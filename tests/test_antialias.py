"""The anti-alias filter: attenuation instead of folding, and no phase shift.

Decimation without band-limiting is not a lossy step, it is a *wrong* one:
content above the output Nyquist does not disappear, it reappears at a mirrored
frequency inside the band, indistinguishable from signal that was really there.
The corpus consequence is what makes it worth a module of its own — a real
front-end band-limits ahead of its ADC, so no real recording can carry that
content, and an unfiltered synthetic trace carrying it is a difference between
the two corpora that nothing downstream could attribute.

Two properties carry the weight here, and the second is the one most likely to
be got subtly wrong:

- energy above the output Nyquist comes back **attenuated, not folded**;
- the filter is **zero-phase**, so a detected activation index does not move.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from myocard_egm_signal import detect_activation
from scipy.signal import butter, sosfilt

from myocard_synthetic_egm_pipeline.simulate.cropping import default_preprocessor
from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import (
    ANTIALIAS_CUTOFF_FRACTION,
    ANTIALIAS_ORDER,
    band_limit,
    downsample,
)

OUTPUT_FS_HZ = 1000.0
#: A capture rate with the awkward property the real ones have: not an integer
#: multiple of the output rate. The two shipped cards give 4060.9 and 4166.7 Hz.
CAPTURE_FS_HZ = 4060.9
N_CAPTURE = 4000


def _tone(
    freq_hz: float, *, n: int = N_CAPTURE, fs: float = CAPTURE_FS_HZ
) -> npt.NDArray[np.float64]:
    t = np.arange(n, dtype=np.float64) / fs
    return np.asarray(np.sin(2.0 * np.pi * freq_hz * t), dtype=np.float64)


def _amplitude_at(signal: npt.NDArray[np.float64], freq_hz: float, *, fs: float) -> float:
    """Peak spectral amplitude within +/- 5 Hz of ``freq_hz``."""
    windowed = (signal - signal.mean()) * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(windowed)) * 2.0 / signal.size
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    band = np.abs(freqs - freq_hz) <= 5.0
    return float(spectrum[band].max())


# ---------------------------------------------------------------------------
# Attenuated, not folded
# ---------------------------------------------------------------------------


def test_decimating_without_the_filter_really_does_fold() -> None:
    """The control. Without this, the next test proves nothing.

    A test that fed the pipeline a signal already inside the band would pass
    whatever the code did — including doing nothing at all. So first establish
    that the fold is real: a 700 Hz tone in a 4060.9 Hz capture is above the
    500 Hz output Nyquist, and plain rate conversion brings it back at
    ``1000 - 700 = 300`` Hz wearing the costume of signal that was never there.

    It arrives at **0.38 of its original amplitude**, not all of it, because
    the rate conversion is a linear interpolation and that is itself a crude
    low-pass. Crude is the operative word: it costs an out-of-band tone about
    half its amplitude and passes the rest straight into the band.
    """
    folded = downsample(_tone(700.0), source_fs_hz=CAPTURE_FS_HZ, target_fs_hz=OUTPUT_FS_HZ)

    alias = _amplitude_at(folded, 300.0, fs=OUTPUT_FS_HZ)
    assert alias > 0.3, (
        f"a 700 Hz tone did not reappear at 300 Hz (amplitude {alias:.3f}); the "
        "premise of the filter test below no longer holds"
    )


@pytest.mark.parametrize(
    ("tone_hz", "alias_hz"),
    [(600.0, 400.0), (700.0, 300.0), (900.0, 100.0), (1200.0, 200.0)],
)
def test_a_tone_above_the_output_nyquist_is_attenuated_not_folded(
    tone_hz: float, alias_hz: float
) -> None:
    """Band-limit first, and the alias never arrives.

    Each tone sits above the 500 Hz output Nyquist and would mirror to
    ``alias_hz`` on decimation. The assertion is on **where the energy ends
    up**: essentially nothing at the mirror frequency, rather than merely "the
    output is smaller".
    """
    capture = _tone(tone_hz)
    limited = band_limit(capture, fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ)
    out = downsample(limited, source_fs_hz=CAPTURE_FS_HZ, target_fs_hz=OUTPUT_FS_HZ)

    unfiltered = downsample(capture, source_fs_hz=CAPTURE_FS_HZ, target_fs_hz=OUTPUT_FS_HZ)
    before = _amplitude_at(unfiltered, alias_hz, fs=OUTPUT_FS_HZ)
    after = _amplitude_at(out, alias_hz, fs=OUTPUT_FS_HZ)

    assert after < 0.02 * before, (
        f"a {tone_hz} Hz tone still folds to {alias_hz} Hz at amplitude {after:.5f} "
        f"against {before:.3f} unfiltered — less than a 50x reduction"
    )


def test_content_below_the_corner_survives() -> None:
    """The other half: the filter must not simply flatten everything.

    An anti-alias filter that removed the signal too would also pass the test
    above, so the passband is asserted as well. 100 and 250 Hz are inside the
    400 Hz corner and come through essentially untouched.
    """
    for freq_hz in (100.0, 250.0):
        capture = _tone(freq_hz)
        limited = band_limit(capture, fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ)

        kept = _amplitude_at(limited, freq_hz, fs=CAPTURE_FS_HZ)
        original = _amplitude_at(capture, freq_hz, fs=CAPTURE_FS_HZ)
        assert kept > 0.97 * original, (
            f"{freq_hz} Hz is inside the {ANTIALIAS_CUTOFF_FRACTION * 500:.0f} Hz "
            f"corner but lost {100 * (1 - kept / original):.1f} % of its amplitude"
        )


# ---------------------------------------------------------------------------
# Zero phase — the regression that matters most
# ---------------------------------------------------------------------------


def _activation_trace(n: int = N_CAPTURE, fs: float = CAPTURE_FS_HZ) -> npt.NDArray[np.float64]:
    """One unambiguous activation, band-limited, a third of the way in."""
    t_ms = np.arange(n, dtype=np.float64) * 1000.0 / fs
    centre = t_ms[n // 3]
    envelope = np.exp(-0.5 * ((t_ms - centre) / 3.0) ** 2)
    carrier = np.sin(2.0 * np.pi * 120.0 * (t_ms - centre) / 1000.0)
    return np.asarray(envelope * carrier, dtype=np.float64)


def test_the_filter_does_not_move_the_detected_activation() -> None:
    """A known activation lands on the same sample either side of the filter.

    This is the failure the whole design turns on. The detected index sets
    ``activation_position`` — a stored column, an asserted value, and the axis
    the controlled-position crop is built on — so a filter with group delay
    would shift every window in every bank while every test that does not look
    at absolute timing carried on passing.

    Asserted **on the finished trace, at the output rate**, because that is
    where the index is taken: the runner detects after the rate conversion, so
    it is the sample grid the bank is stored on that has to be unmoved.
    """
    capture = _activation_trace()
    curve = default_preprocessor()

    reference = downsample(capture, source_fs_hz=CAPTURE_FS_HZ, target_fs_hz=OUTPUT_FS_HZ)
    filtered = downsample(
        band_limit(capture, fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ),
        source_fs_hz=CAPTURE_FS_HZ,
        target_fs_hz=OUTPUT_FS_HZ,
    )

    before = int(detect_activation(reference.astype(np.float32), preprocessor=curve))
    after = int(detect_activation(filtered.astype(np.float32), preprocessor=curve))

    assert after == before, (
        f"the activation moved from sample {before} to {after} — the filter is not "
        "zero-phase, and every crop in every bank would be shifted with it"
    )


def test_a_causal_filter_of_the_same_design_would_move_it() -> None:
    """The control for the test above, and the reason it is not vacuous.

    Same Butterworth, same corner, applied **forwards only**. If this did not
    shift the activation then the zero-phase assertion above would be proving
    nothing about phase — it would just be saying the filter is gentle.

    It shifts by 2 samples at the output rate, which sounds small and is not:
    a window is 192 samples, so 2 samples moves the realized
    ``activation_position`` by 0.010 — larger than the snapping tolerance the
    probe grid is asserted to within, and applied to every trace in every bank.
    """
    capture = _activation_trace()
    curve = default_preprocessor()
    sos = butter(
        ANTIALIAS_ORDER,
        ANTIALIAS_CUTOFF_FRACTION * 0.5 * OUTPUT_FS_HZ,
        btype="low",
        fs=CAPTURE_FS_HZ,
        output="sos",
    )

    reference = downsample(capture, source_fs_hz=CAPTURE_FS_HZ, target_fs_hz=OUTPUT_FS_HZ)
    causal = downsample(
        np.asarray(sosfilt(sos, capture, axis=0)),
        source_fs_hz=CAPTURE_FS_HZ,
        target_fs_hz=OUTPUT_FS_HZ,
    )

    before = int(detect_activation(reference.astype(np.float32), preprocessor=curve))
    after = int(detect_activation(causal.astype(np.float32), preprocessor=curve))

    assert after >= before + 2, (
        f"a causal filter moved the activation only from {before} to {after}; if "
        "group delay is this small the zero-phase test above is not testing phase"
    )


# ---------------------------------------------------------------------------
# What a regenerated bank differs by
# ---------------------------------------------------------------------------


def test_the_filter_changes_the_high_band_and_leaves_the_low_band(
    ambiguous_capture: npt.NDArray[np.float64],
) -> None:
    """Filtered and unfiltered output differ **in the high band specifically**.

    "The traces changed" would be satisfied by any bug at all. What says the
    filter is live rather than merely present is *where* they changed: the
    200-500 Hz band, which is where this capture's out-of-band tones fold to,
    loses essentially all of its energy, while everything below 200 Hz is left
    alone.

    The capture carries tones at 650 and 750 Hz, chosen so that both mirror
    **into the upper band** — 350 and 250 Hz. An alias that landed below 200 Hz
    would make the low band change too, correctly, and the test would then be
    unable to tell "the filter works" from "the filter mangles everything".
    """
    capture = ambiguous_capture
    plain = downsample(capture, source_fs_hz=CAPTURE_FS_HZ, target_fs_hz=OUTPUT_FS_HZ)
    limited = downsample(
        band_limit(capture, fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ),
        source_fs_hz=CAPTURE_FS_HZ,
        target_fs_hz=OUTPUT_FS_HZ,
    )

    def band_power(signal: npt.NDArray[np.float64], low: float, high: float) -> float:
        windowed = (signal - signal.mean()) * np.hanning(signal.size)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / OUTPUT_FS_HZ)
        return float(spectrum[(freqs >= low) & (freqs < high)].sum())

    low_before, low_after = band_power(plain, 0.0, 200.0), band_power(limited, 0.0, 200.0)
    high_before, high_after = band_power(plain, 200.0, 500.0), band_power(limited, 200.0, 500.0)

    assert abs(low_after - low_before) < 0.03 * low_before, (
        "the filter moved the low band, which it has no business touching: "
        f"{low_before:.4g} -> {low_after:.4g}"
    )
    assert high_after < 0.01 * high_before, (
        f"the high band barely moved ({high_before:.4g} -> {high_after:.4g}); a "
        "filter that is present but inert would look exactly like this"
    )


@pytest.fixture
def ambiguous_capture() -> npt.NDArray[np.float64]:
    """A capture with content on both sides of the output Nyquist.

    Both out-of-band tones fold above 200 Hz — 650 to 350 and 750 to 250 — so
    the low band stays a clean measure of what the filter should not touch.
    """
    return np.asarray(
        _activation_trace() + 0.3 * _tone(650.0) + 0.3 * _tone(750.0), dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_corner_above_the_capture_nyquist_is_refused() -> None:
    """Nothing to filter is a configuration error, not a no-op.

    It means the capture is barely oversampled relative to the output, so
    filtering silently does nothing and the fold happens anyway.
    """
    with pytest.raises(ValueError, match="capture Nyquist"):
        band_limit(_tone(100.0), fs_hz=700.0, output_fs_hz=OUTPUT_FS_HZ)


def test_a_capture_too_short_to_filter_is_refused_by_name() -> None:
    """``sosfiltfilt`` pads each end; too short a signal has nothing to pad from."""
    with pytest.raises(ValueError, match="too short"):
        band_limit(_tone(100.0, n=20), fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_a_corner_outside_the_unit_interval_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match="cutoff_fraction"):
        band_limit(
            _tone(100.0),
            fs_hz=CAPTURE_FS_HZ,
            output_fs_hz=OUTPUT_FS_HZ,
            cutoff_fraction=fraction,
        )


def test_one_dimensional_and_multichannel_agree() -> None:
    """A channel filtered alone and in company come out the same.

    The runner hands this a ``(T, n_pairs)`` block, so an axis mistake would
    filter across pairs instead of across time — which on a 20-pair capture is
    a plausible-looking result rather than an obvious one.
    """
    single = _activation_trace()
    block = np.stack([single, 2.0 * single, -single], axis=1)

    alone = band_limit(single, fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ)
    together = band_limit(block, fs_hz=CAPTURE_FS_HZ, output_fs_hz=OUTPUT_FS_HZ)

    assert together.shape == block.shape
    np.testing.assert_allclose(together[:, 0], alone, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(together[:, 1], 2.0 * alone, rtol=1e-9, atol=1e-12)
