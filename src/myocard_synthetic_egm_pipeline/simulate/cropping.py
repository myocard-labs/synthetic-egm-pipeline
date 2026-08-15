"""Controlled-position cropping of a simulation's bipolar traces (SEP2).

The capture is longer than the trace (see
:mod:`~myocard_synthetic_egm_pipeline.simulate.sizing`); this is where the
``T``-sample window is actually cut out of it, with the activation placed at a
sampled fractional position.

**Per trace, not per simulation.** The wavefront sweeps the electrode grid, so
pairs on opposite sides of it activate tens of milliseconds apart. One offset
applied to a whole simulation would put the activation at the requested
position for a single pair and somewhere uncontrolled for all the others —
which reintroduces exactly the positional structure the crop exists to remove.

**Detected, not told.** The stimulus time is a configured number, but the
activation time *at a given pair* is not: the stored trace is a pseudo-EGM, a
distance-weighted sum of membrane current over the whole mesh, so its timing
follows the wavefront's arrival — a function of conduction velocity, the
realized fibrosis draw, the electrode standoff and the pair's position. None of
those is a value the config carries. Detecting also keeps T1 a
detected-versus-detected comparison: the real corpus has nothing but the
electrogram, so handing the synthetic side a privileged ground-truth time would
flatter the very comparison the experiment exists to make (CL-130 / CL-147).

Everything below routes through egm-signal's ``SingleActivationWindower``,
which is ``window_train`` with a **train of one** (CL-134) — the same function
the IAFDB side uses, which is what stops the window geometry drifting between
the two corpora.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from myocard_egm_signal import (
    DEFAULT_BOTTERON_BAND_HZ,
    DEFAULT_BOTTERON_LOWPASS_HZ,
    ActivationPositionGenerator,
    BotteronEnvelope,
    ConstantSignalError,
    DetectionPreprocessor,
    RectifiedDerivative,
    SingleActivationWindower,
    TeagerKaiser,
)

#: The curve names ``activation_position.detection.curve`` accepts, spelled
#: exactly as iafdb-pipeline spells them (``cli/_config.py::DetectionCurve``
#: there). A different spelling on this side would put a translation step inside
#: every cross-corpus comparison, which is where the mistakes go. The *names*
#: match; the block path deliberately does not — see the config builder.
DETECTION_CURVES: tuple[str, ...] = (
    "rectified_derivative",
    "teager_kaiser",
    "botteron_envelope",
)


def default_preprocessor() -> DetectionPreprocessor:
    """The detection curve used when a config names none.

    ``RectifiedDerivative`` — ``g[i] = |x[i] - x[i-1]|``, the ``dV/dt``-max
    convention.

    The original rationale was that a clean simulated trace has no need of the
    smoothed Botteron envelope, which exists for the noise on real recordings.
    CL-167 corrected that framing: ``activation_position`` has to be the *same
    measurand* on both corpora, so the curve is a shared decision with
    iafdb-pipeline rather than a per-corpus convenience — and a decision that
    can only be made once the two sides can be *set* to the same value.
    :func:`build_preprocessor` is that seam (S16a); this stays the default so
    an absent ``activation_position.detection`` block means exactly what it
    meant before it existed.
    """
    return RectifiedDerivative()


def build_preprocessor(
    *,
    curve: str,
    fs_hz: float,
    botteron_band_hz: tuple[float, float] = DEFAULT_BOTTERON_BAND_HZ,
    botteron_lowpass_hz: float = DEFAULT_BOTTERON_LOWPASS_HZ,
) -> DetectionPreprocessor:
    """Map a curve name onto an egm-signal preprocessor.

    The synthetic-side twin of iafdb-pipeline's
    ``export/activation_extract.py::build_preprocessor`` — same three names,
    same two Botteron parameters. Only the curve subset is mirrored: IAFDB runs
    ``detect_activation_train`` (preprocess → threshold → select → suppress →
    refine) while this side runs ``detect_activation``, which is ``argmax g``
    on a trace holding exactly one activation by construction. Its threshold /
    prominence / refractory knobs would be config that provably does nothing
    here, so they are refused at the config layer rather than accepted.

    Parameters
    ----------
    curve
        One of :data:`DETECTION_CURVES`.
    fs_hz
        Sampling rate **of the traces the detector will see** — i.e. the
        *output* rate, not the capture rate. Only ``botteron_envelope`` reads
        it, and it reads it to turn Hz cutoffs into filter coefficients: the
        runner downsamples before it crops, so passing ``fs_capture_hz`` would
        mis-scale the band and the low-pass by the oversample factor and still
        run without complaint — a wrong number, not an error.
    botteron_band_hz, botteron_lowpass_hz
        Botteron only; ignored by the two sample-domain curves, which is why
        the config layer rejects them alongside another curve rather than
        silently dropping them.
    """
    if curve == "rectified_derivative":
        return RectifiedDerivative()
    if curve == "teager_kaiser":
        return TeagerKaiser()
    if curve == "botteron_envelope":
        return BotteronEnvelope(
            fs=fs_hz,
            band_hz=botteron_band_hz,
            lowpass_hz=botteron_lowpass_hz,
        )
    raise ValueError(f"Unknown detection curve {curve!r}; expected one of {DETECTION_CURVES}.")


def out_of_bounds_message(
    *,
    window: Any,
    window_length_samples: int,
    n_capture: int,
    context: str,
) -> str:
    """Why a window did not fit, and which knob buys the room.

    Shared by the random-position crop and the probe sweep (S16b) so the two
    cannot drift into describing the same failure differently. ``context``
    names what was being cut — a pair for the crop, a pair *and a grid point*
    for the probe — because "it does not fit" without a subject sends the
    reader looking through every trace.

    ``window`` is egm-signal's ``Window``; typed loosely because the class is
    not exported and the two attributes read here are stable.
    """
    start = window.activation_index - round(window.realized_position * (window_length_samples - 1))
    off_front = start < 0
    overshoot = (start + window_length_samples) - n_capture
    detail = (
        (
            f"It runs off the FRONT by {-start} samples: the activation arrived before "
            "the stimulus delay allowed for. Raise activation.stimulus_delay_ms."
        )
        if off_front
        else (
            f"It runs off the BACK by {overshoot} samples: the wave took longer to reach "
            f"this pair than the travel allowance covers (activation at "
            f"{window.activation_index}). Raise run.travel_allowance_ms — it needs to "
            f"exceed the largest activation index minus the stimulus delay. Heavy "
            f"fibrosis slows conduction, so a dense substrate needs a larger allowance "
            f"than a clean one."
        )
    )
    return (
        f"{context}: a window of {window_length_samples} samples at realized "
        f"position {window.realized_position:.4f} does not fit inside a "
        f"{n_capture}-sample capture. " + detail
    )


def constant_signal_message(context: str) -> str:
    """Why a flat trace has no activation to anchor on.

    Shared for the same reason as :func:`out_of_bounds_message`: both callers
    hit it, and on a synthetic run it always means the same thing — the
    wavefront never reached that pair.
    """
    return (
        f"{context} carries a constant signal, so no activation can be detected. "
        "The wavefront never reached this pair — check the activation source, "
        "the substrate density, and the capture duration."
    )


class CroppedTraces:
    """Result of cropping one simulation's traces.

    ``signals`` is ``(n_pairs, T)``; ``realized_positions`` is ``(n_pairs,)``.

    **Realized, never requested.** The two differ whenever ``p * (T - 1)`` is
    not an integer, and only the realized value describes where the activation
    actually sits in the stored trace — so it is the one written to the bank
    and the one any positional analysis must read.
    """

    __slots__ = ("realized_positions", "requested_positions", "signals")

    def __init__(
        self,
        *,
        signals: npt.NDArray[np.float32],
        realized_positions: npt.NDArray[np.float64],
        requested_positions: npt.NDArray[np.float64],
    ) -> None:
        self.signals = signals
        self.realized_positions = realized_positions
        self.requested_positions = requested_positions


def crop_traces(
    *,
    traces: npt.NDArray[np.float32],
    position_generator: ActivationPositionGenerator,
    window_length_samples: int,
    preprocessor: DetectionPreprocessor | None = None,
) -> CroppedTraces:
    """Cut one ``T``-sample window per bipolar trace, at a sampled position.

    Parameters
    ----------
    traces
        ``(n_pairs, n_capture)`` at the output rate, where ``n_capture`` is the
        sized capture rather than ``T``.
    position_generator
        egm-signal's stateful generator. Shared across every pair and every
        simulation in a run, so the stream advances rather than repeating —
        two pairs drawing the same position would be a coincidence, not a rule.
    window_length_samples
        ``T``.
    preprocessor
        Detection-curve transform; :func:`default_preprocessor` when omitted.
        A run's curve comes from ``activation_position.detection`` via
        :func:`build_preprocessor` — nested there because this function builds
        one windower out of the preprocessor and the position generator, so the
        two are one decision.

    Raises
    ------
    ValueError
        If any window lands out of bounds. egm-signal deliberately *reports*
        rather than filters, because the real side wants to count its drops —
        but on the synthetic side a drop means the capture was mis-sized, which
        is a bug in :mod:`~myocard_synthetic_egm_pipeline.simulate.sizing` and
        not something to quietly discard a trace over.
    """
    if traces.ndim != 2:
        raise ValueError(f"traces must be (n_pairs, n_samples); got {traces.ndim}-D.")
    n_pairs, n_capture = traces.shape
    if n_capture < window_length_samples:
        raise ValueError(
            f"capture is {n_capture} samples but the window is {window_length_samples}; "
            "the crop has nothing to cut from. Size the capture with "
            "simulate.sizing.required_capture_duration_ms."
        )

    windower = SingleActivationWindower(
        preprocessor=preprocessor if preprocessor is not None else default_preprocessor(),
        position_generator=position_generator,
        window_length_samples=window_length_samples,
    )

    signals = np.empty((n_pairs, window_length_samples), dtype=np.float32)
    realized = np.empty(n_pairs, dtype=np.float64)
    requested = np.empty(n_pairs, dtype=np.float64)

    for pair_index in range(n_pairs):
        try:
            window_set = windower.window(np.asarray(traces[pair_index], dtype=np.float64))
        except ConstantSignalError as exc:
            # A flat trace has no activation to detect. egm-signal raises
            # rather than returning an arbitrary index, and we let it through
            # with the pair named: on a clean simulation this means the
            # wavefront never reached that pair, which is a generation problem,
            # not a windowing one.
            raise ValueError(constant_signal_message(f"pair {pair_index}")) from exc

        (window,) = window_set.windows
        if not window.in_bounds:
            raise ValueError(
                out_of_bounds_message(
                    window=window,
                    window_length_samples=window_length_samples,
                    n_capture=n_capture,
                    context=f"pair {pair_index}",
                )
            )
        assert window.signal is not None  # in-bounds windows always carry their slice
        signals[pair_index] = window.signal.astype(np.float32, copy=False)
        realized[pair_index] = window.realized_position
        requested[pair_index] = window.requested_position

    return CroppedTraces(
        signals=signals, realized_positions=realized, requested_positions=requested
    )


__all__ = [
    "DETECTION_CURVES",
    "CroppedTraces",
    "build_preprocessor",
    "constant_signal_message",
    "crop_traces",
    "default_preprocessor",
    "out_of_bounds_message",
]
