"""Capture sizing for controlled-position cropping.

The arithmetic here decides whether a window fits, so the tests are written
against the *geometry* — place the window and check where it lands — rather
than against the returned numbers alone. A formula that is internally
consistent but disagrees with egm-signal's crop by one sample would pass the
latter and fail the former.

The model under test (docs/simulation_theory.md has the derivation):

    k(p) = round(p·(T-1))          offset from window start to the activation
    D    = k(p_hi)                 stimulus delay — buys the front
    V    = 2T                      assumed travel allowance
    N    = D + V + T - k(p_lo)     capture length — buys the back
"""

from __future__ import annotations

import numpy as np
import pytest
from myocard_egm_signal import UniformPositionGenerator

from myocard_synthetic_egm_pipeline.simulate.sizing import (
    WINDOW_LENGTH_MULTIPLE,
    activation_offset_samples,
    required_capture_duration_ms,
    required_capture_samples,
    required_stimulus_delay_ms,
    required_stimulus_delay_samples,
    travel_allowance_samples,
    window_length_samples,
)

T = 192


def _sized(low: float, high: float) -> tuple[int, int, int]:
    """(delay, travel allowance, capture) for a position range."""
    d = required_stimulus_delay_samples(window_length_samples=T, position_high=high)
    v = travel_allowance_samples(window_length_samples=T)
    n = required_capture_samples(
        window_length_samples=T, position_low=low, stimulus_delay_samples=d
    )
    return d, v, n


def test_window_length_from_duration() -> None:
    """192 ms at 1 kHz is 192 samples, and it is on the 64-grid."""
    n = window_length_samples(trace_duration_ms=192.0, output_fs_hz=1000.0)
    assert n == T
    assert n % WINDOW_LENGTH_MULTIPLE == 0


@pytest.mark.parametrize(("low", "high"), [(0.4, 0.6), (0.25, 0.75), (0.5, 0.5), (0.0, 1.0)])
def test_every_reachable_window_fits(low: float, high: float) -> None:
    """The whole point: place the window everywhere it can land, check it fits.

    Exhaustive over the reachable activation indices — the activation arrives at
    ``D + travel`` for any travel the allowance admits — crossed with the
    extremes and midpoint of the position range.
    """
    delay, travel_allowance, capture = _sized(low, high)

    for travel in range(travel_allowance + 1):
        activation_index = delay + travel
        for position in (low, (low + high) / 2, high):
            start = activation_index - activation_offset_samples(
                window_length_samples=T, position=position
            )
            assert start >= 0, f"off the front at travel={travel}, p={position}"
            assert start + T <= capture, f"off the back at travel={travel}, p={position}"


def test_the_delay_is_what_buys_the_front() -> None:
    """One sample less delay and the earliest arrival no longer fits.

    Confirms the delay is doing real work rather than being incidentally large.
    """
    low, high = 0.4, 0.6
    delay, _, _ = _sized(low, high)

    # Earliest possible arrival is travel = 0, i.e. the activation index IS the delay.
    assert delay - activation_offset_samples(window_length_samples=T, position=high) == 0
    assert delay - 1 < activation_offset_samples(window_length_samples=T, position=high)


def test_the_capture_is_tight_not_merely_sufficient() -> None:
    """The worst-case window ends flush with the capture.

    A capture that over-allocates would pass the fits-everywhere test above
    while wasting solver time on every simulation, so the bound is asserted
    from both sides.
    """
    low, high = 0.4, 0.6
    delay, travel_allowance, capture = _sized(low, high)

    latest_activation = delay + travel_allowance
    window_end = (
        latest_activation - activation_offset_samples(window_length_samples=T, position=low) + T
    )
    assert window_end == capture


def test_the_front_guarantee_needs_no_geometry() -> None:
    """The delay holds for any travel time, which is why CV can change freely.

    Conduction velocity is being recalibrated; a front guarantee that depended
    on it would have to be re-derived afterwards. This one does not.
    """
    high = 0.75
    delay = required_stimulus_delay_samples(window_length_samples=T, position_high=high)
    k = activation_offset_samples(window_length_samples=T, position=high)

    for travel in (0, 1, 50, 500, 10_000):  # any CV, any patch size
        assert (delay + travel) - k >= 0


@pytest.mark.parametrize(
    ("low", "high", "expected_delay", "expected_capture"),
    [
        # D = k(hi);  N = D + V + T - k(lo),  V = 2T = 384
        (0.4, 0.6, 115, 615),
        (0.5, 0.5, 96, 576),
        (0.0, 0.0, 0, 576),
        (1.0, 1.0, 191, 576),
    ],
)
def test_worked_values(low: float, high: float, expected_delay: int, expected_capture: int) -> None:
    """Pin the shipped arithmetic against hand-computed values."""
    delay, _, capture = _sized(low, high)
    assert delay == expected_delay
    assert capture == expected_capture


def test_travel_allowance_is_two_windows() -> None:
    """``V = 2T`` — assumed, not derived.

    It was ``T`` and that proved too small on a 30-60 % fibrosis run: fibrosis
    slows conduction, and the original justification had bounded travel using a
    *clean*-tissue velocity, which is the fastest case rather than the slowest.
    """
    assert travel_allowance_samples(window_length_samples=T) == 2 * T


def test_durations_round_up() -> None:
    """Rounding must only ever over-capture.

    Both values cross into model time units and back into samples; each hop can
    shed a fraction, and rounding up means the residue costs solver time rather
    than a trace.
    """
    delay_ms = required_stimulus_delay_ms(
        trace_duration_ms=192.0, output_fs_hz=1000.0, position_high=0.6
    )
    capture_ms = required_capture_duration_ms(
        trace_duration_ms=192.0,
        output_fs_hz=1000.0,
        position_low=0.4,
        stimulus_delay_ms=delay_ms,
    )
    _, _, capture_samples = _sized(0.4, 0.6)

    assert capture_ms * 1e-3 * 1000.0 >= capture_samples
    assert delay_ms == float(int(delay_ms))
    assert capture_ms == float(int(capture_ms))


@pytest.mark.parametrize("position", [-0.1, 1.5])
def test_positions_outside_the_unit_interval_are_rejected(position: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        required_capture_samples(window_length_samples=T, position_low=position)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        required_stimulus_delay_samples(window_length_samples=T, position_high=position)


def test_negative_delay_is_rejected() -> None:
    with pytest.raises(ValueError, match="stimulus_delay_samples must be >= 0"):
        required_capture_samples(
            window_length_samples=T, position_low=0.4, stimulus_delay_samples=-1
        )


def test_sizing_covers_every_position_a_generator_can_emit() -> None:
    """Sized from the range's bounds, but every *sampled* position must fit too.

    The bounds are the worst cases by construction; this checks that claim
    against the real generator rather than assuming it.
    """
    low, high = 0.2, 0.8
    generator = UniformPositionGenerator(low=low, high=high, seed=0)
    delay, travel_allowance, capture = _sized(low, high)

    positions = generator.generate(500)
    assert positions.min() >= low and positions.max() <= high

    offsets = np.array(
        [activation_offset_samples(window_length_samples=T, position=float(p)) for p in positions]
    )
    for activation_index in (delay, delay + travel_allowance):  # earliest and latest
        starts = activation_index - offsets
        assert starts.min() >= 0
        assert (starts + T).max() <= capture
