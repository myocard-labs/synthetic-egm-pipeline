"""Sizing the simulation so a ``T``-sample window always fits around the activation.

This is the synthetic-side *response* to controlled-position cropping, and the
part egm-signal cannot do. SIG1 owns the crop — the detection curve, the
``argmax`` detection, the fractional-to-index conversion, the slice. What it
cannot own is **how much signal exists to slice from**, nor **where in that
signal the activation lands**; both follow from how the simulation was set up.
A windower handed a trace that is too short can only report ``in_bounds=False``;
only the producer can prevent it.

The goal is to window synthetic traces the way IAFDB traces are windowed. A real
record is arbitrarily long with activations somewhere in the middle, so a window
at any position ``p`` has material either side of it. A simulation has to be
*arranged* to have the same property, with two knobs: fire the stimulus late
enough that there is signal before the activation, and run long enough that
there is signal after it.

Notation
--------
All quantities are in samples at ``output_fs_hz`` unless suffixed ``_ms``.

- ``T``       — window length, the on-disk trace length
- ``p``       — fractional activation position within the window, in ``[0, 1]``,
                drawn per window from ``[p_lo, p_hi]``
- ``k(p)``    — offset from window start to the activation, ``round(p·(T-1))``
- ``a``       — activation index within the capture (measured per pair)
- ``D``       — stimulus delay
- ``V``       — travel allowance: an upper bound on how long the wave takes to
                reach a pair after the stimulus fires
- ``N``       — capture length

Placement and the two constraints
---------------------------------
A window around an activation at ``a`` spans ``[a - k(p), a - k(p) + T)``, so::

    front:  a - k(p) >= 0        i.e. enough signal before the activation
    back:   a - k(p) + T <= N    i.e. enough signal after it

**The front is bought with the delay.** ``a = D + travel`` and ``travel >= 0``,
so ``D >= k(p_hi)`` satisfies the front for *any* patch size, conduction
velocity or electrode position — no geometry knowledge required, which is what
makes it robust to the CV recalibration.

**The back is bought with capture length.** Worst case is the largest ``a`` with
the smallest ``p``. Bounding ``a <= D + V``::

    N >= D + V + T - k(p_lo)

**On ``V``.** Deriving the true travel time needs conduction velocity and the
stimulus-to-pair distance, neither of which the config knows and the first of
which is being recalibrated. So ``V`` is assumed rather than derived, and
defaults to ``2T`` — overridable via ``run.travel_allowance_ms``.

**Why ``2T`` and not ``T``.** ``T`` was the first choice and it was wrong,
caught on a 30-60 % fibrosis run where a pair activated at index 329 against an
allowance of 192. The justification for ``T`` had been "the furthest pair is
~24 mm out, which ``T`` covers for any CV above ~0.13 mm/ms" — computed from
**clean-tissue** CV. Fibrosis slows conduction substantially, so clean tissue is
the *fastest* case, which is the wrong end from which to bound a worst case.
``V`` has to be sized for the slowest substrate a run can draw, not the
quickest.

An over-estimate costs solver time and nothing else; an under-estimate costs
traces. Where even the default is not enough, the crop raises with a message
naming this knob rather than silently producing a short trace.

What this replaced
------------------
An earlier version sized only the back and argued the front away — that the
lead-in was inherently flat, so ``p`` should be held low ("back-bounded"). That
reasoning was retired (plan design note D9): it was reasoned back from
activation indices that later proved to be detection artifacts, and it would
have left synthetic windows positionally disjoint from IAFDB's, which is a
sim-to-real gap rather than a fix for one.
"""

from __future__ import annotations

import math

#: ``T`` must be a multiple of this. egm-classifier's 1D MobileViT halves the
#: sequence six times (2**6 = 64), so a length off the grid fails outright at
#: the first ragged stage rather than degrading (CL-112, design §8.1).
WINDOW_LENGTH_MULTIPLE: int = 64


def window_length_samples(*, trace_duration_ms: float, output_fs_hz: float) -> int:
    """``T`` — the on-disk trace length in samples."""
    return round(trace_duration_ms * 1e-3 * output_fs_hz)


def activation_offset_samples(*, window_length_samples: int, position: float) -> int:
    """``k(p) = round(p·(T-1))`` — offset from window start to the activation.

    Restated here rather than imported so the sizing arithmetic matches
    egm-signal's crop exactly. A one-sample disagreement between the two is a
    window that fits in theory and not in practice.
    """
    return round(position * (window_length_samples - 1))


#: Default travel allowance as a multiple of ``T``. See the module docstring
#: for why this is 2 and not 1 — the first choice was bounded with a
#: clean-tissue conduction velocity, and fibrosis slows conduction.
TRAVEL_ALLOWANCE_WINDOWS: int = 2


def travel_allowance_samples(*, window_length_samples: int) -> int:
    """``V`` — assumed upper bound on stimulus-to-pair travel time.

    The true value is ``distance / conduction_velocity``; neither term is
    available at config time, since the config carries no stimulus-to-pair
    distance and CV is a calibration output rather than an input. So this is a
    deliberate over-estimate, sized for the *slowest* substrate a run may draw
    rather than the fastest.
    """
    return TRAVEL_ALLOWANCE_WINDOWS * window_length_samples


def required_stimulus_delay_samples(*, window_length_samples: int, position_high: float) -> int:
    """``D = k(p_hi)`` — delay that guarantees the front for any geometry.

    The largest position in the range is the worst case: the further back in its
    window the activation sits, the more signal is needed ahead of it.
    """
    _check_position(position_high, "position_high")
    return activation_offset_samples(
        window_length_samples=window_length_samples, position=position_high
    )


def required_capture_samples(
    *,
    window_length_samples: int,
    position_low: float,
    stimulus_delay_samples: int = 0,
    travel_samples: int | None = None,
) -> int:
    """``N = D + V + T - k(p_lo)`` — capture length that guarantees the back.

    The smallest position in the range is the worst case here: the earlier in
    its window the activation sits, the more signal is needed behind it.
    """
    _check_position(position_low, "position_low")
    if stimulus_delay_samples < 0:
        raise ValueError(f"stimulus_delay_samples must be >= 0; got {stimulus_delay_samples}.")
    travel = (
        travel_allowance_samples(window_length_samples=window_length_samples)
        if travel_samples is None
        else travel_samples
    )
    return (
        stimulus_delay_samples
        + travel
        + window_length_samples
        - activation_offset_samples(
            window_length_samples=window_length_samples, position=position_low
        )
    )


def required_stimulus_delay_ms(
    *, trace_duration_ms: float, output_fs_hz: float, position_high: float
) -> float:
    """:func:`required_stimulus_delay_samples` as a duration, rounded up."""
    d = required_stimulus_delay_samples(
        window_length_samples=window_length_samples(
            trace_duration_ms=trace_duration_ms, output_fs_hz=output_fs_hz
        ),
        position_high=position_high,
    )
    return math.ceil(d / output_fs_hz * 1000.0)


def required_capture_duration_ms(
    *,
    trace_duration_ms: float,
    output_fs_hz: float,
    position_low: float,
    stimulus_delay_ms: float = 0.0,
    travel_allowance_ms: float | None = None,
) -> float:
    """:func:`required_capture_samples` as a duration for the backend.

    Rounded **up** to the whole millisecond. The backend converts this to model
    time units and the runner converts back to samples; each hop can shed a
    fraction, and rounding up means the residue costs solver time rather than a
    trace.
    """
    t = window_length_samples(trace_duration_ms=trace_duration_ms, output_fs_hz=output_fs_hz)
    n = required_capture_samples(
        window_length_samples=t,
        position_low=position_low,
        stimulus_delay_samples=round(stimulus_delay_ms * 1e-3 * output_fs_hz),
        travel_samples=(
            None
            if travel_allowance_ms is None
            else round(travel_allowance_ms * 1e-3 * output_fs_hz)
        ),
    )
    return math.ceil(n / output_fs_hz * 1000.0)


def _check_position(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value}.")


__all__ = [
    "TRAVEL_ALLOWANCE_WINDOWS",
    "WINDOW_LENGTH_MULTIPLE",
    "activation_offset_samples",
    "required_capture_duration_ms",
    "required_capture_samples",
    "required_stimulus_delay_ms",
    "required_stimulus_delay_samples",
    "travel_allowance_samples",
    "window_length_samples",
]
