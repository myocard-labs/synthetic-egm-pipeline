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

import numpy as np
import numpy.typing as npt
from myocard_egm_signal import (
    ActivationPositionGenerator,
    ConstantSignalError,
    DetectionPreprocessor,
    RectifiedDerivative,
    SingleActivationWindower,
)


def default_preprocessor() -> DetectionPreprocessor:
    """The detection curve used for synthetic traces.

    ``RectifiedDerivative`` — ``g[i] = |x[i] - x[i-1]|``, the ``dV/dt``-max
    convention.

    **Provisional, pending S38.** The original rationale here was that a clean
    simulated trace has no need of the smoothed Botteron envelope, which exists
    for the noise on real recordings. CL-167 corrected that framing:
    ``activation_position`` has to be the *same measurand* on both corpora, so
    the curve is a shared decision with iafdb-pipeline rather than a per-corpus
    convenience. iafdb-pipeline dispatches all three curves from config, so
    unifying needs a choice made once, not a default asserted twice. Research
    also measured that on a clean trace ``dV/dt``-max and the Botteron envelope
    land at the same index, so this is not expected to move any number — it is
    about the two sides agreeing by construction rather than by coincidence.
    """
    return RectifiedDerivative()


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
            raise ValueError(
                f"pair {pair_index} carries a constant signal, so no activation can be "
                "detected. The wavefront never reached this pair — check the activation "
                "source, the substrate density, and the capture duration."
            ) from exc

        (window,) = window_set.windows
        if not window.in_bounds:
            start = window.activation_index - round(
                window.realized_position * (window_length_samples - 1)
            )
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
            raise ValueError(
                f"pair {pair_index}: a window of {window_length_samples} samples at realized "
                f"position {window.realized_position:.4f} does not fit inside a "
                f"{n_capture}-sample capture. " + detail
            )
        assert window.signal is not None  # in-bounds windows always carry their slice
        signals[pair_index] = window.signal.astype(np.float32, copy=False)
        realized[pair_index] = window.realized_position
        requested[pair_index] = window.requested_position

    return CroppedTraces(
        signals=signals, realized_positions=realized, requested_positions=requested
    )


__all__ = ["CroppedTraces", "crop_traces", "default_preprocessor"]
