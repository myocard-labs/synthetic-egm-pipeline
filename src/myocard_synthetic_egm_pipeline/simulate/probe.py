"""Positional-sensitivity probe — one logical simulation per crop offset (SEP13).

The probe is a **diagnostic bank, not training data**. It emits N simulations
that differ only in where the window was cut, computed by reusing a single
solve. STU8 plots model output *against* activation offset; holding everything
but the offset fixed is what makes the resulting curve attributable to the
offset rather than to a different draw.

One grid point is one ``simulation_id``
---------------------------------------
The single solve is an **implementation detail, not the unit of identity.** The
first version of this made the trace axis ``(pair x grid point)`` inside one
simulation and carried a per-trace pair mapping to go with it; a generated bank
came out with ``pair_index`` running 0-59 against 20 real pairs, because
``pair_index`` had been quietly promoted from *foreign key into the pair list*
to *trace identity*.

That is not a matter of taste. ``egm-studio``'s synthetic-bank loader matches
ClassifierBank traces to their theta companion on ``(simulation_id,
pair_index)`` and **raises when either side's key is non-unique**, so the pair
is a composite primary key and the tiled bank was unloadable by the one
consumer it exists for.

So every grid point becomes its own logical simulation: ``n_pairs`` traces,
``pair_index`` back to ``0..n_pairs-1``, one ``activation_position`` per
simulation, and nothing tiled. The sweep is identified by its **shared seed**,
which the schema already carries per simulation and does not require to be
unique.

Detect once, then shift (D6)
----------------------------
This is not "call the SEP2 crop N times". That path re-runs the detector per
window, which carries detector jitter — fine when the position is a training
augmentation draw, wrong for an axis a study reads off. So the activation is
detected **once per pair** on the source trace and every grid point is placed by
exact integer shift from it. One detection, N windows; the x-axis carries no
jitter, and the verification correspondingly asserts that the emitted
``activation_position`` column *reproduces* the grid rather than approximates it.

The windows still go through egm-signal's ``window_train`` — the same function
the random-position crop and the IAFDB side use — so probe windows and training
windows have identical geometry by construction rather than by two
configurations agreeing (CL-134).

The grid lives on the sample lattice (D6, amended)
--------------------------------------------------
A window is placed at ``s = round(t_a - p(T-1))`` and reports
``realized = (t_a - s)/(T-1)``, so realized equals requested **iff ``p(T-1)`` is
an integer**. At ``T = 192`` the divisor is 191, which is prime: the only
representable positions are ``j/191``, and a grid stated in round fractions
lands on none of them.

So the config states fractions and every point is immediately snapped to
``k = round(p(T-1))``. **The snapped value is the grid** — what the sweep
requests, what the bank stores, and what the tests assert against. The
requested-versus-snapped difference is reported once, at config time, instead of
becoming a per-trace approximation nobody can see.

Two requested points that snap to one ``k`` are an **error**, not a silent
de-duplication: the same crop would otherwise be emitted twice under two
different offset labels, which is a mis-specified study rather than something to
quietly repair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from myocard_egm_signal import (
    ConstantSignalError,
    DetectionPreprocessor,
    UniformPositionGenerator,
    detect_activation,
    window_train,
)

from myocard_synthetic_egm_pipeline.simulate.cropping import (
    constant_signal_message,
    default_preprocessor,
    out_of_bounds_message,
)


@dataclass(frozen=True)
class ProbeGrid:
    """The offsets one probe sweep cuts at, snapped to the sample lattice.

    Attributes
    ----------
    requested_positions
        What the config asked for, kept only so the config layer can report the
        snap once and so an error can name the fraction a user wrote.
    offsets_samples
        ``k = round(p(T-1))`` per point — the grid *as the crop sees it*.
    positions
        ``k/(T-1)`` per point — the grid of record. Computed by the same
        division ``window_train`` uses to report ``realized_position``, so the
        emitted column equals this bit-for-bit rather than within a tolerance.
    window_length_samples
        ``T``. A grid is only valid for the ``T`` it was snapped against.

    **Every pair is swept.** There is no pair-subset option: one grid point is
    one logical simulation, so its traces are that simulation's pairs and
    ``pair_index`` is ``0..n_pairs-1``. Sweeping a subset would need a per-trace
    pair mapping — the field the rework deleted — to keep the column pointing at
    the right electrode pair.
    """

    requested_positions: tuple[float, ...]
    offsets_samples: tuple[int, ...]
    positions: tuple[float, ...]
    window_length_samples: int

    @property
    def n_points(self) -> int:
        return len(self.positions)

    @property
    def low(self) -> float:
        """Smallest **snapped** position — the one that sizes the capture."""
        return min(self.positions)

    @property
    def high(self) -> float:
        """Largest **snapped** position — the one that buys the stimulus delay."""
        return max(self.positions)

    @property
    def max_snap_error(self) -> float:
        """Largest requested-to-snapped distance, for the config-time report.

        Bounded by half a sample by construction, i.e. below what the axis can
        represent at all — worth printing once so the number in the study's
        x-axis is known to be the snapped one, and worth never printing per
        trace.
        """
        return max(
            abs(requested - snapped)
            for requested, snapped in zip(self.requested_positions, self.positions, strict=True)
        )

    @classmethod
    def snapped(
        cls,
        *,
        low: float,
        high: float,
        n_points: int,
        window_length_samples: int,
    ) -> ProbeGrid:
        """Build a grid of ``n_points`` fractions over ``[low, high]``, snapped.

        Raises
        ------
        ValueError
            On a malformed range, on fewer than two points, or when two
            requested fractions land on one sample offset.
        """
        if window_length_samples < 2:
            raise ValueError(
                f"window_length_samples must be at least 2 to carry a position; "
                f"got {window_length_samples}."
            )
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"the probe grid must satisfy 0 <= low <= high <= 1; got low={low}, high={high}."
            )
        if n_points < 2:
            raise ValueError(
                f"a probe grid needs at least 2 points to measure anything; got {n_points}. "
                "To pin every trace to one position, use the ordinary "
                "activation_position range collapsed to a point (low: p, high: p) "
                "rather than a one-point sweep."
            )

        lattice = window_length_samples - 1
        requested = np.linspace(low, high, n_points, dtype=np.float64)
        offsets = np.round(requested * lattice).astype(np.int64)

        seen: dict[int, int] = {}
        for index, offset in enumerate(int(k) for k in offsets):
            if offset in seen:
                first = seen[offset]
                raise ValueError(
                    f"probe grid points {first} (p={requested[first]:.6f}) and "
                    f"{index} (p={requested[index]:.6f}) both snap to sample offset "
                    f"{offset}, so the sweep would emit one crop twice under two "
                    f"different offsets. Only j/{lattice} positions are representable "
                    f"at T={window_length_samples}, which admits at most "
                    f"{window_length_samples} distinct points; lower n_points or widen "
                    "[low, high]."
                )
            seen[offset] = index

        return cls(
            requested_positions=tuple(float(p) for p in requested),
            offsets_samples=tuple(int(k) for k in offsets),
            # The same division window_train performs, so the emitted column is
            # equal to this rather than close to it.
            positions=tuple(float(k) / lattice for k in offsets),
            window_length_samples=window_length_samples,
        )


@dataclass(frozen=True)
class ProbedSweep:
    """One capture, cut at every grid point.

    ``signals`` is ``(n_points, n_pairs, T)``: **one entry per grid point**, each
    a complete set of that simulation's pairs. That shape is the whole point of
    the rework — a grid point becomes a logical simulation, so nothing is tiled
    and ``pair_index`` goes back to meaning what the schema says it means.
    """

    signals: npt.NDArray[np.float32]
    """``(n_points, n_pairs, T)`` — grid point, then pair."""

    positions: npt.NDArray[np.float64]
    """``(n_points,)`` — the snapped grid, one value per logical simulation."""

    activation_index_per_pair: npt.NDArray[np.int64]
    """``(n_pairs,)`` — where the single detection landed, for provenance."""


def sweep_capture(
    *,
    traces: npt.NDArray[np.float32],
    grid: ProbeGrid,
    preprocessor: DetectionPreprocessor | None = None,
) -> ProbedSweep:
    """Cut one capture at every grid point, detecting once per pair.

    Parameters
    ----------
    traces
        ``(n_pairs, n_capture)`` at the output rate — the **capture**, not the
        trace: the sweep needs material either side of every offset to cut from.
    grid
        The snapped offsets, from :meth:`ProbeGrid.snapped`.
    preprocessor
        The **run's** detection curve, not a fresh default. A probe that
        detected differently from the bank it characterises would be measuring
        two things at once; the caller passes
        ``DatasetConfig.detection_preprocessor`` straight through.

    Raises
    ------
    ValueError
        If a grid point places a window outside the capture — the same
        FRONT/BACK diagnostic the random-position crop raises, naming the grid
        point as well as the pair. A probe window that does not fit means the
        simulation was sized for a narrower sweep than the grid asks for, and
        clipping it would silently shorten one point of the study's x-axis.
    """
    if traces.ndim != 2:
        raise ValueError(f"traces must be (n_pairs, n_samples); got {traces.ndim}-D.")
    n_pairs, n_capture = traces.shape
    window_length_samples = grid.window_length_samples
    if n_capture < window_length_samples:
        raise ValueError(
            f"capture is {n_capture} samples but the window is {window_length_samples}; "
            "the probe has nothing to cut from. Size the capture with "
            "simulate.sizing.required_capture_duration_ms."
        )

    detector = preprocessor if preprocessor is not None else default_preprocessor()
    signals = np.empty((grid.n_points, n_pairs, window_length_samples), dtype=np.float32)
    activation_index_per_pair = np.empty(n_pairs, dtype=np.int64)

    for pair_index in range(n_pairs):
        trace = np.asarray(traces[pair_index], dtype=np.float64)

        # Once per pair (D6). Re-detecting per grid point would put the
        # detector's jitter onto the axis the study reads off.
        try:
            activation_index = detect_activation(trace, preprocessor=detector)
        except ConstantSignalError as exc:
            raise ValueError(constant_signal_message(f"pair {pair_index}")) from exc
        activation_index_per_pair[pair_index] = activation_index
        train = np.asarray([activation_index], dtype=np.int64)

        for point_index, position in enumerate(grid.positions):
            window_set = window_train(
                trace,
                train,
                # Collapsed to a point: this is a requested offset, not a draw,
                # so the generator consumes no random stream and needs no seed.
                position_generator=UniformPositionGenerator(low=position, high=position),
                window_length_samples=window_length_samples,
            )
            (window,) = window_set.windows
            if not window.in_bounds:
                raise ValueError(
                    out_of_bounds_message(
                        window=window,
                        window_length_samples=window_length_samples,
                        n_capture=n_capture,
                        context=(
                            f"pair {pair_index}, probe grid point {point_index} "
                            f"(offset {grid.offsets_samples[point_index]} samples, "
                            f"p={position:.6f})"
                        ),
                    )
                )
            assert window.signal is not None  # in-bounds windows carry their slice
            signals[point_index, pair_index] = window.signal.astype(np.float32, copy=False)
            # Every window at this grid point reports the same realized position
            # by construction: the offset is an exact integer, so
            # `(t_a - s)/(T-1)` is the grid value whatever `t_a` was.
            assert window.realized_position == position

    return ProbedSweep(
        signals=signals,
        positions=np.asarray(grid.positions, dtype=np.float64),
        activation_index_per_pair=activation_index_per_pair,
    )


__all__ = ["ProbeGrid", "ProbedSweep", "sweep_capture"]
