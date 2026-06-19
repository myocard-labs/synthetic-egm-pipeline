"""Thin per-simulation orchestrator.

:func:`run_single` is the public entry point that wires the four
strategy specs into a chosen backend, then post-processes the
backend's ``RawSimulationResult`` into the runner-side
:class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationResult`
that label policies and storage consume.

The orchestration is intentionally small:

1. Call ``backend.simulate(...)`` — backend owns the V_m → φ_e step
   internally (see ``project/architecture.md`` for why).
2. Form bipolar pairs from the per-electrode unipolar traces at the
   capture rate.
3. Downsample to the output rate (typically 1 kHz).
4. Truncate / pad to the exact target sample count.
5. Compute per-pair midpoints in physical mm so locality-aware label
   policies don't have to re-derive them.
6. Stamp per-run provenance (geometry / substrate / activation /
   electrode type, plus per-sim sampled scalars like ``stim_edge`` and
   ``electrode_height_mm``) into ``run_metadata``.

This module imports no backend code (Guardrail 1) — it operates on the
:class:`~myocard_synthetic_egm_pipeline.backends.SimulationBackend`
Protocol and the public strategy types only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # Backend types are used only in annotations; importing them at
    # runtime would close the cycle backends → simulate.result →
    # simulate → simulate.runner → backends. ``from __future__ import
    # annotations`` makes all annotations strings so this is safe.
    from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend

from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import (
    bipolar_from_unipolar,
    downsample,
)
from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult
from myocard_synthetic_egm_pipeline.simulate.specs import (
    ActivationSource,
    ElectrodePlacement,
    GeometrySpec,
    SubstrateStrategy,
)


def run_single(
    *,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    backend: SimulationBackend,
    config: RunConfig,
    rng: np.random.Generator,
) -> SimulationResult:
    """Run one simulation and return a finished :class:`SimulationResult`.

    Parameters
    ----------
    geometry, substrate, activation, electrodes
        The four strategy specs; the backend's adapters translate them
        into the backend's native representation.
    backend
        Concrete :class:`SimulationBackend`. The runner is generic over
        which backend ran.
    config
        Per-run knobs (duration, output rate, AP calibration constant,
        oversampling factor).
    rng
        Reproducible source of randomness for any per-sim sampling the
        backend's strategy adapters do (e.g. fibrosis pattern draw).
    """
    raw = backend.simulate(
        geometry=geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        config=config,
        rng=rng,
    )

    # --- 1. Bipolar pairing at the capture rate -------------------------
    bipolar_capture = bipolar_from_unipolar(raw.unipolar_traces, raw.bipolar_pairs)

    # --- 2. Downsample to the output rate -------------------------------
    bipolar_target = downsample(
        bipolar_capture,
        source_fs_hz=raw.fs_capture_hz,
        target_fs_hz=config.output_fs_hz,
    )

    # --- 3. Truncate / pad to exact target samples ----------------------
    # The downsample step's sample count is approximate (integer rounding
    # of duration_ms * fs_hz). We pin to the exact target so every trace
    # in a dataset shares T.
    target_samples = round(config.trace_duration_ms * 1e-3 * config.output_fs_hz)
    n = bipolar_target.shape[0]
    if n >= target_samples:
        bipolar_final = bipolar_target[:target_samples]
    else:
        # Backend came in slightly short (rare; can happen with
        # non-integer downsample ratios near a boundary). Pad with
        # zeros so the bank stays uniform.
        pad = np.zeros(
            (target_samples - n, bipolar_target.shape[1]),
            dtype=bipolar_target.dtype,
        )
        bipolar_final = np.concatenate([bipolar_target, pad], axis=0)

    # --- 4. Reshape (T, n_pairs) → (n_pairs, T) + float32 ---------------
    bipolar_traces: npt.NDArray[np.float32] = bipolar_final.T.astype(np.float32, copy=False)

    # --- 5. Per-pair midpoints (physical mm) ----------------------------
    midpoints = _compute_pair_midpoints_mm(
        positions_mm=raw.electrode_positions_mm,
        bipolar_pairs=raw.bipolar_pairs,
    )

    # --- 6. Run-level provenance ----------------------------------------
    # Duck-typed extraction of per-sim scalars from the strategy specs.
    # Concretes that don't expose the field get None; downstream
    # consumers (ClassifierBank trace_metadata, SyntheticBank columns)
    # handle the None case.
    # Per-pair row index for downstream per-trace metadata. For
    # placements with a row_col helper (CenteredGrid2D) we use it;
    # other placements get None per pair.
    electrode_row_per_pair: list[int | None] = _electrode_row_per_pair(
        electrodes=electrodes, bipolar_pairs=raw.bipolar_pairs
    )

    run_metadata: dict[str, Any] = {
        "geometry_type": geometry.type,
        "substrate_type": substrate.type,
        "activation_type": activation.type,
        "electrode_type": electrodes.type,
        "electrode_height_mm": getattr(electrodes, "height_mm", None),
        "electrode_n_rows": getattr(electrodes, "n_rows", None),
        "electrode_n_cols": getattr(electrodes, "n_cols", None),
        "electrode_spacing_mm": getattr(electrodes, "spacing_mm", None),
        "electrode_row_per_pair": electrode_row_per_pair,
        "stim_edge": getattr(activation, "edge", None),
        "fiber_angle_rad": getattr(geometry, "fiber_angle_rad", None),
        "anisotropy_ratio": getattr(geometry, "anisotropy_ratio", None),
        "patch_size_mm": getattr(geometry, "size_mm", None),
        "patch_dr_mm": getattr(geometry, "dr_mm", None),
        "fibrosis_density_requested": getattr(substrate, "density", None),
        "backend_metadata": dict(raw.backend_metadata),
    }

    return SimulationResult(
        bipolar_traces=bipolar_traces,
        fs_hz=config.output_fs_hz,
        trace_duration_ms=config.trace_duration_ms,
        bipolar_pair_midpoints_mm=midpoints,
        substrate_mask=raw.substrate_mask,
        substrate_mask_dr_mm=raw.substrate_mask_dr_mm,
        electrode_positions_mm=raw.electrode_positions_mm,
        bipolar_pairs=raw.bipolar_pairs,
        substrate_realization_metadata=dict(raw.substrate_realization_metadata),
        run_metadata=run_metadata,
    )


def _electrode_row_per_pair(
    *,
    electrodes: ElectrodePlacement,
    bipolar_pairs: tuple[tuple[int, int], ...],
) -> list[int | None]:
    """Best-effort row lookup per bipolar pair.

    Placements that expose a ``row_col(electrode_index) -> (row, col)``
    helper (Phase 1's :class:`CenteredGrid2D`) get a concrete row index
    per pair (taken from the pair's first electrode). Future placements
    that don't define rows (e.g. an endocardial-surface 3D placement)
    get ``None`` per pair; downstream consumers fall back to
    ``electrode_positions_mm`` for spatial context.
    """
    row_col = getattr(electrodes, "row_col", None)
    if row_col is None:
        return [None] * len(bipolar_pairs)
    return [int(row_col(a)[0]) for a, _b in bipolar_pairs]


def _compute_pair_midpoints_mm(
    *,
    positions_mm: npt.NDArray[np.float64],
    bipolar_pairs: tuple[tuple[int, int], ...],
) -> npt.NDArray[np.float64]:
    """``(n_pairs, 3)`` array of midpoint coordinates per bipolar pair.

    Phase 1 assumes Cartesian midpoints (linear average of the two
    pole positions). 3D placements on a curved surface (Phase 5) may
    want a geodesic midpoint; when that lands, the
    :class:`ElectrodePlacement` Protocol grows a
    ``bipolar_pair_midpoints_mm()`` method and concretes override.
    """
    n_pairs = len(bipolar_pairs)
    midpoints = np.empty((n_pairs, 3), dtype=np.float64)
    for pair_idx, (a, b) in enumerate(bipolar_pairs):
        midpoints[pair_idx] = 0.5 * (positions_mm[a] + positions_mm[b])
    return midpoints
