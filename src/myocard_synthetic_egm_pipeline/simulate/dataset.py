"""N-simulation orchestrator.

:func:`generate_dataset` runs ``n_simulations`` independent simulations
with per-sim sampled parameters (fibrosis density, activation edge,
electrode height), collects each
:class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationResult`,
applies the configured
:class:`~myocard_synthetic_egm_pipeline.simulate.label_policy.LabelPolicy`,
and returns a flat :class:`DatasetResult` ready for the storage layer.

Sampling axes (per ``simulator_v1_spec``):

- **fibrosis density** — uniform over ``fibrosis_density_range``
  (default ``[0.0, 0.5]`` to stay safely below the propagation-failure
  regime). Optional ``fraction_healthy`` forces a portion of sims to
  ``density = 0`` exactly, useful for class-balancing.
- **stimulus edge** — uniform over the four edges. Optional
  ``fixed_stim_edge`` pins the edge for calibration runs.
- **electrode height** — uniform over the placement spec's
  ``height_mm_range``. Sampled inside
  :meth:`~myocard_synthetic_egm_pipeline.simulate.specs.CenteredGrid2D.sample`.

Per-sim RNG is derived deterministically from ``master_seed`` so the
whole dataset is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

if TYPE_CHECKING:
    # Backend types are used only in annotations; importing them at
    # runtime would close the cycle backends → simulate.result →
    # simulate → simulate.dataset → backends. ``from __future__ import
    # annotations`` makes all annotations strings so this is safe.
    from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend

from myocard_synthetic_egm_pipeline.constants import (
    DEFAULT_ELECTRODE_GRID_COLS,
    DEFAULT_ELECTRODE_GRID_ROWS,
    DEFAULT_ELECTRODE_HEIGHT_MM_RANGE,
    DEFAULT_ELECTRODE_SPACING_MM,
    DEFAULT_FIBROSIS_DENSITY_RANGE,
)
from myocard_synthetic_egm_pipeline.simulate.label_policy import LabelPolicy
from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult
from myocard_synthetic_egm_pipeline.simulate.runner import run_single
from myocard_synthetic_egm_pipeline.simulate.specs import (
    EDGES,
    CenteredGrid2D,
    Edge,
    GeometrySpec,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
    random_edge,
)


@dataclass(frozen=True)
class DatasetConfig:
    """N-simulation orchestrator config.

    Phase 1 hard-codes the substrate type
    (:class:`~myocard_synthetic_egm_pipeline.simulate.specs.UniformRandomFibrosis`),
    activation type
    (:class:`~myocard_synthetic_egm_pipeline.simulate.specs.PlanarEdgeStimulus`),
    and electrode placement type
    (:class:`~myocard_synthetic_egm_pipeline.simulate.specs.CenteredGrid2D`).
    Phase 2+ generalizes via the CLI's strategy-type dispatch.

    Attributes
    ----------
    n_simulations
        Number of independent simulations.
    geometry
        Static geometry spec — shared across all simulations.
    fibrosis_density_range
        ``(lo, hi)`` — per-sim density sampled uniformly.
    fraction_healthy
        Optional override: forces this fraction of sims to ``density=0``
        exactly (the rest sample from ``fibrosis_density_range``).
        Default 0.0. Set e.g. to 0.3 to match the v1-spec
        ~30 % healthy / ~70 % fibrotic split.
    fixed_stim_edge
        If set, use this edge for every simulation. Default ``None``:
        randomise per-sim.
    electrode_n_rows, electrode_n_cols, electrode_spacing_mm, electrode_height_mm_range
        Pass-through to :meth:`CenteredGrid2D.sample`.
    label_policy
        Concrete :class:`LabelPolicy` applied to each simulation's
        result. Every simulation must return the same ``labels_dict``;
        a disagreement raises.
    run_config
        Per-run knobs forwarded to the backend through
        :func:`run_single`.
    master_seed
        Master RNG seed; per-sim seeds derived from this.
    show_progress
        If ``True``, render a tqdm bar over the simulation loop.
    """

    n_simulations: int

    geometry: GeometrySpec

    label_policy: LabelPolicy
    run_config: RunConfig

    fibrosis_density_range: tuple[float, float] = DEFAULT_FIBROSIS_DENSITY_RANGE
    fraction_healthy: float = 0.0

    fixed_stim_edge: Edge | None = None

    electrode_n_rows: int = DEFAULT_ELECTRODE_GRID_ROWS
    electrode_n_cols: int = DEFAULT_ELECTRODE_GRID_COLS
    electrode_spacing_mm: float = DEFAULT_ELECTRODE_SPACING_MM
    electrode_height_mm_range: tuple[float, float] = DEFAULT_ELECTRODE_HEIGHT_MM_RANGE

    master_seed: int = 0
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be >= 1.")
        lo, hi = self.fibrosis_density_range
        if not 0.0 <= lo <= hi < 1.0:
            raise ValueError("fibrosis_density_range must satisfy 0 <= lo <= hi < 1.")
        if not 0.0 <= self.fraction_healthy <= 1.0:
            raise ValueError("fraction_healthy must be in [0, 1].")
        if self.fixed_stim_edge is not None and self.fixed_stim_edge not in EDGES:
            raise ValueError(
                f"fixed_stim_edge must be None or one of {EDGES}, got {self.fixed_stim_edge!r}."
            )


@dataclass(frozen=True)
class DatasetResult:
    """Output of :func:`generate_dataset`.

    Attributes
    ----------
    results
        One :class:`SimulationResult` per simulation, ordered by
        ``sim_id``.
    labels
        ``(N_total,)`` int64 — flat labels aligned with the trace order
        in ``results`` (sim 0 traces, then sim 1 traces, ...). N_total
        is the sum of ``r.n_pairs`` across ``results``.
    labels_dict
        Integer-to-name mapping shared across the dataset (every
        per-sim LabelPolicy.apply call must agree).
    sim_ids
        ``(N_total,)`` int64 — sim_id per trace.
    pair_indices
        ``(N_total,)`` int64 — bipolar-pair index within the trace's
        own simulation (0 .. result.n_pairs - 1).
    seeds
        ``(n_simulations,)`` int64 — per-sim RNG seed derived from
        ``master_seed``. Kept for reproducibility provenance.
    """

    results: list[SimulationResult]
    labels: npt.NDArray[np.int64]
    labels_dict: dict[int, str]
    sim_ids: npt.NDArray[np.int64]
    pair_indices: npt.NDArray[np.int64]
    seeds: npt.NDArray[np.int64]


def generate_dataset(
    *,
    config: DatasetConfig,
    backend: SimulationBackend,
) -> DatasetResult:
    """Run ``config.n_simulations`` simulations and return a flat dataset.

    The total trace count is
    ``n_simulations * electrode_n_rows * (electrode_n_cols - 1)`` for a
    centered-grid placement.
    """
    master_rng = np.random.default_rng(config.master_seed)
    lo, hi = config.fibrosis_density_range

    results: list[SimulationResult] = []
    label_chunks: list[npt.NDArray[np.int64]] = []
    sim_id_chunks: list[npt.NDArray[np.int64]] = []
    pair_idx_chunks: list[npt.NDArray[np.int64]] = []
    seeds_list: list[int] = []
    labels_dict: dict[int, str] | None = None

    iterator: Any = range(config.n_simulations)
    if config.show_progress:
        iterator = tqdm(iterator, desc="Simulating", unit="sim")

    for sim_id in iterator:
        # Per-sim deterministic RNG.
        sim_seed = int(master_rng.integers(0, 2**31 - 1))
        sim_rng = np.random.default_rng(sim_seed)
        seeds_list.append(sim_seed)

        # Sample fibrosis density.
        if config.fraction_healthy > 0.0 and sim_rng.uniform() < config.fraction_healthy:
            density = 0.0
        else:
            density = float(sim_rng.uniform(lo, hi))
        substrate = UniformRandomFibrosis(density=density)

        # Sample stim edge (or use the fixed override).
        edge: Edge = (
            config.fixed_stim_edge if config.fixed_stim_edge is not None else random_edge(sim_rng)
        )
        activation = PlanarEdgeStimulus(edge=edge)

        # Sample electrode placement (height drawn inside .sample).
        # The geometry must be a Patch2DGeometry for CenteredGrid2D —
        # the backend will reject other combinations at simulate time;
        # here we trust the runner-level dispatch to surface mismatches.
        from myocard_synthetic_egm_pipeline.simulate.specs import Patch2DGeometry

        if not isinstance(config.geometry, Patch2DGeometry):
            raise TypeError(
                "Phase 1 CenteredGrid2D requires a Patch2DGeometry; "
                f"got geometry type {config.geometry.type!r}."
            )
        electrodes = CenteredGrid2D.sample(
            geometry=config.geometry,
            n_rows=config.electrode_n_rows,
            n_cols=config.electrode_n_cols,
            spacing_mm=config.electrode_spacing_mm,
            height_mm_range=config.electrode_height_mm_range,
            rng=sim_rng,
        )

        # Run one simulation.
        result = run_single(
            geometry=config.geometry,
            substrate=substrate,
            activation=activation,
            electrodes=electrodes,
            backend=backend,
            config=config.run_config,
            rng=sim_rng,
        )

        # Stamp sim_id into the result's run_metadata for downstream
        # provenance. The frozen dataclass means we mutate via dict
        # copy on the metadata field.
        result.run_metadata["sim_id"] = sim_id
        result.run_metadata["sim_seed"] = sim_seed

        results.append(result)

        # Apply the label policy.
        sim_labels, sim_labels_dict = config.label_policy.apply(result)
        if labels_dict is None:
            labels_dict = sim_labels_dict
        elif labels_dict != sim_labels_dict:
            raise ValueError(
                f"LabelPolicy returned disagreeing labels_dicts across simulations: "
                f"sim {sim_id} returned {sim_labels_dict!r}; earlier sims returned "
                f"{labels_dict!r}."
            )
        if sim_labels.shape[0] != result.n_pairs:
            raise ValueError(
                f"LabelPolicy.apply returned {sim_labels.shape[0]} labels "
                f"for {result.n_pairs} pairs (sim {sim_id})."
            )

        label_chunks.append(sim_labels.astype(np.int64, copy=False))
        sim_id_chunks.append(np.full(result.n_pairs, sim_id, dtype=np.int64))
        pair_idx_chunks.append(np.arange(result.n_pairs, dtype=np.int64))

    if labels_dict is None:
        labels_dict = {}
    if label_chunks:
        labels = np.concatenate(label_chunks)
        sim_ids = np.concatenate(sim_id_chunks)
        pair_indices = np.concatenate(pair_idx_chunks)
    else:
        labels = np.zeros(0, dtype=np.int64)
        sim_ids = np.zeros(0, dtype=np.int64)
        pair_indices = np.zeros(0, dtype=np.int64)

    return DatasetResult(
        results=results,
        labels=labels,
        labels_dict=labels_dict,
        sim_ids=sim_ids,
        pair_indices=pair_indices,
        seeds=np.asarray(seeds_list, dtype=np.int64),
    )


__all__ = ["DatasetConfig", "DatasetResult", "generate_dataset"]
