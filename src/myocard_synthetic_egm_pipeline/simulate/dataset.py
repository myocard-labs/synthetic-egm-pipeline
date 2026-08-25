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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

if TYPE_CHECKING:
    # Backend types are used only in annotations; importing them at
    # runtime would close the cycle backends → simulate.result →
    # simulate → simulate.dataset → backends. ``from __future__ import
    # annotations`` makes all annotations strings so this is safe.
    from myocard_egm_signal import ActivationPositionGenerator, DetectionPreprocessor

    from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend

from myocard_synthetic_egm_pipeline.constants import (
    DEFAULT_ELECTRODE_GRID_COLS,
    DEFAULT_ELECTRODE_GRID_ROWS,
    DEFAULT_ELECTRODE_HEIGHT_MM_RANGE,
    DEFAULT_ELECTRODE_SPACING_MM,
    DEFAULT_FIBROSIS_DENSITY_RANGE,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import CellModelSpec
from myocard_synthetic_egm_pipeline.simulate.label_policy import LabelPolicy
from myocard_synthetic_egm_pipeline.simulate.probe import ProbeGrid
from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult
from myocard_synthetic_egm_pipeline.simulate.runner import run_probe_sweep, run_single
from myocard_synthetic_egm_pipeline.simulate.specs import (
    EDGES,
    CenteredGrid2D,
    Edge,
    GeometrySpec,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
    random_edge,
)


def _default_cell_model() -> CellModelSpec:
    """The shipped parameterisation, resolved lazily.

    A default rather than a required field so existing constructions keep
    working; resolved through the card so the default is the *calibrated*
    physics and not a constant that can drift away from it.
    """
    from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card

    return load_model_card("af_remodelled_220ms", dr_mm=0.25, dr_model_units=0.25).solved


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
    detection_preprocessor
        Detection curve the crop anchors on, for every simulation in the run.
        ``None`` means
        :func:`~myocard_synthetic_egm_pipeline.simulate.cropping.default_preprocessor`.
    master_seed
        Master RNG seed; per-sim seeds derived from this.
    show_progress
        If ``True``, render a tqdm bar over the simulation loop.
    """

    n_simulations: int

    geometry: GeometrySpec

    label_policy: LabelPolicy
    run_config: RunConfig
    cell_model: CellModelSpec = field(default_factory=lambda: _default_cell_model())
    """Membrane model for every simulation in the dataset — the fifth spec.

    Dataset-level rather than per-simulation for now: at APD 220 > T = 192 the
    repolarisation marker is already outside the analysis window, so sampling
    APD per simulation buys physiological variation rather than correctness.
    When it is scheduled, this becomes a sampled field like ``density_range``
    — and the planned move of the physiological targets onto
    ``SubstrateStrategy`` makes that nearly free, because the substrate is
    already drawn per simulation.
    """

    detection_preprocessor: DetectionPreprocessor | None = None
    """Detection curve every crop in this run anchors on.

    ``None`` keeps ``cropping.default_preprocessor()`` — ``RectifiedDerivative``
    — so a run that names no curve behaves exactly as every run before the knob
    existed. Read only when ``generate_dataset`` is given a position generator:
    with no crop there is nothing to detect for.

    Stateless, unlike the position generator, so one instance is shared across
    the whole run rather than being a per-simulation object.

    **Not recorded in either bank.** It changes where the window is cut and
    therefore the stored waveform, but neither schema has a field for it; until
    one gains it, the bank's ``description`` is the only record.
    """

    probe_grid: ProbeGrid | None = None
    """Positional-sensitivity sweep — a diagnostic bank, not training data.

    Set, the run emits **one logical simulation per grid point**, all computed
    from a single solve and all carrying the same seed, so the only thing that
    differs between them is where the window was cut. ``None`` is the ordinary
    generative run.

    Mutually exclusive with the position generator ``generate_dataset`` is given,
    and with ``n_simulations > 1``: a probe answers a question about *one*
    substrate, so a second solve is a mis-specified study rather than extra data.
    Note that ``n_simulations`` counts **solves**, not written simulations — a
    probe is configured with 1 and writes ``n_points``.
    """

    fibrosis_density_range: tuple[float, float] = DEFAULT_FIBROSIS_DENSITY_RANGE
    fraction_healthy: float = 0.0

    fixed_stim_edge: Edge | None = None
    #: Delay before the stimulus fires, in ms. 0 = fire at t=0 (the historical
    #: behaviour). Held in ms because every other duration in the config is in
    #: ms; converted to the solver's model units at spec construction, where
    #: ``ap_time_unit_ms`` is in hand. **Diagnostic first:** a delay buys only
    #: *resting* lead-in, since phi_e sums membrane current over the whole mesh
    #: and nothing is depolarising yet.
    stimulus_delay_ms: float = 0.0

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
        if self.stimulus_delay_ms < 0:
            raise ValueError("stimulus_delay_ms must be >= 0.")
        if self.fixed_stim_edge is not None and self.fixed_stim_edge not in EDGES:
            raise ValueError(
                f"fixed_stim_edge must be None or one of {EDGES}, got {self.fixed_stim_edge!r}."
            )
        if self.probe_grid is not None and self.n_simulations != 1:
            raise ValueError(
                f"a probe sweep runs exactly one simulation; got n_simulations="
                f"{self.n_simulations}. The sweep holds morphology, substrate and seed "
                "constant so the only thing varying across its traces is the crop "
                "offset — a second simulation would vary the substrate too, and the "
                "positional curve would no longer be attributable to position."
            )


@dataclass(frozen=True)
class DatasetResult:
    """Output of :func:`generate_dataset`.

    Attributes
    ----------
    results
        One :class:`SimulationResult` per simulation, ordered by
        ``simulation_id``.
    labels
        ``(N_total,)`` int64 — flat labels aligned with the trace order
        in ``results`` (sim 0 traces, then sim 1 traces, ...). N_total
        is the sum of ``r.n_pairs`` across ``results``.
    labels_dict
        Integer-to-name mapping shared across the dataset (every
        per-sim LabelPolicy.apply call must agree).
    simulation_ids
        ``(N_total,)`` int64 — simulation_id per trace.
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
    simulation_ids: npt.NDArray[np.int64]
    pair_indices: npt.NDArray[np.int64]
    seeds: npt.NDArray[np.int64]


def generate_dataset(
    *,
    config: DatasetConfig,
    backend: SimulationBackend,
    position_generator: ActivationPositionGenerator | None = None,
) -> DatasetResult:
    """Run ``config.n_simulations`` simulations and return a flat dataset.

    The total trace count is
    ``n_simulations * electrode_n_rows * (electrode_n_cols - 1)`` for a
    centered-grid placement.

    ``position_generator`` enables controlled-position cropping. The
    **same** generator instance is handed to every simulation on purpose: it is
    stateful, so a shared instance advances one stream across the whole run and
    every trace draws an independent position. Constructing one per simulation
    would restart the stream and give each simulation the same sequence of
    positions — a regularity indistinguishable, downstream, from not having
    varied them at all.

    The curve those positions are measured against is
    ``config.detection_preprocessor``; it rides on the config rather than
    alongside it here precisely because it is stateless.

    ``config.probe_grid`` switches the run to the positional-sensitivity
    probe: **one solve, one logical simulation per crop offset**, all sharing
    a seed. The probe ignores ``position_generator`` because the two are
    mutually exclusive at the config layer — a sweep requests its offsets rather
    than drawing them.
    """
    master_rng = np.random.default_rng(config.master_seed)

    seeded_results = (
        _simulate_probe_sweep(config=config, backend=backend, master_rng=master_rng)
        if config.probe_grid is not None
        else _simulate_sampled(
            config=config,
            backend=backend,
            position_generator=position_generator,
            master_rng=master_rng,
        )
    )

    results: list[SimulationResult] = []
    label_chunks: list[npt.NDArray[np.int64]] = []
    simulation_id_chunks: list[npt.NDArray[np.int64]] = []
    pair_idx_chunks: list[npt.NDArray[np.int64]] = []
    seeds_list: list[int] = []
    labels_dict: dict[int, str] | None = None

    for simulation_id, (result, sim_seed) in enumerate(seeded_results):
        seeds_list.append(sim_seed)

        # Stamp simulation_id into the result's run_metadata for downstream
        # provenance. The frozen dataclass means we mutate via dict
        # copy on the metadata field.
        result.run_metadata["simulation_id"] = simulation_id
        result.run_metadata["sim_seed"] = sim_seed

        results.append(result)

        # Apply the label policy.
        sim_labels, sim_labels_dict = config.label_policy.apply(result)
        if labels_dict is None:
            labels_dict = sim_labels_dict
        elif labels_dict != sim_labels_dict:
            raise ValueError(
                f"LabelPolicy returned disagreeing labels_dicts across simulations: "
                f"sim {simulation_id} returned {sim_labels_dict!r}; earlier sims returned "
                f"{labels_dict!r}."
            )
        if sim_labels.shape[0] != result.n_pairs:
            raise ValueError(
                f"LabelPolicy.apply returned {sim_labels.shape[0]} labels "
                f"for {result.n_pairs} pairs (sim {simulation_id})."
            )

        label_chunks.append(sim_labels.astype(np.int64, copy=False))
        simulation_id_chunks.append(np.full(result.n_pairs, simulation_id, dtype=np.int64))
        pair_idx_chunks.append(np.arange(result.n_pairs, dtype=np.int64))

    if labels_dict is None:
        labels_dict = {}
    if label_chunks:
        labels = np.concatenate(label_chunks)
        simulation_ids = np.concatenate(simulation_id_chunks)
        pair_indices = np.concatenate(pair_idx_chunks)
    else:
        labels = np.zeros(0, dtype=np.int64)
        simulation_ids = np.zeros(0, dtype=np.int64)
        pair_indices = np.zeros(0, dtype=np.int64)

    return DatasetResult(
        results=results,
        labels=labels,
        labels_dict=labels_dict,
        simulation_ids=simulation_ids,
        pair_indices=pair_indices,
        seeds=np.asarray(seeds_list, dtype=np.int64),
    )


def _sample_specs(
    *, config: DatasetConfig, sim_rng: np.random.Generator
) -> tuple[UniformRandomFibrosis, PlanarEdgeStimulus, CenteredGrid2D]:
    """Draw one simulation's substrate, activation and electrode placement."""
    lo, hi = config.fibrosis_density_range
    if config.fraction_healthy > 0.0 and sim_rng.uniform() < config.fraction_healthy:
        density = 0.0
    else:
        density = float(sim_rng.uniform(lo, hi))
    substrate = UniformRandomFibrosis(density=density)

    edge: Edge = (
        config.fixed_stim_edge if config.fixed_stim_edge is not None else random_edge(sim_rng)
    )
    activation = PlanarEdgeStimulus(
        edge=edge,
        time_model_units=config.cell_model.ms_to_model_time(config.stimulus_delay_ms),
    )

    # The geometry must be a Patch2DGeometry for CenteredGrid2D — the backend
    # will reject other combinations at simulate time; here we trust the
    # runner-level dispatch to surface mismatches.
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
    return substrate, activation, electrodes


def _simulate_sampled(
    *,
    config: DatasetConfig,
    backend: SimulationBackend,
    position_generator: ActivationPositionGenerator | None,
    master_rng: np.random.Generator,
) -> list[tuple[SimulationResult, int]]:
    """The ordinary run: ``n_simulations`` independent draws, one solve each."""
    iterator: Any = range(config.n_simulations)
    if config.show_progress:
        iterator = tqdm(iterator, desc="Simulating", unit="sim")

    seeded: list[tuple[SimulationResult, int]] = []
    for _ in iterator:
        sim_seed = int(master_rng.integers(0, 2**31 - 1))
        sim_rng = np.random.default_rng(sim_seed)
        substrate, activation, electrodes = _sample_specs(config=config, sim_rng=sim_rng)
        seeded.append(
            (
                run_single(
                    geometry=config.geometry,
                    substrate=substrate,
                    activation=activation,
                    electrodes=electrodes,
                    backend=backend,
                    cell_model=config.cell_model,
                    config=config.run_config,
                    rng=sim_rng,
                    position_generator=position_generator,
                    detection_preprocessor=config.detection_preprocessor,
                ),
                sim_seed,
            )
        )
    return seeded


def _simulate_probe_sweep(
    *,
    config: DatasetConfig,
    backend: SimulationBackend,
    master_rng: np.random.Generator,
) -> list[tuple[SimulationResult, int]]:
    """The probe: one solve, one logical simulation per grid point.

    Every returned result carries the **same seed**, because they came from one
    solve of one substrate — which is what makes a sweep identifiable in the
    written bank. ``simulation_id`` still runs ``0..n_points-1``, so
    ``(simulation_id, pair_index)`` stays unique and egm-studio's loader can
    join the bank pair.
    """
    assert config.probe_grid is not None
    sim_seed = int(master_rng.integers(0, 2**31 - 1))
    sim_rng = np.random.default_rng(sim_seed)
    substrate, activation, electrodes = _sample_specs(config=config, sim_rng=sim_rng)

    results = run_probe_sweep(
        geometry=config.geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        backend=backend,
        cell_model=config.cell_model,
        config=config.run_config,
        rng=sim_rng,
        grid=config.probe_grid,
        detection_preprocessor=config.detection_preprocessor,
    )
    return [(result, sim_seed) for result in results]


__all__ = ["DatasetConfig", "DatasetResult", "generate_dataset"]
