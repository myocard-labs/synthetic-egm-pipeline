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

from collections.abc import Sequence
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
from myocard_synthetic_egm_pipeline.mixer import MixerConfig
from myocard_synthetic_egm_pipeline.simulate.calibration import ModelCard
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
from myocard_synthetic_egm_pipeline.simulate.sweep import (
    DesignCell,
    InfeasibleCell,
    SweepConfig,
    SweepConfigError,
    apply_cell,
    build_design,
)
from myocard_synthetic_egm_pipeline.simulate.tuning import (
    InfeasibleTargetError,
    TunableRun,
    set_value,
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


# ---------------------------------------------------------------------------
# The sweep harness — a loop AROUND generate_dataset, never inside it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepRun:
    """One design cell's bank, and the parameters it was generated at."""

    cell: DesignCell
    config: DatasetConfig
    result: DatasetResult
    mixer: MixerConfig | None = None
    """The mixer **as this cell set it**, when the design writes a noise knob.

    Carried rather than left to the caller because a cell that swept
    ``mix.snr_db_range`` and was then mixed with the config's original range
    would record one SNR distribution and contain another — the design point
    would be a fiction, and nothing downstream could see it.
    """


@dataclass(frozen=True)
class SweepResult:
    """Every cell that ran, and every cell that could not.

    ``infeasible`` is a first-class half of the answer, not an error log. A
    design with its failures discarded fits an emulator on a biased subset of
    the space and cannot see the boundary; keeping them means the boundary can
    be described.
    """

    runs: tuple[SweepRun, ...]
    infeasible: tuple[InfeasibleCell, ...]

    @property
    def n_cells(self) -> int:
        return len(self.runs) + len(self.infeasible)


def generate_sweep(
    *,
    config: DatasetConfig,
    backend: SimulationBackend,
    sweep: SweepConfig,
    card: ModelCard | None = None,
    mixer: MixerConfig | None = None,
    position_generator: ActivationPositionGenerator | None = None,
) -> SweepResult:
    """Generate one dataset per design cell.

    **A loop around :func:`generate_dataset`, not a change to it.** Each cell
    writes its values onto a bundle through the path resolver, producing a new
    :class:`DatasetConfig`, and that config is generated from by exactly the
    code path a non-swept run uses. Nothing about the per-simulation draw is
    replaced — which is what makes the no-sweep result bit-identical, and it is
    asserted rather than assumed.

    Every cell runs ``config.n_simulations`` simulations, so a design of ``c``
    cells costs ``c`` times a plain run.

    A cell the resolver refuses is recorded and the sweep continues. That is the
    one place this function swallows an exception, and it swallows it into a
    result rather than into silence.
    """
    design = build_design(sweep)
    _reject_unwritable_knobs(TunableRun(dataset=config, card=card, mixer=mixer), design)

    runs: list[SweepRun] = []
    infeasible: list[InfeasibleCell] = []

    for cell in design:
        bundle = TunableRun(dataset=config, card=card, mixer=mixer)
        try:
            applied = apply_cell(bundle, cell)
        except ValueError as exc:
            # Broad on purpose. The resolver raises several distinct types and
            # a backend may add more; a harness that caught only the ones known
            # today would turn a new kind of infeasibility into a crashed sweep
            # rather than a recorded point.
            path, value = _blame(cell, exc)
            infeasible.append(
                InfeasibleCell(
                    cell=cell,
                    path=path,
                    value=value,
                    reason=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            continue

        cell_config = applied.dataset
        assert isinstance(cell_config, DatasetConfig)
        runs.append(
            SweepRun(
                cell=cell,
                config=cell_config,
                mixer=applied.mixer,
                result=generate_dataset(
                    config=cell_config,
                    backend=backend,
                    position_generator=position_generator,
                ),
            )
        )

    return SweepResult(runs=tuple(runs), infeasible=tuple(infeasible))


def _reject_unwritable_knobs(run: TunableRun, design: Sequence[DesignCell]) -> None:
    """Refuse a design naming a knob that can never be written, before running any.

    **A path-level refusal is a malformed design; a value-level one is a
    boundary.** The distinction matters because every cell carries every knob,
    at nominal where it is not the varied one — so a single unwritable path
    would make *every* cell infeasible and the sweep would report a design-space
    boundary where it actually has a typo. One knob spelled wrong would look
    like a physics finding.

    So the knob list is dry-run once against the bundle. Anything the resolver
    refuses for what a path *is* — derived, redrawn per simulation, numerical —
    aborts the sweep with the resolver's own explanation. Only
    :class:`InfeasibleTargetError`, which is about the *value*, survives to be
    recorded per cell.
    """
    if not design:
        return
    probe = design[0]
    for path in sorted(probe.values):
        try:
            set_value(run, path, probe.values[path])
        except InfeasibleTargetError:
            # About this value, not this path. A different level may well work,
            # and if none do, that is a boundary worth recording cell by cell.
            continue
        except ValueError as exc:
            raise SweepConfigError(
                f"sweep.knobs names {path!r}, which cannot be written at all: {exc}"
            ) from exc


def merge_sweep_results(runs: Sequence[SweepRun]) -> DatasetResult:
    """Concatenate every design cell's simulations into one dataset.

    **A sweep produces one bank, not one per cell**, because a bank is the unit
    that holds a design: its theta-spec states which knobs *this bank's sweep*
    varied, and an unswept bank is defined as one with an empty knob list. A
    bank per cell would give each file a spec describing a one-point sweep,
    which is neither of the two things the field can express, and the design
    would be recorded nowhere.

    Nothing is lost by merging. The synthetic bank already stores the generation
    config **per simulation**, so each cell's parameter values survive
    simulation by simulation without needing an artifact of their own.

    ``simulation_id`` is offset per cell. It is the join key between the
    classifier bank and its theta partner and the schema requires it to be
    unique within a bank, so ids restarting at zero in every cell would collide
    — silently, since both banks would agree on the wrong answer.
    """
    results: list[SimulationResult] = []
    labels: list[npt.NDArray[np.int64]] = []
    simulation_ids: list[npt.NDArray[np.int64]] = []
    pair_indices: list[npt.NDArray[np.int64]] = []
    seeds: list[npt.NDArray[np.int64]] = []
    labels_dict: dict[int, str] = {}
    offset = 0

    for run in runs:
        cell = run.result
        if labels_dict and cell.labels_dict != labels_dict:
            raise ValueError(
                "design cells disagree about their label mapping "
                f"({cell.labels_dict!r} against {labels_dict!r}); they cannot be "
                "merged into one bank, because a bank has a single labels dict."
            )
        labels_dict = labels_dict or cell.labels_dict

        for local_id, result in enumerate(cell.results):
            # Restamped, not merely offset in the id column: the per-simulation
            # provenance carries its own copy, and a bank whose two records of
            # one id disagreed would be worse than one that never had them.
            result.run_metadata["simulation_id"] = offset + local_id
            result.run_metadata["design_cell"] = run.cell.index
            results.append(result)

        labels.append(cell.labels)
        simulation_ids.append(cell.simulation_ids + offset)
        pair_indices.append(cell.pair_indices)
        seeds.append(cell.seeds)
        offset += len(cell.results)

    empty = np.zeros(0, dtype=np.int64)
    return DatasetResult(
        results=results,
        labels=np.concatenate(labels) if labels else empty,
        labels_dict=labels_dict,
        simulation_ids=np.concatenate(simulation_ids) if simulation_ids else empty,
        pair_indices=np.concatenate(pair_indices) if pair_indices else empty,
        seeds=np.concatenate(seeds) if seeds else empty,
    )


def _blame(cell: DesignCell, exc: ValueError) -> tuple[str, object]:
    """Which knob the failure is about, from the exception if it says so.

    The resolver's own errors carry the path; anything else is attributed to
    the cell's varied knob, which is the honest guess for a one-at-a-time
    design and is recorded as such rather than as a certainty.
    """
    path = getattr(exc, "path", None)
    if isinstance(path, str) and path in cell.values:
        return path, cell.values[path]
    fallback = cell.varied if cell.varied is not None else "(baseline)"
    return fallback, cell.values.get(fallback)
