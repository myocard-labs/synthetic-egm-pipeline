"""Producer specs → ``synthetic_bank`` 2.0 per-simulation config models.

``synthetic_bank`` 2.0 stores generation parameters as one typed,
``type``-discriminated object per *stable generation function* —
geometry, cell model, substrate, activation, electrodes, backend, label
policy — recorded once per simulation rather than flattened into
per-trace columns. This module is the single place that translates this
repo's strategy specs into those egm-contracts models.

Why it is its own module rather than part of ``builders.py``: the
mapping is the **producer's half of a cross-repo contract**, and the two
sides are meant to be readable against each other. egm-contracts' field
names deliberately follow ``specs.py`` (CL-087), so most of this file is
one-to-one; the places where it *isn't* are the interesting ones and are
commented individually.

Everything here is pure — no I/O, no HDF5, no backend imports
(Guardrails 1 and the no-HDF5-in-this-repo rule). Each function takes a
spec object and returns a Pydantic model.

**Codegen shape note.** Single-variant unions in the schema codegen to a
``RootModel`` wrapper (``Geometry``, ``Substrate``, ``Electrodes``,
``Backend``) while multi-variant unions codegen to a proper discriminated
union used directly (``cell_model``, ``activation``, ``label_policy``).
That asymmetry is egm-contracts' FB-15, not something this module can
fix, so the wrappers are applied explicitly at the call site in
:func:`build_simulation_columns` rather than hidden inside each mapper —
otherwise the mappers would return two different kinds of thing for no
reason visible here.
"""

from __future__ import annotations

from typing import Any

from myocard_egm_contracts._generated.python.synthetic_bank import (
    AlievPanfilovCellModel,
    Backend,
    BipolarPair,
    CourtemancheCellModel,
    ElectrodeIndice,
    Electrodes,
    FinitewaveBackend,
    Geometry,
    GlobalDensityLabel,
    LocalDensityLabel,
    PlanarEdgeActivation,
    PointActivation,
    PositionMm,
    S1S2Activation,
    Simulations,
    Substrate,
    SubstrateSummary,
    Threshold,
)
from myocard_egm_contracts._generated.python.synthetic_bank import (
    Edge as ContractsEdge,
)
from myocard_egm_contracts._generated.python.synthetic_bank import (
    Patch2DGeometry as ContractsPatch2DGeometry,
)
from myocard_egm_contracts._generated.python.synthetic_bank import (
    UniformRandomFibrosis as ContractsUniformRandomFibrosis,
)

from myocard_synthetic_egm_pipeline.simulate.label_policy import (
    GlobalDensityLabel as ProducerGlobalDensityLabel,
)
from myocard_synthetic_egm_pipeline.simulate.label_policy import (
    LabelPolicy,
)
from myocard_synthetic_egm_pipeline.simulate.label_policy import (
    LocalDensityLabel as ProducerLocalDensityLabel,
)
from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult
from myocard_synthetic_egm_pipeline.simulate.specs import (
    ActivationSource,
    CenteredGrid2D,
    ElectrodePlacement,
    GeometrySpec,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    SubstrateStrategy,
    UniformRandomFibrosis,
)

# Union of every activation variant the 2.0 schema accepts. Phase 1.5
# builds only the planar-edge one; `point` / `s1s2` arrive with SEP7.
ActivationModel = PlanarEdgeActivation | PointActivation | S1S2Activation
LabelPolicyModel = GlobalDensityLabel | LocalDensityLabel
CellModelModel = CourtemancheCellModel | AlievPanfilovCellModel


class UnsupportedSpecError(ValueError):
    """A spec has no ``synthetic_bank`` 2.0 representation yet.

    Raised rather than silently writing a partial config: a bank whose
    per-simulation config doesn't describe the simulation that produced
    it is precisely the failure the 2.0 restructure exists to prevent.
    """


# ---------------------------------------------------------------------------
# Per-function mappers
# ---------------------------------------------------------------------------


def geometry_model(geometry: GeometrySpec) -> ContractsPatch2DGeometry:
    """Map a geometry spec to its contracts model."""
    if not isinstance(geometry, Patch2DGeometry):
        raise UnsupportedSpecError(
            f"synthetic_bank 2.0 has no geometry variant for {geometry.type!r}. "
            "Add the variant to egm-contracts' simulation_config schema first."
        )
    return ContractsPatch2DGeometry(
        type="patch_2d",
        size_mm=geometry.size_mm,
        dr_mm=geometry.dr_mm,
        fiber_angle_rad=geometry.fiber_angle_rad,
        anisotropy_ratio=geometry.anisotropy_ratio,
    )


def substrate_model(substrate: SubstrateStrategy) -> ContractsUniformRandomFibrosis:
    """Map a substrate spec to its contracts model.

    This is the **requested** substrate. What the draw actually realized
    is a separate object (:func:`substrate_summary_model`) because grid
    discretization makes the two genuinely differ, and the label is
    computed from the realized one.
    """
    if not isinstance(substrate, UniformRandomFibrosis):
        raise UnsupportedSpecError(
            f"synthetic_bank 2.0 has no substrate variant for {substrate.type!r}. "
            "Add the variant to egm-contracts' simulation_config schema first."
        )
    return ContractsUniformRandomFibrosis(
        type="uniform_random_fibrosis",
        density=substrate.density,
    )


def substrate_summary_model(realization_metadata: dict[str, Any]) -> SubstrateSummary:
    """Map the backend's realization metadata to the summary object.

    Both fields are optional in-schema: a backend that doesn't report a
    realized density leaves it absent rather than defaulting to 0.0,
    which would be a real density meaning "no fibrosis".
    """
    realized = realization_metadata.get("density_realized")
    n_fibrotic = realization_metadata.get("n_fibrotic_nodes")
    return SubstrateSummary(
        realized_density=None if realized is None else float(realized),
        n_fibrotic_nodes=None if n_fibrotic is None else int(n_fibrotic),
    )


def activation_model(activation: ActivationSource) -> ActivationModel:
    """Map an activation spec to its contracts model.

    ``PlanarEdgeStimulus`` holds a single ``edge``; the schema's
    ``planar_edge.edges`` is a **list** (CL-087), deliberately ahead of
    the producer so SEP6's multi-edge feature needs no contracts bump.
    Today's single edge is written as a one-element list.
    """
    if not isinstance(activation, PlanarEdgeStimulus):
        raise UnsupportedSpecError(
            f"synthetic_bank 2.0 has no activation variant wired for {activation.type!r}. "
            "point / s1s2 exist in the schema but are built by SEP7."
        )
    return PlanarEdgeActivation(
        type="planar_edge",
        edges=[ContractsEdge(activation.edge)],
        voltage=activation.voltage,
        time_model_units=activation.time_model_units,
        strip_thickness=activation.strip_thickness,
    )


def electrodes_model(electrodes: ElectrodePlacement) -> Any:
    """Map an electrode placement, including the realized per-pair list.

    ``pairs`` is the detail 1.1 repeated on every trace (electrode row,
    height, midpoint). Storing it once per simulation and having traces
    index it by ``pair_index`` is the normalization the restructure is
    for.
    """
    if not isinstance(electrodes, CenteredGrid2D):
        raise UnsupportedSpecError(
            f"synthetic_bank 2.0 has no electrode variant for {electrodes.type!r}. "
            "Add the variant to egm-contracts' simulation_config schema first."
        )

    midpoints = electrodes.bipolar_pair_midpoints_mm()
    pairs = [
        BipolarPair(
            pair_index=pair_index,
            electrode_indices=[ElectrodeIndice(int(a)), ElectrodeIndice(int(b))],
            electrode_row=int(electrodes.row_col(a)[0]),
            # Phase-1 grids sit at one sampled height, so the per-pair
            # height equals the placement's. It is stored per pair anyway
            # because a future non-planar placement varies it per pair.
            height_mm=electrodes.height_mm,
            midpoint_mm=[float(x) for x in midpoints[pair_index]],
        )
        for pair_index, (a, b) in enumerate(electrodes.bipolar_pairs)
    ]

    from myocard_egm_contracts._generated.python.synthetic_bank import (
        CenteredGrid2DElectrodes,
    )

    return CenteredGrid2DElectrodes(
        type="centered_grid_2d",
        n_rows=electrodes.n_rows,
        n_cols=electrodes.n_cols,
        spacing_mm=electrodes.spacing_mm,
        height_mm=electrodes.height_mm,
        positions_mm=[PositionMm([float(x) for x in row]) for row in electrodes.positions_mm],
        pairs=pairs,
    )


def backend_model(
    backend_metadata: dict[str, Any],
    *,
    output_fs_hz: float,
    capture_oversample: int,
) -> FinitewaveBackend:
    """Map backend provenance to the contracts backend object.

    The typed fields are filled from the backend's own metadata keys
    where they exist — the point of 2.0 is that known facts live in
    named fields rather than a generic bag, so anything the schema has a
    field for should not end up in ``params``.

    ``ap_time_unit_ms`` is deliberately excluded: it is the
    Aliev-Panfilov model-time calibration and therefore belongs to the
    *cell model*, not the backend (see :func:`cell_model_model`).
    """
    version = backend_metadata.get("backend_version") or backend_metadata.get(
        "finitewave_version_pin"
    )
    dt_model_units = backend_metadata.get("dt_model_units") or backend_metadata.get(
        "ap_dt_model_units"
    )
    consumed = {
        "backend_name",
        "backend_version",
        "finitewave_version_pin",
        "dt_model_units",
        "ap_dt_model_units",
        "model_class",
        "ap_time_unit_ms",
    }
    params = {k: v for k, v in backend_metadata.items() if k not in consumed}
    return FinitewaveBackend(
        type="finitewave",
        version=_optional_str(version),
        output_fs_hz=float(output_fs_hz),
        capture_oversample=int(capture_oversample),
        dt_model_units=None if dt_model_units is None else float(dt_model_units),
        params=params or None,
    )


def cell_model_model(backend_metadata: dict[str, Any]) -> CellModelModel:
    """Map the backend's reported model class to a cell-model object.

    Phase 1.5 has no ``CellModelSpec`` yet (that is SEP5 / design note
    D2), so the model identity is still recovered from what the backend
    reported rather than from a spec the caller passed in. When SEP5
    lands, this takes the spec directly and the string sniffing goes.
    """
    model_class = str(backend_metadata.get("model_class", "")).strip()
    ap_time_unit_ms = backend_metadata.get("ap_time_unit_ms")

    if model_class in {"Courtemanche", "Courtemanche2D"}:
        return CourtemancheCellModel(type="courtemanche", params=None)

    if ap_time_unit_ms is None:
        raise UnsupportedSpecError(
            f"Cannot build a cell_model for model_class={model_class!r}: "
            "aliev_panfilov requires ap_time_unit_ms in the backend metadata."
        )
    return AlievPanfilovCellModel(
        type="aliev_panfilov",
        ap_time_unit_ms=float(ap_time_unit_ms),
        params=None,
    )


def label_policy_model(policy: LabelPolicy) -> LabelPolicyModel:
    """Map a label policy, widening its scalar threshold to ``thresholds[]``.

    The producer's dataclasses keep a scalar ``threshold`` — it is the
    ergonomic config surface for a binary task. The contract is an
    ascending **list** (N thresholds = N+1 classes, CL-088), so that
    Phase-2 multiclass severity is a longer list rather than a new
    schema variant. Class *names* are deliberately not written here:
    they live once in the per-simulation ``label_names`` map, and
    duplicating them into the policy would let the two disagree.
    """
    if isinstance(policy, ProducerLocalDensityLabel):
        return LocalDensityLabel(
            type="local_density",
            radius_mm=policy.radius_mm,
            thresholds=[Threshold(policy.threshold)],
        )
    if isinstance(policy, ProducerGlobalDensityLabel):
        return GlobalDensityLabel(type="global_density", thresholds=[Threshold(policy.threshold)])
    raise UnsupportedSpecError(
        f"synthetic_bank 2.0 has no label-policy variant for {policy.type!r}. "
        "Add the variant to egm-contracts' simulation_config schema first."
    )


# ---------------------------------------------------------------------------
# Whole-group assembly
# ---------------------------------------------------------------------------


def build_simulation_columns(
    *,
    results: list[SimulationResult],
    label_policy: LabelPolicy,
    labels_dict: dict[int, str],
    master_seed: int,
    output_fs_hz: float,
    capture_oversample: int,
) -> Simulations:
    """Build the columnar ``simulations/`` group from the run's results.

    One row per simulation, in ``simulation_id`` order. Every column is
    built from :attr:`SimulationResult.specs` — the realized spec
    objects — rather than from the dataset-level config, so a bank
    records what each simulation *was*, not what the run was configured
    to sample from.

    ``seed`` is the exception: it is the **run's master seed**, written
    identically into every row (CL-096). It is bank-scoped like
    ``generation_params``, and 2.0 put it in the per-simulation group by
    oversight — **FB-16** moves it to a root attr in the Phase-2 bump,
    at which point the repetition goes away. Writing the per-simulation
    derived seed here instead would silently change the column's meaning
    with no version bump to signal it, which is the worse failure.
    """
    policy_model = label_policy_model(label_policy)
    # The {int: name} map is bank-wide today (every simulation shares one
    # label vocabulary); egm-data raises if simulations disagree.
    name_map = {str(k): v for k, v in labels_dict.items()}

    simulation_ids: list[int] = []
    seeds: list[int] = []
    geometries: list[Geometry] = []
    cell_models: list[CellModelModel] = []
    substrates: list[Substrate] = []
    summaries: list[SubstrateSummary] = []
    activations: list[ActivationModel] = []
    electrodes_col: list[Electrodes] = []
    backends: list[Backend] = []
    policies: list[LabelPolicyModel] = []
    name_maps: list[dict[str, str]] = []

    for result in results:
        run_meta = result.run_metadata
        backend_meta = dict(run_meta.get("backend_metadata", {}))
        specs = result.specs

        simulation_ids.append(int(run_meta.get("simulation_id", 0)))
        seeds.append(int(master_seed))
        # Single-variant unions need the RootModel wrapper; multi-variant
        # ones are used directly. See the module docstring (FB-15).
        geometries.append(Geometry(geometry_model(specs.geometry)))
        cell_models.append(cell_model_model(backend_meta))
        substrates.append(Substrate(substrate_model(specs.substrate)))
        summaries.append(substrate_summary_model(result.substrate_realization_metadata))
        activations.append(activation_model(specs.activation))
        electrodes_col.append(Electrodes(electrodes_model(specs.electrodes)))
        backends.append(
            Backend(
                backend_model(
                    backend_meta,
                    output_fs_hz=output_fs_hz,
                    capture_oversample=capture_oversample,
                )
            )
        )
        policies.append(policy_model)
        name_maps.append(dict(name_map))

    return Simulations(
        simulation_id=simulation_ids,
        seed=seeds,
        geometry=geometries,
        cell_model=cell_models,
        substrate=substrates,
        substrate_summary=summaries,
        activation=activations,
        electrodes=electrodes_col,
        backend=backends,
        label_policy=policies,
        label_names=name_maps,
    )


def _optional_str(value: Any) -> str | None:
    """``str(value)`` unless it is absent — never the string ``"None"``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ActivationModel",
    "CellModelModel",
    "LabelPolicyModel",
    "UnsupportedSpecError",
    "activation_model",
    "backend_model",
    "build_simulation_columns",
    "cell_model_model",
    "electrodes_model",
    "geometry_model",
    "label_policy_model",
    "substrate_model",
    "substrate_summary_model",
]
