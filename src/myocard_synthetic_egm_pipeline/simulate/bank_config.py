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
names deliberately follow ``specs.py``, so most of this file is
one-to-one; the places where it *isn't* are the interesting ones and are
commented individually.

Everything here is pure — no I/O, no HDF5, no backend imports
(Guardrails 1 and the no-HDF5-in-this-repo rule). Each function takes a
spec object and returns a Pydantic model.

**Codegen shape note.** Single-variant unions in the schema codegen to a
``RootModel`` wrapper (``Geometry``, ``Substrate``, ``Electrodes``,
``Backend``) while multi-variant unions codegen to a proper discriminated
union used directly (``cell_model``, ``activation``, ``label_policy``).
That asymmetry is a codegen artifact on the contracts side, not something
this module can fix, so the wrappers are applied explicitly at the call site in
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

from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel as ProducerAlievPanfilovCellModel,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    CellModelSpec,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    CourtemancheCellModel as ProducerCourtemancheCellModel,
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

# Union of every activation variant the 2.0 schema accepts. This producer
# builds only the planar-edge one; `point` / `s1s2` are schema-only so far.
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
    ``planar_edge.edges`` is a **list**, deliberately ahead of the
    producer: a multi-edge stimulus is a foreseen feature, and having the
    schema already plural means adding it needs no contracts bump and no
    migration of banks written before it. Today's single edge is written
    as a one-element list.
    """
    if not isinstance(activation, PlanarEdgeStimulus):
        raise UnsupportedSpecError(
            f"synthetic_bank 2.0 has no activation variant wired for {activation.type!r}. "
            "point / s1s2 exist in the schema but no producer builds them yet."
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
    *cell model*, not the backend (see :func:`cell_model_model`). No
    backend in this repo emits it any more — the cell-model spec is the
    source — but the exclusion stays: the rule is "this fact lives on the
    cell model", and a backend that reports it anyway must not get a
    second, independently-editable copy of it into ``params``.

    The step is looked for under each model's own key. ``crn_dt_ms`` is the
    same fact as ``ap_dt_model_units`` — Courtemanche's model time unit *is*
    the millisecond — so it fills the same typed field rather than falling
    through into ``params``, where a Courtemanche bank would state its timestep
    in a free-form bag while claiming ``dt_model_units: null``.
    """
    version = backend_metadata.get("backend_version") or backend_metadata.get(
        "finitewave_version_pin"
    )
    dt_model_units = (
        backend_metadata.get("dt_model_units")
        or backend_metadata.get("ap_dt_model_units")
        or backend_metadata.get("crn_dt_ms")
    )
    consumed = {
        "backend_name",
        "backend_version",
        "finitewave_version_pin",
        "dt_model_units",
        "ap_dt_model_units",
        "crn_dt_ms",
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


def cell_model_model(cell_model: CellModelSpec) -> CellModelModel:
    """Map the cell-model spec the simulation ran with to its contracts object.

    Takes the spec, like every other mapper here. It used to take
    ``backend_metadata`` and recover the identity by **string-sniffing**
    ``model_class`` — matching ``"AlievPanfilov2D"`` against the class
    name finitewave happens to use — then read ``ap_time_unit_ms`` back
    out of the same dict. Both facts were the runner's already; the round
    trip through the backend's provenance bag existed only because
    :class:`~...result.SimulationSpecs` had no cell model to hand over.

    ``ap_time_unit_ms`` is still written, because the contract's
    ``AlievPanfilovCellModel`` still requires it. It now comes from
    ``spec.time_unit_ms``, which is the number the backend metadata was
    carrying a copy of, so the bank is unchanged.

    The Courtemanche variant carries **no** time-unit field, in the schema
    as here: it runs in milliseconds, so there is nothing to calibrate and
    nothing to record. Its ``params`` are the conductance scalings, which
    are chosen rather than derived and are what a parameter sweep's
    theta-spec points into by path — hence a mapping rather than named
    fields.
    """
    if isinstance(cell_model, ProducerAlievPanfilovCellModel):
        return AlievPanfilovCellModel(
            type="aliev_panfilov",
            ap_time_unit_ms=float(cell_model.time_unit_ms),
            params=None,
        )
    if isinstance(cell_model, ProducerCourtemancheCellModel):
        return CourtemancheCellModel(
            type="courtemanche",
            # `or None` because the schema's field is optional and an empty
            # mapping is the control parameterisation, not a set of scalings
            # that happens to be empty.
            params={str(k): float(v) for k, v in cell_model.params.items()} or None,
        )
    raise UnsupportedSpecError(
        f"synthetic_bank 2.0 has no cell-model variant wired for {cell_model.type!r}."
    )


def label_policy_model(policy: LabelPolicy) -> LabelPolicyModel:
    """Map a label policy, widening its scalar threshold to ``thresholds[]``.

    The producer's dataclasses keep a scalar ``threshold`` — it is the
    ergonomic config surface for a binary task. The contract is an
    ascending **list** (N thresholds = N+1 classes), so that a later
    multiclass severity label is a longer list rather than a new schema
    variant. Class *names* are deliberately not written here:
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
    identically into every row. It is bank-scoped like
    ``generation_params``, and 2.0 put it in the per-simulation group by
    oversight; a later schema bump moves it to a root attribute, at which
    point the repetition goes away. Writing the per-simulation
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
        # ones are used directly. See the module docstring.
        geometries.append(Geometry(geometry_model(specs.geometry)))
        cell_models.append(cell_model_model(specs.cell_model))
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
