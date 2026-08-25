"""FinitewaveBackend — 2D solver + our EGM tracker forward calc.

This file is the only place in the repo that imports ``finitewave``
(Guardrail 1 in ``project/architecture.md``). It translates the five
public strategy specs into Finitewave's native API:

- :class:`~myocard_synthetic_egm_pipeline.simulate.cell_models.AlievPanfilovCellModel`
  → :class:`finitewave.AlievPanfilov2D`, and
  :class:`~myocard_synthetic_egm_pipeline.simulate.cell_models.CourtemancheCellModel`
  → :class:`finitewave.Courtemanche2D`, via :func:`_build_model_2d`
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.Patch2DGeometry`
  → :class:`finitewave.CardiacTissue2D` with anisotropic diffusion
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.UniformRandomFibrosis`
  → :class:`finitewave.DiffusePattern` applied to the tissue mesh
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.PlanarEdgeStimulus`
  → :class:`finitewave.StimVoltageCoord2D` wrapped in a
  :class:`finitewave.StimSequence`
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.CenteredGrid2D`
  → :class:`~myocard_synthetic_egm_pipeline.backends.finitewave.egm_kernel.EGMTracker`,
  our subclass of Finitewave's tracker. It streams the pseudo-EGM during
  the AP integration, as the stock tracker does, but computes it with a
  kernel we own — see that module for why vendoring was necessary.

Each translation lives in a private ``_apply_*`` helper. The public
:meth:`FinitewaveBackend.simulate` method wires them together, runs
the solver, and returns a
:class:`~myocard_synthetic_egm_pipeline.simulate.result.RawSimulationResult`.

Phase-1 limitations:

- Only ``Patch2DGeometry`` is supported. ``AtrialMesh3D`` would need
  a different Finitewave model (``CardiacTissue3D``) and is out of
  scope for v0.2.0.
- Only ``UniformRandomFibrosis`` is supported. Future substrate
  strategies wire into ``_apply_substrate_2d``.
- Only ``PlanarEdgeStimulus`` is supported. Future activation sources
  wire into ``_install_activation_2d``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import finitewave as fw
import numpy as np
import numpy.typing as npt

from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend
from myocard_synthetic_egm_pipeline.backends.finitewave.egm_kernel import EGMTracker
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel,
    CellModelSpec,
    CourtemancheCellModel,
)
from myocard_synthetic_egm_pipeline.simulate.model_cards import card_provenance
from myocard_synthetic_egm_pipeline.simulate.result import RawSimulationResult
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

# Aliev-Panfilov step defaults, retained ONLY as RunConfig fallbacks.
#
# These were the operative values until the calibration landed; they now live
# on RunConfig (``dt_model_units`` / ``dr_model_units``) so a bank records what
# produced it. Kept here because the RunConfig defaults have to come from
# somewhere and this is where the reasoning lives, but nothing reads them
# during a simulation any more.
_AP_DT_MODEL_UNITS: float = 0.01
"""Finitewave's recommended pairing with ``dr = 0.25`` for stable propagation.

Note this pairing is only stable at ``D_model = 1``. The calibrated defaults
run ``D_model`` near 8, where the bound is ~8x tighter — which is why ``dt`` is
derived from the bound rather than defaulted."""

_AP_DR_MODEL_UNITS: float = 0.25
"""AP model's internal dr (non-dimensional). NOT the same as
PatchSpec.dr_mm — they're independent: dr_mm is the physical mesh
resolution, this is the solver's discretization scale."""

_FINITEWAVE_VERSION_TAG: str = "0.9"
"""Pinned via pyproject (``finitewave>=0.9``). Stamped into
backend_metadata so the bank records which backend it came from."""


class FinitewaveBackend(SimulationBackend):
    """Concrete backend wrapping Finitewave's 2D monodomain solvers.

    Instantiate once; reuse across simulations — the backend itself
    holds no per-simulation state. Which membrane model integrates a given
    simulation is the ``cell_model`` spec's business, not the backend's
    (:func:`_build_model_2d`).
    """

    name: str = "finitewave"

    def simulate(
        self,
        *,
        geometry: GeometrySpec,
        substrate: SubstrateStrategy,
        activation: ActivationSource,
        electrodes: ElectrodePlacement,
        cell_model: CellModelSpec,
        config: RunConfig,
        rng: np.random.Generator,
    ) -> RawSimulationResult:
        # --- 1. Dispatch ----------------------------------------------------
        if geometry.type != "patch_2d":
            raise ValueError(
                f"FinitewaveBackend supports only 'patch_2d' geometry; got {geometry.type!r}."
            )
        if electrodes.type != "centered_grid_2d":
            raise ValueError(
                "FinitewaveBackend supports only 'centered_grid_2d' electrode placement; "
                f"got {electrodes.type!r}."
            )

        # Strategy types are checked at runtime via the dispatch in the
        # private helpers; isinstance narrows the typing here so the
        # rest of this method sees concrete types.
        assert isinstance(geometry, Patch2DGeometry)
        assert isinstance(electrodes, CenteredGrid2D)

        # --- 2. Build the membrane model and the tissue ---------------------
        # Every one of these came out of a module constant or a Finitewave
        # default until the calibration landed. They determine CV and APD, so
        # the no-hardcoding rule required them in config; they arrive here
        # already solved from physiological targets by simulate.calibration.
        native = _build_model_2d(
            cell_model=cell_model,
            geometry=geometry,
            dr_model_units=config.dr_model_units,
        )
        model = native.model

        tissue = _build_tissue_2d(geometry)

        # --- 3. Apply substrate (mutates tissue.mesh in place) --------------
        substrate_meta = _apply_substrate_2d(tissue=tissue, strategy=substrate, rng=rng)

        # Snapshot the realised substrate before the solver runs. The
        # mask survives into SimulationResult so label policies can
        # query it (mesh int8: 0=boundary, 1=healthy, 2=fibrotic).
        substrate_mask: npt.NDArray[np.int8] = tissue.mesh.astype(np.int8, copy=True)

        # Attach the tissue to the model. From here on, model.cardiac_tissue
        # is the source of truth for what the solver sees.
        model.cardiac_tissue = tissue

        # --- 4. Install activation ------------------------------------------
        _install_activation_2d(model=model, source=activation, tissue=tissue)

        # --- 5. Set t_max and install the EGM tracker -----------------------
        # The capture, not the trace: with cropping configured the solver must
        # run past the end of the trace so a window placed around the
        # activation has signal behind it (see simulate.sizing).
        t_max_model_units = cell_model.ms_to_model_time(config.effective_capture_duration_ms)
        model.t_max = t_max_model_units

        capture_step, fs_capture_hz = _pick_capture_step(
            dt_ms=native.dt_ms,
            output_fs_hz=config.output_fs_hz,
            oversample=config.capture_oversample,
        )

        # The tracker takes electrode positions in mesh-index units
        # (coords_grid = coord_mm / dr_mm). We subclass Finitewave's rather
        # than replacing it wholesale: streaming during the solve is why it
        # was chosen, and computing phi_e ourselves would mean holding the
        # whole V_m history (~504 MB per simulation at the production
        # geometry). Only the kernel arithmetic is ours.
        #
        # Coords go across in OUR (x, y, z) order and the kernel knows it:
        # x indexes axis-1 (j), y indexes axis-0 (i). Upstream's kernel reads
        # column 0 as i, which is the transpose — see egm_kernel for what that
        # cost us. The convention is stated in the kernel rather than fixed by
        # a silent column swap here.
        coords_grid = electrodes.positions_mm / geometry.dr_mm
        egm_tracker = EGMTracker(measure_coords=coords_grid)
        egm_tracker.step = capture_step
        tracker_seq = fw.TrackerSequence()
        tracker_seq.add_tracker(egm_tracker)
        model.tracker_sequence = tracker_seq

        # --- 6. Run ---------------------------------------------------------
        model.run()

        # --- 7. Collect unipolar traces ------------------------------------
        # tracker.output shape: (n_capture, n_electrodes).
        unipolar = np.asarray(egm_tracker.output, dtype=np.float64)

        # --- 8. Assemble result --------------------------------------------
        backend_metadata: dict[str, Any] = {
            "backend_name": self.name,
            "finitewave_version_pin": _FINITEWAVE_VERSION_TAG,
            "ap_dr_model_units": float(config.dr_model_units),
            "anisotropy_ratio": float(geometry.anisotropy_ratio),
            # dt / time unit / diffusion / eps are the cell model's to report.
            **cell_model.to_metadata(),
            "t_max_model_units": float(t_max_model_units),
            "capture_step_integration": int(capture_step),
            "fs_capture_hz": float(fs_capture_hz),
            # Which finitewave class integrated this — provenance about the
            # solver, and nothing more. The bank used to recover the
            # *cell model's identity* from this string; that identity now
            # comes from the spec, so no consumer keys off a third party's
            # class name.
            "model_class": native.model_class,
        }
        # Physiological provenance: which named parameterisation produced this,
        # and what it was aiming at. The four solved numbers above are the
        # mechanism; these are what a reader can check against a paper.
        if config.model_card is not None:
            backend_metadata.update(card_provenance(config.model_card))

        return RawSimulationResult(
            unipolar_traces=unipolar,
            fs_capture_hz=fs_capture_hz,
            substrate_mask=substrate_mask,
            substrate_mask_dr_mm=geometry.dr_mm,
            electrode_positions_mm=electrodes.positions_mm.copy(),
            bipolar_pairs=electrodes.bipolar_pairs,
            substrate_realization_metadata=substrate_meta,
            backend_metadata=backend_metadata,
        )


# ---------------------------------------------------------------------------
# Private adapters — strategy spec → Finitewave native API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NativeModel:
    """A configured Finitewave model plus the two facts the caller still needs.

    ``dt_ms`` and ``model_class`` are known only inside the per-model branch and
    are needed outside it, and returning them beats re-deriving them: ``dt_ms``
    is ``dt_model_units * time_unit_ms`` for a dimensionless model and plain
    ``dt_model_units`` for a dimensional one, and ``type(model).__name__`` is
    *not* a substitute for ``model_class`` — ``fw.Courtemanche2D`` is an alias
    for ``fw.Courtemanche``, so introspection would silently rename what the
    bank records.
    """

    model: Any
    dt_ms: float
    """Milliseconds per integration step. What the capture stride is chosen in."""
    model_class: str
    """Finitewave class name, for ``backend_metadata``. Provenance only."""


def _build_model_2d(
    *,
    cell_model: CellModelSpec,
    geometry: Patch2DGeometry,
    dr_model_units: float,
) -> _NativeModel:
    """Dispatch on the cell model; return a Finitewave model configured from it.

    **Refuses an unfamiliar membrane model BY NAME.** Duck-typing into one we
    have no measured calibration for would produce numbers rather than an error,
    which is the worse failure: every model here has constants that were
    measured against *it*, and applying one model's constants to another is
    how this project once reported a transverse conduction velocity under a
    longitudinal label.

    The anisotropy tensor is configured here too, and is deliberately
    **model-agnostic**: :func:`_configure_anisotropy_2d` writes a pure *shape*
    onto the stencil (``D_al = 1``, ``D_ac = 1/ratio^2``) and leaves absolute
    scale to ``model.D_model``, so the two models' very different diffusion
    magnitudes — 7.8 dimensionless for Aliev-Panfilov, 0.154 mm^2/ms for
    Courtemanche — need no per-model constant in it.
    """
    if isinstance(cell_model, AlievPanfilovCellModel):
        model: Any = fw.AlievPanfilov2D()
        model.eps = cell_model.eps
        dt_ms = cell_model.dt_model_units * cell_model.time_unit_ms
        model_class = "AlievPanfilov2D"
        solver = "calibrate_aliev_panfilov"
    elif isinstance(cell_model, CourtemancheCellModel):
        # Courtemanche's space unit is the millimetre (D is in mm^2/ms), so a
        # `dr_model_units` that disagrees with the mesh pitch is not a rescaling
        # — it is a different mesh from the one the geometry describes, and the
        # wave would simply travel at the wrong speed. Aliev-Panfilov is
        # dimensionless and has no such constraint, which is why the check lives
        # in the branch rather than above it.
        if not np.isclose(dr_model_units, geometry.dr_mm, rtol=1e-9):
            raise ValueError(
                f"Courtemanche runs in physical units: run.dr_model_units "
                f"({dr_model_units}) must equal geometry.dr_mm ({geometry.dr_mm}), "
                "because its diffusion coefficient is in mm^2/ms. Aliev-Panfilov "
                "is dimensionless and may differ."
            )
        model = fw.Courtemanche2D()
        _apply_conductance_scalings(model, cell_model.params)
        dt_ms = cell_model.dt_model_units
        model_class = "Courtemanche"
        solver = "calibrate_courtemanche"
    else:
        raise ValueError(
            "FinitewaveBackend integrates the Aliev-Panfilov and Courtemanche "
            f"cell models; got {cell_model.type!r}."
        )

    model.dt = cell_model.dt_model_units
    model.dr = dr_model_units
    model.D_model = cell_model.diffusion

    # The stability bound is a joint property: diffusion is the cell model's,
    # the grid step is the backend's. Checked here because this is the only
    # place both are in hand -- and because violating it does not crash, it
    # writes a well-formed bank full of a diverged field. Asked *after* the
    # dispatch so that a spec which only partly implements the Protocol reaches
    # the refusal above — an error naming the model beats an AttributeError.
    limit = cell_model.stability_limit(dr_model_units=dr_model_units)
    if cell_model.dt_model_units > limit:
        raise ValueError(
            f"dt={cell_model.dt_model_units} exceeds the stability bound "
            f"{limit:.6g} at diffusion={cell_model.diffusion}, "
            f"dr={dr_model_units}. Derive dt with "
            f"simulate.cell_models.{solver} rather than by hand."
        )

    _configure_anisotropy_2d(model, geometry)
    return _NativeModel(model=model, dt_ms=dt_ms, model_class=model_class)


#: Conductance-scaling names → the Finitewave attribute each one multiplies.
#:
#: Explicit rather than derived, so a name the solver cannot honour is refused
#: instead of creating a dead attribute — assigning an unknown name to a Python
#: object is not an error, which is exactly how ``anisotropy_ratio`` spent the
#: life of the project doing nothing.
#:
#: **`g_Kur_scale` is deliberately absent, and an AF-remodelled card will need
#: it.** Finitewave computes I_Kur's conductance inside the kernel as a
#: function of voltage
#: (``gkur = 0.005 + 0.05 / (1 + exp(-(u - 15) / 13))``) rather than reading a
#: parameter, so the cAF -49 % I_Kur scaling cannot be applied by assignment in
#: 0.9.3. It has to be a kernel change or a fork, and finding that out when a
#: remodelling sweep silently moved three currents of four would cost a day.
_CRN_CONDUCTANCE_ATTRS: dict[str, str] = {
    "g_Na_scale": "gna",
    "g_K1_scale": "gk1",
    "g_to_scale": "gto",
    "g_Kr_scale": "gkr",
    "g_Ks_scale": "gks",
    "g_CaL_scale": "gcal",
    "g_bNa_scale": "gnab",
    "g_bCa_scale": "gcab",
}


def _apply_conductance_scalings(model: Any, params: Mapping[str, float]) -> None:
    """Multiply the named conductances in place. No patching required.

    Finitewave exposes every Courtemanche conductance as a plain instance
    attribute read at kernel-run time, so a remodelling severity is an
    assignment rather than a subclass. The multiply is against the *shipped*
    default, so a scaling means what its name says — a fraction of the published
    conductance — regardless of what else was applied.
    """
    for name, scale in params.items():
        attr = _CRN_CONDUCTANCE_ATTRS.get(name)
        if attr is None:
            raise ValueError(
                f"Courtemanche conductance scaling {name!r} is not one this backend "
                f"can apply. Known: {', '.join(sorted(_CRN_CONDUCTANCE_ATTRS))}. "
                "(I_Kur has no scalar conductance in finitewave 0.9.3 — it is "
                "computed from voltage inside the kernel.)"
            )
        baseline = getattr(model, attr)
        setattr(model, attr, float(baseline) * float(scale))


def _build_tissue_2d(geometry: Patch2DGeometry) -> fw.CardiacTissue2D:
    """Build a healthy 2D tissue patch with a uniform fiber field.

    **Fibre components are stored in Finitewave's axis order, not ours.**
    ``fibers[..., 0]`` feeds ``d_xx``, which ``compute_weights`` applies to the
    ``(i-1, j)`` neighbour — so component 0 acts along mesh **axis-0**. Our
    convention is ``x = j`` (axis-1), ``y = i`` (axis-0), and
    ``fiber_angle_rad`` is documented as measured from **+x** (``specs.py``:
    *"0 = along +x, π/2 = along +y"*).

    So the components are crossed relative to the naive reading:

    - our ``x`` (axis-1) is Finitewave's *y* -> ``fibers[..., 1] = cos(theta)``
    - our ``y`` (axis-0) is Finitewave's *x* -> ``fibers[..., 0] = sin(theta)``

    Written the other way round, ``fiber_angle_rad = 0`` ran the fibres along
    our **+y** while every doc and config comment said ``+x``. That is the same
    root cause as the electrode transpose: Finitewave uses
    "x" for axis-0 throughout, and we use it for axis-1, so every physical
    ``(x, y)`` handed across the boundary has to swap.

    Because the tensor makes the along-fibre axis conduct measurably faster
    (3.09x at the shipped settings), getting this backwards did not merely
    mislabel an axis — it put the fast axis at 90 degrees to the intended one,
    and every conduction-velocity measurement taken before it was found was of
    the transverse axis under a longitudinal label.
    """
    import math

    tissue = fw.CardiacTissue2D(shape=geometry.shape)
    n_i, n_j = tissue.mesh.shape
    fibers = np.zeros((n_i, n_j, 2), dtype=np.float64)
    fibers[:, :, 0] = math.sin(geometry.fiber_angle_rad)  # axis-0 = our y
    fibers[:, :, 1] = math.cos(geometry.fiber_angle_rad)  # axis-1 = our x
    tissue.fibers = fibers
    return tissue


def _configure_anisotropy_2d(model: Any, geometry: Patch2DGeometry) -> None:
    """Shape the diffusion tensor so the realized CV ratio is the requested one.

    **The knobs live on the STENCIL, not on the model.** Finitewave
    reads ``self.D_al`` / ``self.D_ac`` inside
    ``AsymmetricStencil2D.compute_diffusion_components``. Until 2026-08-14 this
    helper assigned them to the *model*, where Python created two attributes
    that no reader ever consulted — so ``anisotropy_ratio`` did nothing at all
    for the life of the project, and the realized ratio was always the
    stencil's built-in ``D_al = 1, D_ac = 1/9`` (measured 3.093 for requested
    1.0, 3.0 and 6.0 alike). A silent no-op, because assigning an unknown
    attribute to a Python object is not an error.

    Why ``ratio ** 2``
    ------------------
    :math:`CV \\propto \\sqrt{D}`, so a *diffusion* ratio of :math:`\\rho^2`
    gives a *velocity* ratio of :math:`\\rho`:

    .. math::
        \\frac{CV_{\\parallel}}{CV_{\\perp}}
            = \\sqrt{\\frac{D_{al}}{D_{ac}}}
            = \\sqrt{\\rho^{2}} = \\rho

    Measured after the fix: requested 1.0 gives 1.000, requested 2.0 gives
    2.15, requested 3.0 gives 3.093 — the few-percent excess is discretization
    on a 0.25 mm mesh, not a modelling error.

    Which axis is held fixed, and why it is the along-fibre one
    ----------------------------------------------------------
    A ratio only fixes the *quotient*; something else has to pin the scale.
    We set ``D_al = 1`` and put the whole ratio into ``D_ac``, so **changing
    the anisotropy leaves the along-fibre velocity untouched and slows the
    transverse one**. The alternative — holding the geometric mean fixed, which
    the previous docstring described — would make ``anisotropy_ratio`` shift
    the along-fibre CV as a side effect, and the along-fibre axis is exactly
    what the literature quotes and what we calibrate against. A knob that
    silently moves the quantity you calibrated is the class of surprise this
    step exists to remove.

    Absolute scale is **not** set here. It belongs to ``model.D_model``, which
    multiplies the whole tensor. Finitewave already offers three independent
    multipliers on the same coefficient (``D_model`` x ``D_al`` x
    ``tissue.conductivity``); this helper deliberately drives only one of them,
    so the stencil stays a pure *shape* knob and the scale has a single home.
    The removed ``base_diffusion`` argument was the beginnings of a second one.

    Inert at the shipped default, by construction: ``ratio = 3`` yields exactly
    ``D_al = 1, D_ac = 1/9``, which is what the stencil already defaults to.
    Every bank generated at ``anisotropy_ratio = 3.0`` is therefore
    bit-identical across this change — which is why it ships on its own, ahead
    of the recalibration that changes everything.

    Full analysis: ``intracardiac-platform/project/investigations/
    ap_model_calibration.md`` section 3.
    """
    ratio = float(geometry.anisotropy_ratio)

    stencil = fw.AsymmetricStencil2D()
    stencil.D_al = 1.0
    stencil.D_ac = 1.0 / (ratio * ratio)
    model.stencil = stencil


def _apply_substrate_2d(
    *,
    tissue: fw.CardiacTissue2D,
    strategy: SubstrateStrategy,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Dispatch on strategy type; mutate ``tissue.mesh`` in place.

    Returns realization metadata (requested density, realised density,
    fibrotic node count, etc.) for downstream provenance + labelling.
    """
    if strategy.type == "uniform_random_fibrosis":
        assert isinstance(strategy, UniformRandomFibrosis)
        return _apply_uniform_random_fibrosis_2d(tissue=tissue, strategy=strategy, rng=rng)
    raise ValueError(f"FinitewaveBackend has no adapter for substrate type {strategy.type!r}.")


def _apply_uniform_random_fibrosis_2d(
    *,
    tissue: fw.CardiacTissue2D,
    strategy: UniformRandomFibrosis,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Realize UniformRandomFibrosis via Finitewave's DiffusePattern."""
    if strategy.density == 0.0:
        return {
            "density_requested": 0.0,
            "density_realized": 0.0,
            "n_fibrotic_nodes": 0,
            "strategy_type": strategy.type,
        }

    n_i, n_j = tissue.mesh.shape
    pattern = fw.DiffusePattern(
        density=strategy.density,
        x1=1,
        x2=n_i - 1,
        y1=1,
        y2=n_j - 1,
    )
    # Finitewave's DiffusePattern uses the global ``np.random`` state.
    # Seed it from our reproducible RNG and restore the prior state so
    # we don't bleed determinism into the caller.
    seed = int(rng.integers(0, 2**31 - 1))
    prior_state = np.random.get_state()
    try:
        np.random.seed(seed)
        tissue.add_pattern(pattern)
    finally:
        np.random.set_state(prior_state)

    n_fibrotic = int(np.count_nonzero(tissue.mesh == 2))
    n_interior = int(np.count_nonzero(tissue.mesh != 0))
    density_realized = n_fibrotic / max(n_interior, 1)
    return {
        "density_requested": float(strategy.density),
        "density_realized": float(density_realized),
        "n_fibrotic_nodes": n_fibrotic,
        "strategy_type": strategy.type,
    }


def _install_activation_2d(
    *,
    model: Any,
    source: ActivationSource,
    tissue: fw.CardiacTissue2D,
) -> None:
    """Dispatch on activation type; attach a StimSequence to the model."""
    if source.type == "planar_edge":
        assert isinstance(source, PlanarEdgeStimulus)
        stim = _build_planar_edge_stimulus_2d(source=source, mesh_shape=tissue.mesh.shape)
        stim_seq = fw.StimSequence()
        stim_seq.add_stim(stim)
        model.stim_sequence = stim_seq
        return
    raise ValueError(f"FinitewaveBackend has no adapter for activation type {source.type!r}.")


def _build_planar_edge_stimulus_2d(
    *,
    source: PlanarEdgeStimulus,
    mesh_shape: tuple[int, int],
) -> fw.StimVoltageCoord2D:
    """Build the Finitewave stimulus for the given mesh shape.

    Edge naming (top / bottom / left / right) follows Finitewave's mesh
    convention. See ``specs.PlanarEdgeStimulus`` docstring for the axis
    convention; the geometry is unchanged from the legacy code.
    """
    n_i, n_j = mesh_shape
    t = source.strip_thickness
    if source.edge == "top":
        x1, x2, y1, y2 = 1, t + 1, 1, n_j - 1
    elif source.edge == "bottom":
        x1, x2, y1, y2 = n_i - 1 - t, n_i - 1, 1, n_j - 1
    elif source.edge == "left":
        x1, x2, y1, y2 = 1, n_i - 1, 1, t + 1
    else:  # "right"
        x1, x2, y1, y2 = 1, n_i - 1, n_j - 1 - t, n_j - 1
    return fw.StimVoltageCoord2D(
        time=source.time_model_units,
        volt_value=source.voltage,
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
    )


def _pick_capture_step(
    *,
    dt_ms: float,
    output_fs_hz: float,
    oversample: int,
) -> tuple[int, float]:
    """Choose the tracker's integration-step stride and report the achieved capture rate.

    The tracker captures every ``step`` integration steps. We choose
    ``step`` so the capture rate is approximately
    ``oversample * output_fs_hz``; the exact achieved rate is reported
    back so the runner's downsample step uses it.

    Takes the step **already in milliseconds**. It used to take the step in
    model units together with Aliev-Panfilov's ``time_unit_ms`` and multiply
    them here, which made a dimensionless model's calibration constant an
    argument to a function about sampling rates — and there is nothing to pass
    for it when the model is dimensional. The conversion belongs to whoever
    knows which model this is (:func:`_build_model_2d`), not here.

    Returns
    -------
    (step, achieved_capture_fs_hz)
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    target_capture_fs_hz = oversample * output_fs_hz
    step_f = (1000.0 / target_capture_fs_hz) / dt_ms
    step = max(1, round(step_f))
    achieved_capture_fs_hz = 1000.0 / (step * dt_ms)
    return step, achieved_capture_fs_hz
