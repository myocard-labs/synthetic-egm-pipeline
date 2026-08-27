"""Measure conduction velocity and APD from a Finitewave run.

The other half of :mod:`~myocard_synthetic_egm_pipeline.simulate.calibration`.
That module solves model knobs from physiological targets analytically; this one
runs the solver and reports what the tissue actually did, which is what turns
the calibration into a **round-trip test**: solve, simulate, measure, assert the
measurement returns the targets.

That test is the whole point. The 2026-06-10 calibration hit its conduction
velocity target and destroyed action potential duration doing it, and nothing
noticed for two months because the only artifact was four constants and a
comment. A round-trip has no such blind spot — it asserts on both observables
at once, in physical units, against the numbers a reader can check.

**Why this lives under ``backends/``.** Measuring CV or APD means integrating the
model, so this imports Finitewave. Guardrail 1 (``project/architecture.md``)
makes ``backends/`` the only place allowed to. ``calibration.py`` deliberately
imports nothing from here, so the estimator can solve without dragging a solver
in.

**Measured off V_m, not off the electrograms.** Conduction velocity is a property
of the propagating field; routing it through the pseudo-EGM would fold in
electrode height, the ``1/r`` weighting and bipolar geometry, none of which have
anything to do with how fast the tissue conducts. Measuring the field directly
keeps a calibration failure distinguishable from an electrode-model failure —
which matters, because this project has already spent two days on a conduction
question that turned out to be an electrode transpose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import finitewave as fw
import numpy as np
import numpy.typing as npt

from myocard_synthetic_egm_pipeline.simulate.calibration import MeasuredValues
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    CRN_MAX_DT_MS,
    CRN_PACING_BCL_MS,
    CRN_PACING_BEATS,
    CRN_REFERENCE_CV_CM_S,
    CRN_REFERENCE_DIFFUSION,
    CRN_STIMULUS_DURATION_MS,
    DT_SAFETY_FACTOR,
    MODEL_UNIT_APD90,
    MODEL_UNIT_CV,
    WILHELMS_2012_CRN_CONTROL,
    AlievPanfilovCellModel,
    CellModelSpec,
    CourtemancheCellModel,
)
from myocard_synthetic_egm_pipeline.simulate.specs import (
    _DEFAULT_STRIP_THICKNESS,
    Edge,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
)

#: Fraction of the mesh, centred, used for the velocity fit. Excludes the
#: stimulus edge (where the wave is still forming) and the far boundary (where
#: no-flux reflection distorts arrival), keeping the fit on steady propagation.
_FIT_SPAN: tuple[float, float] = (0.3, 0.7)

#: Threshold on ``u`` for "this node has activated". The AP model runs on
#: [0, 1]; 0.5 sits on the steep part of the upstroke, so the crossing time is
#: insensitive to the exact value.
_ACTIVATION_THRESHOLD: float = 0.5

#: The same, for a model whose potential is in millivolts. -20 mV sits on the
#: steep part of a Courtemanche upstroke, well above the -81 mV resting
#: potential and below the ~0 mV peak a loaded tissue node reaches.
#:
#: **Not interchangeable with the value above**, which is why they are separate
#: constants rather than one default: 0.5 in millivolts is a level a Courtemanche
#: node crosses on the way *up* and again while repolarising, so reusing it would
#: measure something, just not activation.
_ACTIVATION_THRESHOLD_MV: float = -20.0


def _activation_threshold(solved: CellModelSpec) -> float:
    """Which "this node has activated" level applies to this model's units."""
    if isinstance(solved, CourtemancheCellModel):
        return _ACTIVATION_THRESHOLD_MV
    return _ACTIVATION_THRESHOLD


def _ms_per_model_time(solved: CellModelSpec) -> float:
    """Milliseconds per model time unit — the inverse of ``ms_to_model_time``.

    Finitewave reports activation times and step counts in the model's own time
    units, so turning either into a physical duration needs this factor. It is
    1 for Courtemanche, by the same identity that makes ``ms_to_model_time`` the
    identity function.
    """
    if isinstance(solved, AlievPanfilovCellModel):
        return float(solved.time_unit_ms)
    if isinstance(solved, CourtemancheCellModel):
        return 1.0
    raise ValueError(f"no model-time conversion is registered for cell model {solved.type!r}.")


def _run_plane_wave(
    *,
    geometry: Patch2DGeometry,
    solved: CellModelSpec,
    dr_model_units: float,
    edge: Edge,
    t_max_model_units: float,
    shape: tuple[int, int] | None = None,
    strip_thickness: int = _DEFAULT_STRIP_THICKNESS,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One clean-tissue plane wave. Returns (activation times, centre-node AP).

    **Clean tissue is required, not merely convenient.** These measurements
    define what "the conduction velocity of this parameterisation" means, and
    that has to be a property of the tissue model rather than of one fibrosis
    draw. Fibrotic tissue also makes the anisotropy participate in *every*
    direction — the wave diffracts around holes — so an along-fibre measurement
    there is not measuring the along-fibre tensor entry.

    **Built through the backend's own dispatch**, not by instantiating a solver
    class here. A measurement taken through a second, similar-looking
    configuration path is evidence about that path; the CV constants this module
    produces are only meaningful if the tissue that produced them is the tissue
    a generation run integrates.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as _backend

    model = _backend._build_model_2d(
        cell_model=solved,
        geometry=geometry,
        dr_model_units=dr_model_units,
    ).model

    tissue = _backend._build_tissue_2d(
        shape=geometry.shape if shape is None else shape,
        fiber_angle_rad=geometry.fiber_angle_rad,
    )
    _backend._apply_substrate_2d(
        tissue=tissue,
        strategy=UniformRandomFibrosis(density=0.0),
        rng=np.random.default_rng(0),
    )
    model.cardiac_tissue = tissue
    _backend._install_activation_2d(
        model=model,
        source=PlanarEdgeStimulus(edge=edge, strip_thickness=strip_thickness),
        tissue=tissue,
    )
    model.t_max = t_max_model_units

    activation_tracker = fw.ActivationTime2DTracker()
    activation_tracker.threshold = _activation_threshold(solved)
    # Centre of the mesh in BOTH axes. It read ``[n_i // 2, n_i // 2]`` while
    # every mesh was square, which is the same point — and lands on the
    # stimulus edge the moment one is not, which the 1-D cable below is.
    n_i, n_j = tissue.mesh.shape
    potential_tracker = fw.ActionPotential2DTracker()
    potential_tracker.cell_ind = [[n_i // 2, n_j // 2]]

    sequence = fw.TrackerSequence()
    sequence.add_tracker(activation_tracker)
    sequence.add_tracker(potential_tracker)
    model.tracker_sequence = sequence
    model.run()

    return (
        np.asarray(activation_tracker.act_t, dtype=np.float64),
        np.asarray(potential_tracker.output, dtype=np.float64).ravel(),
    )


def _velocity_from_activation(
    activation: npt.NDArray[np.float64],
    *,
    edge: Edge,
    dr_mm: float,
    time_unit_ms: float,
) -> float:
    """Least-squares slope of activation time against index, in cm/s.

    Averages across the wavefront, which is right for a sheet and **wrong for
    a cable**: a mesh three cells wide has two non-conducting boundary rows
    reading zero, so the mean is a third of the arrival time, the slope is a
    third of the truth and the velocity comes out three times too fast. A cable
    supplies its own profile to :func:`_velocity_from_profile` instead.
    """
    axis = 1 if edge in ("left", "right") else 0
    return _velocity_from_profile(
        activation.mean(axis=1 - axis), edge=edge, dr_mm=dr_mm, time_unit_ms=time_unit_ms
    )


def _velocity_from_profile(
    profile: npt.NDArray[np.float64],
    *,
    edge: Edge,
    dr_mm: float,
    time_unit_ms: float,
) -> float:
    """Least-squares slope of activation time against index, in cm/s."""
    lo = int(_FIT_SPAN[0] * len(profile))
    hi = int(_FIT_SPAN[1] * len(profile))
    index = np.arange(lo, hi)
    reached = profile[lo:hi] > 0

    if int(reached.sum()) < 5:
        raise ValueError(
            f"the wave did not cross the fit window for edge={edge!r}: only "
            f"{int(reached.sum())} of {hi - lo} nodes activated. Either t_max is "
            "too short for this conduction velocity, or propagation failed."
        )

    slope = float(np.polyfit(index[reached], profile[lo:hi][reached], 1)[0])
    if slope <= 0:
        raise ValueError(
            f"activation time does not increase along the propagation axis "
            f"(slope {slope}); the wave is not travelling the way edge={edge!r} implies."
        )
    # mm per cell / (model t.u. per cell * ms per t.u.) = mm/ms; x100 -> cm/s.
    return 100.0 * dr_mm / (slope * time_unit_ms)


def _apd90_from_potential(
    potential: npt.NDArray[np.float64],
    *,
    dt_model_units: float,
    time_unit_ms: float,
    activated_above: float = _ACTIVATION_THRESHOLD,
) -> float:
    """APD90 in ms: 10 % upstroke to 90 % repolarisation.

    Definitions differ between papers by tens of milliseconds, so the one used
    here is stated rather than assumed — and it is the same one
    ``MODEL_UNIT_APD90`` was measured with, which is what makes the round-trip
    a closed loop rather than two unrelated numbers.

    Unit-free apart from ``activated_above``: the 10 % and 90 % levels are taken
    relative to *this trace's* rest and peak, so a potential in millivolts and
    one on [0, 1] are handled by the same arithmetic. Only the "did it activate
    at all" guard needs to know the scale.
    """
    if potential.size == 0 or float(potential.max()) < activated_above:
        raise ValueError(
            "the centre node never activated, so APD is undefined. The stimulus "
            "may not have propagated that far within t_max."
        )

    rest = float(potential[0])
    amplitude = float(potential.max()) - rest
    threshold = rest + 0.1 * amplitude

    peak = int(np.argmax(potential))
    upstroke = int(np.argmax(potential > threshold))
    after_peak = np.flatnonzero(potential[peak:] < threshold)
    if after_peak.size == 0:
        raise ValueError(
            "the action potential had not repolarised to 90 % by the end of the "
            "run, so APD90 is a lower bound rather than a measurement. Extend "
            "t_max — it must cover activation plus a full APD."
        )

    duration_steps = (peak + int(after_peak[0])) - upstroke
    return duration_steps * dt_model_units * time_unit_ms


def _crossing_and_repolarisation_budget(
    *,
    geometry: Patch2DGeometry,
    solved: CellModelSpec,
    dr_model_units: float,
) -> float:
    """Model time for the wave to cross the patch twice over, plus repolarisation.

    Generous on purpose: the failure mode of too-short is a confusing exception
    rather than a wrong number, and this runs a handful of times rather than per
    simulation. Per-model because the two halves are read off different
    constants — Aliev-Panfilov's are in model units, Courtemanche's in ms.
    """
    cells = geometry.size_mm / geometry.dr_mm
    span_space_units = cells * dr_model_units

    if isinstance(solved, AlievPanfilovCellModel):
        crossing = span_space_units / (MODEL_UNIT_CV * float(np.sqrt(solved.diffusion)))
        return 2.0 * crossing + 3.0 * MODEL_UNIT_APD90
    if isinstance(solved, CourtemancheCellModel):
        # CV ~ sqrt(D) from the measured reference point, in cm/s -> mm/ms.
        cv_mm_per_ms = (
            CRN_REFERENCE_CV_CM_S
            * float(np.sqrt(solved.diffusion / CRN_REFERENCE_DIFFUSION))
            / 100.0
        )
        crossing = geometry.size_mm / cv_mm_per_ms
        # Two published APD90s rather than three, because Courtemanche's is
        # ~300 ms against Aliev-Panfilov's 220 and every extra millisecond is a
        # full pass over the mesh with a 21-state membrane behind it.
        return 2.0 * crossing + 2.0 * WILHELMS_2012_CRN_CONTROL["apd90_ms"]
    raise ValueError(f"no simulation-time budget is registered for {solved.type!r}.")


def measure(
    *,
    geometry: Patch2DGeometry,
    solved: CellModelSpec,
    dr_model_units: float,
    t_max_model_units: float | None = None,
) -> MeasuredValues:
    """Simulate at ``solved`` and report the physical CV and APD it produces.

    Conduction velocity is measured **along the fibres** (``left`` edge with the
    default ``fiber_angle_rad = 0``), matching what ``calibrate`` targets and
    what the literature quotes.

    **The APD reported here is a first beat into rested tissue**, which is what
    a generation run produces and therefore the number that decides whether
    repolarisation clears the trace window. It is *not* the paced steady-state
    APD that a published single-cell table reports — for Courtemanche the two
    differ by tens of milliseconds and neither is wrong, they are answers to
    different questions — see :func:`measure_single_cell`, which reads the
    paced one at a pinned protocol.
    """
    if t_max_model_units is None:
        t_max_model_units = _crossing_and_repolarisation_budget(
            geometry=geometry, solved=solved, dr_model_units=dr_model_units
        )

    activation, potential = _run_plane_wave(
        geometry=geometry,
        solved=solved,
        dr_model_units=dr_model_units,
        edge="left",
        t_max_model_units=t_max_model_units,
    )
    ms_per_model_time = _ms_per_model_time(solved)

    # The centre node sits half a patch from the stimulus edge — 20 mm at the
    # production geometry — so its upstroke is driven by the arriving wavefront
    # and carries nothing of the stimulus. Only reported for a model whose
    # potential is in millivolts; a rate of change of a dimensionless ``u`` is
    # not a volts-per-second, and writing one would invite the comparison.
    upstroke_v_s = None
    if isinstance(solved, CourtemancheCellModel):
        dt_ms = solved.dt_model_units * ms_per_model_time
        upstroke_v_s = float(np.diff(potential).max() / dt_ms)

    return MeasuredValues(
        conduction_velocity_cm_s=_velocity_from_activation(
            activation,
            edge="left",
            dr_mm=geometry.dr_mm,
            time_unit_ms=ms_per_model_time,
        ),
        apd90_ms=_apd90_from_potential(
            potential,
            dt_model_units=solved.dt_model_units,
            time_unit_ms=ms_per_model_time,
            activated_above=_activation_threshold(solved),
        ),
        upstroke_v_s=upstroke_v_s,
    )


def measure_conduction_velocity(
    *,
    geometry: Patch2DGeometry,
    solved: CellModelSpec,
    dr_model_units: float,
    edge: Edge = "left",
) -> float:
    """Along-fibre conduction velocity alone, in cm/s.

    Separate from :func:`measure` because **it is an order of magnitude cheaper
    for an ionic model**: the wave crosses the patch in ~70 ms and repolarises
    over ~300 more, so a run that also has to report APD integrates the whole
    mesh five times longer for a number the caller may not need. At
    Courtemanche's per-node cost that is the difference between a slow test and
    an overnight one.
    """
    cells = geometry.size_mm / geometry.dr_mm
    span_space_units = cells * dr_model_units
    if isinstance(solved, AlievPanfilovCellModel):
        crossing = span_space_units / (MODEL_UNIT_CV * float(np.sqrt(solved.diffusion)))
    elif isinstance(solved, CourtemancheCellModel):
        cv_mm_per_ms = (
            CRN_REFERENCE_CV_CM_S
            * float(np.sqrt(solved.diffusion / CRN_REFERENCE_DIFFUSION))
            / 100.0
        )
        crossing = geometry.size_mm / cv_mm_per_ms
    else:
        raise ValueError(f"no simulation-time budget is registered for {solved.type!r}.")

    # Transverse propagation is `anisotropy_ratio` times slower, so a `top` or
    # `bottom` run needs its budget scaled or it silently never crosses.
    scale = 1.0 if edge in ("left", "right") else max(1.0, geometry.anisotropy_ratio)

    activation, _ = _run_plane_wave(
        geometry=geometry,
        solved=solved,
        dr_model_units=dr_model_units,
        edge=edge,
        t_max_model_units=2.0 * crossing * scale,
    )
    return _velocity_from_activation(
        activation,
        edge=edge,
        dr_mm=geometry.dr_mm,
        time_unit_ms=_ms_per_model_time(solved),
    )


def measure_anisotropy_ratio(
    *,
    geometry: Patch2DGeometry,
    solved: CellModelSpec,
    dr_model_units: float,
) -> float:
    """Realized along-over-transverse velocity ratio.

    Separate from :func:`measure` because it costs a second solver run and is
    only interesting when the anisotropy itself is under test — which, since
    that knob once spent the life of the project silently doing nothing, is
    worth being able to ask directly.

    **Measured from the activation map, both times.** Reading ``D_al`` and
    ``D_ac`` back off the stencil would restate the assignment rather than test
    it, and restating the assignment is precisely what the code did while the
    knob was inoperative: it plainly appeared to set the tensor, and it was
    setting two attributes on an object nothing read.

    **Any cell model, and that is the point of the signature.** The tensor
    lives on the stencil, which is a separate object from the membrane, so a
    single helper writes a pure shape that every model inherits — there is one
    configuration path and no per-model sibling to drift from it. That is a
    claim about the code, though, of the same species as "the helper sets
    ``D_al``", and this function is what turns it into a measurement.

    Two calls to :func:`measure_conduction_velocity` rather than a private loop
    with its own time budget: that function already carries the per-model
    crossing estimate *and* the transverse scaling, and a second copy here
    would be one more place for the ionic model to be forgotten.
    """
    along = measure_conduction_velocity(
        geometry=geometry, solved=solved, dr_model_units=dr_model_units, edge="left"
    )
    across = measure_conduction_velocity(
        geometry=geometry, solved=solved, dr_model_units=dr_model_units, edge="top"
    )
    return along / across


# ---------------------------------------------------------------------------
# The 1-D cable — the rig a mesh-convergence sweep runs on
# ---------------------------------------------------------------------------

STRIP_LENGTH_MM: float = 20.0
"""Length of the convergence rig's cable, in mm.

Long enough that the central 30-70 % the velocity fit uses is steady
propagation well clear of both ends, and short enough that refining the mesh
stays affordable: cost runs as ``1/dr**3`` once the diffusion stability bound
binds, so halving the pitch is eight times the work.
"""

STRIP_STIMULUS_MM: float = 0.75
"""Physical thickness of the cable's stimulus strip, in mm.

**Held in millimetres, not in cells, and that is the whole point.**
``PlanarEdgeStimulus.strip_thickness`` counts *cells*, so a sweep that left it
alone would shrink the stimulus every time it refined the mesh — 0.75 mm at
``dr = 0.25`` down to 0.225 mm at 0.075 — and would be varying two things at
once.

It bites hardest at **high** diffusion, which is the counter-intuitive part: a
larger ``D`` drains the stimulated region into its neighbours faster, so the
smallest physical stimulus fails to launch a wave first at the top of a
diffusion sweep, not the bottom. At ``dr = 0.075`` and ``D = 0.616`` it stopped
launching altogether and the sweep died with "0 of 106 nodes activated", which
is at least loud. Had it merely launched *weakly*, the sweep would have
returned numbers and blamed the mesh for the stimulus.

0.75 mm is what three cells came to at the coarsest pitch, so the coarse end of
the curve means what it did before.
"""

STRIP_WIDTH_CELLS: int = 3
"""Mesh rows in the cable. Three, giving **one** interior row.

Finitewave's tissue puts a non-conducting boundary on every edge, so a 3-row
mesh is a genuine one-dimensional cable: the interior row has no transverse
neighbour to exchange with, and the transverse diffusion entry never
participates. That is the point — a convergence sweep should vary the pitch
along the direction of propagation and nothing else.
"""


@dataclass(frozen=True)
class StripMeasurement:
    """What one cable run reports."""

    dr_mm: float
    dt_ms: float
    n_cells: int
    conduction_velocity_cm_s: float
    upstroke_v_s: float
    """**Propagated** dV/dt max, at the cable's centre. No stimulus reaches it."""
    apd90_ms: float | None = None
    """``None`` unless the run was given a budget long enough to repolarise."""


def strip_step_ms(*, diffusion: float, dr_mm: float, dimensions: int = 2) -> float:
    """The integration step a cable at this pitch needs, in ms.

    The same rule :func:`~...simulate.cell_models.calibrate_courtemanche`
    applies — safety factor inside the explicit-diffusion bound, capped by the
    ionic ceiling — but reachable at an arbitrary pitch. The calibration itself
    refuses any pitch but the one its reference velocity was measured at, which
    is correct for authoring a card and useless for the sweep that decides what
    that pitch should be.
    """
    cfl = (dr_mm * dr_mm) / (2.0 * dimensions * diffusion)
    return float(min(DT_SAFETY_FACTOR * cfl, CRN_MAX_DT_MS))


def measure_strip(
    *,
    diffusion: float,
    dr_mm: float,
    params: Mapping[str, float] | None = None,
    length_mm: float = STRIP_LENGTH_MM,
    include_apd: bool = False,
) -> StripMeasurement:
    """Run one Courtemanche plane wave down a 1-D cable and report what it did.

    **The rig a mesh-convergence sweep needs, and the reason it is a cable.**
    Refining the pitch on the 40 mm patch costs ``f**4`` — ``f**2`` more nodes
    and ``f**2`` more steps, because holding conduction velocity forces
    ``D`` to scale as ``f**2`` and that tightens the stability bound equally.
    A cable is one row instead of ``n``, so the same question — does the
    realised velocity stop moving as the mesh refines? — is answered at a few
    hundredths of the cost, and the patch price is paid once for the pitch that
    is chosen rather than once per candidate.

    Takes ``diffusion`` and ``dr_mm`` rather than a solved cell model because a
    sweep is exactly the thing that cannot go through
    :func:`~...simulate.cell_models.calibrate_courtemanche`: that refuses any
    pitch but the one its reference velocity was measured at, and this is the
    measurement that decides whether that pitch is right.

    Parameters
    ----------
    diffusion
        ``D_model`` in mm^2/ms. Held fixed across a sweep, so a change in the
        reported velocity is the *mesh* moving and nothing else.
    dr_mm
        Mesh pitch under test.
    params
        Conductance scalings, empty for control.
    length_mm
        Cable length. Defaults to :data:`STRIP_LENGTH_MM`.
    include_apd
        Run long enough to repolarise as well. Roughly twenty times the work,
        so it is off for a velocity sweep and on for a severity sweep.
    """
    solved = CourtemancheCellModel(
        diffusion=diffusion,
        dt_model_units=strip_step_ms(diffusion=diffusion, dr_mm=dr_mm),
        params=dict(params or {}),
    )
    geometry = Patch2DGeometry(size_mm=length_mm, dr_mm=dr_mm)
    n_cells = geometry.n_cells_per_edge
    shape = (STRIP_WIDTH_CELLS, n_cells)
    # Fixed physical stimulus, so refining the mesh varies the mesh alone.
    strip_cells = max(1, round(STRIP_STIMULUS_MM / dr_mm))

    # Crossing time at the velocity this diffusion is expected to give, doubled
    # for margin. The expectation only has to be in the right order — it sizes
    # the run, it does not enter the answer.
    expected_cm_s = CRN_REFERENCE_CV_CM_S * float(np.sqrt(diffusion / CRN_REFERENCE_DIFFUSION))
    crossing_ms = length_mm / (expected_cm_s / 100.0)
    t_max_ms = 2.0 * crossing_ms
    if include_apd:
        t_max_ms += 2.0 * WILHELMS_2012_CRN_CONTROL["apd90_ms"]

    activation, potential = _run_plane_wave(
        geometry=geometry,
        solved=solved,
        dr_model_units=dr_mm,
        edge="left",
        t_max_model_units=t_max_ms,
        shape=shape,
        strip_thickness=strip_cells,
    )

    # The single interior row, not the mean across the mesh: the two boundary
    # rows never activate and averaging them in would divide the slope by three.
    profile = activation[STRIP_WIDTH_CELLS // 2, :]
    velocity = _velocity_from_profile(profile, edge="left", dr_mm=dr_mm, time_unit_ms=1.0)

    dt_ms = float(solved.dt_model_units)
    upstroke = float(np.diff(potential).max() / dt_ms)
    apd90 = None
    if include_apd:
        apd90 = _apd90_from_potential(
            potential,
            dt_model_units=dt_ms,
            time_unit_ms=1.0,
            activated_above=_ACTIVATION_THRESHOLD_MV,
        )

    return StripMeasurement(
        dr_mm=dr_mm,
        dt_ms=dt_ms,
        n_cells=n_cells,
        conduction_velocity_cm_s=velocity,
        upstroke_v_s=upstroke,
        apd90_ms=apd90,
    )


# ---------------------------------------------------------------------------
# Single cell — the protocol a published table is read at
# ---------------------------------------------------------------------------


def _single_cell_model(solved: CourtemancheCellModel) -> Any:
    """A one-node Courtemanche cell: 3x3 mesh, one interior point.

    Shared by the paced measurement and the threshold search so the two cannot
    drift on what "one cell" means — a threshold measured on a differently
    configured cell would be a threshold for a different protocol.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as _backend

    tissue = fw.CardiacTissue2D(shape=(3, 3))
    model = fw.Courtemanche2D()
    model.dt = float(solved.dt_model_units)
    model.dr = 1.0  # No neighbours to diffuse to; the value cannot reach a result.
    model.cardiac_tissue = tissue
    model.prog_bar = False
    _backend._apply_conductance_scalings(model, solved.params)
    return model


def _stimulate(model: Any, *, beat: int, amplitude: float, bcl_ms: float, first: bool) -> None:
    """Arm one stimulus and integrate one cycle.

    One stimulus in the sequence at a time, re-armed per beat rather than 50
    queued at once: ``StimSequence.stimulate_next`` walks the whole list on
    every integration step, so a queued protocol costs ``n_beats`` python calls
    per step -- 125 million of them over a 50-beat run.

    Single-threaded deliberately: the mesh is one node, so every extra thread
    contributes a barrier and no work. Measured 1.57 s per beat on one thread
    against 2.9 s on eight.
    """
    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(
        fw.StimCurrentCoord2D(
            time=beat * bcl_ms,
            curr_value=amplitude,
            duration=CRN_STIMULUS_DURATION_MS,
            x1=1,
            x2=2,
            y1=1,
            y2=2,
        )
    )
    model.stim_sequence = stim_sequence
    model.t_max = (beat + 1) * bcl_ms
    if first:
        model.run(initialize=True, num_of_threads=1)
    else:
        # `initialize=False` keeps the state variables, which is the whole point
        # of pacing; the new stimulus has to be armed by hand because that is
        # what the skipped `initialize` would have done.
        stim_sequence.initialize(model)
        model.run(initialize=False, num_of_threads=1)


def measure_capture_threshold(
    *,
    solved: CourtemancheCellModel,
    bcl_ms: float = CRN_PACING_BCL_MS,
    resolution: float = 0.005,
) -> float:
    """Smallest stimulus amplitude that fires a rested cell, in mV/ms.

    **The number the pacing protocol is defined against.** Wilhelms states the
    single-cell stimulus as *twice the threshold amplitude*, which is only a
    protocol if the threshold is measured rather than guessed — and it is not
    guessable, because it depends on the stimulus *duration* through the
    strength-duration relation. At the pinned 2 ms this returns ~10.9 mV/ms;
    at 1, 5 and 10 ms it returns 21.3, 4.5 and 2.4.

    Measured on a **rested** cell rather than mid-train, which costs nothing in
    accuracy: at a 1 s cycle length the cell is fully recovered by the next
    stimulus, and the threshold after 49 conditioning beats measured 10.884
    against the rested 10.912 — 0.3 % apart, and independent of the amplitude
    those 49 beats were delivered at.

    Bisection, so the cost is ``log2(range / resolution)`` single-beat runs
    rather than a sweep.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    def fires(amplitude: float) -> bool:
        model = _single_cell_model(solved)
        tracker = fw.ActionPotential2DTracker()
        tracker.cell_ind = [[1, 1]]
        tracker.step = 1
        sequence = fw.TrackerSequence()
        sequence.add_tracker(tracker)
        model.tracker_sequence = sequence
        _stimulate(model, beat=0, amplitude=amplitude, bcl_ms=bcl_ms, first=True)
        peak = float(np.asarray(tracker.output, dtype=np.float64).max())
        return peak > _ACTIVATION_THRESHOLD_MV

    low, high = 0.2, 80.0
    if not fires(high):
        raise ValueError(
            f"a {high} mV/ms stimulus does not fire the cell, so no threshold "
            "exists in the searched range. Check the conductance scalings."
        )
    while high - low > resolution:
        middle = 0.5 * (low + high)
        if fires(middle):
            high = middle
        else:
            low = middle
    return high


def measure_single_cell(
    *,
    solved: CourtemancheCellModel,
    stimulus_mv_per_ms: float,
    n_beats: int = CRN_PACING_BEATS,
    bcl_ms: float = CRN_PACING_BCL_MS,
) -> dict[str, float]:
    """Pace one isolated cell and report the last beat's five AP properties.

    Returns amplitude, resting potential, APD50, APD90 and ``dV/dt max`` — the
    vector :data:`WILHELMS_2012_CRN_CONTROL` gives for control Courtemanche, so
    that agreement with a published implementation is checkable rather than
    eyeballed.

    **One cell, not tissue, and that is the point.** A published cell table is a
    membrane measurement; taking it in tissue would fold in electrotonic load
    from the neighbours and diffusion, and disagreement would then be
    uninterpretable — a membrane error and a tissue error look identical in the
    result. The mesh here is 3x3 with a single interior node, so the diffusion
    stencil has nothing to exchange with and ``diffusion`` cannot enter the
    answer at all.

    **The protocol is the fixture.** Courtemanche never reaches steady state
    (:data:`CRN_PACING_BEATS`), so a beat count is as much a part of the
    measurement as the cycle length. ``dV/dt max`` additionally depends on the
    stimulus that elicited it, which is why ``stimulus_mv_per_ms`` is
    **required** rather than defaulted: the reference protocol defines it as
    twice the measured capture threshold (:func:`measure_capture_threshold`),
    and a default here would let a caller take the measurement at whatever
    amplitude happened to be convenient — which is exactly how this number came
    to disagree with its reference by 14 % in the first place.

    Cost: ``n_beats * bcl_ms / dt`` integration steps of a 21-state membrane —
    around two minutes at the shipped defaults, which is why this is a slow test
    and an authoring tool rather than a load-time guard.
    """
    if n_beats < 1:
        raise ValueError("n_beats must be >= 1.")
    if stimulus_mv_per_ms <= 0:
        raise ValueError("stimulus_mv_per_ms must be positive.")
    dt = float(solved.dt_model_units)

    model = _single_cell_model(solved)

    tracker = fw.ActionPotential2DTracker()
    tracker.cell_ind = [[1, 1]]
    tracker.step = 1
    # Only the last beat is reported, and the 49 before it are 2.45 million
    # samples nothing reads. `start_time` is in model time, i.e. ms.
    tracker.start_time = (n_beats - 1) * bcl_ms
    sequence = fw.TrackerSequence()
    sequence.add_tracker(tracker)
    model.tracker_sequence = sequence

    for beat in range(n_beats):
        _stimulate(
            model,
            beat=beat,
            amplitude=stimulus_mv_per_ms,
            bcl_ms=bcl_ms,
            first=(beat == 0),
        )

    return _action_potential_properties(
        np.asarray(tracker.output, dtype=np.float64).ravel(), dt_ms=dt
    )


def _action_potential_properties(
    potential: npt.NDArray[np.float64], *, dt_ms: float
) -> dict[str, float]:
    """The five properties of one beat, in the units a published table uses.

    ``dV/dt max`` is a forward difference rather than a fitted slope, matching
    how it is defined in the literature; at ``dt <= 0.02 ms`` the upstroke is
    resolved by ~30 samples, so the difference and the derivative agree to well
    inside the tolerance anything here is asserted at.

    APDs are measured from the time of ``dV/dt max`` — the upstroke — to the
    return through the 50 % and 90 % repolarisation levels. That is one of
    several published conventions and differs from the 10 %-upstroke reference
    :func:`_apd90_from_potential` uses on a tissue trace by a fraction of a
    millisecond, since the two points are separated by the upstroke itself.
    """
    if potential.size == 0:
        raise ValueError("no samples were captured for this beat.")

    derivative = np.diff(potential) / dt_ms
    upstroke = int(np.argmax(derivative))
    peak_index = int(np.argmax(potential))
    rest = float(potential[0])
    peak = float(potential.max())
    amplitude = peak - rest

    if amplitude < 20.0:
        raise ValueError(
            f"the cell did not fire: peak {peak:.1f} mV against a rest of "
            f"{rest:.1f} mV. Check the stimulus amplitude against the diastolic "
            "threshold."
        )

    properties = {
        "amplitude_mv": amplitude,
        "rmp_mv": rest,
        "dvdt_max_v_s": float(derivative.max()),
    }
    for percent in (50, 90):
        level = peak - (percent / 100.0) * amplitude
        below = np.flatnonzero(potential[peak_index:] < level)
        if below.size == 0:
            raise ValueError(
                f"the action potential had not repolarised to {percent} % by the "
                "end of the beat, so APD is a lower bound rather than a "
                "measurement. A longer cycle length is needed."
            )
        properties[f"apd{percent}_ms"] = (peak_index + int(below[0]) - upstroke) * dt_ms
    return properties


__all__: list[str] = [
    "STRIP_LENGTH_MM",
    "StripMeasurement",
    "measure",
    "measure_anisotropy_ratio",
    "measure_conduction_velocity",
    "measure_single_cell",
    "measure_strip",
    "strip_step_ms",
]
