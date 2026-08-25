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

import finitewave as fw
import numpy as np
import numpy.typing as npt

from myocard_synthetic_egm_pipeline.simulate.calibration import MeasuredValues
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    CRN_PACING_BCL_MS,
    CRN_PACING_BEATS,
    CRN_REFERENCE_CV_CM_S,
    CRN_REFERENCE_DIFFUSION,
    CRN_STIMULUS_AMPLITUDE_MV_PER_MS,
    CRN_STIMULUS_DURATION_MS,
    MODEL_UNIT_APD90,
    MODEL_UNIT_CV,
    WILHELMS_2012_CRN_CONTROL,
    AlievPanfilovCellModel,
    CellModelSpec,
    CourtemancheCellModel,
)
from myocard_synthetic_egm_pipeline.simulate.specs import (
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

    tissue = _backend._build_tissue_2d(geometry)
    _backend._apply_substrate_2d(
        tissue=tissue,
        strategy=UniformRandomFibrosis(density=0.0),
        rng=np.random.default_rng(0),
    )
    model.cardiac_tissue = tissue
    _backend._install_activation_2d(
        model=model, source=PlanarEdgeStimulus(edge=edge), tissue=tissue
    )
    model.t_max = t_max_model_units

    activation_tracker = fw.ActivationTime2DTracker()
    activation_tracker.threshold = _activation_threshold(solved)
    n_i = tissue.mesh.shape[0]
    potential_tracker = fw.ActionPotential2DTracker()
    potential_tracker.cell_ind = [[n_i // 2, n_i // 2]]

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
    """Least-squares slope of activation time against index, in cm/s."""
    axis = 1 if edge in ("left", "right") else 0
    profile = activation.mean(axis=1 - axis)

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
    different questions (:func:`measure_single_cell`, CL-180 trap 1).
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
    solved: AlievPanfilovCellModel,
    dr_model_units: float,
    t_max_model_units: float | None = None,
) -> float:
    """Realized along-over-transverse velocity ratio.

    Separate from :func:`measure` because it costs a second solver run and is
    only interesting when the anisotropy itself is under test — which, since
    that knob spent the project silently doing nothing (CL-172), is worth being
    able to ask directly.
    """
    velocities: dict[Edge, float] = {}
    for edge in ("left", "top"):
        if t_max_model_units is None:
            cells = geometry.size_mm / geometry.dr_mm
            crossing = (cells * dr_model_units) / (MODEL_UNIT_CV * float(np.sqrt(solved.diffusion)))
            # Transverse propagation is `ratio` times slower, so the budget has
            # to scale with it or the `top` run silently never crosses.
            budget = 2.0 * crossing * max(1.0, geometry.anisotropy_ratio) + 3.0 * MODEL_UNIT_APD90
        else:
            budget = t_max_model_units

        activation, _ = _run_plane_wave(
            geometry=geometry,
            solved=solved,
            dr_model_units=dr_model_units,
            edge=edge,
            t_max_model_units=budget,
        )
        velocities[edge] = _velocity_from_activation(
            activation,
            edge=edge,
            dr_mm=geometry.dr_mm,
            time_unit_ms=_ms_per_model_time(solved),
        )

    return velocities["left"] / velocities["top"]


# ---------------------------------------------------------------------------
# Single cell — the protocol a published table is read at
# ---------------------------------------------------------------------------


def measure_single_cell(
    *,
    solved: CourtemancheCellModel,
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
    measurement as the cycle length, and ``dV/dt max`` additionally depends on
    the stimulus that elicited it (:data:`CRN_STIMULUS_AMPLITUDE_MV_PER_MS`).
    Both are arguments with pinned defaults rather than free choices made here.

    Cost: ``n_beats * bcl_ms / dt`` integration steps of a 21-state membrane —
    around two minutes at the shipped defaults, which is why this is a slow test
    and an authoring tool rather than a load-time guard.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as _backend

    if n_beats < 1:
        raise ValueError("n_beats must be >= 1.")
    dt = float(solved.dt_model_units)

    tissue = fw.CardiacTissue2D(shape=(3, 3))
    model = fw.Courtemanche2D()
    model.dt = dt
    model.dr = 1.0  # No neighbours to diffuse to; the value cannot reach a result.
    model.cardiac_tissue = tissue
    model.prog_bar = False
    _backend._apply_conductance_scalings(model, solved.params)

    tracker = fw.ActionPotential2DTracker()
    tracker.cell_ind = [[1, 1]]
    tracker.step = 1
    # Only the last beat is reported, and the 49 before it are 2.45 million
    # samples nothing reads. `start_time` is in model time, i.e. ms.
    tracker.start_time = (n_beats - 1) * bcl_ms
    sequence = fw.TrackerSequence()
    sequence.add_tracker(tracker)
    model.tracker_sequence = sequence

    # One stimulus in the sequence at a time, re-armed per beat rather than 50
    # stimuli queued at once: `StimSequence.stimulate_next` walks the whole list
    # on every integration step, so a queued protocol costs `n_beats` python
    # calls per step -- 125 million of them over this run.
    for beat in range(n_beats):
        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(
            fw.StimCurrentCoord2D(
                time=beat * bcl_ms,
                curr_value=CRN_STIMULUS_AMPLITUDE_MV_PER_MS,
                duration=CRN_STIMULUS_DURATION_MS,
                x1=1,
                x2=2,
                y1=1,
                y2=2,
            )
        )
        model.stim_sequence = stim_sequence
        model.t_max = (beat + 1) * bcl_ms
        # Single-threaded deliberately: the mesh is one node, so every extra
        # thread contributes a barrier and no work. Measured 1.57 s per beat on
        # one thread against 2.9 s on eight.
        if beat == 0:
            model.run(initialize=True, num_of_threads=1)
        else:
            # `initialize=False` keeps the state variables, which is the whole
            # point of pacing; the new stimulus has to be armed by hand because
            # that is what the skipped `initialize` would have done.
            stim_sequence.initialize(model)
            model.run(initialize=False, num_of_threads=1)

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
    "measure",
    "measure_anisotropy_ratio",
    "measure_conduction_velocity",
    "measure_single_cell",
]
