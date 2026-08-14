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
    MODEL_UNIT_APD90,
    MODEL_UNIT_CV,
    AlievPanfilovCellModel,
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


def _run_plane_wave(
    *,
    geometry: Patch2DGeometry,
    solved: AlievPanfilovCellModel,
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
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as _backend

    model = fw.AlievPanfilov2D()
    model.dt = solved.dt_model_units
    model.dr = dr_model_units
    model.D_model = solved.diffusion
    model.eps = solved.eps

    tissue = _backend._build_tissue_2d(geometry)
    _backend._configure_anisotropy_2d(model, geometry)
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
    activation_tracker.threshold = _ACTIVATION_THRESHOLD
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
) -> float:
    """APD90 in ms: 10 % upstroke to 90 % repolarisation.

    Definitions differ between papers by tens of milliseconds, so the one used
    here is stated rather than assumed — and it is the same one
    ``MODEL_UNIT_APD90`` was measured with, which is what makes the round-trip
    a closed loop rather than two unrelated numbers.
    """
    if potential.size == 0 or float(potential.max()) < _ACTIVATION_THRESHOLD:
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


def measure(
    *,
    geometry: Patch2DGeometry,
    solved: AlievPanfilovCellModel,
    dr_model_units: float,
    t_max_model_units: float | None = None,
) -> MeasuredValues:
    """Simulate at ``solved`` and report the physical CV and APD it produces.

    ``t_max`` defaults to enough model time for the wave to cross the patch at
    the *target* velocity plus three action potentials — generous, because the
    failure mode of too-short is a confusing exception rather than a wrong
    number, and this runs a handful of times, not per simulation.

    Conduction velocity is measured **along the fibres** (``left`` edge with the
    default ``fiber_angle_rad = 0``), matching what ``calibrate`` targets and
    what the literature quotes.
    """
    if t_max_model_units is None:
        # Crossing time at the analytic velocity, in model time units, with a
        # 2x margin, plus three APDs so repolarisation is definitely complete.

        cells = geometry.size_mm / geometry.dr_mm
        span_space_units = cells * dr_model_units
        crossing = span_space_units / (MODEL_UNIT_CV * float(np.sqrt(solved.diffusion)))
        t_max_model_units = 2.0 * crossing + 3.0 * MODEL_UNIT_APD90

    activation, potential = _run_plane_wave(
        geometry=geometry,
        solved=solved,
        dr_model_units=dr_model_units,
        edge="left",
        t_max_model_units=t_max_model_units,
    )

    return MeasuredValues(
        conduction_velocity_cm_s=_velocity_from_activation(
            activation,
            edge="left",
            dr_mm=geometry.dr_mm,
            time_unit_ms=solved.time_unit_ms,
        ),
        apd90_ms=_apd90_from_potential(
            potential,
            dt_model_units=solved.dt_model_units,
            time_unit_ms=solved.time_unit_ms,
        ),
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
            time_unit_ms=solved.time_unit_ms,
        )

    return velocities["left"] / velocities["top"]


__all__: list[str] = ["measure", "measure_anisotropy_ratio"]
