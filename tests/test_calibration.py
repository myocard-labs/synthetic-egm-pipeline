"""Calibration: the solve, the model card, and the round-trip that closes them.

The fast tests here are arithmetic and file handling. The **round-trip** — solve,
simulate, measure, assert the measurement returns the targets — is marked slow
because it runs the real solver, and it is the reason this module exists. Four
constants and a comment is what the 2026-06-10 calibration produced, and it hit
its conduction-velocity target while destroying action potential duration with
nobody able to notice for two months. A round-trip has no such blind spot.
"""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import numpy as np
import pytest

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.simulate import (
    CenteredGrid2D,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
)
from myocard_synthetic_egm_pipeline.simulate.calibration import (
    MeasuredValues,
    ModelCard,
    ModelTargets,
    verify_solved,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AP_EPS_PUBLISHED,
    MODEL_UNIT_APD90,
    MODEL_UNIT_CV,
    AlievPanfilovCellModel,
    CellModelSpec,
    calibrate_aliev_panfilov,
)
from myocard_synthetic_egm_pipeline.simulate.model_cards import (
    ModelCardError,
    card_provenance,
    load_model_card,
    parse_model_card,
    shipped_card_names,
)


def _shipped_cell_model() -> CellModelSpec:
    """The calibrated parameterisation, for tests that do not vary it.

    ``run_single`` requires a cell model rather than defaulting to one: it is a
    policy value, and the project rule is that libraries ship no defaults for
    policy values. Tests that are about something else get it from here.
    """
    from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card

    return load_model_card("af_remodelled_220ms", dr_mm=0.25, dr_model_units=0.25).solved


DR_MM = 0.25
DR_MODEL_UNITS = 0.25
#: Kept for readability at call sites that take them as a pair. NOT splatted with
#: ``**`` — a ``dict[str, float]`` cannot be checked against named parameters, so
#: unpacking it hides exactly the argument mistakes a type checker is for.
MESH = {"dr_mm": DR_MM, "dr_model_units": DR_MODEL_UNITS}

#: Named separately because ``ModelTargets.apd90_ms`` is ``float | None`` since
#: S18b — optional *per model*, since Courtemanche measures APD rather than
#: solving it — and the Aliev-Panfilov solve takes a plain float.
ATRIAL_APD90_MS = 220.0
ATRIAL = ModelTargets(conduction_velocity_cm_s=80.0, apd90_ms=ATRIAL_APD90_MS)


def calibrate(
    targets: ModelTargets,
    *,
    dr_mm: float = DR_MM,
    dr_model_units: float = DR_MODEL_UNITS,
) -> AlievPanfilovCellModel:
    """Test-local shim: the solve takes scalars, deliberately (FB-34).

    That signature is the migration seam for moving targets onto
    ``SubstrateStrategy`` — the solve must not learn where they live. Tests
    still want to say *solve for these targets*, so the adaptation happens
    here rather than by widening the production signature.
    """
    assert targets.apd90_ms is not None, (
        "Aliev-Panfilov solves its time unit from the APD target; a card without "
        "one is refused by verify_targets_against_solve rather than defaulted."
    )
    return calibrate_aliev_panfilov(
        conduction_velocity_cm_s=targets.conduction_velocity_cm_s,
        apd90_ms=targets.apd90_ms,
        dr_mm=dr_mm,
        dr_model_units=dr_model_units,
    )


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def test_the_solve_inverts_its_own_forward_model() -> None:
    """Pushing the solved knobs back through the physics returns the targets.

    Not a tautology worth skipping: it is the arithmetic that was wrong in June,
    when a single knob was moved to satisfy two constraints that pull on it in
    opposite directions.
    """
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)

    apd = MODEL_UNIT_APD90 * solved.time_unit_ms
    space_unit_mm = DR_MM / DR_MODEL_UNITS
    cv = (
        100.0
        * MODEL_UNIT_CV
        * float(np.sqrt(solved.diffusion))
        * space_unit_mm
        / solved.time_unit_ms
    )

    assert apd == pytest.approx(ATRIAL.apd90_ms, rel=1e-9)
    assert cv == pytest.approx(ATRIAL.conduction_velocity_cm_s, rel=1e-9)


def test_the_step_size_respects_the_stability_bound() -> None:
    """``dt`` must land strictly inside the bound, not on it.

    Violating it does not raise — it writes a well-formed bank full of a
    diverged field — so the margin is the only thing standing between a
    plausible config and silent garbage.
    """
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    bound = DR_MODEL_UNITS**2 / (4.0 * solved.diffusion)

    assert solved.dt_model_units < bound
    assert solved.dt_model_units == pytest.approx(0.9 * bound, rel=1e-12)


def test_a_faster_target_costs_a_smaller_step() -> None:
    """The cost direction, asserted rather than assumed.

    ``D`` grows as the square of the velocity target and ``dt`` falls as its
    inverse, so asking for faster conduction is quadratically expensive. Worth a
    test because it is the reason the APD target has a runtime consequence at
    all, and that is what research traded against physiology.
    """
    slow = calibrate(
        dataclasses.replace(ATRIAL, conduction_velocity_cm_s=40.0),
        dr_mm=DR_MM,
        dr_model_units=DR_MODEL_UNITS,
    )
    fast = calibrate(
        dataclasses.replace(ATRIAL, conduction_velocity_cm_s=80.0),
        dr_mm=DR_MM,
        dr_model_units=DR_MODEL_UNITS,
    )

    assert fast.diffusion == pytest.approx(4.0 * slow.diffusion, rel=1e-9)
    assert fast.dt_model_units == pytest.approx(slow.dt_model_units / 4.0, rel=1e-9)


def test_apd_is_the_only_thing_that_moves_the_time_scale() -> None:
    """``K`` follows the APD target alone — the separability the solve needs."""
    a = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    b = calibrate(
        dataclasses.replace(ATRIAL, conduction_velocity_cm_s=40.0),
        dr_mm=DR_MM,
        dr_model_units=DR_MODEL_UNITS,
    )
    assert a.time_unit_ms == pytest.approx(b.time_unit_ms, rel=1e-12)


def test_solving_away_from_the_measured_eps_is_refused() -> None:
    """The MODEL_UNIT_* constants were measured at one ``eps``; using them at
    another would be reading a number off the wrong axis, which this project has
    already done once."""
    with pytest.raises(ValueError, match="measured at eps"):
        calibrate_aliev_panfilov(
            conduction_velocity_cm_s=ATRIAL.conduction_velocity_cm_s,
            apd90_ms=ATRIAL_APD90_MS,
            eps=0.01,
            dr_mm=DR_MM,
            dr_model_units=DR_MODEL_UNITS,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conduction_velocity_cm_s", 0.0),
        ("apd90_ms", -1.0),
    ],
)
def test_impossible_targets_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(ATRIAL, **{field: value})


# ---------------------------------------------------------------------------
# The card, and the guard that keeps it honest
# ---------------------------------------------------------------------------


def test_the_shipped_card_still_agrees_with_the_solver() -> None:
    """The load-time guard, exercised on the card we actually ship.

    If :func:`calibrate` is ever changed without re-solving the card, this is
    the test that fails — and it fails here rather than silently producing a
    second physics under the same parameterisation name.
    """
    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)

    assert card.name == "af_remodelled_220ms"
    assert card.targets.apd90_ms == 220.0
    assert isinstance(card.solved, AlievPanfilovCellModel)
    assert card.solved.eps == AP_EPS_PUBLISHED


def test_the_shipped_apd_target_clears_the_trace_duration() -> None:
    """``APD >= T`` is the unconditional no-shortcut rule, so assert it directly.

    Repolarisation leaves the cropped window iff ``APD > T * (1 - p)``. Since
    ``p`` can in principle be small, only ``APD >= T`` guarantees it for the
    whole position range. A 180 ms target — which is what this chat originally
    proposed, and what the AF literature suggests if you do not notice that our
    single stimulus into rested tissue is a *long*-cycle beat — fails for any
    ``p < 0.0625`` (CL-176/178).
    """
    from myocard_synthetic_egm_pipeline.constants import DEFAULT_TRACE_DURATION_MS

    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    # Stated at all, and clearing T: Aliev-Panfilov solves its time unit from
    # this target, so a card for it without one is refused rather than defaulted.
    assert card.targets.apd90_ms is not None
    assert card.targets.apd90_ms >= DEFAULT_TRACE_DURATION_MS


def test_a_drifted_card_is_refused_rather_than_used() -> None:
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    drifted = ModelCard(
        name="drifted",
        targets=ATRIAL,
        solved=dataclasses.replace(solved, diffusion=solved.diffusion * 1.05),
    )
    with pytest.raises(ValueError, match="no longer agrees"):
        verify_solved(drifted, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


def test_a_card_rounded_for_human_reading_still_passes() -> None:
    """The tolerance exists for this and only this.

    Cards are written for people, so their numbers are rounded. The guard has to
    absorb that while still catching a routine change, which moves values by
    whole percent rather than by 0.01 %.
    """
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    rounded = ModelCard(
        name="rounded",
        targets=ATRIAL,
        solved=AlievPanfilovCellModel(
            time_unit_ms=round(solved.time_unit_ms, 4),
            diffusion=round(solved.diffusion, 4),
            eps=solved.eps,
            dt_model_units=round(solved.dt_model_units, 6),
        ),
    )
    verify_solved(rounded, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


def test_a_card_written_before_anyone_measured_it_is_valid() -> None:
    """``measured:`` is optional and may be present-but-empty.

    A card exists as soon as the solve does; the measurement comes from a run
    that has not happened yet. Treating that as malformed would force a lie into
    the file.
    """
    doc = {
        "model": {
            "name": "unmeasured",
            "targets": {
                "conduction_velocity_cm_s": 80.0,
                "apd90_ms": 220.0,
                "anisotropy_ratio": 2.0,
            },
            "solved": {
                "time_unit_ms": 5.709836,
                "diffusion": 7.826387,
                "eps": 0.002,
                "dt_model_units": 0.001797,
            },
            "measured": {"conduction_velocity_cm_s": None, "apd90_ms": None},
        }
    }
    card = parse_model_card(doc, source="test")
    assert card.measured is None


def test_an_unknown_reference_names_the_shipped_alternatives(tmp_path: Path) -> None:
    with pytest.raises(ModelCardError, match="shipped name"):
        load_model_card(
            "no_such_card", config_dir=tmp_path, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
        )


def test_a_card_can_be_a_path_beside_the_config(tmp_path: Path) -> None:
    """The escape hatch: an experiment does not have to ship in the package."""
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    (tmp_path / "mine.yaml").write_text(
        textwrap.dedent(f"""
            model:
              name: mine
              targets:
                conduction_velocity_cm_s: 80.0
                apd90_ms: 220.0
                anisotropy_ratio: 2.0
              solved:
                time_unit_ms: {solved.time_unit_ms}
                diffusion: {solved.diffusion}
                eps: {solved.eps}
                dt_model_units: {solved.dt_model_units}
        """)
    )
    card = load_model_card(
        "mine.yaml", config_dir=tmp_path, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    assert card.name == "mine"


def test_provenance_carries_values_not_a_path() -> None:
    """A bank records what the card *said*, because a path can change under it."""
    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    provenance = card_provenance(
        dataclasses.replace(
            card, measured=MeasuredValues(conduction_velocity_cm_s=86.7, apd90_ms=219.4)
        )
    )

    assert provenance["model_card_name"] == "af_remodelled_220ms"
    assert provenance["model_target_apd90_ms"] == 220.0
    assert provenance["model_measured_conduction_velocity_cm_s"] == 86.7
    assert not any("path" in key for key in provenance)


def test_the_shipped_card_is_discoverable_by_name() -> None:
    assert "af_remodelled_220ms" in shipped_card_names()


# ---------------------------------------------------------------------------
# RunConfig refuses to be quietly wrong
# ---------------------------------------------------------------------------


def test_the_solved_step_sits_inside_the_stability_bound() -> None:
    """The bound is a **joint** property, and that is why it moved.

    ``diffusion`` belongs to the cell model, ``dr`` to the backend, so neither
    object can check it alone — the backend does, at the one point where both
    are in hand. What the cell model can answer is *what the limit is*, and the
    solve has to respect it.

    This matters more than a normal range check: exceeding the bound does not
    crash, it writes a structurally perfect bank full of a diverged field.
    """
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    limit = solved.stability_limit(dr_model_units=DR_MODEL_UNITS)

    assert solved.dt_model_units < limit
    # The legacy pairing (dt 0.01 at D=1) is stable; at the calibrated D near 8
    # it is not, which is the whole reason dt stopped being a constant.
    assert limit < 0.01


def test_the_time_conversion_belongs_to_the_cell_model() -> None:
    """``ms_to_model_time`` is the seam that lets Courtemanche join.

    Callers used to write ``duration_ms / ap_time_unit_ms`` — an Aliev-Panfilov
    idiom leaking into the runner, and one that is simply wrong for a
    dimensional model. Asking the model instead means Courtemanche can answer
    *the same number* and nothing upstream changes.
    """
    solved = calibrate(ATRIAL, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    assert solved.ms_to_model_time(solved.time_unit_ms) == pytest.approx(1.0)
    assert solved.ms_to_model_time(0.0) == 0.0


def test_a_run_config_carries_the_card_without_carrying_its_parameters() -> None:
    """The card rides on ``RunConfig`` as a **label**, not as knobs.

    S38b put ``eps``, ``diffusion`` and the rest here, which design note D2
    forbids — a Courtemanche run would have carried an ``eps`` meaning nothing
    to it. What remains is provenance: which named parameterisation produced
    this bank, in physiological terms rather than four solved numbers.
    """
    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    config = RunConfig(trace_duration_ms=192.0, output_fs_hz=1000.0, model_card=card)

    assert config.model_card is card
    for gone in ("diffusion", "membrane_eps", "dt_model_units", "ap_time_unit_ms"):
        assert not hasattr(config, gone), (
            f"RunConfig still carries {gone!r}; cell-model parameters belong on "
            "the CellModelSpec (D2)."
        )


# The backend's refusal of an unfamiliar cell model moved to
# ``test_courtemanche.py`` when Courtemanche stopped being the unfamiliar one:
# a test that names the supported set belongs beside the model that joined it.


# ---------------------------------------------------------------------------
# The round-trip — the point of all of the above
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_solving_then_simulating_returns_the_targets() -> None:
    """Solve, simulate, measure, assert the measurement returns the targets.

    **This is the test that would have failed in June 2026.** The calibration of
    that date hit conduction velocity by shrinking the time-scale constant,
    which divided APD by 6.5 — and since the only artifact was four constants,
    nothing contradicted it until a repolarisation deflection turned up inside
    the analysis window two months later.

    Tolerances are asymmetric on purpose. APD is set by the time-scale constant
    alone and lands tight. Conduction velocity carries a discretization excess
    the closed form ignores — a larger diffusion coefficient widens the upstroke
    relative to a fixed mesh — so it runs high by a few percent. That is
    recorded in the card's ``measured`` block rather than tuned away, and a few
    percent is immaterial against a literature spread of 88 +/- 9 cm/s.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import measure

    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    # Anisotropy is geometry's, not the card's (FB-34): the default is 2.0.
    geometry = Patch2DGeometry(size_mm=16.0, dr_mm=DR_MM)

    assert isinstance(card.solved, AlievPanfilovCellModel)
    measured = measure(
        geometry=geometry,
        solved=card.solved,
        dr_model_units=DR_MODEL_UNITS,
    )

    assert measured.apd90_ms == pytest.approx(card.targets.apd90_ms, rel=0.05), (
        f"APD90 missed: wanted {card.targets.apd90_ms}, measured {measured.apd90_ms:.1f}"
    )
    assert measured.conduction_velocity_cm_s == pytest.approx(
        card.targets.conduction_velocity_cm_s, rel=0.15
    ), (
        "conduction velocity missed: wanted "
        f"{card.targets.conduction_velocity_cm_s}, measured "
        f"{measured.conduction_velocity_cm_s:.1f}"
    )

    # Printed so the card's `measured:` block can be filled from a real run
    # rather than guessed. pytest shows it with -s or on failure.
    print(
        f"\nmeasured: conduction_velocity_cm_s: {measured.conduction_velocity_cm_s:.1f}"
        f"\nmeasured: apd90_ms: {measured.apd90_ms:.1f}"
    )


@pytest.mark.slow
def test_the_realized_anisotropy_matches_the_card() -> None:
    """The knob S38a made operative, checked at the value S38b ships."""
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import (
        measure_anisotropy_ratio,
    )

    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    geometry = Patch2DGeometry(size_mm=12.0, dr_mm=DR_MM)

    assert isinstance(card.solved, AlievPanfilovCellModel)
    realized = measure_anisotropy_ratio(
        geometry=geometry,
        solved=card.solved,
        dr_model_units=DR_MODEL_UNITS,
    )
    assert realized == pytest.approx(geometry.anisotropy_ratio, rel=0.12)


@pytest.mark.slow
def test_no_second_deflection_survives_inside_the_cropped_window() -> None:
    """The shortcut is gone — asserted where it actually mattered.

    **On the cropped trace, not the raw capture** (CL-178). The simulation runs
    far longer than the 192-sample window, which is cut out of it afterwards at
    position ``p``; "inside the window" is a statement about the crop and does
    not follow from anything measured on the capture. The original 51 ms
    finding was measured on the capture and the window arithmetic was left
    implicit — correct, as it turned out, but unstated.

    At the **smallest ``p``**, because that places the window end furthest after
    the activation and so gives a repolarisation tail the most room to appear.
    Testing at mid-range would pass while the extreme failed.
    """
    from myocard_egm_signal import RectifiedDerivative, detect_activation

    from myocard_synthetic_egm_pipeline.simulate import (
        run_single,
    )
    from myocard_synthetic_egm_pipeline.simulate.sizing import (
        required_capture_duration_ms,
    )

    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)
    geometry = Patch2DGeometry(size_mm=12.0, dr_mm=DR_MM)
    rng = np.random.default_rng(0)
    electrodes = CenteredGrid2D.sample(geometry=geometry, rng=rng)

    position_low = 0.4
    config = RunConfig(
        trace_duration_ms=192.0,
        output_fs_hz=1000.0,
        capture_oversample=1,
        capture_duration_ms=required_capture_duration_ms(
            trace_duration_ms=192.0,
            output_fs_hz=1000.0,
            position_low=position_low,
        ),
        model_card=card,
    )

    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge="left"),
        electrodes=electrodes,
        backend=__import__(
            "myocard_synthetic_egm_pipeline.backends.finitewave",
            fromlist=["FinitewaveBackend"],
        ).FinitewaveBackend(),
        cell_model=_shipped_cell_model(),
        config=config,
        rng=np.random.default_rng(0),
    )

    traces = np.asarray(result.bipolar_traces, dtype=np.float64)
    fs = float(config.output_fs_hz)
    guard = int(0.020 * fs)

    worst = 0.0
    for trace in traces:
        activation_index = detect_activation(trace, preprocessor=RectifiedDerivative())
        slope = np.abs(np.gradient(trace))
        masked = slope.copy()
        masked[max(0, activation_index - guard) : activation_index + guard] = 0.0
        if slope[activation_index] > 0:
            worst = max(worst, float(masked.max() / slope[activation_index]))

    # At APD 51 ms the repolarisation marker measured 9-21 % of the activation
    # deflection at a lag of 51.2 +/- 1.0 ms. Anything that large inside the
    # window now means it is back.
    assert worst < 0.09, (
        f"a secondary deflection reaching {worst:.1%} of the activation slope is "
        "inside the cropped window; at APD 51 ms the repolarisation shortcut "
        "measured 9-21 %, so this looks like it has returned"
    )
