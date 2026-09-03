"""Courtemanche: the seam, the partial solve, the card, and the published vector.

The fast tests here are arithmetic, dispatch and file handling. The **slow**
ones are the point of the module: Courtemanche arrives with no known-good
prior fixture of our own — the recalibration that came just before it moved
every number on purpose, so no earlier run of ours is a baseline — and it is
therefore validated against **Wilhelms et al. 2012**'s published five-element
vector. That is weaker than a self-comparison, and it is exactly why the five
values are asserted rather than eyeballed.
"""

from __future__ import annotations

import dataclasses
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.backends.finitewave.measure import strip_step_ms
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
    CRN_AF_SCALINGS,
    CRN_AF_SEVERITY,
    CRN_CALIBRATION_DR_MM,
    CRN_DIASTOLIC_THRESHOLD_MV_PER_MS,
    CRN_MAX_DT_MS,
    CRN_PACING_BCL_MS,
    CRN_PACING_BEATS,
    CRN_REFERENCE_CV_CM_S,
    CRN_REFERENCE_DIFFUSION,
    CRN_STIMULUS_AMPLITUDE_MV_PER_MS,
    WILHELMS_2012_CRN_CONTROL,
    AlievPanfilovCellModel,
    CellModelSpec,
    CourtemancheCellModel,
    calibrate_courtemanche,
)
from myocard_synthetic_egm_pipeline.simulate.model_cards import (
    ModelCardError,
    card_provenance,
    load_model_card,
    parse_model_card,
    shipped_card_names,
)

DR_MM = CRN_CALIBRATION_DR_MM
DR_MODEL_UNITS = CRN_CALIBRATION_DR_MM

#: The shipped Courtemanche parameterisation. Control conductances, solved for
#: the same conduction velocity the Aliev-Panfilov card targets.
CARD_NAME = "courtemanche_control"

#: The Aliev-Panfilov card's pitch, which is **not** the Courtemanche one.
#:
#: The mesh-convergence sweep moved Courtemanche from 0.25 mm to 0.10; the
#: Aliev-Panfilov card is still solved at 0.25 and is loaded here at its own
#: pitch. That difference is a real open question for the model comparison —
#: everything but the membrane is supposed to match — and it is named here
#: rather than hidden behind a shared constant that would make the two look
#: interchangeable.
AP_DR_MM = 0.25


def shipped_card() -> ModelCard:
    return load_model_card(CARD_NAME, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


# ---------------------------------------------------------------------------
# The seam CellModelSpec was cut for
# ---------------------------------------------------------------------------


def test_ms_to_model_time_is_the_identity_for_courtemanche() -> None:
    """Asserted, not assumed, because the failure it prevents is silent.

    Aliev-Panfilov is dimensionless, so callers used to reach solver time by
    writing ``duration_ms / ap_time_unit_ms``. Courtemanche has no such
    constant. The Courtemanche version of that expression is a division by an
    attribute that does not exist — which in Python is an ``AttributeError``
    several frames from the cause if you are lucky, and a division by whatever
    else was in scope if you are not. Answering *"the same number"* is what the
    fifth spec exists to make possible, so the identity is a fixture rather
    than an implementation detail.
    """
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )

    for duration_ms in (0.0, 1.0, 192.0, 1000.0, 12345.678):
        assert solved.ms_to_model_time(duration_ms) == duration_ms

    # And the attribute whose absence makes the identity necessary really is
    # absent, rather than present and equal to 1.0 — a `time_unit_ms` of 1.0
    # would make every test here pass while restoring the very knob a
    # dimensional model has no business carrying.
    assert not hasattr(solved, "time_unit_ms")


def test_courtemanche_satisfies_the_cell_model_protocol() -> None:
    """Frozen dataclass against a Protocol whose ``type`` is a read-only property.

    Writing it as a bare ``type: str`` demands a *settable* attribute, and this
    exact check is what caught ``AlievPanfilovCellModel`` silently failing to
    satisfy its own Protocol.
    """
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    assert isinstance(solved, CellModelSpec)
    assert solved.type == "courtemanche"


# ---------------------------------------------------------------------------
# The solve: conduction velocity only, by design
# ---------------------------------------------------------------------------


def test_the_solve_takes_a_velocity_target_and_no_apd_target() -> None:
    """APD is measured, never solved — so it is not an argument at all.

    Not a convenience: there is no time-unit constant to invert and no
    closed-form path from the ionic equations to a duration. A signature that
    accepted ``apd90_ms`` would have to either ignore it or pretend to hit it.
    """
    import inspect

    parameters = inspect.signature(calibrate_courtemanche).parameters
    assert "conduction_velocity_cm_s" in parameters
    assert "apd90_ms" not in parameters


def test_doubling_the_velocity_target_quadruples_diffusion() -> None:
    """``CV ~ sqrt(D)``, which is the one law that survives the model swap.

    It is a property of the diffusion operator rather than of the membrane,
    which is why this half of the calibration reads the same for both models —
    and why only the *constant* had to be re-measured for Courtemanche.
    """
    slow = calibrate_courtemanche(
        conduction_velocity_cm_s=40.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    fast = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    assert fast.diffusion == pytest.approx(4.0 * slow.diffusion, rel=1e-9)


def test_the_reference_point_solves_back_to_the_reference_diffusion() -> None:
    """Asking for the velocity that was measured returns the diffusion it was
    measured at. A closed loop over the one measured constant in the solve."""
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=CRN_REFERENCE_CV_CM_S,
        dr_mm=DR_MM,
        dr_model_units=DR_MODEL_UNITS,
    )
    assert solved.diffusion == pytest.approx(CRN_REFERENCE_DIFFUSION, rel=1e-12)


def test_the_solved_step_respects_both_bounds() -> None:
    """Two bounds, and **which one binds moved when the mesh refined**.

    Courtemanche has an ionic ceiling as well as the explicit-diffusion CFL
    condition: its fast sodium current needs a smaller step than the diffusion
    bound permits at a coarse pitch, and because Finitewave integrates the
    gating variables Rush-Larsen a step that is too large does not diverge — it
    flattens the upstroke, which is the one observable an ionic model was added
    to get right. So the limit has to report the *smaller* of the two.

    At the old 0.25 mm pitch the ionic ceiling bound and the CFL bound was
    slack. At the shipped 0.10 mm it is the other way round: CFL scales as
    ``dr**2``, so refining by 2.5x tightened it 6.25x and it now bites first.
    **The previous version of this test asserted the ceiling bound and failed
    when the pitch changed, with the message it carried for exactly that case**
    — which is the test doing its job, not an obstacle. What is asserted here
    is the invariant that survives either regime: the step is inside both.
    """
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    cfl = DR_MODEL_UNITS**2 / (4.0 * solved.diffusion)

    assert solved.dt_model_units <= solved.stability_limit(dr_model_units=DR_MODEL_UNITS)
    assert solved.dt_model_units <= CRN_MAX_DT_MS
    assert solved.dt_model_units < cfl
    # And the binding one at the shipped pitch is the diffusion bound, with the
    # safety factor applied. If this flips, the sentence above is stale.
    assert solved.dt_model_units == pytest.approx(0.9 * cfl, rel=1e-9), (
        "the ionic ceiling has become the binding constraint again — the pitch "
        "or the diffusion moved, and the reasoning above needs re-reading."
    )


def test_a_space_step_that_is_not_millimetres_is_still_refused() -> None:
    """A unit error, and the one mesh fault that is still fatal.

    Courtemanche's diffusion is in mm^2/ms, so its space unit **is** the
    millimetre: a ``dr_model_units`` that disagrees with ``dr_mm`` is not a
    rescaling but a different mesh from the one the geometry describes, and
    nothing downstream could reconstruct which was meant. That is unlike an
    unregistered *pitch*, which is a known quantity measured on the wrong mesh
    and can be borrowed with a warning.
    """
    with pytest.raises(ValueError, match="space unit is the millimetre"):
        calibrate_courtemanche(conduction_velocity_cm_s=80.0, dr_mm=0.10, dr_model_units=0.5)


def test_an_unregistered_pitch_warns_rather_than_refusing() -> None:
    """The under-resolution trap is still real; it is now reported, not blocked.

    A CV-solve absorbs discretisation error into ``diffusion`` and hits its
    target anyway, leaving a physical-looking number that is not. Measured on a
    cable at a **fixed** diffusion, the same tissue conducts at 81.85 cm/s on a
    0.25 mm mesh and 87.94 on a 0.05 mm one — 7.4 % apart with no physics
    changed. So an anchor is evidence about one pitch, and using it at another
    has to be *visible*: it warns here and is written into the bank.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        AnchorSubstitutionWarning,
    )

    with pytest.warns(AnchorSubstitutionWarning, match="ABOVE target"):
        calibrate_courtemanche(conduction_velocity_cm_s=80.0, dr_mm=0.05, dr_model_units=0.05)
    with pytest.warns(AnchorSubstitutionWarning, match="BELOW target"):
        calibrate_courtemanche(conduction_velocity_cm_s=80.0, dr_mm=0.25, dr_model_units=0.25)


def test_a_registered_conductance_set_solves_without_a_warning() -> None:
    """The control on every substitution test above.

    An anchor belongs to a conductance set as much as to a mesh — the shipped
    AF scalings move conduction velocity at fixed diffusion by 2.3 % — so
    solving an unregistered set borrows and warns. This asserts the other side:
    the registered set solves **silently**, so those tests are not passing for
    the trivial reason that every solve warns.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        AnchorSubstitutionWarning,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", AnchorSubstitutionWarning)
        registered = calibrate_courtemanche(
            conduction_velocity_cm_s=80.0,
            dr_mm=DR_MM,
            dr_model_units=DR_MODEL_UNITS,
            params=CRN_AF_SCALINGS,
        )
    assert registered.params == CRN_AF_SCALINGS


def test_a_scaling_that_deletes_a_current_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        CourtemancheCellModel(diffusion=0.3, dt_model_units=0.02, params={"g_to_scale": 0.0})


def test_the_scalings_a_spec_carries_cannot_be_edited_underneath_it() -> None:
    """A card's parameters are provenance, so the mapping is not a shared handle."""
    scalings = {"g_to_scale": 0.5}
    solved = CourtemancheCellModel(diffusion=0.3, dt_model_units=0.02, params=scalings)
    scalings["g_to_scale"] = 99.0

    assert solved.params["g_to_scale"] == 0.5
    with pytest.raises(TypeError):
        solved.params["g_to_scale"] = 99.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_the_shipped_courtemanche_card_still_agrees_with_the_solver() -> None:
    """The load-time guard, on the card we actually ship."""
    card = shipped_card()

    assert card.name == CARD_NAME
    assert CARD_NAME in shipped_card_names()
    assert isinstance(card.solved, CourtemancheCellModel)
    assert card.solved.params == {}, "the shipped card is control, not remodelled"
    assert card.targets.conduction_velocity_cm_s == 80.0


def test_the_shipped_courtemanche_card_states_no_apd_target() -> None:
    """Partial targets are the design, not an omission.

    Courtemanche measures APD rather than solving it, so control CRN's APD90 is
    whatever the published conductances produce. Recording it as a *target*
    would claim it had been aimed at.
    """
    card = shipped_card()

    assert card.targets.apd90_ms is None
    assert card.measured is not None
    assert card.measured.apd90_ms > 0


def test_the_card_records_both_upstrokes_and_says_which_is_which() -> None:
    """Two upstrokes, and conflating them is what produced the 14 % outlier.

    The card's ``measured.upstroke_v_s`` is the **propagated** one — dV/dt max
    at the patch centre, 20 mm from the stimulus, driven by the arriving
    wavefront. The **stimulated** single-cell figure is a different
    measurement, recorded in the card's validation block, and it is the one a
    published cell table reports.

    They are not close: 131 against 215 V/s. An isolated cell puts all its
    sodium current into its own membrane, while a cell in tissue spends much of
    it charging the cells ahead. Asserting they differ substantially is what
    stops a future edit from filling one field with the other's number, which
    would read as perfectly plausible.
    """
    card = shipped_card()
    assert card.measured is not None
    propagated = card.measured.upstroke_v_s
    assert propagated is not None, (
        "the Courtemanche card records no propagated upstroke; it is the card's "
        "physical claim about what a stored trace sees"
    )

    assert 100.0 < propagated < 180.0, (
        f"a propagated upstroke of {propagated} V/s is outside the range loaded "
        "tissue produces — check it has not been filled with the single-cell value"
    )
    stimulated = WILHELMS_2012_CRN_CONTROL["dvdt_max_v_s"]
    assert propagated < 0.8 * stimulated, (
        f"the propagated upstroke ({propagated}) is not clearly below the "
        f"stimulated single-cell figure ({stimulated}); the two have most likely "
        "been conflated"
    )

    provenance = card_provenance(card)
    assert provenance["model_measured_propagated_upstroke_v_s"] == propagated, (
        "the bank records the upstroke under a name that does not say which of "
        "the two measurements it is"
    )


def test_an_aliev_panfilov_card_records_no_upstroke() -> None:
    """A rate of change of a dimensionless ``u`` is not a volts-per-second.

    Optional rather than universal, so the field is absent on a card that
    cannot fill it honestly rather than carrying a number in units it does not
    have — which would invite exactly the cross-model comparison the units
    forbid.
    """
    card = load_model_card("af_remodelled_220ms", dr_mm=AP_DR_MM, dr_model_units=AP_DR_MM)

    assert card.measured is not None
    assert card.measured.upstroke_v_s is None
    assert "model_measured_propagated_upstroke_v_s" not in card_provenance(card)


def test_the_measured_courtemanche_apd_clears_the_trace_duration() -> None:
    """``APD >= T`` is the unconditional no-shortcut rule.

    Aliev-Panfilov reaches it by solving; Courtemanche has to be *checked*,
    because nothing in its solve is aiming at it. Control CRN clears T with
    margin, which is what made this card authorable ahead of the severity
    sweep — full cAF remodelling gives 143.87 ms, below T, and would bring the
    repolarisation shortcut straight back.
    """
    from myocard_synthetic_egm_pipeline.constants import DEFAULT_TRACE_DURATION_MS

    card = shipped_card()
    assert card.measured is not None
    assert card.measured.apd90_ms >= DEFAULT_TRACE_DURATION_MS


def test_both_shipped_cards_aim_at_the_same_conduction_velocity() -> None:
    """What makes the A/B a comparison of models rather than of tissues.

    CV is the target both solves share; upstroke morphology is what the
    comparison is actually isolating, and it only reads as a model difference
    if everything solvable is matched.
    """
    courtemanche = shipped_card()
    aliev_panfilov = load_model_card("af_remodelled_220ms", dr_mm=AP_DR_MM, dr_model_units=AP_DR_MM)
    assert (
        courtemanche.targets.conduction_velocity_cm_s
        == aliev_panfilov.targets.conduction_velocity_cm_s
    )


def test_an_aliev_panfilov_card_still_loads_and_verifies_unchanged() -> None:
    """The regression guard for a change that touched every card path.

    Adding a second cell model gave the parser a dispatch table, made
    ``targets.apd90_ms`` optional and gave verification a branch. None of that
    may move the Aliev-Panfilov card's numbers — a bank generated before and
    after it must be identical.
    """
    card = load_model_card("af_remodelled_220ms", dr_mm=AP_DR_MM, dr_model_units=AP_DR_MM)

    assert isinstance(card.solved, AlievPanfilovCellModel)
    assert card.solved.time_unit_ms == 5.709836
    assert card.solved.diffusion == 7.826387
    assert card.solved.eps == 0.002
    assert card.solved.dt_model_units == 0.001797
    assert card.targets.apd90_ms == 220.0

    provenance = card_provenance(card)
    assert provenance["model_target_apd90_ms"] == 220.0
    assert provenance["ap_diffusion"] == 7.826387
    assert "crn_diffusion" not in provenance


def test_a_card_naming_an_unregistered_model_is_refused_by_name() -> None:
    """By name, and listing what is registered.

    The alternative is worse than an error: a Courtemanche ``diffusion`` is in
    mm^2/ms and an Aliev-Panfilov one is dimensionless, so a card parsed as the
    wrong model would be off by three orders of magnitude and still load.
    """
    doc = {
        "model": {
            "name": "stranger",
            "type": "hodgkin_huxley",
            "targets": {"conduction_velocity_cm_s": 80.0},
            "solved": {"diffusion": 0.3, "dt_ms": 0.02},
        }
    }
    with pytest.raises(ModelCardError, match="hodgkin_huxley"):
        parse_model_card(doc, source="test")


def test_a_courtemanche_card_can_be_a_path_beside_the_config(tmp_path: Path) -> None:
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    (tmp_path / "mine.yaml").write_text(
        textwrap.dedent(f"""
            model:
              name: mine
              type: courtemanche
              targets:
                conduction_velocity_cm_s: 80.0
              solved:
                diffusion: {solved.diffusion}
                dt_ms: {solved.dt_model_units}
        """)
    )
    card = load_model_card(
        "mine.yaml", config_dir=tmp_path, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    assert card.name == "mine"
    assert isinstance(card.solved, CourtemancheCellModel)


def test_a_drifted_courtemanche_card_is_refused_rather_than_used() -> None:
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    drifted = ModelCard(
        name="drifted",
        targets=ModelTargets(conduction_velocity_cm_s=80.0),
        solved=dataclasses.replace(solved, diffusion=solved.diffusion * 1.05),
    )
    with pytest.raises(ValueError, match="no longer agrees"):
        verify_solved(drifted, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


def test_an_aliev_panfilov_card_without_an_apd_target_is_refused() -> None:
    """Optional per model, not optional in general.

    Aliev-Panfilov *solves* its time unit from the APD target; without one there
    is nothing that fixed the recorded ``time_unit_ms`` and nothing to verify
    against. Silently defaulting is how a card would come to mean two things.
    """
    card = ModelCard(
        name="no_apd",
        targets=ModelTargets(conduction_velocity_cm_s=80.0),
        solved=AlievPanfilovCellModel(
            time_unit_ms=5.709836, diffusion=7.826387, eps=0.002, dt_model_units=0.001797
        ),
    )
    with pytest.raises(ValueError, match="states no apd90_ms target"):
        verify_solved(card, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


def test_a_courtemanche_apd_target_is_checked_against_the_measurement() -> None:
    """A stated target that nothing verifies is worse than an absent one.

    Courtemanche cannot solve for APD, so a card that states an APD target is
    claiming its *chosen parameters* reach it — the claim an AF-remodelled
    card's severity sweep makes. The check is two recorded numbers against
    each other, so it stays
    load-time cheap, and it is the reason ``apd90_ms`` may be present on a
    Courtemanche card at all rather than being refused outright.
    """
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    aimed = ModelCard(
        name="aimed",
        targets=ModelTargets(conduction_velocity_cm_s=80.0, apd90_ms=220.0),
        solved=solved,
        measured=MeasuredValues(conduction_velocity_cm_s=80.0, apd90_ms=294.8),
    )
    with pytest.raises(ValueError, match="measures APD rather than solving it"):
        verify_solved(aimed, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)

    reached = dataclasses.replace(
        aimed, measured=MeasuredValues(conduction_velocity_cm_s=80.0, apd90_ms=224.0)
    )
    verify_solved(reached, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def test_the_backend_builds_the_courtemanche_solver_from_the_spec() -> None:
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as be

    card = shipped_card()
    assert isinstance(card.solved, CourtemancheCellModel)
    native = be._build_model_2d(
        cell_model=card.solved,
        geometry=Patch2DGeometry(size_mm=8.0, dr_mm=DR_MM),
        dr_model_units=DR_MODEL_UNITS,
    )

    assert native.model_class == "Courtemanche"
    assert native.model.D_model == card.solved.diffusion
    assert native.model.dt == card.solved.dt_model_units
    # The identity again, from the other side: a step in model units is a step
    # in ms, so the capture stride needs no conversion.
    assert native.dt_ms == card.solved.dt_model_units


def test_the_backend_refuses_a_mesh_pitch_that_is_not_in_millimetres() -> None:
    """Courtemanche's diffusion is in mm^2/ms, so its space unit is fixed.

    Aliev-Panfilov is dimensionless and its ``dr_model_units`` may differ from
    ``dr_mm`` freely — the calibration absorbs the ratio. Reusing that freedom
    here would not rescale anything; it would simulate a different mesh from the
    one the geometry describes and the wave would travel at the wrong speed.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as be

    card = shipped_card()
    with pytest.raises(ValueError, match="physical units"):
        be._build_model_2d(
            cell_model=card.solved,
            geometry=Patch2DGeometry(size_mm=8.0, dr_mm=0.25),
            dr_model_units=0.5,
        )


def test_conductance_scalings_are_applied_by_assignment() -> None:
    """No patching: Finitewave reads these attributes at kernel-run time.

    The multiply is against the *shipped* default, so a scaling means the
    fraction of the published conductance its name says.
    """
    import finitewave as fw

    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as be

    default_gto = fw.Courtemanche2D().gto
    native = be._build_model_2d(
        cell_model=CourtemancheCellModel(
            diffusion=CRN_REFERENCE_DIFFUSION,
            dt_model_units=strip_step_ms(diffusion=CRN_REFERENCE_DIFFUSION, dr_mm=DR_MM),
            params={"g_to_scale": 0.35, "g_K1_scale": 2.1},
        ),
        geometry=Patch2DGeometry(size_mm=8.0, dr_mm=DR_MM),
        dr_model_units=DR_MODEL_UNITS,
    )

    assert native.model.gto == pytest.approx(default_gto * 0.35)
    assert native.model.gk1 == pytest.approx(fw.Courtemanche2D().gk1 * 2.1)


def test_a_scaling_the_solver_cannot_apply_is_refused_by_name() -> None:
    """The alternative is a dead attribute and a sweep that does nothing.

    ``I_Kur`` is the live case and it is no longer hypothetical: finitewave
    0.9.3 computes its conductance from voltage inside the kernel rather than
    reading a parameter, so the cAF -49 % I_Kur change **is not applied by the
    shipped AF card** — which declares the omission rather than working around
    it. Discovering that here cost a message; discovering it from a severity
    sweep that silently moved three currents of four would have cost a day.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as be

    with pytest.raises(ValueError, match="g_Kur_scale"):
        be._build_model_2d(
            cell_model=CourtemancheCellModel(
                diffusion=CRN_REFERENCE_DIFFUSION,
                dt_model_units=strip_step_ms(diffusion=CRN_REFERENCE_DIFFUSION, dr_mm=DR_MM),
                params={"g_Kur_scale": 0.51},
            ),
            geometry=Patch2DGeometry(size_mm=8.0, dr_mm=DR_MM),
            dr_model_units=DR_MODEL_UNITS,
        )


def test_the_backend_still_refuses_an_unfamiliar_cell_model() -> None:
    """Refusal is by name, because the alternative is silent nonsense.

    A backend that duck-typed its way into an unfamiliar membrane model would
    integrate *something* and return numbers.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend

    class _Stranger:
        type = "hodgkin_huxley"
        dt_model_units = 0.01

        def ms_to_model_time(self, duration_ms: float) -> float:
            return duration_ms

        def stability_limit(self, *, dr_model_units: float, dimensions: int = 2) -> float:
            return 1.0

        def to_metadata(self) -> dict[str, object]:
            return {}

    geometry = Patch2DGeometry(size_mm=8.0, dr_mm=DR_MM)
    with pytest.raises(ValueError, match="hodgkin_huxley"):
        FinitewaveBackend().simulate(
            geometry=geometry,
            substrate=UniformRandomFibrosis(density=0.0),
            activation=PlanarEdgeStimulus(edge="left"),
            electrodes=CenteredGrid2D.sample(geometry=geometry, rng=np.random.default_rng(0)),
            cell_model=_Stranger(),
            config=RunConfig(trace_duration_ms=192.0, output_fs_hz=1000.0),
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# What a bank records
# ---------------------------------------------------------------------------


def test_a_courtemanche_bank_records_the_cell_model_from_the_spec() -> None:
    """The contract's Courtemanche variant carries no time-unit field.

    It runs in milliseconds, so there is nothing to calibrate and nothing to
    record — the asymmetry with ``AlievPanfilovCellModel.ap_time_unit_ms`` is
    the schema agreeing with the physics.
    """
    from myocard_synthetic_egm_pipeline.simulate.bank_config import cell_model_model

    control = cell_model_model(shipped_card().solved)
    assert control.type == "courtemanche"
    assert control.params is None
    assert not hasattr(control, "ap_time_unit_ms")

    remodelled = cell_model_model(
        CourtemancheCellModel(diffusion=0.3, dt_model_units=0.02, params={"g_to_scale": 0.35})
    )
    assert remodelled.params == {"g_to_scale": 0.35}


def test_a_courtemanche_bank_records_its_timestep_in_the_typed_field() -> None:
    """``crn_dt_ms`` is the same fact as ``ap_dt_model_units``, so it fills the
    same schema field.

    The 2.0 restructure's rule is that a fact the schema has a field for does
    not go in ``params``. Left alone, a Courtemanche bank would have reported
    ``dt_model_units: null`` beside a ``crn_dt_ms`` in the free-form bag — the
    generic-bag failure, reintroduced by a key rename.
    """
    from myocard_synthetic_egm_pipeline.simulate.bank_config import backend_model

    solved = shipped_card().solved
    assert isinstance(solved, CourtemancheCellModel)
    metadata = {"backend_name": "finitewave", **solved.to_metadata()}
    backend = backend_model(metadata, output_fs_hz=1000.0, capture_oversample=4)

    assert backend.dt_model_units == solved.dt_model_units
    assert "crn_dt_ms" not in (backend.params or {})
    assert (backend.params or {})["crn_diffusion"] == solved.diffusion


def test_the_provenance_a_courtemanche_bank_carries() -> None:
    card = shipped_card()
    provenance = card_provenance(card)

    assert provenance["model_card_name"] == CARD_NAME
    assert provenance["cell_model_type"] == "courtemanche"
    assert provenance["model_target_conduction_velocity_cm_s"] == 80.0
    # No APD target on this card, so no key for it — backend_metadata lands in
    # HDF5 attributes, which have no null, so the alternative would be a
    # sentinel number that reads as a target.
    assert "model_target_apd90_ms" not in provenance
    assert "ap_time_unit_ms" not in provenance


# ---------------------------------------------------------------------------
# The published vector — the point of all of the above
# ---------------------------------------------------------------------------

#: Per-property tolerance on the Wilhelms comparison, as a fraction.
#:
#: **Built from measured sources of variation, not from the observed miss.**
#: Now that the stimulus is pinned at twice the measured capture threshold
#: rather than at an unstated amplitude, there are only three things left that
#: can legitimately move a number here, and all three have been measured:
#:
#: 1. **Where in the pacing train it is read.** Courtemanche never settles, so
#:    "50 beats" is part of the fixture. Measured at 40 / 45 / 50 / 55 / 60
#:    beats, moving the read point by +/- 10 beats moves APD50 by 0.9 %, APD90
#:    by 0.25 %, and amplitude, RMP and ``dV/dt max`` by under 0.1 % each.
#: 2. **The integration step.** Halving it to 0.01 ms moves amplitude 0.7 %,
#:    APD50 1.0 %, APD90 0.6 %, ``dV/dt max`` 0.6 %, RMP 0.2 %.
#: 3. **The stimulus duration, which Wilhelms does not state.** Measured at 1,
#:    2, 5 and 10 ms — each at twice *its own* threshold — amplitude spans
#:    4.5 %, RMP 1.1 %, APD50 1.9 %, APD90 1.9 %, and ``dV/dt max`` spans
#:    **179.6 to 215.1 V/s**, which is -3.8 % to +15.3 % of the published
#:    figure.
#:
#: Summing the three per property and rounding up gives the numbers below. Four
#: of the five tighten substantially — APD50 from 15 % to 6 %, APD90 from 10 %
#: to 5 %.
#:
#: ``dV/dt max`` barely tightens, from 20 % to 18 %, and that is the honest
#: answer rather than a disappointing one: **term 3 dominates it entirely**.
#: The upstroke happens *while the stimulus is still on*, so the measured
#: maximum includes the stimulus's own contribution — raising the amplitude by
#: 1.82 mV/ms raised the measured maximum by 2.13 V/s — and an unstated
#: duration is therefore an unstated fraction of the number. A tolerance under
#: 16 % would be asserting that Wilhelms used our duration, which is not known.
#: The card records the arithmetic that makes the residual interpretable, and
#: records a **propagated** upstroke that no stimulus touches.
WILHELMS_TOLERANCE: dict[str, float] = {
    "amplitude_mv": 0.07,
    "rmp_mv": 0.03,
    "apd50_ms": 0.06,
    "apd90_ms": 0.05,
    "dvdt_max_v_s": 0.18,
}


@pytest.mark.slow
def test_the_shipped_stimulus_is_still_twice_the_measured_threshold() -> None:
    """The protocol constant, re-derived rather than trusted.

    :data:`CRN_STIMULUS_AMPLITUDE_MV_PER_MS` is written as twice
    :data:`CRN_DIASTOLIC_THRESHOLD_MV_PER_MS`, and the threshold is a *measured*
    property of the cell — of its sodium and inward-rectifier conductances, and
    of the stimulus duration. Change any of those and the shipped amplitude
    quietly stops being twice threshold, while every number that depends on it
    carries on looking reasonable. This is the test that notices.

    1 % rather than exact: the search bisects to 0.005 mV/ms and stops there.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import (
        measure_capture_threshold,
    )

    solved = shipped_card().solved
    assert isinstance(solved, CourtemancheCellModel)
    threshold = measure_capture_threshold(solved=solved)

    assert threshold == pytest.approx(CRN_DIASTOLIC_THRESHOLD_MV_PER_MS, rel=0.01), (
        f"the capture threshold measured {threshold:.4f} mV/ms against a recorded "
        f"{CRN_DIASTOLIC_THRESHOLD_MV_PER_MS} — the shipped stimulus is no longer "
        "twice threshold, so the protocol has drifted from the one the published "
        "comparison is made under"
    )
    assert pytest.approx(2.0 * threshold, rel=0.01) == CRN_STIMULUS_AMPLITUDE_MV_PER_MS


@pytest.mark.slow
def test_courtemanche_reproduces_the_published_control_vector() -> None:
    """Wilhelms et al. 2012 Table 1 column C, at the protocol it was read at.

    **The protocol is half the fixture.** Courtemanche never reaches steady
    state: APD90 falls to 83 % of its first beat over 16 minutes of pacing, so
    the same model legitimately reads 295 ms or ~245 ms depending on when you
    look. Wilhelms paces 50 s at BCL 1 s, so this test does too — and pins both
    numbers rather than the cycle length alone, which is the most likely cause
    of a spurious failure in this module.

    **The stimulus is twice the measured capture threshold**, which is the
    protocol Wilhelms states, rather than the round 20 mV/ms an earlier version
    of this test used because it was the textbook 2 nA. Those are nearly the
    same number by luck; only one of them is a protocol.

    This is a claim about agreement with an **independent reimplementation**,
    not about reproducing CRN 1998's own table, which could not be retrieved.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import measure_single_cell

    card = shipped_card()
    assert isinstance(card.solved, CourtemancheCellModel)
    measured = measure_single_cell(
        solved=card.solved, stimulus_mv_per_ms=CRN_STIMULUS_AMPLITUDE_MV_PER_MS
    )

    misses = {
        name: (measured[name], published)
        for name, published in WILHELMS_2012_CRN_CONTROL.items()
        if abs(measured[name] - published) > abs(published) * WILHELMS_TOLERANCE[name]
    }
    assert not misses, (
        f"Wilhelms 2012 control vector missed at BCL {CRN_PACING_BCL_MS} ms / "
        f"{CRN_PACING_BEATS} beats (measured, published): {misses}"
    )


@pytest.mark.slow
def test_the_solved_diffusion_reaches_the_velocity_target() -> None:
    """Solve, simulate, measure — the half of the round-trip Courtemanche has.

    Same tolerance the Aliev-Panfilov round-trip accepts, for the same reason:
    the solve is analytic and discretisation moves the realised velocity by a
    few percent. What is *not* here is an APD assertion, because nothing solved
    for APD; it is measured and recorded on the card instead.

    Measured on the 40 mm patch the reference constant was measured on. A
    smaller patch is cheaper and would be measuring something else — the fit
    window would sit where the wave is still forming.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import (
        measure_conduction_velocity,
    )

    card = shipped_card()
    geometry = Patch2DGeometry(size_mm=40.0, dr_mm=DR_MM)
    measured = measure_conduction_velocity(
        geometry=geometry, solved=card.solved, dr_model_units=DR_MODEL_UNITS
    )

    assert measured == pytest.approx(card.targets.conduction_velocity_cm_s, rel=0.15), (
        f"conduction velocity missed: wanted {card.targets.conduction_velocity_cm_s}, "
        f"measured {measured:.1f}"
    )
    # Printed so the card's `measured:` block can be refreshed from a real run
    # rather than guessed. pytest shows it with -s or on failure.
    print(f"\nmeasured: conduction_velocity_cm_s: {measured:.1f}")


@pytest.mark.slow
def test_a_courtemanche_simulation_runs_end_to_end() -> None:
    """The whole path, in physical units, with nothing converting time.

    The unit tests above check the seam in isolation; this is the one that would
    catch a ``t_max`` or capture stride computed as though a model time unit
    were something other than a millisecond, because both of those produce a
    perfectly well-formed bank of the wrong length.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
    from myocard_synthetic_egm_pipeline.simulate.runner import run_single

    card = shipped_card()
    geometry = Patch2DGeometry(size_mm=8.0, dr_mm=DR_MM)
    rng = np.random.default_rng(0)
    # `dr_model_units` has to be set alongside `geometry.dr_mm`: it defaults to
    # 0.25, which is the Aliev-Panfilov card's pitch, and Courtemanche's space
    # unit IS the millimetre. Leaving the default raises rather than silently
    # simulating a mesh 2.5x coarser than the geometry claims — which is what
    # this line getting it wrong did, the first time the pitch moved.
    config = RunConfig(
        trace_duration_ms=192.0,
        output_fs_hz=1000.0,
        dr_model_units=DR_MM,
        model_card=card,
    )

    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge="left"),
        electrodes=CenteredGrid2D.sample(geometry=geometry, rng=rng),
        cell_model=card.solved,
        backend=FinitewaveBackend(),
        config=config,
        rng=rng,
    )

    assert result.bipolar_traces.shape[1] == 192
    assert np.isfinite(result.bipolar_traces).all()
    # A diverged field is the failure mode a well-formed bank hides, so assert
    # the traces carry signal rather than only that they exist.
    assert float(np.abs(result.bipolar_traces).max()) > 0.0
    assert result.specs.cell_model.type == "courtemanche"
    assert result.run_metadata["backend_metadata"]["model_class"] == "Courtemanche"
    assert result.run_metadata["backend_metadata"]["crn_dt_ms"] == card.solved.dt_model_units


# ---------------------------------------------------------------------------
# The matched AF card, and the omission it declares
# ---------------------------------------------------------------------------

AF_CARD_NAME = "af_remodelled_crn_220ms"


def af_card() -> ModelCard:
    return load_model_card(AF_CARD_NAME, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)


def test_the_severity_scalings_are_three_currents_and_the_arithmetic_is_stated() -> None:
    """``s`` scales three conductances linearly, and ends where it should.

    ``s = 0`` must be exactly control — every multiplier 1.0 — or the severity
    axis does not pass through the card it is supposed to be a remodelling of.
    ``s = 1`` must be the published magnitudes.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import crn_af_scalings

    assert crn_af_scalings(0.0) == {
        "g_to_scale": 1.0,
        "g_CaL_scale": 1.0,
        "g_K1_scale": 1.0,
    }
    full = crn_af_scalings(1.0)
    assert full["g_to_scale"] == pytest.approx(0.35)
    assert full["g_CaL_scale"] == pytest.approx(0.35)
    assert full["g_K1_scale"] == pytest.approx(2.10)

    with pytest.raises(ValueError, match="severity must lie"):
        crn_af_scalings(1.5)


def test_the_af_card_applies_three_of_the_four_published_changes() -> None:
    """**The declared omission, asserted rather than described.**

    The published cAF set is four conductance changes. This card carries three:
    ``I_Kur`` is absent because Finitewave computes its conductance inside the
    ionic kernel from voltage, so no parameter exists to scale.

    Asserting the *absence* is what stops the omission from being quietly
    repaired by someone adding a ``g_Kur_scale`` that the backend would refuse
    anyway — and, more usefully, what stops it from being quietly *compensated*
    by an added ``g_Kr_scale`` or ``g_Ks_scale``. Those are settable, they would
    absorb I_Kur's effect on duration, and doing so would convert a forced,
    declarable deviation into a fabricated one. The card must carry exactly the
    three currents it says it does.
    """
    solved = af_card().solved
    assert isinstance(solved, CourtemancheCellModel)

    assert set(solved.params) == {"g_to_scale", "g_CaL_scale", "g_K1_scale"}
    assert "g_Kur_scale" not in solved.params
    assert not {"g_Kr_scale", "g_Ks_scale"} & set(solved.params), (
        "the card carries a delayed-rectifier scaling, which would mean I_Kur's "
        "omission had been compensated rather than declared"
    )
    # And they are the shipped severity's, not some other set.
    assert dict(solved.params) == pytest.approx(dict(CRN_AF_SCALINGS))


def test_the_af_card_states_an_apd_target_and_the_measurement_backs_it() -> None:
    """Unlike the control card, this one aims at an APD90 — and is checked.

    Courtemanche cannot solve for APD, so the target is a claim about the
    severity sweep. Load-time verification compares it against the card's own
    ``measured`` block, which is the only thing that makes stating it honest.
    """
    card = af_card()

    assert card.targets.apd90_ms == 220.0
    assert card.measured is not None
    assert card.measured.apd90_ms == pytest.approx(220.0, rel=0.02)

    from myocard_synthetic_egm_pipeline.constants import DEFAULT_TRACE_DURATION_MS

    assert card.measured.apd90_ms >= DEFAULT_TRACE_DURATION_MS, (
        "the remodelled APD no longer clears the trace duration, so a "
        "repolarisation deflection is back inside every cropped window"
    )


def test_the_two_courtemanche_cards_are_matched() -> None:
    """The pair is only a *model* comparison if everything else agrees.

    Same conduction-velocity target, same mesh pitch, same cell model — so the
    membrane's remodelling state is the only thing that differs between a bank
    generated under one and a bank generated under the other.
    """
    control, remodelled = shipped_card(), af_card()

    assert control.targets.conduction_velocity_cm_s == remodelled.targets.conduction_velocity_cm_s
    assert control.solved.type == remodelled.solved.type
    # Both were solved at the shipped pitch: loading either at any other one
    # raises, which is what the shared DR_MM in this module is exercising.
    for card in (control, remodelled):
        assert card.measured is not None
        assert card.measured.upstroke_v_s is not None

    # The upstroke is the free variable the comparison reads. Remodelling moves
    # it barely at all, which is what makes the AP-versus-CRN difference
    # attributable to the membrane model rather than to the severity choice.
    assert control.measured is not None and remodelled.measured is not None
    assert remodelled.measured.upstroke_v_s == pytest.approx(
        control.measured.upstroke_v_s, rel=0.05
    )


def test_the_registered_anchors_cover_exactly_the_shipped_cards() -> None:
    """Every registered anchor has a card, and every card has an anchor.

    An anchor with no card is a measurement nobody uses; a card with no anchor
    cannot be solved at all. The registry is small enough that the pairing can
    simply be asserted.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        CRN_REFERENCES,
        find_courtemanche_reference,
    )

    assert {r.name for r in CRN_REFERENCES} == {"control", "af_remodelled"}
    for reference in CRN_REFERENCES:
        assert reference.dr_mm == CRN_CALIBRATION_DR_MM
        assert reference.reference_cv_cm_s > 0

    assert find_courtemanche_reference({}, CRN_CALIBRATION_DR_MM) is not None
    assert find_courtemanche_reference(CRN_AF_SCALINGS, CRN_CALIBRATION_DR_MM) is not None
    # Same conductances, wrong mesh.
    assert find_courtemanche_reference({}, 0.25) is None


def test_the_shipped_severity_is_the_one_the_card_records() -> None:
    """The constant and the card cannot drift apart.

    ``CRN_AF_SEVERITY`` is what the sweep landed on; the card's ``params`` are
    what a run actually uses. If someone edits one, this says so.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import crn_af_scalings

    solved = af_card().solved
    assert isinstance(solved, CourtemancheCellModel)
    assert dict(solved.params) == pytest.approx(crn_af_scalings(CRN_AF_SEVERITY))


# ---------------------------------------------------------------------------
# The cable rig the sweeps ran on
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_cable_reproduces_the_patch_on_the_quantities_it_is_used_for() -> None:
    """The premise of doing the convergence sweep on a cable at all.

    A plane wave in a sheet has no transverse gradient, so a cell in a 1-D
    cable sees the same axial load as one in the patch — which is why the
    propagated upstroke and the APD can be measured on the cable at a
    hundredth of the cost. Conduction velocity is the exception and is
    deliberately not asserted here: the two fit windows sit at different
    distances from the stimulus, so they differ by ~0.5 %, and that is why the
    cards measure CV on the patch instead.

    Run at the coarse pitch on purpose. It is where the comparison was first
    made, and a 40 mm patch at the shipped 0.10 mm would take twenty minutes
    to say the same thing.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import (
        measure,
        measure_strip,
    )

    diffusion, dr_mm = 0.299142, 0.25
    solved = CourtemancheCellModel(
        diffusion=diffusion, dt_model_units=strip_step_ms(diffusion=diffusion, dr_mm=dr_mm)
    )
    patch = measure(
        geometry=Patch2DGeometry(size_mm=40.0, dr_mm=dr_mm),
        solved=solved,
        dr_model_units=dr_mm,
    )
    cable = measure_strip(diffusion=diffusion, dr_mm=dr_mm, include_apd=True)

    assert patch.upstroke_v_s is not None and cable.apd90_ms is not None
    assert cable.upstroke_v_s == pytest.approx(patch.upstroke_v_s, rel=0.01), (
        "the cable and the patch disagree on the propagated upstroke, so the "
        "convergence sweep is not measuring what the cards record"
    )
    assert cable.apd90_ms == pytest.approx(patch.apd90_ms, rel=0.01)


@pytest.mark.slow
def test_conduction_velocity_is_mesh_dependent_at_a_fixed_diffusion() -> None:
    """The reproducibility hazard, demonstrated rather than asserted.

    The solve absorbs discretisation error into ``diffusion``, so ``diffusion``
    is not a physical tissue property: hold it fixed, change only the mesh, and
    the tissue conducts at a different speed. Anyone re-running one of our
    banks at a different pitch gets a different conduction velocity out of the
    same card, which is precisely why ``calibrate_courtemanche`` refuses a
    pitch it has no measured anchor for.

    Asserting the gap is **large** rather than small: a test that allowed it to
    shrink to nothing would pass on a bug that made the mesh inoperative.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import measure_strip

    diffusion = 0.299142
    coarse = measure_strip(diffusion=diffusion, dr_mm=0.25)
    fine = measure_strip(diffusion=diffusion, dr_mm=0.05)

    ratio = fine.conduction_velocity_cm_s / coarse.conduction_velocity_cm_s
    assert ratio > 1.05, (
        f"refining the mesh 5x moved conduction velocity by only "
        f"{100 * (ratio - 1):.1f} % at fixed diffusion; either the mesh stopped "
        "reaching the solver or the model became resolution-independent"
    )


@pytest.mark.slow
def test_the_af_card_reaches_its_apd_target_in_tissue() -> None:
    """Solve, simulate, measure — the AF card's half of the round trip.

    The severity was swept until this landed on 220 ms, so the assertion is
    that the sweep's answer survives being re-derived from the card rather than
    from the sweep's own working.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import measure_strip

    solved = af_card().solved
    assert isinstance(solved, CourtemancheCellModel)
    measured = measure_strip(
        diffusion=solved.diffusion,
        dr_mm=DR_MM,
        params=solved.params,
        include_apd=True,
    )

    assert measured.apd90_ms == pytest.approx(220.0, rel=0.02), (
        f"the shipped severity gives APD90 {measured.apd90_ms:.1f} ms, not 220"
    )


# ---------------------------------------------------------------------------
# A borrowed anchor warns, proceeds, and lands in the artifact
# ---------------------------------------------------------------------------


def test_an_unregistered_combination_borrows_the_nearest_anchor() -> None:
    """It warns and proceeds rather than refusing, and the reason is structural.

    Refusing made ``backend.model`` unreadable without ``geometry.dr_mm``: two
    config sections that were independently reasonable-about stopped being so,
    and every future cross-section rule would accumulate in whichever loader
    noticed first. Cross-section validation gets its own tool later.

    What must not happen is the borrowing being *quiet* — see the provenance
    test below, which is the half that actually protects a reader.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        AnchorSubstitutionWarning,
    )

    with pytest.warns(AnchorSubstitutionWarning, match="0.25"):
        solved = calibrate_courtemanche(
            conduction_velocity_cm_s=80.0, dr_mm=0.25, dr_model_units=0.25
        )

    # Proceeded, and on the control anchor: same conductances, nearest pitch.
    exact = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    assert solved.diffusion == pytest.approx(exact.diffusion), (
        "the borrowed anchor produced a different diffusion from the registered "
        "one, so it was not the anchor that got substituted"
    )


def test_the_nearest_anchor_prefers_the_matching_membrane() -> None:
    """Two tiers, because the two axes are not comparable.

    A different *pitch* rescales an error the convergence sweep has measured; a
    different *membrane* moves conduction velocity by an amount nobody has. So
    an unregistered pitch with known conductances borrows from its own
    membrane, and only a wholly unknown conductance set falls back across the
    registry — where the warning says so.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        resolve_courtemanche_anchor,
    )

    remodelled = resolve_courtemanche_anchor(CRN_AF_SCALINGS, 0.25)
    assert remodelled.substituted
    assert not remodelled.conductances_differ
    assert remodelled.reference.name == "af_remodelled"

    stranger = resolve_courtemanche_anchor({"g_to_scale": 0.35}, DR_MM)
    assert stranger.substituted
    assert stranger.conductances_differ, (
        "an unknown conductance set borrowed an anchor without flagging that the "
        "membrane differs, which is the worse half of the substitution"
    )


def test_a_borrowed_anchor_is_written_into_the_bank() -> None:
    """**The half that matters.** A console warning is gone by the time anyone
    reads the artifact.

    If the substitution is not in the bank, the run is silently wrong to every
    later reader — which is the failure the refusal existed to prevent, moved
    rather than removed. So the card carries which anchor it was verified
    through, and ``card_provenance`` writes it where the backend metadata goes.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        AnchorSubstitutionWarning,
    )

    with pytest.warns(AnchorSubstitutionWarning):
        borrowed = load_model_card(CARD_NAME, dr_mm=0.25, dr_model_units=0.25)

    provenance = card_provenance(borrowed)
    assert provenance["model_anchor_substituted"] is True
    assert provenance["model_anchor_name"] == "control"
    assert provenance["model_anchor_dr_mm"] == pytest.approx(DR_MM)
    assert provenance["model_anchor_run_dr_mm"] == pytest.approx(0.25)
    assert provenance["model_anchor_conductances_differ"] is False


def test_the_anchor_is_recorded_even_when_it_was_the_right_one() -> None:
    """Recorded always, so absence of the key means an old writer and nothing else.

    Writing only the exception would leave a reader unable to tell "this was
    verified against the anchor measured on this very mesh" from "this bank
    predates the field".
    """
    provenance = card_provenance(shipped_card())

    assert provenance["model_anchor_name"] == "control"
    assert provenance["model_anchor_dr_mm"] == pytest.approx(DR_MM)
    assert "model_anchor_substituted" not in provenance


def test_an_aliev_panfilov_card_records_no_anchor() -> None:
    """It has none. Its solve runs off measured constants, not a velocity anchor."""
    card = load_model_card("af_remodelled_220ms", dr_mm=AP_DR_MM, dr_model_units=AP_DR_MM)

    assert card.anchor is None
    assert not any("anchor" in key for key in card_provenance(card))


def test_an_empty_anchor_registry_is_still_a_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one genuine impossibility: nothing to substitute.

    Everything else degrades to a warning, so this is the only place left that
    raises — and it has to, because there is no answer to give rather than a
    worse answer to give.
    """
    from myocard_synthetic_egm_pipeline.simulate import cell_models

    monkeypatch.setattr(cell_models, "CRN_REFERENCES", ())
    with pytest.raises(ValueError, match="nothing to solve through"):
        cell_models.resolve_courtemanche_anchor({}, DR_MM)


def test_the_shipped_courtemanche_example_config_agrees_with_the_shipped_card() -> None:
    """The example is the only thing that makes either card reachable.

    A card nothing points at is a card nobody runs, and the coupling it needs —
    ``geometry.dr_mm`` **and** ``run.dr_model_units`` both moved off their
    Aliev-Panfilov defaults — is exactly the kind a reader gets wrong once.

    Asserting **no substitution warning** is the load-bearing part: it says the
    example's pitch and the card's anchor agree. If either moves without the
    other, this fails here rather than in a bank whose upstroke is 10 % out and
    whose conduction velocity looks perfect.
    """
    import yaml

    from myocard_synthetic_egm_pipeline.cli._config import build_generate_dataset_config
    from myocard_synthetic_egm_pipeline.simulate.cell_models import (
        AnchorSubstitutionWarning,
    )

    path = Path("examples/synthegm_courtemanche.yaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["_config_dir"] = path.parent

    with warnings.catch_warnings():
        warnings.simplefilter("error", AnchorSubstitutionWarning)
        config = build_generate_dataset_config(doc)

    assert isinstance(config.cell_model, CourtemancheCellModel)
    assert isinstance(config.geometry, Patch2DGeometry)
    assert config.geometry.dr_mm == pytest.approx(DR_MM)
    assert config.run_config.dr_model_units == pytest.approx(DR_MM), (
        "the example leaves run.dr_model_units at its Aliev-Panfilov default, "
        "which the backend refuses for a dimensional model"
    )
    assert config.run_config.model_card is not None
    assert config.run_config.model_card.anchor is not None
    assert not config.run_config.model_card.anchor.substituted


# ---------------------------------------------------------------------------
# Anisotropy, measured on the ionic model rather than argued for
# ---------------------------------------------------------------------------

#: Patch extent for the anisotropy measurement, in mm.
#:
#: **Not the 40 mm generation geometry, and the ratio is why.** A velocity
#: *magnitude* is extent-dependent — the fit window sits closer to the stimulus
#: on a small patch, where the wave is still accelerating, and the along-fibre
#: figure duly reads 83.5 cm/s at 6 mm, 82.2 at 12 and 81.2 at 40. A *ratio* is
#: not, because both runs are displaced by the same factor: measured 2.0467 at
#: 6 mm, 2.0500 at 12 mm and 2.0501 at 20 mm — the last two agree to four
#: figures, so the sequence has converged rather than merely been sampled
#: twice. So extent is free to choose here
#: in a way it is not for :func:`measure_conduction_velocity`, and it is chosen
#: on two grounds: 12 mm is what the Aliev-Panfilov anisotropy test uses, so
#: the two models' realized ratios are read over the same tissue and the same
#: fit window in millimetres with only the pitch and the membrane differing;
#: and at 0.1 mm it keeps the fit to 48 nodes starting 3.6 mm from the
#: stimulus, for two runs in well under a minute.
ANISOTROPY_PATCH_MM = 12.0


@pytest.mark.slow
def test_the_ionic_model_realizes_the_requested_anisotropy() -> None:
    """The knob that was inoperative for the life of the project, on Courtemanche.

    **Measured from the activation map, not read back off the stencil.** The
    tensor is written by one model-agnostic helper, so "Courtemanche inherits
    it" is true of the code — and that is the same species of claim as "the
    helper sets ``D_al``", which was also true of the code for two years while
    the helper assigned to the model instead of the stencil and every bank came
    out at the built-in 3.093 whatever was requested. Only a velocity measured
    off a wave distinguishes the two.

    **2.0 is a discriminating request, and it has to be.** The stencil's own
    default is ``D_al = 1, D_ac = 1/9``, i.e. a realized ratio near 3 — so a
    test run at ``anisotropy_ratio = 3.0`` would pass unchanged on a helper
    that did nothing at all. At 2.0 the historical no-op reads 3.09 and fails
    by 55 %.

    Expected slightly **above** the requested value, which is the transverse
    run being the under-resolved one: ``D_across`` is ``D/4``, so the
    transverse space constant is half the along-fibre one and 0.1 mm resolves
    its upstroke half as well. Under-resolution depresses conduction velocity
    (asserted from the other side by the mesh-dependence test above), and that
    velocity is the denominator. Aliev-Panfilov shows the same sign: 2.15 for a
    requested 2.0 at its own coarser pitch.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import (
        measure_anisotropy_ratio,
    )

    card = shipped_card()
    geometry = Patch2DGeometry(size_mm=ANISOTROPY_PATCH_MM, dr_mm=DR_MM)

    realized = measure_anisotropy_ratio(
        geometry=geometry, solved=card.solved, dr_model_units=DR_MODEL_UNITS
    )

    # 5 %, against a discretisation residual measured at 2.5 % — twice the
    # observed excess, and far tighter than every way the tensor can be wrong:
    # the pre-fix stencil default gives 3.09 (+55 %), an isotropic tensor 1.0
    # (-50 %), and a ratio applied without the square gives sqrt(2) = 1.41
    # (-29 %). Nothing legitimate lives between 5 % and those.
    assert realized == pytest.approx(geometry.anisotropy_ratio, rel=0.05), (
        f"requested anisotropy {geometry.anisotropy_ratio}, realized {realized:.4f} "
        "on the Courtemanche membrane"
    )
