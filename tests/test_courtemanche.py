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
    """The ionic ceiling, not the CFL condition, is what binds here.

    Aliev-Panfilov's limit is purely the explicit-diffusion bound. Courtemanche's
    fast sodium current needs a much smaller step than that bound permits, and
    because Finitewave integrates the gating variables Rush-Larsen the coarse
    step does not diverge — it flattens the upstroke, which is the one
    observable an ionic model was added to get right. A limit reporting only
    the CFL bound would report the slack constraint.
    """
    solved = calibrate_courtemanche(
        conduction_velocity_cm_s=80.0, dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
    cfl = DR_MODEL_UNITS**2 / (4.0 * solved.diffusion)

    assert solved.dt_model_units <= solved.stability_limit(dr_model_units=DR_MODEL_UNITS)
    assert solved.dt_model_units == CRN_MAX_DT_MS
    assert cfl > CRN_MAX_DT_MS, (
        "the CFL bound has become the binding constraint, so CRN_MAX_DT_MS is no "
        "longer doing anything — re-check which bound the step is respecting."
    )


@pytest.mark.parametrize(
    ("dr_mm", "dr_model_units", "match"),
    [
        (0.25, 0.5, "space unit is the millimetre"),
        (0.1, 0.1, "measured at dr=0.25"),
    ],
)
def test_a_mesh_the_constant_was_not_measured_on_is_refused(
    dr_mm: float, dr_model_units: float, match: str
) -> None:
    """Both halves of the under-resolution trap, refused rather than absorbed.

    A CV-solve will happily absorb discretisation error into ``diffusion`` and
    hit its target anyway, leaving a physical-looking number that is not — the
    same shape as "CV ran 8 % high at D ~ 10". Courtemanche is where that bites:
    its upstroke is ~0.59 ms, about 1.9 cells wide at 0.25 mm, where monodomain
    practice wants 5-10. The measured constant is therefore evidence about one
    pitch, and using it at another is reading a number off the wrong axis.
    """
    with pytest.raises(ValueError, match=match):
        calibrate_courtemanche(
            conduction_velocity_cm_s=80.0, dr_mm=dr_mm, dr_model_units=dr_model_units
        )


def test_remodelled_conductances_are_refused_until_their_velocity_is_measured() -> None:
    """The reference CV was measured at control conductances.

    Sodium conductance moves conduction velocity directly, so solving a
    remodelled set through the control constant puts the whole error into
    diffusion. The remodelled reference has to be measured and registered
    before any card can be solved through it.
    """
    with pytest.raises(ValueError, match="control conductances"):
        calibrate_courtemanche(
            conduction_velocity_cm_s=80.0,
            dr_mm=DR_MM,
            dr_model_units=DR_MODEL_UNITS,
            params={"g_to_scale": 0.35},
        )


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
    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)

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
    aliev_panfilov = load_model_card(
        "af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS
    )
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
    card = load_model_card("af_remodelled_220ms", dr_mm=DR_MM, dr_model_units=DR_MODEL_UNITS)

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
            dt_model_units=0.02,
            params={"g_to_scale": 0.35, "g_K1_scale": 2.1},
        ),
        geometry=Patch2DGeometry(size_mm=8.0, dr_mm=DR_MM),
        dr_model_units=DR_MODEL_UNITS,
    )

    assert native.model.gto == pytest.approx(default_gto * 0.35)
    assert native.model.gk1 == pytest.approx(fw.Courtemanche2D().gk1 * 2.1)


def test_a_scaling_the_solver_cannot_apply_is_refused_by_name() -> None:
    """The alternative is a dead attribute and a sweep that does nothing.

    ``I_Kur`` is the live case, not a hypothetical: finitewave 0.9.3 computes its
    conductance from voltage inside the kernel rather than reading a parameter,
    so the cAF -49 % I_Kur scaling an AF-remodelled card needs cannot be set
    by assignment at all. Discovering that here costs a message; discovering it
    from a severity
    sweep that silently moved three of four currents costs a day.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as be

    with pytest.raises(ValueError, match="g_Kur_scale"):
        be._build_model_2d(
            cell_model=CourtemancheCellModel(
                diffusion=CRN_REFERENCE_DIFFUSION,
                dt_model_units=0.02,
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
    config = RunConfig(trace_duration_ms=192.0, output_fs_hz=1000.0, model_card=card)

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
