"""Courtemanche: the seam, the partial solve, the card, and the published vector.

The fast tests here are arithmetic, dispatch and file handling. The **slow**
ones are the point of the step: Courtemanche arrives with no known-good prior
fixture of our own — S17 was meant to leave a clean numerical baseline and was
absorbed into S38c, which moved every number on purpose — so it is validated
against **Wilhelms et al. 2012**'s published five-element vector rather than
against a previous run of ours. That is weaker than a self-comparison, and it is
exactly why the five values are asserted rather than eyeballed.
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
    CRN_MAX_DT_MS,
    CRN_PACING_BCL_MS,
    CRN_PACING_BEATS,
    CRN_REFERENCE_CV_CM_S,
    CRN_REFERENCE_DIFFUSION,
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
    # would make every test here pass while restoring the very knob D2 says a
    # dimensional model must not carry.
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
    step does not diverge — it flattens the upstroke, which is the observable
    SEP5 exists to compare. A limit reporting only the CFL bound would report
    the slack constraint.
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
    """Both halves of CL-180's trap 3, refused by name rather than absorbed.

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
    diffusion. S18c measures the remodelled reference and registers it.
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


def test_the_measured_courtemanche_apd_clears_the_trace_duration() -> None:
    """``APD >= T`` is the unconditional no-shortcut rule (CL-176/178).

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

    CV is the target both solves share; upstroke morphology is what SEP5 is
    actually isolating, and it only reads as a model difference if everything
    solvable is matched.
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
    """The regression guard for a step that touched every card path.

    Parsing gained a dispatch table, ``targets.apd90_ms`` became optional and
    verification gained a branch. None of that may move the Aliev-Panfilov
    card's numbers — a bank generated before and after this step must be
    identical.
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
    claiming its *chosen parameters* reach it — the claim S18c's severity sweep
    will make. The check is two recorded numbers against each other, so it stays
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
    so the cAF -49 % I_Kur scaling S18c needs cannot be set by assignment at
    all. Discovering that here costs a message; discovering it from a severity
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
#: Stated per quantity rather than as one number because the quantities are not
#: equally determined by the model. RMP and APD90 are membrane properties read
#: off a settled trace and should land close. ``dV/dt max`` is the loosest, and
#: honestly so: it is measured *inside* the 2 ms stimulus window, so it carries
#: the stimulus amplitude with it — 165 to 227 V/s across a plausible range of
#: stimulus strengths — and Wilhelms does not state the amplitude used. 20 % on
#: that quantity is what agreement with an independent reimplementation can
#: mean when the protocol is only partly specified; a tighter figure would be a
#: claim we cannot support.
WILHELMS_TOLERANCE: dict[str, float] = {
    "amplitude_mv": 0.10,
    "rmp_mv": 0.03,
    "apd50_ms": 0.15,
    "apd90_ms": 0.10,
    "dvdt_max_v_s": 0.20,
}


@pytest.mark.slow
def test_courtemanche_reproduces_the_published_control_vector() -> None:
    """Wilhelms et al. 2012 Table 1 column C, at the protocol it was read at.

    **The protocol is half the fixture.** Courtemanche never reaches steady
    state: APD90 falls to 83 % of its first beat over 16 minutes of pacing, so
    the same model legitimately reads 295 ms or ~245 ms depending on when you
    look. Wilhelms paces 50 s at BCL 1 s, so this test does too — and pins both
    numbers rather than the cycle length alone, which is CL-180's most likely
    cause of a spurious failure here.

    This is a claim about agreement with an **independent reimplementation**,
    not about reproducing CRN 1998's own table, which could not be retrieved.
    """
    from myocard_synthetic_egm_pipeline.backends.finitewave.measure import measure_single_cell

    card = shipped_card()
    assert isinstance(card.solved, CourtemancheCellModel)
    measured = measure_single_cell(solved=card.solved)

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
