"""The parameter-path resolver: what it reaches, and what it refuses.

The round-trip test **enumerates its own cases from the spec objects**. A
hand-written list of paths would be a second description of the same thing, and
the two drift the moment a spec gains a field — which is exactly when the new
field is least likely to have been thought about. Reading the dataclasses means
a field that appears gains coverage in the same commit that adds it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from myocard_synthetic_egm_pipeline.mixer import MixerConfig
from myocard_synthetic_egm_pipeline.simulate import (
    CenteredGrid2D,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel,
    CourtemancheCellModel,
)
from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card
from myocard_synthetic_egm_pipeline.simulate.result import SimulationSpecs
from myocard_synthetic_egm_pipeline.simulate.tuning import (
    DerivedParameterError,
    InfeasibleTargetError,
    NumericalParameterError,
    SimVaryingParameterError,
    TunableRun,
    UnknownParameterError,
    enumerate_paths,
    get_value,
    set_value,
)


def _specs_of(run: TunableRun) -> SimulationSpecs:
    """The fixture always builds specs; narrow once rather than at every use."""
    assert run.specs is not None
    return run.specs


def _diffusion(spec: object) -> float:
    """Read a solved diffusion off either concrete cell model.

    The Protocol does not declare it — deliberately, since a third model need
    not have one — so the tests narrow rather than widen the contract.
    """
    assert isinstance(spec, AlievPanfilovCellModel | CourtemancheCellModel)
    return float(spec.diffusion)


AP_CARD = ("af_remodelled_220ms", 0.25)
CRN_AF_CARD = ("af_remodelled_crn_220ms", 0.10)


def specs_for(card_name: str, dr_mm: float, *, mixer: bool = True) -> TunableRun:
    geometry = Patch2DGeometry(dr_mm=dr_mm)
    card = load_model_card(card_name, dr_mm=dr_mm, dr_model_units=dr_mm)
    return TunableRun(
        specs=SimulationSpecs(
            geometry=geometry,
            substrate=UniformRandomFibrosis(density=0.2),
            activation=PlanarEdgeStimulus(edge="left"),
            electrodes=CenteredGrid2D.sample(geometry=geometry, rng=np.random.default_rng(0)),
            cell_model=card.solved,
        ),
        card=card,
        mixer=MixerConfig() if mixer else None,
        dr_model_units=dr_mm,
    )


@pytest.fixture(params=[AP_CARD, CRN_AF_CARD], ids=["aliev_panfilov", "courtemanche_af"])
def specs(request: pytest.FixtureRequest) -> TunableRun:
    """Both cell models, because their chosen/derived splits differ."""
    return specs_for(*request.param)


# ---------------------------------------------------------------------------
# Round-trip, over every path the objects offer
# ---------------------------------------------------------------------------


def test_every_reachable_path_round_trips(specs: TunableRun) -> None:
    """Read a value, write it back, read it again — for all of them.

    Writing the value it already had is the strongest form: any path that
    survives this is one a sweep can address without the resolver quietly
    substituting something else.
    """
    paths = enumerate_paths(specs)
    assert paths, "no paths were enumerated, so this test asserts nothing"

    for path in paths:
        original = get_value(specs, path)
        updated = set_value(specs, path, original)
        assert get_value(updated, path) == original, path


def test_every_reachable_path_accepts_a_changed_value(specs: TunableRun) -> None:
    """And the new value comes back, which round-tripping alone would not show.

    Setting a field to what it already held passes even against a resolver whose
    write does nothing at all. This one perturbs each value first, so a write
    that silently no-ops fails here.
    """
    for path in enumerate_paths(specs):
        original = get_value(specs, path)
        if path.startswith("cell_model.params.") or path == "cell_model.eps":
            # Both re-solve, and both have dedicated tests. A conductance write
            # warns about a borrowed anchor; `eps` is an input the calibration
            # accepts and then pins to the published value, so perturbing it is
            # refused on purpose. Neither belongs in a blanket round-trip.
            continue
        if path == "activation.edge":
            changed: object = "top" if original == "left" else "left"
        elif isinstance(original, bool):
            changed = not original
        elif isinstance(original, int):
            # n_cols and n_rows have floors, and strip_thickness must stay
            # inside the mesh; growing by one is safe for all of them.
            changed = original + 1
        else:
            changed = float(original) * 0.5 + 0.125

        updated = set_value(specs, path, changed)
        assert get_value(updated, path) == changed, path
        assert get_value(specs, path) == original, f"{path} mutated the original specs"


def test_the_contract_example_paths_all_resolve() -> None:
    """The three paths the parameter contract itself gives as examples.

    ``cell_model.params.g_CaL_scale`` resolves against a card that declares that
    scaling. The control card's ``params`` is empty — meaning every conductance
    at its published value — and a sweep varies what a card declares rather than
    introducing a key, so the example is checked against the AF card.
    """
    specs = specs_for(*CRN_AF_CARD)

    assert get_value(specs, "substrate.density") == pytest.approx(0.2)
    assert get_value(specs, "cell_model.params.g_CaL_scale") == pytest.approx(0.636)
    assert get_value(specs, "electrodes.height_mm") > 0


# ---------------------------------------------------------------------------
# A categorical is not a float
# ---------------------------------------------------------------------------


def test_a_categorical_path_round_trips(specs: TunableRun) -> None:
    """A non-numeric path gets and sets without anything assuming float.

    ``mix.bandpass_clean`` rather than ``activation.edge``: the edge is now
    read-only, because the per-simulation sampler redraws it. Both are
    categorical, and this is the one a sweep can actually write.
    """
    for flag in (False, True):
        updated = set_value(specs, "mix.bandpass_clean", flag)
        assert get_value(updated, "mix.bandpass_clean") is flag


def test_a_categorical_read_returns_every_value_it_was_built_with() -> None:
    """The edge is still readable, and reads are what a bank records.

    Built four ways rather than written four times, since writing it is refused.
    """
    for edge in ("top", "bottom", "left", "right"):
        run = specs_for(*AP_CARD)
        run = replace(run, specs=replace(_specs_of(run), activation=PlanarEdgeStimulus(edge=edge)))
        assert get_value(run, "activation.edge") == edge


def test_an_invalid_value_is_refused_by_the_spec_that_owns_the_rule(
    specs: TunableRun,
) -> None:
    """The resolver does not re-validate; it lets the spec speak.

    Duplicating a rule here would give it two homes, and the copy would be the
    one that went stale. Checked on ``geometry``, which is writable — the edge's
    own validation is now unreachable through a write.
    """
    with pytest.raises(ValueError, match="anisotropy_ratio must be"):
        set_value(specs, "geometry.anisotropy_ratio", 0.5)


# ---------------------------------------------------------------------------
# Derived fields are refused, and the message says what to do instead
# ---------------------------------------------------------------------------


def test_a_solved_membrane_field_is_refused_and_names_the_alternative(
    specs: TunableRun,
) -> None:
    """The trap this resolver exists to not fall into.

    A card re-verifies its solved values on every load, so a bank generated with
    ``diffusion`` overridden would claim one parameterisation and contain
    another. The refusal has to name the alternative, or the next person simply
    reaches for the field again.
    """
    with pytest.raises(DerivedParameterError) as excinfo:
        set_value(specs, "cell_model.diffusion", 0.3)

    message = str(excinfo.value)
    assert "cell_model.diffusion" in message
    assert "conduction_velocity_cm_s" in message, "the alternative is not named"


def test_a_solved_field_can_still_be_read(specs: TunableRun) -> None:
    """Only writing is refused. Reading one is how a run records what it solved."""
    assert get_value(specs, "cell_model.diffusion") > 0


def test_the_chosen_and_derived_split_is_what_we_think_it_is() -> None:
    """Pins the split, so a signature change cannot flip a field in silence.

    The mechanism reads the calibration signature, which is what keeps it from
    going stale. This is the other half: the mechanism is automatic, and this
    records what it currently concludes, so a rename that turned a derived field
    into a sweepable one fails here rather than in a bank six weeks later.
    """
    ap = enumerate_paths(specs_for(*AP_CARD))
    crn = enumerate_paths(specs_for(*CRN_AF_CARD))

    # eps is an *input* to calibrate_aliev_panfilov, so it is chosen.
    assert "cell_model.eps" in ap
    # Everything the solve produces is refused.
    for solved in ("cell_model.diffusion", "cell_model.dt_model_units"):
        assert solved not in ap, solved
        assert solved not in crn, solved
    assert "cell_model.time_unit_ms" not in ap
    # Courtemanche has no eps and no time unit; its chosen knobs are the
    # conductance scalings, which are inputs to its calibration.
    assert "cell_model.eps" not in crn
    assert "cell_model.params.g_CaL_scale" in crn


# ---------------------------------------------------------------------------
# Fields nothing reads: the write that would have silently done nothing
# ---------------------------------------------------------------------------


def test_the_electrode_rebuild_still_happens_on_a_write_that_is_allowed(
    specs: TunableRun,
) -> None:
    """The coupling that made the rebuild necessary, exercised where writes land.

    ``CenteredGrid2D`` records ``height_mm`` and computes ``positions_mm``, and
    the backend reads **only** ``positions_mm``. Writing the scalar and leaving
    the array stale would give every simulation identical electrode geometry
    while the bank recorded different heights.

    Direct electrode writes are now refused for a different reason (they do not
    survive the per-simulation sampler), so the rebuild is reached through
    ``geometry``, which does survive. Asserting on the array rather than the
    scalar is the point either way.
    """
    before = _specs_of(specs).electrodes.positions_mm.copy()
    updated = set_value(specs, "geometry.size_mm", 60.0)

    assert isinstance(_specs_of(updated).electrodes, CenteredGrid2D)
    assert not np.allclose(before[:, :2], _specs_of(updated).electrodes.positions_mm[:, :2])


def test_growing_the_patch_recentres_the_electrodes(specs: TunableRun) -> None:
    """The same coupling from the other side, and why normalisation is unconditional.

    Electrode positions are centred on the patch, so ``geometry.size_mm`` moves
    them as surely as the grid's own layout does. A rule that rebuilt only after
    writes to ``electrodes.*`` would miss this one.
    """
    before = _specs_of(specs).electrodes.positions_mm.copy()
    updated = set_value(specs, "geometry.size_mm", 60.0)

    assert not np.allclose(before[:, :2], _specs_of(updated).electrodes.positions_mm[:, :2]), (
        "the grid was not recentred, so it now sits off-centre in the larger patch"
    )


def test_a_computed_field_is_refused_and_names_the_layout(specs: TunableRun) -> None:
    with pytest.raises(DerivedParameterError, match="computed from the other fields"):
        set_value(specs, "electrodes.positions_mm", np.zeros((25, 3)))


def test_computed_fields_are_not_enumerated(specs: TunableRun) -> None:
    """They are not sweepable, so they must not appear as sweepable paths."""
    paths = enumerate_paths(specs)
    assert "electrodes.positions_mm" not in paths
    assert "electrodes.bipolar_pairs" not in paths


# ---------------------------------------------------------------------------
# Unknown paths
# ---------------------------------------------------------------------------


def test_an_unknown_root_errors_and_names_the_path(specs: TunableRun) -> None:
    with pytest.raises(UnknownParameterError) as excinfo:
        get_value(specs, "mixer.snr_db")
    assert "mixer.snr_db" in str(excinfo.value)
    assert "geometry" in str(excinfo.value), "the valid roots are not listed"


def test_an_unknown_field_errors_and_suggests_what_is_available(
    specs: TunableRun,
) -> None:
    with pytest.raises(UnknownParameterError) as excinfo:
        get_value(specs, "substrate.densty")
    message = str(excinfo.value)
    assert "substrate.densty" in message
    assert "substrate.density" in message, "the near-miss is not offered"
    # Suggested from the fields the object has, not from what a sweep may
    # write — otherwise a read-only root would answer "no such field" with an
    # empty list and imply the correct spelling did not exist either.
    assert "read-only" in message


def test_an_unknown_mapping_key_errors_and_lists_the_present_ones() -> None:
    specs = specs_for(*CRN_AF_CARD)
    with pytest.raises(UnknownParameterError) as excinfo:
        set_value(specs, "cell_model.params.g_Kur_scale", 0.51)
    message = str(excinfo.value)
    assert "g_Kur_scale" in message
    assert "g_CaL_scale" in message


def test_the_discriminator_is_not_a_parameter(specs: TunableRun) -> None:
    """Writing ``type`` would relabel a spec without changing its class."""
    with pytest.raises(DerivedParameterError, match="discriminator"):
        set_value(specs, "geometry.type", "patch_3d")
    assert "geometry.type" not in enumerate_paths(specs)


def test_a_path_that_stops_at_a_root_says_so(specs: TunableRun) -> None:
    with pytest.raises(UnknownParameterError, match="names the 'substrate' spec itself"):
        get_value(specs, "substrate")


def test_a_path_deeper_than_the_grammar_is_refused(specs: TunableRun) -> None:
    with pytest.raises(UnknownParameterError, match="deeper than this grammar goes"):
        get_value(specs, "geometry.size_mm.units")


# ---------------------------------------------------------------------------
# The targets root — a write re-runs the calibration
# ---------------------------------------------------------------------------


def test_a_target_write_re_solves_the_card(specs: TunableRun) -> None:
    """The point of the root: each sweep value gets a correctly solved card.

    Writing a target and leaving the old solved values beside it would produce
    exactly the card the verification exists to reject. So the write re-runs the
    calibration, and both halves have to be true afterwards — the recorded
    target is what was asked for, *and* the solved values moved.
    """
    assert specs.card is not None
    before = _diffusion(specs.card.solved)
    target = get_value(specs, "cell_model.targets.conduction_velocity_cm_s")

    updated = set_value(specs, "cell_model.targets.conduction_velocity_cm_s", target * 1.25)

    assert updated.card is not None
    assert get_value(updated, "cell_model.targets.conduction_velocity_cm_s") == pytest.approx(
        target * 1.25
    )
    assert _diffusion(updated.card.solved) != pytest.approx(before), (
        "the target changed but the solve did not, so the card now records a "
        "target nothing was solved from"
    )
    # CV goes as sqrt(D), so a 1.25x target is a 1.5625x diffusion.
    assert _diffusion(updated.card.solved) == pytest.approx(before * 1.25**2, rel=1e-6)


def test_the_re_solved_card_also_replaces_the_spec_the_run_would_use(
    specs: TunableRun,
) -> None:
    """The card and the specs must not disagree about what is being integrated.

    The backend reads ``specs.cell_model``, not the card. A re-solve that
    updated only the card would leave the run integrating the old membrane while
    the bank's provenance claimed the new one.
    """
    updated = set_value(specs, "cell_model.targets.conduction_velocity_cm_s", 95.0)
    assert updated.card is not None
    assert _specs_of(updated).cell_model == updated.card.solved


def test_a_re_solve_clears_the_stale_measurement(specs: TunableRun) -> None:
    """``measured`` described the previous solve, so it cannot be carried over.

    Keeping it would attach a real simulation's numbers to a parameterisation
    that never produced them.
    """
    assert specs.card is not None and specs.card.measured is not None
    updated = set_value(specs, "cell_model.targets.conduction_velocity_cm_s", 95.0)
    assert updated.card is not None
    assert updated.card.measured is None


def test_an_infeasible_target_raises_and_is_not_clamped(specs: TunableRun) -> None:
    """The failure that makes the feasible boundary visible.

    A harness records this as an explicit infeasible design point. Clamping
    would move the cell somewhere the design never asked for and the emulator
    would fit it as though it were the requested one; dropping it silently
    would bias the fit and hide the boundary entirely.
    """
    with pytest.raises(InfeasibleTargetError) as excinfo:
        set_value(specs, "cell_model.targets.conduction_velocity_cm_s", -10.0)

    error = excinfo.value
    assert error.path == "cell_model.targets.conduction_velocity_cm_s"
    assert error.value == -10.0
    assert "positive" in error.reason, "the reason is not carried"
    assert "do not clamp" in str(error)


def test_an_infeasible_target_leaves_the_run_untouched(specs: TunableRun) -> None:
    """Nothing half-written survives a refusal."""
    assert specs.card is not None
    before = _diffusion(specs.card.solved)
    with pytest.raises(InfeasibleTargetError):
        set_value(specs, "cell_model.targets.conduction_velocity_cm_s", 0.0)
    assert _diffusion(specs.card.solved) == before


def test_an_unstable_solve_is_refused_even_though_no_target_reaches_it() -> None:
    """The stability guard, tested where it can actually fire.

    The shipped calibrators derive ``dt`` *from* the stability bound, so no
    target value produces a step that violates it — probing conduction
    velocities from 1e-6 to 1e12 cm/s never does. The guard is therefore
    exercised directly, against a spec built to violate it, rather than through
    a target that cannot. A test that swept targets looking for divergence would
    pass for the wrong reason: it would never have reached the branch.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import CourtemancheCellModel
    from myocard_synthetic_egm_pipeline.simulate.tuning import _ensure_stable

    reckless = CourtemancheCellModel(diffusion=0.26, dt_model_units=999.0, params={})
    with pytest.raises(InfeasibleTargetError, match="stability bound"):
        _ensure_stable(reckless, dr_model_units=0.1, path="cell_model.targets.x", value=1.0)


def test_a_courtemanche_apd_target_is_readable_but_not_sweepable() -> None:
    """The subtler half of the derived rule, and it caught a real gap.

    ``af_remodelled_crn_220ms`` *does* state ``apd90_ms: 220`` — it is the
    matched card, and 220 ms is what its conductance severity was swept to
    reach. But ``calibrate_courtemanche`` does not take an APD argument: that
    duration emerges from twelve interacting currents and is measured, never
    solved. So the target is a faithful record of what the sweep aimed at, and
    writing it would change the claim while re-solving nothing.

    Readable, therefore, and refused for writing — with the actual mechanism
    named, because "sweep the conductances instead" is the only useful answer.
    """
    crn = specs_for(*CRN_AF_CARD)
    assert crn.card is not None and crn.card.targets.apd90_ms == pytest.approx(220.0)

    assert get_value(crn, "cell_model.targets.apd90_ms") == pytest.approx(220.0)
    assert "cell_model.targets.apd90_ms" not in enumerate_paths(crn)

    with pytest.raises(DerivedParameterError, match="does not take 'apd90_ms'"):
        set_value(crn, "cell_model.targets.apd90_ms", 240.0)


def test_aliev_panfilov_does_solve_from_an_apd_target() -> None:
    """The same rule, opposite answer, so the mechanism is not vacuously strict."""
    ap = specs_for(*AP_CARD)
    assert "cell_model.targets.apd90_ms" in enumerate_paths(ap)

    updated = set_value(ap, "cell_model.targets.apd90_ms", 260.0)
    assert updated.card is not None
    assert updated.card.targets.apd90_ms == pytest.approx(260.0)
    assert updated.card.solved != ap.card.solved  # type: ignore[union-attr]


def test_sweeping_a_conductance_re_solves_and_warns_about_the_anchor() -> None:
    """A real limitation of conductance sweeps, surfaced rather than swallowed.

    A conductance scaling is an *input* to the Courtemanche calibration, so
    changing one changes the solved diffusion — leaving the old value beside it
    would be the same corruption a direct write to ``diffusion`` causes. But the
    re-solve then has no measured conduction-velocity anchor for that membrane,
    so every point of such a sweep is solved through a borrowed one. The warning
    is the only thing that tells anyone.
    """
    from myocard_synthetic_egm_pipeline.simulate.cell_models import AnchorSubstitutionWarning

    crn = specs_for(*CRN_AF_CARD)
    assert crn.card is not None
    before = _diffusion(crn.card.solved)

    with pytest.warns(AnchorSubstitutionWarning, match="CONDUCTANCES ALSO DIFFER"):
        updated = set_value(crn, "cell_model.params.g_CaL_scale", 0.4)

    assert updated.card is not None
    assert get_value(updated, "cell_model.params.g_CaL_scale") == pytest.approx(0.4)
    assert _diffusion(updated.card.solved) != pytest.approx(before)


# ---------------------------------------------------------------------------
# The mixer root — the distribution, never the realized sample
# ---------------------------------------------------------------------------


def test_the_snr_range_round_trips(specs: TunableRun) -> None:
    updated = set_value(specs, "mix.snr_db_range.low", 5.0)
    assert get_value(updated, "mix.snr_db_range.low") == pytest.approx(5.0)
    assert updated.mixer is not None
    assert updated.mixer.snr_db_range[0] == pytest.approx(5.0)
    assert updated.mixer.snr_db_range[1] == specs.mixer.snr_db_range[1]  # type: ignore[union-attr]


def test_the_realized_per_trace_snr_is_not_addressable(specs: TunableRun) -> None:
    """The invariant, asserted: θ addresses distributions, banks record samples.

    ``snr_db`` is a per-trace column produced by drawing from the range. There
    is no such field to write, and a sweep that thinks it is setting one is
    asking for something the mixer cannot honour — every trace would still draw.
    """
    assert "mix.snr_db" not in enumerate_paths(specs)
    with pytest.raises(UnknownParameterError, match="has no field 'snr_db'"):
        get_value(specs, "mix.snr_db")


def test_the_mixer_seed_is_not_a_parameter(specs: TunableRun) -> None:
    """A seed selects which sample is drawn, not the distribution it comes from.

    By the same invariant that makes the range addressable, the seed is not:
    it belongs with the fibrosis pattern and the noise segment, recorded so a
    run reproduces and never searched over.
    """
    paths = enumerate_paths(specs)
    assert "mix.master_seed" not in paths
    assert "mix.show_progress" not in paths
    assert "mix.description" not in paths
    assert "mix.snr_db_range.low" in paths


def test_a_mixer_path_on_a_clean_run_is_refused_by_name() -> None:
    """No mixer configured is a real state, not a missing field."""
    clean = specs_for(*AP_CARD, mixer=False)
    with pytest.raises(UnknownParameterError, match="no mixer config"):
        get_value(clean, "mix.snr_db_range.low")


def test_a_range_is_addressed_by_its_endpoints(specs: TunableRun) -> None:
    with pytest.raises(UnknownParameterError, match="addressed by its endpoints"):
        get_value(specs, "mix.snr_db_range.middle")


# ---------------------------------------------------------------------------
# The widening is additive
# ---------------------------------------------------------------------------


def test_every_path_the_narrower_resolver_reached_still_resolves(
    specs: TunableRun,
) -> None:
    """S22 addressed the five spec roots; none of them moved.

    Enumerated from the spec objects, the same way the resolver does, so this
    cannot drift from what the specs actually offer.
    """
    from myocard_synthetic_egm_pipeline.simulate.tuning import SPEC_ROOTS

    paths = enumerate_paths(specs)
    spec_paths = [p for p in paths if p.split(".")[0] in SPEC_ROOTS]
    assert spec_paths, "the spec roots vanished from the enumeration"

    for path in spec_paths:
        assert get_value(specs, path) is not None, path


# ---------------------------------------------------------------------------
# Sim-varying roots: readable, never writable
# ---------------------------------------------------------------------------

SIM_VARYING_CASES = [
    ("substrate.density", 0.35, "substrate.density_range"),
    ("activation.edge", "top", "activation.fixed_edge"),
    ("electrodes.height_mm", 0.42, "electrodes.height_mm_range"),
]


@pytest.mark.parametrize(("path", "value", "alternative"), SIM_VARYING_CASES)
def test_a_write_to_a_sim_varying_root_is_refused_and_names_the_distribution(
    specs: TunableRun, path: str, value: object, alternative: str
) -> None:
    """The third silent no-op this module closes.

    ``_sample_specs`` rebuilds substrate, activation and electrodes from the
    dataset config for every simulation, so a value written here never reaches
    the backend. Until now that was a docstring warning and the write still
    succeeded — the same protection that failed for the anisotropy ratio and for
    ``OMP_NUM_THREADS``.

    Refusing is only half of it: the message has to name the distribution, or
    the next person concludes the parameter cannot be swept at all when in fact
    it can, through its range.
    """
    with pytest.raises(SimVaryingParameterError) as excinfo:
        set_value(specs, path, value)

    error = excinfo.value
    assert error.path == path
    assert error.alternative is not None
    assert alternative in str(error), "the distribution to sweep is not named"
    assert "discarded before the backend sees it" in str(error)


@pytest.mark.parametrize(("path", "value", "alternative"), SIM_VARYING_CASES)
def test_a_sim_varying_path_still_reads(
    specs: TunableRun, path: str, value: object, alternative: str
) -> None:
    """Reads stay allowed: the realized value records what that run actually did."""
    assert get_value(specs, path) is not None


def test_sim_varying_paths_are_not_offered_as_sweepable(specs: TunableRun) -> None:
    """``enumerate_paths`` says what a sweep can vary, so they must not appear."""
    paths = enumerate_paths(specs)
    for path, _, _ in SIM_VARYING_CASES:
        assert path not in paths, path
    assert any(p.startswith("geometry.") for p in paths), "the surviving roots vanished too"


def test_the_two_refusals_are_distinct_types(specs: TunableRun) -> None:
    """A caller can tell "never sweepable" from "sweep it through its range".

    They are separate because the remedies differ: nothing chooses a derived
    field, whereas a sim-varying one is chosen — by a draw from a distribution
    that a sweep can own. Collapsing them into one exception would force a
    caller wanting to fall back to the range to parse the message.
    """
    with pytest.raises(DerivedParameterError):
        set_value(specs, "cell_model.diffusion", 0.3)
    with pytest.raises(SimVaryingParameterError):
        set_value(specs, "substrate.density", 0.35)

    assert not issubclass(SimVaryingParameterError, DerivedParameterError)
    assert not issubclass(DerivedParameterError, SimVaryingParameterError)


def test_a_computed_field_keeps_its_own_message_under_a_sim_varying_root(
    specs: TunableRun,
) -> None:
    """Ordering: the precise refusal wins over the coarser one.

    ``electrodes.positions_mm`` is refusable for two reasons at once. The useful
    one says it is computed from the layout; the root-level rule would only say
    electrodes are redrawn, which is true and less actionable.
    """
    with pytest.raises(DerivedParameterError, match="computed from the other fields"):
        set_value(specs, "electrodes.positions_mm", np.zeros((25, 3)))


def test_the_roots_that_survive_are_still_writable(specs: TunableRun) -> None:
    """geometry and cell_model reach the backend, so they are unaffected."""
    assert get_value(
        set_value(specs, "geometry.anisotropy_ratio", 3.0), "geometry.anisotropy_ratio"
    ) == pytest.approx(3.0)
    target = get_value(specs, "cell_model.targets.conduction_velocity_cm_s")
    updated = set_value(specs, "cell_model.targets.conduction_velocity_cm_s", target * 1.1)
    assert get_value(updated, "cell_model.targets.conduction_velocity_cm_s") == pytest.approx(
        target * 1.1
    )


# ---------------------------------------------------------------------------
# Mesh pitch is numerical, not physiological
# ---------------------------------------------------------------------------


def test_writing_the_mesh_pitch_is_refused(specs: TunableRun) -> None:
    """The last silently-wrong write on a surviving root.

    ``geometry.dr_mm`` survives ``_sample_specs`` and reaches the backend, so
    unlike the sim-varying roots a write here really does change the run — it
    just leaves the card solved for a pitch the run no longer uses, which is
    worse than being discarded.

    The message has to carry both halves. Someone reaching for this is usually
    trying to do something legitimate (study convergence) by an illegitimate
    route, so a bare refusal would leave them stuck.
    """
    with pytest.raises(NumericalParameterError) as excinfo:
        set_value(specs, "geometry.dr_mm", 0.05)

    message = str(excinfo.value)
    assert "solved at this mesh pitch" in message, "the card coupling is not named"
    assert "in the config across separate runs" in message, "the alternative is not named"


def test_reading_the_mesh_pitch_still_works(specs: TunableRun) -> None:
    """The pitch a simulation ran at is a fact worth recording."""
    assert get_value(specs, "geometry.dr_mm") == pytest.approx(specs.geometry.dr_mm)


def test_the_mesh_pitch_is_not_offered_as_sweepable(specs: TunableRun) -> None:
    assert "geometry.dr_mm" not in enumerate_paths(specs)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("geometry.fiber_angle_rad", 0.5),
        ("geometry.anisotropy_ratio", 3.0),
        ("geometry.size_mm", 60.0),
    ],
)
def test_the_other_geometry_fields_stay_writable(
    specs: TunableRun, path: str, value: float
) -> None:
    """Genuine tissue properties, which no calibration is solved against.

    This is what keeps the guard from being over-broad: anisotropy in
    particular is deliberately arranged so it leaves the along-fibre velocity —
    the quantity the card is calibrated on — untouched, so it is independent of
    the solve rather than merely absent from its signature.
    """
    assert path in enumerate_paths(specs)
    assert get_value(set_value(specs, path, value), path) == pytest.approx(value)


def test_only_the_pitch_is_card_coupled() -> None:
    """Pins what the mechanical rule concludes, so a signature change is loud.

    The set is read from the calibrator signatures rather than written down, so
    it cannot go stale — and this records what it currently yields, so a new
    geometry argument to a calibration shows up here rather than as a quietly
    widened refusal.
    """
    from myocard_synthetic_egm_pipeline.simulate.tuning import _card_coupled_geometry_names

    assert _card_coupled_geometry_names() == frozenset({"dr_mm"})


def test_all_three_refusals_are_distinct_types(specs: TunableRun) -> None:
    """Three reasons, three types, three remedies.

    Derived: nothing chooses it. Sim-varying: sweep its distribution instead.
    Numerical: do not sweep it at all, vary it across runs. A caller can act on
    each differently, which is the whole reason they are not one exception.
    """
    with pytest.raises(DerivedParameterError):
        set_value(specs, "cell_model.diffusion", 0.3)
    with pytest.raises(SimVaryingParameterError):
        set_value(specs, "substrate.density", 0.35)
    with pytest.raises(NumericalParameterError):
        set_value(specs, "geometry.dr_mm", 0.05)

    types = (DerivedParameterError, SimVaryingParameterError, NumericalParameterError)
    for one in types:
        for other in types:
            if one is not other:
                assert not issubclass(one, other), (one, other)
