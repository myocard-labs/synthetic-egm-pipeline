"""Physiological targets, what a run measured, and the card that binds them.

**Universal by construction.** Everything here survives a cell-model swap and a
backend swap: every membrane model targets a conduction velocity and an action
potential duration, and every one of them benefits from a named parameterisation
that can be cited and verified. The model-specific *solve* lives in
:mod:`~myocard_synthetic_egm_pipeline.simulate.cell_models`; this module only
knows that a card has one.

Why a routine rather than four constants
----------------------------------------
The 2026-06-10 calibration wrote four numbers into ``constants.py`` with a
comment saying where they came from. One was wrong for two months — it bought
conduction velocity by destroying action potential duration — and nothing could
catch it, because there was no executable statement of what the numbers were
*for*. Making the physiology the input and the knobs the output buys three
things a constant cannot: the derivation is reviewable, it is testable by
round-trip, and it is the same solve STU4 needs per proposal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel,
    CellModelSpec,
    CourtemancheCellModel,
    calibrate_aliev_panfilov,
    calibrate_courtemanche,
)

SOLVED_MATCH_RTOL: float = 1e-3
"""Tolerance when checking a card's recorded knobs against a fresh solve.

Loose on purpose. The check exists to catch the **solve changing** — a different
formula, a re-measured constant, a new safety factor — which moves values by
whole percent. It is not a tamper check. The tolerance has to absorb the
rounding in a human-readable YAML file, where ``5.710`` stands for
``5.70983...``; demanding exactness would mean writing 17 significant figures
into a file whose entire purpose is being read by people.
"""

MEASURED_APD_RTOL: float = 0.05
"""How close a *measured* APD90 must land to a stated APD90 target.

Applies only to a model that measures APD rather than solving it, where the
target records what the parameters were chosen to reach rather than what a
formula guarantees. Far looser than :data:`SOLVED_MATCH_RTOL` because it is
absorbing a one-dimensional authoring sweep stopping at a grid point rather
than the rounding in a YAML file: 5 % of 220 ms is 11 ms, well inside the
literature's own spread for the quantity (95-287 ms across the cited cAF
studies, CL-180).
"""


@dataclass(frozen=True)
class ModelTargets:
    """What we asked the tissue to do, in units a cardiologist would use.

    The block the white paper cites and the block STU4 searches.

    **Anisotropy is deliberately absent.** It is fibre architecture, not membrane
    behaviour, and it already lives on
    :class:`~myocard_synthetic_egm_pipeline.simulate.specs.Patch2DGeometry`. An
    earlier draft carried it here too, giving one number two homes with nothing
    reconciling them — and the solve never read it.

    **Long-term home: `SubstrateStrategy`** (Daniel, 2026-08-15; FB-34). In the
    ablation literature "substrate" means the arrhythmogenic tissue state —
    structural *and* electrical remodelling together — and a shortened APD is
    textbook AF remodelling, which is why the shipped card is called
    ``af_remodelled_220ms``. Today that leaves one half of substrate remodelling
    (fibrosis) on the substrate spec and the other half (APD, CV) here.

    The move is deferred because it renames schema-adjacent metadata keys, but
    this class is kept **migration-shaped** for it:

    - no upward imports — it depends on nothing, so it can be lifted wholesale;
    - **the solve takes scalars, not this object** (see
      :func:`~...cell_models.calibrate_aliev_panfilov`), so re-homing the targets
      does not touch the calibration at all;
    - verification takes ``targets`` and ``solved`` separately, with the
      card-level wrapper thin, so only the wrapper moves.

    The payoff is bigger than tidiness: **substrate is already sampled per
    simulation** (``density_range`` draws from ``sim_rng`` in the dataset loop),
    so per-simulation APD sampling over 200-260 ms — deferred out of S38b for
    want of a sampling mechanism — falls out for free once targets live there.
    """

    conduction_velocity_cm_s: float
    """Along-fibre conduction velocity. Human atrial free wall measures
    88 +/- 9 cm/s intra-operatively in sinus rhythm (Hansson 1998).

    **The universal half.** Every membrane model is embedded in the same
    diffusion operator, so every one of them can be solved for a velocity, and
    two banks stating the same figure means the same thing under both. That is
    what makes a Courtemanche-versus-Aliev-Panfilov comparison a comparison of
    *models* rather than of two unrelated tissues.
    """

    apd90_ms: float | None = None
    """APD90, 10 % upstroke to 90 % repolarisation. **Optional, per model.**

    Aliev-Panfilov solves it: its time axis is arbitrary, so a duration in
    milliseconds is set by choosing what a model time unit means, and a card
    without this number cannot be solved at all. Courtemanche cannot: APD falls
    out of the ionic equations with no time-unit constant and no closed-form
    inverse, so it is **measured and recorded, never solved**, and a Courtemanche
    card may legitimately state no APD target. Stating one anyway is allowed and
    is checked against the ``measured:`` block instead of against a solve
    (:func:`verify_targets_against_solve`) — a target nothing verifies is worse
    than an absent one.

    **Must exceed the trace duration T** where it is stated. Repolarisation
    leaves the cropped window iff ``APD > T * (1 - p)``, so ``APD >= T`` is the
    unconditional guarantee across the whole position range. At T = 192 ms an
    APD of 180 — plausible from the AF literature, which quotes short-cycle
    rates we do not simulate — fails for any ``p < 0.0625`` (CL-176),
    reintroducing the exact defect the calibration exists to remove.
    """

    def __post_init__(self) -> None:
        if self.conduction_velocity_cm_s <= 0:
            raise ValueError("conduction_velocity_cm_s must be positive.")
        if self.apd90_ms is not None and self.apd90_ms <= 0:
            raise ValueError("apd90_ms must be positive.")


@dataclass(frozen=True)
class MeasuredValues:
    """What a simulation actually produced at the solved parameters.

    Recorded because the analytic solve is a few percent off and pretending
    otherwise would be a small dishonesty in a document meant to be cited.
    Discretization widens the upstroke relative to a fixed mesh, so CV lands
    high by roughly 1.5 % at ``D ~ 5`` and 8 % at ``D ~ 10``; APD lands within
    0.5 %. A few percent of CV is immaterial against a literature spread of
    88 +/- 9 cm/s. Reporting the *target* as though it were the *achieved* value
    would not be.
    """

    conduction_velocity_cm_s: float
    apd90_ms: float


@dataclass(frozen=True)
class ModelCard:
    """A named parameterisation: targets, the solved cell model, what it measured.

    A file rather than a block inside a generation config so that a
    parameterisation has an **identity** — citable, diffable, referenceable from
    a methods section — and because STU4's output is a *region* of parameters,
    which means writing many of these.
    """

    name: str
    targets: ModelTargets
    solved: CellModelSpec
    measured: MeasuredValues | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A model card needs a name; it is how banks refer to it.")


def verify_targets_against_solve(
    targets: ModelTargets,
    solved: CellModelSpec,
    *,
    label: str,
    dr_mm: float,
    dr_model_units: float,
    measured: MeasuredValues | None = None,
    dimensions: int = 2,
) -> None:
    """Re-solve a card's targets and confirm the recorded cell model still matches.

    Runs on **every load**, which is the point: a CI-only guard protects one
    machine, a load-time guard protects every bank anyone generates. The solve is
    analytic, so this costs microseconds.

    Catches the solve **changing** under a card that did not — a re-measured
    constant, a corrected formula, a different safety factor. Any of those would
    otherwise leave two banks bearing the same parameterisation name and
    different physics, detectable only by someone thinking to compare them.

    Does **not** catch a card whose values were never produced by this routine.
    :class:`MeasuredValues` and the round-trip test cover that; this is about
    drift, not authenticity.

    Dispatches on the cell-model type rather than assuming Aliev-Panfilov, so
    Courtemanche joins by adding a branch and nothing here has to be rewritten.

    ``measured`` is optional and is only consulted for a target the model does
    not solve. It is passed separately rather than read off a card so that this
    function keeps taking the pieces it checks — the shape FB-34's move of
    ``targets`` onto ``SubstrateStrategy`` needs.
    """
    fields: tuple[str, ...]
    if isinstance(solved, AlievPanfilovCellModel):
        if targets.apd90_ms is None:
            raise ValueError(
                f"{label} states no apd90_ms target, but Aliev-Panfilov solves its "
                "time unit from one — without it there is nothing to verify and "
                "nothing that fixed the recorded time_unit_ms. Add the target, or "
                "use a model that measures APD instead of solving it."
            )
        fresh: CellModelSpec = calibrate_aliev_panfilov(
            conduction_velocity_cm_s=targets.conduction_velocity_cm_s,
            apd90_ms=targets.apd90_ms,
            dr_mm=dr_mm,
            dr_model_units=dr_model_units,
            eps=solved.eps,
            dimensions=dimensions,
        )
        fields = ("time_unit_ms", "diffusion", "eps", "dt_model_units")
    elif isinstance(solved, CourtemancheCellModel):
        fresh = calibrate_courtemanche(
            conduction_velocity_cm_s=targets.conduction_velocity_cm_s,
            dr_mm=dr_mm,
            dr_model_units=dr_model_units,
            params=solved.params,
            dimensions=dimensions,
        )
        fields = ("diffusion", "dt_model_units")
        _verify_measured_apd_target(targets, measured, label=label)
    else:
        raise ValueError(
            f"no calibration is registered for cell model {solved.type!r}, so its "
            "card cannot be verified. Add a branch here when the model gains a "
            "solve — an unverifiable card is worse than no card, because it looks "
            "checked."
        )

    drifted = [
        f"  {field}: card says {getattr(solved, field)!r}, "
        f"the solve now gives {getattr(fresh, field)!r}"
        for field in fields
        if not math.isclose(
            float(getattr(solved, field)),
            float(getattr(fresh, field)),
            rel_tol=SOLVED_MATCH_RTOL,
        )
    ]

    if drifted:
        raise ValueError(
            f"{label} no longer agrees with the calibration:\n"
            + "\n".join(drifted)
            + "\n\nThe solve has changed since this card was written. Re-solve the "
            "targets and update the card (and re-measure its measured block), or "
            "pin the old routine. Do NOT relax the tolerance: two banks under one "
            "parameterisation name with different physics is the failure this "
            "check exists to prevent."
        )


def _verify_measured_apd_target(
    targets: ModelTargets,
    measured: MeasuredValues | None,
    *,
    label: str,
) -> None:
    """For a model that measures APD, check the target against the measurement.

    The alternative was to ignore ``apd90_ms`` on a Courtemanche card, and this
    project has been bitten three times by a knob that silently does nothing —
    an ``anisotropy_ratio`` that was a no-op for the life of the project being
    the worst of them. A stated target has to be either verified or refused.

    Nothing is simulated here: the card records what a run measured, so the
    check is two recorded numbers against each other and stays load-time cheap.
    A card that states the target but has not been measured yet is a real
    intermediate state (the solve exists before the run does), so it passes.
    """
    if targets.apd90_ms is None or measured is None:
        return
    if not math.isclose(
        float(measured.apd90_ms), float(targets.apd90_ms), rel_tol=MEASURED_APD_RTOL
    ):
        raise ValueError(
            f"{label} aims at apd90_ms={targets.apd90_ms} but records a measured "
            f"{measured.apd90_ms}, which is outside {MEASURED_APD_RTOL:.0%}. This "
            "model measures APD rather than solving it, so the target is only a "
            "claim about the parameters chosen for it — re-author them until the "
            "measurement lands on the target, or state the target the card "
            "actually reached."
        )


def verify_solved(
    card: ModelCard,
    *,
    dr_mm: float,
    dr_model_units: float,
    dimensions: int = 2,
) -> None:
    """Card-level wrapper over :func:`verify_targets_against_solve`.

    Deliberately thin. The real check takes ``targets`` and ``solved`` as
    separate arguments precisely so that when targets move to
    ``SubstrateStrategy`` (FB-34) this wrapper is the only thing that has to
    change — the verification logic never learns where its inputs came from.
    """
    verify_targets_against_solve(
        card.targets,
        card.solved,
        label=f"model card {card.name!r}",
        dr_mm=dr_mm,
        dr_model_units=dr_model_units,
        measured=card.measured,
        dimensions=dimensions,
    )


__all__ = [
    "MEASURED_APD_RTOL",
    "SOLVED_MATCH_RTOL",
    "MeasuredValues",
    "ModelCard",
    "ModelTargets",
    "verify_solved",
    "verify_targets_against_solve",
]
