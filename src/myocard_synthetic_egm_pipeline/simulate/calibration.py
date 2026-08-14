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
    calibrate_aliev_panfilov,
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
    88 +/- 9 cm/s intra-operatively in sinus rhythm (Hansson 1998)."""

    apd90_ms: float
    """APD90, 10 % upstroke to 90 % repolarisation.

    **Must exceed the trace duration T.** Repolarisation leaves the cropped
    window iff ``APD > T * (1 - p)``, so ``APD >= T`` is the unconditional
    guarantee across the whole position range. At T = 192 ms an APD of 180 —
    plausible from the AF literature, which quotes short-cycle rates we do not
    simulate — fails for any ``p < 0.0625`` (CL-176), reintroducing the exact
    defect the calibration exists to remove.
    """

    def __post_init__(self) -> None:
        if self.conduction_velocity_cm_s <= 0:
            raise ValueError("conduction_velocity_cm_s must be positive.")
        if self.apd90_ms <= 0:
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
    """
    if isinstance(solved, AlievPanfilovCellModel):
        fresh: CellModelSpec = calibrate_aliev_panfilov(
            conduction_velocity_cm_s=targets.conduction_velocity_cm_s,
            apd90_ms=targets.apd90_ms,
            dr_mm=dr_mm,
            dr_model_units=dr_model_units,
            eps=solved.eps,
            dimensions=dimensions,
        )
        fields = ("time_unit_ms", "diffusion", "eps", "dt_model_units")
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
        dimensions=dimensions,
    )


__all__ = [
    "SOLVED_MATCH_RTOL",
    "MeasuredValues",
    "ModelCard",
    "ModelTargets",
    "verify_solved",
    "verify_targets_against_solve",
]
