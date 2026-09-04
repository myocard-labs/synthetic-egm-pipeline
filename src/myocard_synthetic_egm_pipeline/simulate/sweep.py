"""Design matrices over the tunable knobs, and the harness that applies them.

A **sweep** generates several banks' worth of simulations at different points in
parameter space, so that a later step can ask how a feature responds to a knob.
This module produces the design — which knob takes which value in which cell —
and applies each cell through the path resolver. It does not simulate; the
dataset orchestrator does that, once per cell.

Additive by construction
------------------------
**With no sweep configured, nothing here runs and the sampling is untouched.**
That is not a convention, it is the property the whole design rests on: every
bank generated before this module existed must still be reproducible, byte for
byte, from the same seed. The harness is a loop *around* the existing
per-simulation draw, never a replacement for it, and a test asserts the
bit-identical result rather than trusting the reading.

What a design cell may write
----------------------------
Only what survives to the backend, which is narrower than it looks. The
per-simulation sampler rebuilds the substrate, activation and electrode specs
for every simulation, so a cell that wrote ``substrate.density`` would be
generating the same design point over and over. The resolver refuses those
paths, and a cell writes the **distribution** instead —
``substrate.density_range.low`` and ``.high``. See
:mod:`~myocard_synthetic_egm_pipeline.simulate.tuning` for the three refusals
and their remedies.

Roles come from the θ-spec
--------------------------
``role`` is ``label_param`` or ``nuisance``, read off the knob rather than
inferred from its name. **No sampler here special-cases a parameter name**: a
sampler that knows "density is the label" is one that silently stops being right
the day a label policy changes. :class:`OATSampler` treats the two roles
identically — a screen varies everything it is given — and carries the role
through into the design so that interpretation, which does depend on it, has it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from myocard_synthetic_egm_pipeline.simulate.tuning import (
    TunableRun,
    set_value,
)

__all__ = [
    "DesignCell",
    "InfeasibleCell",
    "Knob",
    "OATSampler",
    "Sampler",
    "SweepConfig",
    "apply_cell",
    "build_design",
]

#: Roles the parameter contract defines. Not re-litigated here — the schema's
#: own wording is exactly right, and a second enum would be a second source of
#: truth for one fact.
ROLES = ("label_param", "nuisance")

#: Transforms the contract allows, and the space a sampler spaces levels in.
TRANSFORMS = ("identity", "log", "logit")

#: Paths a sampler must never vary, whatever a config asks.
#:
#: **The stimulus edge, and the reason is scientific rather than technical.**
#: The edge is randomised per simulation precisely so that activation direction
#: cannot become a shortcut feature — a classifier that learns "waves arriving
#: from the left are fibrotic" has learned the generator, not the physiology.
#: A swept edge gives every simulation in a cell one fixed direction, which
#: reinstates exactly that shortcut inside each bank.
#:
#: The resolver already refuses the write, so an attempt would raise rather than
#: corrupt anything. This is the earlier, clearer failure: refused when the
#: design is built, naming the reason, instead of once per cell at apply time.
#: A deliberate direction-sensitivity study is a config-level choice —
#: ``activation.fixed_edge`` across separate runs — not a swept axis.
UNSWEEPABLE_PATHS = frozenset({"activation.edge", "activation.fixed_edge"})


class SweepConfigError(ValueError):
    """A sweep specification is malformed or names something unsweepable."""


@dataclass(frozen=True)
class Knob:
    """One entry of the θ-spec: what is swept, over what range, meaning what.

    Mirrors the parameter contract's ``TunedParam`` — ``path`` and ``bounds``
    required, ``transform``, ``role`` and ``nominal`` optional — rather than
    inventing a parallel vocabulary. ``nominal`` defaults to the midpoint of
    ``bounds`` in the knob's transformed space, because a one-at-a-time design
    is only self-describing if the pinned value is known, and a missing one
    would otherwise make two screens incomparable.
    """

    path: str
    bounds: tuple[float, float]
    role: str = "nuisance"
    transform: str = "identity"
    nominal: float | None = None

    def __post_init__(self) -> None:
        low, high = self.bounds
        if not math.isfinite(low) or not math.isfinite(high):
            raise SweepConfigError(f"{self.path}: bounds must be finite, got {self.bounds}.")
        if low > high:
            raise SweepConfigError(f"{self.path}: bounds must be ascending, got [{low}, {high}].")
        if self.role not in ROLES:
            raise SweepConfigError(
                f"{self.path}: role must be one of {', '.join(ROLES)}; got {self.role!r}."
            )
        if self.transform not in TRANSFORMS:
            raise SweepConfigError(
                f"{self.path}: transform must be one of {', '.join(TRANSFORMS)}; "
                f"got {self.transform!r}."
            )
        if self.transform == "log" and low <= 0:
            raise SweepConfigError(
                f"{self.path}: a log transform needs strictly positive bounds; got [{low}, {high}]."
            )
        if self.transform == "logit" and not (low > 0.0 and high < 1.0):
            raise SweepConfigError(
                f"{self.path}: a logit transform needs bounds inside (0, 1); got [{low}, {high}]."
            )
        if self.path in UNSWEEPABLE_PATHS:
            raise SweepConfigError(
                f"{self.path!r} must not be swept. The stimulus edge is randomised "
                "per simulation so that activation direction cannot become a "
                "shortcut feature; fixing it per design cell reinstates that "
                "shortcut inside every bank. For a deliberate direction study, "
                "set activation.fixed_edge in the config and compare separate runs."
            )
        if self.nominal is None:
            object.__setattr__(self, "nominal", self._midpoint())

    def _midpoint(self) -> float:
        low, high = self.bounds
        return self._from_unit(0.5) if low != high else low

    def _from_unit(self, unit: float) -> float:
        """Place a level, spaced in the knob's own transformed space.

        A log-scaled knob gets geometric spacing and a logit-scaled one gets
        spacing that stays off the boundary — which is the point of recording a
        transform at all. ``bounds`` stay in natural units either way, as the
        contract requires, so a reader need not know the transform to know what
        was swept.
        """
        low, high = self.bounds
        if low == high:
            return low
        if self.transform == "log":
            return math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
        if self.transform == "logit":
            return _expit(_logit(low) + unit * (_logit(high) - _logit(low)))
        return low + unit * (high - low)

    def levels(self, count: int) -> tuple[float, ...]:
        """``count`` values spanning the bounds, endpoints included."""
        if count < 1:
            raise SweepConfigError(f"{self.path}: need at least one level, got {count}.")
        if count == 1:
            assert self.nominal is not None
            return (self.nominal,)
        return tuple(self._from_unit(i / (count - 1)) for i in range(count))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class DesignCell:
    """One point of the design: every knob's value, and which one this cell is about.

    ``values`` carries **every** knob, not only the varied one. A cell that
    recorded only what changed would leave a reader unable to say what the rest
    were held at, and two screens run against different nominals would look
    comparable when they are not.
    """

    index: int
    values: Mapping[str, float]
    varied: str | None
    """The knob this cell varies, or ``None`` for the all-nominal baseline."""


@dataclass(frozen=True)
class InfeasibleCell:
    """A design cell that could not be applied, kept rather than dropped.

    **Dropping one silently is the failure this type exists to prevent.** A
    design with its failures removed fits an emulator on a biased subset of the
    space and leaves it unable to see where the boundary is — and the missing
    cells are exactly the informative ones. Recording the reason means the
    boundary can be described afterwards instead of guessed at.

    Deliberately **not** shaped around any particular failure. The stability
    bound was expected to be the source of these and, measured, is not: the
    calibrators derive the timestep from the bound rather than checking a
    proposal against it, so the conduction-velocity axis has no infeasible
    region at all. Infeasibility is real for other reasons — a target outside a
    physical domain, a geometry a grid does not fit in — so the record is
    general and stores whatever was raised.
    """

    cell: DesignCell
    path: str
    value: Any
    reason: str
    error_type: str


@runtime_checkable
class Sampler(Protocol):
    """Turns a knob list into a design matrix.

    The whole harness is generic over this. A one-at-a-time screen and a
    space-filling design differ only here, which is why the sweep step is one
    issue rather than two: swapping the sampler is the entire difference.
    """

    @property
    def type(self) -> str:
        """Discriminator, as a read-only property — frozen dataclasses satisfy it."""
        ...

    def design(self, knobs: Sequence[Knob]) -> tuple[DesignCell, ...]:
        """Every cell to generate, in the order they should be generated."""
        ...


@dataclass(frozen=True)
class OATSampler:
    """One knob at a time: vary one, pin the rest at ``nominal``.

    The screening design. It answers "which knobs matter at all" in
    ``1 + sum(levels - 1)`` cells rather than the product a grid would need, and
    that is what a later space-filling design is spent on once the list is
    short.

    It cannot see interactions, by construction — that is the trade, not an
    oversight. A knob whose effect only appears alongside another reads as inert
    here.
    """

    levels: int = 3
    include_baseline: bool = True
    type: str = field(default="oat", init=False)

    def __post_init__(self) -> None:
        if self.levels < 2:
            raise SweepConfigError(
                f"an OAT screen needs at least 2 levels per knob to show a "
                f"response; got {self.levels}. Use 2 for endpoints only."
            )

    def design(self, knobs: Sequence[Knob]) -> tuple[DesignCell, ...]:
        nominal = {k.path: _nominal_of(k) for k in knobs}
        cells: list[DesignCell] = []

        if self.include_baseline:
            # The all-nominal cell, generated first. Every varied cell differs
            # from it in exactly one coordinate, so it is the reference the
            # whole screen is read against.
            cells.append(DesignCell(index=0, values=dict(nominal), varied=None))

        for knob in knobs:
            for level in knob.levels(self.levels):
                if self.include_baseline and math.isclose(level, nominal[knob.path]):
                    # Already generated as the baseline; regenerating it would
                    # spend a cell to re-measure the same point.
                    continue
                values = dict(nominal)
                values[knob.path] = level
                cells.append(DesignCell(index=len(cells), values=values, varied=knob.path))

        return tuple(cells)


def _nominal_of(knob: Knob) -> float:
    assert knob.nominal is not None  # filled in by __post_init__
    return knob.nominal


@dataclass(frozen=True)
class SweepConfig:
    """The ``sweep:`` block: which sampler, over which knobs."""

    sampler: Sampler
    knobs: tuple[Knob, ...]

    def design(self) -> tuple[DesignCell, ...]:
        return self.sampler.design(self.knobs)


def build_design(config: SweepConfig) -> tuple[DesignCell, ...]:
    """The design matrix this sweep asks for."""
    if not config.knobs:
        raise SweepConfigError("a sweep needs at least one knob; sweep.knobs is empty.")
    return config.design()


def apply_cell(run: TunableRun, cell: DesignCell) -> TunableRun:
    """Write every value in ``cell`` onto ``run``, in path order.

    Raises whatever the resolver raises. The caller decides whether that is a
    fatal error or an infeasible point to record — this function does not
    swallow it, because a harness that quietly skipped a value would produce a
    bank claiming a design point it never reached.
    """
    for path in sorted(cell.values):
        run = set_value(run, path, cell.values[path])
    return run
