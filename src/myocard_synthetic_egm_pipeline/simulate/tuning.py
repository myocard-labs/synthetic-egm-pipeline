"""Resolve a dotted parameter path against the per-simulation specs.

A sweep names the parameters it varies by **path** — ``substrate.density``,
``cell_model.params.g_CaL_scale``, ``activation.edge`` — and this module is what
turns such a string into a value it can read and a new spec set it can write.
Membership of the swept set therefore becomes a per-sweep choice in a config
file, with no new types and no contract change each time it moves.

This module defines the grammar
------------------------------
The parameter contract leaves ``path`` deliberately unconstrained — no pattern,
no enum, no grammar — on the stated grounds that fixing a shape before anything
resolved one would bake in something untested. This is the resolver, so this is
where the shape gets decided. It is kept as small as it can be:

    ``<root>.<field>``            e.g. ``substrate.density``
    ``<root>.<field>.<key>``      e.g. ``cell_model.params.g_CaL_scale``

Roots are the five strategy specs, named exactly as they are on
:class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationSpecs`. The
second form exists for one case — a mapping of named conductance scalings — and
recurses no further. Anything a later sweep needs beyond this is an **additive**
extension, which is the property that made deferring the grammar safe.

Two ways a write can be wrong, and both are silent
--------------------------------------------------
**Derived fields.** A model card records what its calibration solved and
re-verifies it on every load. Writing ``cell_model.diffusion`` directly would
give a bank whose card claims one parameterisation and whose physics is another
— the exact failure the card system exists to prevent, arriving through a new
door. These are refused, with the thing to sweep instead named in the message.

**Fields nothing reads.** ``CenteredGrid2D`` records ``height_mm`` and computes
``positions_mm`` from it, and the backend reads *only* ``positions_mm``. So
``dataclasses.replace(electrodes, height_mm=0.7)`` changes a number that no
solver, kernel or tracker ever consults: every simulation in the sweep would
record a different height and every one would have identical electrode
geometry. That is the same shape of failure as the anisotropy knob that spent
the life of the project assigning to an object nothing read — and it sits on one
of the contract's own example paths. Writes here therefore **rebuild** the
computed companions rather than editing the scalar beside them; see
:func:`_normalised`.

Three roots are read-only, because a write to them would be discarded
---------------------------------------------------------------------
``dataset._sample_specs`` rebuilds ``substrate``, ``activation`` and
``electrodes`` from the dataset config for **every** simulation, so a value
written to any of them is overwritten before the backend sees it. They are
therefore refused for writing, with :class:`SimVaryingParameterError`.

This was a docstring warning first, with the write still succeeding — which is
the same strength of protection that failed for the anisotropy ratio, for the
detection curve before it was wired, and for ``OMP_NUM_THREADS``: a silent
no-op with a comment beside it. A comment is not a guard.

Reads stay allowed. The realized value is a faithful record of what that
simulation actually ran with, which is exactly what a bank should carry.

``geometry`` and ``cell_model`` are passed through untouched, so writes to them
do reach the backend — with one exception on each. ``cell_model``'s solved
fields are refused because a calibration produces them; ``geometry.dr_mm`` is
refused because a card is *solved against* it, and because a mesh pitch is a
numerical choice rather than a tissue property. See
:class:`NumericalParameterError`.

Three refusals, three reasons, three types
------------------------------------------
============================  ==================================  ==================================
raised for                    because                             what to do instead
============================  ==================================  ==================================
:class:`DerivedParameterError` a solve produces it                sweep what the solve is solved *from*
:class:`SimVaryingParameterError` it is redrawn every simulation  sweep the distribution it is drawn from
:class:`NumericalParameterError`  it is a discretisation choice   vary it across runs, in the config
============================  ==================================  ==================================

They are separate types because the remedies differ and a caller could
reasonably handle each differently — a sampler that fell back to sweeping a
range on the second would be wrong to try it on the first or the third.

It is also the clearest statement of the invariant below. Density, edge and
height are *drawn* per simulation, so what a sweep can own is the distribution
they are drawn from, and those distributions live in the dataset config rather
than on the realized spec objects this resolver addresses.

Distributions, not samples
--------------------------
**θ addresses distribution parameters; banks record realized samples.** The
mixer root addresses ``snr_db_range``, never the per-trace ``snr_db`` column.
The same shape recurs throughout: a target conduction velocity is solved to a
diffusion and realized as a measured velocity; a height *range* is sampled to a
per-simulation height. The addressable thing is always the parameter of the
distribution, and the bank is where the draw is recorded.

How derived fields are recognised
---------------------------------
**Not by a hand-written deny-list**, which is right on the day it is written and
silently wrong the first time a field is added. A field of a cell model is
*chosen* exactly when the calibration function accepts it as an argument, and
*derived* otherwise — that is what those two words mean here, and
:func:`inspect.signature` can be asked directly.

So ``eps`` is chosen (``calibrate_aliev_panfilov`` takes an ``eps``) while
``diffusion`` is derived (it does not). Add a solved output and it is refused
without anyone remembering to add it anywhere; add a genuine input and it
becomes sweepable only by being threaded through the calibration, which is the
deliberate act that makes it an input in the first place. The failure direction
is safe too: a mechanism that loses track refuses a legitimate sweep, which is
noticed immediately, rather than permitting a corrupting one, which is not.

``tests/test_tuning.py`` pins the resulting sets, so a rename that quietly flips
a field from derived to chosen fails there rather than in a bank.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np

from myocard_synthetic_egm_pipeline.mixer import MixerConfig
from myocard_synthetic_egm_pipeline.simulate.calibration import ModelCard, ModelTargets
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel,
    CellModelSpec,
    CourtemancheCellModel,
    calibrate_aliev_panfilov,
    calibrate_courtemanche,
)
from myocard_synthetic_egm_pipeline.simulate.result import SimulationSpecs
from myocard_synthetic_egm_pipeline.simulate.specs import CenteredGrid2D, Patch2DGeometry

__all__ = [
    "DerivedParameterError",
    "InfeasibleTargetError",
    "NumericalParameterError",
    "SimVaryingParameterError",
    "TunableRun",
    "UnknownParameterError",
    "enumerate_paths",
    "get_value",
    "set_value",
]


@dataclass(frozen=True)
class TunableRun:
    """Everything one run's parameters can address.

    **The subject is a bundle, not the specs alone.** Two of the three things a
    sweep needs to reach are not per-simulation objects: a calibration target
    lives on the model card, and the noise distribution lives in the mixer
    config. The alternative — a resolver per root — would force every caller to
    know which resolver a given path needs, which is the taxonomy leaking into
    everything that touches a path. One subject, one resolver, one grammar.

    The bundle is also exactly what a re-solve needs. Re-running a calibration
    requires the mesh pitch it is being solved for, and that is
    ``specs.geometry.dr_mm`` — already here.

    ``card`` and ``mixer`` are optional because not every caller has them: a
    clean run has no mixer, and a bundle built for a spec-only question needs no
    card. A path into a root that is absent is refused by name rather than
    resolving to nothing.
    """

    specs: SimulationSpecs | None = None
    card: ModelCard | None = None
    mixer: MixerConfig | None = None
    dataset: Any = None
    """The :class:`~...simulate.dataset.DatasetConfig`, when the caller has one.

    Typed loosely to avoid a circular import — ``dataset`` imports the runner,
    which imports this module's neighbours. Its presence is what makes the
    *distribution* paths resolvable: ``substrate.density_range`` and
    ``electrodes.height_mm_range`` live here, not on any per-simulation spec,
    which is precisely why writes to the realized values were refused.
    """
    dr_model_units: float | None = None
    """The solver's own space step. ``None`` means "the mesh pitch", which is
    the only value Courtemanche permits and the shipped Aliev-Panfilov default.
    Held because a re-solve needs it and it is a ``RunConfig`` knob rather than
    a property of any spec here."""

    def __post_init__(self) -> None:
        if self.specs is None and self.dataset is None:
            raise ValueError(
                "a TunableRun needs per-simulation specs, a dataset config, or "
                "both: with neither there is nothing for a path to resolve "
                "against."
            )

    @property
    def geometry(self) -> Any:
        """The geometry, from whichever half of the bundle holds one.

        The dataset config wins when both are present: it is what a generation
        run reads, and the specs are a record of one simulation drawn from it.
        """
        if self.dataset is not None:
            return self.dataset.geometry
        assert self.specs is not None
        return self.specs.geometry

    @property
    def cell_model(self) -> Any:
        if self.dataset is not None:
            return self.dataset.cell_model
        assert self.specs is not None
        return self.specs.cell_model

    @property
    def effective_dr_model_units(self) -> float:
        if self.dr_model_units is not None:
            return self.dr_model_units
        return float(self.geometry.dr_mm)


#: Roots that resolve against the per-simulation specs, named as they are on
#: :class:`SimulationSpecs`. Read off the type rather than written out, so a
#: sixth spec becomes addressable by being added there and nowhere else.
SPEC_ROOTS: tuple[str, ...] = tuple(f.name for f in fields(SimulationSpecs))

#: The mixer root. Separate because it is not a simulation spec: it describes
#: how noise is added to traces after every simulation has finished.
MIX_ROOT = "mix"

#: The pseudo-field under ``cell_model`` that reaches the card's calibration
#: targets. Not a field of any cell-model spec — a spec holds what the solve
#: *produced*, and this is what it was asked for.
TARGETS_FIELD = "targets"

ROOTS: tuple[str, ...] = (*SPEC_ROOTS, MIX_ROOT)

#: Roots ``dataset._sample_specs`` rebuilds for every simulation.
#:
#: A write to any of them is discarded before the backend runs, so they are
#: **read-only here**. Structural rather than per-field: the sampler
#: reconstructs each of these spec objects wholesale, so nothing on them
#: survives, and a rule about the root cannot go stale field by field.
#:
#: **This is permanent, not scaffolding for S23.** When the ranges become
#: addressable, ``substrate.density`` still will not survive — it is still
#: redrawn each simulation. What changes is that ``substrate.density_range``
#: starts working *alongside* this refusal.
#:
#: ``geometry`` and ``cell_model`` are absent because they are passed through
#: from the dataset config untouched, so writes to them do reach the backend.
SIM_VARYING_ROOTS: frozenset[str] = frozenset({"substrate", "activation", "electrodes"})

#: Path → the config key that owns its distribution, for the refusal message.
#:
#: Hand-maintained, and safe to be: it supplies only the *suggestion*. The
#: refusal itself comes from :data:`SIM_VARYING_ROOTS` above, so a path missing
#: from this table is still refused — it just gets a vaguer message. The
#: failure mode of staleness here is a worse sentence, not a lost guard.
_SIM_VARYING_ALTERNATIVE: Mapping[str, str] = {
    "substrate.density": "substrate.density_range",
    "activation.edge": "activation.fixed_edge (null randomises it per simulation)",
    "activation.time_model_units": "activation.stimulus_delay_ms",
    "electrodes.height_mm": "electrodes.height_mm_range",
    "electrodes.n_rows": "electrodes.n_rows",
    "electrodes.n_cols": "electrodes.n_cols",
    "electrodes.spacing_mm": "electrodes.spacing_mm",
}

#: Distribution paths → the ``DatasetConfig`` field that holds them.
#:
#: **These are the writable counterparts of the read-only realized values.**
#: ``substrate.density`` is redrawn every simulation and refused; the range it
#: is drawn from lives on the dataset config and is what a sweep owns. The path
#: names follow the config file's own blocks rather than the dataclass field
#: names, so a sweep and a config spell the same knob the same way.
#:
#: Point-valued or endpoint-valued, and the rule rather than a taste call:
#: **a knob drawn per simulation or per trace is addressed by its
#: distribution's endpoints; a knob fixed for the whole bank is addressed as a
#: scalar.** Conduction-velocity target, anisotropy and a conductance scaling
#: are one value for every simulation in a bank, so they are one θ dimension
#: each. Density, electrode height and SNR are *drawn*, so their realism
#: includes their spread — pin SNR to a point and every trace in a design cell
#: carries an identical value while the real corpus has a distribution, and the
#: cell's distance is then dominated by that mismatch rather than by anything
#: being tuned. Density has the same problem with a sharper edge: pinned to one
#: value, a cell's class balance would be set entirely by ``fraction_healthy``,
#: which is not a swept knob, so an unswept quantity would drive the label
#: distribution.
#:
#: Nothing is foreclosed by this: a point-valued design is the degenerate range
#: with ``low == high``, so endpoints are strictly the more expressive choice.
_DISTRIBUTION_PATHS: Mapping[str, str] = {
    "substrate.density_range": "fibrosis_density_range",
    "electrodes.height_mm_range": "electrode_height_mm_range",
}

#: Mixer fields a sweep may address.
#:
#: Excluded, and the rule is the invariant rather than taste: **θ addresses
#: distribution parameters; banks record realized samples.** ``master_seed``
#: selects *which* sample is drawn from a distribution, not the distribution, so
#: it is not a θ parameter — it is the same category as the fibrosis pattern or
#: the noise segment, recorded so a run reproduces and never searched over.
#: ``show_progress`` and ``description`` do not touch the data at all.
_MIX_EXCLUDED = frozenset({"master_seed", "show_progress", "description"})

#: Endpoint names for a two-element range field, so a sweep addresses the two
#: numbers a range *is* rather than a tuple. A GP emulator needs θ as a
#: continuous numeric vector; a pair is not one, and its endpoints are.
_RANGE_KEYS = ("low", "high")

#: Cell model → the function that solves it. The source of truth for which of
#: its fields are chosen and which are derived; see the module docstring.
_CALIBRATORS: Mapping[type, Callable[..., Any]] = {
    AlievPanfilovCellModel: calibrate_aliev_panfilov,
    CourtemancheCellModel: calibrate_courtemanche,
}

#: What to sweep instead of a solved membrane knob. These are calibration
#: *targets*: the physiology a card aims at, which the solve inverts. Sweeping
#: one means a card per sweep point rather than an edit to a spec — see the
#: open question in ``project/phase_1_5_plan.md``.
_DERIVED_ALTERNATIVE = (
    "sweep the calibration target the card solves from — "
    "conduction_velocity_cm_s, or apd90_ms for a model that solves it — "
    "rather than the value it solves to"
)


class UnknownParameterError(ValueError):
    """A path does not name anything on the tunable run."""


class NumericalParameterError(ValueError):
    """A path names a discretisation parameter, not a physiological one.

    **A third exception rather than a stretch of the other two, because the
    reason and the remedy are both different.** A derived field is never
    chosen — a solve produces it. A sim-varying field is chosen, just not here —
    a distribution owns it. This one *is* chosen, *does* survive to the backend,
    and is still refused, on the ground that it does not belong in a parameter
    search at all: it is a property of the numerical scheme, validated by a
    convergence study, not a property of the tissue tuned for realism.

    **The concrete hazard is an estimator gaming discretisation error.** A
    coarse mesh makes conduction velocity read high — the convergence sweep
    measured the local exponent in ``CV ∝ D**n`` falling from 0.625 to 0.542
    toward the continuum ½ as the pitch refined, which is discretisation error
    by signature. Make the pitch searchable and a corpus-similarity objective
    can improve its score by finding a mesh whose error happens to flatter the
    match, rather than by getting the physiology right — exploiting precisely
    the artefact the convergence study existed to remove.

    Refusing costs nothing. A convergence study is run by editing configs and
    comparing runs, which is how this project's own was done, and it never
    needed the pitch to be addressable by path.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path!r} is not a sweepable parameter. {reason}")


class SimVaryingParameterError(ValueError):
    """A path names a value that is redrawn per simulation, so writing it is lost.

    **Distinct from :class:`DerivedParameterError` because the remedy differs.**
    A derived field can never be swept — nothing chooses it, a solve produces
    it. A sim-varying one *can* be swept, just not here: it is drawn from a
    distribution, and the distribution is the thing to write. A caller that
    wanted to fall back to sweeping a range could act on this and not on the
    other, which is why they are not one exception with two messages.

    Carries the offending path and the distribution to write instead.
    """

    def __init__(self, path: str, alternative: str | None) -> None:
        self.path = path
        self.alternative = alternative
        remedy = (
            f"Sweep the distribution it is drawn from: {alternative}."
            if alternative
            else (
                "It is not exposed in the dataset config, so there is no "
                "distribution to sweep — it would need a config key first."
            )
        )
        super().__init__(
            f"{path!r} is rebuilt per simulation from the dataset config, so a "
            f"value written here is discarded before the backend sees it. {remedy}"
        )


class InfeasibleTargetError(ValueError):
    """A target is unreachable: no valid calibration solves it.

    **Raised, never clamped, and swallowing it is a bug.** A sweep point whose
    target cannot be solved is a real fact about the target space — the
    boundary of the feasible region — and the harness's job is to record it as
    an explicit infeasible point. Nudging the value into range instead would
    move a design cell somewhere the design never asked for, and the emulator
    would fit that point as though it were the one requested. Dropping the cell
    silently is just as bad in the other direction: the emulator is then fitted
    on a biased subset of the design and cannot see the boundary at all.

    Carries the offending path, the value, and the underlying reason, so the
    caller can record all three without re-deriving them.
    """

    def __init__(self, path: str, value: object, reason: str) -> None:
        self.path = path
        self.value = value
        self.reason = reason
        super().__init__(
            f"{path!r} = {value!r} is infeasible: {reason} "
            "Record this as an explicit infeasible design point — do not clamp "
            "the value and do not drop the point."
        )


class DerivedParameterError(ValueError):
    """A path names a real field that must not be written directly."""


def _is_leaf(value: Any) -> bool:
    """Is this a scalar a sweep can set?

    Arrays and tuples are excluded rather than enumerated element-wise. They
    are, in every current case, computed from scalars beside them, so the
    sweepable thing is the scalar; and a path grammar with indices in it is a
    much larger commitment than this step should make on a guess.
    """
    return isinstance(value, str | bool | int | float) and not isinstance(value, np.ndarray)


def _is_range(value: Any) -> bool:
    """A two-element numeric tuple — a ``(low, high)`` distribution parameter."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    )


def _solved_field_names(spec: Any) -> frozenset[str]:
    """Fields of ``spec`` its calibration produces rather than accepts."""
    calibrator = _CALIBRATORS.get(type(spec))
    if calibrator is None:
        return frozenset()
    accepted = set(inspect.signature(calibrator).parameters)
    return frozenset(f.name for f in fields(spec) if f.name not in accepted)


def _computed_field_names(spec: Any) -> frozenset[str]:
    """Fields ``spec`` computes from other fields of itself.

    Currently one case, stated on the type it belongs to: a grid's positions
    and pairs come from its layout. Sourced from the constructor that rebuilds
    them for the same reason the solved set is sourced from the calibration —
    the code that produces a field is the honest authority on whether it is
    produced.
    """
    if isinstance(spec, CenteredGrid2D):
        accepted = set(inspect.signature(CenteredGrid2D.at).parameters)
        return frozenset(f.name for f in fields(spec) if f.name not in accepted)
    return frozenset()


def _distribution_field(path: str) -> str | None:
    """The dataset-config field a distribution path names, if it is one."""
    head = ".".join(path.split(".")[:2])
    return _DISTRIBUTION_PATHS.get(head)


def _subject_for(run: TunableRun, root: str, path: str) -> Any:
    """The object a root resolves against, or a refusal naming what is missing."""
    if root == MIX_ROOT:
        if run.mixer is None:
            raise UnknownParameterError(
                f"{path!r} addresses the mixer, but this run has no mixer config — "
                "it is a clean run. Add a mix block before sweeping noise."
            )
        return run.mixer
    if _distribution_field(path) is not None:
        if run.dataset is None:
            raise UnknownParameterError(
                f"{path!r} addresses a distribution, which lives on the dataset "
                "config, and this run carries none. Build the bundle with it to "
                "sweep the range rather than the drawn value."
            )
        return run.dataset
    if root in ("geometry", "cell_model"):
        return getattr(run, root)
    if run.specs is None:
        # The commonest mistake reaches here: a sweep naming the drawn value
        # rather than the range it is drawn from. A harness bundle carries no
        # specs, so without this the answer would be "there are no specs" — true,
        # unhelpful, and silent about the path that would have worked.
        alternative = _SIM_VARYING_ALTERNATIVE.get(path)
        remedy = (
            f" To sweep it, write {alternative} instead — the distribution it is drawn from."
            if alternative
            else ""
        )
        raise UnknownParameterError(
            f"{path!r} names a per-simulation value, and this run carries no "
            f"specs — only a dataset config. {root} is drawn per simulation, so "
            f"there is no realized value until one has been.{remedy}"
        )
    return getattr(run.specs, root)


def _root_of(run: TunableRun, path: str) -> tuple[str, Any, list[str]]:
    head, _, rest = path.partition(".")
    if head not in ROOTS:
        raise UnknownParameterError(
            f"{path!r} does not start with a known root. A path is "
            f"<root>.<field>, where root is one of: {', '.join(ROOTS)}."
        )
    subject = _subject_for(run, head, path)
    if not rest:
        raise UnknownParameterError(
            f"{path!r} names the {head!r} spec itself, not a parameter on it. "
            f"Add a field, e.g. {head}.{fields(subject)[0].name}."
        )
    return head, subject, rest.split(".")


def enumerate_paths(run: TunableRun) -> tuple[str, ...]:
    """Every path these specs can round-trip, derived from the objects.

    **Enumerated from the dataclass fields, never from a list.** A list is a
    second description of the same thing and the two drift; this cannot, because
    it is a reading of the first one. It is what the round-trip test iterates,
    so a spec that gains a field gains coverage automatically.
    """
    found: list[str] = []
    for root in SPEC_ROOTS:
        # Readable but never writable, so they are not sweepable paths and do
        # not belong in a list whose purpose is to say what a sweep can vary.
        if root in SIM_VARYING_ROOTS:
            continue
        spec = getattr(run.specs, root)
        if not is_dataclass(spec):
            continue
        refused = _solved_field_names(spec) | _computed_field_names(spec)
        if isinstance(spec, Patch2DGeometry):
            refused |= _card_coupled_geometry_names()
        for field in fields(spec):
            # The discriminator is structural, not a parameter: changing the
            # string would not change the class, so it is never a path.
            if field.name == "type" or field.name in refused:
                continue
            value = getattr(spec, field.name)
            if isinstance(value, Mapping):
                found.extend(f"{root}.{field.name}.{key}" for key in value)
            elif _is_leaf(value):
                found.append(f"{root}.{field.name}")

    # The card's targets, reached under `cell_model` because that is what they
    # parameterise. `apd90_ms` appears only for a model that solves from one:
    # Courtemanche measures its APD instead, so its cards state no target and
    # there is nothing there to sweep.
    if run.card is not None:
        solvable = _solvable_target_names(run.cell_model)
        for field in fields(ModelTargets):
            if field.name in solvable and getattr(run.card.targets, field.name) is not None:
                found.append(f"cell_model.{TARGETS_FIELD}.{field.name}")

    if run.dataset is not None:
        for dist_path, field_name in _DISTRIBUTION_PATHS.items():
            if _is_range(getattr(run.dataset, field_name)):
                found.extend(f"{dist_path}.{key}" for key in _RANGE_KEYS)

    if run.mixer is not None:
        for field in fields(MixerConfig):
            if field.name in _MIX_EXCLUDED:
                continue
            value = getattr(run.mixer, field.name)
            if _is_range(value):
                found.extend(f"{MIX_ROOT}.{field.name}.{key}" for key in _RANGE_KEYS)
            elif _is_leaf(value):
                found.append(f"{MIX_ROOT}.{field.name}")

    return tuple(found)


def get_value(run: TunableRun, path: str) -> Any:
    """Read the value ``path`` names.

    Reads are permitted on derived fields — reading one is how you record what
    a solve produced. Only writing is refused.
    """
    root, spec, parts = _root_of(run, path)
    field_name = _distribution_field(path)
    if field_name is not None:
        current = getattr(spec, field_name)
        if len(parts) == 1:
            return current
        return current[_range_index(path, parts[1])]

    if root == "cell_model" and parts[0] == TARGETS_FIELD:
        return _target_value(run, path, parts)

    name = parts[0]
    if not any(f.name == name for f in fields(spec)):
        raise UnknownParameterError(_unknown_field_message(run, path, root, spec, name))
    value = getattr(spec, name)

    if len(parts) == 1:
        return value
    if len(parts) == 2 and isinstance(value, Mapping):
        key = parts[1]
        if key not in value:
            raise UnknownParameterError(
                f"{path!r}: {root}.{name} has no key {key!r}. "
                f"Present: {', '.join(sorted(value)) or '(none)'}."
            )
        return value[key]
    if len(parts) == 2 and _is_range(value):
        return value[_range_index(path, parts[1])]
    raise UnknownParameterError(
        f"{path!r} is deeper than this grammar goes. A path is <root>.<field>, "
        "or <root>.<field>.<key> where the field is a mapping or a range."
    )


def _range_index(path: str, key: str) -> int:
    if key not in _RANGE_KEYS:
        raise UnknownParameterError(
            f"{path!r}: a range is addressed by its endpoints, "
            f"{' or '.join(_RANGE_KEYS)}, not {key!r}."
        )
    return _RANGE_KEYS.index(key)


def _target_value(run: TunableRun, path: str, parts: list[str]) -> Any:
    """Read a calibration target off the card."""
    if run.card is None:
        raise UnknownParameterError(
            f"{path!r} addresses a calibration target, but this run carries no "
            "model card, so there is nothing that was solved from one."
        )
    if len(parts) != 2:
        raise UnknownParameterError(
            f"{path!r}: a target path is cell_model.{TARGETS_FIELD}.<name>, where "
            f"name is one of: {', '.join(f.name for f in fields(ModelTargets))}."
        )
    name = parts[1]
    if not any(f.name == name for f in fields(ModelTargets)):
        raise UnknownParameterError(
            f"{path!r}: a card has no target {name!r}. "
            f"Targets: {', '.join(f.name for f in fields(ModelTargets))}."
        )
    value = getattr(run.card.targets, name)
    if value is None:
        raise UnknownParameterError(
            f"{path!r}: this card states no {name!r} target. Courtemanche measures "
            "its action potential duration rather than solving from it, so its "
            "cards record a measurement and there is no target to sweep."
        )
    return value


def set_value(run: TunableRun, path: str, value: Any) -> TunableRun:
    """Return a new :class:`SimulationSpecs` with ``path`` set to ``value``.

    Returns rather than mutates: the specs are frozen, and a sweep wants each
    point to be an independent object it can record beside its results.

    Validation is **not** repeated here. Each spec validates in its own
    ``__post_init__``, so an out-of-range density or an invalid edge is refused
    by the type that owns the rule, in the words that type already uses.
    """
    root, spec, parts = _root_of(run, path)
    field_name = _distribution_field(path)
    if field_name is not None:
        # Checked before every refusal below, because this is the path those
        # refusals point at: `substrate.density` is refused and told to write
        # `substrate.density_range`, and it would be absurd for the remedy to be
        # refused by the same root rule that sent you to it.
        current = getattr(spec, field_name)
        if len(parts) == 1:
            updated_range = tuple(value)
        else:
            endpoints = list(current)
            endpoints[_range_index(path, parts[1])] = value
            updated_range = tuple(endpoints)
        return replace(run, dataset=replace(spec, **{field_name: updated_range}))

    if root == "cell_model" and parts[0] == TARGETS_FIELD:
        # Read first, so an unknown or absent target is refused before anything
        # is rebuilt. Then set the target and re-solve from it, which is the
        # whole reason a target is writable at all.
        _target_value(run, path, parts)
        card = run.card
        assert card is not None  # _target_value refuses a run without one
        name = parts[1]
        solvable = _solvable_target_names(run.cell_model)
        if name not in solvable:
            calibrator = _CALIBRATORS.get(type(run.cell_model))
            solver = calibrator.__name__ if calibrator else "this model's calibration"
            raise DerivedParameterError(
                f"{path!r} is not solved from. {solver} does not take {name!r}, so "
                f"writing it would change what the card claims to aim at without "
                f"changing anything that was solved. For Courtemanche the action "
                f"potential duration emerges from the conductances and is measured "
                f"— sweep cell_model.params.* and record the duration it produces. "
                f"Solvable targets here: {', '.join(sorted(solvable)) or '(none)'}."
            )
        try:
            targets = replace(card.targets, **{name: value})
        except ValueError as exc:
            # ModelTargets validates on construction, so an out-of-domain target
            # is refused before the calibration ever sees it. Same fact, same
            # handling: it is an infeasible design point, not a bad call.
            raise InfeasibleTargetError(path, value, f"{exc}") from exc
        staged = replace(run, card=replace(card, targets=targets))
        return _resolved_card(staged, path=path, value=value)

    name = parts[0]
    if not any(f.name == name for f in fields(spec)):
        raise UnknownParameterError(_unknown_field_message(run, path, root, spec, name))

    if name == "type":
        raise DerivedParameterError(
            f"{path!r} is the strategy discriminator, not a parameter. Writing it "
            "would relabel the spec without changing its class, so the backend "
            "would dispatch on one thing and integrate another. Choose a "
            "different spec instead."
        )
    if name in _solved_field_names(spec):
        raise DerivedParameterError(
            f"{path!r} is solved by {_CALIBRATORS[type(spec)].__name__}, not chosen. "
            f"A model card records it and re-verifies it on every load, so a bank "
            f"written with it overridden would claim one parameterisation and "
            f"contain another. Instead: {_DERIVED_ALTERNATIVE}."
        )
    if name in _computed_field_names(spec):
        raise DerivedParameterError(
            f"{path!r} is computed from the other fields of {type(spec).__name__} "
            f"and is what the backend actually reads. Set the layout instead — "
            f"{root}.n_rows, {root}.n_cols, {root}.spacing_mm, {root}.height_mm — "
            "and it is rebuilt from them."
        )
    if isinstance(spec, Patch2DGeometry) and name in _card_coupled_geometry_names():
        raise NumericalParameterError(
            path,
            "The model card's diffusion was solved at this mesh pitch, by "
            "inverting a conduction velocity through an anchor measured at it, "
            "so writing the pitch here would leave the card solved for a mesh "
            "the run no longer uses — with nothing downstream able to tell. "
            "It is also not a parameter to search: mesh pitch is a numerical "
            "choice validated by a convergence study, not a tissue property "
            "tuned for realism, and a coarse mesh makes conduction velocity run "
            "high. Left searchable, a similarity objective could improve its "
            "score by picking a mesh whose discretisation error flatters the "
            "match. To study convergence, vary geometry.dr_mm in the config "
            "across separate runs and compare them, re-solving the card at each "
            "pitch — that is how the pitch was chosen in the first place.",
        )
    # Last of the three refusals, and deliberately last: the two above are
    # field-specific and say something more precise than "this root does not
    # survive", so a computed field keeps its own message rather than being
    # swallowed by the coarser rule.
    if root in SIM_VARYING_ROOTS:
        raise SimVaryingParameterError(path, _SIM_VARYING_ALTERNATIVE.get(path))

    current = getattr(spec, name)
    if len(parts) == 1:
        updated = replace(spec, **{name: value})
    elif len(parts) == 2 and isinstance(current, Mapping):
        key = parts[1]
        if key not in current:
            raise UnknownParameterError(
                f"{path!r}: {root}.{name} has no key {key!r}. "
                f"Present: {', '.join(sorted(current)) or '(none)'}. A sweep may "
                "vary an existing scaling, not introduce one — an unknown name "
                "would be silently ignored by the backend."
            )
        updated = replace(spec, **{name: {**current, key: value}})
    elif len(parts) == 2 and _is_range(current):
        index = _range_index(path, parts[1])
        endpoints = list(current)
        endpoints[index] = value
        updated = replace(spec, **{name: tuple(endpoints)})
    else:
        raise UnknownParameterError(
            f"{path!r} is deeper than this grammar goes. A path is <root>.<field>, "
            "or <root>.<field>.<key> where the field is a mapping or a range."
        )

    if root == MIX_ROOT:
        return replace(run, mixer=updated)

    # Written to both halves of the bundle where both exist. The dataset config
    # is what a generation run reads; the specs are the record of one simulation
    # drawn from it, and letting them disagree is how a bank comes to describe
    # something other than what ran.
    written = run
    if run.dataset is not None and root in ("geometry", "cell_model"):
        written = replace(written, dataset=replace(run.dataset, **{root: updated}))
    if run.specs is not None:
        written = replace(written, specs=_normalised(replace(run.specs, **{root: updated})))
    if root == "cell_model":
        # A conductance scaling is an *input* to the calibration, so changing
        # one changes what the solve returns. Leaving the card's solved values
        # beside a membrane they were not solved for is the same corruption a
        # direct write to `diffusion` would cause, arriving indirectly.
        return _resolved_card(written, path=path, value=value)
    return written


def _card_coupled_geometry_names() -> frozenset[str]:
    """Geometry fields a calibration is solved *against*.

    Mechanical, from the calibrator signatures, like every other classification
    in this module — not a hand-listed path. Today it returns exactly
    ``{"dr_mm"}``: it is the only geometry field either calibrator accepts and
    the only one a conduction-velocity anchor is keyed on. ``size_mm``,
    ``fiber_angle_rad`` and ``anisotropy_ratio`` appear in neither, which is why
    they stay writable — they are tissue properties the solve never sees.

    If a calibration later takes another geometry field, that field joins this
    set on the same commit, with nobody remembering to add it.
    """
    accepted: set[str] = set()
    for calibrator in _CALIBRATORS.values():
        accepted |= set(inspect.signature(calibrator).parameters)
    return frozenset(f.name for f in fields(Patch2DGeometry) if f.name in accepted)


def _solvable_target_names(spec: Any) -> frozenset[str]:
    """Targets the model's calibration actually solves from.

    The same signature rule as everywhere else in this module, applied to the
    other side of the card. Aliev-Panfilov's calibration takes both
    ``conduction_velocity_cm_s`` and ``apd90_ms``, so both are sweepable.
    Courtemanche's takes only the first: its action potential duration emerges
    from twelve interacting currents and is **measured**, never solved.

    So an APD target on a Courtemanche card is a record of what a conductance
    sweep was aimed at, not an input to anything. Writing it would change that
    record and re-solve nothing — the card would then claim a duration no step
    of the pipeline had pursued.
    """
    calibrator = _CALIBRATORS.get(type(spec))
    if calibrator is None:
        return frozenset()
    accepted = set(inspect.signature(calibrator).parameters)
    return frozenset(f.name for f in fields(ModelTargets) if f.name in accepted)


def _chosen_inputs(spec: Any) -> dict[str, object]:
    """The calibration arguments this spec already carries.

    The mirror image of :func:`_solved_field_names`, from the same signature:
    whatever the calibrator accepts *and* the spec holds is an input that has to
    be carried into a re-solve, or re-solving would quietly reset it to a
    default. ``eps`` for Aliev-Panfilov, the conductance scalings for
    Courtemanche.
    """
    calibrator = _CALIBRATORS.get(type(spec))
    if calibrator is None:
        return {}
    accepted = set(inspect.signature(calibrator).parameters)
    return {f.name: getattr(spec, f.name) for f in fields(spec) if f.name in accepted}


def _ensure_stable(spec: Any, *, dr_model_units: float, path: str, value: object) -> None:
    """Refuse a solve whose step exceeds the explicit-scheme bound.

    **THE CONDUCTION-VELOCITY AXIS HAS NO INFEASIBLE REGION, and this was
    expected to have one.** The plan asserted that a target could violate the
    stability bound and that such failures would make the infeasible parts of
    the target space explicit. Measured, that is false. Conduction velocities
    from 1e-6 to 1e12 cm/s were probed against both calibrators and **none**
    produced a step exceeding its bound, because the calibrators *derive* ``dt``
    from that bound rather than checking a proposal against it. The solve always
    returns a valid step; an extreme target buys an absurdly small one, not an
    invalid one.

    Stated flatly because a later reader would otherwise design boundary
    handling for a boundary that is not there — an emulator's feasible-region
    logic, a sampler that expects rejections, a design that reserves cells for
    failures that never come. Along the CV axis every target succeeds.

    The guard is still correct to exist, for two reasons that are not that one:
    non-positive targets *do* raise, from the domain checks; and a calibrator
    added later — or a backend with a genuine hard limit, such as an implicit
    scheme or a fixed-step solver — has no reason to know this invariant unless
    something enforces it. What the guard must not do is imply a boundary along
    an axis that has none.

    Tested directly against a hand-built spec rather than through a target,
    because no target reaches it: a test that swept targets hunting for
    divergence would pass without ever entering this branch.
    """
    limit = spec.stability_limit(dr_model_units=dr_model_units)
    if spec.dt_model_units > limit:
        raise InfeasibleTargetError(
            path,
            value,
            f"the solve returned dt={spec.dt_model_units:.6g} against a stability "
            f"bound of {limit:.6g} at dr={dr_model_units}, so the integration "
            "would diverge rather than produce a slow but valid run.",
        )


def _resolved_card(run: TunableRun, *, path: str, value: object) -> TunableRun:
    """Re-run the calibration and put its answer on both the card and the specs.

    **This is what makes a target write legal.** A card records what its
    calibration solved and re-verifies it on load, so a sweep cannot simply
    write a new target and leave the old solved values beside it. Re-solving
    keeps the card's own invariant true at every sweep point, which is why the
    target is addressable at all.

    ``measured`` is cleared. It described a simulation run at the *previous*
    solve, and carrying it forward would attach a real measurement to a
    parameterisation that never produced it — a card that lies in the specific
    way the type exists to prevent. ``None`` is a state the card format already
    has: solved but not yet simulated.
    """
    card = run.card
    if card is None:
        return run

    inputs = _chosen_inputs(run.cell_model)
    dr_mm = float(run.geometry.dr_mm)
    dr_model_units = run.effective_dr_model_units

    try:
        if isinstance(run.cell_model, AlievPanfilovCellModel):
            if card.targets.apd90_ms is None:
                raise ValueError(
                    "Aliev-Panfilov solves its time unit from an apd90_ms target "
                    "and this card states none."
                )
            solved: CellModelSpec = calibrate_aliev_panfilov(
                conduction_velocity_cm_s=card.targets.conduction_velocity_cm_s,
                apd90_ms=card.targets.apd90_ms,
                dr_mm=dr_mm,
                dr_model_units=dr_model_units,
                **inputs,  # type: ignore[arg-type]
            )
        elif isinstance(run.cell_model, CourtemancheCellModel):
            solved = calibrate_courtemanche(
                conduction_velocity_cm_s=card.targets.conduction_velocity_cm_s,
                dr_mm=dr_mm,
                dr_model_units=dr_model_units,
                **inputs,  # type: ignore[arg-type]
            )
        else:
            raise ValueError(
                f"no calibration is registered for cell model "
                f"{run.cell_model.type!r}, so its targets cannot be swept."
            )
    except InfeasibleTargetError:
        raise
    except ValueError as exc:
        raise InfeasibleTargetError(path, value, f"the calibration refused it: {exc}") from exc

    _ensure_stable(solved, dr_model_units=dr_model_units, path=path, value=value)

    resolved = replace(run, card=replace(card, solved=solved, measured=None))
    if run.specs is not None:
        resolved = replace(resolved, specs=replace(run.specs, cell_model=solved))
    if run.dataset is not None:
        resolved = replace(resolved, dataset=replace(run.dataset, cell_model=solved))
    return resolved


def _normalised(specs: SimulationSpecs) -> SimulationSpecs:
    """Rebuild whatever is computed from what was just written.

    Applied after **every** write rather than only the ones known to matter,
    because the couplings are not local: an electrode grid's positions depend on
    its own layout *and* on the patch it is centred in, so ``geometry.size_mm``
    moves them just as ``electrodes.height_mm`` does. One idempotent pass is
    easier to keep correct than a set of rules about which writes need it.
    """
    electrodes = specs.electrodes
    if isinstance(electrodes, CenteredGrid2D):
        electrodes = CenteredGrid2D.at(
            geometry=specs.geometry,  # type: ignore[arg-type]
            n_rows=electrodes.n_rows,
            n_cols=electrodes.n_cols,
            spacing_mm=electrodes.spacing_mm,
            height_mm=electrodes.height_mm,
        )
        return replace(specs, electrodes=electrodes)
    return specs


def _unknown_field_message(run: TunableRun, path: str, root: str, spec: Any, name: str) -> str:
    """Suggest from what the object *has*, not from what a sweep may write.

    Those two sets differ now that some roots are read-only, and the readable
    one is right here: this error means "no such field", so answering it with
    the sweepable subset would tell someone their correctly-spelled path does
    not exist. Whether it can be *written* is a separate question, with its own
    error and its own remedy.
    """
    readable = [f.name for f in fields(spec) if f.name != "type"]
    note = f" (read-only: {root} is rebuilt per simulation)" if root in SIM_VARYING_ROOTS else ""
    listed = ", ".join(f"{root}.{n}" for n in readable) or "(none)"
    return (
        f"{path!r}: {type(spec).__name__} has no field {name!r}. Fields on {root}{note}: {listed}."
    )
