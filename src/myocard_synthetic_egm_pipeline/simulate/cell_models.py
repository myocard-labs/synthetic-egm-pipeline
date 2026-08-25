"""Cell-model specs — the fifth strategy spec: the membrane kinetics each node runs.

A **cell model** is what each node of the mesh does on its own: the membrane
kinetics. It is a separate axis from the backend, because Courtemanche runs on
Finitewave *and* on TorchCor, and separate from geometry, because fibre
architecture is tissue structure rather than membrane behaviour.

Why these parameters live here and nowhere else
-----------------------------------------------
The test is: **does this survive a cell-model swap?**

- ``apd90_ms`` survives — it is a fact about atrium, so it is a *target*
  (:mod:`~myocard_synthetic_egm_pipeline.simulate.calibration`).
- ``anisotropy_ratio`` survives — it is fibre architecture, and it already lives
  on :class:`~...specs.Patch2DGeometry`.
- ``eps`` does **not** survive. Courtemanche has no such parameter.
- ``time_unit_ms`` does **not** survive either, and for a deeper reason:
  Aliev-Panfilov is **dimensionless**, so it needs a mapping to milliseconds at
  all. Courtemanche is already in milliseconds. The very *existence* of that
  knob is a property of one model.

An earlier calibration put ``eps``, ``diffusion``, ``dt_model_units`` and
``ap_time_unit_ms`` on ``RunConfig`` — the config object shared by *every*
backend — and that is the arrangement this module exists to correct. A config
object with no place for model-specific parameters gets them anyway, so a
Courtemanche run would have carried an ``eps`` field meaning nothing to it and
a TorchCor backend would have carried both models' fields at once. The rule
that follows is: **a parameter belongs to the narrowest thing that can change
it independently**, and for membrane kinetics that is the cell model.

Time conversion belongs to the model, not the caller
----------------------------------------------------
Callers used to write ``duration_ms / config.ap_time_unit_ms`` to reach solver
time. That is an Aliev-Panfilov idiom leaking into the runner, and it silently
breaks for a dimensional model. :meth:`CellModelSpec.ms_to_model_time` moves the
conversion behind the spec, where Courtemanche can answer *"the same number"*
and nothing upstream has to know why.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Measured properties of Aliev-Panfilov as Finitewave integrates it
# ---------------------------------------------------------------------------

AP_EPS_PUBLISHED: float = 0.002
"""Aliev-Panfilov's own baseline recovery rate.

``gamma`` in the Göktepe/Kuhl notation, ``gamma_0`` in the EP-PINNs papers,
where it is described as the parameter controlling action potential duration.
**Finitewave ships 0.01**, five times this — a defensible default for a package
whose examples run for 10 dimensionless time units and claim no physical time
scale, but we inherited it unnoticed, and it is half of why APD came out at
51 ms.
"""

MODEL_UNIT_APD90: float = 38.53
"""APD90 in model time units at :data:`AP_EPS_PUBLISHED`. **Measured**, not read
from a paper — Finitewave assigns no physical units anywhere.

Measured 2026-08-13 on clean tissue as the interval from 10 % upstroke to 90 %
repolarisation. Across the sweep: 38.53 at eps 0.002, 33.03 at 0.004, 25.92 at
0.010, 20.76 at 0.020 — a weak response, roughly ``eps ** -0.27``. Independent
of diffusion (25.89 / 25.89 / 25.89 / 25.88 / 25.87 over a 16-fold sweep), which
is what makes the two-target solve separable.
"""

MODEL_UNIT_CV: float = 1.6328
"""Along-fibre conduction velocity in model space units per model time unit at
``D = 1`` and :data:`AP_EPS_PUBLISHED`. **Measured.**

``CV`` scales as ``sqrt(D)`` — measured 0.797 / 1.612 / 3.248 at D = 0.25 / 1 / 4
against a sqrt-law prediction of 0.5 / 1.0 / 2.0 in ratio.
"""

# ---------------------------------------------------------------------------
# Measured properties of Courtemanche as Finitewave integrates it
# ---------------------------------------------------------------------------

CRN_REFERENCE_DIFFUSION: float = 0.154
"""Finitewave's shipped ``Courtemanche.D_model``, in mm^2/ms.

Used only as the point the CV measurement below was taken at; the operative
value is solved from a conduction-velocity target, exactly as Aliev-Panfilov's
is. Note it is three orders of magnitude below Aliev-Panfilov's calibrated
``diffusion`` (~7.8) — the two models carry different units, which is why the
number cannot be carried across a model swap and why it lives on the spec.
"""

CRN_REFERENCE_CV_CM_S: float = 57.400
"""Along-fibre conduction velocity in cm/s at :data:`CRN_REFERENCE_DIFFUSION`,
control conductances and :data:`CRN_CALIBRATION_DR_MM`. **Measured**, like
:data:`MODEL_UNIT_CV` — Finitewave publishes no such number.

Measured 2026-08-15 on clean tissue, along the fibres, on the 40 mm production
patch, by the same least-squares fit over the central 30-70 % of the mesh that
fills a card's ``measured`` block. The ``sqrt(D)`` law it is extrapolated with
was checked rather than assumed — see the sweep recorded in
``models/courtemanche_control.yaml``.
"""

CRN_CALIBRATION_DR_MM: float = 0.25
"""Mesh pitch :data:`CRN_REFERENCE_CV_CM_S` was measured at.

**Not a free parameter of the solve.** Discretisation widens the upstroke
relative to the mesh, so the measured CV constant is a property of the model
*and this pitch*; using it at another pitch would be reading a number off the
wrong axis — which this project has already done once, when a conduction
velocity measured *across* the fibres was recorded under a longitudinal label.

Courtemanche is the model where the pitch matters most: its upstroke is
~0.59 ms wide, so at 80 cm/s the wavefront spans ~0.47 mm, which is about 1.9
cells at 0.25 mm where monodomain practice wants 5-10. Until a mesh-convergence
sweep re-measures this constant at a finer pitch, a card solved here is a card
valid at 0.25 mm only — which :func:`calibrate_courtemanche` enforces rather
than trusts, because a CV-solve will otherwise absorb the discretisation error
into ``diffusion`` and hit its target with a number that is no longer physical.
"""

CRN_MAX_DT_MS: float = 0.02
"""Ceiling on the integration step, in ms. **Measured**, not inherited.

The explicit-diffusion CFL bound is not the binding constraint for Courtemanche:
at ``D = 0.154`` and ``dr = 0.25`` it permits ``dt = 0.101``, while the fast
sodium current needs far less. Finitewave integrates the gating variables
Rush-Larsen, so the model does not diverge at a coarse step — it quietly
mis-reports the upstroke — which is the one observable an ionic model was
added to get right, since electrogram amplitude scales with ``dV/dt``.

Chosen from a convergence check of the pinned single-cell protocol: halving the
step to 0.01 ms moves every one of the five action-potential properties by
<= 1 %, so 0.02 is on the plateau and halving again was not run. The numbers are
in ``models/courtemanche_control.yaml``. This is a *ceiling*: the CFL bound still
applies and :meth:`CourtemancheCellModel.stability_limit` returns whichever is
smaller.
"""

CRN_PACING_BCL_MS: float = 1000.0
"""Basic cycle length of the single-cell protocol the published vector is read at.

**Half of a two-part pin, and neither half is optional** — see
:data:`CRN_PACING_BEATS`.
"""

CRN_PACING_BEATS: int = 50
"""Number of paced beats before the properties below are read.

**Courtemanche never reaches steady state** (Wilhelms 2012 §3.1): APD90 falls to
83 % of its first-beat value over the first 16 minutes of pacing, and APD50
falls 42 % over 20 minutes at BCL 1 s. So "Courtemanche's APD90" is not a
number — the same model legitimately reads 295 ms or ~245 ms depending on when
you look, and a test that pins only the cycle length would drift with whatever
run length it happened to use. Wilhelms paces **50 s at BCL 1 s**, so that is
what the reference vector below means and what
:func:`~...backends.finitewave.measure.measure_single_cell` runs by default.
"""

CRN_STIMULUS_AMPLITUDE_MV_PER_MS: float = 20.0
"""Single-cell stimulus, as the ``dV/dt`` Finitewave's ``StimCurrent`` adds.

20 mV/ms for 2 ms is the textbook Courtemanche protocol — 2 nA into
``Cm = 100 pF`` — and about 1.9x the diastolic threshold measured here
(capture between 10 and 11 mV/ms at this duration).

**It is pinned because ``dV/dt max`` depends on it**, and not weakly: measured
165 / 195 / 218 / 227 V/s at 12 / 15 / 21 / 30 mV/ms on the first beat, because
the maximum falls inside the 2 ms stimulus window rather than after it. That
sensitivity is a property of the protocol, not of the model, which is precisely
why the protocol is part of the fixture instead of an incidental choice.
"""

CRN_STIMULUS_DURATION_MS: float = 2.0
"""Duration of the single-cell stimulus. See
:data:`CRN_STIMULUS_AMPLITUDE_MV_PER_MS`."""

WILHELMS_2012_CRN_CONTROL: Mapping[str, float] = MappingProxyType(
    {
        "amplitude_mv": 110.11,
        "rmp_mv": -81.04,
        "apd50_ms": 165.16,
        "apd90_ms": 294.83,
        "dvdt_max_v_s": 186.58,
    }
)
"""Control Courtemanche at BCL 1 s, from **Wilhelms et al., Front Physiol
2012;3:487**, Table 1, column C.

**This is an independent reimplementation, not the original paper's own table.**
Courtemanche/Ramirez/Nattel 1998 is paywalled and its own table could not be
retrieved. Matching Wilhelms is therefore a claim that our
implementation agrees with *someone else's* — arguably the stronger check, since
it is a five-element vector rather than one number and a second implementation
is a genuine independent replicate — but it is a **different** claim from
"reproduces the original", and the card says which.

Only meaningful together with :data:`CRN_PACING_BEATS` and
:data:`CRN_PACING_BCL_MS`.
"""

DT_SAFETY_FACTOR: float = 0.9
"""How far inside the stability bound to place ``dt``.

The bound is the marginal case, and marginal is no place to run an explicit
scheme: violating it does not raise, it silently diverges. 10 % of margin costs
10 % of runtime and removes a failure mode invisible in the output.
"""


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@runtime_checkable
class CellModelSpec(Protocol):
    """What one node of the mesh does. The fifth strategy spec.

    Mirrors the shape of the other four — a ``type`` discriminator the backend
    dispatches on, and no backend imports — so a backend that cannot run a given
    model refuses it by name rather than by duck-typing its way into nonsense.
    """

    @property
    def type(self) -> str:
        """Discriminator the backend dispatches on.

        A read-only ``@property``, matching the other four strategy Protocols:
        the bare ``type: str`` form demands a *settable* attribute, which frozen
        dataclass concretes cannot supply. Declaring it the other way is why
        ``GeometrySpec`` and friends accept frozen concretes cleanly, and
        writing it wrong here made ``AlievPanfilovCellModel`` silently fail to
        satisfy its own Protocol.
        """
        ...

    @property
    def dt_model_units(self) -> float:
        """Integration step, in this model's own time units.

        On the Protocol rather than in each backend's narrowed branch because
        every consumer of a cell model needs it and none of them cares which
        model it is: the runner sizes a capture with it, the backend checks it
        against a stability bound, the bank records it. What the *unit* is
        varies (milliseconds for a dimensional model, arbitrary for a
        dimensionless one), which is what :meth:`ms_to_model_time` is for.

        A read-only ``@property`` for the same reason ``type`` is — a frozen
        dataclass field satisfies it, a settable declaration would not.
        """
        ...

    def ms_to_model_time(self, duration_ms: float) -> float:
        """Convert a physical duration to this model's own time units."""
        ...

    def stability_limit(self, *, dr_model_units: float, dimensions: int = 2) -> float:
        """Largest ``dt`` this model tolerates on a grid of the given pitch.

        The bound is a **joint** property — diffusion belongs to the model, the
        grid step to the backend — so it is a method taking the half the model
        does not own. Which bound binds is the model's business: an explicit
        diffusion CFL condition for a smooth reaction term, something tighter
        where a stiff current sets the pace.
        """
        ...

    def to_metadata(self) -> dict[str, Any]:
        """Flatten for ``backend_metadata``, so a bank records what ran."""
        ...


@dataclass(frozen=True)
class AlievPanfilovCellModel:
    """Aliev-Panfilov 1996, parameterised for a physiological target.

    Every field here is **derived** by :func:`calibrate_aliev_panfilov` rather
    than chosen, which is the point: the June 2026 calibration set one of them
    by hand to hit a conduction-velocity target and destroyed action potential
    duration doing it, because ``time_unit_ms`` appears in the two physical
    observables in *opposite* senses.
    """

    time_unit_ms: float
    """K, milliseconds per model time unit. Exists only because AP is
    dimensionless — a dimensional model has no analogue."""

    diffusion: float
    """``model.D_model``. Sets absolute conduction velocity as ``sqrt(D)``.

    One of three multipliers Finitewave applies to the same coefficient
    (``D_model`` x ``stencil.D_al`` x ``tissue.conductivity``); we drive this one
    only, so the stencil stays a pure anisotropy *shape* knob."""

    eps: float
    """Baseline recovery rate; sets APD in model units. Held at the published
    value rather than solved — see :func:`calibrate_aliev_panfilov`."""

    dt_model_units: float
    """Integration step.

    Scheme-flavoured rather than membrane-flavoured, but it is **solved jointly
    with** ``diffusion`` and is meaningless apart from it, so it travels with the
    model and the backend validates it against its own stencil."""

    type: str = "aliev_panfilov"

    def __post_init__(self) -> None:
        for name in ("time_unit_ms", "diffusion", "eps", "dt_model_units"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    def ms_to_model_time(self, duration_ms: float) -> float:
        """Physical ms to dimensionless model time."""
        return duration_ms / self.time_unit_ms

    def stability_limit(self, *, dr_model_units: float, dimensions: int = 2) -> float:
        """Largest stable ``dt`` for an explicit scheme on a regular grid.

        Lives on the model because it depends on ``diffusion``; takes the grid
        step as an argument because that is the *backend's* business. The seam
        between the two axes is exactly here.
        """
        return (dr_model_units * dr_model_units) / (2.0 * dimensions * self.diffusion)

    def to_metadata(self) -> dict[str, Any]:
        """Provenance for ``backend_metadata``.

        ``time_unit_ms`` is **not** here. It used to be, and the bank's
        cell-model object was rebuilt from that copy — a round trip through
        the backend's provenance bag, since replaced by the bank reading the
        spec directly. It never reached disk through this path anyway
        (:func:`~...bank_config.backend_model` excludes it from
        ``params`` precisely because it belongs to the cell model), so
        what remains here is what genuinely has nowhere else to go.
        """
        return {
            "cell_model_type": self.type,
            "ap_diffusion": float(self.diffusion),
            "ap_membrane_eps": float(self.eps),
            "ap_dt_model_units": float(self.dt_model_units),
        }


@dataclass(frozen=True)
class CourtemancheCellModel:
    """Courtemanche-Ramirez-Nattel 1998, the human-atrial ionic model.

    **Deliberately shorter than its Aliev-Panfilov sibling.** Aliev-Panfilov is
    phenomenological, so every knob that produces a physiological observable has
    to be solved for. Courtemanche's membrane is already human atrium: its
    conductances are measured quantities, not fitting parameters, and its time
    axis is already milliseconds. What is left to solve is the *tissue* half —
    how fast the wave travels — which is diffusion, not membrane.

    So the split is:

    - ``diffusion`` is **solved** from a conduction-velocity target, exactly as
      it is for Aliev-Panfilov (``CV ~ sqrt(D)`` is a property of the diffusion
      operator and survives the model swap).
    - APD90 is **measured, never solved**. There is no ``time_unit_ms`` to
      divide by and no closed-form inverse from the ionic equations; it is what
      the conductances produce. A card records it under ``measured:``.
    - ``params`` are **chosen**, not derived — see below.
    """

    diffusion: float
    """``model.D_model`` in mm^2/ms. Solved by :func:`calibrate_courtemanche`.

    Physical units, unlike Aliev-Panfilov's, which is why the two models' values
    differ by three orders of magnitude and why neither can be read as a
    correction of the other."""

    dt_model_units: float
    """Integration step. **The model time unit is the millisecond**, so this is
    a step in ms and no conversion applies — the same identity that makes
    :meth:`ms_to_model_time` the identity function.

    The field keeps the sibling's name because the backend and the bank schema
    ask every cell model for "the step in its own time units", and answering
    "the same as ms" is the honest answer rather than a missing one."""

    params: Mapping[str, float] = field(default_factory=dict)
    """Conductance scalings applied on top of the model's published defaults,
    as ``{name: multiplier}`` — ``{}`` for control.

    **Chosen, not derived**, which is why they are a free-form mapping rather
    than solved fields: which conductances are worth varying is an experimental
    question (the schema's ``CourtemancheCellModel.params`` is deliberately open
    for the same reason, and a parameter sweep's theta-spec points into it by
    path rather than by field name). The
    backend owns the mapping from these names to its solver's attributes and
    refuses a name it cannot apply — a scaling that silently did nothing would
    be the ``anisotropy_ratio`` no-op again — a knob that spent the life of the
    project silently doing nothing because it was assigned to an object that
    never read it.

    An AF-remodelled card fills this in by sweeping remodelling severity along
    the published van Wagoner / Bosch / Dobrev axis; the shipped card is
    control, so its mapping is empty."""

    type: str = "courtemanche"

    def __post_init__(self) -> None:
        for name in ("diffusion", "dt_model_units"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for key, value in self.params.items():
            if value <= 0:
                raise ValueError(
                    f"conductance scaling {key!r} must be positive; got {value}. "
                    "A zero knocks the current out entirely and a negative one "
                    "reverses it — neither is a remodelling severity."
                )
        # Frozen means the *field* cannot be rebound; without this the mapping
        # behind it could still be edited in place, and a card's parameters are
        # provenance.
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def ms_to_model_time(self, duration_ms: float) -> float:
        """The identity. **This is the seam** :class:`CellModelSpec` exists for.

        Courtemanche integrates in milliseconds, so there is nothing to convert.
        The method is not decoration: callers used to write
        ``duration_ms / config.ap_time_unit_ms`` inline, and the Courtemanche
        version of that expression is a division by a constant that does not
        exist — which in Python is not an error but a ``0.0`` or an
        ``AttributeError`` three call frames from the cause, depending on what
        happened to be in scope. Returning the argument unchanged is asserted in
        the tests for exactly that reason.
        """
        return duration_ms

    def stability_limit(self, *, dr_model_units: float, dimensions: int = 2) -> float:
        """Largest usable ``dt``, in ms. **Two bounds, and the CFL one rarely wins.**

        Aliev-Panfilov's limit is purely the explicit-diffusion CFL condition:
        its reaction term is smooth, so the diffusion operator is what
        destabilises. Courtemanche's fast sodium current is stiff — the upstroke
        is ~0.59 ms — and Finitewave integrates the gating variables
        Rush-Larsen, which keeps a coarse step from *diverging* while it quietly
        flattens the upstroke. A limit that reported only the CFL bound would
        therefore be reporting the slack constraint: 0.101 ms at the shipped
        mesh, five times the step the upstroke actually needs.

        Returning the smaller of the two keeps the backend's one-line check
        ("is dt inside the limit?") meaningful for both models without the
        backend having to know which bound bit.
        """
        cfl = (dr_model_units * dr_model_units) / (2.0 * dimensions * self.diffusion)
        return min(cfl, CRN_MAX_DT_MS)

    def to_metadata(self) -> dict[str, Any]:
        """Provenance for ``backend_metadata``.

        The conductance scalings are flattened one key per scaling rather than
        nested, because ``backend_metadata`` lands in HDF5 attributes where a
        nested mapping has no representation. ``crn_`` prefixes keep them
        distinguishable from Aliev-Panfilov's ``ap_`` block in a bank that
        someone is comparing the two models, which is the whole point of
        shipping both.
        """
        meta: dict[str, Any] = {
            "cell_model_type": self.type,
            "crn_diffusion": float(self.diffusion),
            "crn_dt_ms": float(self.dt_model_units),
        }
        meta.update({f"crn_param_{key}": float(value) for key, value in self.params.items()})
        return meta


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def calibrate_aliev_panfilov(
    *,
    conduction_velocity_cm_s: float,
    apd90_ms: float,
    dr_mm: float,
    dr_model_units: float,
    eps: float = AP_EPS_PUBLISHED,
    dimensions: int = 2,
) -> AlievPanfilovCellModel:
    """Solve AP knobs from physiological targets. Analytic, no simulation.

    .. math::
        K = \\frac{\\text{APD}_{ms}}{\\text{APD}^{*}(\\varepsilon)}
        \\qquad
        D = \\left(\\frac{\\text{CV}\\,K}{c^{*}\\,s}\\right)^{2}
        \\qquad
        \\Delta t = \\frac{\\alpha \\, \\Delta r^{2}}{2 \\, \\text{dim} \\, D}

    with :math:`s = \\delta / \\Delta r` millimetres per model space unit.

    ``eps`` is **held, not solved.** It is the free direction in a
    four-unknowns-from-three-targets system, and spending it on the published
    value is also the cheapest choice: a larger APD in model units means a
    smaller ``K``, hence a smaller ``D`` and a larger ``dt``. It is additionally
    the direction the data cannot constrain, since fixing the repolarisation
    shortcut makes APD unobservable inside the analysis window by
    construction.

    **Deliberately analytic.** An iterate-and-measure loop would hit the targets
    exactly but costs seconds to minutes, so it could not run as a load-time
    guard — the guard would move to CI, protecting only us rather than every run
    on every machine. The residual (CV lands a few percent high, because
    discretization widens the upstroke relative to a fixed mesh) is recorded in
    the model card's ``measured`` block instead of hidden.
    """
    if conduction_velocity_cm_s <= 0:
        raise ValueError("conduction_velocity_cm_s must be positive.")
    if apd90_ms <= 0:
        raise ValueError("apd90_ms must be positive.")
    if dr_mm <= 0 or dr_model_units <= 0:
        raise ValueError("dr_mm and dr_model_units must be positive.")
    if dimensions < 1:
        raise ValueError("dimensions must be >= 1.")
    if eps != AP_EPS_PUBLISHED:
        # The MODEL_UNIT_* constants were measured AT the published eps. Using
        # them elsewhere would be reading a number off the wrong axis, which
        # this project has already done once — a conduction velocity measured
        # across the fibres and recorded under a longitudinal label.
        raise ValueError(
            f"the calibration constants were measured at eps={AP_EPS_PUBLISHED}; "
            f"got eps={eps}. Re-measure MODEL_UNIT_APD90 and MODEL_UNIT_CV at the "
            "new value first — see investigations/ap_model_calibration.md 2.1."
        )

    time_unit_ms = apd90_ms / MODEL_UNIT_APD90
    space_unit_mm = dr_mm / dr_model_units
    cv_mm_per_ms = conduction_velocity_cm_s / 100.0

    sqrt_diffusion = cv_mm_per_ms * time_unit_ms / (MODEL_UNIT_CV * space_unit_mm)
    diffusion = sqrt_diffusion * sqrt_diffusion

    limit = (dr_model_units * dr_model_units) / (2.0 * dimensions * diffusion)
    dt_model_units = DT_SAFETY_FACTOR * limit

    if not math.isfinite(dt_model_units) or dt_model_units <= 0:
        raise ValueError(
            f"derived dt is not usable ({dt_model_units}). A very high conduction "
            "velocity or a very short APD drives diffusion up and the step to zero."
        )

    return AlievPanfilovCellModel(
        time_unit_ms=time_unit_ms,
        diffusion=diffusion,
        eps=eps,
        dt_model_units=dt_model_units,
    )


def calibrate_courtemanche(
    *,
    conduction_velocity_cm_s: float,
    dr_mm: float,
    dr_model_units: float,
    params: Mapping[str, float] | None = None,
    dimensions: int = 2,
) -> CourtemancheCellModel:
    """Solve Courtemanche's tissue knobs from a conduction-velocity target.

    .. math::
        D = D^{*} \\left(\\frac{\\text{CV}}{\\text{CV}^{*}}\\right)^{2}
        \\qquad
        \\Delta t = \\min\\left(
            \\alpha \\frac{\\Delta r^{2}}{2\\,\\text{dim}\\,D},\\;
            \\Delta t_{\\max}
        \\right)

    **Conduction velocity only, and the asymmetry is the design.** Its sibling
    solves two targets because Aliev-Panfilov's time axis is arbitrary, so APD
    is set by choosing what a model time unit means. Courtemanche has no such
    constant: APD90 falls out of the ionic equations and the conductance
    scalings with no closed-form inverse, so it is **measured and recorded**,
    never solved. A card's ``targets`` block is therefore per-model partial —
    the shared half is the conduction velocity, which is what lets an
    Aliev-Panfilov bank and a Courtemanche bank state that they aimed at the
    same tissue and makes the A/B between them interpretable.

    ``CV ~ sqrt(D)`` survives the model swap because it is a property of the
    diffusion operator rather than of the membrane; what does *not* survive is
    the constant of proportionality, which is why
    :data:`CRN_REFERENCE_CV_CM_S` had to be measured for this model rather than
    scaled from :data:`MODEL_UNIT_CV`.

    **Gap junctions are inside the target, not beside it.** Wilhelms additionally
    reduces intracellular conductivity 30 % for AF gap-junctional remodelling in
    tissue. Applying that here as well would double-count: this solve *derives*
    diffusion from the velocity we want, so a 30 % reduction would simply be
    cancelled by a 30 % larger solved ``D`` — a no-op that leaves a diffusion
    coefficient meaning nothing physical. Conduction slowing
    belongs in ``conduction_velocity_cm_s``.
    """
    if conduction_velocity_cm_s <= 0:
        raise ValueError("conduction_velocity_cm_s must be positive.")
    if dr_mm <= 0 or dr_model_units <= 0:
        raise ValueError("dr_mm and dr_model_units must be positive.")
    if dimensions < 1:
        raise ValueError("dimensions must be >= 1.")
    if not math.isclose(dr_model_units, dr_mm, rel_tol=1e-9):
        # Aliev-Panfilov is dimensionless, so its mesh may be scaled freely and
        # `space_unit_mm = dr_mm / dr_model_units` absorbs the difference.
        # Courtemanche's diffusion is in mm^2/ms, so its space unit IS the
        # millimetre and the two steps are the same number or the solve is
        # describing a different mesh from the one being simulated.
        raise ValueError(
            f"Courtemanche is dimensional: its space unit is the millimetre, so "
            f"dr_model_units must equal dr_mm. Got dr_model_units={dr_model_units} "
            f"and dr_mm={dr_mm}. (Aliev-Panfilov may differ; it is dimensionless.)"
        )
    if not math.isclose(dr_mm, CRN_CALIBRATION_DR_MM, rel_tol=1e-9):
        # Same guard, same reason, as calibrate_aliev_panfilov's eps check: the
        # constant was measured on one axis and is evidence about that axis only.
        raise ValueError(
            f"CRN_REFERENCE_CV_CM_S was measured at dr={CRN_CALIBRATION_DR_MM} mm; "
            f"got dr_mm={dr_mm}. Conduction velocity on a discrete mesh depends on "
            "the pitch, and the failure is silent: a CV-solve absorbs the "
            "discretisation error into diffusion and hits the target anyway, "
            "leaving a physical-looking number that is not. Re-measure the "
            "constant at the new pitch first."
        )

    scalings = dict(params or {})
    if scalings:
        # The reference CV was measured at control conductances. Sodium
        # conductance in particular moves CV directly, so a remodelled set makes
        # the constant an answer to a different question.
        raise ValueError(
            f"CRN_REFERENCE_CV_CM_S was measured at control conductances; got "
            f"scalings {sorted(scalings)}. A remodelled set changes conduction "
            "velocity, so solving through this constant would put the error into "
            "diffusion. Measure the reference CV for the remodelled set and "
            "register it before calibrating against it."
        )

    velocity_ratio = conduction_velocity_cm_s / CRN_REFERENCE_CV_CM_S
    diffusion = CRN_REFERENCE_DIFFUSION * velocity_ratio * velocity_ratio

    cfl = (dr_model_units * dr_model_units) / (2.0 * dimensions * diffusion)
    dt_model_units = min(DT_SAFETY_FACTOR * cfl, CRN_MAX_DT_MS)

    if not math.isfinite(dt_model_units) or dt_model_units <= 0:
        raise ValueError(
            f"derived dt is not usable ({dt_model_units}). A very high conduction "
            "velocity drives diffusion up and the step to zero."
        )

    return CourtemancheCellModel(
        diffusion=diffusion,
        dt_model_units=dt_model_units,
        params=scalings,
    )


__all__ = [
    "AP_EPS_PUBLISHED",
    "CRN_CALIBRATION_DR_MM",
    "CRN_MAX_DT_MS",
    "CRN_PACING_BCL_MS",
    "CRN_PACING_BEATS",
    "CRN_REFERENCE_CV_CM_S",
    "CRN_REFERENCE_DIFFUSION",
    "CRN_STIMULUS_AMPLITUDE_MV_PER_MS",
    "CRN_STIMULUS_DURATION_MS",
    "DT_SAFETY_FACTOR",
    "MODEL_UNIT_APD90",
    "MODEL_UNIT_CV",
    "WILHELMS_2012_CRN_CONTROL",
    "AlievPanfilovCellModel",
    "CellModelSpec",
    "CourtemancheCellModel",
    "calibrate_aliev_panfilov",
    "calibrate_courtemanche",
]
