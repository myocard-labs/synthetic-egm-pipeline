"""Cell-model specs — the fifth strategy spec that design note D2 called for.

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

S38b put ``eps``, ``diffusion``, ``dt_model_units`` and ``ap_time_unit_ms`` on
``RunConfig`` — the config object shared by every backend — which is exactly
what D2 ruled against: *"would put model-specific parameters into a config
object that has no place for them."* A Courtemanche run would have carried an
``eps`` field meaning nothing. This module is the correction.

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
from dataclasses import dataclass
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
    """What one node of the mesh does. The fifth strategy spec (D2).

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

    def ms_to_model_time(self, duration_ms: float) -> float:
        """Convert a physical duration to this model's own time units."""
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
        return {
            "cell_model_type": self.type,
            "ap_time_unit_ms": float(self.time_unit_ms),
            "ap_diffusion": float(self.diffusion),
            "ap_membrane_eps": float(self.eps),
            "ap_dt_model_units": float(self.dt_model_units),
        }


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
    shortcut makes APD unobservable inside the analysis window by construction
    (CL-176).

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
        # this project has already done once (CL-170).
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


__all__ = [
    "AP_EPS_PUBLISHED",
    "DT_SAFETY_FACTOR",
    "MODEL_UNIT_APD90",
    "MODEL_UNIT_CV",
    "AlievPanfilovCellModel",
    "CellModelSpec",
    "calibrate_aliev_panfilov",
]
