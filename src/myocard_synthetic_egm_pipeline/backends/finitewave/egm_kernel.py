"""Our own pseudo-**EGM** kernel, vendored from Finitewave's pseudo-ECG tracker.

Why ``egm`` and not ``ecg``
--------------------------
Upstream calls this an ECG, and that name is for surface leads. What is actually
computed here is the extracellular potential at *intracardiac electrode
positions* — an **electrogram**. The mismatch is not cosmetic: it framed a
two-day investigation around the wrong mental model, with far-field reasoning
applied to a near-field measurement. The project's terminology is EGM
throughout; this module keeps it.

Why vendor at all
-----------------
Two defects sit in the stock kernel's arithmetic, and both need fixing:

1. **an axis transpose** — the kernel differences coordinate column 0 against
   mesh axis-0, while our convention is ``x = j`` (axis-1), ``y = i`` (axis-0).
   That reflects the electrode grid across the diagonal, leaving every bipolar
   pair perpendicular to a ``left`` wavefront so the *near* field cancels;
2. **a mixed weighting** — the source term is a Laplacian (the diffusion
   increment) but the weight is ``1/r²``, which belongs to the gradient
   formulation. One term from each of two equivalent forms.

Defect 1 is **ours**: upstream's API means ``[i, j, z]`` and we hand it
``[x, y, z]``. Defect 2 is upstream's, and they have already fixed it on the
unreleased ``solvers`` branch — restoring the ``sqrt``, defaulting
``distance_power`` to 1, and adding the missing ``1/(4·pi·sigma_e)`` prefactor.

The alternatives were worse. Tracking that branch means a git pin on a moving
branch of a *runtime* dependency, and it restructures the package around a
numba/jax/mlx abstraction, so adopting it is a port rather than a bump.
Switching to :func:`~myocard_synthetic_egm_pipeline.simulate.pseudo_egm.compute_phi_e`
means holding the whole ``V_m`` history in memory — roughly 504 MB per
simulation at the production geometry — which is precisely why the streaming
tracker was chosen in the first place. Vendoring ~20 lines keeps the streaming
behaviour and puts both fixes somewhere we control.

**This module is deliberately a faithful copy for now.** It reproduces stock
0.9.3 *exactly*: same ``1/r²``, same axis handling, same absent prefactor. The
fixes land in later steps (S39, S40), each on its own, so that each change is
attributable. The verification for this step is that generated banks are
**byte-identical** to ones produced by the stock tracker — a check that means
nothing the moment a real change rides along with it.

Full analysis, including the derivations and the cross-simulator comparison:
``intracardiac-platform/project/investigations/pseudo_egm_axes_and_weighting.md``.
"""

from __future__ import annotations

from typing import Any

import finitewave as fw
import numpy as np

# numba ships no type information, so mypy cannot see through `prange` (it is a
# plain function to the type checker and a parallel range only under `njit`).
# Ignored at the import rather than at each use site, since every use is inside
# the compiled kernel where mypy's view is wrong by construction.
from numba import njit, prange


@njit(parallel=True)
def egm_kernel_2d(  # pragma: no cover - njit-compiled, exercised via the tracker
    u_tr: Any, u: Any, coords: Any, dr: float, indexes: Any
) -> Any:
    """Pseudo-EGM at each measurement point, for a 2D mesh.

    **Verbatim reproduction of Finitewave 0.9.3's ``_compute_ecg_2d``.** Every
    quirk below is intentional at this step and is corrected later:

    - ``coords[c, 0]`` is differenced against ``i`` (mesh axis-0), which is the
      transpose relative to our ``x = j`` convention — corrected in S39;
    - ``d`` holds the **squared** grid distance and the accumulator divides by
      ``d`` rather than ``sqrt(d)``, giving ``1/r²`` — corrected in S40;
    - no ``1/(4·pi·sigma_e)`` prefactor — added in S40.

    Parameters
    ----------
    u_tr, u
        Potential after and before the diffusion step. Their difference is the
        diffusion increment, i.e. the Laplacian source term.
    coords
        ``(n_points, 3)`` measurement points **in grid-index units**, so the
        caller has already divided physical millimetres by ``dr``.
    dr
        Grid spacing in mm per cell. Converts grid distance to physical.
    indexes
        Flat indices of myocardium nodes; the sum runs over these only.
    """
    n_j = u.shape[1]
    n_c = coords.shape[0]
    egm = np.zeros(n_c)

    for c in range(n_c):
        x = coords[c, 0]
        y = coords[c, 1]
        z = coords[c, 2]  # height above the plane
        acc = 0.0

        for ind in prange(len(indexes)):  # type: ignore[no-untyped-call, attr-defined]
            ii = indexes[ind]
            i = ii // n_j
            j = ii % n_j

            d = (x - i) * (x - i) + (y - j) * (y - j) + z * z
            if d > 0.0:
                acc += (u_tr[i, j] - u[i, j]) / (d * dr)

        egm[c] = acc

    return egm


class EGMTracker(fw.ECGTracker):  # type: ignore[misc]
    """Finitewave tracker that computes φ_e with **our** kernel.

    Subclasses rather than replaces, so all of upstream's plumbing — the
    diffusion-kernel call, the step scheduling, the output accumulation — is
    inherited untouched. Only the kernel selection changes.

    Overriding :meth:`initialize` alone is enough: upstream's ``calc_ecg``
    dispatches through ``self._compute``, so reassigning that after
    ``super().initialize()`` swaps the arithmetic without duplicating the
    surrounding code. Overriding ``calc_ecg`` instead would mean copying the
    diffusion-kernel invocation, which is real logic we have no reason to own.

    3D is refused rather than silently falling back to upstream's kernel. The
    repo is 2D-only today, and a quiet fallback is how a 3D geometry would end
    up on the unfixed arithmetic without anyone noticing.
    """

    def initialize(self, model: Any) -> None:
        super().initialize(model)
        if model.u.ndim != 2:
            raise ValueError(
                f"EGMTracker supports 2D meshes only; got model.u.ndim={model.u.ndim}. "
                "A 3D kernel needs the same axis and weighting review this one had — "
                "see investigations/pseudo_egm_axes_and_weighting.md."
            )
        self._compute = egm_kernel_2d


__all__ = ["EGMTracker", "egm_kernel_2d"]
