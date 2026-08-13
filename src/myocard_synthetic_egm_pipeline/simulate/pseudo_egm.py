"""Okenov 2024 pseudo-EGM forward + bipolar pairing + downsampling.

These three steps are backend-agnostic — given V_m on a 2D mesh, the
electrode positions in physical mm, and the bipolar pair list, the math
is the same regardless of which solver produced V_m. Future backends
that don't ship their own ECG tracker (openCARP's pseudo-bidomain, for
example, is listed as "under-testing" in TorchCor) call into this
module the same way Finitewave's V_m capture does.

Per ``project/architecture.md``: this module imports no backend code.

Formula
-------
The Okenov 2024 / Plonsey pseudo-EGM at electrode :math:`e` with
position :math:`\\vec r_e`:

.. math::

    \\phi_e(t) = \\sum_{i \\in \\text{mesh}} \\frac{I_{m,i}(t)}{4\\pi \\|\\vec r_e - \\vec r_i\\|}

where :math:`I_m \\approx D \\nabla^2 V_m` is the transmembrane current
source per node and the sum is over all interior mesh nodes. The
implementation below uses a discrete Laplacian-of-V_m proxy as the
current source (the diffusion-step delta after one time integration
step), matching what Finitewave's ``ECG2DTracker`` computes internally.

For the case where a backend already provides per-electrode φ_e (e.g.
Finitewave's ``ECG2DTracker.output``), the runner can skip
:func:`compute_phi_e` entirely and pass the tracker output straight to
:func:`bipolar_from_unipolar` + :func:`downsample`.

Which of these three the pipeline actually calls
------------------------------------------------
Worth stating, because the answer surprised us: **two of the three.**

- :func:`bipolar_from_unipolar` — production, Step 6 of the runner;
- :func:`downsample` — production, Step 7;
- :func:`compute_phi_e` — **not** production. Every bank this repo has written
  came from the Finitewave-side tracker (now
  :mod:`~myocard_synthetic_egm_pipeline.backends.finitewave.egm_kernel`), never
  from here.

That gap is how two kernels came to disagree on the physics unnoticed: this one
was thoroughly unit-tested and irrelevant, while the one that produced the data
was untested and authoritative. From S40 :func:`compute_phi_e` is promoted to
the **numerical reference** the production kernel is checked against — but only
on isotropic, unmasked tissue, because the Laplacian here is a plain 5-point
stencil while the production path uses Finitewave's anisotropic, myocardium-
masked diffusion kernel. See ``project/phase_1_5_plan.md`` S40.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def compute_phi_e(
    v_m_history: npt.NDArray[np.floating[Any]],
    *,
    electrode_positions_mm: npt.NDArray[np.float64],
    dr_mm: float,
    conductivity: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Compute per-electrode pseudo-EGM from a V_m history on a 2D grid.

    Implements the Okenov 2024 pseudo-EGM formula
    :math:`\\phi_e(t) \\propto \\sum_i I_{m,i}(t) / r_i` where the current
    source :math:`I_m` is approximated by the discrete Laplacian of V_m
    (consistent with Finitewave's ECG tracker internals).

    Parameters
    ----------
    v_m_history
        ``(T_capture, n_i, n_j)`` float array of V_m on a 2D mesh per
        capture step.
    electrode_positions_mm
        ``(n_electrodes, 3)`` float64 — electrode positions in physical
        mm relative to the patch origin (axis 0 = x along columns,
        axis 1 = y along rows, axis 2 = z above the surface).
    dr_mm
        Spatial step of the mesh, in mm.

    Returns
    -------
    np.ndarray
        ``(T_capture, n_electrodes)`` float64 array of per-electrode
        unipolar EGMs at the capture rate of ``v_m_history``.

    Notes
    -----
    The 4π normalization is intentionally dropped here — for a
    classifier the absolute scale is irrelevant and the relative
    amplitude across pairs is what carries the morphology signal. If an
    absolute mV calibration is ever needed, scale this function's
    output downstream rather than baking the factor in.
    """
    if v_m_history.ndim != 3:
        raise ValueError(f"v_m_history must be 3-D (T, n_i, n_j); got {v_m_history.ndim}-D")
    if electrode_positions_mm.ndim != 2 or electrode_positions_mm.shape[1] != 3:
        raise ValueError("electrode_positions_mm must have shape (n_electrodes, 3).")
    if dr_mm <= 0:
        raise ValueError("dr_mm must be positive.")

    n_capture, n_i, n_j = v_m_history.shape
    n_electrodes = electrode_positions_mm.shape[0]

    # 5-point discrete Laplacian of V_m per timestep (interior only).
    # Boundary cells contribute zero current, matching the
    # zero-flux boundary condition Finitewave imposes.
    lap = np.zeros_like(v_m_history)
    lap[:, 1:-1, 1:-1] = (
        v_m_history[:, 2:, 1:-1]
        + v_m_history[:, :-2, 1:-1]
        + v_m_history[:, 1:-1, 2:]
        + v_m_history[:, 1:-1, :-2]
        - 4.0 * v_m_history[:, 1:-1, 1:-1]
    ) / (dr_mm * dr_mm)

    # Mesh-node positions in mm. Axis 0 is i (rows, mapped to y in mm),
    # axis 1 is j (cols, mapped to x in mm).
    j_idx, i_idx = np.meshgrid(np.arange(n_j), np.arange(n_i), indexing="xy")
    node_x_mm = j_idx.astype(np.float64) * dr_mm
    node_y_mm = i_idx.astype(np.float64) * dr_mm

    # Per-electrode φ_e = sum over nodes of lap / distance.
    phi_e = np.zeros((n_capture, n_electrodes), dtype=np.float64)
    for e_idx in range(n_electrodes):
        ex, ey, ez = electrode_positions_mm[e_idx]
        dx = node_x_mm - ex
        dy = node_y_mm - ey
        # 3D distance: nodes sit at z=0; electrode at z=ez.
        r = np.sqrt(dx * dx + dy * dy + ez * ez)
        # Distance can never hit zero because ez > 0 by construction
        # (electrode height bounded away from the surface — see
        # CenteredGrid2D.sample). Floor defensively anyway.
        r = np.maximum(r, dr_mm * 1e-3)
        weight = 1.0 / r  # (n_i, n_j)
        # Sum over interior nodes for each capture step.
        phi_e[:, e_idx] = (lap * weight[np.newaxis, :, :]).sum(axis=(1, 2))

    # The 1/(4 pi sigma_e) of the Plonsey / Gima-Rudy integral. A constant
    # scale, so it changes no morphology and no classification — added (S40)
    # only so this and the production kernel are directly comparable, which is
    # the whole point of keeping this function around.
    phi_e *= 1.0 / (4.0 * np.pi * conductivity)

    return phi_e


def bipolar_from_unipolar(
    unipolar: npt.NDArray[np.floating[Any]],
    bipolar_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> npt.NDArray[np.floating[Any]]:
    """Difference unipolar electrodes into bipolar pairs.

    Parameters
    ----------
    unipolar
        ``(T, n_electrodes)`` array of per-electrode unipolar traces.
    bipolar_pairs
        Sequence of ``(a_idx, b_idx)`` index pairs. The bipolar trace
        is ``unipolar[:, a] - unipolar[:, b]``.

    Returns
    -------
    np.ndarray
        ``(T, n_pairs)`` array of bipolar traces.
    """
    if unipolar.ndim != 2:
        raise ValueError(f"unipolar must be 2-D, got {unipolar.ndim}-D")
    pairs = list(bipolar_pairs)
    out = np.empty((unipolar.shape[0], len(pairs)), dtype=unipolar.dtype)
    for pair_idx, (a, b) in enumerate(pairs):
        out[:, pair_idx] = unipolar[:, a] - unipolar[:, b]
    return out


def downsample(
    traces: npt.NDArray[np.floating[Any]],
    *,
    source_fs_hz: float,
    target_fs_hz: float,
) -> npt.NDArray[np.floating[Any]]:
    """Resample ``traces`` from ``source_fs_hz`` to ``target_fs_hz``.

    Integer-ratio stride fast path; linear-interpolation slow path
    when the ratio isn't integer (the AP simulator's effective capture
    rate doesn't always divide cleanly into 1 kHz due to
    :data:`~myocard_synthetic_egm_pipeline.constants.AP_TIME_UNIT_MS`
    rounding).

    Linear interpolation (no anti-alias filter) is acceptable for
    Phase 1 because (a) the AP membrane potential has no spectral
    energy above a few hundred Hz, well below the typical 4-8 kHz
    capture rate; (b) the source rate is several times the target.
    For Phase 2 / Courtemanche (faster wavefronts), swap for
    ``scipy.signal.resample_poly`` with a polyphase filter.

    Parameters
    ----------
    traces
        ``(T, n_channels)`` or 1-D ``(T,)`` array.
    source_fs_hz, target_fs_hz
        Source and target sample rates. ``source_fs_hz >= target_fs_hz``;
        upsampling is rejected.
    """
    if source_fs_hz <= 0 or target_fs_hz <= 0:
        raise ValueError("Both sampling rates must be positive.")
    if source_fs_hz < target_fs_hz:
        raise ValueError(
            f"source_fs_hz ({source_fs_hz}) must be >= target_fs_hz "
            f"({target_fs_hz}); upsampling is not supported here."
        )

    ratio_f = source_fs_hz / target_fs_hz
    ratio = round(ratio_f)
    # Fast path: integer ratio, stride.
    if ratio >= 1 and abs(ratio - ratio_f) <= 1e-9:
        if ratio == 1:
            return traces
        return traces[::ratio]

    # Slow path: linear interpolation onto the target time grid.
    n_source = traces.shape[0]
    duration_s = n_source / source_fs_hz
    n_target = round(duration_s * target_fs_hz)
    t_source = np.arange(n_source, dtype=np.float64) / source_fs_hz
    t_target = np.arange(n_target, dtype=np.float64) / target_fs_hz

    if traces.ndim == 1:
        interpolated = np.asarray(np.interp(t_target, t_source, traces))
        return interpolated.astype(traces.dtype, copy=False)
    out = np.empty((n_target, traces.shape[1]), dtype=traces.dtype)
    for c in range(traces.shape[1]):
        out[:, c] = np.interp(t_target, t_source, traces[:, c])
    return out
