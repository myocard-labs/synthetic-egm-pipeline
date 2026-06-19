"""Pure-math tests for pseudo_egm: compute_phi_e, bipolar pairing, downsample."""

from __future__ import annotations

import numpy as np
import pytest

from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import (
    bipolar_from_unipolar,
    compute_phi_e,
    downsample,
)

# ---------------------------------------------------------------------------
# compute_phi_e
# ---------------------------------------------------------------------------


def test_compute_phi_e_shape() -> None:
    """Output shape is (T_capture, n_electrodes) regardless of mesh size."""
    v_m = np.random.default_rng(0).standard_normal((20, 16, 16))
    electrodes = np.array([[1.0, 1.0, 0.5], [3.0, 3.0, 0.5]])
    phi = compute_phi_e(v_m, electrode_positions_mm=electrodes, dr_mm=0.25)
    assert phi.shape == (20, 2)


def test_compute_phi_e_zero_v_m_gives_zero_phi() -> None:
    """If V_m is flat (no spatial variation), the Laplacian is zero, so phi_e is zero."""
    v_m = np.full((10, 16, 16), 0.5, dtype=np.float64)
    electrodes = np.array([[1.0, 1.0, 0.5]])
    phi = compute_phi_e(v_m, electrode_positions_mm=electrodes, dr_mm=0.25)
    np.testing.assert_allclose(phi, 0.0, atol=1e-12)


def test_compute_phi_e_responds_to_localised_dipole() -> None:
    """An electrode closer to a spatial-Laplacian source feature should
    record a larger-magnitude phi_e response than a distant electrode.

    Set up a V_m with an obvious Laplacian feature near one corner; place
    one electrode close to the feature and one far away. The near
    electrode should see a larger-amplitude response.
    """
    # Single timestep: V_m bumps up at mesh node (3, 3) only.
    v_m = np.zeros((1, 16, 16), dtype=np.float64)
    v_m[0, 3, 3] = 1.0
    near = np.array([[0.75, 0.75, 0.5]])  # node (3, 3) sits at (0.75, 0.75) mm
    far = np.array([[3.5, 3.5, 0.5]])
    phi_near = compute_phi_e(v_m, electrode_positions_mm=near, dr_mm=0.25)
    phi_far = compute_phi_e(v_m, electrode_positions_mm=far, dr_mm=0.25)
    assert abs(float(phi_near[0, 0])) > abs(float(phi_far[0, 0]))


def test_compute_phi_e_rejects_2d_v_m() -> None:
    v_m = np.zeros((16, 16))
    with pytest.raises(ValueError, match="3-D"):
        compute_phi_e(v_m, electrode_positions_mm=np.array([[0, 0, 0]]), dr_mm=0.25)


def test_compute_phi_e_rejects_2d_electrode_array_wrong_cols() -> None:
    v_m = np.zeros((10, 16, 16))
    with pytest.raises(ValueError, match="electrode_positions_mm"):
        compute_phi_e(v_m, electrode_positions_mm=np.zeros((4, 2)), dr_mm=0.25)


def test_compute_phi_e_rejects_nonpositive_dr() -> None:
    v_m = np.zeros((10, 16, 16))
    with pytest.raises(ValueError, match="dr_mm"):
        compute_phi_e(v_m, electrode_positions_mm=np.zeros((1, 3)), dr_mm=0.0)


# ---------------------------------------------------------------------------
# bipolar_from_unipolar
# ---------------------------------------------------------------------------


def test_bipolar_from_unipolar_shape() -> None:
    """Output is (T, n_pairs)."""
    unipolar = np.random.default_rng(0).standard_normal((100, 5))
    pairs = ((0, 1), (1, 2), (3, 4))
    out = bipolar_from_unipolar(unipolar, pairs)
    assert out.shape == (100, 3)


def test_bipolar_from_unipolar_subtraction() -> None:
    """Each pair (a, b) yields unipolar[:, a] - unipolar[:, b]."""
    unipolar = np.arange(20, dtype=np.float64).reshape(4, 5)
    pairs = ((0, 1), (2, 3))
    out = bipolar_from_unipolar(unipolar, pairs)
    np.testing.assert_array_equal(out[:, 0], unipolar[:, 0] - unipolar[:, 1])
    np.testing.assert_array_equal(out[:, 1], unipolar[:, 2] - unipolar[:, 3])


def test_bipolar_from_unipolar_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        bipolar_from_unipolar(np.zeros(10), [(0, 1)])


# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


def test_downsample_integer_ratio_strides() -> None:
    """An integer downsample ratio takes the fast path (stride)."""
    sig = np.arange(20, dtype=np.float64).reshape(20, 1)
    out = downsample(sig, source_fs_hz=4000.0, target_fs_hz=1000.0)
    # 4x stride keeps every fourth sample.
    np.testing.assert_array_equal(out[:, 0], np.arange(0, 20, 4))


def test_downsample_passthrough_when_ratio_is_one() -> None:
    sig = np.arange(10, dtype=np.float64).reshape(10, 1)
    out = downsample(sig, source_fs_hz=1000.0, target_fs_hz=1000.0)
    np.testing.assert_array_equal(out, sig)


def test_downsample_non_integer_uses_interp() -> None:
    """A non-integer ratio falls back to linear interpolation."""
    # Source 4123 Hz, target 1000 Hz → ratio ≈ 4.123 (not integer).
    sig = np.linspace(0, 1, 4123).reshape(-1, 1)
    out = downsample(sig, source_fs_hz=4123.0, target_fs_hz=1000.0)
    # Expected sample count: round(duration_s * target_fs_hz) = round(1.0 * 1000) = 1000.
    assert out.shape[0] == 1000


def test_downsample_rejects_upsampling() -> None:
    sig = np.zeros((100, 1))
    with pytest.raises(ValueError, match="upsampling"):
        downsample(sig, source_fs_hz=1000.0, target_fs_hz=4000.0)


def test_downsample_rejects_nonpositive_rates() -> None:
    sig = np.zeros((100, 1))
    with pytest.raises(ValueError, match="positive"):
        downsample(sig, source_fs_hz=0.0, target_fs_hz=1000.0)


def test_downsample_handles_1d_input() -> None:
    """1-D inputs are valid and produce 1-D outputs."""
    sig = np.arange(20, dtype=np.float64)
    out = downsample(sig, source_fs_hz=4000.0, target_fs_hz=1000.0)
    assert out.ndim == 1
    np.testing.assert_array_equal(out, np.arange(0, 20, 4))
