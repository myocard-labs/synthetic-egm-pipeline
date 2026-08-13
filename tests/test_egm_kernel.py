"""The vendored EGM kernel: identity with stock (S37), then the axis fix (S39).

S37 vendored the kernel changing no physics; S39 corrects the electrode
transpose and the fibre-component order. Each landed separately so that each is
attributable — the fixes are a line or two apiece, so bundling would have been
tempting and would have destroyed the checks below.

S37 proved the vendored kernel was byte-identical to the stock tracker. S39
then changed the axis pairing deliberately, so that check was **sharpened
rather than deleted**: our kernel given ``(x, y, z)`` must still equal the stock
kernel given ``(y, x, z)``, exactly — proving the change is precisely a
transpose and nothing else.

The guard that keeps any of it meaningful: **our kernel actually ran**. A
subclass that failed to install its kernel would inherit upstream's and satisfy
an identity check trivially.

Marked ``slow``: these run the real solver, because the point is to compare the
production path against the thing it replaced.
"""

from __future__ import annotations

import finitewave as fw
import numpy as np
import numpy.typing as npt
import pytest
from finitewave.cpuwave.tracker.ecg_tracker import _compute_ecg_2d

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
from myocard_synthetic_egm_pipeline.backends.finitewave.egm_kernel import (
    EGMTracker,
    egm_kernel_2d,
)
from myocard_synthetic_egm_pipeline.simulate import (
    CenteredGrid2D,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
    run_single,
)
from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import compute_phi_e
from myocard_synthetic_egm_pipeline.simulate.specs import Edge


class _FakeModel:
    """Minimal stand-in for a Finitewave model.

    ``initialize`` only reads ``model.u`` (for its shape and ndim), so a bare
    array is enough — and it keeps these two tests off the solver, which is what
    lets them run in the fast suite.
    """

    def __init__(self, u: npt.NDArray[np.float64]) -> None:
        self.u = u


def _inputs() -> tuple[Patch2DGeometry, CenteredGrid2D, RunConfig]:
    geometry = Patch2DGeometry(size_mm=10.0, dr_mm=0.25)
    electrodes = CenteredGrid2D.sample(
        geometry=geometry, rng=np.random.default_rng(0), n_rows=2, n_cols=2
    )
    config = RunConfig(
        trace_duration_ms=192.0,
        output_fs_hz=1000.0,
        ap_time_unit_ms=1.97,
        capture_oversample=4,
    )
    return geometry, electrodes, config


def _run(backend: FinitewaveBackend) -> npt.NDArray[np.float32]:
    geometry, electrodes, config = _inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.2),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        config=config,
        backend=backend,
        rng=np.random.default_rng(7),
    )
    return np.asarray(result.bipolar_traces)


def test_our_tracker_is_the_one_installed() -> None:
    """The subclass installs our kernel, replacing upstream's.

    Guards the failure mode that would make the identity test below vacuous: if
    ``initialize`` did not reassign ``_compute``, the stock kernel would run and
    every byte would match for the wrong reason.
    """
    tracker = EGMTracker(measure_coords=np.array([[5.0, 5.0, 0.5]]))
    tracker.initialize(_FakeModel(np.zeros((8, 8))))

    # `_compute` is a closure binding the two physics parameters, so identity
    # is asserted against the kernel the class declares rather than the bound
    # wrapper. `kernel` exists for exactly this reason.
    assert EGMTracker.kernel is egm_kernel_2d
    assert tracker._compute is not fw.ECGTracker.__dict__.get("_compute")
    assert tracker.distance_power == 1.0  # 1/r, the corrected weighting
    assert isinstance(tracker, fw.ECGTracker)  # still upstream's plumbing


def test_three_dimensional_meshes_are_refused() -> None:
    """A 3D mesh must raise rather than fall back to the unfixed kernel.

    Silent fallback is how a future 3D geometry would end up on arithmetic that
    was never given the axis and weighting review this kernel had.
    """

    tracker = EGMTracker(measure_coords=np.array([[1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="2D meshes only"):
        tracker.initialize(_FakeModel(np.zeros((4, 4, 4))))


def test_our_kernel_reduces_to_stock_under_the_old_settings() -> None:
    """The running invariant: our kernel is stock plus *exactly* the known fixes.

    Carried forward and re-sharpened at each step rather than deleted:

    - **S37** — identical to stock, full stop.
    - **S39** — identical once the coordinates are pre-swapped, proving the
      change was precisely a transpose.
    - **S40** — identical once the coordinates are pre-swapped **and** the two
      new parameters are dialled back to their pre-fix values:
      ``distance_power = 2`` reinstates ``1/r²``, and
      ``conductivity = 1/(4 pi)`` cancels the new prefactor.

    Holding this pins the cumulative diff against upstream to a transpose, a
    square root and a constant — with nothing else having drifted in. That is a
    far stronger statement than "the tests still pass", and it costs one
    parametrised call.
    """
    rng = np.random.default_rng(11)
    n_i, n_j = 24, 31  # deliberately non-square, so an i/j mix-up cannot hide
    u = rng.standard_normal((n_i, n_j))
    u_tr = rng.standard_normal((n_i, n_j))
    indexes = np.arange(n_i * n_j, dtype=np.int64)
    coords = np.array([[3.0, 17.0, 0.5], [22.5, 4.25, 1.0], [11.0, 11.0, 0.2]])

    ours_as_stock = egm_kernel_2d(u_tr, u, coords, 0.25, indexes, 2.0, 1.0 / (4.0 * np.pi))
    stock_swapped = _compute_ecg_2d(u_tr, u, coords[:, [1, 0, 2]], 0.25, indexes)

    assert np.allclose(ours_as_stock, stock_swapped, rtol=1e-12, atol=0.0), (
        "our kernel no longer reduces to the stock one under pre-fix settings; "
        "something beyond the transpose, the sqrt and the prefactor has changed "
        f"(max rel diff {np.abs(ours_as_stock / stock_swapped - 1).max():.3e})"
    )


def test_the_shipped_defaults_are_the_corrected_physics() -> None:
    """Defaults must be the *right* values, not the reproducible-old ones.

    ``distance_power`` exists to reproduce pre-fix banks for comparison, which
    is a debugging use. If it ever defaulted to 2 the fix would be silently
    off in production while every test that passes it explicitly still passed.
    """
    rng = np.random.default_rng(5)
    u = rng.standard_normal((16, 21))
    u_tr = rng.standard_normal((16, 21))
    indexes = np.arange(16 * 21, dtype=np.int64)
    coords = np.array([[7.0, 5.0, 0.5]])

    default = egm_kernel_2d(u_tr, u, coords, 0.25, indexes)
    explicit_correct = egm_kernel_2d(u_tr, u, coords, 0.25, indexes, 1.0, 1.0)
    old_behaviour = egm_kernel_2d(u_tr, u, coords, 0.25, indexes, 2.0, 1.0)

    assert np.array_equal(default, explicit_correct)
    assert not np.allclose(default, old_behaviour)


def test_the_transpose_actually_changes_the_answer() -> None:
    """Guard the test above against a square-mesh or symmetric-input coincidence.

    If swapping the coordinates made no difference, the comparison would pass
    for reasons unrelated to the fix.
    """
    rng = np.random.default_rng(11)
    u = rng.standard_normal((24, 31))
    u_tr = rng.standard_normal((24, 31))
    indexes = np.arange(24 * 31, dtype=np.int64)
    coords = np.array([[3.0, 17.0, 0.5]])

    ours = egm_kernel_2d(u_tr, u, coords, 0.25, indexes)
    stock_unswapped = _compute_ecg_2d(u_tr, u, coords, 0.25, indexes)

    assert not np.allclose(ours, stock_unswapped)


@pytest.mark.slow
def test_the_traces_are_not_trivially_empty() -> None:
    """Guard against the identity test passing on two identical piles of zeros.

    A tracker that silently produced nothing would satisfy byte-equality
    perfectly, so the comparison is only meaningful if there is signal in it.
    """
    traces = _run(FinitewaveBackend())

    assert np.isfinite(traces).all()
    assert np.abs(traces).max() > 0.0
    assert np.unique(traces).size > 100  # real waveform, not a constant


# ---------------------------------------------------------------------------
# S39 — one axis convention across electrodes, fibres and stimulus
# ---------------------------------------------------------------------------


def _directional_run(edge: Edge, *, anisotropy_ratio: float = 1.0) -> npt.NDArray[np.float32]:
    """One clean, uniform simulation with the pairs laid along physical x.

    ``CenteredGrid2D`` separates the poles of each pair along **x**, so a
    ``left``/``right`` stimulus propagates *along* the pairs and a ``top``/
    ``bottom`` one propagates *across* them. Density 0 and no anisotropy keep
    orientation the only variable.
    """
    geometry = Patch2DGeometry(size_mm=12.0, dr_mm=0.25, anisotropy_ratio=anisotropy_ratio)
    electrodes = CenteredGrid2D.sample(
        geometry=geometry, rng=np.random.default_rng(0), n_rows=2, n_cols=2
    )
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge=edge),
        electrodes=electrodes,
        config=RunConfig(
            trace_duration_ms=192.0,
            output_fs_hz=1000.0,
            ap_time_unit_ms=1.97,
            capture_oversample=4,
        ),
        backend=FinitewaveBackend(),
        rng=np.random.default_rng(3),
    )
    return np.asarray(result.bipolar_traces)


@pytest.mark.slow
def test_a_wave_along_the_pair_beats_a_wave_across_it() -> None:
    """Bipolar directional sensitivity — the assertion that would have caught this.

    A bipole differences two nearby unipolar potentials, so its amplitude scales
    with ``d_AB · n̂``: **parallel** to propagation gives the poles a real time
    offset and a large biphasic deflection; **perpendicular** puts both poles on
    the same wavefront, firing together, and the near field cancels.

    Under the transpose every pair was perpendicular to a ``left`` wave, which
    is what produced the ``1.35e-6`` "dead healthy tissue" reading. With the
    axes agreed, ``left`` (along the pairs) must now dominate ``top`` (across).
    """
    along = np.ptp(_directional_run("left"), axis=1).max()
    across = np.ptp(_directional_run("top"), axis=1).max()

    assert along > 0.0
    assert along > 10.0 * across, (
        "a wave travelling ALONG the electrode pairs should dominate one travelling "
        f"ACROSS them; got along={along:.3e} across={across:.3e}. If across wins, the "
        "electrode coordinates are transposed again."
    )


@pytest.mark.slow
def test_healthy_uniform_tissue_produces_a_real_activation() -> None:
    """Density-0 tissue must give a real EGM, not a near-cancelled blob.

    This is the check that retires the CL-168 hypothesis. The ~2000x amplitude
    jump between density 0 and density 0.01 was read as "a planar wave over
    uniform tissue is degenerate"; it was the transpose. Healthy tissue can and
    must produce a normal local activation.
    """
    traces = _directional_run("left")

    peak_to_peak = np.ptp(traces, axis=1)
    assert peak_to_peak.min() > 1e-3, (
        f"healthy uniform tissue is still near-cancelled (min p2p {peak_to_peak.min():.3e}); "
        "the pairs are probably still perpendicular to the wavefront"
    )


@pytest.mark.slow
def test_the_fast_conduction_axis_is_the_intended_one() -> None:
    """``fiber_angle_rad = 0`` must run the fibres along **+x**, as documented.

    ``specs.Patch2DGeometry`` says *"0 = along +x"*. Finitewave stores fibre
    component 0 against mesh axis-0, which is our **y** — so writing
    ``fibers[..., 0] = cos(theta)`` put the fast axis 90 degrees off. With
    ``anisotropy_ratio``, conduction along the fibres is ``sqrt(ratio)`` faster,
    so the wave clears the mesh sooner in the fibre direction.
    """
    ratio = 9.0  # sqrt(9) = 3x, comfortably outside numerical noise

    def clearing_sample(edge: Edge) -> int:
        traces = _directional_run(edge, anisotropy_ratio=ratio)
        envelope = np.abs(traces).max(axis=0)
        live = np.flatnonzero(envelope > 0.01 * envelope.max())
        return int(live.max())

    along_fibres = clearing_sample("left")  # propagates +x = fibre direction
    across_fibres = clearing_sample("top")  # propagates +y

    assert along_fibres < across_fibres, (
        "a wave along the fibres should clear the mesh sooner than one across them; "
        f"got along={along_fibres} across={across_fibres}. If reversed, the fibre "
        "components are transposed."
    )


# ---------------------------------------------------------------------------
# S40 — 1/r weighting, cross-checked against the independent reference
# ---------------------------------------------------------------------------


def test_the_production_kernel_agrees_with_compute_phi_e() -> None:
    """Two independent implementations of the same formula must agree.

    This is where ``compute_phi_e`` stops being decorative. It has been
    thoroughly unit-tested since Wave 1 and **never called by the pipeline** —
    which is exactly how two kernels came to disagree on the physics unnoticed:
    the tested one was irrelevant and the untested one was authoritative.

    **Isotropic, unmasked tissue only, and that is deliberate** (plan S40). The
    two compute *different Laplacians*: ``compute_phi_e`` applies a plain
    5-point stencil over every node, while the production path uses
    Finitewave's **anisotropic**, myocardium-**masked** diffusion kernel. At
    production settings they must disagree, and an unconstrained ``allclose``
    would fail for entirely correct reasons. Constraining the tissue keeps the
    check aimed at what it is for — arithmetic errors in the weighting and the
    axes, both visible on the simplest possible tissue. The anisotropy and the
    mask are the solver's business, already covered by the reduces-to-stock
    invariant above.

    The inputs are matched by construction: our kernel receives the diffusion
    increment ``u_tr - u``, so feeding it the same physical Laplacian
    ``compute_phi_e`` computes internally makes the two directly comparable.
    """
    rng = np.random.default_rng(19)
    n_i, n_j, dr = 20, 27, 0.25  # non-square again
    v = rng.standard_normal((n_i, n_j))

    # The physical Laplacian compute_phi_e forms internally: 5-point, /dr^2,
    # zero on the boundary ring.
    lap = np.zeros_like(v)
    lap[1:-1, 1:-1] = (
        v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4.0 * v[1:-1, 1:-1]
    ) / (dr * dr)

    positions_mm = np.array([[3.0, 2.0, 0.5], [1.5, 4.25, 0.8]])

    ours = egm_kernel_2d(
        v + lap,  # u_tr - u == lap, the same source term
        v,
        positions_mm / dr,  # every column in cells, standoff included
        dr,
        np.arange(n_i * n_j, dtype=np.int64),
    )
    reference = compute_phi_e(v[np.newaxis], electrode_positions_mm=positions_mm, dr_mm=dr)[0]

    assert np.allclose(ours, reference, rtol=1e-10, atol=0.0), (
        "the production kernel and compute_phi_e disagree on isotropic clean tissue; "
        f"max rel diff {np.abs(ours / reference - 1).max():.3e}"
    )


def test_the_reference_check_would_catch_a_wrong_exponent() -> None:
    """The cross-check above must be sensitive to the thing it guards.

    A comparison that passed for any ``distance_power`` would prove nothing
    about the weighting, which is the defect it exists to detect.
    """
    rng = np.random.default_rng(19)
    n_i, n_j, dr = 20, 27, 0.25
    v = rng.standard_normal((n_i, n_j))
    lap = np.zeros_like(v)
    lap[1:-1, 1:-1] = (
        v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4.0 * v[1:-1, 1:-1]
    ) / (dr * dr)
    positions_mm = np.array([[3.0, 2.0, 0.5]])

    wrong = egm_kernel_2d(
        v + lap, v, positions_mm / dr, dr, np.arange(n_i * n_j, dtype=np.int64), 2.0, 1.0
    )
    reference = compute_phi_e(v[np.newaxis], electrode_positions_mm=positions_mm, dr_mm=dr)[0]

    assert not np.allclose(wrong, reference, rtol=1e-3)
