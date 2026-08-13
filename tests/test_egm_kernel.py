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

    assert tracker._compute is egm_kernel_2d
    assert isinstance(tracker, fw.ECGTracker)  # still upstream's plumbing


def test_three_dimensional_meshes_are_refused() -> None:
    """A 3D mesh must raise rather than fall back to the unfixed kernel.

    Silent fallback is how a future 3D geometry would end up on arithmetic that
    was never given the axis and weighting review this kernel had.
    """

    tracker = EGMTracker(measure_coords=np.array([[1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="2D meshes only"):
        tracker.initialize(_FakeModel(np.zeros((4, 4, 4))))


def test_our_kernel_differs_from_stock_by_exactly_a_transpose() -> None:
    """S37's byte-identity gate, sharpened for S39 rather than deleted.

    Until S39 our kernel reproduced the stock tracker exactly, and identity was
    the proof that vendoring changed nothing. S39 changes the axis pairing on
    purpose, so plain identity can no longer hold — but the *claim* it protected
    still can, in a stronger form:

        our kernel given ``(x, y, z)``  ==  stock kernel given ``(y, x, z)``

    exactly. Both then compute ``(y-i)² + (x-j)² + z²``. Holding this proves the
    change is **precisely** a transpose and nothing else — no weighting drifted,
    no stray edit rode along. Deleting the test would have given up that
    guarantee at the moment it became most useful.

    Calls both kernels directly rather than running the solver twice: the claim
    is about arithmetic, so the inputs may as well be arbitrary, and it keeps
    the check in the fast suite.
    """
    rng = np.random.default_rng(11)
    n_i, n_j = 24, 31  # deliberately non-square, so an i/j mix-up cannot hide
    u = rng.standard_normal((n_i, n_j))
    u_tr = rng.standard_normal((n_i, n_j))
    indexes = np.arange(n_i * n_j, dtype=np.int64)
    coords = np.array([[3.0, 17.0, 0.5], [22.5, 4.25, 1.0], [11.0, 11.0, 0.2]])

    ours = egm_kernel_2d(u_tr, u, coords, 0.25, indexes)
    stock_swapped = _compute_ecg_2d(u_tr, u, coords[:, [1, 0, 2]], 0.25, indexes)

    assert np.array_equal(ours, stock_swapped), (
        "our kernel is not a pure transpose of the stock one; something other than "
        f"the axis pairing changed (max abs diff {np.abs(ours - stock_swapped).max():.3e})"
    )


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
