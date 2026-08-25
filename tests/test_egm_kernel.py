"""The vendored EGM kernel: identity with stock, then the axis fix.

Vendoring came first and changed no physics; the electrode transpose and the
fibre-component order were corrected after. Each landed separately so that each
is attributable — the fixes are a line or two apiece, so bundling would have
been tempting and would have destroyed the checks below.

The first step proved the vendored kernel byte-identical to the stock tracker.
The second changed the axis pairing deliberately, so that check was **sharpened
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
from myocard_synthetic_egm_pipeline.backends.finitewave import backend as backend_module
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
from myocard_synthetic_egm_pipeline.simulate.cell_models import CellModelSpec
from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import compute_phi_e
from myocard_synthetic_egm_pipeline.simulate.specs import Edge


def _shipped_cell_model() -> CellModelSpec:
    """The calibrated parameterisation, for tests that do not vary it.

    ``run_single`` requires a cell model rather than defaulting to one: it is a
    policy value, and the project rule is that libraries ship no defaults for
    policy values. Tests that are about something else get it from here.
    """
    from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card

    return load_model_card("af_remodelled_220ms", dr_mm=0.25, dr_model_units=0.25).solved


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
        cell_model=_shipped_cell_model(),
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

    - **at vendoring** — identical to stock, full stop.
    - **after the axis fix** — identical once the coordinates are pre-swapped,
      proving the change was precisely a transpose.
    - **after the weighting fix** — identical once the coordinates are
      pre-swapped **and** the two
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
# One axis convention across electrodes, fibres and stimulus
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
        cell_model=_shipped_cell_model(),
        config=RunConfig(
            trace_duration_ms=192.0,
            output_fs_hz=1000.0,
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

    This retires an earlier hypothesis. The ~2000x amplitude jump between
    density 0 and density 0.01 was read as "a planar wave over uniform tissue
    is degenerate"; it was the transpose. Healthy tissue can and
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
    ``fibers[..., 0] = cos(theta)`` put the fast axis 90 degrees off, and the
    wave clears the mesh sooner along whichever axis the fibres actually run.

    **The magnitude reasoning here used to be wrong, and the test passed
    anyway.** It requested ``anisotropy_ratio = 9`` and argued conduction was
    ``sqrt(9) = 3x`` faster along the fibres. In fact the request did nothing —
    ``anisotropy_ratio`` was written to an object that never read it — and the
    3x came from the stencil's built-in default. What the test genuinely checks
    is the **direction**, which is set by the fibre field and was always real,
    so the transpose fix stayed verified throughout. But a test that would not have
    failed if its own input were ignored was proving less than it claimed, so
    the ratio is now a plain 3.0 and the *magnitude* claim lives in
    :func:`test_the_requested_anisotropy_is_the_realized_one`, which does fail
    if the knob is ignored.
    """
    ratio = 3.0  # ~3x along vs across, comfortably outside numerical noise

    def mean_arrival_sample(edge: Edge) -> float:
        """Mean activation index across pairs — *arrival*, not clearing.

        The original proxy was the **last** sample above 1 % of peak, i.e. when
        the trace went quiet. That worked at APD 51 ms and stopped working the
        moment the calibration lengthened APD to 220 ms: the window is 192 ms,
        so the trace
        is still repolarising at the final sample in **both** directions and the
        proxy saturates at 191 for each. It failed for the right reason — the
        calibration fix doing exactly what it was meant to.

        Arrival time is the quantity the test was always reaching for, and it
        does not saturate: a faster axis activates earlier regardless of how
        long repolarisation then takes.
        """
        traces = _directional_run(edge, anisotropy_ratio=ratio)
        return float(np.mean([int(np.argmax(np.abs(np.gradient(t)))) for t in traces]))

    along_fibres = mean_arrival_sample("left")  # propagates +x = fibre direction
    across_fibres = mean_arrival_sample("top")  # propagates +y

    assert along_fibres < across_fibres, (
        "a wave along the fibres should arrive sooner than one across them; "
        f"got along={along_fibres:.1f} across={across_fibres:.1f}. If reversed, "
        "the fibre components are transposed."
    )


# ---------------------------------------------------------------------------
# The 1/r weighting, cross-checked against the independent reference
# ---------------------------------------------------------------------------


def test_the_production_kernel_agrees_with_compute_phi_e() -> None:
    """Two independent implementations of the same formula must agree.

    This is where ``compute_phi_e`` stops being decorative. It has been
    thoroughly unit-tested since Wave 1 and **never called by the pipeline** —
    which is exactly how two kernels came to disagree on the physics unnoticed:
    the tested one was irrelevant and the untested one was authoritative.

    **Isotropic, unmasked tissue only, and that is deliberate.** The
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


# ---------------------------------------------------------------------------
# anisotropy_ratio is operative — it spent the project inoperative
# ---------------------------------------------------------------------------


def _cv_model_units(edge: Edge, ratio: float) -> float:
    """Plane-wave conduction velocity in model units, straight from V_m.

    Measured off ``ActivationTime2DTracker`` rather than off the EGM, because
    conduction velocity is a property of the propagating field and routing it
    through the electrode model would fold in the pseudo-EGM's own behaviour.
    """
    geometry = Patch2DGeometry(size_mm=12.0, dr_mm=0.25, anisotropy_ratio=ratio)
    model = fw.AlievPanfilov2D()
    model.dt = backend_module._AP_DT_MODEL_UNITS
    model.dr = backend_module._AP_DR_MODEL_UNITS
    tissue = backend_module._build_tissue_2d(geometry)
    backend_module._configure_anisotropy_2d(model, geometry)
    backend_module._apply_substrate_2d(
        tissue=tissue,
        strategy=UniformRandomFibrosis(density=0.0),
        rng=np.random.default_rng(0),
    )
    model.cardiac_tissue = tissue
    backend_module._install_activation_2d(
        model=model, source=PlanarEdgeStimulus(edge=edge), tissue=tissue
    )
    model.t_max = 200.0

    tracker = fw.ActivationTime2DTracker()
    tracker.threshold = 0.5
    sequence = fw.TrackerSequence()
    sequence.add_tracker(tracker)
    model.tracker_sequence = sequence
    model.run()

    # Average out the transverse direction, then fit activation time against
    # index over the middle 40 % — away from the stimulus and the far boundary.
    activation = np.asarray(tracker.act_t)
    axis = 1 if edge == "left" else 0
    profile = activation.mean(axis=1 - axis)
    lo, hi = int(0.3 * len(profile)), int(0.7 * len(profile))
    index = np.arange(lo, hi)
    reached = profile[lo:hi] > 0
    assert reached.sum() > 5, f"wave never crossed the fit window for edge={edge!r}"
    slope = np.polyfit(index[reached], profile[lo:hi][reached], 1)[0]
    return float(geometry.dr_mm / slope)


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_the_requested_anisotropy_is_the_realized_one(ratio: float) -> None:
    """``anisotropy_ratio`` must actually reach the solver.

    Until 2026-08-14 it did not. ``_configure_anisotropy_2d`` assigned ``D_al``
    and ``D_ac`` to the **model**, while Finitewave reads them off the
    **stencil** — so Python created two attributes nobody consulted and the
    realized ratio was the stencil's built-in 3.09 whatever was requested.
    Measured 3.093 for requested 1.0, 3.0 and 6.0 alike.

    **Ratio 1.0 is the load-bearing case** — it is the one the old code could
    not produce, so a test at 3.0 alone would have passed against the bug.
    2.0 is the shipped default — the atrial working-myocardium value — covered
    here so the knob is exercised at the setting real banks use. 3.0 is deliberately
    absent: it is covered exactly, not approximately, by
    :func:`test_the_default_ratio_reproduces_the_stencil_defaults_exactly`.

    Tolerance is 10 %: the realized ratio carries a few percent of
    discretization excess on a 0.25 mm mesh (2.0 measures ~2.15), which is a
    property of the grid rather than of the tensor.
    """
    along = _cv_model_units("left", ratio)  # +x, the fibre direction
    across = _cv_model_units("top", ratio)  # +y
    realized = along / across

    assert realized == pytest.approx(ratio, rel=0.10), (
        f"requested anisotropy_ratio={ratio} but measured CV ratio {realized:.3f} "
        f"(along {along:.4f}, across {across:.4f}). A realized ~3.09 at every "
        "requested value means D_al/D_ac is being set somewhere nothing reads."
    )


@pytest.mark.slow
def test_the_shipped_ratio_leaves_the_along_fibre_velocity_alone() -> None:
    """Changing the anisotropy must not move the axis we calibrate against.

    ``D_al`` is pinned at 1 and the whole ratio goes into ``D_ac``, so raising
    the anisotropy slows the transverse axis and leaves the longitudinal one
    untouched. The rejected alternative — holding the geometric mean fixed —
    would make this knob shift the along-fibre CV as a side effect, which is
    the quantity every calibration and every published number refers to.

    **This invariance is a property of a planar wave in homogeneous tissue, and
    only of that.** ``_cv_model_units`` uses ``density = 0`` for exactly that
    reason. With ``fiber_angle_rad = 0`` the tensor is
    ``diag(D_ac, D_al) = diag(1/ratio**2, 1)``, so the entry a wave travelling
    along the fibres rides is 1.0 at every ratio — it never samples the
    anisotropy. Add fibrosis and the wave diffracts around every hole, sampling
    the transverse entry continuously, and the invariance stops holding. That is
    correct behaviour, not a leak: measured on a fibrotic 40 mm patch, along-fibre
    propagation at ratio 1 vs 3 differs by more than the same change measured
    across the fibres. The practical consequence is for generation, not for
    this test — in the substrate we actually generate, the ratio changes every
    bank whatever the fibre orientation.
    """
    isotropic = _cv_model_units("left", 1.0)
    anisotropic = _cv_model_units("left", 3.0)

    assert anisotropic == pytest.approx(isotropic, rel=0.02), (
        f"along-fibre CV moved from {isotropic:.4f} to {anisotropic:.4f} when only "
        "the anisotropy changed; the ratio is leaking into the absolute scale."
    )


def test_the_default_ratio_reproduces_the_stencil_defaults_exactly() -> None:
    """At ``anisotropy_ratio = 3.0`` this change must be a no-op, bit for bit.

    ``D_al = 1, D_ac = 1/9`` is precisely what ``AsymmetricStencil2D`` already
    defaults to, so every bank generated at the shipped ratio is unchanged.
    That is what lets this fix ship on its own, ahead of the recalibration that
    changes every trace — and it is worth an explicit test rather than an
    explicit comment, because it is the claim the step's safety rests on.
    """
    model = fw.AlievPanfilov2D()
    backend_module._configure_anisotropy_2d(model, Patch2DGeometry(anisotropy_ratio=3.0))
    stock = fw.AsymmetricStencil2D()

    assert model.stencil.D_al == stock.D_al
    assert model.stencil.D_ac == stock.D_ac
