"""The vendored EGM kernel reproduces the stock tracker exactly (S37).

This step changes no physics. Its whole verification is *"nothing changed"* —
which is precisely why it is a step of its own: the two fixes it enables (the
axis transpose in S39, the ``1/r`` weighting in S40) are each a line or two, so
bundling them would be tempting, and would destroy this check.

Two things have to hold, and the second is the one that could silently rot:

1. the traces are **byte-identical** to the stock tracker's;
2. **our kernel actually ran**. A subclass that failed to install its kernel
   would inherit upstream's and pass (1) trivially, so identity alone proves
   nothing without it.

Marked ``slow``: these run the real solver, because the point is to compare the
production path against the thing it replaced.
"""

from __future__ import annotations

import finitewave as fw
import numpy as np
import numpy.typing as npt
import pytest

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


@pytest.mark.slow
def test_vendored_kernel_is_byte_identical_to_the_stock_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of S37: same numbers, different owner.

    Runs the production path twice — once with our tracker, once with the
    subclass neutered so upstream's kernel is used — and requires exact
    equality. Not ``allclose``: this step is a refactor, and a refactor that
    moves the last bit is not a refactor.
    """
    ours = _run(FinitewaveBackend())

    # Neuter the override so `initialize` leaves upstream's `_compute` in place.
    monkeypatch.setattr(EGMTracker, "initialize", fw.ECGTracker.initialize)
    stock = _run(FinitewaveBackend())

    assert ours.shape == stock.shape
    assert np.array_equal(ours, stock), (
        "vendored kernel diverged from the stock tracker; S37 is meant to change "
        f"no physics (max abs diff {np.abs(ours - stock).max():.3e})"
    )


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
