"""Strategy spec validation + Protocol satisfaction + CenteredGrid2D geometry."""

from __future__ import annotations

import numpy as np
import pytest

from myocard_synthetic_egm_pipeline.simulate.specs import (
    EDGES,
    ActivationSource,
    CenteredGrid2D,
    Edge,
    ElectrodePlacement,
    GeometrySpec,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    SubstrateStrategy,
    UniformRandomFibrosis,
    random_edge,
)

# ---------------------------------------------------------------------------
# Patch2DGeometry
# ---------------------------------------------------------------------------


def test_patch_2d_defaults() -> None:
    """Defaults match the project's Phase 1 spec (40 mm patch at 0.25 mm)."""
    g = Patch2DGeometry()
    assert g.size_mm == 40.0
    assert g.dr_mm == 0.25
    # 2.0 since S38b: atrial WALL is ~2:1 (Hansson 1998); the higher ratios in
    # the literature belong to specialised bundles, not working myocardium.
    assert g.anisotropy_ratio == 2.0
    assert g.fiber_angle_rad == 0.0
    assert g.type == "patch_2d"
    assert g.n_cells_per_edge == 160
    assert g.shape == (160, 160)


def test_patch_2d_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="size_mm"):
        Patch2DGeometry(size_mm=0.0)


def test_patch_2d_rejects_nonpositive_dr() -> None:
    with pytest.raises(ValueError, match="dr_mm"):
        Patch2DGeometry(dr_mm=-0.1)


def test_patch_2d_rejects_anisotropy_below_one() -> None:
    """Anisotropy ratio < 1 is non-physical for our along/across convention."""
    with pytest.raises(ValueError, match="anisotropy_ratio"):
        Patch2DGeometry(anisotropy_ratio=0.5)


def test_patch_2d_satisfies_protocol() -> None:
    """Phase 1 concrete must structurally satisfy the public Protocol."""
    assert isinstance(Patch2DGeometry(), GeometrySpec)


# ---------------------------------------------------------------------------
# UniformRandomFibrosis
# ---------------------------------------------------------------------------


def test_uniform_random_fibrosis_defaults_ok() -> None:
    s = UniformRandomFibrosis(density=0.3)
    assert s.density == 0.3
    assert s.type == "uniform_random_fibrosis"


@pytest.mark.parametrize("bad_density", [-0.01, 1.0, 1.5])
def test_uniform_random_fibrosis_rejects_out_of_range(bad_density: float) -> None:
    """Density must satisfy 0 <= d < 1 (closed-open)."""
    with pytest.raises(ValueError, match="density"):
        UniformRandomFibrosis(density=bad_density)


def test_uniform_random_fibrosis_satisfies_protocol() -> None:
    assert isinstance(UniformRandomFibrosis(density=0.0), SubstrateStrategy)


# ---------------------------------------------------------------------------
# PlanarEdgeStimulus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge", EDGES)
def test_planar_edge_stimulus_accepts_each_valid_edge(edge: Edge) -> None:
    s = PlanarEdgeStimulus(edge=edge)
    assert s.edge == edge
    assert s.type == "planar_edge"


def test_planar_edge_stimulus_rejects_unknown_edge() -> None:
    with pytest.raises(ValueError, match="edge"):
        PlanarEdgeStimulus(edge="diagonal")  # type: ignore[arg-type]


def test_planar_edge_stimulus_rejects_nonpositive_voltage() -> None:
    with pytest.raises(ValueError, match="voltage"):
        PlanarEdgeStimulus(edge="top", voltage=0.0)


def test_planar_edge_stimulus_rejects_thin_strip() -> None:
    with pytest.raises(ValueError, match="strip_thickness"):
        PlanarEdgeStimulus(edge="top", strip_thickness=0)


def test_planar_edge_stimulus_satisfies_protocol() -> None:
    assert isinstance(PlanarEdgeStimulus(edge="top"), ActivationSource)


def test_random_edge_returns_one_of_four() -> None:
    """random_edge should only ever pick from the four valid edges."""
    rng = np.random.default_rng(0)
    seen = {random_edge(rng) for _ in range(50)}
    # Probabilistically very likely we've seen all four in 50 draws.
    assert seen.issubset(set(EDGES))


# ---------------------------------------------------------------------------
# CenteredGrid2D
# ---------------------------------------------------------------------------


def test_centered_grid_2d_sample_basic_shape() -> None:
    """Default 5x5 grid produces 25 electrodes and 20 bipolar pairs."""
    g = Patch2DGeometry(size_mm=40.0, dr_mm=0.25)
    e = CenteredGrid2D.sample(geometry=g, rng=np.random.default_rng(0))
    assert e.n_rows == 5
    assert e.n_cols == 5
    assert e.n_electrodes == 25
    assert e.n_bipolar_pairs == 20
    assert e.positions_mm.shape == (25, 3)
    assert len(e.bipolar_pairs) == 20


def test_centered_grid_2d_height_within_range() -> None:
    """Sampled height must fall inside the configured range."""
    g = Patch2DGeometry()
    e = CenteredGrid2D.sample(
        geometry=g,
        height_mm_range=(0.2, 1.0),
        rng=np.random.default_rng(0),
    )
    assert 0.2 <= e.height_mm <= 1.0
    # All electrodes share the sampled height (uniform plane).
    np.testing.assert_allclose(e.positions_mm[:, 2], e.height_mm)


def test_centered_grid_2d_rejects_grid_that_does_not_fit() -> None:
    """A grid wider than the patch should be flagged at sample time."""
    g = Patch2DGeometry(size_mm=4.0, dr_mm=0.25)
    with pytest.raises(ValueError, match="does not fit"):
        CenteredGrid2D.sample(
            geometry=g,
            n_rows=5,
            n_cols=5,
            spacing_mm=10.0,  # 5x10 mm = 50 mm grid in a 4 mm patch
            rng=np.random.default_rng(0),
        )


def test_centered_grid_2d_rejects_bad_height_range() -> None:
    """Height range must satisfy 0 < lo <= hi."""
    g = Patch2DGeometry()
    with pytest.raises(ValueError, match="height_mm_range"):
        CenteredGrid2D.sample(
            geometry=g,
            height_mm_range=(0.0, 1.0),
            rng=np.random.default_rng(0),
        )


def test_centered_grid_2d_row_col_round_trip() -> None:
    """row_col maps flat electrode indices back to (row, col)."""
    g = Patch2DGeometry()
    e = CenteredGrid2D.sample(geometry=g, rng=np.random.default_rng(0))
    for idx in range(e.n_electrodes):
        r, c = e.row_col(idx)
        assert 0 <= r < e.n_rows
        assert 0 <= c < e.n_cols
        # Recompute the flat index from (r, c).
        assert r * e.n_cols + c == idx


def test_centered_grid_2d_bipolar_pairs_within_row() -> None:
    """Phase-1 convention: pairs are consecutive within rows."""
    g = Patch2DGeometry()
    e = CenteredGrid2D.sample(geometry=g, rng=np.random.default_rng(0))
    for a, b in e.bipolar_pairs:
        row_a, _ = e.row_col(a)
        row_b, _ = e.row_col(b)
        assert row_a == row_b
        assert b - a == 1


def test_centered_grid_2d_midpoints_shape() -> None:
    """Midpoint array shape matches the bipolar pair count."""
    g = Patch2DGeometry()
    e = CenteredGrid2D.sample(geometry=g, rng=np.random.default_rng(0))
    midpoints = e.bipolar_pair_midpoints_mm()
    assert midpoints.shape == (e.n_bipolar_pairs, 3)


def test_centered_grid_2d_midpoint_values() -> None:
    """Per-pair midpoint = (pos_a + pos_b) / 2."""
    g = Patch2DGeometry()
    e = CenteredGrid2D.sample(geometry=g, rng=np.random.default_rng(0))
    midpoints = e.bipolar_pair_midpoints_mm()
    for pair_idx, (a, b) in enumerate(e.bipolar_pairs):
        expected = 0.5 * (e.positions_mm[a] + e.positions_mm[b])
        np.testing.assert_allclose(midpoints[pair_idx], expected)


def test_centered_grid_2d_satisfies_protocol() -> None:
    g = Patch2DGeometry()
    e = CenteredGrid2D.sample(geometry=g, rng=np.random.default_rng(0))
    assert isinstance(e, ElectrodePlacement)
