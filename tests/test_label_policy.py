"""Label policy tests — Global vs Local density on crafted substrate masks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from myocard_synthetic_egm_pipeline.simulate import (
    GlobalDensityLabel,
    LocalDensityLabel,
    SimulationResult,
)

# ---------------------------------------------------------------------------
# GlobalDensityLabel
# ---------------------------------------------------------------------------


def test_global_density_labels_all_pairs_uniformly(
    four_pair_simulation_result: SimulationResult,
) -> None:
    """Every trace in one sim gets the same global-density label."""
    labels, ldict = GlobalDensityLabel(threshold=0.1).apply(four_pair_simulation_result)
    # The fixture's density_realized is 0.15 > 0.1 → every pair labeled 1.
    assert labels.tolist() == [1, 1, 1, 1]
    assert ldict == {0: "healthy", 1: "fibrotic"}


def test_global_density_label_below_threshold(
    four_pair_simulation_result: SimulationResult,
) -> None:
    """Density 0.15 is below threshold 0.2 → all pairs labeled 0."""
    labels, _ = GlobalDensityLabel(threshold=0.2).apply(four_pair_simulation_result)
    assert labels.tolist() == [0, 0, 0, 0]


def test_global_density_custom_label_names(
    four_pair_simulation_result: SimulationResult,
) -> None:
    """Custom healthy/fibrotic names surface in the labels_dict."""
    _labels, ldict = GlobalDensityLabel(
        threshold=0.1, healthy_name="normal", fibrotic_name="scar"
    ).apply(four_pair_simulation_result)
    assert ldict == {0: "normal", 1: "scar"}


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.0, 1.5])
def test_global_density_rejects_threshold_out_of_range(bad_threshold: float) -> None:
    """Threshold must satisfy 0 <= t < 1."""
    with pytest.raises(ValueError, match="threshold"):
        GlobalDensityLabel(threshold=bad_threshold)


# ---------------------------------------------------------------------------
# LocalDensityLabel
# ---------------------------------------------------------------------------


def test_local_density_all_healthy_mask(
    healthy_substrate_mask: npt.NDArray[np.int8],
    four_pair_midpoints: npt.NDArray[np.float64],
) -> None:
    """With no fibrotic cells, every pair is labeled 0 regardless of radius."""
    bipolar = np.zeros((4, 50), dtype=np.float32)
    result = SimulationResult(
        bipolar_traces=bipolar,
        fs_hz=1000.0,
        trace_duration_ms=50.0,
        bipolar_pair_midpoints_mm=four_pair_midpoints,
        substrate_mask=healthy_substrate_mask,
        substrate_mask_dr_mm=0.25,
        electrode_positions_mm=np.zeros((8, 3)),
        bipolar_pairs=tuple((i, i + 1) for i in range(4)),
        substrate_realization_metadata={"density_realized": 0.0},
        run_metadata={},
    )
    labels, _ = LocalDensityLabel(radius_mm=0.5, threshold=0.1).apply(result)
    assert labels.tolist() == [0, 0, 0, 0]


def test_local_density_picks_only_pairs_near_patch(
    four_pair_simulation_result: SimulationResult,
) -> None:
    """Pairs whose midpoint sits over the fibrotic patch get label 1;
    pairs far away get label 0.

    Fibrotic patch at mask[2:6, 2:6] → physical x/y in [0.5, 1.25] mm.
    Midpoint pairs (from the fixture):
      - pair 0: (1.0, 1.0) mm — over the patch
      - pair 1: (0.75, 1.0) mm — adjacent (still within 0.5 mm)
      - pair 2: (3.0, 3.0) mm — far
      - pair 3: (3.5, 0.5) mm — far

    A small radius (0.5 mm) catches the patch for pairs 0 + 1 only;
    pairs 2 + 3 fall outside.
    """
    labels, ldict = LocalDensityLabel(radius_mm=0.5, threshold=0.1).apply(
        four_pair_simulation_result
    )
    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[2] == 0
    assert labels[3] == 0
    assert ldict == {0: "healthy", 1: "fibrotic"}


def test_local_density_large_radius_behaves_globally(
    four_pair_simulation_result: SimulationResult,
) -> None:
    """A radius that covers the whole patch reduces to a global density check.

    The fibrotic patch is 4x4 cells = 16 cells; the interior is 14x14 = 196
    cells. Local density ≈ 16/196 ≈ 0.082 < 0.1 threshold → all pairs
    labeled 0.
    """
    labels, _ = LocalDensityLabel(radius_mm=10.0, threshold=0.1).apply(four_pair_simulation_result)
    assert labels.tolist() == [0, 0, 0, 0]


def test_local_density_rejects_3d_mask(
    four_pair_midpoints: npt.NDArray[np.float64],
) -> None:
    """A 3D substrate mask requires a 3D-aware label policy; fail loudly."""
    bipolar = np.zeros((4, 50), dtype=np.float32)
    result = SimulationResult(
        bipolar_traces=bipolar,
        fs_hz=1000.0,
        trace_duration_ms=50.0,
        bipolar_pair_midpoints_mm=four_pair_midpoints,
        substrate_mask=np.ones((8, 8, 8), dtype=np.int8),
        substrate_mask_dr_mm=0.25,
        electrode_positions_mm=np.zeros((8, 3)),
        bipolar_pairs=tuple((i, i + 1) for i in range(4)),
        substrate_realization_metadata={"density_realized": 0.0},
        run_metadata={},
    )
    with pytest.raises(ValueError, match="2-D substrate mask"):
        LocalDensityLabel(radius_mm=0.5, threshold=0.1).apply(result)


def test_local_density_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError, match="radius_mm"):
        LocalDensityLabel(radius_mm=0.0, threshold=0.1)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.0])
def test_local_density_rejects_threshold_out_of_range(bad_threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        LocalDensityLabel(radius_mm=1.0, threshold=bad_threshold)
