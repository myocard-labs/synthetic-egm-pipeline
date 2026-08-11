"""Tests for run_single using a MockBackend fixture.

The MockBackend (in conftest.py) returns deterministic canned
RawSimulationResults so we can exercise the runner's post-processing
(bipolar pairing, downsampling, midpoint computation, run_metadata
stamping) without spinning up Finitewave.
"""

from __future__ import annotations

import numpy as np

from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend
from myocard_synthetic_egm_pipeline.constants import DEFAULT_TRACE_DURATION_MS
from myocard_synthetic_egm_pipeline.simulate import (
    CenteredGrid2D,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    UniformRandomFibrosis,
    run_single,
)


def _build_run_inputs() -> tuple[Patch2DGeometry, CenteredGrid2D, RunConfig]:
    """Common run inputs reused across the runner tests."""
    geometry = Patch2DGeometry(size_mm=40.0, dr_mm=0.25)
    electrodes = CenteredGrid2D.sample(geometry=geometry, rng=np.random.default_rng(0))
    config = RunConfig(
        trace_duration_ms=DEFAULT_TRACE_DURATION_MS,
        output_fs_hz=1000.0,
        ap_time_unit_ms=1.97,
        capture_oversample=4,
    )
    return geometry, electrodes, config


def test_run_single_output_shape(mock_backend: SimulationBackend) -> None:
    """SimulationResult.bipolar_traces is (n_pairs, T_target_samples)."""
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.2),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    expected_t = round(config.trace_duration_ms * 1e-3 * config.output_fs_hz)
    assert result.bipolar_traces.shape == (electrodes.n_bipolar_pairs, expected_t)


def test_run_single_pinned_to_exact_duration(mock_backend: SimulationBackend) -> None:
    """The runner pins T to exactly trace_duration_ms * output_fs_hz."""
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.2),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert result.trace_duration_ms == config.trace_duration_ms
    assert result.fs_hz == config.output_fs_hz


def test_run_single_midpoints_shape(mock_backend: SimulationBackend) -> None:
    """Midpoint array shape matches the bipolar pair count."""
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert result.bipolar_pair_midpoints_mm.shape == (electrodes.n_bipolar_pairs, 3)


def test_run_single_run_metadata_keys(mock_backend: SimulationBackend) -> None:
    """run_metadata carries the per-sim provenance fields downstream consumers expect."""
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.35),
        activation=PlanarEdgeStimulus(edge="bottom"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    md = result.run_metadata
    # Strategy-type discriminators land verbatim.
    assert md["geometry_type"] == "patch_2d"
    assert md["substrate_type"] == "uniform_random_fibrosis"
    assert md["activation_type"] == "planar_edge"
    assert md["electrode_type"] == "centered_grid_2d"
    # Per-sim sampled scalars duck-type-extracted from the input specs.
    assert md["stim_edge"] == "bottom"
    assert md["fibrosis_density_requested"] == 0.35
    assert md["electrode_height_mm"] == electrodes.height_mm
    # Backend metadata round-trips into run_metadata.
    assert md["backend_metadata"]["backend_name"] == "mock"


def test_run_single_substrate_mask_forwarded(mock_backend: SimulationBackend) -> None:
    """The substrate mask survives the runner unchanged.

    Label policies need the mask to compute neighborhood statistics
    without re-running the simulation. Verify the mask's shape +
    resolution match what the (mock) backend produced.
    """
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert result.substrate_mask.shape == geometry.shape
    assert result.substrate_mask_dr_mm == geometry.dr_mm


def test_run_single_electrode_row_per_pair(mock_backend: SimulationBackend) -> None:
    """For a CenteredGrid2D, the runner extracts row indices per pair."""
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    rows = result.run_metadata["electrode_row_per_pair"]
    assert len(rows) == electrodes.n_bipolar_pairs
    # Phase-1 CenteredGrid2D has 5 rows x 4 pairs per row = 20 pairs;
    # each row should appear exactly four times.
    expected = sorted([r for r in range(electrodes.n_rows) for _ in range(electrodes.n_cols - 1)])
    assert sorted(rows) == expected


def test_run_single_carries_the_realized_specs(mock_backend: SimulationBackend) -> None:
    """``SimulationResult.specs`` holds the exact spec objects passed in.

    Identity, not equality: ``synthetic_bank`` 2.0 serializes the
    per-simulation config from these, so the result must carry the
    realized objects themselves rather than reconstructed copies. The
    per-sim sampled values (density, edge, height) live only here and in
    ``run_metadata``'s flattened view.
    """
    geometry, electrodes, config = _build_run_inputs()
    substrate = UniformRandomFibrosis(density=0.42)
    activation = PlanarEdgeStimulus(edge="left")

    result = run_single(
        geometry=geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )

    assert result.specs.geometry is geometry
    assert result.specs.substrate is substrate
    assert result.specs.activation is activation
    assert result.specs.electrodes is electrodes
    # The sampled per-sim values are recoverable from the specs, which is
    # the point — run_metadata's copies are a lossy convenience view.
    assert result.specs.substrate.density == 0.42
    assert result.specs.activation.edge == "left"
    assert result.specs.electrodes.height_mm == electrodes.height_mm


def test_run_metadata_join_key_is_simulation_id(mock_backend: SimulationBackend) -> None:
    """The join key is spelled ``simulation_id`` everywhere (CL-024 §3).

    ``synthetic_bank`` 2.0, egm-data's converter and the T4 bank-to-bank
    join all key on ``simulation_id``; the producer used to write
    ``sim_id`` on its direct path, so the same artifact carried two names
    depending on which writer made it.
    """
    geometry, electrodes, config = _build_run_inputs()
    result = run_single(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=0.1),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=electrodes,
        backend=mock_backend,
        config=config,
        rng=np.random.default_rng(0),
    )
    result.run_metadata["simulation_id"] = 7

    assert "sim_id" not in result.run_metadata
    assert result.run_metadata["simulation_id"] == 7
