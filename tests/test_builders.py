"""Builder round-trip tests.

Each builder is a pure function — same input, same output, no I/O.
The fixtures in conftest.py construct a minimal DatasetResult /
ClassifierBank that exercises every per-trace metadata field and
every top-level provenance field.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from myocard_egm_contracts._generated.python.synthetic_bank import (
    AlievPanfilovCellModel,
    PlanarEdgeActivation,
)
from myocard_egm_data.banks import (
    read_synthetic_bank_hdf5,
    synthetic_bank_to_classifier,
    write_synthetic_bank,
)

from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig,
    DatasetResult,
    build_classifier_bank_from_dataset,
    build_clean_trace_metadata,
    build_synthetic_bank_from_dataset,
)
from myocard_synthetic_egm_pipeline.simulate.builders import AMP_TYPE

# ---------------------------------------------------------------------------
# build_clean_trace_metadata
# ---------------------------------------------------------------------------


def test_build_clean_trace_metadata_field_set() -> None:
    """Helper produces the canonical 9-key dict; downstream readers
    rely on these keys existing."""
    md = build_clean_trace_metadata(
        simulation_id=3,
        pair_idx=7,
        electrode_row=1,
        fibrosis_density_requested=0.25,
        fibrosis_density_realized=0.23,
        electrode_height_mm=0.7,
        stim_edge="top",
        sim_seed=42,
    )
    assert set(md.keys()) == {
        "simulation_id",
        "pair_index",
        "electrode_row",
        "fibrosis_density_requested",
        "fibrosis_density_realized",
        "electrode_height_mm",
        "stim_edge",
        "sim_seed",
        "patient_id",
    }
    # patient_id is set to str(simulation_id) so the patient-aware split treats
    # each simulation as one patient.
    assert md["patient_id"] == "3"
    assert md["pair_index"] == 7


def test_build_clean_trace_metadata_handles_null_sim_id() -> None:
    """If simulation_id is missing, patient_id falls back to the empty string."""
    md = build_clean_trace_metadata(
        simulation_id=None,
        pair_idx=0,
        electrode_row=None,
        fibrosis_density_requested=None,
        fibrosis_density_realized=None,
        electrode_height_mm=None,
        stim_edge=None,
        sim_seed=None,
    )
    assert md["patient_id"] == ""


# ---------------------------------------------------------------------------
# build_classifier_bank_from_dataset
# ---------------------------------------------------------------------------


def test_classifier_bank_trace_count_matches_dataset(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Bank carries one ClassifierTrace per bipolar pair across all sims."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
    )
    assert len(bank.traces) == sum(r.n_pairs for r in small_dataset_result.results)


def test_classifier_bank_labels_propagate(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Per-trace label_truth matches the flat labels array."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
    )
    for trace, expected in zip(bank.traces, small_dataset_result.labels, strict=True):
        assert trace.label_truth == int(expected)


def test_classifier_bank_metadata_includes_provenance(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Bank-level metadata stamps the simulator name + config knobs."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
        description="round-trip test",
    )
    assert len(bank.banks) == 1
    meta = bank.banks[0]
    assert meta.bank_type == "synthetic_egm_pipeline"
    assert meta.bank_metadata["description"] == "round-trip test"
    assert meta.bank_metadata["backend"] == "mock"
    assert meta.bank_metadata["cell_model"] == "aliev_panfilov"
    assert meta.bank_metadata["n_simulations"] == 3
    assert meta.bank_metadata["geometry_size_mm"] == 4.0


def test_classifier_bank_amp_type_is_synthetic_au(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Every trace's amp_type advertises the synthetic_au convention."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
    )
    assert all(t.amp_type == AMP_TYPE for t in bank.traces)


# ---------------------------------------------------------------------------
# build_synthetic_bank_from_dataset (pre-mixer)
# ---------------------------------------------------------------------------


def test_synthetic_bank_premixer_columns(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Pre-mixer SyntheticBank has snr_db = NaN and empty noise audit columns."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert all(math.isnan(v) for v in bank.traces.snr_db)
    assert all(s == "" for s in bank.traces.noise_record)
    assert all(s == "" for s in bank.traces.noise_channel)


def test_synthetic_bank_writes_per_simulation_config(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The generation config is one typed row per simulation, not per trace.

    This is the whole point of the 2.0 restructure: 1.1 repeated
    ``fibrosis_density`` / ``stim_edge`` / ``electrode_height_mm`` on
    every trace of a simulation, inviting the reading that traces from
    one simulation differed in those respects.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    sims = bank.simulations
    n_sims = len(small_dataset_result.results)

    assert len(sims.simulation_id) == n_sims
    assert len(bank.traces.signal) > n_sims  # many traces per simulation

    # Each per-function column decodes to its typed, discriminated object.
    assert sims.geometry[0].root.type == "patch_2d"
    assert sims.substrate[0].root.type == "uniform_random_fibrosis"
    assert sims.activation[0].type == "planar_edge"
    assert sims.electrodes[0].root.type == "centered_grid_2d"
    assert sims.backend[0].root.type == "finitewave"
    assert sims.cell_model[0].type == "aliev_panfilov"


def test_synthetic_bank_activation_edges_is_a_list(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``planar_edge.edges`` is a list even for today's single edge (CL-087).

    The schema is deliberately ahead of the producer here so SEP6's
    multi-edge stimulation needs no contracts bump.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    activation = bank.simulations.activation[0]
    assert isinstance(activation, PlanarEdgeActivation)
    assert isinstance(activation.edges, list)
    assert len(activation.edges) == 1


def test_synthetic_bank_label_policy_uses_thresholds_list(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The policy carries ``thresholds[]`` and no class names (CL-088).

    Names live once in the per-simulation ``label_names`` map; writing
    them into the policy too would let the two disagree with nothing to
    catch it. A one-element list is the binary case; Phase-2 multiclass
    is a longer list rather than a new schema variant.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    policy = bank.simulations.label_policy[0]
    assert isinstance(policy.thresholds, list)
    assert len(policy.thresholds) == 1
    assert not hasattr(policy, "healthy_name")
    assert not hasattr(policy, "fibrotic_name")
    assert bank.simulations.label_names[0] == {"0": "healthy", "1": "fibrotic"}


def test_synthetic_bank_electrode_pairs_are_per_simulation(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The realized per-pair detail 1.1 repeated per trace lives once per sim."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    pairs = bank.simulations.electrodes[0].root.pairs
    n_pairs_first_sim = small_dataset_result.results[0].n_pairs

    assert len(pairs) == n_pairs_first_sim
    assert [p.pair_index for p in pairs] == list(range(n_pairs_first_sim))
    assert pairs[0].midpoint_mm is not None
    # Traces index this list by pair_index — the FK the restructure adds.
    # Constrained scalar columns codegen to RootModel wrappers, so the
    # value is under .root (egm-data has an _unwrap helper for the same).
    assert max(int(p.root) for p in bank.traces.pair_index) < len(pairs)


def test_synthetic_bank_trace_columns_are_collapsed(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``traces/`` keeps only the signal, the two FKs, the label and noise."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    n = len(bank.traces.signal)
    assert len(bank.traces.simulation_id) == n
    assert len(bank.traces.pair_index) == n
    assert len(bank.traces.label) == n
    # The 1.1 generation columns are gone, not relocated.
    for removed in (
        "fibrosis_density",
        "fibrosis_density_realized",
        "electrode_row",
        "electrode_height_mm",
        "stim_edge",
        "seed",
    ):
        assert not hasattr(bank.traces, removed), removed


def test_synthetic_bank_labels_match_the_dataset(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Per-trace labels are plain ints matching the DatasetResult exactly."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert [int(x.root) for x in bank.traces.label] == [int(x) for x in small_dataset_result.labels]


def test_synthetic_bank_activation_position_absent_in_wave_1(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """No controlled crop yet, so the column is absent — never 0.0.

    The schema is explicit that absence means *unknown*; 0.0 is a
    legitimate position (activation on the first sample), so defaulting
    would fabricate a spike at the low edge of the distribution. SEP2
    populates it.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert bank.traces.activation_position is None


def test_theta_spec_records_the_regime_with_no_knobs(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Wave 1 sweeps nothing, but the regime is still stated explicitly."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    theta = bank.generation_params
    assert theta.knobs == []
    assert theta.regime["geometry"] == "patch_2d"
    assert theta.regime["substrate"] == "uniform_random_fibrosis"
    assert theta.regime["cell_model"] == "aliev_panfilov"


def test_synthetic_bank_noise_columns_when_mixed(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The inline mixer path supplies signals + the three noise columns.

    2.0's per-simulation config cannot be rebuilt from a mixed
    ClassifierBank, so the noise-mixed bank is written from the same
    DatasetResult with only the signals and noise provenance replaced.
    """
    n = len(small_dataset_result.labels)
    mixed = [
        np.full(small_dataset_result.results[0].n_samples, 0.5, dtype=np.float32) for _ in range(n)
    ]
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        mixed_signals=mixed,
        snr_db=[12.0] * n,
        noise_record=["iaf3_afw"] * n,
        noise_channel=["CS12"] * n,
        noise_bank_source="iafdb v1.0.0",
    )
    assert bank.noise_bank_source == "iafdb v1.0.0"
    assert all(v == 12.0 for v in bank.traces.snr_db)
    assert all(r == "iaf3_afw" for r in bank.traces.noise_record)
    # The generation config is still the clean run's — only signals changed.
    assert len(bank.simulations.simulation_id) == len(small_dataset_result.results)


def test_synthetic_bank_rejects_mismatched_mixed_signal_count(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """A short mixed-signal list is a bug, not something to pad around."""
    with pytest.raises(ValueError, match="mixed_signals"):
        build_synthetic_bank_from_dataset(
            dataset_result=small_dataset_result,
            config=small_dataset_config,
            mixed_signals=[np.zeros(4, dtype=np.float32)],
        )


def test_clean_trace_metadata_uses_simulation_id() -> None:
    """The ClassifierBank's per-trace join key is ``simulation_id``.

    egm-data's ``synthetic_bank_to_classifier`` writes ``simulation_id``;
    the producer's direct-write path used to write ``sim_id``, so a
    ClassifierBank carried a differently-named join key depending on
    which of the two paths produced it (CL-008 / CL-024 §3). Both write
    one name now, and the T4 bank-to-bank join depends on it.
    """
    meta = build_clean_trace_metadata(
        simulation_id=3,
        pair_idx=1,
        electrode_row=0,
        fibrosis_density_requested=0.2,
        fibrosis_density_realized=0.21,
        electrode_height_mm=0.5,
        stim_edge="top",
        sim_seed=42,
    )

    assert meta["simulation_id"] == 3
    assert "sim_id" not in meta
    # patient_id is derived from the same value — the patient-aware split
    # treats one simulation as one patient.
    assert meta["patient_id"] == "3"


# ---------------------------------------------------------------------------
# Round-trip through the real egm-data writer/reader
# ---------------------------------------------------------------------------


def test_synthetic_bank_round_trips_through_egm_data(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
    tmp_path: Path,
) -> None:
    """A written 2.0 bank reads back with its config and FKs intact.

    The unit tests above check the in-memory model; this is the one that
    proves the producer and egm-data agree on the on-disk layout — the
    Wave-1 gate's real question. It also exercises the writer's two
    refusals (no bank_id, orphan trace) by not tripping them.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        description="round-trip fixture",
    )
    path = write_synthetic_bank(bank, tmp_path / "rt.synthetic.h5")
    reloaded = read_synthetic_bank_hdf5(path)

    assert str(reloaded.schema_version.value) == "2.0"
    assert reloaded.bank_id == bank.bank_id
    assert len(reloaded.simulations.simulation_id) == len(bank.simulations.simulation_id)
    assert len(reloaded.traces.signal) == len(bank.traces.signal)
    # The typed per-function config survives the JSON-column encoding.
    assert reloaded.simulations.geometry[0].root.size_mm == pytest.approx(
        bank.simulations.geometry[0].root.size_mm
    )
    reloaded_activation = reloaded.simulations.activation[0]
    written_activation = bank.simulations.activation[0]
    assert isinstance(reloaded_activation, PlanarEdgeActivation)
    assert isinstance(written_activation, PlanarEdgeActivation)
    assert reloaded_activation.edges == written_activation.edges
    assert reloaded.simulations.label_policy[0].thresholds == (
        bank.simulations.label_policy[0].thresholds
    )
    # Every trace's FK resolves into simulations/ — the invariant JSON
    # Schema cannot express and the restructure exists to guarantee.
    sim_ids = {int(x) for x in reloaded.simulations.simulation_id}
    assert {int(x) for x in reloaded.traces.simulation_id} <= sim_ids


def test_written_bank_converts_to_a_classifier_bank(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
    tmp_path: Path,
) -> None:
    """egm-data can derive a ClassifierBank from what we write.

    We do **not** use this path in the producer (design note D7 — the
    converter hardcodes ``amp_type="mv"``, which is wrong for
    relative-unit synthetic traces; see FB-17). But a consumer may, so
    the bank we emit has to be convertible: labels come off the bank
    itself and the join keys land in ``trace_metadata``.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    path = write_synthetic_bank(bank, tmp_path / "conv.synthetic.h5")
    cb = synthetic_bank_to_classifier(read_synthetic_bank_hdf5(path), bank_path=path)

    assert len(cb.traces) == len(bank.traces.signal)
    assert cb.traces[0].trace_metadata["simulation_id"] == 0
    assert cb.traces[0].trace_metadata["patient_id"] == "0"
    assert cb.labels == {0: "healthy", 1: "fibrotic"}


def test_backend_object_fills_its_typed_fields_from_metadata(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Known backend facts land in named fields, not the generic bag.

    The backend reports its version as ``finitewave_version_pin`` and
    its timestep as ``ap_dt_model_units``; the schema has ``version``
    and ``dt_model_units`` for exactly those. Leaving them in ``params``
    would defeat the reason 2.0 has typed fields at all.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    backend = bank.simulations.backend[0].root

    assert backend.version is not None
    assert backend.dt_model_units is not None
    assert backend.output_fs_hz == small_dataset_config.run_config.output_fs_hz
    # ap_time_unit_ms belongs to the cell model, not the backend.
    assert "ap_time_unit_ms" not in (backend.params or {})
    cell_model = bank.simulations.cell_model[0]
    assert isinstance(cell_model, AlievPanfilovCellModel)
    assert cell_model.ap_time_unit_ms is not None


def test_substrate_summary_carries_the_realized_draw(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The realized density/count are recorded, not just the request.

    Requested and realized genuinely differ through grid discretization,
    and the label is computed from the realized one — so a bank that
    stored only the request could not explain its own labels.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    summary = bank.simulations.substrate_summary[0]
    assert summary.realized_density is not None
    assert summary.n_fibrotic_nodes is not None


def test_seed_column_carries_the_run_master_seed(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``simulations/seed`` is the run's master seed, repeated per row.

    It is a bank-scoped fact that 2.0 placed in the per-simulation group
    by oversight (CL-096); FB-16 moves it to a root attr. Writing the
    per-simulation derived seed here instead would change the column's
    meaning with no version bump to signal it — so the repetition is
    deliberate, not an accident of the loop.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    seeds = [int(x) for x in bank.simulations.seed]

    assert seeds == [small_dataset_config.master_seed] * len(small_dataset_result.results)
