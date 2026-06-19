"""Builder round-trip tests.

Each builder is a pure function — same input, same output, no I/O.
The fixtures in conftest.py construct a minimal DatasetResult /
ClassifierBank that exercises every per-trace metadata field and
every top-level provenance field.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from myocard_egm_data.banks import ClassifierBank

from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig,
    DatasetResult,
    build_classifier_bank_from_dataset,
    build_clean_trace_metadata,
    build_synthetic_bank_from_classifier,
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
        sim_id=3,
        pair_idx=7,
        electrode_row=1,
        fibrosis_density_requested=0.25,
        fibrosis_density_realized=0.23,
        electrode_height_mm=0.7,
        stim_edge="top",
        sim_seed=42,
    )
    assert set(md.keys()) == {
        "sim_id",
        "pair_index",
        "electrode_row",
        "fibrosis_density_requested",
        "fibrosis_density_realized",
        "electrode_height_mm",
        "stim_edge",
        "sim_seed",
        "patient_id",
    }
    # patient_id is set to str(sim_id) so the patient-aware split treats
    # each simulation as one patient.
    assert md["patient_id"] == "3"
    assert md["pair_index"] == 7


def test_build_clean_trace_metadata_handles_null_sim_id() -> None:
    """If sim_id is missing, patient_id falls back to the empty string."""
    md = build_clean_trace_metadata(
        sim_id=None,
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


def test_synthetic_bank_premixer_no_mixer_config(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Pre-mixer SyntheticBank has no mixer config block."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert bank.mixer_config is None
    assert bank.noise_bank_source is None


def test_synthetic_bank_premixer_rejects_bad_stim_edge(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """A non-enum stim_edge in run_metadata is caught at build time."""
    small_dataset_result.results[0].run_metadata["stim_edge"] = "diagonal"
    with pytest.raises(ValueError, match="stim_edge"):
        build_synthetic_bank_from_dataset(
            dataset_result=small_dataset_result,
            config=small_dataset_config,
        )


# ---------------------------------------------------------------------------
# build_synthetic_bank_from_classifier (post-mixer)
# ---------------------------------------------------------------------------


def test_hybrid_synthetic_bank_reads_mixer_audit_fields(
    small_classifier_bank: ClassifierBank,
) -> None:
    """A hybrid ClassifierBank (with mixer audit fields stamped) converts
    cleanly to a SyntheticBank with the audit columns populated."""
    # Simulate the mixer's audit-field stamping.
    for i, trace in enumerate(small_classifier_bank.traces):
        trace.trace_metadata["snr_db"] = 12.0 + i
        trace.trace_metadata["noise_record"] = f"iaf{i + 1}_afw"
        trace.trace_metadata["noise_channel"] = "CS12"
    # And append a "mixer" provenance entry.
    from myocard_egm_data.banks import ClassifierBankMetaData

    small_classifier_bank.banks.append(
        ClassifierBankMetaData(
            bank_id=1,
            bank_type="mixer",
            bank_path="<test>",
            bank_metadata={
                "snr_db_range": [10.0, 25.0],
                "bandpass_clean": True,
                "band_hz": [30.0, 300.0],
                "noise_bank_source": "iafdb_noise_v1.h5",
                "master_seed": 0,
            },
        )
    )
    bank = build_synthetic_bank_from_classifier(hybrid_bank=small_classifier_bank)
    assert bank.traces.snr_db == [12.0, 13.0, 14.0, 15.0]
    assert bank.traces.noise_record == ["iaf1_afw", "iaf2_afw", "iaf3_afw", "iaf4_afw"]
    assert bank.traces.noise_channel == ["CS12"] * 4
    assert bank.noise_bank_source == "iafdb_noise_v1.h5"
    assert bank.mixer_config is not None
    assert bank.mixer_config["snr_db_range"] == [10.0, 25.0]


def test_hybrid_synthetic_bank_rejects_empty_bank() -> None:
    """An empty ClassifierBank has nothing to convert; surface a clear error."""
    empty = ClassifierBank(banks=[], traces=[], labels={})
    with pytest.raises(ValueError, match="no traces"):
        build_synthetic_bank_from_classifier(hybrid_bank=empty)


def test_hybrid_synthetic_bank_rejects_invalid_stim_edge(
    small_classifier_bank: ClassifierBank,
) -> None:
    """A non-enum stim_edge in trace_metadata is caught."""
    small_classifier_bank.traces[0].trace_metadata["stim_edge"] = "diagonal"
    with pytest.raises(ValueError, match="stim_edge"):
        build_synthetic_bank_from_classifier(hybrid_bank=small_classifier_bank)
