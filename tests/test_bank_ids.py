"""Stable cross-artifact id stamping (synthetic-egm-pipeline v0.3.0).

Covers the ids.py derive/validate/resolve helpers, the clean-path id
stamping (ClassifierBank + SyntheticBank), and the noise_mixed-path mixer ids
(noise-source entry read from the sidecar, noise_mixed bank's own derived id).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_data.banks import ClassifierBank
from myocard_egm_data.records import build_noise_bank_run_record, write_noise_bank_run_record

from myocard_synthetic_egm_pipeline.ids import (
    derive_noise_ref_id,
    derive_synthetic_bank_id,
    resolve_noise_bank_id,
    validate_artifact_id,
)
from myocard_synthetic_egm_pipeline.mixer import MixerConfig, mix_classifier_bank
from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig,
    DatasetResult,
    build_classifier_bank_from_dataset,
    build_synthetic_bank_from_dataset,
)


def _write_noise_sidecar(noise_path: Path, bank_id: str) -> None:
    """Write an iafdb-style noise run-record sidecar next to ``noise_path``."""
    record = build_noise_bank_run_record(
        source="iafdb v1.0.0",
        fs_hz=1000.0,
        window_ms=200.0,
        window_samples=200,
        hop_ms=100.0,
        band_hz=[30.0, 300.0],
        calibration_method="none",
        calibration_target_qrs_pp_mv=None,
        threshold_mode="percentile",
        threshold_value=20.0,
        source_records=["iaf1_afw"],
        bank_id=bank_id,
    )
    write_noise_bank_run_record(noise_path.with_name(noise_path.stem + "_run_record.json"), record)


# ---------------------------------------------------------------------------
# ids.py units
# ---------------------------------------------------------------------------


def test_derive_synthetic_bank_id_clean_and_noise_mixed() -> None:
    clean = derive_synthetic_bank_id("aliev_panfilov")
    noise_mixed = derive_synthetic_bank_id("aliev_panfilov", noise_mixed=True)
    assert clean.startswith("tbank_synthetic_aliev_panfilov_")
    assert noise_mixed.startswith("tbank_synthetic_aliev_panfilov_noise_mixed_")
    # Both must satisfy the egm-contracts ArtifactId pattern.
    validate_artifact_id(clean)
    validate_artifact_id(noise_mixed)


def test_validate_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="valid stable artifact id"):
        validate_artifact_id("NOT VALID")


def test_resolve_noise_bank_id_override() -> None:
    assert (
        resolve_noise_bank_id(None, override="nbank_iafdb_2026-06-27") == "nbank_iafdb_2026-06-27"
    )


def test_resolve_noise_bank_id_derived_when_no_sidecar(tmp_path: Path) -> None:
    assert resolve_noise_bank_id(tmp_path / "noise.h5") == derive_noise_ref_id()


def test_resolve_noise_bank_id_reads_sidecar(tmp_path: Path) -> None:
    noise_path = tmp_path / "iafdb_noise_v1.h5"
    _write_noise_sidecar(noise_path, "nbank_iafdb_2026-06-15")
    assert resolve_noise_bank_id(noise_path) == "nbank_iafdb_2026-06-15"


# ---------------------------------------------------------------------------
# clean path
# ---------------------------------------------------------------------------


def test_clean_classifier_bank_stamps_derived_id(
    small_dataset_result: DatasetResult, small_dataset_config: DatasetConfig
) -> None:
    """The clean ClassifierBank's own id, its source entry, and every trace
    all carry the derived synthetic id."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<t>"),
    )
    expected = derive_synthetic_bank_id("aliev_panfilov")
    assert bank.id == expected
    assert bank.banks[0].bank_id == expected
    assert all(t.bank_id == expected for t in bank.traces)


def test_clean_classifier_bank_override(
    small_dataset_result: DatasetResult, small_dataset_config: DatasetConfig
) -> None:
    explicit = "tbank_synthetic_courtemanche_v1_5_2026-06-27"
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<t>"),
        bank_id=explicit,
    )
    assert bank.id == explicit
    assert all(t.bank_id == explicit for t in bank.traces)


def test_clean_classifier_bank_rejects_malformed_id(
    small_dataset_result: DatasetResult, small_dataset_config: DatasetConfig
) -> None:
    with pytest.raises(ValueError, match="valid stable artifact id"):
        build_classifier_bank_from_dataset(
            dataset_result=small_dataset_result,
            config=small_dataset_config,
            bank_path=Path("<t>"),
            bank_id="bad id",
        )


def test_synthetic_bank_stamps_id(
    small_dataset_result: DatasetResult, small_dataset_config: DatasetConfig
) -> None:
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert bank.bank_id == derive_synthetic_bank_id("aliev_panfilov")


# ---------------------------------------------------------------------------
# noise_mixed path (mixer)
# ---------------------------------------------------------------------------


def test_mixer_noise_entry_id_derived_without_sidecar(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
        noise_bank_path="/no/such/noise.h5",
    )
    assert noise_mixed.banks[1].bank_type == "mixer"
    assert noise_mixed.banks[1].bank_id == derive_noise_ref_id()


def test_mixer_noise_entry_id_from_sidecar(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank, tmp_path: Path
) -> None:
    noise_path = tmp_path / "iafdb_noise_v1.h5"
    _write_noise_sidecar(noise_path, "nbank_iafdb_2026-06-15")
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
        noise_bank_path=str(noise_path),
    )
    assert noise_mixed.banks[1].bank_id == "nbank_iafdb_2026-06-15"


def test_mixer_noise_mixed_bank_id_derived(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """The noise_mixed bank gets its own _noise_mixed id derived from the clean
    source's cell model, and the mixed traces keep the clean bank's id."""
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
    )
    assert noise_mixed.id == derive_synthetic_bank_id("aliev_panfilov", noise_mixed=True)
    # Traces still reference the clean source bank's id (additive noise).
    assert all(t.bank_id == small_classifier_bank.traces[0].bank_id for t in noise_mixed.traces)


def test_mixer_noise_mixed_bank_id_override(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    explicit = "tbank_synthetic_aliev_panfilov_noise_mixed_v2_2026-06-27"
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
        noise_mixed_bank_id=explicit,
    )
    assert noise_mixed.id == explicit
