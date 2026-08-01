"""Stable cross-artifact id stamping (synthetic-egm-pipeline v0.3.0).

Covers the ids.py derive/validate/resolve helpers, the clean-path id
stamping (ClassifierBank + SyntheticBank), and the noise_mixed-path mixer ids
(noise-source entry read from the sidecar, noise_mixed bank's own derived id).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_data.banks import ClassifierBank, ClassifierBankMetaData
from myocard_egm_data.records import build_noise_bank_run_record, write_noise_bank_run_record

from myocard_synthetic_egm_pipeline.constants import THETA_BANK_SOURCE
from myocard_synthetic_egm_pipeline.ids import (
    derive_noise_ref_id,
    derive_synthetic_bank_id,
    noise_mixed_id_from,
    resolve_noise_bank_id,
    theta_bank_id_from,
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
    """The synthetic bank stamps the run's base id plus the theta marker.

    It is derived from the same base as the paired ClassifierBank rather
    than independently, so the two are distinct but visibly one run's
    output.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert bank.bank_id == theta_bank_id_from(derive_synthetic_bank_id("aliev_panfilov"))


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
    """The mixed bank's id is derived from the clean bank's **id**.

    It used to be re-derived from the cell model read out of the clean
    bank's ``bank_metadata``; that coupling broke silently when the
    generation parameters were cleaned off the ClassifierBank, yielding
    ``synthetic_unknown_noise_mixed``. One string now anchors the family.
    """
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
    )
    assert noise_mixed.id == noise_mixed_id_from(str(small_classifier_bank.id))
    assert "unknown" not in str(noise_mixed.id)
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


# ---------------------------------------------------------------------------
# The two banks a run writes take distinct ids from one base
# ---------------------------------------------------------------------------


def test_theta_marker_goes_before_a_trailing_date() -> None:
    """The marker is part of the descriptive name, not appended after the date.

    ``ArtifactId`` is ``<prefix>_<descriptive_name>[_<YYYY-MM-DD>]``, so a
    marker tacked on after the date would put the id outside the grammar.
    """
    assert (
        theta_bank_id_from("tbank_synthetic_aliev_panfilov_2026-08-01")
        == "tbank_synthetic_aliev_panfilov_theta_2026-08-01"
    )


def test_theta_marker_appends_when_there_is_no_date() -> None:
    """Dates are optional on a hand-set id; the marker still lands correctly."""
    assert theta_bank_id_from("tbank_run7") == "tbank_run7_theta"


def test_theta_id_composes_with_the_noise_mixed_marker() -> None:
    """A noise-mixed run keeps both markers, in a stable order."""
    assert theta_bank_id_from("tbank_synthetic_ap_noise_mixed_2026-08-01") == (
        "tbank_synthetic_ap_noise_mixed_theta_2026-08-01"
    )


def test_derived_ids_differ_between_the_two_banks(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Without an override, the pair still gets distinct ids.

    The phase manifest keys artifacts by stable id, so two files sharing
    one id collide. Before both banks were always written this was
    harmless — the synthetic bank was an optional view of the same data —
    but they are parallel artifacts now.
    """
    classifier = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("ids.classifier.h5"),
    )
    synthetic = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )

    assert classifier.id != synthetic.bank_id
    assert synthetic.bank_id == theta_bank_id_from(str(classifier.id))


def test_one_override_names_both_banks(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``output.bank_id`` is the base for the pair, not just the ClassifierBank.

    One knob keeps the two in step. Two independent overrides would let a
    caller set one and leave the other derived, producing a pair that
    looks unrelated with nothing to detect it.
    """
    classifier = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("ids.classifier.h5"),
        bank_id="tbank_run7",
    )
    synthetic = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_id="tbank_run7",
    )

    assert str(classifier.id) == "tbank_run7"
    assert synthetic.bank_id == "tbank_run7_theta"


# ---------------------------------------------------------------------------
# The companion entry must name the theta bank actually written beside it
# ---------------------------------------------------------------------------


def test_mixed_bank_companion_names_the_mixed_theta_bank(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """Mixing re-points the theta companion at the *mixed* run's theta bank.

    The mixer copies the clean bank's provenance entries forward, but the
    theta companion is not shared history: a noise-mixed run writes its
    own ``synthetic_bank`` under its own id. Carrying the clean entry
    through left the mixed ClassifierBank naming a *different* artifact
    than the one written beside it — exactly the mispairing this entry
    exists to prevent.
    """
    clean = ClassifierBank(
        id=small_classifier_bank.id,
        banks=[
            *small_classifier_bank.banks,
            ClassifierBankMetaData(
                bank_id=theta_bank_id_from(str(small_classifier_bank.id)),
                bank_type=THETA_BANK_SOURCE,
                bank_path="clean.synthetic.h5",
                bank_metadata={"join_key": "simulation_id"},
            ),
        ],
        traces=small_classifier_bank.traces,
        labels=small_classifier_bank.labels,
    )

    mixed = mix_classifier_bank(
        clean_bank=clean,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
    )

    companion = next(b for b in mixed.banks if b.bank_type == THETA_BANK_SOURCE)
    assert companion.bank_id == theta_bank_id_from(str(mixed.id))
    # i.e. it moved off the clean run's theta bank
    assert companion.bank_id != theta_bank_id_from(str(clean.id))


def test_noise_mixed_id_is_idempotent() -> None:
    """Re-mixing an already-mixed id must not stack the marker."""
    once = noise_mixed_id_from("tbank_synthetic_ap_2026-08-01")

    assert noise_mixed_id_from(once) == once


def test_mixer_refuses_a_clean_bank_with_no_id(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """An id-less clean bank fails loudly instead of yielding ``unknown``.

    egm-data refuses to *write* a bank without an id, so an id-less bank
    is never a real artifact; deriving a plausible-looking id from it was
    how ``synthetic_unknown_noise_mixed`` got produced.
    """
    nameless = ClassifierBank(
        banks=small_classifier_bank.banks,
        traces=small_classifier_bank.traces,
        labels=small_classifier_bank.labels,
    )

    with pytest.raises(ValueError, match="carries no id"):
        mix_classifier_bank(
            clean_bank=nameless,
            noise_bank=small_noise_bank,
            config=MixerConfig(show_progress=False),
        )
