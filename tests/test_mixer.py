"""Mixer tests — pure math + full bank-mixing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_data.banks import ClassifierBank

from myocard_synthetic_egm_pipeline.mixer import (
    MixerConfig,
    mix_classifier_bank,
    sample_noise_for_length,
    snr_scale,
)

# ---------------------------------------------------------------------------
# snr_scale
# ---------------------------------------------------------------------------


def test_snr_scale_hits_target_for_well_conditioned_signals() -> None:
    """For a sine signal + Gaussian noise, the realized SNR after applying
    the computed alpha must match the target within rounding noise."""
    rng = np.random.default_rng(0)
    sig = np.sin(np.linspace(0, 4 * np.pi, 1000)).astype(np.float64)
    noise = rng.standard_normal(1000)
    target_snr_db = 20.0
    alpha = snr_scale(sig, noise, target_snr_db=target_snr_db)
    # Compute realized SNR from definitions: 10 log10(P_s / (alpha^2 P_n)).
    p_s = float(np.mean(sig**2))
    p_n_scaled = float(np.mean((alpha * noise) ** 2))
    realized_snr_db = 10.0 * np.log10(p_s / p_n_scaled)
    assert abs(realized_snr_db - target_snr_db) < 0.01


def test_snr_scale_zero_noise_returns_zero_alpha() -> None:
    """Zero-power noise → alpha = 0 (no-op mix; trace stays clean)."""
    sig = np.ones(100, dtype=np.float64)
    noise = np.zeros(100, dtype=np.float64)
    assert snr_scale(sig, noise, target_snr_db=10.0) == 0.0


def test_snr_scale_zero_signal_returns_unit_alpha() -> None:
    """Zero-power signal → alpha = 1 (avoids div-by-zero; mix is raw noise)."""
    sig = np.zeros(100, dtype=np.float64)
    noise = np.ones(100, dtype=np.float64)
    assert snr_scale(sig, noise, target_snr_db=10.0) == 1.0


# ---------------------------------------------------------------------------
# sample_noise_for_length
# ---------------------------------------------------------------------------


def test_sample_noise_for_length_matches_when_lengths_equal(
    small_noise_bank: NoiseBank,
) -> None:
    """When n_samples matches the bank segment length, the sample is returned as-is."""
    rng = np.random.default_rng(0)
    sig, record, channel = sample_noise_for_length(
        noise_bank=small_noise_bank, n_samples=200, rng=rng
    )
    assert sig.shape == (200,)
    # Audit fields come straight from the bank's source columns.
    assert record in set(small_noise_bank.traces.source_record)
    assert channel in set(small_noise_bank.traces.source_channel)


def test_sample_noise_for_length_tiles_short_segments(
    small_noise_bank: NoiseBank,
) -> None:
    """Asking for a longer trace than the bank's segments triggers tiling."""
    rng = np.random.default_rng(0)
    sig, _, _ = sample_noise_for_length(noise_bank=small_noise_bank, n_samples=400, rng=rng)
    assert sig.shape == (400,)


def test_sample_noise_for_length_crops_long_segments() -> None:
    """A short request from a long-segment bank crops to exact length."""
    from datetime import datetime, timezone

    from myocard_egm_contracts._generated.python.noise_bank import (
        SchemaVersion,
    )
    from myocard_egm_contracts._generated.python.noise_bank import (
        Traces as NoiseTraces,
    )

    # A bank with 500-sample segments.
    long_bank = NoiseBank(
        schema_version=SchemaVersion.field_1_0,
        created_utc=datetime.now(timezone.utc),
        source="test",
        fs_hz=1000.0,
        traces=NoiseTraces(
            signal=[np.zeros(500).tolist() for _ in range(4)],
            source_record=["r"] * 4,
            source_channel=["c"] * 4,
        ),
    )
    rng = np.random.default_rng(0)
    sig, _, _ = sample_noise_for_length(noise_bank=long_bank, n_samples=200, rng=rng)
    assert sig.shape == (200,)


def test_sample_noise_for_length_rejects_empty_bank() -> None:
    """An empty noise bank is a fatal mixer config error."""
    from datetime import datetime, timezone

    from myocard_egm_contracts._generated.python.noise_bank import SchemaVersion
    from myocard_egm_contracts._generated.python.noise_bank import (
        Traces as NoiseTraces,
    )

    empty = NoiseBank(
        schema_version=SchemaVersion.field_1_0,
        created_utc=datetime.now(timezone.utc),
        source="test",
        fs_hz=1000.0,
        traces=NoiseTraces(signal=[], source_record=[], source_channel=[]),
    )
    with pytest.raises(ValueError, match="empty"):
        sample_noise_for_length(
            noise_bank=empty,
            n_samples=200,
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# MixerConfig validation
# ---------------------------------------------------------------------------


def test_mixer_config_rejects_inverted_snr_range() -> None:
    with pytest.raises(ValueError, match="snr_db_range"):
        MixerConfig(snr_db_range=(25.0, 10.0))


def test_mixer_config_rejects_bad_band() -> None:
    with pytest.raises(ValueError, match="band_hz"):
        MixerConfig(band_hz=(300.0, 30.0))


# ---------------------------------------------------------------------------
# mix_classifier_bank
# ---------------------------------------------------------------------------


def test_mix_classifier_bank_stamps_audit_fields(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """Every output trace gains snr_db / noise_record / noise_channel."""
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(snr_db_range=(15.0, 15.0), show_progress=False),
    )
    for trace in noise_mixed.traces:
        assert "snr_db" in trace.trace_metadata
        assert "noise_record" in trace.trace_metadata
        assert "noise_channel" in trace.trace_metadata
        # Fixed-SNR range collapses to a point.
        assert trace.trace_metadata["snr_db"] == 15.0


def test_mix_classifier_bank_appends_mixer_provenance(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """A 'mixer' ClassifierBankMetaData entry is appended; the clean entry survives."""
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
        noise_bank_path="/path/to/noise_bank.h5",
    )
    # Two entries: the original synthetic source + the new mixer entry.
    assert len(noise_mixed.banks) == 2
    mixer_entry = noise_mixed.banks[1]
    assert mixer_entry.bank_type == "mixer"
    assert mixer_entry.bank_metadata["snr_db_range"] == [10.0, 25.0]
    assert mixer_entry.bank_metadata["bandpass_clean"] is True
    assert mixer_entry.bank_metadata["noise_bank_path"] == "/path/to/noise_bank.h5"


def test_mix_classifier_bank_rejects_fs_mismatch(
    small_classifier_bank: ClassifierBank,
) -> None:
    """Sample-rate mismatch between clean + noise must fail loudly, not resample."""
    from datetime import datetime, timezone

    from myocard_egm_contracts._generated.python.noise_bank import (
        SchemaVersion,
    )
    from myocard_egm_contracts._generated.python.noise_bank import (
        Traces as NoiseTraces,
    )

    mismatched = NoiseBank(
        schema_version=SchemaVersion.field_1_0,
        created_utc=datetime.now(timezone.utc),
        source="test",
        fs_hz=500.0,  # clean is 1000.0
        traces=NoiseTraces(
            signal=[np.zeros(200).tolist() for _ in range(4)],
            source_record=["r"] * 4,
            source_channel=["c"] * 4,
        ),
    )
    with pytest.raises(ValueError, match="Sample-rate mismatch"):
        mix_classifier_bank(
            clean_bank=small_classifier_bank,
            noise_bank=mismatched,
            config=MixerConfig(show_progress=False),
        )


def test_mix_classifier_bank_rejects_empty_input(small_noise_bank: NoiseBank) -> None:
    """An empty clean bank has nothing to mix."""
    empty = ClassifierBank(banks=[], traces=[], labels={})
    with pytest.raises(ValueError, match="no traces"):
        mix_classifier_bank(
            clean_bank=empty,
            noise_bank=small_noise_bank,
            config=MixerConfig(show_progress=False),
        )


def test_mix_classifier_bank_preserves_trace_count(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """Mixing is per-trace 1:1 — no traces added or dropped."""
    n_clean = len(small_classifier_bank.traces)
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
    )
    assert len(noise_mixed.traces) == n_clean


def test_mix_classifier_bank_preserves_labels(
    small_classifier_bank: ClassifierBank, small_noise_bank: NoiseBank
) -> None:
    """Per-trace label_truth survives mixing unchanged."""
    noise_mixed = mix_classifier_bank(
        clean_bank=small_classifier_bank,
        noise_bank=small_noise_bank,
        config=MixerConfig(show_progress=False),
    )
    for clean, mixed in zip(small_classifier_bank.traces, noise_mixed.traces, strict=True):
        assert clean.label_truth == mixed.label_truth
