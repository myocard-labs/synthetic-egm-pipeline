"""Additive bandpass-domain mixing of clean synthetic + iafdb-extracted noise.

The mixer's job:

1. Optionally band-pass filter the clean trace to the bipolar EGM band
   so noise and signal are characterised in the same band the
   downstream classifier will see.
2. Draw a noise segment uniformly at random from the
   :class:`~myocard_egm_contracts._generated.python.noise_bank.NoiseBank`.
3. Scale the noise to hit the per-trace target SNR sampled from
   ``config.snr_db_range``.
4. Sum and emit a modified
   :class:`~myocard_egm_data.banks.ClassifierBank` with the mixer's
   per-trace audit fields (``snr_db``, ``noise_record``,
   ``noise_channel``) stamped into ``trace_metadata``.

SNR definition: :math:`\\mathrm{SNR}_{\\mathrm{dB}} =
10 \\log_{10}(P_s / P_n)` computed over the *bandpassed* clean trace
and the (already-bandpassed at extraction time) noise.

Noise-length handling: short segments are tiled to the trace length;
long segments are cropped at a random offset. Cross-fading at tile
joins is deliberately not done in v0.2.0 — the noise is broadband
anyway so the joint discontinuities sit well below physiological
amplitude. Add a polyphase cross-fade if a downstream artefact ever
flags it.

This module imports no backend code (Guardrail 1). It depends on
``myocard-egm-data`` (ClassifierBank + reader), ``myocard-egm-contracts``
(NoiseBank Pydantic), and ``myocard-egm-signal`` (the shared
``bandpass`` primitive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_data.banks import ClassifierBank, ClassifierBankMetaData, ClassifierTrace
from myocard_egm_signal import DEFAULT_BIPOLAR_BAND_HZ, bandpass
from tqdm import tqdm

from myocard_synthetic_egm_pipeline import __version__
from myocard_synthetic_egm_pipeline.constants import THETA_BANK_SOURCE
from myocard_synthetic_egm_pipeline.ids import (
    companion_path,
    noise_mixed_id_from,
    resolve_noise_bank_id,
    theta_bank_id_from,
    validate_artifact_id,
)

DEFAULT_SNR_DB_RANGE: tuple[float, float] = (10.0, 25.0)
"""Per-trace target SNR range (dB). Defaults match Sánchez 2021's
hybrid synthetic+real corpus and the legacy synthetic_egm_pipeline's
v1 production setting."""


@dataclass(frozen=True)
class MixerConfig:
    """Configuration for one mixing run.

    Attributes
    ----------
    snr_db_range
        ``(lo, hi)`` — per-trace target SNR sampled uniformly from this
        range. Tighten the range (e.g. ``(15.0, 15.0)``) for ablation
        studies where every trace gets the same SNR.
    bandpass_clean
        Apply the clinical bipolar band-pass to clean traces before
        mixing. The noise bank is already band-passed at extraction
        time (iafdb-pipeline does this), so passing through
        ``True`` keeps the two characterised in the same band.
    band_hz
        ``(low_hz, high_hz)`` for the bandpass. Defaults to
        :data:`myocard_egm_signal.DEFAULT_BIPOLAR_BAND_HZ`
        (≈ 30 - 300 Hz per Sánchez 2021 / Unger 2019).
    master_seed
        Master RNG seed for trace-noise pairing + SNR sampling.
    show_progress
        Render a tqdm progress bar over the trace loop.
    description
        Free-form note stamped into the output bank's mixer metadata.
    """

    snr_db_range: tuple[float, float] = DEFAULT_SNR_DB_RANGE
    bandpass_clean: bool = True
    band_hz: tuple[float, float] = DEFAULT_BIPOLAR_BAND_HZ
    master_seed: int = 0
    show_progress: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        lo, hi = self.snr_db_range
        if lo > hi:
            raise ValueError("snr_db_range must have lo <= hi.")
        b_lo, b_hi = self.band_hz
        if not (0 < b_lo < b_hi):
            raise ValueError("band_hz must satisfy 0 < low < high.")


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------


def snr_scale(
    signal: npt.NDArray[np.floating[Any]],
    noise: npt.NDArray[np.floating[Any]],
    target_snr_db: float,
) -> float:
    """Return the scalar :math:`\\alpha` such that mixing
    ``signal + alpha * noise`` yields ``target_snr_db``.

    Derivation. SNR is defined as
    :math:`10 \\log_{10}(P_s / (\\alpha^2 P_n))`; solving for
    :math:`\\alpha` gives :math:`\\alpha = \\sqrt{P_s / (10^{\\mathrm{SNR}/10} P_n)}`.

    Degenerate cases handled defensively:

    - If the noise has zero power, return 0.0 (no-op mix; the trace
      stays clean).
    - If the signal has zero power, return 1.0 (avoids dividing by
      zero in the formula; the resulting mix is just the
      raw noise, which is the right behaviour for a "silent" clean
      trace).
    """
    p_s = float(np.mean(signal**2))
    p_n = float(np.mean(noise**2))
    if p_n <= 0:
        return 0.0
    if p_s <= 0:
        return 1.0
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    return float(np.sqrt(p_s / (snr_linear * p_n)))


def sample_noise_for_length(
    *,
    noise_bank: NoiseBank,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], str, str]:
    """Draw one noise array of length ``n_samples`` from the bank.

    Returns the signal plus the ``(source_record, source_channel)``
    audit fields from the chosen segment. Short segments are tiled to
    ``n_samples``; long segments are cropped at a random offset.
    """
    n_segments = len(noise_bank.traces.signal)
    if n_segments == 0:
        raise ValueError("Noise bank is empty.")
    idx = int(rng.integers(0, n_segments))
    raw = np.asarray(noise_bank.traces.signal[idx], dtype=np.float64)
    record = str(noise_bank.traces.source_record[idx])
    channel = str(noise_bank.traces.source_channel[idx])

    if raw.shape[0] == n_samples:
        return raw, record, channel
    if raw.shape[0] > n_samples:
        start = int(rng.integers(0, raw.shape[0] - n_samples + 1))
        return raw[start : start + n_samples], record, channel
    # Tile to cover; crop to exact length. No cross-fade.
    reps = int(np.ceil(n_samples / raw.shape[0]))
    tiled = np.tile(raw, reps)[:n_samples]
    return tiled, record, channel


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_noise_mixed_id(override: str | None, clean_bank: ClassifierBank) -> str:
    """Resolve the noise-mixed bank's own stable id.

    Derived from the **clean bank's id** by inserting the ``noise_mixed``
    marker. It used to be re-derived from the cell model, read out of the
    clean bank's ``bank_metadata`` — a coupling that broke silently the
    moment generation parameters were cleaned off the ClassifierBank and
    moved to its ``synthetic_bank`` partner: the lookup fell back to
    ``"unknown"`` and produced a wrong-but-valid id. One string now anchors
    the whole family.
    """
    if override is not None:
        return validate_artifact_id(override)
    if clean_bank.id is None:
        raise ValueError(
            "Cannot derive a noise-mixed bank id: the clean bank carries no id. "
            "Pass noise_mixed_bank_id explicitly."
        )
    return validate_artifact_id(noise_mixed_id_from(str(clean_bank.id)))


def _rewrite_theta_companion(
    entries: list[ClassifierBankMetaData],
    mixed_bank_id: str,
    *,
    theta_bank_path: str | None,
    output_bank_path: str | None,
) -> list[ClassifierBankMetaData]:
    """Re-point the theta companion entry at the *mixed* run's theta bank.

    The mixer copies the clean bank's provenance entries forward, but the
    theta companion is not shared history — the mixed run writes its own
    ``synthetic_bank`` (same per-simulation config, mixed signals), under
    its own id. Carrying the clean entry through would leave the mixed
    ClassifierBank pointing at a *different* artifact than the one
    written beside it, which is the failure this rewrite exists to prevent.

    **Both** the id and the path are rewritten. Rewriting only the id was
    survivable while the clean bank's entry happened to name the mixed theta
    file — the two banks shared one theta artifact, which is exactly the
    defect that gave the clean intermediate its own id base and its own theta
    partner in the first place. Now that the clean bank names its *own*
    theta partner, inheriting its path would point the mixed bank at the
    clean theta file: right id, wrong file, and a consumer joining on it
    would read clean waveforms out of an artifact it believes is mixed.

    ``theta_bank_path=None`` means **the caller is not writing a theta bank
    for these mixed signals**, and the entry is *dropped* rather than
    rewritten. That is the standalone ``synthegm-mix`` case: it post-processes
    a ClassifierBank on disk, and schema 2.0's per-simulation config is not
    recoverable from per-trace metadata, so no theta partner can exist. The
    old code rewrote the id and kept the clean bank's path, leaving the mixed
    bank naming a mixed-derived id at the *clean* theta file — the same
    id-names-one-artifact-file-holds-another divergence, one CLI over.
    Absence is the honest answer: no theta partner, said by omission, not by a
    pointer to the wrong artifact.
    """
    rewritten: list[ClassifierBankMetaData] = []
    for entry in entries:
        if entry.bank_type == THETA_BANK_SOURCE:
            if theta_bank_path is None:
                continue
            entry = ClassifierBankMetaData(
                bank_id=theta_bank_id_from(mixed_bank_id),
                bank_type=entry.bank_type,
                bank_path=companion_path(theta_bank_path, relative_to=output_bank_path),
                bank_metadata=dict(entry.bank_metadata),
            )
        rewritten.append(entry)
    return rewritten


def mix_classifier_bank(
    *,
    clean_bank: ClassifierBank,
    noise_bank: NoiseBank,
    config: MixerConfig | None = None,
    noise_bank_path: str = "",
    noise_bank_id: str | None = None,
    noise_mixed_bank_id: str | None = None,
    output_bank_path: str | None = None,
    theta_bank_path: str | None = None,
) -> ClassifierBank:
    """Return a new ClassifierBank with each trace mixed against sampled noise.

    The bank's ``banks`` provenance is extended with a "mixer" entry
    recording the mixer config + noise bank source. Each trace's
    ``trace_metadata`` gains three audit fields:

    - ``snr_db`` — realised per-trace target SNR (dB).
    - ``noise_record`` — source dataset record (e.g. ``"iaf1_afw"``).
    - ``noise_channel`` — source bipolar channel (e.g. ``"CS12"``).

    Sample rates must match — clean and noise must both be at the
    same Hz. The function refuses to mix banks at different rates
    rather than silently resampling.
    """
    cfg = config if config is not None else MixerConfig()
    rng = np.random.default_rng(cfg.master_seed)

    if not clean_bank.traces:
        raise ValueError("Clean ClassifierBank has no traces to mix.")

    fs_hz = float(clean_bank.traces[0].freq_hz)
    if abs(noise_bank.fs_hz - fs_hz) > 1e-9:
        raise ValueError(
            f"Sample-rate mismatch: noise bank fs={noise_bank.fs_hz} Hz "
            f"vs clean traces fs={fs_hz} Hz."
        )

    snr_lo, snr_hi = cfg.snr_db_range
    new_traces: list[ClassifierTrace] = []

    iterator: Any = clean_bank.traces
    if cfg.show_progress:
        iterator = tqdm(clean_bank.traces, desc="Mixing", unit="trace")

    for clean in iterator:
        sig = np.asarray(clean.signal, dtype=np.float64)
        if cfg.bandpass_clean:
            sig_bp = bandpass(sig, fs=fs_hz, low_hz=cfg.band_hz[0], high_hz=cfg.band_hz[1])
        else:
            sig_bp = sig

        noise, src_record, src_channel = sample_noise_for_length(
            noise_bank=noise_bank, n_samples=sig_bp.shape[0], rng=rng
        )
        target_snr_db = float(rng.uniform(snr_lo, snr_hi))
        alpha = snr_scale(sig_bp, noise, target_snr_db)
        mixed = sig_bp + alpha * noise

        mixed_metadata = dict(clean.trace_metadata)
        mixed_metadata["snr_db"] = target_snr_db
        mixed_metadata["noise_record"] = src_record
        mixed_metadata["noise_channel"] = src_channel

        new_traces.append(
            ClassifierTrace(
                bank_id=clean.bank_id,
                signal=mixed.astype(np.float32, copy=False),
                freq_hz=clean.freq_hz,
                amp_type=clean.amp_type,
                split=clean.split,
                label_truth=clean.label_truth,
                prediction=clean.prediction,
                trace_metadata=mixed_metadata,
            )
        )

    # Extend the bank provenance with a mixer entry. The clean source
    # banks are preserved verbatim so a downstream consumer can still
    # see where the underlying signal came from.
    resolved_noise_id = resolve_noise_bank_id(noise_bank_path or None, override=noise_bank_id)
    # A COMPANION entry: the noise bank contributed to these traces but is
    # not where they came from. Its path is rendered relative to the bank
    # being written when they are neighbours.
    mixer_entry = ClassifierBankMetaData(
        bank_id=resolved_noise_id,
        bank_type="mixer",
        bank_path=companion_path(noise_bank_path or None, relative_to=output_bank_path),
        bank_metadata={
            "producer": "synthetic_egm_pipeline.mixer",
            "producer_version": __version__,
            "snr_db_range": list(cfg.snr_db_range),
            "bandpass_clean": cfg.bandpass_clean,
            "band_hz": list(cfg.band_hz),
            "noise_bank_source": str(noise_bank.source),
            "noise_bank_path": str(noise_bank_path or ""),
            "master_seed": cfg.master_seed,
            "description": cfg.description,
        },
    )

    resolved_noise_mixed_id = _resolve_noise_mixed_id(noise_mixed_bank_id, clean_bank)
    return ClassifierBank(
        id=resolved_noise_mixed_id,
        banks=[
            *_rewrite_theta_companion(
                list(clean_bank.banks),
                resolved_noise_mixed_id,
                theta_bank_path=theta_bank_path,
                output_bank_path=output_bank_path,
            ),
            mixer_entry,
        ],
        traces=new_traces,
        labels=dict(clean_bank.labels),
    )
