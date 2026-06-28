"""Additive bandpass-domain noise overlay.

Reads a noise_bank.h5 (produced by iafdb-pipeline), samples noise per
trace at a target SNR, adds it to a clean ClassifierBank's signal
column, emits a noise-mixed ClassifierBank. Per-trace mixer audit fields
(``snr_db``, ``noise_record``, ``noise_channel``) land in
``trace_metadata``; a "mixer" provenance entry is appended to the
bank's ``banks`` list.

Public API:

- :class:`MixerConfig` — SNR range, band, RNG seed, bandpass-clean flag.
- :func:`mix_classifier_bank` — orchestrator; takes a clean
  ClassifierBank + a Pydantic NoiseBank, returns a noise-mixed
  ClassifierBank.
- :func:`snr_scale`, :func:`sample_noise_for_length` — pure-math
  helpers exposed for tests + offline analysis.
- :func:`write_noise_mixed_synthetic_bank_from_classifier` — optional
  Pydantic SyntheticBank sibling output (only useful for offline
  analysis tools that read the per-trace columns directly).
"""

from __future__ import annotations

from myocard_synthetic_egm_pipeline.mixer.mixing import (
    DEFAULT_SNR_DB_RANGE,
    MixerConfig,
    mix_classifier_bank,
    sample_noise_for_length,
    snr_scale,
)
from myocard_synthetic_egm_pipeline.mixer.storage import (
    write_noise_mixed_synthetic_bank_from_classifier,
)

__all__ = [
    "DEFAULT_SNR_DB_RANGE",
    "MixerConfig",
    "mix_classifier_bank",
    "sample_noise_for_length",
    "snr_scale",
    "write_noise_mixed_synthetic_bank_from_classifier",
]
