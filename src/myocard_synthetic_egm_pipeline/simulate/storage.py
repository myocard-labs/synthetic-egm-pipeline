"""Storage wrappers — build a bank in memory, then hand to egm-data's writer.

Two thin write helpers around the pure-in-memory builders in
:mod:`~myocard_synthetic_egm_pipeline.simulate.builders`:

- :func:`write_classifier_bank_from_dataset` — clean
  :class:`DatasetResult` → ClassifierBank.h5 on disk.
- :func:`write_synthetic_bank_from_dataset` — clean
  :class:`DatasetResult` → SyntheticBank.h5 on disk (optional sibling
  output for offline analysis tools that prefer the columnar format).

The noise-mixed (post-mixer) write wrappers live in
:mod:`~myocard_synthetic_egm_pipeline.mixer.storage`; the build logic
they share with this module is in
:mod:`~myocard_synthetic_egm_pipeline.simulate.builders`.

This module imports no backend code (Guardrail 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from myocard_egm_data.banks import write_classifier_bank, write_synthetic_bank

from myocard_synthetic_egm_pipeline.simulate.builders import (
    build_classifier_bank_from_dataset,
    build_synthetic_bank_from_dataset,
)
from myocard_synthetic_egm_pipeline.simulate.dataset import DatasetConfig, DatasetResult


def write_classifier_bank_from_dataset(
    *,
    dataset_result: DatasetResult,
    config: DatasetConfig,
    output_path: Path | str,
    description: str = "",
    overwrite: bool = False,
    bank_id: str | None = None,
    synthetic_bank_path: Path | str | None = None,
) -> Path:
    """Build + write a ClassifierBank for a finished DatasetResult.

    Each bipolar trace lands as one ``ClassifierTrace`` with its
    integer label, per-sim sampled scalars in ``trace_metadata``, and
    ``patient_id`` = ``simulation_id`` so the patient-aware split treats each
    simulation as one patient.
    """
    output_path = Path(output_path)
    bank = build_classifier_bank_from_dataset(
        dataset_result=dataset_result,
        config=config,
        bank_path=output_path,
        description=description,
        bank_id=bank_id,
        synthetic_bank_path=synthetic_bank_path,
    )
    return write_classifier_bank(bank, output_path, overwrite=overwrite)


def write_synthetic_bank_from_dataset(
    *,
    dataset_result: DatasetResult,
    config: DatasetConfig,
    output_path: Path | str,
    description: str = "",
    overwrite: bool = False,
    bank_id: str | None = None,
    mixed_signals: list[npt.NDArray[np.float32]] | None = None,
    snr_db: list[float] | None = None,
    noise_record: list[str] | None = None,
    noise_channel: list[str] | None = None,
    noise_bank_source: str | None = None,
    bank_id_base: str | None = None,
) -> Path:
    """Build + write a ``synthetic_bank`` 2.0 for a finished DatasetResult.

    This is the artifact that carries θ and the per-simulation
    generation config — the parallel bank egm-studio's T4 views read
    from, joined to the ClassifierBank on ``simulation_id``.

    Clean runs leave the mixer arguments unset. The **inline mixer
    path** passes the mixed signals plus the three per-trace noise
    columns, so a noise-mixed bank is written from this same
    ``DatasetResult`` — 2.0's per-simulation config cannot be
    reconstructed from a mixed ClassifierBank's per-trace metadata, so
    building it here is the only correct route.
    """
    output_path = Path(output_path)
    bank = build_synthetic_bank_from_dataset(
        dataset_result=dataset_result,
        config=config,
        description=description,
        bank_id=bank_id,
        mixed_signals=mixed_signals,
        snr_db=snr_db,
        noise_record=noise_record,
        noise_channel=noise_channel,
        noise_bank_source=noise_bank_source,
        bank_id_base=bank_id_base,
    )
    return write_synthetic_bank(bank, output_path, overwrite=overwrite)


__all__ = [
    "write_classifier_bank_from_dataset",
    "write_synthetic_bank_from_dataset",
]
