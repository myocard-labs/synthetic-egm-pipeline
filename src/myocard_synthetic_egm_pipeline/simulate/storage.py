"""Storage wrappers — build a bank in memory, then hand to egm-data's writer.

Two thin write helpers around the pure-in-memory builders in
:mod:`~myocard_synthetic_egm_pipeline.simulate.builders`:

- :func:`write_classifier_bank_from_dataset` — clean
  :class:`DatasetResult` → ClassifierBank.h5 on disk.
- :func:`write_synthetic_bank_from_dataset` — clean
  :class:`DatasetResult` → SyntheticBank.h5 on disk (optional sibling
  output for offline analysis tools that prefer the columnar format).

The hybrid (post-mixer) write wrappers live in
:mod:`~myocard_synthetic_egm_pipeline.mixer.storage`; the build logic
they share with this module is in
:mod:`~myocard_synthetic_egm_pipeline.simulate.builders`.

This module imports no backend code (Guardrail 1).
"""

from __future__ import annotations

from pathlib import Path

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
) -> Path:
    """Build + write a ClassifierBank for a finished DatasetResult.

    Each bipolar trace lands as one ``ClassifierTrace`` with its
    integer label, per-sim sampled scalars in ``trace_metadata``, and
    ``patient_id`` = ``sim_id`` so the patient-aware split treats each
    simulation as one patient.
    """
    output_path = Path(output_path)
    bank = build_classifier_bank_from_dataset(
        dataset_result=dataset_result,
        config=config,
        bank_path=output_path,
        description=description,
    )
    return write_classifier_bank(bank, output_path, overwrite=overwrite)


def write_synthetic_bank_from_dataset(
    *,
    dataset_result: DatasetResult,
    config: DatasetConfig,
    output_path: Path | str,
    description: str = "",
    overwrite: bool = False,
) -> Path:
    """Build + write a Pydantic SyntheticBank for a finished DatasetResult.

    Useful for offline analysis tools that read the per-trace columns
    directly (notebooks, egm-viewer's Inspection tab). The legacy
    ``synthetic_bank`` v1.0 schema is preserved verbatim — no growth,
    no migration. Pre-mixer values: ``snr_db = NaN``,
    ``noise_record = ""``, ``noise_channel = ""``.
    """
    output_path = Path(output_path)
    bank = build_synthetic_bank_from_dataset(
        dataset_result=dataset_result,
        config=config,
        description=description,
    )
    return write_synthetic_bank(bank, output_path, overwrite=overwrite)


__all__ = [
    "write_classifier_bank_from_dataset",
    "write_synthetic_bank_from_dataset",
]
