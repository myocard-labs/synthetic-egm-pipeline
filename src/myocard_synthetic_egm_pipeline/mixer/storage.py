"""Hybrid-bank write helpers.

The mixer's default output is a hybrid
:class:`~myocard_egm_data.banks.ClassifierBank` with the mixer
metadata stamped into each trace's ``trace_metadata`` and a "mixer"
provenance entry appended to ``banks``. That bank goes straight
through :func:`myocard_egm_data.banks.write_classifier_bank` — no
new writer needed.

This module owns the *optional* Pydantic
:class:`~myocard_egm_contracts._generated.python.synthetic_bank.SyntheticBank`
sibling output, parallel to
:func:`myocard_synthetic_egm_pipeline.simulate.storage.write_synthetic_bank_from_dataset`.
The in-memory transformation (hybrid ClassifierBank → Pydantic
SyntheticBank) lives in
:func:`myocard_synthetic_egm_pipeline.simulate.builders.build_synthetic_bank_from_classifier`;
this module is the thin disk-write wrapper around it.

This module imports no backend code (Guardrail 1).
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_data.banks import ClassifierBank, write_synthetic_bank

from myocard_synthetic_egm_pipeline.simulate.builders import (
    build_synthetic_bank_from_classifier,
)


def write_hybrid_synthetic_bank_from_classifier(
    *,
    hybrid_bank: ClassifierBank,
    output_path: Path | str,
    description: str = "",
    overwrite: bool = False,
) -> Path:
    """Build + write a hybrid Pydantic SyntheticBank sibling.

    The hybrid ClassifierBank must have been produced by
    :func:`~myocard_synthetic_egm_pipeline.mixer.mixing.mix_classifier_bank`
    — this writer reads the mixer audit fields from the bank's
    per-trace ``trace_metadata`` and from the "mixer" provenance
    entry on ``bank.banks``.

    Raises
    ------
    ValueError
        If the bank is empty, or per-trace metadata is missing fields
        the SyntheticBank schema requires (e.g. ``sim_id``,
        ``stim_edge``, ``electrode_height_mm``).
    """
    output_path = Path(output_path)
    bank = build_synthetic_bank_from_classifier(
        hybrid_bank=hybrid_bank,
        description=description,
    )
    return write_synthetic_bank(bank, output_path, overwrite=overwrite)


__all__ = [
    "write_hybrid_synthetic_bank_from_classifier",
]
