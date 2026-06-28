"""Stable cross-artifact id helpers for synthetic-egm-pipeline outputs.

The producer stamps an egm-contracts ``ArtifactId`` on every bank it writes.
The clean synthetic ClassifierBank (and the optional SyntheticBank sibling)
get a training-bank id derived from the cell model:
``tbank_synthetic_<cell_model>_<date>`` (e.g.
``tbank_synthetic_aliev_panfilov_2026-06-27``, matching the egm-data
fixtures). The noise-mixed (post-mixer) bank gets a ``_noise_mixed`` variant. Each id
is overridable via the CLI config / a function kwarg.

The mixer additionally needs the *noise* bank's id for its provenance entry.
The noise bank (iafdb-produced) records its id on the
``noise_bank_run_record.json`` sidecar, not the HDF5;
:func:`resolve_noise_bank_id` reads it from there, with a derived fallback.

The id *pattern* is single-sourced in egm-contracts' ``common.ArtifactId``;
this module only composes candidate strings and validates them.
"""

from __future__ import annotations

import datetime as _dt
import re as _re
from pathlib import Path
from typing import Any

from myocard_egm_contracts import common as _contracts_common
from myocard_egm_data.records import load_noise_bank_run_record
from pydantic import ValidationError

DATASET_TAG = "synthetic"
"""Descriptor root for this producer's banks."""

NOISE_DATASET_TAG = "iafdb"
"""Descriptor root for the noise bank the mixer references."""


def _today_utc() -> str:
    """Today's date (UTC) as ``YYYY-MM-DD`` for the id's date segment."""
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def _idstr(value: Any) -> str:
    """Normalize a contracts id field (RootModel or str) to a plain string."""
    return value.root if hasattr(value, "root") else str(value)


def _sanitize(text: str) -> str:
    """Collapse arbitrary text to the ``[a-z0-9_]`` id-descriptor charset."""
    cleaned = _re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or "unknown"


def validate_artifact_id(value: str) -> str:
    """Validate ``value`` against the egm-contracts ArtifactId pattern.

    Returns it unchanged on success; raises ``ValueError`` with a
    producer-friendly message on a malformed id. Mirrors egm-data's
    ClassifierBank-id validation so an explicit override fails fast at the
    producer boundary.
    """
    try:
        _contracts_common.ArtifactId(value)
    except ValidationError as exc:
        raise ValueError(
            f"bank_id {value!r} is not a valid stable artifact id "
            "(egm-contracts ArtifactId pattern, e.g. "
            "'tbank_synthetic_aliev_panfilov_2026-06-27')."
        ) from exc
    return value


def derive_synthetic_bank_id(
    cell_model: str, *, noise_mixed: bool = False, today: str | None = None
) -> str:
    """Default id for a synthetic ClassifierBank / SyntheticBank.

    ``tbank_synthetic_<cell_model>[_noise_mixed]_<date>``. Synthetic banks are
    always labeled training banks, so the role prefix is always ``tbank_``;
    the ``_noise_mixed`` marker distinguishes a post-mixer bank from its clean
    source.
    """
    desc = _sanitize(cell_model)
    suffix = "_noise_mixed" if noise_mixed else ""
    return f"tbank_{DATASET_TAG}_{desc}{suffix}_{today or _today_utc()}"


def derive_noise_ref_id(*, today: str | None = None) -> str:
    """Fallback id for the mixer's noise-source entry when the sidecar is absent."""
    return f"nbank_{NOISE_DATASET_TAG}_{today or _today_utc()}"


def resolve_noise_bank_id(
    noise_bank_path: Path | str | None, *, override: str | None = None
) -> str:
    """Resolve the noise bank's stable id for the mixer's provenance entry.

    Precedence: an explicit ``override`` -> the id recorded on the noise
    bank's ``<stem>_run_record.json`` sidecar -> a derived
    ``nbank_iafdb_<date>``. The sidecar is the iafdb-pipeline noise
    run-record (its ``bank_id`` field); a missing or unreadable sidecar
    falls back to the derived id.
    """
    if override is not None:
        return validate_artifact_id(override)
    if noise_bank_path is not None and str(noise_bank_path) != "":
        p = Path(noise_bank_path)
        sidecar = p.with_name(p.stem + "_run_record.json")
        if sidecar.is_file():
            record = None
            try:
                record = load_noise_bank_run_record(sidecar)
            except Exception:  # a corrupt sidecar must not break a mix; fall back
                record = None
            if record is not None and record.bank_id is not None:
                return validate_artifact_id(_idstr(record.bank_id))
    return derive_noise_ref_id()
