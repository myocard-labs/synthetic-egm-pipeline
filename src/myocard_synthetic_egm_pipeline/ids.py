"""Stable cross-artifact id helpers for synthetic-egm-pipeline outputs.

The producer stamps an egm-contracts ``ArtifactId`` on every bank it writes.
A run writes **two** banks and they take **distinct** ids derived from one
base, because the phase manifest keys artifacts by id and two files sharing
one id collide:

- the ClassifierBank keeps the base id —
  ``tbank_synthetic_<cell_model>[_noise_mixed]_<date>``, derived from the
  cell model (e.g. ``tbank_synthetic_aliev_panfilov_2026-06-27``);
- the ``synthetic_bank`` gets the same id with a ``theta`` marker in the
  descriptive name (:func:`theta_bank_id_from`).

One override (`output.bank_id`) therefore names both, keeping the pair in
step; see :func:`theta_bank_id_from` for why that beats two independent
overrides.

The mixer additionally needs the *noise* bank's id for its provenance entry.
The noise bank (iafdb-produced) records its id on the
``noise_bank_run_record.json`` sidecar, not the HDF5;
:func:`resolve_noise_bank_id` reads it from there, with a derived fallback.

The id *pattern* is single-sourced in egm-contracts' ``common.ArtifactId``;
this module only composes candidate strings and validates them.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
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
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
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


#: ISO date suffix an ArtifactId may end with (optional per the grammar).
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Marker appended to a run's base id to name its ``synthetic_bank``.
#: The ClassifierBank keeps the bare stem, so existing ids are unchanged.
THETA_BANK_MARKER: str = "theta"


def derive_synthetic_bank_id(
    cell_model: str, *, noise_mixed: bool = False, today: str | None = None
) -> str:
    """Default id for a synthetic ClassifierBank.

    ``tbank_synthetic_<cell_model>[_noise_mixed]_<date>``. Synthetic banks are
    always labeled training banks, so the role prefix is always ``tbank_``;
    the ``_noise_mixed`` marker distinguishes a post-mixer bank from its clean
    source.

    The paired ``synthetic_bank`` id comes from
    :func:`theta_bank_id_from` rather than being derived independently —
    see that function for why.
    """
    desc = _sanitize(cell_model)
    suffix = "_noise_mixed" if noise_mixed else ""
    return f"tbank_{DATASET_TAG}_{desc}{suffix}_{today or _today_utc()}"


def theta_bank_id_from(classifier_bank_id: str) -> str:
    """The ``synthetic_bank`` id paired with a ClassifierBank's id.

    A run writes two artifacts — the ClassifierBank it trains on and the
    ``synthetic_bank`` carrying theta plus the per-simulation config —
    and they need **distinct** stable ids, because the phase manifest
    keys artifacts by id and two files sharing one id collide.

    They are derived from a single base rather than configured
    independently. The two are a pair produced by one run, and sharing a
    stem makes that visible: they sort together in a manifest or a
    directory listing, and one override keeps them in step. Two separate
    overrides would let a caller set one and leave the other on a derived
    id, silently producing a pair that looks unrelated — with nothing but
    the ``simulation_id`` join to recover the relationship afterwards.

    The marker is inserted **before** any trailing date so the result
    still matches the ``ArtifactId`` grammar
    (``<prefix>_<descriptive_name>[_<YYYY-MM-DD>]``): the marker is part
    of the descriptive name, not something after the date.

        ``tbank_synthetic_aliev_panfilov_2026-08-01``
        -> ``tbank_synthetic_aliev_panfilov_theta_2026-08-01``
        ``tbank_run7`` -> ``tbank_run7_theta``
    """
    return _insert_marker(classifier_bank_id, THETA_BANK_MARKER)


#: Marker distinguishing a post-mixer bank from its clean source.
NOISE_MIXED_MARKER: str = "noise_mixed"


def _insert_marker(bank_id: str, marker: str) -> str:
    """Insert ``marker`` into ``bank_id``'s descriptive name, before any date.

    ``ArtifactId`` is ``<prefix>_<descriptive_name>[_<YYYY-MM-DD>]``, so a
    marker appended after the date would put the id outside its own
    grammar.
    """
    head, sep, tail = bank_id.rpartition("_")
    if sep and _DATE_RE.fullmatch(tail):
        return f"{head}_{marker}_{tail}"
    return f"{bank_id}_{marker}"


def noise_mixed_id_from(clean_bank_id: str) -> str:
    """The noise-mixed bank's id, derived from its clean source's id.

    Derived from the **id** rather than re-derived from the cell model,
    which is what the mixer used to do by reading ``cell_model`` out of
    the clean bank's ``bank_metadata``. That coupling broke the moment
    the generation parameters were cleaned off the ClassifierBank and
    moved to its ``synthetic_bank`` partner: the lookup silently fell back
    to ``"unknown"`` and the mixed bank got a wrong-but-valid id — the worst
    kind, because nothing downstream refuses it. Deriving from the id keeps
    the whole family — clean, noise-mixed, and their theta partners —
    anchored to one string.

        ``tbank_synthetic_ap_2026-08-01``
        -> ``tbank_synthetic_ap_noise_mixed_2026-08-01``
    """
    if f"_{NOISE_MIXED_MARKER}_" in clean_bank_id or clean_bank_id.endswith(
        f"_{NOISE_MIXED_MARKER}"
    ):
        return clean_bank_id
    return _insert_marker(clean_bank_id, NOISE_MIXED_MARKER)


def companion_path(target: Path | str | None, *, relative_to: Path | str | None) -> str:
    """Render a companion artifact's path for a bank's provenance entry.

    Companion artifacts — the noise bank a mixer drew from, the
    ``synthetic_bank`` holding theta — normally sit beside the bank that
    references them, so a **bare filename** is both the common case and
    the portable one: move the directory and the reference still
    resolves.

    Relative whenever the two share a **real common ancestor** — any
    directory above the filesystem root. That is the test for "these live
    in the same project tree", and it is what makes the reference survive
    the tree being moved or copied. `../noise/iafdb.h5` qualifies: banks
    routinely sit in sibling directories under one project root, and that
    whole root moves as a unit.

    Falls back to absolute only when there is genuinely nothing to be
    relative *to* — different drives on Windows, or two paths whose only
    shared ancestor is the root itself, which means they belong to
    unrelated trees and no amount of ``..`` expresses a stable
    relationship.

    Returns ``""`` when there is no target; callers that mean "the data
    is in this file" want :data:`LOCAL_BANK_PATH` instead.
    """
    if target is None or str(target) == "":
        return ""
    target_path = Path(target)
    if relative_to is None:
        return str(target_path)

    anchor = Path(relative_to)
    anchor_dir = anchor.parent if anchor.suffix else anchor
    try:
        relative = Path(os.path.relpath(target_path, anchor_dir))
    except ValueError:
        # Different drives on Windows — no relative path exists at all.
        return str(target_path)

    # Only a root-level common ancestor means the two are in unrelated
    # trees; a relative path between them would express nothing stable.
    if _shares_only_the_root(target_path, anchor_dir):
        return str(target_path)
    return relative.as_posix()


def _shares_only_the_root(a: Path, b: Path) -> bool:
    """True when ``a`` and ``b``'s deepest common ancestor is the root."""
    a_parts = Path(os.path.abspath(a)).parts
    b_parts = Path(os.path.abspath(b)).parts
    common = 0
    for x, y in zip(a_parts, b_parts, strict=False):
        if x != y:
            break
        common += 1
    # parts[0] is the root ("/" or "C:\\"); anything beyond it is a real
    # shared directory.
    return common <= 1


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
