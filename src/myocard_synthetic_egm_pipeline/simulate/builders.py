"""Pure in-memory builders — DatasetResult / ClassifierBank → bank objects.

This module owns the per-trace-metadata + bank-assembly logic shared
across the producer pipeline:

- :func:`build_classifier_bank_from_dataset` — clean
  :class:`DatasetResult` → in-memory :class:`ClassifierBank`. The CLI's
  inline-mixer path needs the bank in memory before writing; the
  storage layer wraps this with a writer call for the disk-only path.
- :func:`build_synthetic_bank_from_dataset` — clean
  :class:`DatasetResult` → in-memory Pydantic :class:`SyntheticBank`.
  Pre-mixer values: ``snr_db = NaN``, ``noise_record = ""``,
  ``noise_channel = ""``.
- :func:`build_synthetic_bank_from_classifier` — noise-mixed (post-mixer)
  :class:`ClassifierBank` → in-memory Pydantic :class:`SyntheticBank`.
  Reads the mixer audit fields from each trace's ``trace_metadata``
  and the "mixer" provenance entry on ``bank.banks``.
- :func:`build_clean_trace_metadata`,
  :func:`build_bank_metadata_for_classifier_bank` — small helpers the
  three larger builders share.

These are pure functions (no I/O). The corresponding ``write_*``
wrappers in :mod:`~myocard_synthetic_egm_pipeline.simulate.storage`
and :mod:`~myocard_synthetic_egm_pipeline.mixer.storage` are thin
shims that call a builder and hand the result to the egm-data writer.

This module imports no backend code (Guardrail 1).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.synthetic_bank import (
    SchemaVersion,
    StimEdgeEnum,
    SyntheticBank,
    Traces,
)
from myocard_egm_contracts.schema_info import current_version
from myocard_egm_data.banks import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
)

from myocard_synthetic_egm_pipeline import __version__
from myocard_synthetic_egm_pipeline.constants import BANK_SOURCE
from myocard_synthetic_egm_pipeline.ids import derive_synthetic_bank_id, validate_artifact_id
from myocard_synthetic_egm_pipeline.simulate.dataset import DatasetConfig, DatasetResult

# Amplitude convention for ClassifierBank.amp_type. The Phase-1
# pseudo-EGM forward calc drops the 4*pi/sigma_e normalisation so
# synthetic bipolar traces are NOT in absolute mV; downstream
# consumers that need absolute mV scale themselves.
AMP_TYPE: str = "synthetic_au"


# ---------------------------------------------------------------------------
# Shared per-trace + bank-level metadata helpers
# ---------------------------------------------------------------------------


def build_clean_trace_metadata(
    *,
    sim_id: int | None,
    pair_idx: int,
    electrode_row: int | None,
    fibrosis_density_requested: float | None,
    fibrosis_density_realized: float | None,
    electrode_height_mm: float | None,
    stim_edge: str | None,
    sim_seed: int | None,
) -> dict[str, Any]:
    """Canonical per-trace metadata dict for a clean trace.

    Used by both the ClassifierBank builder and (indirectly via the
    same call sites) the SyntheticBank-from-dataset builder. The mixer
    extends this dict at mix time with three audit fields
    (``snr_db``, ``noise_record``, ``noise_channel``); see
    :func:`~myocard_synthetic_egm_pipeline.mixer.mixing.mix_classifier_bank`.

    ``patient_id`` is set to ``str(sim_id)`` so the patient-aware split
    in egm-data treats each simulation as one patient.
    """
    return {
        "sim_id": sim_id,
        "pair_index": int(pair_idx),
        "electrode_row": electrode_row,
        "fibrosis_density_requested": fibrosis_density_requested,
        "fibrosis_density_realized": fibrosis_density_realized,
        "electrode_height_mm": electrode_height_mm,
        "stim_edge": stim_edge,
        "sim_seed": sim_seed,
        "patient_id": str(sim_id) if sim_id is not None else "",
    }


def build_bank_metadata_for_classifier_bank(
    *,
    config: DatasetConfig,
    dataset_result: DatasetResult,
    bank_path: Path,
    description: str,
) -> dict[str, Any]:
    """Bank-level provenance blob for the synthetic source bank entry."""
    first_backend_meta: dict[str, Any] = {}
    if dataset_result.results:
        first_backend_meta = dict(
            dataset_result.results[0].run_metadata.get("backend_metadata", {})
        )
    return {
        "producer": "synthetic_egm_pipeline",
        "producer_version": __version__,
        "description": description,
        "backend": first_backend_meta.get("backend_name"),
        "cell_model": _cell_model_from_backend_meta(first_backend_meta),
        "n_simulations": config.n_simulations,
        "label_policy_name": config.label_policy.name,
        "label_policy_type": config.label_policy.type,
        "fibrosis_density_range": list(config.fibrosis_density_range),
        "fraction_healthy": config.fraction_healthy,
        "fixed_stim_edge": config.fixed_stim_edge,
        "geometry_type": config.geometry.type,
        "geometry_size_mm": getattr(config.geometry, "size_mm", None),
        "geometry_dr_mm": getattr(config.geometry, "dr_mm", None),
        "geometry_anisotropy_ratio": getattr(config.geometry, "anisotropy_ratio", None),
        "electrode_n_rows": config.electrode_n_rows,
        "electrode_n_cols": config.electrode_n_cols,
        "electrode_spacing_mm": config.electrode_spacing_mm,
        "electrode_height_mm_range": list(config.electrode_height_mm_range),
        "fs_hz": config.run_config.output_fs_hz,
        "trace_duration_ms": config.run_config.trace_duration_ms,
        "ap_time_unit_ms": config.run_config.ap_time_unit_ms,
        "bank_path": str(bank_path),
    }


# ---------------------------------------------------------------------------
# ClassifierBank from DatasetResult
# ---------------------------------------------------------------------------


def build_classifier_bank_from_dataset(
    *,
    dataset_result: DatasetResult,
    config: DatasetConfig,
    bank_path: Path | str,
    description: str = "",
    bank_id: str | None = None,
) -> ClassifierBank:
    """Build an in-memory ClassifierBank from a DatasetResult.

    Each bipolar trace becomes one :class:`ClassifierTrace` with the
    integer label from ``dataset_result.labels`` and the per-sim
    sampled scalars in ``trace_metadata``. ``patient_id`` is set to
    the trace's ``sim_id`` so the patient-aware split treats each
    simulation as one patient.

    No I/O — pair with
    :func:`myocard_egm_data.banks.write_classifier_bank` to write to
    disk.
    """
    bank_path = Path(bank_path)
    first_backend_meta: dict[str, Any] = {}
    if dataset_result.results:
        first_backend_meta = dict(
            dataset_result.results[0].run_metadata.get("backend_metadata", {})
        )
    resolved_bank_id = (
        validate_artifact_id(bank_id)
        if bank_id is not None
        else derive_synthetic_bank_id(_cell_model_from_backend_meta(first_backend_meta))
    )
    bank_meta = ClassifierBankMetaData(
        bank_id=resolved_bank_id,
        bank_type=BANK_SOURCE,
        bank_path=str(bank_path),
        bank_metadata=build_bank_metadata_for_classifier_bank(
            config=config,
            dataset_result=dataset_result,
            bank_path=bank_path,
            description=description,
        ),
    )

    traces: list[ClassifierTrace] = []
    flat_idx = 0
    for result in dataset_result.results:
        run_meta = result.run_metadata
        substrate_meta = result.substrate_realization_metadata
        sim_id = run_meta.get("sim_id")
        sim_seed = run_meta.get("sim_seed")
        electrode_row_per_pair = run_meta.get("electrode_row_per_pair") or [None] * result.n_pairs

        for pair_idx in range(result.n_pairs):
            label_value = int(dataset_result.labels[flat_idx])
            trace_meta = build_clean_trace_metadata(
                sim_id=sim_id,
                pair_idx=pair_idx,
                electrode_row=electrode_row_per_pair[pair_idx],
                fibrosis_density_requested=run_meta.get("fibrosis_density_requested"),
                fibrosis_density_realized=substrate_meta.get("density_realized"),
                electrode_height_mm=run_meta.get("electrode_height_mm"),
                stim_edge=run_meta.get("stim_edge"),
                sim_seed=sim_seed,
            )
            traces.append(
                ClassifierTrace(
                    bank_id=resolved_bank_id,
                    signal=result.bipolar_traces[pair_idx],
                    freq_hz=result.fs_hz,
                    amp_type=AMP_TYPE,
                    split=None,
                    label_truth=label_value,
                    prediction=None,
                    trace_metadata=trace_meta,
                )
            )
            flat_idx += 1

    return ClassifierBank(
        id=resolved_bank_id,
        banks=[bank_meta],
        traces=traces,
        labels=dict(dataset_result.labels_dict),
    )


# ---------------------------------------------------------------------------
# SyntheticBank from DatasetResult (clean / pre-mixer)
# ---------------------------------------------------------------------------


def build_synthetic_bank_from_dataset(
    *,
    dataset_result: DatasetResult,
    config: DatasetConfig,
    description: str = "",
    bank_id: str | None = None,
) -> SyntheticBank:
    """Build an in-memory Pydantic SyntheticBank from a DatasetResult.

    Pre-mixer values land in the noise + SNR columns:

    - ``snr_db = NaN`` per trace.
    - ``noise_record = ""`` per trace.
    - ``noise_channel = ""`` per trace.

    For noise-mixed (post-mixer) SyntheticBanks, use
    :func:`build_synthetic_bank_from_classifier` instead — that path
    reads the mixer audit fields from a noise-mixed ClassifierBank's
    ``trace_metadata``.

    No I/O — pair with
    :func:`myocard_egm_data.banks.write_synthetic_bank` to write.
    """
    signal_rows: list[list[float]] = []
    simulation_id: list[int] = []
    pair_index: list[int] = []
    electrode_row: list[int] = []
    fibrosis_density: list[float] = []
    fibrosis_density_realized: list[float] = []
    electrode_height_mm: list[float] = []
    seed_col: list[int] = []
    snr_db: list[float] = []
    stim_edge: list[StimEdgeEnum] = []
    noise_record: list[str] = []
    noise_channel: list[str] = []

    backend_meta_first: dict[str, Any] = {}

    for result in dataset_result.results:
        run_meta = result.run_metadata
        substrate_meta = result.substrate_realization_metadata
        if not backend_meta_first:
            backend_meta_first = dict(run_meta.get("backend_metadata", {}))

        sim_id = int(run_meta.get("sim_id", 0))
        sim_seed = int(run_meta.get("sim_seed", 0))
        row_per_pair = run_meta.get("electrode_row_per_pair") or [0] * result.n_pairs
        edge_val = run_meta.get("stim_edge", "top")
        if edge_val not in {"top", "bottom", "left", "right"}:
            raise ValueError(
                f"SyntheticBank stim_edge enum only accepts top/bottom/left/right; "
                f"got {edge_val!r}. Pre-mixer banks must use PlanarEdgeStimulus."
            )
        edge_enum = StimEdgeEnum(edge_val)
        height = float(run_meta.get("electrode_height_mm") or 0.0)
        density_req = float(run_meta.get("fibrosis_density_requested") or 0.0)
        density_realized = float(substrate_meta.get("density_realized", 0.0))

        for pair_idx in range(result.n_pairs):
            signal_rows.append(result.bipolar_traces[pair_idx].astype(float).tolist())
            simulation_id.append(sim_id)
            pair_index.append(int(pair_idx))
            electrode_row.append(int(row_per_pair[pair_idx] or 0))
            fibrosis_density.append(density_req)
            fibrosis_density_realized.append(density_realized)
            electrode_height_mm.append(height)
            seed_col.append(sim_seed)
            snr_db.append(math.nan)
            stim_edge.append(edge_enum)
            noise_record.append("")
            noise_channel.append("")

    traces = Traces(
        signal=signal_rows,
        simulation_id=simulation_id,
        pair_index=pair_index,
        electrode_row=electrode_row,
        fibrosis_density=fibrosis_density,
        fibrosis_density_realized=fibrosis_density_realized,
        electrode_height_mm=electrode_height_mm,
        seed=seed_col,
        snr_db=snr_db,
        stim_edge=stim_edge,
        noise_record=noise_record,
        noise_channel=noise_channel,
    )

    first_result = dataset_result.results[0] if dataset_result.results else None
    first_run_meta = first_result.run_metadata if first_result is not None else {}

    resolved_bank_id = (
        validate_artifact_id(bank_id)
        if bank_id is not None
        else derive_synthetic_bank_id(_cell_model_from_backend_meta(backend_meta_first))
    )

    return SyntheticBank(
        schema_version=SchemaVersion(current_version("synthetic_bank")),
        bank_id=resolved_bank_id,
        created_utc=datetime.now(timezone.utc),
        description=description or None,
        fs_hz=float(
            first_result.fs_hz if first_result is not None else config.run_config.output_fs_hz
        ),
        trace_duration_ms=float(
            first_result.trace_duration_ms
            if first_result is not None
            else config.run_config.trace_duration_ms
        ),
        simulator=str(backend_meta_first.get("backend_name") or "unknown"),
        cell_model=_cell_model_from_backend_meta(backend_meta_first),
        patch_size_mm=float(first_run_meta.get("patch_size_mm") or 0.0) or None,
        patch_dr_mm=float(first_run_meta.get("patch_dr_mm") or 0.0) or None,
        ap_time_unit_ms=float(backend_meta_first.get("ap_time_unit_ms") or 0.0) or None,
        fibrosis_strategy_name=str(first_run_meta.get("substrate_type") or "unknown"),
        fibrosis_params={
            "density_range": list(config.fibrosis_density_range),
            "fraction_healthy": config.fraction_healthy,
        },
        electrode_config={
            "n_rows": config.electrode_n_rows,
            "n_cols": config.electrode_n_cols,
            "spacing_mm": config.electrode_spacing_mm,
            "height_mm_range": list(config.electrode_height_mm_range),
        },
        mixer_config=None,
        experiment_config={
            "n_simulations": config.n_simulations,
            "master_seed": config.master_seed,
            "fixed_stim_edge": config.fixed_stim_edge,
            "label_policy_name": config.label_policy.name,
            "label_policy_type": config.label_policy.type,
            "producer_version": __version__,
        },
        noise_bank_source=None,
        traces=traces,
    )


# ---------------------------------------------------------------------------
# SyntheticBank from noise-mixed ClassifierBank (post-mixer)
# ---------------------------------------------------------------------------


def build_synthetic_bank_from_classifier(
    *,
    noise_mixed_bank: ClassifierBank,
    description: str = "",
    bank_id: str | None = None,
) -> SyntheticBank:
    """Build an in-memory Pydantic SyntheticBank from a noise-mixed ClassifierBank.

    The bank must have been produced by
    :func:`~myocard_synthetic_egm_pipeline.mixer.mixing.mix_classifier_bank`
    — this builder reads the mixer metadata from the bank's "mixer"
    provenance entry and per-trace ``trace_metadata`` to populate the
    SyntheticBank's mixer-aware columns (``snr_db``, ``noise_record``,
    ``noise_channel``).

    No I/O — pair with
    :func:`myocard_egm_data.banks.write_synthetic_bank` to write.
    """
    if not noise_mixed_bank.traces:
        raise ValueError("Noise-mixed bank has no traces.")

    mixer_entry = _find_bank_entry(noise_mixed_bank, "mixer")
    clean_entry = _find_bank_entry(noise_mixed_bank, BANK_SOURCE)
    clean_meta = clean_entry.bank_metadata if clean_entry is not None else {}
    mixer_meta = mixer_entry.bank_metadata if mixer_entry is not None else {}

    fs_hz = float(noise_mixed_bank.traces[0].freq_hz)
    n_samples = int(noise_mixed_bank.traces[0].signal.shape[0])
    trace_duration_ms = float(n_samples / fs_hz * 1000.0)

    signal_rows: list[list[float]] = []
    simulation_id: list[int] = []
    pair_index: list[int] = []
    electrode_row: list[int] = []
    fibrosis_density: list[float] = []
    fibrosis_density_realized: list[float] = []
    electrode_height_mm: list[float] = []
    seed_col: list[int] = []
    snr_db: list[float] = []
    stim_edge: list[StimEdgeEnum] = []
    noise_record: list[str] = []
    noise_channel: list[str] = []

    for trace_idx, trace in enumerate(noise_mixed_bank.traces):
        md = trace.trace_metadata
        edge_val = md.get("stim_edge")
        if edge_val not in {"top", "bottom", "left", "right"}:
            raise ValueError(
                f"trace {trace_idx}: SyntheticBank stim_edge enum needs one of "
                f"top/bottom/left/right; got {edge_val!r}."
            )

        signal_rows.append(trace.signal.astype(float).tolist())
        simulation_id.append(int(md.get("sim_id") or 0))
        pair_index.append(int(md.get("pair_index") or 0))
        electrode_row.append(int(md.get("electrode_row") or 0))
        fibrosis_density.append(float(md.get("fibrosis_density_requested") or 0.0))
        fibrosis_density_realized.append(float(md.get("fibrosis_density_realized") or 0.0))
        electrode_height_mm.append(float(md.get("electrode_height_mm") or 0.0))
        seed_col.append(int(md.get("sim_seed") or 0))
        snr_db.append(float(md.get("snr_db") or 0.0))
        stim_edge.append(StimEdgeEnum(edge_val))
        noise_record.append(str(md.get("noise_record") or ""))
        noise_channel.append(str(md.get("noise_channel") or ""))

    traces = Traces(
        signal=signal_rows,
        simulation_id=simulation_id,
        pair_index=pair_index,
        electrode_row=electrode_row,
        fibrosis_density=fibrosis_density,
        fibrosis_density_realized=fibrosis_density_realized,
        electrode_height_mm=electrode_height_mm,
        seed=seed_col,
        snr_db=snr_db,
        stim_edge=stim_edge,
        noise_record=noise_record,
        noise_channel=noise_channel,
    )

    if bank_id is not None:
        resolved_bank_id = validate_artifact_id(bank_id)
    elif noise_mixed_bank.id is not None:
        resolved_bank_id = validate_artifact_id(str(noise_mixed_bank.id))
    else:
        resolved_bank_id = derive_synthetic_bank_id(
            str(clean_meta.get("cell_model") or "unknown"), noise_mixed=True
        )

    return SyntheticBank(
        schema_version=SchemaVersion(current_version("synthetic_bank")),
        bank_id=resolved_bank_id,
        created_utc=datetime.now(timezone.utc),
        description=description or None,
        fs_hz=fs_hz,
        trace_duration_ms=trace_duration_ms,
        simulator=str(clean_meta.get("backend") or "unknown"),
        cell_model=str(clean_meta.get("cell_model") or "unknown"),
        patch_size_mm=float(clean_meta.get("geometry_size_mm") or 0.0) or None,
        patch_dr_mm=float(clean_meta.get("geometry_dr_mm") or 0.0) or None,
        ap_time_unit_ms=float(clean_meta.get("ap_time_unit_ms") or 0.0) or None,
        fibrosis_strategy_name=_fibrosis_strategy_name_from(clean_meta),
        fibrosis_params={
            "density_range": list(clean_meta.get("fibrosis_density_range") or []),
            "fraction_healthy": clean_meta.get("fraction_healthy", 0.0),
        },
        electrode_config={
            "n_rows": clean_meta.get("electrode_n_rows"),
            "n_cols": clean_meta.get("electrode_n_cols"),
            "spacing_mm": clean_meta.get("electrode_spacing_mm"),
            "height_mm_range": list(clean_meta.get("electrode_height_mm_range") or []),
        },
        mixer_config={
            "snr_db_range": list(mixer_meta.get("snr_db_range") or []),
            "bandpass_clean": mixer_meta.get("bandpass_clean", True),
            "band_hz": list(mixer_meta.get("band_hz") or []),
            "master_seed": mixer_meta.get("master_seed"),
        },
        experiment_config={
            "producer_version": __version__,
            "n_simulations": clean_meta.get("n_simulations"),
            "label_policy_name": clean_meta.get("label_policy_name"),
            "label_policy_type": clean_meta.get("label_policy_type"),
        },
        noise_bank_source=str(mixer_meta.get("noise_bank_source") or "") or None,
        traces=traces,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_bank_entry(bank: ClassifierBank, bank_type: str) -> ClassifierBankMetaData | None:
    """Return the first ``ClassifierBankMetaData`` matching ``bank_type``, or None."""
    for entry in bank.banks:
        if entry.bank_type == bank_type:
            return entry
    return None


def _cell_model_from_backend_meta(backend_meta: dict[str, Any]) -> str:
    """Best-effort cell-model name from backend metadata.

    Finitewave's :class:`AlievPanfilov2D` reports
    ``model_class="AlievPanfilov2D"``; we lower-case-snake it for the
    ``synthetic_bank.cell_model`` field. Future backends emitting
    different model names land here too.
    """
    model_class = str(backend_meta.get("model_class", "")).strip()
    if model_class == "AlievPanfilov2D":
        return "aliev_panfilov"
    if not model_class:
        return "unknown"
    out: list[str] = []
    for i, ch in enumerate(model_class):
        if ch.isupper() and i > 0 and not model_class[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _fibrosis_strategy_name_from(clean_meta: dict[str, Any]) -> str:
    """Best-effort fibrosis-strategy name for the SyntheticBank field.

    Phase 1 only has one substrate type so the hard-coded fallback is
    safe; future multi-strategy releases can read the actual type
    from per-trace metadata (which the producer already stamps via
    ``substrate_realization_metadata["strategy_type"]``).
    """
    del clean_meta  # reserved for future per-strategy dispatch
    return "uniform_random_fibrosis"


__all__ = [
    "AMP_TYPE",
    "build_bank_metadata_for_classifier_bank",
    "build_classifier_bank_from_dataset",
    "build_clean_trace_metadata",
    "build_synthetic_bank_from_classifier",
    "build_synthetic_bank_from_dataset",
]
