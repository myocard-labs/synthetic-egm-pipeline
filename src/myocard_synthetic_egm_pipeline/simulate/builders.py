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

import numpy as np
import numpy.typing as npt
from myocard_egm_contracts._generated.python.synthetic_bank import (
    GenerationParams,
    LabelItem,
    PairIndexItem,
    SchemaVersion,
    Simulations,
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
from myocard_synthetic_egm_pipeline.simulate.bank_config import build_simulation_columns
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
    simulation_id: int | None,
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

    ``patient_id`` is set to ``str(simulation_id)`` so the patient-aware split
    in egm-data treats each simulation as one patient.
    """
    return {
        "simulation_id": simulation_id,
        "pair_index": int(pair_idx),
        "electrode_row": electrode_row,
        "fibrosis_density_requested": fibrosis_density_requested,
        "fibrosis_density_realized": fibrosis_density_realized,
        "electrode_height_mm": electrode_height_mm,
        "stim_edge": stim_edge,
        "sim_seed": sim_seed,
        "patient_id": str(simulation_id) if simulation_id is not None else "",
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
    the trace's ``simulation_id`` so the patient-aware split treats each
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
        simulation_id = run_meta.get("simulation_id")
        sim_seed = run_meta.get("sim_seed")
        electrode_row_per_pair = run_meta.get("electrode_row_per_pair") or [None] * result.n_pairs

        for pair_idx in range(result.n_pairs):
            label_value = int(dataset_result.labels[flat_idx])
            trace_meta = build_clean_trace_metadata(
                simulation_id=simulation_id,
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
# SyntheticBank 2.0 from DatasetResult
# ---------------------------------------------------------------------------


def build_synthetic_bank_from_dataset(
    *,
    dataset_result: DatasetResult,
    config: DatasetConfig,
    description: str = "",
    bank_id: str | None = None,
    mixed_signals: list[npt.NDArray[np.float32]] | None = None,
    snr_db: list[float] | None = None,
    noise_record: list[str] | None = None,
    noise_channel: list[str] | None = None,
    noise_bank_source: str | None = None,
) -> SyntheticBank:
    """Build an in-memory ``synthetic_bank`` 2.0 from a DatasetResult.

    The generation config is written **per simulation** as typed,
    ``type``-discriminated objects (see
    :mod:`~myocard_synthetic_egm_pipeline.simulate.bank_config`); the
    ``traces/`` group collapses to the signal, the two foreign keys, the
    integer label and the noise provenance.

    Noise columns
    -------------
    Clean banks leave the mixer arguments unset: ``snr_db`` is NaN and
    the two noise strings are empty, matching the schema's "the mixer
    was off" convention. The **inline mixer path** passes
    ``mixed_signals`` plus the three per-trace noise columns, so a
    noise-mixed bank is written from the same in-memory
    :class:`DatasetResult` rather than reconstructed from a
    ClassifierBank — the 1.1 round-trip that 2.0 makes impossible, since
    the per-simulation config is not recoverable from per-trace
    metadata (see ``project/architecture.md``).

    ``activation_position`` is left absent: Wave 1 applies no controlled
    crop, and the schema is explicit that absence means *unknown*, never
    0.0 (which is a legitimate position). SEP2 populates it.

    No I/O — pair with :func:`myocard_egm_data.banks.write_synthetic_bank`.
    """
    results = dataset_result.results
    if not results:
        raise ValueError("DatasetResult has no simulations to write.")

    first = results[0]
    backend_meta_first = dict(first.run_metadata.get("backend_metadata", {}))

    simulations = build_simulation_columns(
        results=results,
        label_policy=config.label_policy,
        labels_dict=dataset_result.labels_dict,
        master_seed=config.master_seed,
        output_fs_hz=config.run_config.output_fs_hz,
        capture_oversample=config.run_config.capture_oversample,
    )

    # Trace order is positional and must match dataset_result's flat
    # simulation_id / pair_index / label arrays, which are built in the
    # same nested order by generate_dataset.
    signal_rows: list[list[float]] = []
    for result in results:
        for pair_idx in range(result.n_pairs):
            signal_rows.append(result.bipolar_traces[pair_idx].astype(float).tolist())

    n_traces = len(signal_rows)
    if mixed_signals is not None:
        if len(mixed_signals) != n_traces:
            raise ValueError(
                f"mixed_signals has {len(mixed_signals)} entries for {n_traces} traces."
            )
        signal_rows = [sig.astype(float).tolist() for sig in mixed_signals]

    traces = Traces(
        signal=signal_rows,
        simulation_id=[int(x) for x in dataset_result.simulation_ids],
        pair_index=[PairIndexItem(int(x)) for x in dataset_result.pair_indices],
        label=[LabelItem(int(x)) for x in dataset_result.labels],
        activation_position=None,
        snr_db=list(snr_db) if snr_db is not None else [math.nan] * n_traces,
        noise_record=list(noise_record) if noise_record is not None else [""] * n_traces,
        noise_channel=(list(noise_channel) if noise_channel is not None else [""] * n_traces),
    )

    resolved_bank_id = (
        validate_artifact_id(bank_id)
        if bank_id is not None
        else derive_synthetic_bank_id(
            _cell_model_from_backend_meta(backend_meta_first),
            noise_mixed=mixed_signals is not None,
        )
    )

    return SyntheticBank(
        schema_version=SchemaVersion(current_version("synthetic_bank")),
        bank_id=resolved_bank_id,
        created_utc=datetime.now(timezone.utc),
        description=description or None,
        fs_hz=float(first.fs_hz),
        trace_duration_ms=float(first.trace_duration_ms),
        noise_bank_source=noise_bank_source,
        generation_params=build_theta_spec(simulations),
        simulations=simulations,
        traces=traces,
    )


def build_theta_spec(simulations: Simulations) -> GenerationParams:
    """Build the bank-scoped theta-spec.

    The regime is the set of structural ``type`` discriminators held
    fixed across the bank; ``knobs`` is the list of swept parameters.
    Wave 1 sweeps nothing, so the knob list is empty — which the schema
    requires anyway rather than allowing omission, because "nothing
    varied" is worth stating explicitly. SEP11 fills the list.
    """
    regime: dict[str, str] = {}
    if simulations.simulation_id:
        regime = {
            "geometry": simulations.geometry[0].root.type,
            "cell_model": simulations.cell_model[0].type,
            "substrate": simulations.substrate[0].root.type,
            "activation": simulations.activation[0].type,
            "electrodes": simulations.electrodes[0].root.type,
            "backend": simulations.backend[0].root.type,
            "label_policy": simulations.label_policy[0].type,
        }
    return GenerationParams(regime=regime, knobs=[])


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
    "build_synthetic_bank_from_dataset",
    "build_theta_spec",
]
