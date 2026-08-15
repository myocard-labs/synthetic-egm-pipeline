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
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from myocard_egm_contracts._generated.python.synthetic_bank import (
    ActivationPosition,
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
from myocard_synthetic_egm_pipeline.constants import (
    BANK_SOURCE,
    LOCAL_BANK_PATH,
    THETA_BANK_SOURCE,
)
from myocard_synthetic_egm_pipeline.ids import (
    companion_path,
    derive_synthetic_bank_id,
    theta_bank_id_from,
    validate_artifact_id,
)
from myocard_synthetic_egm_pipeline.simulate.bank_config import build_simulation_columns
from myocard_synthetic_egm_pipeline.simulate.dataset import DatasetConfig, DatasetResult
from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult

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
) -> dict[str, Any]:
    """Per-trace metadata for a ClassifierBank trace.

    **Identity only.** The ClassifierBank is a *source-agnostic ML
    compression*: signal, label, and the keys needed to join back to
    where the trace came from. Generation parameters are the raw
    material of the signal-realism / parameter-estimation work, not of
    classification, so they live on the ``synthetic_bank`` and consumers
    reach them through ``simulation_id``
    (``synthetic_bank_source_of_truth.md`` section 12).

    Until Phase 1.5 this dict also carried ``fibrosis_density_requested``
    / ``fibrosis_density_realized`` / ``electrode_row`` /
    ``electrode_height_mm`` / ``stim_edge`` / ``sim_seed`` — the same
    flat per-trace generation columns the ``synthetic_bank`` restructure
    removed. Taking them out of one artifact and leaving the copy in the
    other would have kept the duplication and the drift risk; every one
    is now recoverable per-simulation.

    ``patient_id`` is ``str(simulation_id)`` so egm-classifier's
    patient-aware split treats one simulation as one patient — traces
    from a simulation share a substrate, so splitting across them leaks.

    The mixer adds ``snr_db`` / ``noise_record`` / ``noise_channel`` when
    it runs. They are **absent on a clean bank** rather than NaN/empty:
    a present-but-empty noise field reads as "mixed, details unknown",
    which is the opposite of the truth.
    """
    return {
        "simulation_id": simulation_id,
        "pair_index": int(pair_idx),
        "patient_id": str(simulation_id) if simulation_id is not None else "",
    }


def build_bank_metadata_for_classifier_bank(
    *,
    config: DatasetConfig,
    dataset_result: DatasetResult,
    bank_path: Path,
    description: str,
) -> dict[str, Any]:
    """Bank-level provenance for the ClassifierBank's origin entry.

    Deliberately small. The generation config — geometry, substrate
    ranges, electrode grid, cell model, backend — used to be copied here
    too; it is the ``synthetic_bank``'s subject and is reachable through
    the companion entry, so keeping a second copy only created two
    things to disagree.

    What stays, and why:

    - ``producer`` / ``producer_version`` — **reproducibility**: which
      code wrote this. Not generation physics, and ``synthetic_bank``
      2.0 has nowhere to record it, so dropping it would lose the fact
      rather than de-duplicate it.
    - ``description`` — free text about the run.
    - ``trace_duration_ms`` — the classifier's input-length contract, so
      genuinely training-relevant.
    - ``label_policy`` — the policy's **identity only**. It defines what
      the classification task *is*, which is the ClassifierBank's whole
      subject; its thresholds are generation detail and stay on the
      source bank.
    """
    del bank_path  # the entry's own bank_path records locality
    policy_types = sorted(
        {
            str(r.run_metadata.get("label_policy_type", config.label_policy.type))
            for r in dataset_result.results
        }
    ) or [config.label_policy.type]
    return {
        "producer": "synthetic_egm_pipeline",
        "producer_version": __version__,
        "description": description,
        "trace_duration_ms": config.run_config.trace_duration_ms,
        "label_policy": "+".join(policy_types),
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
    synthetic_bank_path: Path | str | None = None,
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
    # The ORIGIN entry: these traces were produced by this run, not loaded
    # from another bank, so there is no source file to name. Writing one in
    # anyway is how this field came to name a bank that is never written
    # (clean runs) or one whose traces differ from the file's (noise-mixed).
    bank_meta = ClassifierBankMetaData(
        bank_id=resolved_bank_id,
        bank_type=BANK_SOURCE,
        bank_path=LOCAL_BANK_PATH,
        bank_metadata=build_bank_metadata_for_classifier_bank(
            config=config,
            dataset_result=dataset_result,
            bank_path=bank_path,
            description=description,
        ),
    )

    # The COMPANION entry: the synthetic_bank this run also writes, holding
    # theta and the per-simulation generation config. Nothing referenced it
    # before, so pairing the two artifacts was a fact that lived only in
    # someone's memory. No trace points at this entry — it describes the
    # traces without being where they came from.
    theta_meta = ClassifierBankMetaData(
        bank_id=theta_bank_id_from(resolved_bank_id),
        bank_type=THETA_BANK_SOURCE,
        bank_path=companion_path(synthetic_bank_path, relative_to=bank_path),
        bank_metadata={
            "join_key": "simulation_id",
            "description": ("Per-simulation generation config + theta-spec for these traces."),
        },
    )

    # The join key comes from the DatasetResult's own columns — the same arrays
    # the synthetic_bank is built from — rather than being recomputed here.
    #
    # It used to be recomputed: `range(result.n_pairs)` for the pair index and
    # `run_metadata["simulation_id"]` for the simulation, while the theta path
    # read `dataset_result.pair_indices` / `.simulation_ids`. One quantity, two
    # computations, and when the probe changed the layout only one of them was
    # updated — so the two banks disagreed on `pair_index` and egm-studio, which
    # joins them on `(simulation_id, pair_index)` and raises on a non-unique
    # key, could not load the pair. The layout has since been fixed, which is
    # exactly why this stays: with both readings correct again the duplication
    # is invisible until the next change reintroduces the skew.
    n_traces = int(dataset_result.labels.shape[0])
    for name, column in (
        ("simulation_ids", dataset_result.simulation_ids),
        ("pair_indices", dataset_result.pair_indices),
    ):
        if column.shape[0] != n_traces:
            raise ValueError(
                f"DatasetResult.{name} has {column.shape[0]} entries for {n_traces} "
                "traces; the ClassifierBank and the synthetic_bank are keyed on these "
                "columns and must agree trace for trace."
            )

    traces: list[ClassifierTrace] = []
    flat_idx = 0
    for result in dataset_result.results:
        for pair_idx in range(result.n_pairs):
            traces.append(
                ClassifierTrace(
                    bank_id=resolved_bank_id,
                    signal=result.bipolar_traces[pair_idx],
                    freq_hz=result.fs_hz,
                    amp_type=AMP_TYPE,
                    split=None,
                    label_truth=int(dataset_result.labels[flat_idx]),
                    prediction=None,
                    trace_metadata=build_clean_trace_metadata(
                        simulation_id=int(dataset_result.simulation_ids[flat_idx]),
                        pair_idx=int(dataset_result.pair_indices[flat_idx]),
                    ),
                )
            )
            flat_idx += 1
    if flat_idx != n_traces:
        raise ValueError(
            f"walked {flat_idx} traces across the results but the DatasetResult's "
            f"columns carry {n_traces}; the signal axis and the key columns disagree."
        )

    return ClassifierBank(
        id=resolved_bank_id,
        banks=[bank_meta, theta_meta],
        traces=traces,
        labels=dict(dataset_result.labels_dict),
    )


# ---------------------------------------------------------------------------
# SyntheticBank 2.0 from DatasetResult
# ---------------------------------------------------------------------------


def _activation_position_column(
    results: Sequence[SimulationResult],
) -> list[ActivationPosition] | None:
    """Flatten per-simulation realized positions into the trace column.

    ``None`` when the run applied no crop. Mixed state — some simulations
    cropped, some not — cannot arise from a single run (one position generator
    is passed to every ``run_single`` call or none is), so it is treated as a
    programming error rather than silently half-filling the column.
    """
    cropped = [r.activation_positions is not None for r in results]
    if not any(cropped):
        return None
    if not all(cropped):
        raise ValueError(
            "some simulations were cropped and others were not; a bank's "
            "activation_position column must be all-or-nothing."
        )
    return [
        ActivationPosition(float(p))
        for result in results
        for p in result.activation_positions  # type: ignore[union-attr]
    ]


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
    bank_id_base: str | None = None,
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

    ``activation_position`` carries each trace's **realized** crop position
    when the run configured a position policy, and is **absent** when it did
    not. Absence means *no crop happened* — the schema is explicit that it
    never means 0.0, which is a legitimate position. Realized rather than
    requested: the two differ whenever ``p * (T - 1)`` is not an integer, and
    only the realized value describes the stored trace.

    A run either crops every trace or none of them, so the column is all-or-
    nothing rather than per-trace-optional. A partially-populated column would
    be indistinguishable from a run where detection failed on some pairs, and
    the crop raises on that case instead.

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
        activation_position=_activation_position_column(results),
        snr_db=list(snr_db) if snr_db is not None else [math.nan] * n_traces,
        noise_record=list(noise_record) if noise_record is not None else [""] * n_traces,
        noise_channel=(list(noise_channel) if noise_channel is not None else [""] * n_traces),
    )

    # The synthetic_bank takes the run's base id with a `theta` marker, so
    # it is distinct from its paired ClassifierBank (which keeps the bare
    # id) while still visibly belonging to the same run. `bank_id` is the
    # base for BOTH banks — one override keeps the pair in step.
    # `bank_id_base` is the paired ClassifierBank's id when the caller has
    # one (the mixed path passes the mixer's output id, so the two banks a
    # noise-mixed run writes stay a matched pair). Otherwise derive it.
    if bank_id_base is not None:
        base_bank_id = validate_artifact_id(bank_id_base)
    elif bank_id is not None:
        base_bank_id = validate_artifact_id(bank_id)
    else:
        base_bank_id = derive_synthetic_bank_id(
            _cell_model_from_backend_meta(backend_meta_first),
            noise_mixed=mixed_signals is not None,
        )
    resolved_bank_id = validate_artifact_id(theta_bank_id_from(base_bank_id))

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
