"""Builder round-trip tests.

Each builder is a pure function — same input, same output, no I/O.
The fixtures in conftest.py construct a minimal DatasetResult /
ClassifierBank that exercises every per-trace metadata field and
every top-level provenance field.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from myocard_egm_contracts._generated.python.synthetic_bank import (
    AlievPanfilovCellModel,
    PlanarEdgeActivation,
)
from myocard_egm_data.banks import (
    read_synthetic_bank_hdf5,
    synthetic_bank_to_classifier,
    write_synthetic_bank,
)

from myocard_synthetic_egm_pipeline.constants import LOCAL_BANK_PATH, THETA_BANK_SOURCE
from myocard_synthetic_egm_pipeline.ids import companion_path
from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig,
    DatasetResult,
    build_classifier_bank_from_dataset,
    build_clean_trace_metadata,
    build_synthetic_bank_from_dataset,
)
from myocard_synthetic_egm_pipeline.simulate.bank_config import (
    UnsupportedSpecError,
    cell_model_model,
)
from myocard_synthetic_egm_pipeline.simulate.builders import AMP_TYPE
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel as ProducerAlievPanfilovCellModel,
)

# ---------------------------------------------------------------------------
# build_clean_trace_metadata
# ---------------------------------------------------------------------------


def test_trace_metadata_is_identity_only() -> None:
    """A ClassifierBank trace carries identity, not generation parameters.

    The bank is a source-agnostic ML compression; generation params are
    the subject of the ``synthetic_bank`` and reachable through
    ``simulation_id``. Keeping a copy here would be the same flat
    per-trace duplication the 2.0 restructure removed from the other
    artifact.
    """
    md = build_clean_trace_metadata(simulation_id=3, pair_idx=7)

    assert set(md.keys()) == {"simulation_id", "pair_index", "patient_id"}
    # patient_id groups one simulation as one patient for the split.
    assert md["patient_id"] == "3"
    assert md["pair_index"] == 7


def test_trace_metadata_carries_no_generation_params() -> None:
    """Named explicitly, so a re-added generation key fails loudly.

    Each of these had a copy here *and* in the synthetic bank; the
    duplicate is what drifts.
    """
    md = build_clean_trace_metadata(simulation_id=0, pair_idx=0)

    for gone in (
        "fibrosis_density_requested",
        "fibrosis_density_realized",
        "electrode_row",
        "electrode_height_mm",
        "stim_edge",
        "sim_seed",
    ):
        assert gone not in md, gone


def test_clean_trace_metadata_has_no_noise_fields() -> None:
    """A clean bank says nothing about noise rather than saying NaN.

    A present-but-empty ``snr_db`` reads as "mixed, SNR unknown", which
    is the opposite of the truth. The mixer adds the three noise fields
    when it actually runs.
    """
    md = build_clean_trace_metadata(simulation_id=0, pair_idx=0)

    for noise_key in ("snr_db", "noise_record", "noise_channel"):
        assert noise_key not in md, noise_key


def test_build_clean_trace_metadata_handles_null_simulation_id() -> None:
    """If simulation_id is missing, patient_id falls back to the empty string."""
    md = build_clean_trace_metadata(simulation_id=None, pair_idx=0)

    assert md["patient_id"] == ""


# ---------------------------------------------------------------------------
# build_classifier_bank_from_dataset
# ---------------------------------------------------------------------------


def test_classifier_bank_trace_count_matches_dataset(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Bank carries one ClassifierTrace per bipolar pair across all sims."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
    )
    assert len(bank.traces) == sum(r.n_pairs for r in small_dataset_result.results)


def test_classifier_bank_labels_propagate(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Per-trace label_truth matches the flat labels array."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
    )
    for trace, expected in zip(bank.traces, small_dataset_result.labels, strict=True):
        assert trace.label_truth == int(expected)


def test_classifier_bank_metadata_includes_provenance(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Bank-level metadata stamps the simulator name + config knobs."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
        description="round-trip test",
    )
    # Two entries: the origin entry for these traces, and a companion
    # entry naming the synthetic_bank that holds their generation config.
    assert len(bank.banks) == 2
    meta = bank.banks[0]
    assert meta.bank_type == "synthetic_egm_pipeline"
    assert meta.bank_metadata["description"] == "round-trip test"
    # Reproducibility: which code wrote this. synthetic_bank 2.0 has
    # nowhere to record it, so dropping it would lose the fact.
    assert meta.bank_metadata["producer"] == "synthetic_egm_pipeline"
    assert meta.bank_metadata["producer_version"]
    # The task definition, by identity only — thresholds are generation
    # detail and stay on the source bank.
    assert meta.bank_metadata["label_policy"] == "global_density"
    # The generation config is NOT copied here.
    for gone in (
        "backend",
        "cell_model",
        "geometry_size_mm",
        "electrode_spacing_mm",
        "fibrosis_density_range",
        "fixed_stim_edge",
        "ap_time_unit_ms",
    ):
        assert gone not in meta.bank_metadata, gone
    # Everything above is recoverable per-simulation from the companion bank.
    assert meta.bank_metadata["trace_duration_ms"] > 0


def test_classifier_bank_amp_type_is_synthetic_au(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Every trace's amp_type advertises the synthetic_au convention."""
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("<test>"),
    )
    assert all(t.amp_type == AMP_TYPE for t in bank.traces)


# ---------------------------------------------------------------------------
# build_synthetic_bank_from_dataset (pre-mixer)
# ---------------------------------------------------------------------------


def test_synthetic_bank_premixer_columns(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Pre-mixer SyntheticBank has snr_db = NaN and empty noise audit columns."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert all(math.isnan(v) for v in bank.traces.snr_db)
    assert all(s == "" for s in bank.traces.noise_record)
    assert all(s == "" for s in bank.traces.noise_channel)


def test_synthetic_bank_writes_per_simulation_config(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The generation config is one typed row per simulation, not per trace.

    This is the whole point of the 2.0 restructure: 1.1 repeated
    ``fibrosis_density`` / ``stim_edge`` / ``electrode_height_mm`` on
    every trace of a simulation, inviting the reading that traces from
    one simulation differed in those respects.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    sims = bank.simulations
    n_sims = len(small_dataset_result.results)

    assert len(sims.simulation_id) == n_sims
    assert len(bank.traces.signal) > n_sims  # many traces per simulation

    # Each per-function column decodes to its typed, discriminated object.
    assert sims.geometry[0].root.type == "patch_2d"
    assert sims.substrate[0].root.type == "uniform_random_fibrosis"
    assert sims.activation[0].type == "planar_edge"
    assert sims.electrodes[0].root.type == "centered_grid_2d"
    assert sims.backend[0].root.type == "finitewave"
    assert sims.cell_model[0].type == "aliev_panfilov"


def test_synthetic_bank_activation_edges_is_a_list(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``planar_edge.edges`` is a list even for today's single edge (CL-087).

    The schema is deliberately ahead of the producer here so SEP6's
    multi-edge stimulation needs no contracts bump.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    activation = bank.simulations.activation[0]
    assert isinstance(activation, PlanarEdgeActivation)
    assert isinstance(activation.edges, list)
    assert len(activation.edges) == 1


def test_synthetic_bank_label_policy_uses_thresholds_list(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The policy carries ``thresholds[]`` and no class names (CL-088).

    Names live once in the per-simulation ``label_names`` map; writing
    them into the policy too would let the two disagree with nothing to
    catch it. A one-element list is the binary case; Phase-2 multiclass
    is a longer list rather than a new schema variant.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    policy = bank.simulations.label_policy[0]
    assert isinstance(policy.thresholds, list)
    assert len(policy.thresholds) == 1
    assert not hasattr(policy, "healthy_name")
    assert not hasattr(policy, "fibrotic_name")
    assert bank.simulations.label_names[0] == {"0": "healthy", "1": "fibrotic"}


def test_synthetic_bank_electrode_pairs_are_per_simulation(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The realized per-pair detail 1.1 repeated per trace lives once per sim."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    pairs = bank.simulations.electrodes[0].root.pairs
    n_pairs_first_sim = small_dataset_result.results[0].n_pairs

    assert len(pairs) == n_pairs_first_sim
    assert [p.pair_index for p in pairs] == list(range(n_pairs_first_sim))
    assert pairs[0].midpoint_mm is not None
    # Traces index this list by pair_index — the FK the restructure adds.
    # Constrained scalar columns codegen to RootModel wrappers, so the
    # value is under .root (egm-data has an _unwrap helper for the same).
    assert max(int(p.root) for p in bank.traces.pair_index) < len(pairs)


def test_synthetic_bank_trace_columns_are_collapsed(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``traces/`` keeps only the signal, the two FKs, the label and noise."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    n = len(bank.traces.signal)
    assert len(bank.traces.simulation_id) == n
    assert len(bank.traces.pair_index) == n
    assert len(bank.traces.label) == n
    # The 1.1 generation columns are gone, not relocated.
    for removed in (
        "fibrosis_density",
        "fibrosis_density_realized",
        "electrode_row",
        "electrode_height_mm",
        "stim_edge",
        "seed",
    ):
        assert not hasattr(bank.traces, removed), removed


def test_synthetic_bank_labels_match_the_dataset(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Per-trace labels are plain ints matching the DatasetResult exactly."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert [int(x.root) for x in bank.traces.label] == [int(x) for x in small_dataset_result.labels]


def test_synthetic_bank_activation_position_absent_in_wave_1(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """No controlled crop yet, so the column is absent — never 0.0.

    The schema is explicit that absence means *unknown*; 0.0 is a
    legitimate position (activation on the first sample), so defaulting
    would fabricate a spike at the low edge of the distribution. SEP2
    populates it.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    assert bank.traces.activation_position is None


def test_theta_spec_records_the_regime_with_no_knobs(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Wave 1 sweeps nothing, but the regime is still stated explicitly."""
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    theta = bank.generation_params
    assert theta.knobs == []
    assert theta.regime["geometry"] == "patch_2d"
    assert theta.regime["substrate"] == "uniform_random_fibrosis"
    assert theta.regime["cell_model"] == "aliev_panfilov"


def test_synthetic_bank_noise_columns_when_mixed(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The inline mixer path supplies signals + the three noise columns.

    2.0's per-simulation config cannot be rebuilt from a mixed
    ClassifierBank, so the noise-mixed bank is written from the same
    DatasetResult with only the signals and noise provenance replaced.
    """
    n = len(small_dataset_result.labels)
    mixed = [
        np.full(small_dataset_result.results[0].n_samples, 0.5, dtype=np.float32) for _ in range(n)
    ]
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        mixed_signals=mixed,
        snr_db=[12.0] * n,
        noise_record=["iaf3_afw"] * n,
        noise_channel=["CS12"] * n,
        noise_bank_source="iafdb v1.0.0",
    )
    assert bank.noise_bank_source == "iafdb v1.0.0"
    assert all(v == 12.0 for v in bank.traces.snr_db)
    assert all(r == "iaf3_afw" for r in bank.traces.noise_record)
    # The generation config is still the clean run's — only signals changed.
    assert len(bank.simulations.simulation_id) == len(small_dataset_result.results)


def test_synthetic_bank_rejects_mismatched_mixed_signal_count(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """A short mixed-signal list is a bug, not something to pad around."""
    with pytest.raises(ValueError, match="mixed_signals"):
        build_synthetic_bank_from_dataset(
            dataset_result=small_dataset_result,
            config=small_dataset_config,
            mixed_signals=[np.zeros(4, dtype=np.float32)],
        )


# ---------------------------------------------------------------------------
# Round-trip through the real egm-data writer/reader
# ---------------------------------------------------------------------------


def test_synthetic_bank_round_trips_through_egm_data(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
    tmp_path: Path,
) -> None:
    """A written 2.0 bank reads back with its config and FKs intact.

    The unit tests above check the in-memory model; this is the one that
    proves the producer and egm-data agree on the on-disk layout — the
    Wave-1 gate's real question. It also exercises the writer's two
    refusals (no bank_id, orphan trace) by not tripping them.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        description="round-trip fixture",
    )
    path = write_synthetic_bank(bank, tmp_path / "rt.synthetic.h5")
    reloaded = read_synthetic_bank_hdf5(path)

    assert str(reloaded.schema_version.value) == "2.0"
    assert reloaded.bank_id == bank.bank_id
    assert len(reloaded.simulations.simulation_id) == len(bank.simulations.simulation_id)
    assert len(reloaded.traces.signal) == len(bank.traces.signal)
    # The typed per-function config survives the JSON-column encoding.
    assert reloaded.simulations.geometry[0].root.size_mm == pytest.approx(
        bank.simulations.geometry[0].root.size_mm
    )
    reloaded_activation = reloaded.simulations.activation[0]
    written_activation = bank.simulations.activation[0]
    assert isinstance(reloaded_activation, PlanarEdgeActivation)
    assert isinstance(written_activation, PlanarEdgeActivation)
    assert reloaded_activation.edges == written_activation.edges
    assert reloaded.simulations.label_policy[0].thresholds == (
        bank.simulations.label_policy[0].thresholds
    )
    # Every trace's FK resolves into simulations/ — the invariant JSON
    # Schema cannot express and the restructure exists to guarantee.
    sim_ids = {int(x) for x in reloaded.simulations.simulation_id}
    assert {int(x) for x in reloaded.traces.simulation_id} <= sim_ids


def test_written_bank_converts_to_a_classifier_bank(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
    tmp_path: Path,
) -> None:
    """egm-data can derive a ClassifierBank from what we write.

    We do **not** use this path in the producer (design note D7 — the
    converter hardcodes ``amp_type="mv"``, which is wrong for
    relative-unit synthetic traces; see FB-17). But a consumer may, so
    the bank we emit has to be convertible: labels come off the bank
    itself and the join keys land in ``trace_metadata``.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    path = write_synthetic_bank(bank, tmp_path / "conv.synthetic.h5")
    cb = synthetic_bank_to_classifier(read_synthetic_bank_hdf5(path), bank_path=path)

    assert len(cb.traces) == len(bank.traces.signal)
    assert cb.traces[0].trace_metadata["simulation_id"] == 0
    assert cb.traces[0].trace_metadata["patient_id"] == "0"
    assert cb.labels == {0: "healthy", 1: "fibrotic"}


def test_backend_object_fills_its_typed_fields_from_metadata(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Known backend facts land in named fields, not the generic bag.

    The backend reports its version as ``finitewave_version_pin`` and
    its timestep as ``ap_dt_model_units``; the schema has ``version``
    and ``dt_model_units`` for exactly those. Leaving them in ``params``
    would defeat the reason 2.0 has typed fields at all.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    backend = bank.simulations.backend[0].root

    assert backend.version is not None
    assert backend.dt_model_units is not None
    assert backend.output_fs_hz == small_dataset_config.run_config.output_fs_hz
    # ap_time_unit_ms belongs to the cell model, not the backend.
    assert "ap_time_unit_ms" not in (backend.params or {})
    cell_model = bank.simulations.cell_model[0]
    assert isinstance(cell_model, AlievPanfilovCellModel)
    assert cell_model.ap_time_unit_ms is not None


# ---------------------------------------------------------------------------
# The cell model comes from the spec, not from the backend's class name (S18a)
# ---------------------------------------------------------------------------


def _with_backend_metadata(dataset_result: DatasetResult, **overrides: object) -> DatasetResult:
    """Copy a DatasetResult with every result's ``backend_metadata`` overridden.

    The lever these tests pull. They are the "what if this input were
    ignored?" check for the identity path: the metadata is made to *lie*,
    and the bank must not notice.
    """
    results = []
    for result in dataset_result.results:
        run_metadata = dict(result.run_metadata)
        run_metadata["backend_metadata"] = {
            **dict(run_metadata.get("backend_metadata", {})),
            **overrides,
        }
        results.append(replace(result, run_metadata=run_metadata))
    return replace(dataset_result, results=results)


def test_cell_model_identity_ignores_the_backends_class_name(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``model_class`` is provenance about the solver, not the model's identity.

    Until S18a the bank recovered the cell model by matching this string
    against ``"AlievPanfilov2D"`` — so the identity of every simulation
    this project has recorded rode on the class name a third-party
    package happened to choose, and a rename upstream would have written
    a bank that mislabelled its own physics.

    The metadata here says something else entirely; the spec says
    Aliev-Panfilov, and the spec is what ran.
    """
    lying = _with_backend_metadata(
        small_dataset_result,
        model_class="TotallyDifferentModel3D",
    )

    bank = build_synthetic_bank_from_dataset(
        dataset_result=lying,
        config=small_dataset_config,
    )
    classifier = build_classifier_bank_from_dataset(
        dataset_result=lying,
        config=small_dataset_config,
        bank_path=Path("bank.h5"),
    )

    assert bank.simulations.cell_model[0].type == "aliev_panfilov"
    # The derived bank ids named the model from the same string.
    assert classifier.id is not None
    assert bank.bank_id is not None
    assert "aliev_panfilov" in classifier.id
    assert "totally_different_model3_d" not in classifier.id
    assert "aliev_panfilov" in bank.bank_id


def test_ap_time_unit_comes_from_the_spec_not_the_metadata(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The contract still wants ``ap_time_unit_ms``; its source moved.

    S18a changed **where** the number comes from, not what lands on
    disk — so the check is that a bank built from a result whose
    metadata carries a *different* value still records the spec's. Read
    the other way round, this is what stops the calibration the solver
    actually integrated and the calibration the bank claims from drifting
    apart: there is now one copy, and it is the one the backend was
    handed.
    """
    spec = small_dataset_result.results[0].specs.cell_model
    assert isinstance(spec, ProducerAlievPanfilovCellModel)
    spec_value = spec.time_unit_ms
    lying = _with_backend_metadata(small_dataset_result, ap_time_unit_ms=spec_value * 3.0)

    bank = build_synthetic_bank_from_dataset(
        dataset_result=lying,
        config=small_dataset_config,
    )

    cell_model = bank.simulations.cell_model[0]
    assert isinstance(cell_model, AlievPanfilovCellModel)
    assert cell_model.ap_time_unit_ms == pytest.approx(spec_value)


def test_cell_model_mapper_refuses_an_unwired_spec() -> None:
    """A spec the mapper has no branch for refuses rather than guesses.

    A spec the mapper has no branch for is a bank whose per-simulation
    config would not describe the simulation that produced it — the
    failure the 2.0 restructure exists to prevent — so it raises, exactly
    as the activation and electrode mappers do for their unbuilt
    variants. Courtemanche was the placeholder here until S18b wired it;
    the guard is about the *next* model, so it now names one the schema
    itself does not have.
    """

    class _NotYetBuilt:
        dt_model_units = 0.01

        @property
        def type(self) -> str:
            return "hodgkin_huxley"

        def ms_to_model_time(self, duration_ms: float) -> float:
            return duration_ms

        def stability_limit(self, *, dr_model_units: float, dimensions: int = 2) -> float:
            return 1.0

        def to_metadata(self) -> dict[str, object]:
            return {}

    with pytest.raises(UnsupportedSpecError, match="hodgkin_huxley"):
        cell_model_model(_NotYetBuilt())


def test_substrate_summary_carries_the_realized_draw(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The realized density/count are recorded, not just the request.

    Requested and realized genuinely differ through grid discretization,
    and the label is computed from the realized one — so a bank that
    stored only the request could not explain its own labels.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    summary = bank.simulations.substrate_summary[0]
    assert summary.realized_density is not None
    assert summary.n_fibrotic_nodes is not None


def test_seed_column_carries_the_run_master_seed(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """``simulations/seed`` is the run's master seed, repeated per row.

    It is a bank-scoped fact that 2.0 placed in the per-simulation group
    by oversight (CL-096); FB-16 moves it to a root attr. Writing the
    per-simulation derived seed here instead would change the column's
    meaning with no version bump to signal it — so the repetition is
    deliberate, not an accident of the loop.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    seeds = [int(x) for x in bank.simulations.seed]

    assert seeds == [small_dataset_config.master_seed] * len(small_dataset_result.results)


# ---------------------------------------------------------------------------
# SEP12.3b — the two banks a run emits must agree
# ---------------------------------------------------------------------------


def test_both_banks_from_one_run_agree(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The ClassifierBank and the synthetic_bank describe the same traces.

    A run emits both, and they are **parallel artifacts joined on
    ``simulation_id``** rather than one derived from the other (design
    note D4). Because the producer builds them through two independent
    code paths (D7 — we deliberately do not route through egm-data's
    converter, which would hardcode ``amp_type="mv"`` onto relative-unit
    synthetic traces, FB-17), nothing structural forces them to match.

    This is the guard that replaces the by-construction guarantee: same
    trace count, same order, same labels, same join key. If either
    builder's ordering or labelling drifts, the join that T4 depends on
    silently starts pairing the wrong rows — which would not show up as
    an error anywhere, just as wrong science.
    """
    classifier = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("agree.classifier.h5"),
    )
    synthetic = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )

    assert len(classifier.traces) == len(synthetic.traces.signal)

    classifier_sim_ids = [t.trace_metadata["simulation_id"] for t in classifier.traces]
    synthetic_sim_ids = [int(x) for x in synthetic.traces.simulation_id]
    assert classifier_sim_ids == synthetic_sim_ids

    classifier_labels = [t.label_truth for t in classifier.traces]
    synthetic_labels = [int(x.root) for x in synthetic.traces.label]
    assert classifier_labels == synthetic_labels

    # Same waveform in the same row of both banks.
    assert classifier.traces[0].signal == pytest.approx(
        np.asarray(synthetic.traces.signal[0], dtype=np.float32)
    )


def test_both_banks_agree_on_the_noise_mixed_path(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The agreement has to survive mixing, where the two paths diverge most.

    The ClassifierBank is mixed by the mixer; the synthetic_bank is
    rebuilt from the DatasetResult with the mixed signals passed in. Two
    different routes to the same waveforms is exactly where a drift
    would appear.
    """
    n = len(small_dataset_result.labels)
    mixed = [
        np.full(small_dataset_result.results[0].n_samples, 0.25, dtype=np.float32) for _ in range(n)
    ]
    synthetic = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        mixed_signals=mixed,
        snr_db=[10.0] * n,
        noise_record=["iaf1_afw"] * n,
        noise_channel=["CS12"] * n,
    )
    classifier = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("agree_mixed.classifier.h5"),
    )

    assert [int(x) for x in synthetic.traces.simulation_id] == [
        t.trace_metadata["simulation_id"] for t in classifier.traces
    ]
    assert [int(x.root) for x in synthetic.traces.label] == [
        t.label_truth for t in classifier.traces
    ]
    # The mixed signal is what landed in the synthetic bank, not the clean one.
    assert synthetic.traces.signal[0][0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# SEP12.7 — the Wave-1 equivalence gate
# ---------------------------------------------------------------------------


def test_per_simulation_config_recovers_the_1_1_per_trace_values(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Everything 1.1 stored per trace is recoverable from the 2.0 config.

    This is what the Wave-1 gate actually asks: the restructure moved
    fields, so nothing may be *lost*. 1.1 wrote ``fibrosis_density``,
    ``stim_edge``, ``electrode_height_mm`` and ``electrode_row`` onto
    every trace; 2.0 stores each once per simulation (and the row on the
    realized pair). A value that no longer round-trips is a regression
    the schema itself cannot catch, because the new bank would still
    validate.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )

    for i, result in enumerate(small_dataset_result.results):
        run_meta = result.run_metadata
        assert bank.simulations.substrate[i].root.density == pytest.approx(
            run_meta["fibrosis_density_requested"]
        )
        activation = bank.simulations.activation[i]
        assert isinstance(activation, PlanarEdgeActivation)
        assert run_meta["stim_edge"] in [e.value for e in activation.edges]

        electrodes = bank.simulations.electrodes[i].root
        assert electrodes.height_mm == pytest.approx(run_meta["electrode_height_mm"])
        expected_rows = run_meta["electrode_row_per_pair"]
        assert [p.electrode_row for p in electrodes.pairs] == list(expected_rows)


def test_signals_are_written_unchanged(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """Serialization is lossless: bank signals == simulator output.

    The restructure touched only how a bank is *shaped*; if a waveform
    changed on the way to disk, the phase's science would rest on
    different data than the simulator produced.
    """
    bank = build_synthetic_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
    )
    expected = np.concatenate([r.bipolar_traces for r in small_dataset_result.results], axis=0)
    written = np.asarray(bank.traces.signal, dtype=np.float32)

    assert np.array_equal(written, expected)


@pytest.mark.slow
def test_wave_1_equivalence_against_the_real_backend(tmp_path: Path) -> None:
    """The same checks, but with Finitewave actually solving.

    The MockBackend returns canned traces, so on its own it proves the
    plumbing rather than the pipeline. This runs two real simulations and
    asserts the same three invariants end to end — signals unchanged,
    labels unchanged, per-simulation config recovering the 1.1 per-trace
    values. Opt-in (``pytest -m slow``) to keep the default suite ~1 s.
    """
    from myocard_synthetic_egm_pipeline.backends import RunConfig
    from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
    from myocard_synthetic_egm_pipeline.simulate import (
        LocalDensityLabel,
        Patch2DGeometry,
        generate_dataset,
    )

    config = DatasetConfig(
        n_simulations=2,
        geometry=Patch2DGeometry(size_mm=10.0, dr_mm=0.25, anisotropy_ratio=3.0),
        label_policy=LocalDensityLabel(radius_mm=2.0, threshold=0.1),
        run_config=RunConfig(trace_duration_ms=40.0, output_fs_hz=1000.0),
        fibrosis_density_range=(0.0, 0.4),
        electrode_n_rows=3,
        electrode_n_cols=3,
        electrode_spacing_mm=2.0,
        master_seed=42,
        show_progress=False,
    )
    dataset_result = generate_dataset(config=config, backend=FinitewaveBackend())

    synthetic = build_synthetic_bank_from_dataset(dataset_result=dataset_result, config=config)
    classifier = build_classifier_bank_from_dataset(
        dataset_result=dataset_result, config=config, bank_path=tmp_path / "x.h5"
    )

    expected = np.concatenate([r.bipolar_traces for r in dataset_result.results], axis=0)
    assert np.array_equal(np.asarray(synthetic.traces.signal, dtype=np.float32), expected)
    assert np.array_equal(np.stack([t.signal for t in classifier.traces]), expected)
    assert [int(x.root) for x in synthetic.traces.label] == [int(x) for x in dataset_result.labels]
    for i, result in enumerate(dataset_result.results):
        assert synthetic.simulations.substrate[i].root.density == pytest.approx(
            result.run_metadata["fibrosis_density_requested"]
        )


# ---------------------------------------------------------------------------
# SEP12.9 — what a `banks` entry means
# ---------------------------------------------------------------------------


def test_origin_entry_does_not_claim_a_source_file(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The traces originate here, so no source path is claimed.

    ``bank_path`` is documented as where a source bank was *loaded from*.
    A producer has no such file, and filling one in anyway is how this
    field came to name a bank that is never written (clean runs) or one
    whose traces differ from the file's own (noise-mixed runs).
    """
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("/banks/out.classifier.h5"),
    )
    origin = bank.banks[0]

    assert origin.bank_path == LOCAL_BANK_PATH
    # Distinguishable from "nobody filled this in", which is now a bug signal.
    assert origin.bank_path != ""
    # And from any real path, because < > cannot appear in a portable one.
    assert "<" in origin.bank_path


def test_companion_entry_links_to_the_theta_bank(
    small_dataset_result: DatasetResult,
    small_dataset_config: DatasetConfig,
) -> None:
    """The ClassifierBank names the synthetic_bank holding its generation config.

    Without it, pairing the two artifacts a run writes is a fact that
    lives only in someone's memory — and egm-studio has to be told.
    """
    bank = build_classifier_bank_from_dataset(
        dataset_result=small_dataset_result,
        config=small_dataset_config,
        bank_path=Path("/banks/run7.classifier.h5"),
        bank_id="tbank_run7",
        synthetic_bank_path=Path("/banks/run7.synthetic.h5"),
    )
    companion = next(b for b in bank.banks if b.bank_type == THETA_BANK_SOURCE)

    assert companion.bank_id == "tbank_run7_theta"
    # Siblings, so a bare filename — portable if the directory moves.
    assert companion.bank_path == "run7.synthetic.h5"
    assert companion.bank_metadata["join_key"] == "simulation_id"
    # No trace points at the companion: it describes the traces without
    # being where they came from.
    assert all(t.bank_id != companion.bank_id for t in bank.traces)


def test_companion_path_falls_back_to_absolute_when_far_away() -> None:
    """A distant companion gets an absolute path, not a chain of `..`.

    Portable-in-principle beats readable only up to a point; `../../../..`
    is neither, and an absolute path at least resolves where it was
    written.
    """
    near = companion_path("/banks/run7.synthetic.h5", relative_to="/banks/run7.classifier.h5")
    far = companion_path("/elsewhere/a/b/c/noise.h5", relative_to="/banks/run7.classifier.h5")

    assert near == "run7.synthetic.h5"
    assert Path(far).is_absolute()


def test_no_companion_path_when_target_is_unknown() -> None:
    """No target means empty — the caller had nothing to point at."""
    assert companion_path(None, relative_to="/banks/x.classifier.h5") == ""


def test_companion_path_uses_relative_across_sibling_directories() -> None:
    """A companion one directory over is still relative, not absolute.

    Banks routinely sit in sibling directories under one project root,
    and that root moves as a unit — so `../noise/x.h5` is portable and
    should be preferred. An earlier rule bailed to absolute on any `..`,
    which gave up exactly where relative was still useful.
    """
    assert (
        companion_path("/repo/noise/iafdb.h5", relative_to="/repo/banks/x.classifier.h5")
        == "../noise/iafdb.h5"
    )


def test_companion_path_absolute_only_across_unrelated_trees() -> None:
    """Two paths sharing only the filesystem root get an absolute path.

    Nothing stable links them, so a chain of `..` would express a
    relationship that does not exist.
    """
    result = companion_path("/other/tree/n.h5", relative_to="/repo/banks/x.classifier.h5")

    assert Path(result).is_absolute()
