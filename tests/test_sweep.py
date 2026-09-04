"""The sweep harness: the design it produces, and that it changes nothing else.

**The load-bearing test in this file is the bit-identical one.** Everything else
checks that the harness does what it says; that one checks it did not quietly
rewrite the per-simulation sampling while adding a loop around it. A sweep
feature that perturbed the no-sweep path would invalidate every bank generated
before it, and the only way to know is to compare the arrays.

It runs against the **mock backend**, which keeps it in the fast suite. That is
sound here because the property under test is about *which parameters reach the
backend and in what order the random stream is consumed* — not about the
physics. A mock that returns `rng.standard_normal(...)` is in fact a sharper
instrument for it than the solver would be: any disturbance to the draw order
shows up immediately as different noise, where a real solve might absorb it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend
from myocard_synthetic_egm_pipeline.constants import DEFAULT_TRACE_DURATION_MS
from myocard_synthetic_egm_pipeline.mixer import MixerConfig
from myocard_synthetic_egm_pipeline.simulate import Patch2DGeometry, UniformRandomFibrosis
from myocard_synthetic_egm_pipeline.simulate.dataset import (
    DatasetConfig,
    DatasetResult,
    generate_dataset,
    generate_sweep,
)
from myocard_synthetic_egm_pipeline.simulate.label_policy import GlobalDensityLabel
from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card
from myocard_synthetic_egm_pipeline.simulate.sweep import (
    DesignCell,
    Knob,
    OATSampler,
    Sampler,
    SweepConfig,
    SweepConfigError,
    apply_cell,
    build_design,
)
from myocard_synthetic_egm_pipeline.simulate.tuning import TunableRun


def traces_of(result: DatasetResult) -> npt.NDArray[np.float64]:
    """Stack every simulation's bipolar traces.

    ``DatasetResult`` holds per-simulation results rather than one array, so the
    comparison has to assemble it — which is also the honest thing to compare,
    since it is what a bank is built from.
    """
    return np.concatenate([np.asarray(r.bipolar_traces, dtype=np.float64) for r in result.results])


def _density_of(sim: object) -> float:
    """The realized fibrosis density one simulation actually ran at."""
    substrate = sim.specs.substrate  # type: ignore[attr-defined]
    assert isinstance(substrate, UniformRandomFibrosis)
    return float(substrate.density)


def dataset_config(**overrides: object) -> DatasetConfig:
    base = {
        "n_simulations": 2,
        "geometry": Patch2DGeometry(size_mm=12.0, dr_mm=0.25),
        "label_policy": GlobalDensityLabel(threshold=0.1),
        "run_config": RunConfig(trace_duration_ms=DEFAULT_TRACE_DURATION_MS, output_fs_hz=1000.0),
        "fibrosis_density_range": (0.0, 0.5),
        "fraction_healthy": 0.0,
        "electrode_height_mm_range": (0.2, 1.0),
        "master_seed": 20260903,
        "show_progress": False,
    }
    base.update(overrides)
    return DatasetConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------


def test_an_oat_design_varies_exactly_one_knob_per_cell() -> None:
    """The defining property of a one-at-a-time screen.

    Every cell differs from the all-nominal baseline in **at most one**
    coordinate. Checked against the baseline rather than against the previous
    cell, because "one knob changed since the last row" would also be true of a
    design that drifted a knob and never returned it.
    """
    knobs = (
        Knob(path="substrate.density_range.high", bounds=(0.1, 0.5), role="label_param"),
        Knob(path="mix.snr_db_range.low", bounds=(5.0, 15.0)),
        Knob(path="geometry.anisotropy_ratio", bounds=(1.0, 4.0)),
    )
    cells = build_design(SweepConfig(sampler=OATSampler(levels=3), knobs=knobs))

    baseline = next(c for c in cells if c.varied is None)
    assert baseline.values == {k.path: k.nominal for k in knobs}

    for cell in cells:
        differing = [p for p, v in cell.values.items() if v != baseline.values[p]]
        if cell.varied is None:
            assert differing == []
        else:
            assert differing == [cell.varied], (cell.index, differing)


def test_every_cell_records_all_knobs_not_just_the_varied_one() -> None:
    """Otherwise a screen cannot say what the others were held at."""
    knobs = (
        Knob(path="substrate.density_range.high", bounds=(0.1, 0.5)),
        Knob(path="mix.snr_db_range.low", bounds=(5.0, 15.0)),
    )
    for cell in build_design(SweepConfig(sampler=OATSampler(), knobs=knobs)):
        assert set(cell.values) == {k.path for k in knobs}


def test_levels_span_the_bounds_and_the_baseline_is_not_regenerated() -> None:
    """Endpoints are included, and the midpoint is the baseline rather than a repeat."""
    knob = Knob(path="mix.snr_db_range.low", bounds=(5.0, 15.0))
    cells = build_design(SweepConfig(sampler=OATSampler(levels=3), knobs=(knob,)))

    values = sorted(c.values[knob.path] for c in cells)
    assert values == pytest.approx([5.0, 10.0, 15.0])
    assert len(cells) == 3, "the mid level was regenerated instead of reusing the baseline"


def test_a_log_transform_spaces_levels_geometrically() -> None:
    """The point of recording a transform: it changes where the levels land.

    Bounds stay in natural units, as the contract requires, so this is visible
    as the spacing rather than as a change of scale.
    """
    knob = Knob(path="cell_model.params.g_CaL_scale", bounds=(0.1, 10.0), transform="log")
    assert knob.levels(3) == pytest.approx([0.1, 1.0, 10.0])

    linear = Knob(path="cell_model.params.g_CaL_scale", bounds=(0.1, 10.0))
    assert linear.levels(3) == pytest.approx([0.1, 5.05, 10.0])


def test_a_logit_transform_keeps_levels_off_the_boundary() -> None:
    knob = Knob(path="substrate.density_range.high", bounds=(0.01, 0.99), transform="logit")
    levels = knob.levels(5)
    assert all(0.0 < v < 1.0 for v in levels)
    assert levels[2] == pytest.approx(0.5)


def test_the_sampler_protocol_is_satisfied_structurally() -> None:
    """The harness is generic over the sampler; swapping it is the whole difference."""
    assert isinstance(OATSampler(), Sampler)
    assert OATSampler().type == "oat"


# ---------------------------------------------------------------------------
# What must not be swept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["activation.edge", "activation.fixed_edge"])
def test_the_stimulus_edge_cannot_be_made_a_knob(path: str) -> None:
    """Refused when the design is built, not once per cell at apply time.

    The resolver would refuse the write anyway, so nothing could be corrupted.
    This is the earlier and more useful failure: it names the scientific reason
    while someone is still designing the sweep.
    """
    with pytest.raises(SweepConfigError, match="shortcut feature"):
        Knob(path=path, bounds=(0.0, 1.0))


def test_bounds_must_be_ascending_and_finite() -> None:
    with pytest.raises(SweepConfigError, match="ascending"):
        Knob(path="mix.snr_db_range.low", bounds=(15.0, 5.0))
    with pytest.raises(SweepConfigError, match="finite"):
        Knob(path="mix.snr_db_range.low", bounds=(0.0, float("inf")))


def test_a_role_outside_the_contract_is_refused() -> None:
    with pytest.raises(SweepConfigError, match="role must be one of"):
        Knob(path="mix.snr_db_range.low", bounds=(5.0, 15.0), role="label")


def test_a_log_transform_needs_positive_bounds() -> None:
    with pytest.raises(SweepConfigError, match="strictly positive"):
        Knob(path="mix.snr_db_range.low", bounds=(0.0, 10.0), transform="log")


# ---------------------------------------------------------------------------
# Applying a cell
# ---------------------------------------------------------------------------


def test_a_cell_writes_through_to_the_dataset_config() -> None:
    """A design cell reaches the distributions, which is what a sweep can own."""
    run = TunableRun(dataset=dataset_config(), mixer=MixerConfig())
    cell = DesignCell(
        index=0,
        values={"substrate.density_range.high": 0.25, "mix.snr_db_range.low": 7.5},
        varied=None,
    )
    applied = apply_cell(run, cell)

    assert applied.dataset.fibrosis_density_range == pytest.approx((0.0, 0.25))
    assert applied.mixer is not None
    assert applied.mixer.snr_db_range[0] == pytest.approx(7.5)


def test_applying_a_cell_does_not_mutate_the_original() -> None:
    config = dataset_config()
    run = TunableRun(dataset=config, mixer=MixerConfig())
    apply_cell(run, DesignCell(0, {"substrate.density_range.high": 0.25}, None))
    assert config.fibrosis_density_range == pytest.approx((0.0, 0.5))


# ---------------------------------------------------------------------------
# The property everything else rests on
# ---------------------------------------------------------------------------


def test_the_no_sweep_path_is_bit_identical(mock_backend: SimulationBackend) -> None:
    """Adding the harness changed nothing about an unswept run.

    **The check that proves this step is additive rather than a rewrite.** If it
    ever fails, every bank generated before the sweep landed is no longer
    reproducible from its recorded seed, which is a provenance failure and not
    merely a regression.

    Compared on the arrays, at the same master seed. ``assert_array_equal``
    rather than a tolerance: the two runs execute the same code over the same
    random stream, so anything but exact equality means the stream moved.
    """
    config = dataset_config()

    first = generate_dataset(config=config, backend=mock_backend)
    second = generate_dataset(config=config, backend=mock_backend)

    np.testing.assert_array_equal(traces_of(first), traces_of(second))
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.seeds, second.seeds)


def test_a_single_cell_sweep_reproduces_the_unswept_run_exactly(
    mock_backend: SimulationBackend,
) -> None:
    """The harness is a loop around the draw, not a replacement for it.

    A design whose only cell writes each knob back at the value the config
    already held must produce the same bank as not sweeping at all. This is the
    bit-identical property stated where it can actually catch a rewrite: if the
    harness reseeded, reordered, or re-drew anything, the arrays diverge.
    """
    config = dataset_config()
    unswept = generate_dataset(config=config, backend=mock_backend)

    # bounds collapsed to the config's own value, so the single level is a no-op
    knob = Knob(
        path="substrate.density_range.high",
        bounds=(0.5, 0.5),
        nominal=0.5,
    )
    sweep = SweepConfig(sampler=OATSampler(levels=2), knobs=(knob,))
    result = generate_sweep(config=config, backend=mock_backend, sweep=sweep)

    assert result.n_cells == 1, "a degenerate knob should give exactly the baseline cell"
    swept = result.runs[0].result

    np.testing.assert_array_equal(traces_of(swept), traces_of(unswept))
    np.testing.assert_array_equal(swept.labels, unswept.labels)
    np.testing.assert_array_equal(swept.seeds, unswept.seeds)


# ---------------------------------------------------------------------------
# The values have to survive into the bank
# ---------------------------------------------------------------------------


def test_a_cells_values_survive_into_the_generated_bank(
    mock_backend: SimulationBackend,
) -> None:
    """Asserted on the produced data, not on the config that was handed in.

    **Surviving is exactly what the refused paths fail to do.** A config-level
    assertion would pass against a harness that wrote into the void — set
    ``substrate.density``, watch ``_sample_specs`` overwrite it, and the config
    still says what you set. So this reads the realized densities back off the
    simulations and requires them to lie inside the swept range.
    """
    config = dataset_config(n_simulations=6, fraction_healthy=0.0)
    knob = Knob(path="substrate.density_range.high", bounds=(0.05, 0.05), nominal=0.05)
    sweep = SweepConfig(sampler=OATSampler(levels=2), knobs=(knob,))

    result = generate_sweep(config=config, backend=mock_backend, sweep=sweep)
    run = result.runs[0]

    densities = [_density_of(sim) for sim in run.result.results]
    assert densities, "no simulations ran"
    assert max(densities) <= 0.05 + 1e-12, (
        f"realized densities {densities} exceed the swept range's upper bound; "
        "the design cell did not reach the per-simulation draw"
    )
    # And the range really was narrowed relative to the unswept config, so this
    # is not passing because everything happens to be small.
    assert config.fibrosis_density_range[1] == pytest.approx(0.5)


def test_two_cells_produce_visibly_different_banks(
    mock_backend: SimulationBackend,
) -> None:
    """A sweep that varied nothing would pass every test above but this one."""
    config = dataset_config(n_simulations=4, fraction_healthy=0.0)
    knob = Knob(path="substrate.density_range.high", bounds=(0.02, 0.40))
    sweep = SweepConfig(sampler=OATSampler(levels=2, include_baseline=False), knobs=(knob,))

    result = generate_sweep(config=config, backend=mock_backend, sweep=sweep)
    assert len(result.runs) == 2

    low, high = ([_density_of(s) for s in run.result.results] for run in result.runs)
    assert max(low) < max(high), (low, high)


# ---------------------------------------------------------------------------
# Infeasible cells
# ---------------------------------------------------------------------------


def test_an_infeasible_cell_is_recorded_rather_than_dropped(
    mock_backend: SimulationBackend,
) -> None:
    """A dropped cell biases the emulator and hides the boundary.

    The failure used here is a **value-level** one — a non-positive conduction
    velocity, which the target's own domain check refuses. That distinction is
    the point: a path that can never be written is a malformed design and aborts
    the sweep, while a value that cannot be solved is a boundary and is
    recorded.

    Note what is *not* being tested: the stability bound. It was expected to be
    the source of infeasible cells and, measured, is not — the calibrators
    derive the timestep from the bound rather than checking a proposal against
    it, so the conduction-velocity axis has no infeasible region from that
    direction. The recording is general for exactly that reason.
    """
    card = load_model_card("af_remodelled_220ms", dr_mm=0.25, dr_model_units=0.25)
    config = dataset_config(cell_model=card.solved)
    knob = Knob(
        path="cell_model.targets.conduction_velocity_cm_s",
        bounds=(-10.0, 80.0),
        nominal=80.0,
    )
    sweep = SweepConfig(sampler=OATSampler(levels=2, include_baseline=False), knobs=(knob,))

    result = generate_sweep(config=config, backend=mock_backend, sweep=sweep, card=card)

    assert len(result.infeasible) == 1, "the unsolvable cell was dropped"
    assert len(result.runs) == 1, "the solvable cell did not run"
    assert result.n_cells == 2, "the failed cell vanished from the count"

    record = result.infeasible[0]
    assert record.path == "cell_model.targets.conduction_velocity_cm_s"
    assert record.value == pytest.approx(-10.0)
    assert record.error_type == "InfeasibleTargetError"
    assert "positive" in record.reason


def test_a_knob_that_can_never_be_written_aborts_the_design(
    mock_backend: SimulationBackend,
) -> None:
    """A typo must not masquerade as a design-space boundary.

    Every cell carries every knob, at nominal where it is not the varied one.
    So an unwritable path would make *every* cell infeasible, and the sweep
    would report a boundary where it actually has a misspelled or refused knob.
    Caught once, up front, with the resolver's own explanation attached.
    """
    config = dataset_config()
    sweep = SweepConfig(
        sampler=OATSampler(levels=2),
        knobs=(Knob(path="geometry.dr_mm", bounds=(0.1, 0.3)),),
    )
    with pytest.raises(SweepConfigError, match="cannot be written at all"):
        generate_sweep(config=config, backend=mock_backend, sweep=sweep)


def test_a_refused_realized_path_is_also_caught_up_front(
    mock_backend: SimulationBackend,
) -> None:
    """The commonest mistake: sweeping the drawn value instead of its range.

    The message carries the resolver's remedy, so someone who reached for
    ``substrate.density`` is told to write ``substrate.density_range`` rather
    than left to wonder why every cell failed.
    """
    config = dataset_config()
    sweep = SweepConfig(
        sampler=OATSampler(levels=2),
        knobs=(Knob(path="substrate.density", bounds=(0.1, 0.4)),),
    )
    with pytest.raises(SweepConfigError, match=r"substrate\.density_range"):
        generate_sweep(config=config, backend=mock_backend, sweep=sweep)


# ---------------------------------------------------------------------------
# Reachable from a config file, which is the point of the block
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_swept_run_writes_exactly_one_bank_pair(tmp_path: Path) -> None:
    """One pair for the whole sweep, not one per design cell.

    **A bank is the unit that holds a design.** Its theta-spec states which
    knobs *this bank's sweep* varied, and an unswept bank is defined as one with
    an empty knob list — so a bank per cell would give every file a spec
    describing a one-point sweep, which is neither, and the design would be
    recorded nowhere.

    Slow-marked: it drives the real backend, which is the only way to know the
    config block reaches generation rather than merely parsing.
    """
    import yaml

    from myocard_synthetic_egm_pipeline.cli import generate_dataset_cmd

    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"n_simulations": 1, "master_seed": 3, "show_progress": False},
                "geometry": {"size_mm": 8.0, "dr_mm": 0.25},
                "electrodes": {"n_rows": 1, "n_cols": 2, "spacing_mm": 2.0},
                "run": {"trace_duration_ms": 64.0, "output_fs_hz": 1000.0},
                "sweep": {
                    "sampler": {"type": "oat", "levels": 2},
                    "knobs": [
                        {
                            "path": "substrate.density_range.high",
                            "bounds": [0.05, 0.3],
                            "role": "nuisance",
                        }
                    ],
                },
                "output": {"classifier_bank": str(tmp_path / "screen.h5")},
            }
        ),
        encoding="utf-8",
    )

    assert generate_dataset_cmd.main([str(config)]) == 0

    written = sorted(p.name for p in tmp_path.glob("*.h5"))
    assert written == ["screen.h5", "screen.synthetic.h5"], written
    assert not list(tmp_path.glob("*cell*")), "per-cell artifacts are still being written"


@pytest.mark.slow
def test_the_written_bank_carries_the_whole_design(tmp_path: Path) -> None:
    """Read back off the artifact: ids, per-simulation values, and the theta-spec.

    Every assertion here is on the **written bank**, because that is the thing a
    later step consumes. A design cell whose values reached the config but not
    the file would pass any config-level check and be useless.
    """
    import yaml
    from myocard_egm_data.banks import read_synthetic_bank_hdf5

    from myocard_synthetic_egm_pipeline.cli import generate_dataset_cmd

    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"n_simulations": 2, "master_seed": 7, "show_progress": False},
                "geometry": {"size_mm": 8.0, "dr_mm": 0.25},
                "electrodes": {"n_rows": 1, "n_cols": 2, "spacing_mm": 2.0},
                "run": {"trace_duration_ms": 64.0, "output_fs_hz": 1000.0},
                "substrate": {"density_range": [0.0, 0.5], "fraction_healthy": 0.0},
                "sweep": {
                    "sampler": {"type": "oat", "levels": 2},
                    "knobs": [
                        {
                            "path": "substrate.density_range.high",
                            "bounds": [0.04, 0.32],
                            "role": "nuisance",
                            "nominal": 0.18,
                        }
                    ],
                },
                "output": {"classifier_bank": str(tmp_path / "screen.h5")},
            }
        ),
        encoding="utf-8",
    )
    assert generate_dataset_cmd.main([str(config)]) == 0

    bank = read_synthetic_bank_hdf5(tmp_path / "screen.synthetic.h5")

    # --- simulation_id is unique across the bank and spans every cell --------
    ids = list(bank.simulations.simulation_id)
    assert len(ids) == 6, f"3 cells x 2 simulations expected, got {len(ids)}"
    assert len(set(ids)) == len(ids), f"simulation_id repeats across cells: {ids}"
    assert sorted(ids) == list(range(6)), ids

    # --- each cell's parameter values are recoverable per simulation ---------
    # Read the realized densities back off the bank. Three cells at three
    # different upper bounds must show three different ceilings.
    densities = [float(d.root.density) for d in bank.simulations.substrate]
    assert len(densities) == 6
    ceilings = {round(max(densities[i : i + 2]), 6) for i in range(0, 6, 2)}
    assert len(ceilings) == 3, (
        f"the three design cells are indistinguishable in the bank: {densities}"
    )

    # --- the theta-spec describes the whole design, not one point -----------
    knobs = bank.generation_params.knobs
    assert len(knobs) == 1, "the design's knob list did not reach the bank"
    knob = knobs[0]
    assert knob.path == "substrate.density_range.high"
    assert [float(b) for b in knob.bounds] == pytest.approx([0.04, 0.32]), (
        "bounds record the swept range, which is the emulator's input domain"
    )
    assert knob.role is not None
    assert knob.role.value == "nuisance"
    assert knob.nominal is not None
    assert float(knob.nominal) == pytest.approx(0.18)


@pytest.mark.slow
def test_an_infeasible_cell_is_recorded_in_the_artifact(tmp_path: Path) -> None:
    """Not merely absent from disk.

    A missing file was never a record of an infeasible cell: absence is
    indistinguishable from a run that never happened, or from one that crashed
    halfway. With a single bank there are no per-cell files at all, so the
    record has to be *in* the artifact.

    It lands in the description, and that is a compromise worth stating:
    ``generation_params`` is closed — ``regime`` and ``knobs``,
    ``additionalProperties: false`` — and the bank has no free-form metadata
    slot, so a structured field would be a contracts change.
    """
    import yaml
    from myocard_egm_data.banks import read_synthetic_bank_hdf5

    from myocard_synthetic_egm_pipeline.cli import generate_dataset_cmd

    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"n_simulations": 1, "master_seed": 5, "show_progress": False},
                "geometry": {"size_mm": 8.0, "dr_mm": 0.25},
                "electrodes": {"n_rows": 1, "n_cols": 2, "spacing_mm": 2.0},
                "run": {"trace_duration_ms": 64.0, "output_fs_hz": 1000.0},
                "backend": {"model": "af_remodelled_220ms"},
                "sweep": {
                    "sampler": {"type": "oat", "levels": 2},
                    "knobs": [
                        {
                            "path": "cell_model.targets.conduction_velocity_cm_s",
                            "bounds": [-5.0, 80.0],
                            "nominal": 80.0,
                        }
                    ],
                },
                "output": {
                    "classifier_bank": str(tmp_path / "screen.h5"),
                    "description": "screen",
                },
            }
        ),
        encoding="utf-8",
    )
    assert generate_dataset_cmd.main([str(config)]) == 0

    bank = read_synthetic_bank_hdf5(tmp_path / "screen.synthetic.h5")
    assert bank.description is not None
    assert "infeasible design cells" in bank.description
    assert "cell_model.targets.conduction_velocity_cm_s" in bank.description
    assert "InfeasibleTargetError" in bank.description
