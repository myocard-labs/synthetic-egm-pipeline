"""Tests for the positional-sensitivity probe.

One solve, **one logical simulation per crop offset**. The bank is a diagnostic
— the study it feeds plots classifier output against activation offset — so two
things have to be right: the x-axis must *reproduce* the grid rather than approximate it, and the
bank pair must be loadable at all.

That second one is why this file exists in its present form. The first
implementation made the trace axis ``(pair x grid point)`` inside a single
simulation, and a generated bank came out with ``pair_index`` running 0-59
against 20 real pairs. ``egm-studio``'s loader joins the ClassifierBank to its
theta companion on ``(simulation_id, pair_index)`` and **raises when either
side's key is non-unique**, so that bank was unloadable by the one consumer it
exists for. This repo cannot import egm-studio to find that out, so the
invariant is asserted here directly.

Three that would be easy to write vacuously, and are not:

- **Key uniqueness** is asserted on the written bank, not inferred from the
  shape of the code that wrote it.
- **The two banks are compared to each other**, trace for trace. The defect was
  the ClassifierBank and the theta bank *disagreeing*; every single-bank assert
  passed while it was present.
- **The cross-correlation** is read from two genuinely different grid points and
  its peak lag asserted to equal the offset difference and to be non-zero. A
  trace correlated against itself peaks at lag 0 whatever the sweep did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from myocard_egm_data.banks import load_classifier_bank, read_synthetic_bank_hdf5
from myocard_egm_signal import RectifiedDerivative, TeagerKaiser

from conftest import ambiguous_complex, ambiguous_complex_backend_for
from myocard_synthetic_egm_pipeline.cli import generate_dataset_cmd
from myocard_synthetic_egm_pipeline.cli._config import (
    ConfigError,
    build_generate_dataset_config,
)
from myocard_synthetic_egm_pipeline.simulate.probe import ProbeGrid, sweep_capture

T_SAMPLES = 192
"""``T`` every config here runs at, so the lattice is ``j/191``."""

LATTICE = T_SAMPLES - 1

N_ROWS, N_COLS = 2, 3
N_PAIRS = N_ROWS * (N_COLS - 1)
"""The electrode grid these configs use, and the pair count that follows."""

GRID_LOW, GRID_HIGH, GRID_N_POINTS = 0.3, 0.7, 5
GRID: dict[str, Any] = {"low": GRID_LOW, "high": GRID_HIGH, "n_points": GRID_N_POINTS}
"""The sweep these configs run, as scalars and as the config block."""


def _reference_grid() -> ProbeGrid:
    """The grid the runs below are expected to have snapped to."""
    return ProbeGrid.snapped(
        low=GRID_LOW,
        high=GRID_HIGH,
        n_points=GRID_N_POINTS,
        window_length_samples=T_SAMPLES,
    )


# ---------------------------------------------------------------------------
# Config + run helpers
# ---------------------------------------------------------------------------


def _config_doc(
    *,
    out_dir: Path,
    grid: dict[str, Any] | None = None,
    detection: dict[str, Any] | None = None,
    n_simulations: int = 1,
    stem: str = "probe",
) -> dict[str, Any]:
    """A complete generate-dataset config running a probe sweep."""
    activation_position: dict[str, Any] = {"grid": dict(GRID) if grid is None else grid}
    if detection is not None:
        activation_position["detection"] = detection
    return {
        "dataset": {
            "n_simulations": n_simulations,
            "master_seed": 3,
            "show_progress": False,
        },
        "backend": {"type": "finitewave"},
        "geometry": {"type": "patch_2d", "size_mm": 8.0, "dr_mm": 0.25},
        "substrate": {"type": "uniform_random_fibrosis", "density_range": [0.3, 0.3]},
        "activation": {"type": "planar_edge", "fixed_edge": "top"},
        "electrodes": {
            "type": "centered_grid_2d",
            "n_rows": N_ROWS,
            "n_cols": N_COLS,
            "spacing_mm": 2.0,
            "height_mm_range": [0.5, 0.5],
        },
        "label_policy": {"type": "global_density", "threshold": 0.1},
        "run": {
            "trace_duration_ms": float(T_SAMPLES),
            "output_fs_hz": 1000.0,
            "capture_oversample": 4,
        },
        "activation_position": activation_position,
        "output": {
            "classifier_bank": str(out_dir / f"{stem}.classifier.h5"),
            "description": "probe test",
        },
    }


def _typed_config(doc: dict[str, Any], tmp_path: Path) -> Any:
    return build_generate_dataset_config({**doc, "_config_dir": tmp_path})


def _run_probe(
    tmp_path: Path,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    grid: dict[str, Any] | None = None,
    detection: dict[str, Any] | None = None,
    stem: str = "probe",
) -> tuple[Path, Path]:
    """Run the CLI end to end; return ``(classifier_path, theta_path)``.

    Both, because the defect the probe was reworked to fix was the two banks
    disagreeing — a helper that returned only one would make that untestable.
    """
    doc = _config_doc(out_dir=tmp_path, grid=grid, detection=detection, stem=stem)
    config_path = tmp_path / f"{stem}.yaml"
    config_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setattr(
        generate_dataset_cmd,
        "FinitewaveBackend",
        lambda: ambiguous_complex_backend_for(mock_backend),
    )
    assert generate_dataset_cmd.main([str(config_path)]) == 0
    return tmp_path / f"{stem}.classifier.h5", tmp_path / f"{stem}.synthetic.h5"


def _root(value: Any) -> Any:
    """Unwrap an egm-contracts ``RootModel`` scalar."""
    return getattr(value, "root", value)


def _column(bank: Any, name: str) -> list[Any]:
    return [_root(v) for v in getattr(bank.traces, name)]


def _theta_keys(theta: Any) -> list[tuple[int, int]]:
    """``(simulation_id, pair_index)`` per trace, in bank order."""
    return list(
        zip(
            [int(v) for v in _column(theta, "simulation_id")],
            [int(v) for v in _column(theta, "pair_index")],
            strict=True,
        )
    )


def _classifier_keys(classifier: Any) -> list[tuple[int, int]]:
    return [
        (int(t.trace_metadata["simulation_id"]), int(t.trace_metadata["pair_index"]))
        for t in classifier.traces
    ]


def _theta_signals(theta: Any) -> npt.NDArray[np.float64]:
    return np.asarray(theta.traces.signal, dtype=np.float64)


# ---------------------------------------------------------------------------
# The composite key egm-studio joins on
# ---------------------------------------------------------------------------


def test_simulation_id_and_pair_index_are_unique_across_the_bank(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``(simulation_id, pair_index)`` is the bank pair's composite primary key.

    Asserted here because the consumer that enforces it lives in another repo:
    egm-studio's synthetic-bank loader matches traces to their theta companion
    on this key "rather than by position" and raises when either side repeats.
    A probe that emitted 60 traces under 20 pair indices produced exactly that
    duplicate, and nothing in *this* repo noticed.
    """
    classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)

    for label, keys in (
        ("theta", _theta_keys(read_synthetic_bank_hdf5(theta_path))),
        ("classifier", _classifier_keys(load_classifier_bank(classifier_path))),
    ):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        assert not duplicates, f"{label} bank repeats (simulation_id, pair_index): {duplicates}"
        assert len(keys) == GRID_N_POINTS * N_PAIRS


def test_both_banks_agree_on_the_key_trace_for_trace(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ClassifierBank and its theta companion carry the same keys, in order.

    This is the assert the original defect needed. `pair_index` was computed
    twice — `range(result.n_pairs)` on the ClassifierBank path and
    `dataset_result.pair_indices` on the theta path — so the two banks
    disagreed while each was internally consistent. Both now read the same
    array; comparing them is what keeps it that way.
    """
    classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)

    assert _classifier_keys(load_classifier_bank(classifier_path)) == _theta_keys(
        read_synthetic_bank_hdf5(theta_path)
    )


def test_pair_index_stays_inside_the_simulations_pair_list(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pair_index`` is a foreign key, on both banks.

    The failure it guards is silent: a value past the end of
    ``electrodes.pairs`` is an orphan reference that resolves to nothing, and
    no writer in this repo checks it.
    """
    classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)
    theta = read_synthetic_bank_hdf5(theta_path)

    # One entry per logical simulation now, all describing the same solve.
    pair_counts = {len(entry.root.pairs) for entry in theta.simulations.electrodes}
    assert pair_counts == {N_PAIRS}
    n_pairs = N_PAIRS

    for label, keys in (
        ("theta", _theta_keys(theta)),
        ("classifier", _classifier_keys(load_classifier_bank(classifier_path))),
    ):
        out_of_range = sorted({pair for _sim, pair in keys if not 0 <= pair < n_pairs})
        assert not out_of_range, (
            f"{label} bank has pair_index {out_of_range} against {n_pairs} pairs"
        )


def test_one_grid_point_is_one_simulation(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``simulation_id`` covers ``0..n_points-1`` and every trace shares a seed.

    The shared seed is what identifies a sweep in the written bank — the schema
    carries ``seed`` per simulation and does not require it to be unique, so it
    is the field that can say "these N simulations came from one solve".
    """
    _classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)
    theta = read_synthetic_bank_hdf5(theta_path)

    assert theta.simulations.simulation_id == list(range(GRID_N_POINTS))
    assert len(set(theta.simulations.seed)) == 1, "the grid points came from different solves"
    # Every simulation contributes exactly its pairs.
    per_simulation = [int(v) for v in _column(theta, "simulation_id")]
    assert [per_simulation.count(i) for i in range(GRID_N_POINTS)] == [N_PAIRS] * GRID_N_POINTS


def test_a_sweep_holds_the_substrate_constant(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All N simulations describe one solve, so their configs are identical.

    Without this the "only the offset varies" claim is unchecked, and a future
    refactor that solved per grid point would look the same from the outside
    while quietly making the x-axis confounded.
    """
    _classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)
    simulations = read_synthetic_bank_hdf5(theta_path).simulations

    for field_name in (
        "substrate",
        "substrate_summary",
        "activation",
        "geometry",
        "electrodes",
        "cell_model",
    ):
        column = getattr(simulations, field_name)
        assert all(entry == column[0] for entry in column), (
            f"{field_name} differs across grid points; the sweep is not one solve"
        )


# ---------------------------------------------------------------------------
# The grid of record
# ---------------------------------------------------------------------------


def test_activation_position_column_is_the_snapped_grid(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One value per logical simulation, equal to the snapped grid in float32.

    Exact equality, in the dtype the bank stores. Both sides are ``k/191``
    computed by the same division — the probe requests ``k/(T-1)`` and
    ``window_train`` reports ``(t_a - s)/(T-1)`` — so there is nothing for a
    tolerance to absorb, and reaching for one would hide a real offset error
    behind a plausible-looking epsilon.
    """
    _classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)
    theta = read_synthetic_bank_hdf5(theta_path)

    grid = _reference_grid()
    # The column is per trace, so each simulation's value appears once per pair.
    expected = np.repeat(np.asarray(grid.positions, dtype=np.float32), N_PAIRS)
    written = np.asarray(_column(theta, "activation_position"), dtype=np.float32)

    assert written.tobytes() == expected.tobytes()


def test_the_grid_is_snapped_to_the_sample_lattice() -> None:
    """``k = round(p(T-1))``, and ``k/(T-1)`` is the grid of record.

    ``T - 1 = 191`` is prime, so a grid stated in round fractions lands on none
    of the representable positions. The snap is what makes "the probe hits every
    point exactly" true rather than aspirational.
    """
    grid = ProbeGrid.snapped(low=0.1, high=0.9, n_points=9, window_length_samples=T_SAMPLES)

    assert grid.offsets_samples == (19, 38, 57, 76, 96, 115, 134, 153, 172)
    assert all(
        position == offset / LATTICE
        for position, offset in zip(grid.positions, grid.offsets_samples, strict=True)
    )
    # Sub-sample by construction: below what the axis can represent at all.
    assert grid.max_snap_error <= 0.5 / LATTICE


def test_colliding_grid_points_error_naming_the_pair() -> None:
    """Two fractions snapping to one offset is a mis-specified study.

    Silently de-duplicating would emit one crop twice under two different offset
    labels — the same waveform at two x-positions, which is a defect the plot
    cannot show.
    """
    with pytest.raises(ValueError, match="both snap to sample offset") as excinfo:
        # 0.500 and 0.502 are 0.38 samples apart at T=192.
        ProbeGrid.snapped(low=0.5, high=0.502, n_points=2, window_length_samples=T_SAMPLES)

    message = str(excinfo.value)
    assert "0.500000" in message and "0.502000" in message, "the colliding pair is not named"
    assert "n_points" in message, "the error does not say what to change"


def test_a_grid_too_fine_for_the_lattice_is_rejected(tmp_path: Path) -> None:
    """The same collision, surfaced at config load rather than mid-run."""
    doc = _config_doc(out_dir=tmp_path, grid={"low": 0.2, "high": 0.8, "n_points": 500})

    with pytest.raises(ConfigError, match=r"activation_position\.grid"):
        _typed_config(doc, tmp_path)


# ---------------------------------------------------------------------------
# Same waveform, different offset
# ---------------------------------------------------------------------------


def test_two_grid_points_are_one_waveform_at_the_expected_lag(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep shifts a window; it does not re-cut a different signal.

    Read from two genuinely different grid points **of the same pair**, and the
    cross-correlation peak asserted to be exactly the difference in their sample
    offsets — and non-zero, because a trace compared against itself peaks at lag
    0 no matter what the sweep did.
    """
    _classifier_path, theta_path = _run_probe(tmp_path, mock_backend, monkeypatch)
    signals = _theta_signals(read_synthetic_bank_hdf5(theta_path))
    grid = _reference_grid()

    # Traces are simulation-major: simulation i's pairs, then simulation i+1's.
    pair = 0
    first, last = 0, grid.n_points - 1
    expected_lag = grid.offsets_samples[last] - grid.offsets_samples[first]
    assert expected_lag != 0, "the two points chosen are the same offset; nothing is being tested"

    a = signals[first * N_PAIRS + pair]
    b = signals[last * N_PAIRS + pair]
    a = a - a.mean()
    b = b - b.mean()
    correlation = np.correlate(b, a, mode="full")
    lags = np.arange(-(a.size - 1), a.size)
    peak_lag = int(lags[int(np.argmax(correlation))])

    # A window placed at a LARGER offset starts EARLIER in the capture, so the
    # later grid point's content is delayed by exactly the offset difference.
    assert peak_lag == expected_lag, (
        f"cross-correlation peaks at lag {peak_lag}, expected {expected_lag} "
        f"(offsets {grid.offsets_samples[first]} -> {grid.offsets_samples[last]})"
    )


# ---------------------------------------------------------------------------
# Detection: once per pair, on the run's curve
# ---------------------------------------------------------------------------


def test_the_probe_inherits_the_runs_curve(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep detects with the configured curve, not with the default.

    Asserted on the emitted signals: the fixture capture is deliberately
    ambiguous, so ``teager_kaiser`` anchors on a different sample from the
    default ``rectified_derivative`` and every window in the sweep moves with
    it. A probe that quietly used the default would characterise a bank it does
    not share a detector with.
    """
    _, default_theta = _run_probe(tmp_path, mock_backend, monkeypatch, stem="default")
    _, teager_theta = _run_probe(
        tmp_path,
        mock_backend,
        monkeypatch,
        detection={"curve": "teager_kaiser"},
        stem="teager",
    )

    default_signals = _theta_signals(read_synthetic_bank_hdf5(default_theta))
    configured = _theta_signals(read_synthetic_bank_hdf5(teager_theta))

    assert configured.shape == default_signals.shape
    assert not np.array_equal(configured, default_signals), (
        "the configured curve produced the same windows as the default — the "
        "probe did not use the run's preprocessor"
    )


def test_detection_runs_once_per_pair_not_once_per_grid_point() -> None:
    """One detection per pair, then exact shifts.

    Counted rather than inferred. Re-detecting per grid point would put the
    detector's jitter on the axis the study reads off, and the cost would scale
    with ``n_points`` for no benefit.
    """
    calls: list[int] = []

    class _CountingCurve(RectifiedDerivative):
        def _compute(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            calls.append(signal.size)
            computed: npt.NDArray[np.float64] = super()._compute(signal)
            return computed

    n_pairs, n_points = 3, 7
    capture = np.tile(ambiguous_complex(600), (n_pairs, 1)).astype(np.float32)
    grid = ProbeGrid.snapped(low=0.3, high=0.6, n_points=n_points, window_length_samples=T_SAMPLES)

    sweep_capture(traces=capture, grid=grid, preprocessor=_CountingCurve())

    assert len(calls) == n_pairs, (
        f"the detector ran {len(calls)} times for {n_pairs} pairs x {n_points} grid "
        "points; a probe detects once per pair and shifts"
    )


def test_the_sweep_uses_the_preprocessor_it_is_given() -> None:
    """The seam itself, at the function boundary.

    Two curves over one capture put the windows in different places, so the
    preprocessor argument cannot be quietly dropped the way ``crop_traces``'s
    was before the curve became configurable.
    """
    capture = np.tile(ambiguous_complex(600), (2, 1)).astype(np.float32)
    grid = ProbeGrid.snapped(low=0.3, high=0.6, n_points=4, window_length_samples=T_SAMPLES)

    rectified = sweep_capture(traces=capture, grid=grid, preprocessor=RectifiedDerivative())
    teager = sweep_capture(traces=capture, grid=grid, preprocessor=TeagerKaiser())

    assert not np.array_equal(rectified.signals, teager.signals)
    # The grid is the grid whichever curve found the activation — only the
    # window contents move, never the reported offsets.
    assert np.array_equal(rectified.positions, teager.positions)


def test_the_sweep_returns_one_entry_per_grid_point() -> None:
    """Shape check at the boundary: ``(n_points, n_pairs, T)``, nothing flattened."""
    n_pairs, n_points = 4, 6
    capture = np.tile(ambiguous_complex(600), (n_pairs, 1)).astype(np.float32)
    grid = ProbeGrid.snapped(low=0.3, high=0.6, n_points=n_points, window_length_samples=T_SAMPLES)

    swept = sweep_capture(traces=capture, grid=grid)

    assert swept.signals.shape == (n_points, n_pairs, T_SAMPLES)
    assert swept.positions.shape == (n_points,)
    assert swept.activation_index_per_pair.shape == (n_pairs,)


# ---------------------------------------------------------------------------
# Sizing failures
# ---------------------------------------------------------------------------


def test_a_grid_point_outside_the_sizing_errors_with_the_front_diagnostic() -> None:
    """No clipping: a window that does not fit is a mis-sized simulation.

    Clipping would silently shorten one point of the study's x-axis, and the
    resulting curve would have a kink nothing in the bank explains. The message
    is the crop's own FRONT/BACK diagnostic, with the grid point named as well
    as the pair.
    """
    # Activation early in the capture, so a large offset has no room in front.
    capture = np.zeros((1, 400), dtype=np.float32)
    capture[0, 40] = 1.0
    capture[0, 41] = -1.0
    grid = ProbeGrid.snapped(low=0.9, high=0.95, n_points=2, window_length_samples=T_SAMPLES)

    with pytest.raises(ValueError, match="runs off the FRONT") as excinfo:
        sweep_capture(traces=capture, grid=grid)

    message = str(excinfo.value)
    assert "probe grid point" in message, "the failing grid point is not named"
    assert "stimulus_delay_ms" in message, "the message does not name the knob that buys room"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_grid_and_a_position_range_together_are_rejected(tmp_path: Path) -> None:
    """Two sources for one number — the trap the model card already refuses."""
    doc = _config_doc(out_dir=tmp_path)
    doc["activation_position"]["low"] = 0.4
    doc["activation_position"]["high"] = 0.6

    with pytest.raises(ConfigError, match="both 'grid' and 'low'/'high'"):
        _typed_config(doc, tmp_path)


def test_a_probe_with_several_solves_is_rejected(tmp_path: Path) -> None:
    """``n_simulations`` counts solves, and a probe answers about one substrate."""
    doc = _config_doc(out_dir=tmp_path, n_simulations=4)

    with pytest.raises(ConfigError, match="exactly one simulation"):
        _typed_config(doc, tmp_path)


@pytest.mark.parametrize("missing", ["low", "high", "n_points"])
def test_the_grid_block_needs_its_three_keys(tmp_path: Path, missing: str) -> None:
    grid = dict(GRID)
    del grid[missing]
    doc = _config_doc(out_dir=tmp_path, grid=grid)

    with pytest.raises(ConfigError, match=missing):
        _typed_config(doc, tmp_path)


def test_pair_indices_is_rejected_by_name(tmp_path: Path) -> None:
    """The removed subset knob errors rather than being ignored.

    It existed in the first implementation of the probe and is exactly what
    forced the per-trace pair mapping that made a bank unloadable. Silently
    dropping the key would leave a config
    reading as though it swept three pairs while the bank held twenty.
    """
    doc = _config_doc(out_dir=tmp_path, grid={**GRID, "pair_indices": [0, 2]})

    with pytest.raises(ConfigError, match="pair_indices is not supported"):
        _typed_config(doc, tmp_path)


def test_the_capture_is_sized_from_the_snapped_bounds(tmp_path: Path) -> None:
    """Sizing reads the grid the sweep will actually cut at.

    Sizing for the requested fractions would size for offsets the probe never
    uses — a half-sample error today, but the kind that stops being harmless the
    moment a grid is stated in coarser units.
    """
    doc = _config_doc(out_dir=tmp_path, grid={"low": 0.2, "high": 0.8, "n_points": 13})
    cfg = _typed_config(doc, tmp_path)

    assert cfg.probe_grid is not None
    assert cfg.position_generator is None, "a sweep requests offsets, it does not draw them"
    # D = k(p_hi) in ms, rounded up.
    assert cfg.stimulus_delay_ms == float(np.ceil(cfg.probe_grid.offsets_samples[-1]))
    # N = D + V + T - k(p_lo).
    assert cfg.run_config.capture_duration_ms == (
        cfg.stimulus_delay_ms + 2 * T_SAMPLES + T_SAMPLES - cfg.probe_grid.offsets_samples[0]
    )


def test_the_run_summary_reports_the_snap_and_the_simulation_count(
    tmp_path: Path,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The snap is reported once, and the summary says how many simulations landed.

    Both facts surprise a reader otherwise: the offsets are not the fractions
    the config named, and the bank holds ``n_points`` simulations where the
    config said one. Per trace it would be noise; once per run it is provenance.
    """
    _run_probe(tmp_path, mock_backend, monkeypatch, stem="summary")

    printed = capsys.readouterr().out
    grid = _reference_grid()
    assert "Probe grid:" in printed
    assert f"{grid.n_points} offsets" in printed
    assert "snapped by at most" in printed
    assert f"Probe simulations:      {grid.n_points}" in printed
    assert f"N simulations:          {grid.n_points}" in printed


def test_the_example_probe_config_runs_end_to_end(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped example is a config that works, not an illustration.

    Only the output path is redirected — every substantive value is the
    example's own, so a knob that stops loading is caught here rather than by
    whoever next runs the probe. The bank it writes is checked for the one
    property the study needs: a unique, agreeing join key.
    """
    doc = yaml.safe_load(Path("examples/synthegm_probe.yaml").read_text(encoding="utf-8"))
    doc["output"]["classifier_bank"] = str(tmp_path / "example.classifier.h5")
    config_path = tmp_path / "example.yaml"
    config_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setattr(
        generate_dataset_cmd,
        "FinitewaveBackend",
        lambda: ambiguous_complex_backend_for(mock_backend),
    )

    assert generate_dataset_cmd.main([str(config_path)]) == 0

    theta = read_synthetic_bank_hdf5(tmp_path / "example.synthetic.h5")
    classifier = load_classifier_bank(tmp_path / "example.classifier.h5")
    keys = _theta_keys(theta)

    n_points = doc["activation_position"]["grid"]["n_points"]
    n_pairs = doc["electrodes"]["n_rows"] * (doc["electrodes"]["n_cols"] - 1)
    assert len(keys) == n_points * n_pairs
    assert len(set(keys)) == len(keys)
    assert _classifier_keys(classifier) == keys
