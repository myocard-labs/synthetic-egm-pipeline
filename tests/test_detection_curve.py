"""Tests for the configurable detection curve (S16a).

``crop_traces`` has always taken a ``preprocessor`` and ``run_single`` never
passed one, so every synthetic bank in the project's history was windowed with
``RectifiedDerivative`` and no config could say otherwise. These tests are about
the seam that closes: ``activation_position.detection`` ->
``GenerateDatasetCLIConfig`` -> ``DatasetConfig`` -> ``run_single`` ->
``crop_traces``.

The block nests **inside** ``activation_position`` because detection is part of
the crop — ``crop_traces`` builds one windower out of the preprocessor and the
position generator — and because that makes a curve configured for a run that
never crops impossible to write down.

**Asserted on the emitted signal, not on the config object.** The natural test
here — ``cfg.detection_preprocessor.name == "botteron_envelope"`` — passes with
the runner still ignoring the value, which is the exact bug being fixed. So the
load-bearing tests run the CLI end to end and diff one bank against another.
The two config-object assertions that remain (Botteron's ``fs``, band and
low-pass) are paired with an end-to-end diff that only passes if those numbers
reached the crop.

The backend is not under test: it is the mock, wrapped so that its capture
carries a **deliberately ambiguous** complex — a sharp one-sample deflection and
a broad 100 Hz burst, which the three curves resolve to three different indices.
A clean simulated activation is *not* ambiguous (research measured dV/dt-max and
the Botteron envelope landing on the same sample), so a fixture built from one
would make every diff below vacuous.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from myocard_egm_data.banks import load_classifier_bank, read_synthetic_bank_hdf5
from myocard_egm_signal import (
    BotteronEnvelope,
    RectifiedDerivative,
    TeagerKaiser,
    detect_activation,
)

from conftest import ambiguous_complex, ambiguous_complex_backend_for
from myocard_synthetic_egm_pipeline.cli import generate_dataset_cmd
from myocard_synthetic_egm_pipeline.cli._config import (
    ConfigError,
    build_generate_dataset_config,
)
from myocard_synthetic_egm_pipeline.simulate import generate_dataset
from myocard_synthetic_egm_pipeline.simulate.cropping import (
    DETECTION_CURVES,
    build_preprocessor,
)

#: Output rate every config in this module runs at. The fixture waveform's
#: 100 Hz burst is written in ms, so it is only the shape described below at
#: this rate.
FS_HZ = 1000.0


# ---------------------------------------------------------------------------
# Config + run helpers
# ---------------------------------------------------------------------------


def _config_doc(
    *,
    out_dir: Path,
    detection: dict[str, Any] | None = None,
    stem: str = "run",
) -> dict[str, Any]:
    """A complete generate-dataset config with cropping switched on.

    ``activation_position`` is not optional here, and the detection block nests
    inside it: with no position policy there is no crop, so nothing detects and
    there is no curve to configure.
    """
    activation_position: dict[str, Any] = {"low": 0.4, "high": 0.6}
    if detection is not None:
        activation_position["detection"] = detection
    return {
        "dataset": {"n_simulations": 2, "master_seed": 7, "show_progress": False},
        "backend": {"type": "finitewave"},
        "geometry": {"type": "patch_2d", "size_mm": 8.0, "dr_mm": 0.25},
        "substrate": {"type": "uniform_random_fibrosis", "density_range": [0.0, 0.4]},
        "activation": {"type": "planar_edge", "fixed_edge": "top"},
        "electrodes": {
            "type": "centered_grid_2d",
            "n_rows": 2,
            "n_cols": 2,
            "spacing_mm": 2.0,
            "height_mm_range": [0.2, 1.0],
        },
        "label_policy": {"type": "global_density", "threshold": 0.1},
        "run": {
            "trace_duration_ms": 192.0,
            "output_fs_hz": FS_HZ,
            "capture_oversample": 4,
        },
        "activation_position": activation_position,
        "output": {
            "classifier_bank": str(out_dir / f"{stem}.classifier.h5"),
            "description": "S16a detection-curve test",
        },
    }


def _typed_config(doc: dict[str, Any], tmp_path: Path) -> Any:
    """Build the typed CLI config from an in-memory doc."""
    return build_generate_dataset_config({**doc, "_config_dir": tmp_path})


def _run_cli(
    tmp_path: Path,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    detection: dict[str, Any] | None = None,
    stem: str = "run",
) -> Path:
    """Write the config, run ``main()``, return the ClassifierBank path."""
    doc = _config_doc(out_dir=tmp_path, detection=detection, stem=stem)
    config_path = tmp_path / f"{stem}.yaml"
    config_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setattr(
        generate_dataset_cmd,
        "FinitewaveBackend",
        lambda: ambiguous_complex_backend_for(mock_backend),
    )
    assert generate_dataset_cmd.main([str(config_path)]) == 0
    return tmp_path / f"{stem}.classifier.h5"


def _bank_signals(path: Path) -> npt.NDArray[np.float32]:
    """Every trace in a written ClassifierBank, stacked in bank order."""
    bank = load_classifier_bank(path)
    return np.asarray([np.asarray(t.signal, dtype=np.float32) for t in bank.traces])


def _bank_positions(path: Path) -> npt.NDArray[np.float32]:
    """The written synthetic_bank's ``activation_position`` column, in float32."""
    theta = read_synthetic_bank_hdf5(path)
    assert theta.traces.activation_position is not None
    return np.asarray([p.root for p in theta.traces.activation_position], dtype=np.float32)


def _capture_samples(cfg: Any) -> int:
    """Length in samples of the capture the runner will crop from."""
    run = cfg.run_config
    samples: int = round(run.effective_capture_duration_ms * 1e-3 * run.output_fs_hz)
    return samples


# ---------------------------------------------------------------------------
# The fixture's own precondition
# ---------------------------------------------------------------------------


def test_the_three_curves_disagree_on_the_fixture_capture(tmp_path: Path) -> None:
    """Guard the diffs below: they mean nothing if the curves agree here.

    Every "changing the curve changes the bank" test rests on the fixture
    waveform being genuinely ambiguous. If a future egm-signal release moved
    one curve onto another's peak, those tests would fail with a bank diff and
    no hint why; this one fails saying exactly what stopped being true.
    """
    cfg = _typed_config(_config_doc(out_dir=tmp_path), tmp_path)
    capture = ambiguous_complex(_capture_samples(cfg))

    detected = {
        curve: detect_activation(capture, preprocessor=build_preprocessor(curve=curve, fs_hz=FS_HZ))
        for curve in DETECTION_CURVES
    }

    assert len(set(detected.values())) == len(DETECTION_CURVES), (
        f"the fixture capture no longer separates the curves: {detected}"
    )


# ---------------------------------------------------------------------------
# The emitted signal — the assertions that fail if the runner ignores the curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("curve", DETECTION_CURVES)
def test_each_curve_writes_a_different_bank(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch, curve: str
) -> None:
    """Setting ``detection.curve`` changes **the stored waveform**.

    The whole point of S16a. Two runs identical in every other respect — same
    master seed, same substrate draws, same position stream — differ only in the
    curve, so any difference in the banks is the curve reaching the crop. With
    the runner still dropping the preprocessor on the floor (the defect), both
    banks are byte-identical and this fails.
    """
    baseline = _bank_signals(_run_cli(tmp_path, mock_backend, monkeypatch, stem="baseline"))
    configured = _bank_signals(
        _run_cli(tmp_path, mock_backend, monkeypatch, detection={"curve": curve}, stem=curve)
    )

    assert configured.shape == baseline.shape
    if curve == "rectified_derivative":
        # The default spelled out explicitly must reproduce the default.
        assert configured.tobytes() == baseline.tobytes()
    else:
        assert configured.tobytes() != baseline.tobytes(), (
            f"curve={curve!r} produced the same waveforms as the default — the "
            "preprocessor did not reach crop_traces"
        )


def test_the_three_curves_write_three_different_banks(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three dispatch, and each cuts its window somewhere else.

    Pairwise-distinct rather than "differs from the default", so a mapping that
    collapsed two names onto one preprocessor cannot pass.
    """
    written = {
        curve: _bank_signals(
            _run_cli(tmp_path, mock_backend, monkeypatch, detection={"curve": curve}, stem=curve)
        ).tobytes()
        for curve in DETECTION_CURVES
    }

    assert len(set(written.values())) == len(DETECTION_CURVES), (
        "two curve names produced identical banks"
    )


def test_botteron_band_changes_the_stored_waveform(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Botteron parameters reach the crop, not just the constructor.

    ``[200, 450]`` Hz keeps the sharp one-sample spike and rejects the 100 Hz
    burst, so the envelope's peak moves from the burst to the spike. Asserting
    the band on the constructed object would pass with the object discarded;
    this cannot.
    """
    default_band = _bank_signals(
        _run_cli(
            tmp_path,
            mock_backend,
            monkeypatch,
            detection={"curve": "botteron_envelope"},
            stem="band_default",
        )
    )
    high_band = _bank_signals(
        _run_cli(
            tmp_path,
            mock_backend,
            monkeypatch,
            detection={"curve": "botteron_envelope", "botteron_band_hz": [200.0, 450.0]},
            stem="band_high",
        )
    )

    assert high_band.tobytes() != default_band.tobytes(), (
        "the configured band did not change where the window was cut"
    )


def test_omitting_the_block_reproduces_the_pre_s16a_bank(
    tmp_path: Path, mock_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``detection`` block -> byte-for-byte what the producer wrote before.

    Two references, because one alone leaves a hole:

    - ``detection_preprocessor=None`` is literally the pre-S16a call — the
      ``crop_traces`` invocation with no preprocessor argument at all. It
      catches an absent block resolving to some other curve.
    - an explicit ``RectifiedDerivative()`` pins *which* curve that is. Without
      it, redefining ``default_preprocessor`` would move both the run and its
      reference together and this test would pass through the change.
    """
    written = _bank_signals(_run_cli(tmp_path, mock_backend, monkeypatch, stem="default"))

    references = {
        # The call the runner made before the parameter was threaded through.
        "pre-S16a (no preprocessor)": None,
        # ... and what that default has to be.
        "explicit RectifiedDerivative": RectifiedDerivative(),
    }
    for label, preprocessor in references.items():
        # The config is rebuilt per reference to get a fresh, identically
        # seeded position stream — the generator is stateful, so a reused one
        # would be mid-sequence and every window would land elsewhere.
        cfg = _typed_config(_config_doc(out_dir=tmp_path, stem="default"), tmp_path)
        reference = generate_dataset(
            config=replace(
                generate_dataset_cmd._build_dataset_config(cfg, show_progress=False),
                detection_preprocessor=preprocessor,
            ),
            backend=ambiguous_complex_backend_for(mock_backend),
            position_generator=cfg.position_generator,
        )
        signals = np.concatenate([r.bipolar_traces for r in reference.results])
        assert written.tobytes() == signals.tobytes(), f"differs from {label}"

        # The other thing the crop emits: where it says the activation landed.
        positions = np.concatenate(
            [
                r.activation_positions
                for r in reference.results
                if r.activation_positions is not None
            ]
        )
        assert (
            _bank_positions(tmp_path / "default.synthetic.h5").tobytes()
            == positions.astype(np.float32).tobytes()
        ), f"activation_position differs from {label}"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_no_position_block_means_no_preprocessor(tmp_path: Path) -> None:
    """No crop, no detection — not a default curve nobody uses.

    The curve is only meaningful as part of a crop, so with no position policy
    the config resolves to ``None`` rather than to a preprocessor that would be
    built, carried through two config objects and then ignored by the runner.
    """
    doc = _config_doc(out_dir=tmp_path)
    del doc["activation_position"]

    cfg = _typed_config(doc, tmp_path)

    assert cfg.position_generator is None
    assert cfg.detection_preprocessor is None


def test_a_curve_cannot_be_configured_for_a_run_that_never_crops(tmp_path: Path) -> None:
    """The nesting makes the mismatch unwritable, not merely detected.

    A ``detection`` block can only be reached through ``activation_position``,
    and that block is invalid without its bounds — so "a curve for a run with no
    crop" fails on the bounds it is missing. Under the first spelling the same
    config parsed happily and the curve was silently dropped.
    """
    doc = _config_doc(out_dir=tmp_path)
    doc["activation_position"] = {"detection": {"curve": "botteron_envelope"}}

    with pytest.raises(ConfigError, match="both 'low' and 'high'"):
        _typed_config(doc, tmp_path)


@pytest.mark.parametrize("with_position_block", [True, False])
def test_the_old_activation_detection_path_is_rejected(
    tmp_path: Path, with_position_block: bool
) -> None:
    """``activation.detection`` errors, naming where the block moved to.

    Silently ignoring it is the failure mode that matters here: the old path
    falls back to ``rectified_derivative``, so a config asking for Botteron would
    keep running and keep producing default-windowed banks. Rejected in both
    shapes — with a position block, where the run would have cropped on the
    wrong curve, and without one, where nothing would have cropped at all.
    """
    doc = _config_doc(out_dir=tmp_path)
    doc["activation"]["detection"] = {"curve": "botteron_envelope"}
    if not with_position_block:
        del doc["activation_position"]

    with pytest.raises(ConfigError, match=r"activation_position\.detection") as excinfo:
        _typed_config(doc, tmp_path)

    assert "activation.detection has moved" in str(excinfo.value)


def test_unknown_curve_is_rejected_naming_the_valid_ones(tmp_path: Path) -> None:
    """A typo has to say what to write instead, or it just relocates the guess."""
    doc = _config_doc(out_dir=tmp_path, detection={"curve": "hilbert_envelope"})

    with pytest.raises(ConfigError, match=r"activation_position\.detection\.curve") as excinfo:
        _typed_config(doc, tmp_path)

    message = str(excinfo.value)
    assert "hilbert_envelope" in message
    for valid in DETECTION_CURVES:
        assert valid in message, f"the error does not name {valid!r} as an option"


@pytest.mark.parametrize(
    "dead_key",
    [
        {"threshold": {"rule": "median_mad", "c": 1.0}},
        {"threshold_rule": "percentile"},
        {"threshold_q": 99.0},
        {"min_prominence": 0.2},
        {"refractory_ms": 50.0},
        {"refine": {"curve": "rectified_derivative", "radius_ms": 10.0}},
        {"refine_curve": "rectified_derivative"},
        {"refine_radius_ms": 10.0},
    ],
)
def test_multi_activation_keys_are_refused_by_name(
    tmp_path: Path, dead_key: dict[str, Any]
) -> None:
    """iafdb's threshold / prominence / refractory / refine knobs do nothing here.

    Synthetic detection is ``argmax g`` over a trace with exactly one activation
    by construction, so accepting these would mint fresh config surface that
    provably has no effect — the same unreachable-seam defect S16a removes.
    Both the nested (``threshold:``) and flat (``threshold_rule:``) spellings are
    refused, so a block copied from an iafdb config fails whichever way it was
    written.
    """
    doc = _config_doc(out_dir=tmp_path, detection={"curve": "botteron_envelope", **dead_key})

    with pytest.raises(ConfigError, match="multi-activation detection") as excinfo:
        _typed_config(doc, tmp_path)

    assert next(iter(dead_key)) in str(excinfo.value)


@pytest.mark.parametrize("curve", ["rectified_derivative", "teager_kaiser"])
def test_botteron_parameters_beside_another_curve_are_refused(tmp_path: Path, curve: str) -> None:
    """Silently ignoring a deliberate setting is how a config comes to lie."""
    doc = _config_doc(
        out_dir=tmp_path, detection={"curve": curve, "botteron_band_hz": [40.0, 250.0]}
    )

    with pytest.raises(ConfigError, match="botteron_band_hz"):
        _typed_config(doc, tmp_path)


def test_a_malformed_botteron_band_fails_at_config_load(tmp_path: Path) -> None:
    """egm-signal's own validation, surfaced before the N-sim run rather than in it."""
    doc = _config_doc(
        out_dir=tmp_path,
        detection={"curve": "botteron_envelope", "botteron_band_hz": [250.0, 40.0]},
    )

    with pytest.raises(ConfigError, match=r"activation_position\.detection"):
        _typed_config(doc, tmp_path)


def test_botteron_gets_the_output_rate_not_the_capture_rate(tmp_path: Path) -> None:
    """``fs`` is the rate the detector sees, and the runner downsamples first.

    Passing ``fs_capture_hz`` would mis-scale the band and the low-pass by the
    oversample factor and still run without complaint — a wrong number, not an
    error, which is why it is worth pinning. The config below separates the two
    rates by 8x so the assertion can tell them apart.
    """
    doc = _config_doc(out_dir=tmp_path, detection={"curve": "botteron_envelope"})
    doc["run"] = {
        "trace_duration_ms": 256.0,
        "output_fs_hz": 500.0,
        "capture_oversample": 8,
    }

    preprocessor = _typed_config(doc, tmp_path).detection_preprocessor

    assert isinstance(preprocessor, BotteronEnvelope)
    # 500 Hz is the output rate; the backend captures at 8x that (4000 Hz).
    assert preprocessor.fs == 500.0


def test_botteron_parameters_land_on_the_preprocessor(tmp_path: Path) -> None:
    """The two Botteron knobs carry through with the values the config gave."""
    doc = _config_doc(
        out_dir=tmp_path,
        detection={
            "curve": "botteron_envelope",
            "botteron_band_hz": [30.0, 200.0],
            "botteron_lowpass_hz": 15.0,
        },
    )

    preprocessor = _typed_config(doc, tmp_path).detection_preprocessor

    assert isinstance(preprocessor, BotteronEnvelope)
    assert preprocessor.band_hz == (30.0, 200.0)
    assert preprocessor.lowpass_hz == 15.0


def test_absent_block_resolves_to_the_documented_default(tmp_path: Path) -> None:
    """The default is ``rectified_derivative``, resolved rather than left None."""
    cfg = _typed_config(_config_doc(out_dir=tmp_path), tmp_path)

    assert isinstance(cfg.detection_preprocessor, RectifiedDerivative)
    assert cfg.detection_preprocessor.name == "rectified_derivative"


# ---------------------------------------------------------------------------
# build_preprocessor dispatch
# ---------------------------------------------------------------------------


def test_build_preprocessor_dispatches_all_three_names() -> None:
    """The three names map onto the three egm-signal curves."""
    assert isinstance(
        build_preprocessor(curve="rectified_derivative", fs_hz=FS_HZ), RectifiedDerivative
    )
    assert isinstance(build_preprocessor(curve="teager_kaiser", fs_hz=FS_HZ), TeagerKaiser)
    assert isinstance(build_preprocessor(curve="botteron_envelope", fs_hz=FS_HZ), BotteronEnvelope)


def test_build_preprocessor_rejects_an_unknown_curve() -> None:
    """Direct callers get the same list of options the config layer gives."""
    with pytest.raises(ValueError, match="Unknown detection curve") as excinfo:
        build_preprocessor(curve="wavelet", fs_hz=FS_HZ)

    for valid in DETECTION_CURVES:
        assert valid in str(excinfo.value)


# ---------------------------------------------------------------------------
# The run summary — the only record of the curve until FB-35
# ---------------------------------------------------------------------------


def test_run_summary_reports_the_resolved_curve(
    tmp_path: Path,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bank cannot carry the curve (FB-35), so the run has to say it.

    This is the string that goes into ``output.description`` by hand, which is
    the record until the schema grows a field for it.
    """
    _run_cli(
        tmp_path,
        mock_backend,
        monkeypatch,
        detection={"curve": "botteron_envelope"},
        stem="summary",
    )

    printed = capsys.readouterr().out
    assert "Detection curve:" in printed
    assert "botteron_envelope" in printed


def test_run_summary_omits_the_curve_when_nothing_was_cropped(tmp_path: Path) -> None:
    """With no ``activation_position`` block there is no curve to report.

    Naming one would describe a window placement that never happened — the
    confident-but-wrong provenance this line exists to prevent.
    """
    doc = _config_doc(out_dir=tmp_path)
    del doc["activation_position"]
    cfg = _typed_config(doc, tmp_path)

    summary = generate_dataset_cmd._format_result(
        cfg=cfg,
        dataset_result=_empty_dataset_result(),
        classifier_path=tmp_path / "x.classifier.h5",
        clean_intermediate_path=None,
        synthetic_path=None,
        mixed=False,
    )

    assert "Detection curve" not in summary
    for curve in DETECTION_CURVES:
        assert curve not in summary


def _empty_dataset_result() -> Any:
    """A DatasetResult with no simulations — enough for ``_format_result``."""
    from myocard_synthetic_egm_pipeline.simulate import DatasetResult

    empty = np.zeros(0, dtype=np.int64)
    return DatasetResult(
        results=[],
        labels=empty,
        labels_dict={},
        simulation_ids=empty,
        pair_indices=empty,
        seeds=empty,
    )
