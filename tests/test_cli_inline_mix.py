"""CLI-level tests for the inline-mixer path (``synthegm-generate-dataset``).

These drive :func:`generate_dataset_cmd.main` end to end — config file in,
banks on disk out — with the Finitewave backend swapped for the test
``MockBackend``. The backend is not what is under test; the **CLI wiring** is.

That distinction is the whole reason this module exists. The defect it was
written for was one where the clean intermediate ClassifierBank named a θ
companion that did not match the θ file written beside it, and *every
builder-level test passed while it was present*, because each builder was
individually correct. Only the
composition was wrong, and only a run through ``main()`` composes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_data.banks import (
    join_traces_with_simulations,
    load_classifier_bank,
    read_synthetic_bank_hdf5,
    write_noise_bank,
)

from myocard_synthetic_egm_pipeline.cli import generate_dataset_cmd


def _config_doc(*, out_dir: Path, noise_bank: Path, clean_intermediate: bool) -> dict[str, Any]:
    """A minimal but complete generate-dataset config with inline mixing."""
    doc: dict[str, Any] = {
        "dataset": {"n_simulations": 2, "master_seed": 7, "show_progress": False},
        "backend": {"type": "finitewave"},
        "geometry": {
            "type": "patch_2d",
            "size_mm": 8.0,
            "dr_mm": 0.25,
            "fiber_angle_rad": 0.0,
            "anisotropy_ratio": 3.0,
        },
        "substrate": {
            "type": "uniform_random_fibrosis",
            "density_range": [0.0, 0.5],
            "fraction_healthy": 0.5,
        },
        "activation": {"type": "planar_edge", "fixed_edge": None},
        "electrodes": {
            "type": "centered_grid_2d",
            "n_rows": 2,
            "n_cols": 2,
            "spacing_mm": 2.0,
            "height_mm_range": [0.2, 1.0],
        },
        "label_policy": {
            "type": "global_density",
            "threshold": 0.1,
            "healthy_name": "healthy",
            "fibrotic_name": "fibrotic",
        },
        "run": {
            "trace_duration_ms": 192.0,
            "output_fs_hz": 1000.0,
            "capture_oversample": 4,
        },
        "output": {
            "classifier_bank": str(out_dir / "run.classifier.h5"),
            "description": "inline-mix CLI test",
        },
        "mix": {
            "noise_bank": str(noise_bank),
            "snr_db_range": [10.0, 25.0],
            "bandpass_clean": False,
            "master_seed": 3,
        },
    }
    if clean_intermediate:
        doc["output"]["clean_intermediate"] = str(out_dir / "run.clean.classifier.h5")
    return doc


@pytest.fixture
def inline_mix_run(
    tmp_path: Path,
    small_noise_bank: NoiseBank,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run the CLI once with inline mixing + a clean intermediate."""
    noise_path = write_noise_bank(small_noise_bank, tmp_path / "noise.h5")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _config_doc(out_dir=tmp_path, noise_bank=noise_path, clean_intermediate=True)
        )
    )
    monkeypatch.setattr(generate_dataset_cmd, "FinitewaveBackend", lambda: mock_backend)
    assert generate_dataset_cmd.main([str(config_path)]) == 0
    return tmp_path


def _theta_companion(bank: Any) -> Any:
    """The single θ-companion provenance entry on a ClassifierBank."""
    from myocard_synthetic_egm_pipeline.constants import THETA_BANK_SOURCE

    entries = [e for e in bank.banks if e.bank_type == THETA_BANK_SOURCE]
    assert len(entries) == 1, f"expected exactly one θ companion, got {len(entries)}"
    return entries[0]


def _idstr(value: Any) -> str:
    return str(value.root if hasattr(value, "root") else value)


def test_inline_mix_writes_four_files(inline_mix_run: Path) -> None:
    """A mix run with a clean intermediate produces two complete pairs.

    Every ClassifierBank gets its own θ partner, so the four files are
    mixed bank + mixed θ, clean bank + clean θ. Sharing one θ file would hand
    a consumer mixed waveforms for a trace it joined as clean.
    """
    for name in (
        "run.classifier.h5",
        "run.synthetic.h5",
        "run.clean.classifier.h5",
        "run.clean.synthetic.h5",
    ):
        assert (inline_mix_run / name).is_file(), f"missing {name}"


@pytest.mark.parametrize(
    ("classifier_name", "theta_name"),
    [
        ("run.classifier.h5", "run.synthetic.h5"),
        ("run.clean.classifier.h5", "run.clean.synthetic.h5"),
    ],
)
def test_each_bank_joins_against_its_own_theta(
    inline_mix_run: Path, classifier_name: str, theta_name: str
) -> None:
    """Regression: **both** banks join, each against its own θ file.

    The clean intermediate used to name the *mixed* run's θ bank with an id
    derived from its own — so the id did not match the file, and egm-data
    refused the join. The refusal was correct: ``simulation_id`` restarts at 0
    in every bank, so a permissive join would silently pair traces with another
    run's config.
    """
    classifier = load_classifier_bank(inline_mix_run / classifier_name)
    theta = read_synthetic_bank_hdf5(inline_mix_run / theta_name)

    companion = _theta_companion(classifier)
    assert _idstr(companion.bank_id) == _idstr(theta.bank_id), (
        "the θ companion entry names an id the θ file beside it does not carry"
    )
    assert Path(str(companion.bank_path)).name == theta_name

    joined = join_traces_with_simulations(classifier, theta)
    assert len(joined) == len(classifier.traces)


def test_clean_theta_holds_clean_signals(inline_mix_run: Path) -> None:
    """The clean bank's θ partner must carry the **clean** waveforms.

    This is the assertion that fails under the rejected shared-θ design. A θ
    bank stores ``traces/signal``, not just config — so pointing the clean
    ClassifierBank at the mixed run's θ file would hand a consumer *mixed*
    waveforms for a trace it joined as clean, with nothing flagging the swap.
    """
    clean = load_classifier_bank(inline_mix_run / "run.clean.classifier.h5")
    clean_theta = read_synthetic_bank_hdf5(inline_mix_run / "run.clean.synthetic.h5")
    mixed_theta = read_synthetic_bank_hdf5(inline_mix_run / "run.synthetic.h5")

    # ClassifierBank.traces is a list of trace objects; the synthetic bank
    # stores its traces columnar, so `traces.signal` is already the stack.
    clean_sig = [list(t.signal) for t in clean.traces]
    clean_theta_sig = [list(row) for row in clean_theta.traces.signal]
    mixed_theta_sig = [list(row) for row in mixed_theta.traces.signal]

    assert clean_theta_sig == clean_sig, "clean θ bank does not hold the clean signals"
    assert clean_theta_sig != mixed_theta_sig, (
        "clean and mixed θ banks hold identical signals — the mixer did not run, "
        "or both θ banks were written from the same source"
    )


def test_clean_and_mixed_ids_are_distinct_and_related(inline_mix_run: Path) -> None:
    """Four artifacts, four ids, one family — related by marker, not by luck."""
    ids = {
        name: _idstr(load_classifier_bank(inline_mix_run / name).id)
        for name in ("run.classifier.h5", "run.clean.classifier.h5")
    }
    ids |= {
        name: _idstr(read_synthetic_bank_hdf5(inline_mix_run / name).bank_id)
        for name in ("run.synthetic.h5", "run.clean.synthetic.h5")
    }
    assert len(set(ids.values())) == 4, f"ids collide: {ids}"

    clean_base = ids["run.clean.classifier.h5"]
    assert "noise_mixed" not in clean_base
    assert "noise_mixed" in ids["run.classifier.h5"]
    assert ids["run.clean.synthetic.h5"].startswith(clean_base.rsplit("_", 1)[0])


def test_mix_run_without_clean_intermediate_writes_one_pair(
    tmp_path: Path,
    small_noise_bank: NoiseBank,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No clean intermediate requested → no clean θ bank either.

    The clean θ bank exists to give the clean ClassifierBank a partner. With no
    clean ClassifierBank on disk there is nothing to partner, and writing one
    anyway would leave an orphan θ file that no artifact names.
    """
    noise_path = write_noise_bank(small_noise_bank, tmp_path / "noise.h5")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _config_doc(out_dir=tmp_path, noise_bank=noise_path, clean_intermediate=False)
        )
    )
    monkeypatch.setattr(generate_dataset_cmd, "FinitewaveBackend", lambda: mock_backend)
    assert generate_dataset_cmd.main([str(config_path)]) == 0

    assert (tmp_path / "run.classifier.h5").is_file()
    assert (tmp_path / "run.synthetic.h5").is_file()
    assert not (tmp_path / "run.clean.classifier.h5").exists()
    assert not (tmp_path / "run.clean.synthetic.h5").exists()


def test_clean_intermediate_bank_id_override(
    tmp_path: Path,
    small_noise_bank: NoiseBank,
    mock_backend: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``output.clean_intermediate_bank_id`` names the clean pair.

    One override per pair, matching how ``output.bank_id`` names the mixed
    pair. The clean bank previously had no override at all and fell back to a
    cell-model-derived id, which is how it came to name a θ artifact that did
    not exist.
    """
    noise_path = write_noise_bank(small_noise_bank, tmp_path / "noise.h5")
    doc = _config_doc(out_dir=tmp_path, noise_bank=noise_path, clean_intermediate=True)
    doc["output"]["clean_intermediate_bank_id"] = "tbank_synthetic_clean_probe_2026-08-11"
    doc["output"]["bank_id"] = "tbank_synthetic_mixed_probe_2026-08-11"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(generate_dataset_cmd, "FinitewaveBackend", lambda: mock_backend)
    assert generate_dataset_cmd.main([str(config_path)]) == 0

    clean = load_classifier_bank(tmp_path / "run.clean.classifier.h5")
    mixed = load_classifier_bank(tmp_path / "run.classifier.h5")
    assert _idstr(clean.id) == "tbank_synthetic_clean_probe_2026-08-11"
    assert _idstr(mixed.id) == "tbank_synthetic_mixed_probe_2026-08-11"

    # Each override still carries through to its own theta partner.
    clean_theta = read_synthetic_bank_hdf5(tmp_path / "run.clean.synthetic.h5")
    mixed_theta = read_synthetic_bank_hdf5(tmp_path / "run.synthetic.h5")
    assert _idstr(clean_theta.bank_id) == "tbank_synthetic_clean_probe_theta_2026-08-11"
    assert _idstr(mixed_theta.bank_id) == "tbank_synthetic_mixed_probe_theta_2026-08-11"
    assert _idstr(_theta_companion(clean).bank_id) == _idstr(clean_theta.bank_id)
    assert _idstr(_theta_companion(mixed).bank_id) == _idstr(mixed_theta.bank_id)


def test_clean_bank_id_without_clean_intermediate_is_rejected(tmp_path: Path) -> None:
    """An id for a bank that is not being written is a config mistake.

    Silently ignoring it is the failure mode the retired
    ``also_emit_synthetic_bank`` flag taught us to refuse: the run looks like
    it honoured the override and the artifact never appears.
    """
    from myocard_synthetic_egm_pipeline.cli._config import (
        ConfigError,
        build_generate_dataset_config,
    )

    doc = _config_doc(out_dir=tmp_path, noise_bank=tmp_path / "n.h5", clean_intermediate=False)
    doc["output"]["clean_intermediate_bank_id"] = "tbank_synthetic_orphan_2026-08-11"
    doc["_config_dir"] = tmp_path

    with pytest.raises(ConfigError, match="clean_intermediate_bank_id"):
        build_generate_dataset_config(doc)
