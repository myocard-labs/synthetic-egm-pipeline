"""Tests for the YAML config loader + typed config builders.

Mirrors iafdb-pipeline's test_cli_config.py pattern: a small
``_write_yaml`` helper writes a YAML body to a tmp file, then we call
``load_yaml`` + ``build_generate_dataset_config`` / ``build_mix_config``
to verify each path through the validation logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from myocard_synthetic_egm_pipeline.cli._config import (
    ConfigError,
    build_generate_dataset_config,
    build_mix_config,
    load_yaml,
)
from myocard_synthetic_egm_pipeline.simulate import Patch2DGeometry


def _write_yaml(tmp_path: Path, body: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------


def test_load_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    """A YAML file whose top level is a list isn't a config."""
    path = _write_yaml(tmp_path, "- one\n- two\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(path)


def test_load_yaml_adds_config_dir(tmp_path: Path) -> None:
    """The loader stashes the parent dir for downstream path resolution."""
    path = _write_yaml(tmp_path, "key: value\n")
    doc = load_yaml(path)
    assert doc["_config_dir"] == tmp_path.resolve()


# ---------------------------------------------------------------------------
# build_generate_dataset_config — minimum required fields + defaults
# ---------------------------------------------------------------------------


def test_generate_dataset_minimum_required_fields(tmp_path: Path) -> None:
    """The minimum a config can specify is dataset.n_simulations +
    output.classifier_bank; every other field has a default."""
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        output:
          classifier_bank: ./out.h5
        """,
    )
    cfg = build_generate_dataset_config(load_yaml(path))
    assert cfg.n_simulations == 5
    assert cfg.master_seed == 0
    assert cfg.show_progress is True
    assert cfg.backend_type == "finitewave"
    # Geometry defaults to the project's Phase 1 patch.
    assert cfg.geometry.type == "patch_2d"
    # cfg.geometry is typed as the GeometrySpec Protocol, which exposes only
    # `type` by design (Guardrail 2). Narrow to the concrete before reading a
    # concrete's field, rather than reaching through the Protocol.
    assert isinstance(cfg.geometry, Patch2DGeometry)
    assert cfg.geometry.size_mm == 40.0
    # Output path resolves against the YAML's directory.
    assert cfg.classifier_bank_output == (tmp_path / "out.h5").resolve()
    # No mix block → cfg.mix is None.
    assert cfg.mix is None


def test_generate_dataset_rejects_zero_n_simulations(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 0
        output:
          classifier_bank: ./out.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"n_simulations"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_rejects_missing_required_output(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        """,
    )
    with pytest.raises(ConfigError, match=r"classifier_bank"):
        build_generate_dataset_config(load_yaml(path))


# ---------------------------------------------------------------------------
# Enum validation — fail loudly on unknown discriminator values
# ---------------------------------------------------------------------------


def test_generate_dataset_rejects_unknown_geometry(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        geometry:
          type: cube_3d
        output:
          classifier_bank: ./out.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"geometry.type"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_rejects_unknown_substrate(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        substrate:
          type: vibes_based
        output:
          classifier_bank: ./out.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"substrate.type"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_rejects_unknown_label_policy(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        label_policy:
          type: pickle_of_cats
        output:
          classifier_bank: ./out.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"label_policy.type"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_rejects_unknown_stim_edge(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        activation:
          fixed_edge: diagonal
        output:
          classifier_bank: ./out.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"fixed_edge"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_rejects_unknown_backend(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        backend:
          type: torchcor
        output:
          classifier_bank: ./out.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"backend.type"):
        build_generate_dataset_config(load_yaml(path))


# ---------------------------------------------------------------------------
# Builders for non-default policy types + mix block
# ---------------------------------------------------------------------------


def test_generate_dataset_builds_local_density_policy(tmp_path: Path) -> None:
    """The local_density policy reads radius_mm + threshold from YAML."""
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        label_policy:
          type: local_density
          radius_mm: 3.0
          threshold: 0.2
        output:
          classifier_bank: ./out.h5
        """,
    )
    cfg = build_generate_dataset_config(load_yaml(path))
    assert cfg.label_policy.type == "local_density"
    assert cfg.label_policy.name == "local_density"


def test_generate_dataset_with_inline_mix_block(tmp_path: Path) -> None:
    """A mix block populates cfg.mix; absent block leaves it None."""
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        output:
          classifier_bank: ./out.h5
        mix:
          noise_bank: ./noise.h5
          snr_db_range: [12.0, 22.0]
          master_seed: 7
        """,
    )
    cfg = build_generate_dataset_config(load_yaml(path))
    assert cfg.mix is not None
    assert cfg.mix.noise_bank_path == (tmp_path / "noise.h5").resolve()
    assert cfg.mix.mixer_config.snr_db_range == (12.0, 22.0)
    assert cfg.mix.mixer_config.master_seed == 7


def test_generate_dataset_mix_requires_noise_bank(tmp_path: Path) -> None:
    """Empty mix block (no noise_bank) raises."""
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        output:
          classifier_bank: ./out.h5
        mix:
          snr_db_range: [10.0, 20.0]
        """,
    )
    with pytest.raises(ConfigError, match=r"mix.noise_bank"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_rejects_malformed_bank_id(tmp_path: Path) -> None:
    """A malformed output.bank_id override fails at config-load (fail-fast),
    not after the N-simulation run (S8-3)."""
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        output:
          classifier_bank: ./out.h5
          bank_id: NOT-a-valid-id
        """,
    )
    with pytest.raises(ConfigError, match=r"output.bank_id"):
        build_generate_dataset_config(load_yaml(path))


def test_generate_dataset_accepts_valid_bank_id(tmp_path: Path) -> None:
    """A well-formed stable id passes config-load and reaches the typed config."""
    path = _write_yaml(
        tmp_path,
        """
        dataset:
          n_simulations: 5
        output:
          classifier_bank: ./out.h5
          bank_id: tbank_synthetic_aliev_panfilov_2026-06-27
        """,
    )
    cfg = build_generate_dataset_config(load_yaml(path))
    assert cfg.bank_id == "tbank_synthetic_aliev_panfilov_2026-06-27"


# ---------------------------------------------------------------------------
# build_mix_config — standalone mixer
# ---------------------------------------------------------------------------


def test_mix_config_minimum_required_fields(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        input:
          classifier_bank: ./clean.h5
          noise_bank: ./noise.h5
        output:
          classifier_bank: ./noise_mixed.h5
        """,
    )
    cfg = build_mix_config(load_yaml(path))
    assert cfg.input_classifier_bank == (tmp_path / "clean.h5").resolve()
    assert cfg.noise_bank_path == (tmp_path / "noise.h5").resolve()
    assert cfg.output_classifier_bank == (tmp_path / "noise_mixed.h5").resolve()
    # Mixer defaults reproduce the project's Phase 1 values.
    assert cfg.mixer_config.snr_db_range == (10.0, 25.0)
    assert cfg.mixer_config.bandpass_clean is True


def test_mix_config_rejects_missing_input(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        output:
          classifier_bank: ./noise_mixed.h5
        """,
    )
    with pytest.raises(ConfigError, match=r"input"):
        build_mix_config(load_yaml(path))


def test_mix_config_rejects_inverted_snr_range(tmp_path: Path) -> None:
    """Mixer's __post_init__ rejects lo > hi; surfaces through the builder."""
    path = _write_yaml(
        tmp_path,
        """
        input:
          classifier_bank: ./clean.h5
          noise_bank: ./noise.h5
        output:
          classifier_bank: ./noise_mixed.h5
        mixer:
          snr_db_range: [25.0, 10.0]
        """,
    )
    with pytest.raises(ValueError, match="snr_db_range"):
        build_mix_config(load_yaml(path))


def test_mix_config_rejects_malformed_bank_id(tmp_path: Path) -> None:
    """A malformed output.bank_id override on a mix config fails at load."""
    path = _write_yaml(
        tmp_path,
        """
        input:
          classifier_bank: ./clean.h5
          noise_bank: ./noise.h5
        output:
          classifier_bank: ./noise_mixed.h5
          bank_id: NOT-a-valid-id
        """,
    )
    with pytest.raises(ConfigError, match=r"output.bank_id"):
        build_mix_config(load_yaml(path))


# ---------------------------------------------------------------------------
# Retired output.also_emit_synthetic_bank (SEP12.5)
# ---------------------------------------------------------------------------


def _minimal_generate_doc(tmp_path: Path) -> dict[str, object]:
    """Smallest generate-dataset config the builder accepts."""
    return {
        "_config_dir": tmp_path,
        "dataset": {"n_simulations": 1},
        "backend": {"type": "finitewave"},
        "geometry": {"type": "patch_2d", "size_mm": 40.0, "dr_mm": 0.25},
        "substrate": {"type": "uniform_random_fibrosis", "density_range": [0.0, 0.5]},
        "activation": {"type": "planar_edge"},
        "electrodes": {"type": "centered_grid_2d"},
        "label_policy": {"type": "global_density", "threshold": 0.1},
        "run": {
            "trace_duration_ms": 192.0,
            "output_fs_hz": 1000.0,
            "ap_time_unit_ms": 1.97,
        },
        "output": {"classifier_bank": "out.classifier.h5"},
    }


def test_retired_synthetic_bank_flag_is_rejected(tmp_path: Path) -> None:
    """A config still setting the retired flag fails loudly.

    Silently ignoring it would be worse: the failure would surface much
    later as a missing artifact at analysis time, with the config
    apparently saying it had been requested.
    """
    doc = _minimal_generate_doc(tmp_path)
    doc["output"]["also_emit_synthetic_bank"] = True  # type: ignore[index]

    with pytest.raises(ConfigError, match="also_emit_synthetic_bank was retired"):
        build_generate_dataset_config(doc)


def test_retired_flag_rejected_even_when_false(tmp_path: Path) -> None:
    """``false`` is rejected too — the key means nothing now either way.

    Accepting ``false`` would imply the writer still honours it, which
    is exactly the wrong thing to leave a reader believing.
    """
    doc = _minimal_generate_doc(tmp_path)
    doc["output"]["also_emit_synthetic_bank"] = False  # type: ignore[index]

    with pytest.raises(ConfigError, match="also_emit_synthetic_bank was retired"):
        build_generate_dataset_config(doc)


def test_synthetic_bank_path_defaults_to_a_sibling(tmp_path: Path) -> None:
    """Omitting output.synthetic_bank lands it beside the ClassifierBank."""
    cfg = build_generate_dataset_config(_minimal_generate_doc(tmp_path))

    assert cfg.synthetic_bank_output == (tmp_path / "out.synthetic.h5").resolve()
    assert cfg.classifier_bank_output == (tmp_path / "out.classifier.h5").resolve()


def test_explicit_synthetic_bank_path_is_respected(tmp_path: Path) -> None:
    """An explicit path overrides the derived sibling."""
    doc = _minimal_generate_doc(tmp_path)
    doc["output"]["synthetic_bank"] = "elsewhere/theta.synthetic.h5"  # type: ignore[index]

    cfg = build_generate_dataset_config(doc)

    assert cfg.synthetic_bank_output == (tmp_path / "elsewhere/theta.synthetic.h5").resolve()


def test_mix_config_rejects_synthetic_bank_keys(tmp_path: Path) -> None:
    """synthegm-mix refuses both keys that would ask it for a synthetic bank.

    It cannot write one — standalone mixing has no access to the
    generation config — so accepting either key would leave a config
    that reads as though it requested a theta artifact that never
    appears on disk.
    """
    (tmp_path / "in.classifier.h5").touch()
    (tmp_path / "noise.h5").touch()
    base: dict[str, object] = {
        "_config_dir": tmp_path,
        "input": {"classifier_bank": "in.classifier.h5", "noise_bank": "noise.h5"},
        "output": {"classifier_bank": "out.classifier.h5"},
    }

    for key, value in (
        ("also_emit_synthetic_bank", True),
        ("synthetic_bank", "somewhere.synthetic.h5"),
    ):
        doc = {**base, "output": {**base["output"], key: value}}  # type: ignore[dict-item]
        with pytest.raises(ConfigError, match="not supported by synthegm-mix"):
            build_mix_config(doc)
