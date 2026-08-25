"""CLI: generate one N-simulation synthetic EGM dataset.

Wired as the ``synthegm-generate-dataset`` console_script. All
substantive parameters live in the YAML config; only ``--overwrite``
and ``--no-progress`` remain as run-time flags. See
``examples/synthegm_*.yaml`` for ready-to-use config files.

Usage
-----
::

    synthegm-generate-dataset CONFIG.yaml [--overwrite] [--no-progress]

Examples
--------
::

    # Project-default 100-sim clean dataset.
    synthegm-generate-dataset examples/synthegm_v1_baseline.yaml

    # Same scope, with inline mixing against an iafdb-pipeline noise bank.
    synthegm-generate-dataset examples/synthegm_v1_noise_mixed.yaml

    # Calibration run with deterministic edge / fixed density.
    synthegm-generate-dataset examples/synthegm_calibration.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from myocard_egm_data.banks import (
    read_noise_bank_hdf5,
    write_classifier_bank,
)

from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
from myocard_synthetic_egm_pipeline.cli._config import (
    ConfigError,
    GenerateDatasetCLIConfig,
    build_generate_dataset_config,
    load_yaml,
)
from myocard_synthetic_egm_pipeline.mixer import mix_classifier_bank
from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig,
    DatasetResult,
    build_classifier_bank_from_dataset,
    generate_dataset,
    write_classifier_bank_from_dataset,
    write_synthetic_bank_from_dataset,
)


def _build_dataset_config(cfg: GenerateDatasetCLIConfig, show_progress: bool) -> DatasetConfig:
    """Translate the CLI's typed config into the orchestrator's DatasetConfig."""
    return DatasetConfig(
        n_simulations=cfg.n_simulations,
        geometry=cfg.geometry,
        label_policy=cfg.label_policy,
        run_config=cfg.run_config,
        cell_model=cfg.cell_model,
        detection_preprocessor=cfg.detection_preprocessor,
        probe_grid=cfg.probe_grid,
        fibrosis_density_range=cfg.fibrosis_density_range,
        fraction_healthy=cfg.fraction_healthy,
        fixed_stim_edge=cfg.fixed_stim_edge,
        stimulus_delay_ms=cfg.stimulus_delay_ms,
        electrode_n_rows=cfg.electrode_n_rows,
        electrode_n_cols=cfg.electrode_n_cols,
        electrode_spacing_mm=cfg.electrode_spacing_mm,
        electrode_height_mm_range=cfg.electrode_height_mm_range,
        master_seed=cfg.master_seed,
        show_progress=show_progress,
    )


def _format_result(
    *,
    cfg: GenerateDatasetCLIConfig,
    dataset_result: DatasetResult,
    classifier_path: Path,
    clean_intermediate_path: Path | None,
    synthetic_path: Path | None,
    mixed: bool,
    clean_theta_path: Path | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"Wrote classifier bank:    {classifier_path}")
    if synthetic_path is not None:
        lines.append(f"Wrote synthetic bank:     {synthetic_path}")
    if clean_intermediate_path is not None:
        lines.append(f"Wrote clean intermediate: {clean_intermediate_path}")
    if clean_theta_path is not None:
        lines.append(f"Wrote clean synthetic:    {clean_theta_path}")
    lines.append(f"  N simulations:          {len(dataset_result.results)}")
    lines.append(f"  N traces:               {dataset_result.labels.size}")
    lines.append(f"  Label policy:           {cfg.label_policy.name}")
    # Printed because nothing else records it: the curve decides where each
    # window was cut, and neither bank schema has a field for it. This is the
    # string to paste into output.description.
    #
    # Absent from the summary when the run did not crop, which is the only case
    # where there is no curve — the config nests it inside activation_position,
    # so a run without that block has none. Printing one anyway would report a
    # window placement that never happened.
    if cfg.detection_preprocessor is not None:
        lines.append(f"  Detection curve:        {cfg.detection_preprocessor.name}")
    if cfg.probe_grid is not None:
        # The snap reported ONCE, here, rather than per trace: the config states
        # fractions and the sweep cuts at sample offsets, and the difference is
        # a fact about the run rather than about any one window. It also says
        # plainly that this bank holds n_points simulations, not the one the
        # config asked for.
        grid = cfg.probe_grid
        lines.append(
            f"  Probe grid:             {grid.n_points} offsets, "
            f"{grid.offsets_samples[0]}..{grid.offsets_samples[-1]} samples "
            f"(p {grid.low:.4f}..{grid.high:.4f}), "
            f"snapped by at most {grid.max_snap_error:.4f}"
        )
        lines.append(f"  Probe simulations:      {grid.n_points} (one per offset, shared seed)")
    keys, counts = _unique_counts(dataset_result.labels)
    label_counts = {int(k): int(v) for k, v in zip(keys, counts, strict=False)}
    label_descrs = ", ".join(
        f"{dataset_result.labels_dict.get(k, str(k))}={n}" for k, n in label_counts.items()
    )
    lines.append(f"  By label:               {label_descrs}")
    if mixed:
        assert cfg.mix is not None
        lines.append(f"  Mixer:                  ON ({cfg.mix.noise_bank_path})")
    return "\n".join(lines)


def _unique_counts(
    arr: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per-label counts; thin wrapper around ``np.unique`` for typing."""
    keys, counts = np.unique(arr, return_counts=True)
    return keys, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="synthegm-generate-dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML config file (see examples/synthegm_*.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs. Off by default to protect previous runs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the per-sim tqdm progress bar (overrides the YAML's show_progress).",
    )
    args = parser.parse_args(argv)

    try:
        doc = load_yaml(args.config)
        cfg = build_generate_dataset_config(doc)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    show_progress = cfg.show_progress and not args.no_progress

    try:
        # Phase 1 backend is finitewave by config validation. When new
        # backends land, dispatch on cfg.backend_type here.
        backend = FinitewaveBackend()
        dataset_cfg = _build_dataset_config(cfg, show_progress=show_progress)
        dataset_result = generate_dataset(
            config=dataset_cfg,
            backend=backend,
            position_generator=cfg.position_generator,
        )
    except Exception as exc:
        print(f"ERROR during simulation: {exc}", file=sys.stderr)
        return 1

    try:
        if cfg.mix is None:
            # Clean-only output path.
            classifier_path = write_classifier_bank_from_dataset(
                dataset_result=dataset_result,
                config=dataset_cfg,
                output_path=cfg.classifier_bank_output,
                description=cfg.description,
                overwrite=args.overwrite,
                bank_id=cfg.bank_id,
                synthetic_bank_path=cfg.synthetic_bank_output,
            )
            # Both banks, always: the ClassifierBank for training and the
            # synthetic_bank for theta + per-simulation provenance, joined
            # on simulation_id.
            synthetic_path: Path | None = write_synthetic_bank_from_dataset(
                dataset_result=dataset_result,
                config=dataset_cfg,
                output_path=cfg.synthetic_bank_output,
                description=cfg.description,
                overwrite=args.overwrite,
                bank_id=cfg.bank_id,
            )
            print(
                _format_result(
                    cfg=cfg,
                    dataset_result=dataset_result,
                    classifier_path=classifier_path,
                    clean_intermediate_path=None,
                    synthetic_path=synthetic_path,
                    mixed=False,
                )
            )
            return 0

        # Inline-mixer path. Build clean bank in memory, optionally
        # write the clean intermediate to disk, mix, then write the
        # noise-mixed bank as the primary output.
        #
        # When a clean intermediate is requested it is a *published artifact*,
        # not scratch: it gets its own id base and its own theta partner, so
        # the pair is joinable on its own terms. It used to be
        # built with no bank_id at all — falling back to a cell-model-derived
        # id — while naming the mixed run's theta file, so its companion id
        # matched no artifact and egm-data refused the join.
        writing_clean = cfg.clean_intermediate_output is not None
        clean_bank = build_classifier_bank_from_dataset(
            dataset_result=dataset_result,
            config=dataset_cfg,
            bank_path=cfg.clean_intermediate_output or cfg.classifier_bank_output,
            description=cfg.description,
            bank_id=cfg.clean_intermediate_bank_id,
            synthetic_bank_path=(
                cfg.clean_theta_output if writing_clean else cfg.synthetic_bank_output
            ),
        )

        clean_intermediate_path: Path | None = None
        clean_theta_path: Path | None = None
        if writing_clean:
            assert cfg.clean_intermediate_output is not None
            assert cfg.clean_theta_output is not None
            clean_intermediate_path = write_classifier_bank(
                clean_bank, cfg.clean_intermediate_output, overwrite=args.overwrite
            )
            # The clean bank's own theta partner: same per-simulation config as
            # the mixed one, but the *clean* signals. Sharing the mixed theta
            # file would hand a consumer mixed waveforms for a trace it joined
            # as clean, with nothing flagging the swap.
            clean_theta_path = write_synthetic_bank_from_dataset(
                dataset_result=dataset_result,
                config=dataset_cfg,
                output_path=cfg.clean_theta_output,
                description=cfg.description,
                overwrite=args.overwrite,
                bank_id_base=str(clean_bank.id),
            )

        noise_bank = read_noise_bank_hdf5(cfg.mix.noise_bank_path)
        noise_mixed_bank = mix_classifier_bank(
            clean_bank=clean_bank,
            noise_bank=noise_bank,
            config=cfg.mix.mixer_config,
            noise_bank_path=str(cfg.mix.noise_bank_path),
            noise_bank_id=cfg.mix.noise_bank_id,
            noise_mixed_bank_id=cfg.bank_id,
            output_bank_path=str(cfg.classifier_bank_output),
            theta_bank_path=str(cfg.synthetic_bank_output),
        )
        classifier_path = write_classifier_bank(
            noise_mixed_bank, cfg.classifier_bank_output, overwrite=args.overwrite
        )
        # Built from the in-memory DatasetResult, not reconstructed from
        # the mixed ClassifierBank: schema 2.0's per-simulation config is
        # not recoverable from per-trace metadata. Only the signals and
        # the three noise columns come from the mixer.
        synthetic_path = write_synthetic_bank_from_dataset(
            dataset_result=dataset_result,
            config=dataset_cfg,
            output_path=cfg.synthetic_bank_output,
            description=cfg.description,
            overwrite=args.overwrite,
            bank_id=cfg.bank_id,
            mixed_signals=[t.signal for t in noise_mixed_bank.traces],
            snr_db=[float(t.trace_metadata["snr_db"]) for t in noise_mixed_bank.traces],
            noise_record=[str(t.trace_metadata["noise_record"]) for t in noise_mixed_bank.traces],
            noise_channel=[str(t.trace_metadata["noise_channel"]) for t in noise_mixed_bank.traces],
            noise_bank_source=str(noise_bank.source),
        )
        print(
            _format_result(
                cfg=cfg,
                dataset_result=dataset_result,
                classifier_path=classifier_path,
                clean_intermediate_path=clean_intermediate_path,
                clean_theta_path=clean_theta_path,
                synthetic_path=synthetic_path,
                mixed=True,
            )
        )
        return 0
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Hint: pass --overwrite to replace.", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR while writing outputs: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
