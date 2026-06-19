"""CLI: mix an already-written clean ClassifierBank with a noise bank.

Wired as the ``synthegm-mix`` console_script. Standalone counterpart to
the ``mix:`` block in ``synthegm-generate-dataset`` — use this when
you have a clean bank on disk and want to overlay noise without
re-running the simulator.

Usage
-----
::

    synthegm-mix CONFIG.yaml [--overwrite] [--no-progress]

Examples
--------
::

    # Mix a clean bank with the default IAFDB noise bank.
    synthegm-mix examples/synthegm_mix.yaml

    # Fixed SNR for ablation studies.
    synthegm-mix examples/synthegm_mix_fixed_snr.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from myocard_egm_data.banks import (
    load_classifier_bank,
    read_noise_bank_hdf5,
    write_classifier_bank,
)

from myocard_synthetic_egm_pipeline.cli._config import (
    ConfigError,
    MixCLIConfig,
    build_mix_config,
    load_yaml,
)
from myocard_synthetic_egm_pipeline.mixer import mix_classifier_bank
from myocard_synthetic_egm_pipeline.mixer.storage import (
    write_hybrid_synthetic_bank_from_classifier,
)


def _format_result(
    *,
    cfg: MixCLIConfig,
    n_traces: int,
    classifier_path: Path,
    synthetic_path: Path | None,
) -> str:
    lines: list[str] = []
    lines.append(f"Wrote hybrid classifier bank: {classifier_path}")
    if synthetic_path is not None:
        lines.append(f"Wrote hybrid synthetic bank:  {synthetic_path}")
    lines.append(f"  N traces:                   {n_traces}")
    lines.append(f"  Noise bank:                 {cfg.noise_bank_path}")
    lo, hi = cfg.mixer_config.snr_db_range
    lines.append(f"  Target SNR range:           [{lo}, {hi}] dB")
    lines.append(f"  Bandpass clean:             {cfg.mixer_config.bandpass_clean}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="synthegm-mix",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML config file (see examples/synthegm_mix*.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs. Off by default to protect previous runs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the per-trace tqdm progress bar (overrides the YAML's show_progress).",
    )
    args = parser.parse_args(argv)

    try:
        doc = load_yaml(args.config)
        cfg = build_mix_config(doc)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Honour --no-progress override on the mixer's own progress bar.
    mixer_config = cfg.mixer_config
    if args.no_progress and mixer_config.show_progress:
        from dataclasses import replace

        mixer_config = replace(mixer_config, show_progress=False)

    try:
        clean_bank = load_classifier_bank(cfg.input_classifier_bank)
        noise_bank = read_noise_bank_hdf5(cfg.noise_bank_path)
    except FileNotFoundError as exc:
        print(f"ERROR reading input: {exc}", file=sys.stderr)
        return 1

    try:
        hybrid_bank = mix_classifier_bank(
            clean_bank=clean_bank,
            noise_bank=noise_bank,
            config=mixer_config,
            noise_bank_path=str(cfg.noise_bank_path),
        )
    except Exception as exc:
        print(f"ERROR during mixing: {exc}", file=sys.stderr)
        return 1

    try:
        classifier_path = write_classifier_bank(
            hybrid_bank, cfg.output_classifier_bank, overwrite=args.overwrite
        )
        synthetic_path: Path | None = None
        if cfg.also_emit_synthetic_bank:
            output_path = cfg.output_synthetic_bank or _sibling_synthetic(
                cfg.output_classifier_bank
            )
            synthetic_path = write_hybrid_synthetic_bank_from_classifier(
                hybrid_bank=hybrid_bank,
                output_path=output_path,
                description=mixer_config.description,
                overwrite=args.overwrite,
            )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Hint: pass --overwrite to replace.", file=sys.stderr)
        return 1

    print(
        _format_result(
            cfg=cfg,
            n_traces=len(hybrid_bank.traces),
            classifier_path=classifier_path,
            synthetic_path=synthetic_path,
        )
    )
    return 0


def _sibling_synthetic(primary: Path) -> Path:
    """Default Pydantic-SyntheticBank sibling path next to ``primary``."""
    return primary.with_name(f"{primary.stem}.synthetic.h5")


if __name__ == "__main__":
    raise SystemExit(main())
