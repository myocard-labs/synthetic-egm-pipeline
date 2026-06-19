# Using myocard-synthetic-egm-pipeline

`myocard-synthetic-egm-pipeline` is the Finitewave-driven synthetic
intracardiac-EGM producer for the myocard-labs stack. It runs a
Phase-1 simulator (Aliev-Panfilov model on a 2D atrial patch with
uniform-random fibrosis), captures bipolar EGMs through a 5×5
electrode grid, and writes two on-disk artifacts:

- `<name>.classifier.h5` — labelled `ClassifierBank` ready for
  `myocard-egm-classifier` training. Default output.
- `<name>.synthetic.h5` — Pydantic `SyntheticBank` sibling (optional
  via config flag) for offline analysis tools that read the per-trace
  columnar layout.

The mixer overlays low-amplitude IAFDB noise segments (from
[`myocard-iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline))
onto a clean ClassifierBank to produce a hybrid bank that the
classifier trains on.

Every output validates against a schema in
[`myocard-egm-contracts`](https://github.com/myocard-labs/egm-contracts);
this package owns the simulator orchestration, the pseudo-EGM
forward calc, the label policy, and the mixer.

The CLIs are the primary surface. Programmatic use is supported (the
strategy types, builder functions, and `mix_classifier_bank` are all
importable), but you'll typically drive the pipeline from YAML
configs.

## Install

During pre-1.0 iteration:

```bash
pip install "myocard-synthetic-egm-pipeline @ git+https://github.com/myocard-labs/synthetic-egm-pipeline.git"
```

Editable for development:

```bash
git clone https://github.com/myocard-labs/synthetic-egm-pipeline.git
cd synthetic-egm-pipeline
pip install -e ".[dev]"
pre-commit install
```

The runtime deps are `myocard-egm-contracts`, `myocard-egm-data`,
`myocard-egm-signal`, `numpy`, `scipy`, `finitewave`, `tqdm`, and
`pyyaml`. Finitewave is a hard runtime dep — the whole point of this
package is the simulator pipeline.

## CLI reference

The package ships two console scripts. Both are wired in
`[project.scripts]`; `pip install` puts them on the PATH.

| Command | Purpose |
|---|---|
| `synthegm-generate-dataset` | Run an N-simulation Finitewave dataset; optionally overlay noise inline. |
| `synthegm-mix` | Standalone mixer — overlay noise on an already-written clean ClassifierBank. |

Both take a positional YAML config plus `--overwrite` and
`--no-progress`. All substantive parameters live in the YAML.

### `synthegm-generate-dataset`

The headline producer. Runs `n_simulations` Finitewave simulations
with per-sim sampled parameters (fibrosis density, activation edge,
electrode height), labels each bipolar pair via the configured
`LabelPolicy`, and writes a `ClassifierBank.h5`. If a `mix:` block is
present, the mixer runs inline against the just-generated clean bank
and writes the hybrid output as the primary artifact.

```bash
synthegm-generate-dataset CONFIG.yaml [--overwrite] [--no-progress]
```

The shipped examples cover the three common scenarios:

- `examples/synthegm_v1_baseline.yaml` — Phase 1 default: 100-sim
  clean corpus, `LocalDensityLabel(r=2 mm, t=0.1)`, no mixer.
- `examples/synthegm_v1_hybrid.yaml` — Same scope, with inline IAFDB
  noise overlay. Persists the clean intermediate too for ablation
  studies.
- `examples/synthegm_calibration.yaml` — 4-sim deterministic run
  (fixed edge, fixed density, pinned electrode height) for verifying
  CV calibration or eyeballing bipolar morphology before scaling up.

Run one with:

```bash
synthegm-generate-dataset examples/synthegm_v1_baseline.yaml
```

The CLI prints a summary on exit: output paths, simulation count,
trace count, label-policy name, label distribution, and (if mixed)
the noise bank that was overlaid.

### `synthegm-mix`

Standalone mixer. Reads an existing clean `ClassifierBank.h5` and an
IAFDB `noise_bank.h5`, overlays noise per trace at the configured SNR
distribution, and writes a hybrid `ClassifierBank.h5`. Use this when
you want clean + hybrid outputs from one simulator run (run
`synthegm-generate-dataset` once, then `synthegm-mix` once or many
times against different noise banks or SNR ranges).

```bash
synthegm-mix CONFIG.yaml [--overwrite] [--no-progress]
```

Shipped examples:

- `examples/synthegm_mix.yaml` — Standard mix with v1 default SNR
  range `[10, 25]` dB.
- `examples/synthegm_mix_fixed_snr.yaml` — Collapsed SNR
  (`[15, 15]`) for SNR-vs-AUROC ablation sweeps.

## YAML schema reference

### `synthegm-generate-dataset` config

```yaml
dataset:
  n_simulations: 100                       # required; >= 1
  master_seed: 42                          # default 0
  show_progress: true                      # default true

backend:
  type: finitewave                         # default 'finitewave' (Phase 1: only option)

geometry:
  type: patch_2d                           # default 'patch_2d' (Phase 1: only option)
  size_mm: 40.0                            # default
  dr_mm: 0.25                              # default
  fiber_angle_rad: 0.0                     # default
  anisotropy_ratio: 3.0                    # default

substrate:
  type: uniform_random_fibrosis            # default (Phase 1: only option)
  density_range: [0.0, 0.5]                # default; sampled uniformly per sim
  fraction_healthy: 0.0                    # default; fraction of sims forced to density=0

activation:
  type: planar_edge                        # default (Phase 1: only option)
  fixed_edge: null                         # default: randomise per sim; or 'top'/'bottom'/'left'/'right'

electrodes:
  type: centered_grid_2d                   # default (Phase 1: only option)
  n_rows: 5                                # default
  n_cols: 5                                # default
  spacing_mm: 2.0                          # default
  height_mm_range: [0.2, 1.0]              # default; per-sim height sampled uniformly

label_policy:
  type: global_density                     # default; or 'local_density'
  threshold: 0.1                           # default
  radius_mm: 2.0                           # only used when type=local_density
  healthy_name: healthy                    # default; surfaces in ClassifierBank.labels
  fibrotic_name: fibrotic                  # default

run:
  trace_duration_ms: 200.0                 # default
  output_fs_hz: 1000.0                     # default (matches IAFDB)
  ap_time_unit_ms: 1.97                    # default (calibrated 2026-06-10)
  capture_oversample: 4                    # default; backend captures at 4x output_fs_hz

output:
  classifier_bank: ../banks/synthegm_v1.classifier.h5   # required
  clean_intermediate: null                 # default: don't persist; only used with mix block
  also_emit_synthetic_bank: false          # default; set true to write SyntheticBank sibling
  synthetic_bank: null                     # default: <classifier_bank>.synthetic.h5 sibling
  description: ""                          # default; stamped into bank_metadata

# Optional inline mixer block — omit (or set null) for clean-only output.
mix:
  noise_bank: ../banks/iafdb_noise_v1.h5   # required when mix block present
  snr_db_range: [10.0, 25.0]               # default
  bandpass_clean: true                     # default; filter clean to bipolar band before mixing
  band_hz: [30.0, 300.0]                   # default
  master_seed: 0                           # default (mixer's RNG; independent of simulator's)
  show_progress: true                      # default
  description: ""                          # default
```

Per-field reference:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dataset.n_simulations` | int | (required) | Number of independent simulations. |
| `dataset.master_seed` | int | 0 | Master RNG seed; per-sim seeds derive from this. |
| `dataset.show_progress` | bool | true | tqdm bar over the simulator loop. |
| `backend.type` | `finitewave` | `finitewave` | Backend dispatch. Phase 1 only has finitewave. |
| `geometry.type` | `patch_2d` | `patch_2d` | Geometry dispatch. Phase 1 only has patch_2d. |
| `geometry.size_mm` | float | 40.0 | Patch edge length. |
| `geometry.dr_mm` | float | 0.25 | Spatial step (mesh cell size). |
| `geometry.fiber_angle_rad` | float | 0.0 | Fiber orientation. |
| `geometry.anisotropy_ratio` | float | 3.0 | CV_along / CV_across; atrial ~2-3:1. |
| `substrate.type` | `uniform_random_fibrosis` | `uniform_random_fibrosis` | Substrate dispatch. |
| `substrate.density_range` | `[lo, hi]` | `[0.0, 0.5]` | Per-sim density sampled uniformly. |
| `substrate.fraction_healthy` | float [0, 1] | 0.0 | Fraction of sims forced to density=0 exactly. |
| `activation.type` | `planar_edge` | `planar_edge` | Activation dispatch. |
| `activation.fixed_edge` | str/null | null | If set, every sim fires from this edge. |
| `electrodes.type` | `centered_grid_2d` | `centered_grid_2d` | Electrode dispatch. |
| `electrodes.n_rows` / `n_cols` | int | 5 / 5 | Grid shape; 20 bipolar pairs per sim for 5x5. |
| `electrodes.spacing_mm` | float | 2.0 | Intra-row + inter-row spacing. |
| `electrodes.height_mm_range` | `[lo, hi]` | `[0.2, 1.0]` | Per-sim height sampled uniformly. |
| `label_policy.type` | `global_density` / `local_density` | `global_density` | Label policy dispatch. |
| `label_policy.threshold` | float | 0.1 | Density above which a trace is labeled fibrotic. |
| `label_policy.radius_mm` | float | 2.0 | Used by `local_density` only. |
| `run.trace_duration_ms` | float | 200.0 | Per-trace length. |
| `run.output_fs_hz` | float | 1000.0 | Output sample rate. |
| `run.ap_time_unit_ms` | float | 1.97 | AP non-dimensional time → ms calibration. |
| `run.capture_oversample` | int >=1 | 4 | Backend captures at oversample × output_fs_hz. |
| `output.classifier_bank` | path | (required) | Primary output path. |
| `output.clean_intermediate` | path/null | null | Persist clean bank pre-mix; only meaningful with mix block. |
| `output.also_emit_synthetic_bank` | bool | false | Write SyntheticBank sibling. |
| `output.synthetic_bank` | path/null | null | Override sibling path (default `<classifier_bank>.synthetic.h5`). |
| `output.description` | str | `""` | Stamped into the bank's metadata. |
| `mix.noise_bank` | path | (required if block present) | iafdb-pipeline noise_bank.h5. |
| `mix.snr_db_range` | `[lo, hi]` | `[10.0, 25.0]` | Per-trace SNR sampled uniformly. |
| `mix.bandpass_clean` | bool | true | Filter clean to bipolar band before mixing. |
| `mix.band_hz` | `[lo, hi]` | `[30.0, 300.0]` | Bandpass band. |
| `mix.master_seed` | int | 0 | Mixer RNG seed. |

### `synthegm-mix` config

```yaml
input:
  classifier_bank: ../banks/synthegm_v1_clean.classifier.h5   # required
  noise_bank:      ../banks/iafdb_noise_v1.h5                 # required

output:
  classifier_bank: ../banks/synthegm_v1_hybrid.classifier.h5  # required
  also_emit_synthetic_bank: false                             # default
  synthetic_bank: null                                        # default sibling

mixer:
  snr_db_range: [10.0, 25.0]               # default
  bandpass_clean: true                     # default
  band_hz: [30.0, 300.0]                   # default
  master_seed: 0                           # default
  show_progress: true                      # default
  description: ""                          # default
```

Per-field reference:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `input.classifier_bank` | path | (required) | Clean ClassifierBank to overlay noise on. |
| `input.noise_bank` | path | (required) | iafdb-pipeline noise_bank.h5. |
| `output.classifier_bank` | path | (required) | Hybrid output path. |
| `output.also_emit_synthetic_bank` | bool | false | Write hybrid SyntheticBank sibling. |
| `output.synthetic_bank` | path/null | null | Override sibling path. |
| `mixer.snr_db_range` | `[lo, hi]` | `[10.0, 25.0]` | Per-trace target SNR. |
| `mixer.bandpass_clean` | bool | true | Filter clean to bipolar band before mixing. |
| `mixer.band_hz` | `[lo, hi]` | `[30.0, 300.0]` | Bandpass band. |
| `mixer.master_seed` | int | 0 | Mixer RNG seed. |

## End-to-end walkthroughs

### Producing the project-default clean dataset

```bash
synthegm-generate-dataset examples/synthegm_v1_baseline.yaml
```

The bank lands at `../banks/synthegm_v1_baseline.classifier.h5`
(relative to the example config's directory). egm-classifier consumes
it directly via `load_classifier_bank`; the patient-aware split
treats each `sim_id` as one patient.

### Producing a hybrid dataset in one step (simulate + mix)

```bash
synthegm-generate-dataset examples/synthegm_v1_hybrid.yaml
```

Two files come out:

- `../banks/synthegm_v1_hybrid.classifier.h5` — the primary
  artifact (post-mixer hybrid ClassifierBank).
- `../banks/synthegm_v1_clean.classifier.h5` — the pre-mix clean
  intermediate (preserved because the example config sets
  `output.clean_intermediate`). Useful for ablation studies comparing
  clean-only vs hybrid training without re-running the simulator.

### Producing a hybrid dataset in two steps (decoupled mix)

```bash
synthegm-generate-dataset examples/synthegm_v1_baseline.yaml
synthegm-mix              examples/synthegm_mix.yaml
```

Equivalent output to the one-step hybrid path. Useful when you want
to overlay several different noise banks (or several different SNR
ranges) on the same clean simulation run.

### Running the SNR sweep ablation

For an AUROC-vs-SNR curve at several operating points, create one
copy of `synthegm_mix_fixed_snr.yaml` per SNR point, each with a
unique output filename + `snr_db_range: [X, X]`:

```bash
for snr in 5 10 15 20 25; do
  sed "s/15.0, 15.0/$snr.0, $snr.0/; s/snr15/snr$snr/" \
      examples/synthegm_mix_fixed_snr.yaml > /tmp/mix_snr$snr.yaml
  synthegm-mix /tmp/mix_snr$snr.yaml
done
```

egm-classifier can then evaluate against each hybrid bank
independently and you'll have one AUROC per operating point.

## Programmatic use

If you'd rather drive the producer from Python (notebook analysis,
custom ablation studies, integration tests), the strategy types,
builders, and orchestrators are all importable:

```python
from pathlib import Path
import numpy as np

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig,
    GlobalDensityLabel,
    Patch2DGeometry,
    build_classifier_bank_from_dataset,
    generate_dataset,
    write_classifier_bank_from_dataset,
)

config = DatasetConfig(
    n_simulations=10,
    geometry=Patch2DGeometry(),
    label_policy=GlobalDensityLabel(threshold=0.1),
    run_config=RunConfig(trace_duration_ms=200.0, output_fs_hz=1000.0, ap_time_unit_ms=1.97),
    fibrosis_density_range=(0.0, 0.5),
    fraction_healthy=0.3,
    master_seed=42,
    show_progress=True,
)
result = generate_dataset(config=config, backend=FinitewaveBackend())
print(result.labels.size, "traces;", result.labels_dict)

# Either write to disk directly:
write_classifier_bank_from_dataset(
    dataset_result=result,
    config=config,
    output_path=Path("out/synthegm_dev.classifier.h5"),
    overwrite=True,
)

# Or build the ClassifierBank in memory (for a notebook plot or a custom mix):
bank = build_classifier_bank_from_dataset(
    dataset_result=result,
    config=config,
    bank_path=Path("out/synthegm_dev.classifier.h5"),
)
```

The mixer is similar:

```python
from myocard_egm_data.banks import load_classifier_bank, read_noise_bank_hdf5, write_classifier_bank
from myocard_synthetic_egm_pipeline.mixer import MixerConfig, mix_classifier_bank

clean = load_classifier_bank("out/synthegm_dev.classifier.h5")
noise = read_noise_bank_hdf5("banks/iafdb_noise_v1.h5")
hybrid = mix_classifier_bank(
    clean_bank=clean,
    noise_bank=noise,
    config=MixerConfig(snr_db_range=(10.0, 25.0)),
)
write_classifier_bank(hybrid, "out/synthegm_dev_hybrid.classifier.h5", overwrite=True)
```

Strategy specs are pure-data dataclasses; new substrate / activation /
electrode-placement / label-policy concretes drop into
`simulate/specs.py` + `simulate/label_policy.py` without changing any
of the Protocol interfaces. See `project/architecture.md` for the
plug-in shape.

## Where to read more

- For the design rationale (why four strategy Protocols + a
  Backend boundary, what the three guardrails are, what the
  Option A → Option B migration looks like): `project/architecture.md`.
- For the physics of one simulation run (tissue + substrate + AP
  solver + pseudo-EGM forward calc + labels + storage), with paper
  references and code anchors per step: `docs/simulation_theory.md`.
- For the mixer math (bandpass + SNR scaling + additive overlay +
  per-trace SNR variability), with paper references and code anchors:
  `docs/mixer_theory.md`.
- For what's planned but not in v0.2.0 (noise selection strategies,
  multi-substrate-type composition, 3D geometry trigger,
  compatibility validator): `project/roadmap.md`.
- For the egm-contracts schemas the outputs validate against:
  `myocard-egm-contracts/docs/schemas/`.
- For the egm-data writers the storage layer delegates to:
  `myocard-egm-data/docs/usage.md`.
