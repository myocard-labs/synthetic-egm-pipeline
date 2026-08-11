# Using myocard-synthetic-egm-pipeline

`myocard-synthetic-egm-pipeline` is the Finitewave-driven synthetic
intracardiac-EGM producer for the myocard-labs stack. It runs a
Phase-1 simulator (Aliev-Panfilov model on a 2D atrial patch with
uniform-random fibrosis), captures bipolar EGMs through a 5×5
electrode grid, and writes two on-disk artifacts:

- `<name>.classifier.h5` — labelled `ClassifierBank` ready for
  `myocard-egm-classifier` training. Deliberately **source-agnostic**:
  signal, label, and the `simulation_id` key. Nothing about *how* the
  signal was generated.
- `<name>.synthetic.h5` — `synthetic_bank` 2.0 carrying the
  **per-simulation generation config** (geometry, cell model, substrate,
  activation, electrodes, backend, label policy) and the θ-spec.

**Both are written on every run.** They are parallel artifacts joined on
`simulation_id`, not one derived from the other: the ClassifierBank is
what you train on, the `synthetic_bank` is what you read θ and
provenance from. There is no flag to disable either — a switch that
could turn the θ artifact off is a switch that could silently leave the
analysis work with nothing to read.

The mixer overlays low-amplitude IAFDB noise segments (from
[`myocard-iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline))
onto a clean ClassifierBank to produce a noise-mixed bank that the
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
and writes the noise-mixed output as the primary artifact.

```bash
synthegm-generate-dataset CONFIG.yaml [--overwrite] [--no-progress]
```

The shipped examples cover the three common scenarios:

- `examples/synthegm_v1_baseline.yaml` — Phase 1 default: 100-sim
  clean corpus, `LocalDensityLabel(r=2 mm, t=0.1)`, no mixer.
- `examples/synthegm_v1_noise_mixed.yaml` — Same scope, with inline IAFDB
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
distribution, and writes a noise-mixed `ClassifierBank.h5`. Use this when
you want clean + noise-mixed outputs from one simulator run (run
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
  trace_duration_ms: 192.0                 # default (192 = 3x64; T must be a multiple of 64)
  output_fs_hz: 1000.0                     # default (matches IAFDB)
  ap_time_unit_ms: 1.97                    # default (calibrated 2026-06-10)
  capture_oversample: 4                    # default; backend captures at 4x output_fs_hz

output:
  classifier_bank: ../banks/synthegm_v1.classifier.h5   # required
  # clean_intermediate: ../banks/clean.classifier.h5  # optional; omit to skip.
  #   Only meaningful with a mix block. Setting it makes the run write FOUR
  #   files, because the clean bank gets its own theta partner (see below).
  # synthetic_bank: ../banks/theta.synthetic.h5   # optional; omit for a
  #   `<classifier_bank>.synthetic.h5` sibling. It sets WHERE the second
  #   bank lands, not WHETHER — both banks are always written.
  description: ""                          # default; stamped into bank_metadata
  # bank_id: tbank_synthetic_aliev_panfilov_2026-06-27  # optional; auto-derived from cell model when omitted
  # clean_intermediate_bank_id: tbank_synthetic_ap_clean_2026-06-27  # optional;
  #   names the CLEAN pair as bank_id names the mixed one. Only valid alongside
  #   output.clean_intermediate.

# Optional inline mixer block — omit (or set null) for clean-only output.
mix:
  noise_bank: ../banks/iafdb_noise_v1.h5   # required when mix block present
  # noise_bank_id: nbank_iafdb_2026-06-15  # optional; default reads it from the noise run-record sidecar
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
| `output.clean_intermediate` | path | omit to skip | Persist the clean bank pre-mix; only meaningful with a mix block. Its `synthetic_bank` partner is written automatically beside it as `<stem>.synthetic.h5`. |
| `output.synthetic_bank` | path | omit for the sibling default | Where the `synthetic_bank` lands. Sets the **path**, not whether it is written — both banks are written on every run. Default: `<classifier_bank>.synthetic.h5`. |
| `output.description` | str | `""` | Stamped into the bank's metadata. |
| `output.bank_id` | str (ArtifactId) | auto: `tbank_synthetic_<cell_model>_<date>` | Optional explicit id for the primary bank (the noise-mixed bank in the mix path). See [Stable bank IDs](#stable-bank-ids). |
| `output.clean_intermediate_bank_id` | str (ArtifactId) | auto: `tbank_synthetic_<cell_model>_<date>` | Optional explicit id base for the **clean** pair. Rejected if `output.clean_intermediate` is not set — an id for a bank that is never written would be silently ignored. |
| `mix.noise_bank` | path | (required if block present) | iafdb-pipeline noise_bank.h5. |
| `mix.noise_bank_id` | str (ArtifactId) | auto: read from the noise sidecar | Optional override for the mixed noise bank's id (the mixer's provenance entry). |
| `mix.snr_db_range` | `[lo, hi]` | `[10.0, 25.0]` | Per-trace SNR sampled uniformly. |
| `mix.bandpass_clean` | bool | true | Filter clean to bipolar band before mixing. |
| `mix.band_hz` | `[lo, hi]` | `[30.0, 300.0]` | Bandpass band. |
| `mix.master_seed` | int | 0 | Mixer RNG seed. |

### `synthegm-mix` config

```yaml
input:
  classifier_bank: ../banks/synthegm_v1_clean.classifier.h5   # required
  noise_bank:      ../banks/iafdb_noise_v1.h5                 # required
  # noise_bank_id: nbank_iafdb_2026-06-15                     # optional; default reads the noise sidecar

output:
  classifier_bank: ../banks/synthegm_v1_noise_mixed.classifier.h5  # required
  # No synthetic_bank key here: synthegm-mix writes only a ClassifierBank
  #   (see below). Setting `synthetic_bank` or `also_emit_synthetic_bank`
  #   in a mix config is an error, not a no-op.
  # bank_id: tbank_synthetic_aliev_panfilov_noise_mixed_2026-06-27 # optional; auto-derived (_noise_mixed) when omitted

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
| `input.noise_bank_id` | str (ArtifactId) | auto: read from the noise sidecar | Optional override for the noise bank's id (the mixer's provenance entry). |
| `output.classifier_bank` | path | (required) | Noise-mixed output path. |

| `output.bank_id` | str (ArtifactId) | auto: `tbank_synthetic_<cell_model>_noise_mixed_<date>` | Optional explicit id for the noise-mixed bank. See [Stable bank IDs](#stable-bank-ids). |
| `mixer.snr_db_range` | `[lo, hi]` | `[10.0, 25.0]` | Per-trace target SNR. |
| `mixer.bandpass_clean` | bool | true | Filter clean to bipolar band before mixing. |
| `mixer.band_hz` | `[lo, hi]` | `[30.0, 300.0]` | Bandpass band. |
| `mixer.master_seed` | int | 0 | Mixer RNG seed. |

## Stable bank IDs

Every bank the producer writes carries a stable cross-artifact ID — an egm-contracts `ArtifactId` (added in egm-contracts v0.5.0 for the cross-artifact-linkage system). The intracardiac-platform phase manifests and the provenance graph key on it.

**Finding the two banks that belong together.** The ClassifierBank's `banks` list carries a `synthetic_generation_params` entry naming its `synthetic_bank` (`bank_metadata.join_key` = `simulation_id`), so a consumer can pair them from the file rather than being told. The entry describing the ClassifierBank's *own* traces has `bank_path` = `<local>` — the traces are in that file, there is no source bank. A path that *is* present points at a real companion (the noise bank, the θ bank), relative when it sits inside the same directory tree.

**Clean path.** A run writes two banks, and they take **distinct IDs derived from one base** — the phase manifest keys artifacts by stable ID, so two files sharing one ID collide. The base is derived from the cell model: `tbank_synthetic_<cell_model>_<date>` — e.g. `tbank_synthetic_aliev_panfilov_2026-06-27` (`<date>` is the write-time UTC date). The **ClassifierBank keeps the base**; the **`synthetic_bank` gets a `theta` marker** in the descriptive name, before the date:

| Artifact | ID |
|---|---|
| ClassifierBank | `tbank_synthetic_aliev_panfilov_2026-06-27` |
| `synthetic_bank` | `tbank_synthetic_aliev_panfilov_theta_2026-06-27` |

`output.bank_id` overrides the **base**, so one setting names both and keeps the pair in step — `bank_id: tbank_run7` gives `tbank_run7` and `tbank_run7_theta`. Sharing a stem is deliberate: the two are one run's output, and IDs that sort together make that visible. The ID is stamped on the bank's own `id`, on its source-bank entry, and on every trace. Synthetic banks are always labeled training banks, so the role prefix is always `tbank_`.

**Noise-mixed (mixer) path.** The noise-mixed ClassifierBank gets a `_noise_mixed` variant (`tbank_synthetic_<cell_model>_noise_mixed_<date>`). Its mixed traces keep the **clean** source bank's ID — the noise is additive, so the clean synthetic is the primary source. The mixer also appends a "noise source" provenance entry whose ID is the **noise bank's own** stable ID: it reads that from the iafdb noise run-record sidecar (`<noise_bank>_run_record.json`), falling back to a derived `nbank_iafdb_<date>` if the sidecar is absent.

**Overrides.** Set `output.bank_id` (the primary bank's ID) or `mix.noise_bank_id` / `input.noise_bank_id` (the noise reference) in the config — or pass `bank_id=` / `noise_bank_id=` / `noise_mixed_bank_id=` to the orchestrators. An explicit ID is validated against the ArtifactId pattern (`^[a-z]+_[a-z0-9_]+_\d{4}-\d{2}-\d{2}(_v\d+)?$`) and rejected up front if malformed.

## End-to-end walkthroughs

### Producing the project-default clean dataset

```bash
synthegm-generate-dataset examples/synthegm_v1_baseline.yaml
```

Two files land beside each other (paths relative to the example
config's directory):

- `../banks/synthegm_v1_baseline.classifier.h5` — the ClassifierBank
  egm-classifier consumes directly via `load_classifier_bank`. The
  patient-aware split treats each `simulation_id` as one patient, since
  traces from one simulation share a substrate.
- `../banks/synthegm_v1_baseline.synthetic.h5` — the `synthetic_bank`
  carrying θ and the per-simulation generation config, named by the
  ClassifierBank's `synthetic_generation_params` entry.

### Producing a noise-mixed dataset in one step (simulate + mix)

```bash
synthegm-generate-dataset examples/synthegm_v1_noise_mixed.yaml
```

Three files come out:

- `../banks/synthegm_v1_noise_mixed.classifier.h5` — the primary
  artifact (post-mixer noise-mixed ClassifierBank).
- `../banks/synthegm_v1_noise_mixed.synthetic.h5` — its `synthetic_bank`
  partner, carrying the same per-simulation generation config with the
  mixed signals and per-trace noise provenance.
- `../banks/synthegm_v1_clean.classifier.h5` — the pre-mix clean
  intermediate (preserved because the example config sets
  `output.clean_intermediate`). Useful for ablation studies comparing
  clean-only vs noise-mixed training without re-running the simulator.
- `../banks/synthegm_v1_clean.synthetic.h5` — **the clean intermediate's own
  `synthetic_bank` partner**, carrying the same per-simulation config with the
  *clean* signals.

**Every ClassifierBank has exactly one `synthetic_bank` partner whose id it
names.** That is why a mix run with `output.clean_intermediate` produces four
files rather than three. The clean bank used to point at the mixed run's
`synthetic_bank`, which failed twice over: the id it derived matched no
artifact, so `join_traces_with_simulations` refused it outright — and had the
ids matched, the join would have handed back **mixed** waveforms for traces
read as clean, since a `synthetic_bank` stores `traces/signal`, not just
config.

The mixed pair takes `_noise_mixed` ids: the ClassifierBank
`tbank_…_noise_mixed_<date>` and its partner
`tbank_…_noise_mixed_theta_<date>`. The clean pair keeps the bare stem:
`tbank_…_<date>` and `tbank_…_theta_<date>`.

> **Standalone `synthegm-mix` writes no `synthetic_bank`**, so its output has
> **no** theta partner and carries no theta entry at all. Schema 2.0's
> per-simulation config cannot be recovered from a ClassifierBank's per-trace
> metadata, so there is nothing to write. Use the inline path when you need the
> theta artifact.

### Producing a noise-mixed dataset in two steps (decoupled mix)

```bash
synthegm-generate-dataset examples/synthegm_v1_baseline.yaml
synthegm-mix              examples/synthegm_mix.yaml
```

Note the two-step route produces **no `synthetic_bank` for the mixed
output** — `synthegm-mix` post-processes a ClassifierBank on disk and
has no access to the generation config. Use the one-step route above
when the mixed bank needs a θ partner.

Equivalent output to the one-step noise-mixed path. Useful when you want
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

egm-classifier can then evaluate against each noise-mixed bank
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
noise_mixed = mix_classifier_bank(
    clean_bank=clean,
    noise_bank=noise,
    config=MixerConfig(snr_db_range=(10.0, 25.0)),
)
write_classifier_bank(noise_mixed, "out/synthegm_dev_noise_mixed.classifier.h5", overwrite=True)
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
