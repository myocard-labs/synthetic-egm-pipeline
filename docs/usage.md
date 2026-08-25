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
- `examples/synthegm_probe.yaml` — **positional-sensitivity probe**: one solve
  emitted as one simulation per crop offset. A diagnostic bank, not training
  data — see [The positional-sensitivity
  probe](#the-positional-sensitivity-probe).

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
  # Membrane parameterisation, as a shipped card name or a path to your own
  # YAML. Omit it and you still get the calibrated card — a run never falls
  # back to bare constants, because uncalibrated physics that looks like a
  # normal run is the failure S38 exists to remove.
  model: af_remodelled_220ms               # default; targets CV 80 cm/s, APD90 220 ms
  # model: courtemanche_control            # the human-atrial ionic model (SEP5)

geometry:
  type: patch_2d                           # default 'patch_2d' (Phase 1: only option)
  size_mm: 40.0                            # default
  dr_mm: 0.25                              # default
  fiber_angle_rad: 0.0                     # default
  anisotropy_ratio: 2.0                    # default (2.0, NOT 3.0 — changed in S38b)

substrate:
  type: uniform_random_fibrosis            # default (Phase 1: only option)
  density_range: [0.0, 0.5]                # default; sampled uniformly per sim
  fraction_healthy: 0.0                    # default; fraction of sims forced to density=0

activation:
  type: planar_edge                        # default (Phase 1: only option)
  fixed_edge: null                         # default: randomise per sim; or 'top'/'bottom'/'left'/'right'
  # How long the solver runs before the stimulus fires. Leave it out when
  # `activation_position` is set and it is DERIVED as k(high) — the smallest
  # delay that guarantees signal in front of the activation. Without a crop it
  # defaults to 0. Raise it if the crop reports a window running off the FRONT.
  # stimulus_delay_ms: 115.0

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
  capture_oversample: 4                    # default; backend captures at 4x output_fs_hz
  dr_model_units: 0.25                     # default; the solver's own space step
  # Upper bound on how long the wave may take to reach a pair, which sizes the
  # capture. Default 2 x trace_duration_ms (384 ms at T=192). THIS IS AN
  # ASSUMPTION, NOT A BOUND: it is sized for a substrate slower than any
  # measured, and a dense one can still exceed it. Raise it when the crop
  # reports a window running off the BACK. See the sizing note below.
  # travel_allowance_ms: 500.0

  # RETIRED — ap_time_unit_ms, diffusion, membrane_eps and dt_model_units used
  # to live here. They are now solved from physiological targets and live on the
  # model card named by `backend.model`. Setting any of them here is an ERROR,
  # not an override.

# Optional controlled-position cropping (SEP2). Omit the block for no
# cropping. Both bounds are required; use the same value twice for the
# anchored arm of the position A/B.
activation_position:
  low: 0.25                                # smallest position; DRIVES capture length
  high: 0.75                               # largest position
  # seed: 7                                # default: dataset.master_seed
  # The curve the crop anchors each window on. Omit for rectified_derivative,
  # which is what every bank before this knob existed was windowed with. It
  # nests here, not under `activation:`, because it is part of the crop rather
  # than part of how the wave is launched.
  detection:
    curve: rectified_derivative            # default; or 'teager_kaiser' / 'botteron_envelope'
    # Botteron's two knobs, shown commented because setting them beside any
    # other curve is an ERROR, not an override — they would otherwise be a
    # deliberate setting silently ignored. Uncomment them WITH
    # `curve: botteron_envelope`.
    # botteron_band_hz: [40.0, 250.0]      # default, Botteron 1995
    # botteron_lowpass_hz: 20.0            # default; read at run.output_fs_hz
  # Positional-sensitivity probe (SEP13) — mutually exclusive with low/high
  # above, and requires dataset.n_simulations: 1 (which counts SOLVES: the run
  # writes one simulation per grid point). See "The positional-sensitivity
  # probe" below; it is a diagnostic bank, not training data.
  # grid:
  #   low: 0.2                             # smallest offset; sizes the capture
  #   high: 0.8                            # largest offset; buys the stimulus delay
  #   n_points: 13                         # snapped to the sample lattice

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
| `backend.model` | shipped card name or path | `af_remodelled_220ms` | **Membrane parameterisation, and which cell model runs.** Names a card under `src/myocard_synthetic_egm_pipeline/models/`, or a path to your own. The card states physiological *targets* and the *solved* knobs that reach them; the solve is re-run and checked against the recorded values on **every load**, so a card cannot drift from the routine that produced it. Not optional-with-a-fallback: omitting it still loads the default card, because a run producing uncalibrated physics that looks normal is the failure this replaced. To change the physics, write a card — do not look for knobs in `run:`. See the shipped cards below. |
| `geometry.type` | `patch_2d` | `patch_2d` | Geometry dispatch. Phase 1 only has patch_2d. |
| `geometry.size_mm` | float | 40.0 | Patch edge length. |
| `geometry.dr_mm` | float | 0.25 | Spatial step (mesh cell size). |
| `geometry.fiber_angle_rad` | float | 0.0 | Fiber orientation. With the default 0.0 **every bipole is parallel to the fibres**, since pairs are grid-x separated and fibres run along x — so the anisotropy ratio has less effect on bipolar morphology than it looks. |
| `geometry.anisotropy_ratio` | float | **2.0** | CV_along / CV_across. 2:1 is the standard atrial-wall value (Hansson 1998: RA free wall 88 ± 9 cm/s, only weakly direction-dependent); higher ratios belong to bundles, not working myocardium. **Was 3.0 before S38b** — at 3.0 the transverse velocity falls below the physiological range. Had no effect at all before S38a, when the knob was a silent no-op. |
| `substrate.type` | `uniform_random_fibrosis` | `uniform_random_fibrosis` | Substrate dispatch. |
| `substrate.density_range` | `[lo, hi]` | `[0.0, 0.5]` | Per-sim density sampled uniformly. |
| `substrate.fraction_healthy` | float [0, 1] | 0.0 | Fraction of sims forced to density=0 exactly. |
| `activation.type` | `planar_edge` | `planar_edge` | Activation dispatch. |
| `activation.fixed_edge` | str/null | null | If set, every sim fires from this edge. |
| `activation.stimulus_delay_ms` | float >= 0 | derived, else 0.0 | Solver time before the stimulus fires. With `activation_position` set and this omitted, it is **derived** as `k(high)` — the smallest delay guaranteeing signal in front of the activation at the largest position. Without a crop it is 0. Raise it when the crop reports a window off the **FRONT**. It buys *resting* lead-in only: phi_e sums membrane current over the whole mesh, so while nothing is depolarising the lead-in is flat. |
| `activation_position.detection.curve` | `rectified_derivative` / `teager_kaiser` / `botteron_envelope` | `rectified_derivative` | Detection curve the crop anchors each window on. Same three names iafdb-pipeline uses, so the two corpora can be windowed the same way. **Not recorded in either bank; see below.** |
| `activation_position.detection.botteron_band_hz` | `[lo, hi]` | `[40.0, 250.0]` | Band-pass before rectification (Botteron 1995). `botteron_envelope` only — setting it beside another curve is rejected rather than ignored. |
| `activation_position.detection.botteron_lowpass_hz` | float | 20.0 | Envelope smoothing cutoff. `botteron_envelope` only. Interpreted at `run.output_fs_hz`, not the capture rate: the runner downsamples before it crops. |
| `activation_position.grid.low` / `.high` | float 0..1 | (required if `grid` present) | Ends of the swept offset range, in fractions. **Snapped** to the sample lattice; the snapped values are what sizes the capture. |
| `activation_position.grid.n_points` | int >= 2 | (required if `grid` present) | Points in the sweep, spaced linearly over `[low, high]` before snapping. Two points snapping to one sample offset is an error. **Each point becomes one `simulation_id`**, so the bank holds `n_points x n_pairs` traces. |
| `electrodes.type` | `centered_grid_2d` | `centered_grid_2d` | Electrode dispatch. |
| `electrodes.n_rows` / `n_cols` | int | 5 / 5 | Grid shape; 20 bipolar pairs per sim for 5x5. |
| `electrodes.spacing_mm` | float | 2.0 | Intra-row + inter-row spacing. |
| `electrodes.height_mm_range` | `[lo, hi]` | `[0.2, 1.0]` | Per-sim height sampled uniformly. |
| `label_policy.type` | `global_density` / `local_density` | `global_density` | Label policy dispatch. |
| `label_policy.threshold` | float | 0.1 | Density above which a trace is labeled fibrotic. |
| `label_policy.radius_mm` | float | 2.0 | Used by `local_density` only. |
| `run.trace_duration_ms` | float | 192.0 | Per-trace length **on disk** (T). Must give a sample count that is a multiple of 64 — rejected at config load otherwise (CL-112). With `activation_position` set this is *not* how long the solver runs; see that block. |
| `run.output_fs_hz` | float | 1000.0 | Output sample rate. Shared with IAFDB by decision, not coincidence — catch22 lag features depend on it — so changing it is a both-sides-or-neither call. |
| `run.capture_oversample` | int >=1 | 4 | Backend captures at oversample × output_fs_hz. |
| `run.dr_model_units` | float | 0.25 | The solver's own space step, in the backend's units. Backend/scheme, not physiology: it is paired with the model card's `dt` through the explicit-scheme stability bound `dt <= dr^2 / (2 · dim · D)`. **Must equal `geometry.dr_mm` for a Courtemanche card** — that model's diffusion coefficient is in mm²/ms, so its space unit is the millimetre and a different value simulates a different mesh from the one the geometry describes. Aliev-Panfilov is dimensionless and has no such constraint. Refused rather than absorbed. |
| `run.travel_allowance_ms` | float > 0 | `2 × trace_duration_ms` (384 ms at T=192) | **Assumed upper bound** on stimulus-to-pair travel time; sizes the capture via `N = D + V + T - k(low)`. Raise it when the crop reports a window off the **BACK**. **This is an assumption, not a bound** — the true value is distance/CV, and neither term is known at config time. It has already been too small twice: originally `T`, which failed on a fibrosis run; now `2T`, which failed at density 0.5 (measured travel 414 ms). Heavy fibrosis conducts far slower than a clean patch, so **a dense substrate needs a larger allowance**, and because density is drawn per simulation the failure is a *tail event* — a config that ran fine yesterday can fail today on a different draw. Over-estimating costs solver time; under-estimating costs the run. FB-36 replaces it with a derived value. |
| `activation_position.low` | float 0..1 | (required if block present) | Smallest fractional activation position. **Sizes the capture** — the smaller it is, the more signal a window needs after the activation, so the longer the solver runs. |
| `activation_position.high` | float 0..1 | (required if block present) | Largest fractional position. Kept below 1: the front cannot be extended, so a far-back position fills the window with flat pre-activation baseline. |
| `activation_position.seed` | int | `dataset.master_seed` | Seeds the position generator, which is stateful and owns its own stream. |
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
| `mix.master_seed` | int | 0 | Mixer RNG seed. Independent of `dataset.master_seed`. |
| `mix.show_progress` | bool | true | Progress bar over the mixing loop. |
| `mix.description` | str | `""` | Stamped into the mixed bank's metadata. |

#### Shipped model cards

`backend.model` picks both the parameterisation **and the cell model** — the
card's `type` says which membrane the backend integrates.

| Card | Cell model | Targets | What it is |
|---|---|---|---|
| `af_remodelled_220ms` | `aliev_panfilov` | CV 80 cm/s, APD90 220 ms | The Phase-1.5 default. Phenomenological, two-variable, cheap. |
| `courtemanche_control` | `courtemanche` | CV 80 cm/s | Courtemanche-Ramirez-Nattel 1998 human atrial myocyte, control (un-remodelled) conductances. The ionic half of the SEP5 A/B. |

Both cards target the **same conduction velocity**, which is what makes a
comparison between them a comparison of *membrane models* rather than of two
unrelated tissues: the free variable left is upstroke morphology, and EGM
amplitude scales with `dV/dt`.

`courtemanche_control` states **no APD90 target**, and that is deliberate rather
than an omission. Aliev-Panfilov solves its time-scale constant from an APD
target; Courtemanche has no such constant and no closed-form inverse from the
ionic equations, so its APD is **measured and recorded** under `measured:`
instead. Card targets are therefore per-model partial. A Courtemanche card *may*
state an APD90 target — S18c's AF-matched card will, since its conductances are
chosen to reach one — and where it does, loading checks it against the card's own
`measured:` block rather than against a solve.

Two costs worth knowing before you generate with it:

- **It is far more expensive.** 21 state variables and gating exponentials per
  node against Aliev-Panfilov's 2, at a timestep the sodium current pins to
  0.02 ms.
- **It is authored at `dr = 0.25 mm`, which is under-resolved for it.** The
  upstroke spans about 1.9 mesh cells where monodomain practice wants 5-10. The
  card says so, and the machinery refuses rather than absorbing it: loading
  `courtemanche_control` against any other `geometry.dr_mm` raises, because the
  reference conduction velocity it solves through was measured at that pitch.

#### Retired keys — these error rather than being ignored

Each was removed because leaving it accepted-but-inert is how a config comes to
read as though it asked for something it did not get. The loader names the
replacement in the error.

| Retired key | Replaced by | Why |
|---|---|---|
| `run.ap_time_unit_ms`, `run.diffusion`, `run.membrane_eps`, `run.dt_model_units` | `backend.model` (a model card) | These are *solved* from physiological targets, not chosen. `time_unit_ms` in particular sets CV and APD90 in **opposite** directions, so hand-tuning it to fix one silently breaks the other — which is exactly what happened between June and August 2026. Two configs setting them independently would let two banks claim one parameterisation with different physics. |
| `activation.detection` | `activation_position.detection` | Shipped one level too high in S16a. Here `activation:` is the activation *source* — how the wave is launched — while detection belongs to the crop: `crop_traces` builds one windower from the preprocessor and the position generator. (iafdb-pipeline nests it under `activation:` because there that block *means* activation-based windowing; this side matches its curve and parameter names, not its block path.) |
| `output.also_emit_synthetic_bank` | nothing — both banks are always written | The `synthetic_bank` is not an optional extra; it carries the per-simulation generation config the ClassifierBank deliberately does not. `output.synthetic_bank` sets **where** it lands, never **whether**. |

#### When the crop says a window "does not fit"

The crop refuses rather than sliding the window, because sliding it would change
the realized position without saying so. The message names which end failed, and
**the two ends take different knobs** — reaching for the wrong one is the common
mistake:

```
pair 0: a window of 192 samples at realized position 0.2408 does not fit
inside a 653-sample capture. It runs off the BACK by 22 samples ...
(activation at 529)
```

- **off the FRONT** — the activation arrived *earlier* than the stimulus delay
  allowed for. Raise `activation.stimulus_delay_ms`. `run.travel_allowance_ms`
  does nothing here.
- **off the BACK** — the wave took longer to reach the pair than the travel
  allowance covers. Raise `run.travel_allowance_ms`, past
  `activation index − stimulus delay`.

**Reproduce before you conclude it is fixed.** Fibrosis density is drawn *per
simulation*, so a back failure is usually a **tail event**: the run that failed
drew an unusually slow substrate. Changing the seed, `n_simulations` or the
density range redraws every substrate and the tail case simply does not recur —
which looks like a fix and is not. The hazard is a long generation run dying
partway through with the solver time already spent.

**A back failure is also worth reading as physics, not only as sizing.** Divide
the stimulus-to-pair distance by `activation index − stimulus delay` to get the
effective conduction velocity. If that lands far below the card's target — the
example above works out to 4-6 cm/s against a calibrated 83 cm/s — the substrate
is close to percolation rather than merely fibrotic, and raising the allowance
buys a bank of tissue that conducts an order of magnitude slower than diseased
atrium. Fibrosis is modelled as **insulating holes** (replacement scar), so
`substrate.density_range` values near the 0.5 cap remove enough myocardium to
nearly disconnect the mesh. See FB-36.

#### The detection curve is not recorded in the bank (FB-35)

`activation_position.detection.curve` decides **where each window is cut**, so it changes
the stored waveform — and neither `ClassifierBank` nor `synthetic_bank` has a
field to record it in. Recording it properly is an egm-contracts change flowing
into most of the constellation, which ships as its own wave (FB-35); it is
deliberately *not* stuffed into `backend_metadata`, which describes the
simulator's capture and would read as authoritative about something it does not
know.

**Until FB-35 lands, `output.description` is the record, maintained by hand.**
Two banks generated with different curves are otherwise indistinguishable from
their contents. The run summary prints the resolved curve for exactly this
reason — paste it into the description:

```
Wrote classifier bank:    ../banks/synthegm_v1.classifier.h5
Wrote synthetic bank:     ../banks/synthegm_v1.synthetic.h5
  N simulations:          100
  N traces:               2000
  Label policy:           global_density
  Detection curve:        botteron_envelope
  By label:               healthy=812, fibrotic=1188
```

A probe run adds two lines, reporting the snap once and stating plainly how
many simulations the bank ended up with:

```
  Probe grid:             13 offsets, 38..153 samples (p 0.1990..0.8010), snapped by at most 0.0026
  Probe simulations:      13 (one per offset, shared seed)
```

A run with no `activation_position` block does not crop, so it has no curve and
the summary has no such line.

#### The positional-sensitivity probe

`activation_position.grid` switches the run into **probe mode**: one solve,
emitted as **one logical simulation per crop offset**. Ships as
`examples/synthegm_probe.yaml`.

**It is a diagnostic bank, not training data.** It answers one question — how
much does a model's output move when the *only* thing that changes is where the
activation sits in the window? Every simulation in a sweep shares one substrate
draw, one electrode height and one seed, which is what makes the resulting curve
attributable to the offset rather than to a different draw. The corollary is
that the traces are near-duplicates of one another by construction: train on
them and the model sees one simulation repeated `n_points` times, so **do not
mix a probe bank into a training corpus.**

> **The bank reports N simulations where the config asked for one.**
> `dataset.n_simulations: 1` counts **solves**; a 13-point grid writes 13
> `simulation_id` values, all carrying the same `seed`. Two consequences to know
> before you read one:
>
> - **The shared `seed` is what identifies a sweep.** The schema carries it per
>   simulation and does not require it to be unique, so equal seeds across
>   consecutive `simulation_id`s is the signature of one probe run.
> - **Patient-aware splitting would treat each grid point as a separate
>   patient.** `patient_id` is `str(simulation_id)`, so a splitter would happily
>   put offset 0.2 in train and offset 0.3 in test — the same waveform either
>   side of the split. Harmless *because a probe bank is never training data*,
>   and stated here so it is known rather than discovered.
>
> The alternative — one simulation holding every offset — was tried first and is
> not available: `pair_index` is a foreign key into that simulation's
> `electrodes.pairs`, and egm-studio's loader joins the ClassifierBank to its
> theta companion on `(simulation_id, pair_index)` and **raises when either
> side's key repeats**. A bank with 260 traces under 20 pair indices is
> unloadable by the one consumer it exists for.

Four things worth knowing before you read one:

- **The grid is snapped to the sample lattice, and the snapped value is the
  grid of record.** A window is placed at `s = round(t_a - p(T-1))`, so the
  realized position equals the requested one only when `p(T-1)` is an integer.
  At `T = 192` the divisor is 191, which is prime — a grid stated in round
  fractions lands on none of the representable positions. So each point is
  snapped to `k = round(p(T-1))` at config load, and `k/(T-1)` is what the
  sweep requests and what the `activation_position` column stores. The
  requested-versus-snapped difference is under half a sample. Two points that
  snap to one `k` are an error naming the pair, not a silent de-duplication.
- **Detection runs once per pair, not once per grid point.** Re-detecting per
  window would put the detector's jitter onto the axis the study reads off. The
  probe detects once on the source trace and places every offset by exact
  integer shift from it — so two grid points hold the same waveform at a known
  sample lag.
- **The probe inherits the run's detection curve** (`activation_position.detection`),
  never a private default. A sweep characterising a bank through a different
  detector would be measuring two things at once.
- **Every pair is swept, and the trace axis is ordinary.** Each grid point is a
  complete simulation of `n_pairs` traces, so `pair_index` runs `0..n_pairs-1`
  and `activation_position` is one value per simulation. There is no
  pair-subset knob: selecting pairs would need a per-trace pair mapping, which
  is what made the first version of this bank unloadable. Select pairs when you
  analyse the bank instead.

Sizing needs no special handling: the capture is bought from the grid's smallest
snapped offset and the stimulus delay from its largest, which is the ordinary
rule with a fixed range instead of a sampled one. A grid point that does not fit
raises the usual FRONT/BACK diagnostic naming the grid point, rather than
clipping — a clipped point would put a kink in the study's x-axis that nothing
in the bank explains.

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
    # Membrane parameters are NOT on RunConfig — they are solved from
    # physiological targets and live on a cell model. Omitting `cell_model`
    # loads the default card (af_remodelled_220ms); pass one explicitly with
    #   from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card
    #   cell_model=load_model_card("af_remodelled_220ms", dr_mm=0.25, dr_model_units=0.25).solved
    run_config=RunConfig(trace_duration_ms=192.0, output_fs_hz=1000.0),
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
