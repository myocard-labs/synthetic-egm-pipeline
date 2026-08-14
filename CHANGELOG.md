# Changelog

All notable changes to `synthetic-egm-pipeline` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Model parameters are solved from physiological targets, not hardcoded
  (S38b, CL-173/174/176/178).** `simulate.calibration.calibrate` takes a
  conduction velocity, an APD90 and an anisotropy ratio and returns the four
  knobs the solver eats. The knobs were previously module constants and
  untouched Finitewave defaults — a standing violation of the rule that any
  parameter a generated bank depends on lives in config, and the reason the
  June calibration could be wrong for two months with nothing to contradict it.
  **Named model cards** (`backend.model: af_remodelled_220ms`) carry three
  blocks: the physiological `targets`, the `solved` knobs, and what a
  simulation `measured`. Loading one **re-runs the solve and raises** if the
  recorded knobs no longer match — so a future change to `calibrate` cannot
  quietly put two different physics under one parameterisation name. The solve
  is analytic precisely so that guard can run on every load rather than only in
  CI. Cards resolve by shipped name or by path beside the config, and **partial
  overrides are refused**: setting `run.ap_time_unit_ms` alongside a card is an
  error, because the whole value of a name is that it means one thing.
  Banks record the **resolved contents**, not the path.
  **Verified by round-trip** — solve, simulate, measure, assert the measurement
  returns the targets. That test would have failed in June. A companion check
  asserts no secondary deflection survives **inside the cropped 192-sample
  window at the smallest activation position** (CL-178), which is where the
  defect actually lived; the original measurement was taken on the raw capture
  and left the window arithmetic implicit.

- **Our own pseudo-EGM kernel** (`backends/finitewave/egm_kernel.py`), replacing
  Finitewave's `ECG2DTracker` in the production path. **`egm`, not `ecg`** — this
  computes the extracellular potential at *intracardiac* electrode positions,
  which is an electrogram; upstream's surface-lead naming framed a two-day
  investigation around the wrong mental model.
  **No physics changes here.** The kernel reproduces stock 0.9.3 exactly — same
  `1/r²`, same axis handling, same absent prefactor — and generated banks are
  byte-identical. It exists so the two arithmetic defects it enables fixing (an
  electrode-coordinate transpose, and a `1/r²` weight paired with a Laplacian
  source) can each land as their own attributable change.
  Vendoring rather than the alternatives: upstream's fix is on an unreleased
  branch that restructures the package around a numba/jax/mlx abstraction, and
  computing φ_e ourselves would mean holding the whole V_m history — roughly
  504 MB per simulation at the production geometry, which is why the streaming
  tracker was chosen originally.

- **Controlled-position cropping, part 2: the crop itself (SEP2 + SEP10).** Each
  bipolar trace is now cut to a `T`-sample window with its **detected**
  activation at a position sampled per window, and the realized position is
  written to the `synthetic_bank`'s `activation_position` column. Per trace, not
  per simulation — the wavefront sweeps the grid, so one per-simulation offset
  would control the position for a single pair and leave it uncontrolled for the
  rest. Routed through egm-signal's `SingleActivationWindower`, which is
  `window_train` with a train of one: the same function the IAFDB side uses, so
  the window geometry cannot drift between corpora.
  **The simulation is now arranged to make that possible**, which is what the
  step actually turned on. A stimulus delay `D = round(p_high·(T−1))` puts the
  activation far enough into the capture that a window fits in front of it —
  guaranteed for any patch size or conduction velocity, since travel time is
  non-negative. The capture then runs `N = D + V + T − round(p_low·(T−1))`,
  where `V` is an assumed travel allowance defaulting to `2T` and overridable
  via the new `run.travel_allowance_ms`. Full derivation in
  `docs/simulation_theory.md`.
  `activation_position` ranges now match iafdb-pipeline's `[0.4, 0.6]`:
  windowing the two corpora differently would make position itself a
  "which corpus is this" cue.
  **SEP10 rides along** — the anchored arm is `[0.5, 0.5]`, the same class with
  its range collapsed, so both arms of the A/B are configuration rather than
  code paths.

- **Controlled-position cropping, part 1: the position policy and the capture it
  requires (SEP2).** A new optional `activation_position:` block configures
  egm-signal's `UniformPositionGenerator`, and the capture is sized from it so a
  `T`-sample window placed around the activation always has signal behind it.
  Sizing is driven by the range's **lower** bound — a small `p` puts the
  activation early in its window and so demands the most signal after it. At the
  Phase-1.5 defaults (`T = 192`, `p ∈ [0.25, 0.75]`) the solver runs 335 ms to
  yield a 192 ms trace.
  The block is **opt-in, with no default**: the position policy sets the
  positional structure of every bank a run writes, and neither arm of the §8.9
  A/B is safe to fall into silently. `examples/synthegm_v1_anchored.yaml` shows
  the fixed arm — the same class with the range collapsed to a point.
  *Cropping itself is not wired yet;* this step sizes the capture and the trace
  is still the first `T` samples. The window is cut in the next step.

### Changed

- **`anisotropy_ratio` default 3.0 -> 2.0, and APD90 51 ms -> 220 ms (CL-176).**
  Atrial working myocardium is only weakly direction-dependent — Hansson 1998
  measures the right-atrial free wall at 88 +/- 9 cm/s with 74-81 cm/s across
  four propagation directions — and the high ratios in the literature belong to
  specialised bundles. At 2.0 the transverse velocity lands near 41 cm/s, inside
  the usual 30-50; at 3.0 it was forced to 26.5, below it.
  The APD target is **220 ms and must exceed the 192 ms trace duration**:
  repolarisation leaves the cropped window only if `APD > T(1-p)`, so `APD >= T`
  is the unconditional guarantee across the position range. An earlier proposal
  of 180 ms — plausible from the AF literature, which quotes short-cycle rates
  we do not simulate — fails that rule for any `p < 0.0625`.
  **Every generated bank changes.** In fibrotic tissue the anisotropy ratio
  affects propagation regardless of fibre orientation, because the wave
  diffracts around every hole; the along-fibre invariance that holds for a plane
  wave does not survive a substrate.
  All six example configs restated `ap_time_unit_ms` and `anisotropy_ratio`
  explicitly, so changing the constants alone would have been cosmetic — the
  same trap as the `trace_duration_ms` default in S12. They now name a card
  instead.

- **A short capture now raises instead of zero-padding.** The old fallback
  padded "so the bank stays uniform" — uniform in the worst way, since the pad
  is perfectly flat and always at the tail, giving every short trace exactly the
  positional regularity that varying the crop position exists to remove. Under
  correct sizing the case is unreachable; if it fires, the sizing is wrong and
  manufacturing data would hide that.
- **`run.trace_duration_ms` must give a sample count that is a multiple of 64**,
  rejected at config load with the nearest valid lengths named. The alternative
  is an N-simulation run that completes, writes a bank, and fails only when
  egm-classifier tries to train on it (CL-112).
- **`RunConfig` separates the capture from the trace.** `trace_duration_ms` is
  what lands on disk; `capture_duration_ms` (via `effective_capture_duration_ms`)
  is how long the solver runs. They were one number until cropping needed them
  to differ.

### Fixed

- **`geometry.anisotropy_ratio` had never done anything (CL-172).**
  `_configure_anisotropy_2d` assigned `D_al` / `D_ac` to the **model**;
  Finitewave reads them off the **stencil**. Python created two attributes that
  no reader ever consulted, so the requested ratio was silently discarded and
  the realized value was always the stencil's built-in `D_al = 1, D_ac = 1/9`.
  Measured 3.093 for requested 1.0, 3.0 **and** 6.0 alike. The v0.2.0 changelog
  claimed this helper made the value prescriptive; that claim is **retracted** —
  the v0.1.0 behaviour it described ("~3.05 regardless") was still exactly what
  happened.
  **Nothing generated is wrong.** Every config has always requested 3.0 and the
  accidental value was 3.09, so the knob was inoperative rather than mis-set,
  and this fix is **bit-identical at the shipped default** — `D_al = 1,
  D_ac = 1/ratio²` reproduces the stencil defaults exactly at ratio 3. That is
  asserted as its own test, since it is the claim the step's safety rests on,
  and it is why this ships ahead of the recalibration that changes every trace.
  **Which axis stays fixed is a deliberate change of meaning.** `D_al` is pinned
  at 1 and the whole ratio goes into `D_ac`, so raising the anisotropy slows the
  transverse axis and leaves the longitudinal one alone. The old docstring
  described holding the *geometric mean* fixed, which would have made this knob
  shift the along-fibre velocity as a side effect — and the along-fibre axis is
  what the literature quotes and what we calibrate against. The `base_diffusion`
  argument is gone with it: absolute scale belongs to `model.D_model`, and
  Finitewave already offers three multipliers on one coefficient without us
  adding a fourth.
  **The invariance is a plane-wave property only.** Along-fibre propagation
  rides a tensor entry that is 1.0 at every ratio, so two banks differing only
  in the ratio are bit-identical *on homogeneous tissue*. Add fibrosis and the
  wave diffracts around every hole, sampling the transverse entry continuously —
  measured on a fibrotic 40 mm patch, along-fibre propagation at ratio 1 vs 3
  differs by **more** than the same change measured across the fibres. So in the
  substrate we actually generate, the ratio matters whatever the fibre
  orientation.

- **The distance weighting was `1/r²` where the source term wanted `1/r` (CL-166).**
  Stock Finitewave 0.9.3 divides by `d`, the **squared** grid distance, and never
  takes a square root — an inverse-square weight applied to a Laplacian source.
  That is one term from each of two equivalent formulations: `1/r²` pairs with
  the *gradient* form `∫∇V_m·∇(1/r)` (what openCARP computes), while the
  diffusion increment we actually sum is the *Laplacian*, which pairs with `1/r`.
  Restoring the `sqrt` is the fix. The missing `1/(4πσ_e)` prefactor is now
  applied as well, in this kernel and in `compute_phi_e`.
  **Not a judgement call.** Upstream reached the same conclusion independently on
  their unreleased `solvers` branch — `sqrt` restored, `distance_power` defaulting
  to 1 — and Finitewave's own class docstring said *"the inverse of the distance"*
  the whole time.
  **What it changes.** On clean tissue: amplitude 4.1× smaller (median
  peak-to-peak `1.08e-01 → 2.62e-02`, mostly the prefactor), waveform correlation
  with the old output 0.961, detected activation indices unchanged. A scale and a
  mild re-weighting toward near nodes, not a new morphology — expected for a
  near-field measurement where the nearest sources already dominate.
  `distance_power` is exposed (default `1.0`) so pre-fix banks stay reproducible
  for comparison, and is deliberately **not** a config field: a debugging knob,
  not a generation mode.
  **`compute_phi_e` stops being decorative.** It has been thoroughly unit-tested
  since Wave 1 and never called by the pipeline — which is how two kernels came to
  disagree on the physics unnoticed. It is now a cross-check against the
  production kernel, on isotropic unmasked tissue only, since the two form
  different Laplacians (plain 5-point vs Finitewave's anisotropic, myocardium-
  masked stencil). A companion test asserts the comparison *fails* at
  `distance_power = 2`, so it cannot silently stop guarding the exponent.

- **Electrode coordinates and fibre orientation were transposed (CL-169 + CL-170).**
  Finitewave uses "x" for mesh axis-0 throughout; we use it for axis-1. Anything
  handed across that boundary in physical `x`/`y` had to swap, and neither did.
  **Electrodes** — the grid was reflected across the diagonal. Since bipolar
  pairs separate along `x`, that left every pair *perpendicular* to a `left`
  wavefront, so the near field cancelled and healthy tissue read as silent. On a
  clean 40 mm patch, changing only the stimulus edge: `left` 1.35e-06 →
  **1.08e-01**, `top` 1.61e-01 → 1.34e-06 — exactly mirrored.
  **Fibres** — `fibers[...,0]` acts along axis-0, so `fiber_angle_rad = 0` ran
  fibres along `+y` while the spec and every config said `+x`. At
  `anisotropy_ratio = 3` this put the fast conduction axis 90° from the intended
  one, which corrupts any conduction-velocity measurement taken along it.
  **Stimulus edges were already correct** and are unchanged: they are named by
  index (`top` = a strip at low `i`), so there was no physical direction to
  mistranslate.
  This retires the earlier reading that a planar wave over uniform tissue is
  degenerate and that the healthy class was physically meaningless — that was
  the transpose, not physics.

- **A clean intermediate is now joinable (CL-143 + B13).** In an inline-mix run
  the clean ClassifierBank named the *mixed* run's `synthetic_bank` under an id
  derived from its own, so the id matched no artifact and egm-data's
  `join_traces_with_simulations` refused it. Refusing was correct —
  `simulation_id` restarts at 0 in every bank, so a permissive join would pair
  traces with another run's config. The clean bank now gets **its own id base**
  (`output.clean_intermediate_bank_id`, auto-derived when omitted) and **its own
  `synthetic_bank`**, written beside it as `<stem>.synthetic.h5`. A mix run with
  `output.clean_intermediate` therefore writes **four** files.
  Matching the ids alone would not have been enough: a `synthetic_bank` stores
  `traces/signal`, so a shared θ file would have returned **mixed** waveforms
  for traces joined as clean.
- **The mixer re-points the θ companion's path, not just its id.** Rewriting
  only the id was survivable while both banks shared one θ file; with the clean
  bank now owning its own, an inherited path would aim the mixed bank at the
  clean θ artifact.
- **Standalone `synthegm-mix` no longer emits a θ companion entry.** It cannot
  write a `synthetic_bank` (2.0's per-simulation config is not recoverable from
  a ClassifierBank's per-trace metadata), so it previously left a mixed-derived
  id pointing at the *clean* θ file — the same id-versus-file divergence as
  CL-143, one CLI over. Absence is now the answer, matching the rule the noise
  columns already follow: a field a run did not produce is omitted, never faked.

### Changed

- **Wave-2 dependency re-pin** — egm-contracts `v0.6.0 → v0.6.1`, egm-data
  `v0.6.0 → v0.6.2`, egm-signal `v0.2.0 → v0.4.0` (the SIG1 release carrying
  `SingleActivationWindower`, which SEP2 calls).
- **`__version__` is read from installed distribution metadata** instead of a
  hardcoded literal (CL-117). The literal had drifted to `"0.2.0"` against a
  `v0.3.0` tag, and it is stamped into every bank as `producer_version` — so
  **every bank written before this fix names a version that never produced
  it**. Banks generated from here on carry the true version; older ones cannot
  be trusted on that field and should be regenerated if provenance matters.
- **Default `trace_duration_ms` is 192 ms, was 200** (CL-112, design §8.1). At
  1 kHz that is T = 192 samples, and T must be a multiple of 64: egm-classifier's
  1D MobileViT halves the sequence six times, so an off-grid length fails
  outright rather than degrading. **All five `examples/` configs set this
  explicitly and have been updated too** — changing only the default would have
  left every example generating unusable banks. Test fixtures now derive their
  trace length from the same constant, so the suite exercises the shipped value.
- **`numpy` capped at `>=1.26,<2.5`** (CL-118) — a deliberate, project-lead-blessed
  exception to "never cap a runtime dependency": numpy 2.5's PEP-695 generic
  stubs are unparseable by mypy at our `python_version = "3.10"` floor. It is a
  type-checking constraint, not a runtime one; lift it when the floor moves.

### Changed — Wave 1 (`synthetic_bank` 2.0)

- **BREAKING — `synthetic_bank` 2.0 migration** (Phase 1.5 Wave 1; egm-contracts
  v0.6.0 + egm-data v0.6.0). Generation parameters move out of flat per-trace
  columns into typed, `type`-discriminated objects stored **once per simulation**
  (geometry / cell_model / substrate / activation / electrodes / backend /
  label_policy), plus a bank-scoped θ-spec. `traces/` collapses to the signal,
  the two foreign keys, an integer label and the noise provenance. **1.1 banks
  are not readable** — regenerate; there is no migration path by design.
- **Both banks are now written on every run**, joined on `simulation_id`: the
  ClassifierBank stays a source-agnostic ML artifact (signal + label + key), and
  the `synthetic_bank` carries θ and per-simulation provenance.
  `output.also_emit_synthetic_bank` is **retired** — a config still setting it
  raises a `ConfigError` rather than being silently ignored.
  `output.synthetic_bank` is optional and defaults to a sibling path.
- **The two banks now take distinct stable IDs derived from one base.** Both
  previously received the *same* `ArtifactId` — harmless while the synthetic
  bank was an optional sibling view, but a collision now that both are always
  written and the phase manifest keys artifacts by ID. The ClassifierBank keeps
  the base; the `synthetic_bank` gets a `theta` marker in the descriptive name
  (before any date, so the ID stays inside the `ArtifactId` grammar).
  `output.bank_id` sets the **base** for both, so one override keeps the pair in
  step rather than letting them drift apart.
- **Generation parameters removed from the ClassifierBank.** Per-trace
  `fibrosis_density_requested`, `fibrosis_density_realized`, `electrode_row`,
  `electrode_height_mm`, `stim_edge` and `sim_seed` are gone, along with ~14
  bank-level generation keys (geometry, electrode grid, substrate ranges, cell
  model, backend). All are recoverable per-simulation from the `synthetic_bank`
  through `simulation_id`. They were the same flat per-trace columns the 2.0
  restructure removed from the other artifact — keeping the copy here preserved
  both the duplication and the drift risk. The bank now carries identity
  (`simulation_id`, `pair_index`, `patient_id`), the label, and — **only when
  the mixer ran** — `snr_db` / `noise_record` / `noise_channel`; a clean bank
  omits them rather than writing NaN, which would read as "mixed, SNR unknown".
  Bank-level metadata keeps `producer` / `producer_version` (reproducibility;
  `synthetic_bank` 2.0 has nowhere to record them), `description`,
  `trace_duration_ms`, and the `label_policy` **identity**.
- **A ClassifierBank now names the `synthetic_bank` it belongs with.** Its
  `banks` list gains a `synthetic_generation_params` companion entry carrying
  the θ bank's id, its path, and `join_key = simulation_id`. Previously nothing
  in either artifact recorded the pairing, so a consumer had to be told which
  two files went together. **Unambiguous only for a single-run bank** —
  concatenating two would give two origin and two companion entries with no way
  to pair them; the general fix is Phase-2 work.
- **`bank_path` stops claiming a source file that doesn't exist.** The entry
  describing a bank's *own* traces now carries the sentinel `<local>` instead of
  a path: the traces originate here, there is no source bank. This fixes two
  wrong values — a clean run named its own not-yet-written output, and a
  noise-mixed run named a clean bank whose traces differ from the file's.
  `<local>` rather than `""` so a blank stays available as a bug signal, and
  angle brackets specifically because they are illegal in Windows filenames and
  so cannot collide with a real path. Companion paths (noise bank, θ bank) are
  now **relative when the target sits inside the bank's own directory tree** and
  absolute otherwise — relative only where it actually buys portability.
- **The join key is `simulation_id` everywhere.** The producer's direct-write
  path previously wrote `sim_id` into `ClassifierTrace.trace_metadata` while
  egm-data's converter wrote `simulation_id`, so the same artifact type carried
  a differently-named key depending on which code wrote it.
- `SimulationResult` gained `specs` — the realized strategy specs a simulation
  ran with. A widening of the concrete result type; the four strategy Protocols
  are unchanged (see `project/architecture.md` → Guardrail 2).
- Re-pinned `egm-contracts v0.5.3 → v0.6.0` and `egm-data v0.5.0 → v0.6.0`.

### Removed

- `build_synthetic_bank_from_classifier` and
  `write_noise_mixed_synthetic_bank_from_classifier`. Schema 2.0's
  per-simulation config cannot be reconstructed from a noise-mixed
  ClassifierBank's per-trace metadata, so the inline mixer path now builds the
  bank from the in-memory `DatasetResult`. **Standalone `synthegm-mix` no longer
  emits a `synthetic_bank` at all** — it is a post-process over an existing
  bank, not a synthetic run, and asking it for one is a configuration error
  rather than a degraded output.

### Fixed

- The backend's reported version and solver timestep now land in the
  `synthetic_bank`'s typed `version` / `dt_model_units` fields instead of being
  dropped into the generic `params` bag.

### Development

- mypy type-checks `tests` as well as `src`, and CI pins `ruff==0.15.17` to match
  the pre-commit hook.
- Real-Finitewave tests are marked `slow` and deselected by default, so the
  MockBackend suite still runs in about a second. Run them with `pytest -m slow`.

### Previously unreleased

- Re-pin `egm-signal v0.1.0 → v0.2.0` — align on the current egm-signal (v0.2.0 is purely
  additive; surfaced by the S8-7 integration smoke test, which installs one consistent
  egm-signal across the whole constellation).
- Fix a stale `hybrid` reference in the `pyproject.toml` header comment (the noise-mixed
  output was still described as `<name>.hybrid.h5`).

## [0.3.0] — 2026-06-28

### Added

- **Stable cross-artifact bank IDs** (egm-contracts v0.5.0 / egm-data v0.4.0). Every bank
  carries an egm-contracts `ArtifactId`: the clean ClassifierBank + optional SyntheticBank
  derive `tbank_synthetic_<cell_model>_<date>`; the noise-mixed bank gets a `_noise_mixed`
  variant. Derivation + validation live in `ids.py`; the mixer's noise-source provenance
  entry carries the noise bank's real id (read from the iafdb noise run-record sidecar).
  Overridable via `output.bank_id` / `mix.noise_bank_id`.

### Changed

- **"hybrid" → "noise-mixed"** rename across code identifiers, examples, and prose (the
  noise-conditioned synthetic bank is not a Sánchez-style hybrid dataset).

## [0.2.0] — 2026-06-19

First real release after python-template personalization: a Phase-1 Finitewave simulator,
a bandpass-domain noise mixer, and two YAML-driven CLIs.

### Added

- **Simulator core** — four strategy Protocols + Phase-1 concretes (`Patch2DGeometry`,
  `UniformRandomFibrosis`, `PlanarEdgeStimulus`, `CenteredGrid2D`), pseudo-EGM math
  (`compute_phi_e` / `bipolar_from_unipolar` / `downsample`), label policies
  (`GlobalDensityLabel` / `LocalDensityLabel`), the `SimulationBackend` Protocol, and the
  `FinitewaveBackend` (the only place that imports `finitewave`), driven by `run_single`
  and the `generate_dataset` N-sim orchestrator.
- **Bank assembly** — in-memory builders (`build_classifier_bank_from_dataset`,
  `build_synthetic_bank_from_dataset`, `build_synthetic_bank_from_classifier`) + thin
  egm-data write wrappers. ClassifierBank is the default output; SyntheticBank is opt-in.
- **Mixer** — additive bandpass-domain overlay with per-trace SNR sampling
  (`mix_classifier_bank`, `snr_scale`, `sample_noise_for_length`), the bandpass primitive
  from `myocard-egm-signal`, and a full per-trace provenance trail.
- **Two CLIs** — `synthegm-generate-dataset` (N-sim, optional inline mixer) and
  `synthegm-mix` (standalone), both YAML-config-driven; five example configs.
- **Tests + docs** — 106 tests (with a `MockBackend` so the suite runs in ~1 s), plus
  `docs/usage.md`, `docs/simulation_theory.md`, and `docs/mixer_theory.md`.

### Dependencies

Pins `egm-contracts v0.2.0`, `egm-data v0.2.0`, `egm-signal v0.1.0`.

[0.3.0]: https://github.com/myocard-labs/synthetic-egm-pipeline/releases/tag/v0.3.0
[0.2.0]: https://github.com/myocard-labs/synthetic-egm-pipeline/releases/tag/v0.2.0
