# Changelog

All notable changes to `synthetic-egm-pipeline` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
