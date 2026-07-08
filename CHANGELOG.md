# Changelog

All notable changes to `synthetic-egm-pipeline` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
