# synthetic-egm-pipeline — roadmap

Future work only — shipped history lives in [`CHANGELOG.md`](../CHANGELOG.md). Internal
doc; public users read the README + `docs/usage.md`.

Work lands here as it's identified, sits in the **Backlog** until a phase-planning session
promotes it into a **Phase** cluster, then moves to the CHANGELOG once shipped. Phase
clusters mirror the science Project Phases in
`intracardiac-platform/project/project_plan.md` (this repo's old local "Phase 2 /
Phase 5" subsection numbering was retired to stop the two phase-number planes from
drifting). Items scheduled into cross-cutting Phase work carry a
`→ tracked at intracardiac-platform Phase X` annotation; the rest are component-internal.

## Phase 1.5 — sim-realism

### Courtemanche cell model

Swap Aliev-Panfilov → Courtemanche 1998 (human atrial ionic model) as a new model class in
`backends/finitewave/backend.py`. The strategy specs + `RawSimulationResult` shape stay the
same; only the AP-specific anisotropy helper needs a Courtemanche-aware variant (a
`_configure_anisotropy_2d_courtemanche` sibling or dispatch on a `cell_model` knob in
`RunConfig`). Strict realism upgrade — the model Sánchez 2021 uses.

### Additional activation sources (realism subset)

`PointStimulus(position_mm)` (focal source for spiral-wave studies), `S1S2Protocol(...)`
(re-entry inducibility), and **multi-edge stimulation** (Sánchez stimulates from three
sides; v1's `PlanarEdgeStimulus` uses a single edge). These need the polymorphic
`stimulation` schema (Phase 2, below). See [[reference-multi-beat-consensus]].

### Pluggable noise-selection strategy (mixer-side)

Today the mixer samples noise segments uniformly at random across the full noise bank
(`mixer/mixing.py::sample_noise_for_length`), breaking the within-patient noise-correlation
structure of real recordings. Add a fifth strategy Protocol (`NoiseSelectionStrategy`)
parallel to the four simulation ones, with `UniformRandomNoiseSelection` as the
current-behaviour default plus `PerSimPatientNoiseSelection` (each sim draws one IAFDB
patient; highest leverage) and `PerSimRecordNoiseSelection`. Config via a
`noise_selection.type` block inside `mix:`.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

## Phase 2 — multiclass severity

### Polymorphic `stimulation` schema (replace `stim_edge`)

The additional activation sources need a richer schema: replace `stim_edge` with a
polymorphic `stimulation` object (type discriminator + per-type params). A **coordinated
release** — egm-contracts v0.6.0+ + egm-data v0.5.x + synthetic-egm-pipeline v0.4.0+ —
bundled at the Phase 2 inflection (the "first big publish" per
[[project-publishing-timing]]; bundling avoids fragmenting the work across releases).

### `FibroticTypeLabel` — multi-class label policy

Multi-class label requiring the multi-type `HeterogeneousMix` substrate (Phase 3) to be
meaningful. A new class in `simulate/label_policy.py`; no Protocol / schema changes.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 2.

## Phase 3 — pattern classification

### Additional substrate strategies

`InterstitialFibrosis` (banded), `PatchyFibrosis` (discrete islands), `CompactFibrosis`
(solid scar), `HeterogeneousMix` (composed) — the four substrate types **are** the pattern
classes the classifier learns to distinguish. Each is a new concrete in `simulate/specs.py`
+ a `_apply_<name>_2d` adapter; no Protocol / schema / LabelPolicy changes (labels read the
realized mask, not the strategy type).

### `NeighborhoodCompositionLabel`

Fractional composition per fibrosis type in the pair's neighborhood, then thresholded —
needs the four Phase-3 substrate types.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 3.

## Phase 4 — multi-beat

### `PacingTrain(period_ms, n_beats)` activation source

Steady-state pacing protocol for multi-beat sequence classification. Also needs the
polymorphic `stimulation` schema (Phase 2).

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 4.

## Phase 7 — 3D geometry

### 3D atrial geometry

The most likely trigger for the Option A → Option B backend migration in `architecture.md`.
New concretes: `AtrialMesh3D` (GeometrySpec, path to a `.pts/.elem/.lon` mesh) and
`EndocardialSurface3D` (ElectrodePlacement). Likely a new `backends/torchcor/` (Finitewave
is finite-difference on regular grids; 3D unstructured meshes are TorchCor's wheelhouse) —
the strategy Protocols stay identical (Guardrail 2 holds). Phase 8 extends
`EndocardialSurface3D` into explicit 3D catheter geometry.

### Compatibility validator for strategy combinations

A pre-flight `validate_simulation_specs(...)` in a new `simulate/compatibility.py`, called
once after YAML load, that raises a precise `CompatibilityError` before any simulation.
Today an incompatible combo (e.g. `PlanarEdgeStimulus` on a 3D mesh, `LocalDensityLabel` on
a non-2D substrate) only fails at `simulate()` time, after the AP-solver startup cost.
Triggered when the second geometry type lands.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 7 (3D substrate geometry).

## Backlog (unscheduled — promoted into a phase at a planning session)

### Paper visualization data + opt-in `SimulationResult` export

`SimulationResult` already carries what a Sánchez-style substrate-with-electrodes overlay
needs (substrate mask + mm scale, electrode positions, bipolar-pair midpoints, per-pair
waveforms), so a notebook can already produce per-sim figures. Open question: which subset
of sims gets pickled to disk for paper reproducibility (saving every sim's full result is
feasible but not the default). Add an opt-in `output.save_full_results: list[int]` flag and
decide the figure set alongside the intracardiac-papers work.

> → Cross-cutting; track at `intracardiac-platform/project/project_plan.md` alongside the
> egm-studio + egm-classifier figure needs. Recipes live in intracardiac-papers; the figure
> CLI (`egm-studio-render`) ships in egm-studio; only the producer-side pickling lives here.

### `DistanceToNearestFibroticLabel` — regression target

Minimum distance from the pair's midpoint to a fibrotic node; a continuous regression
target. Component-internal — a different model arch (regression head), so not on any phase
critical path.

## Known issues

None open.

## Open architectural questions for later

- **Should `LabelPolicy.apply` take the substrate mask explicitly** rather than reading it
  off `SimulationResult`? Today the result bundles the mask in, so the result type knows
  about labeling — borderline cross-cutting. If a third use of the mask emerges (e.g. a
  visualization library reading it directly) the pattern self-justifies; if only LabelPolicy
  uses it, passing it explicitly is cleaner.
- **Should the producer ship its own visualization helpers (`viz.py`)?** Today notebooks
  plot from `SimulationResult` ad-hoc. If the paper figures become producer-emitted
  reproducibility artifacts, a `viz/` subpackage may be worth it.
- **Should `RunConfig` grow a `cell_model` discriminator?** One backend dispatching on
  `RunConfig` vs. separate backends per model. Decide when Phase 2 (Courtemanche) starts.

## v1.0 — what graduating Phase 1 means

Promoting to v1.0 needs: two backends in the wild (Finitewave + at least one other) so the
migration-cost story stops being theoretical; white-paper figures locked in with the
producer emitting everything they need without follow-up CLI hacks; and schema versions
stable across a release cycle. Until then, every minor bump may break the on-disk format
and consumers pin exact versions.

## Won't-do (out of scope, but documented to save the question)

- **No HDF5 I/O inside this repo.** Every writer lives in `myocard-egm-data`; an h5py import
  in this source tree is a smell.
- **No DSP primitives.** Filters, threshold strategies, calibration live in
  `myocard-egm-signal` — the mixer's bandpass is the only use today.
- **No model code.** No torch dependency, no `nn.Module`; egm-classifier consumes our output.
- **No real-data preprocessing.** IAFDB download / calibration / healthy-segment extraction
  lives in `myocard-iafdb-pipeline`; we consume its noise bank in the mixer but never call
  into it.
