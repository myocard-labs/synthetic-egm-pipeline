# synthetic-egm-pipeline — roadmap

What's planned for future releases. Internal doc — public users see the
README and `docs/usage.md`. Phase numbering follows
``simulator_v1_spec`` (and the meta repo's ``project_plan.md``).

## v0.2.0 — shipped

Scope (recap, see `architecture.md` for the design rationale):

**Simulator core**

- Four strategy Protocols + Phase-1 concretes (`Patch2DGeometry`,
  `UniformRandomFibrosis`, `PlanarEdgeStimulus`, `CenteredGrid2D`)
  in `simulate/specs.py`. Protocols use the read-only `@property`
  attribute pattern so frozen-dataclass concretes satisfy them
  cleanly.
- `RawSimulationResult` (per-electrode unipolar traces + spatial
  metadata) + `SimulationResult` (bipolar traces + substrate mask
  for label policies) public dataclasses in `simulate/result.py`.
- `compute_phi_e` + `bipolar_from_unipolar` + `downsample` math
  helpers in `simulate/pseudo_egm.py` (public; used by future
  backends without native pseudo-EGM).
- `LabelPolicy` Protocol + `GlobalDensityLabel` + `LocalDensityLabel`
  in `simulate/label_policy.py`.
- `SimulationBackend` Protocol + `RunConfig` in `backends/__init__.py`.
- `FinitewaveBackend` in `backends/finitewave/backend.py` (the only
  backend Phase 1, the only place in the repo that imports
  `finitewave`). Two fixes from the legacy: anisotropy ratio plumbed
  to `D_al` / `D_ac`; substrate mask snapshotted before solver run.
  Adapter helpers `_2d`-suffixed where they touch 2D-specific
  Finitewave types.
- `run_single` thin orchestrator (`simulate/runner.py`) +
  `generate_dataset` N-sim orchestrator (`simulate/dataset.py`).

**Bank assembly**

- Pure in-memory builders in `simulate/builders.py`:
  `build_classifier_bank_from_dataset`,
  `build_synthetic_bank_from_dataset`,
  `build_synthetic_bank_from_classifier` + the shared per-trace
  metadata helper.
- Thin write wrappers in `simulate/storage.py` + `mixer/storage.py`
  (build → egm-data writer).
- ClassifierBank is the default on-disk output; SyntheticBank is an
  opt-in sibling via config flag.

**Mixer**

- `mixer/mixing.py`: `MixerConfig`, `snr_scale`,
  `sample_noise_for_length`, `mix_classifier_bank`. Additive
  bandpass-domain overlay; per-trace SNR sampled from the configured
  range; the bandpass primitive comes from `myocard-egm-signal`.
- `mixer/storage.py`: thin write wrapper for the hybrid
  Pydantic SyntheticBank sibling output.
- Provenance trail: per-trace `snr_db` / `noise_record` /
  `noise_channel` stamped into `trace_metadata`, plus a "mixer"
  `ClassifierBankMetaData` entry appended to `bank.banks`.

**CLIs**

- `cli/_config.py` — YAML loader + typed config dataclasses
  (`GenerateDatasetCLIConfig`, `MixCLIConfig`, `InlineMixConfig`) +
  per-section builder functions that translate YAML to concrete
  strategy / policy / backend instances.
- `synthegm-generate-dataset` — N-sim dataset CLI with optional
  inline mixer block.
- `synthegm-mix` — standalone mixer CLI against an already-written
  clean ClassifierBank.
- Both YAML-config-driven; `--overwrite` + `--no-progress` are the
  only flags.

**Example configs**

- `examples/synthegm_v1_baseline.yaml` (Phase 1 default clean
  corpus).
- `examples/synthegm_v1_hybrid.yaml` (same scope + inline mixer).
- `examples/synthegm_calibration.yaml` (4-sim deterministic
  calibration).
- `examples/synthegm_mix.yaml` (standalone mixer at default SNR
  range).
- `examples/synthegm_mix_fixed_snr.yaml` (collapsed SNR for ablation
  sweeps).

**Tests**

- 106 tests across `tests/test_specs.py` (28), `test_pseudo_egm.py`
  (15), `test_label_policy.py` (13), `test_builders.py` (12),
  `test_mixer.py` (15), `test_cli_config.py` (17),
  `test_runner.py` (6).
- A `MockBackend` fixture in `conftest.py` exercises the runner's
  post-processing without spinning up Finitewave; the full suite
  runs in ~1 s.

**Docs**

- `docs/usage.md` — external CLI + full YAML schema reference.
- `docs/simulation_theory.md` — step-by-step walk through one
  simulation run with code anchors per step + annotated paper
  references.
- `docs/mixer_theory.md` — analogous walk through the mixer math.
- `project/architecture.md` — locked Option A design, three
  guardrails, Option B migration plan, module-by-module rationale.
- `project/roadmap.md` — this file.
- Top-level `README.md` — pitch, install, quick-start, citation.

## v0.3.0+ — concrete next steps

### Compatibility validator for strategy combinations

A strategy combo can be incompatible in subtle ways:
``PlanarEdgeStimulus`` requires a geometry with well-defined edges
(works for ``Patch2DGeometry``; not for an arbitrary 3D atrial mesh
where "edge" isn't a single locus). ``CenteredGrid2D`` requires a 2D
geometry. ``LocalDensityLabel`` only knows how to walk a 2D substrate
mask today.

The current behavior is that each backend's dispatch raises a clear
``ValueError`` at ``simulate()`` time when it sees a combo it can't
handle. That's correct but late — the user has already paid the AP
solver startup cost. We want a pre-flight validator that runs at
config-load time and tells the user the offending combo before any
simulation starts.

Shape:

```python
# In a new simulate/compatibility.py
def validate_simulation_specs(
    *,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    backend: SimulationBackend,
    label_policy: LabelPolicy,
) -> None:
    """Raise CompatibilityError with a precise message if the combo is invalid."""
```

The CLI calls this once after YAML load, before
``generate_dataset``. Each strategy can register declared
"compatibility constraints" against the others — likely a small table
keyed on the four `type` discriminators. Triggered when the second
geometry type lands (3D arrives in Phase 5).

### Paper visualization data + opt-in mesh export

`SimulationResult` already carries everything a Sánchez-style
substrate-with-electrodes-overlay visualization needs:

- ``substrate_mask`` `(n_i, n_j)` int8 — color by ``mesh != 2``
  (healthy vs fibrotic).
- ``substrate_mask_dr_mm`` — mm scale for the axes.
- ``electrode_positions_mm`` — overlay dots.
- ``bipolar_pair_midpoints_mm`` — annotate which pair maps to which
  substrate region.
- ``bipolar_traces`` — per-pair waveform inset.

So a notebook can produce per-sim figures from any
`SimulationResult`. The open question is which subset of sims gets
saved to disk for paper reproducibility. Saving the full
`SimulationResult` for every sim (~26 kB substrate mask per sim ×
hundreds of sims) is feasible but not the default — most sims won't
end up in a figure.

Plan to revisit after the v0.2.0 refactor:

- Decide which figures the paper needs (substrate overlay, per-pair
  EGM panels, density-vs-classifier-output plots, etc.).
- For each figure, identify whether `SimulationResult` already has
  what's needed or whether the producer needs to emit something
  extra.
- Add an opt-in config flag like
  ``output.save_full_results: list[int]`` (sim_ids whose
  `SimulationResult` gets pickled to disk for the notebook to load).
  Default off; the paper-figure runs flip it on for a handful of
  sims.

Cross-cutting work item — track at the meta repo level in
``project_plan.md`` so it sits alongside the egm-viewer +
egm-classifier figure needs.

### Additional substrate strategies

- ``InterstitialFibrosis`` — banded patterns between myocyte bundles.
- ``PatchyFibrosis`` — discrete fibrotic islands of configurable size.
- ``CompactFibrosis`` — solid scar regions.
- ``HeterogeneousMix`` — multiple strategies composed (e.g. a patchy
  scar surrounded by interstitial border zone).

Each is a new concrete in `simulate/specs.py` + a new
`_apply_<name>_2d` adapter in `backends/finitewave/backend.py`. No
Protocol changes; no schema changes; no LabelPolicy changes (because
labels read the realized mask, not the strategy type).

### Additional activation sources

- ``PointStimulus(position_mm)`` — focal source for spiral-wave studies.
- ``PacingTrain(period_ms, n_beats)`` — for steady-state pacing
  protocols (the Phase-2 multi-beat work).
- ``S1S2Protocol(s1_interval_ms, s2_interval_ms)`` — for re-entry
  inducibility studies.

These need a richer ``stim_edge`` schema in
``synthetic_bank`` (the migration note in the current schema points
at this). The plan: replace ``stim_edge`` with a polymorphic
``stimulation`` object (type discriminator + per-type params) when
the second activation source lands. Coordinated bump:
egm-contracts v0.3.0 + egm-data v0.3.0 + synthetic-egm-pipeline
v0.3.0 in one release.

### Pluggable noise-selection strategy (mixer-side)

Today's mixer samples noise segments uniformly at random across the
full noise bank — see ``mixer/mixing.py::sample_noise_for_length``.
That maximises augmentation diversity but breaks the cross-pair
noise correlation structure present in real recordings (all bipolar
pairs from one IAFDB record share the same patient state, drug
state, ambient EM, electrode contact characteristics, etc.).

The current default is defensible for the Phase 1 per-trace 1D CNN
classifier (which has no cross-pair input) and matches Sánchez 2021's
approach, but becomes a real distribution-shift issue if a future
classifier consumes cross-pair signal OR if eval metrics get
stratified by noise source.

Plan: add a fifth strategy Protocol parallel to the four simulation
ones:

```python
class NoiseSelectionStrategy(Protocol):
    type: str
    def sample(
        self,
        *,
        noise_bank: NoiseBank,
        n_samples: int,
        sim_id: int,
        pair_index: int,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], str, str]: ...
```

Phase-1-equivalent default: ``UniformRandomNoiseSelection`` (preserves
current behaviour). Plausible concretes:

- **``PerSimPatientNoiseSelection``** — each sim draws one IAFDB
  patient_id uniformly; all pairs in that sim sample from that
  patient's records. Highest leverage. Patient parsed from
  ``noise_bank.traces.source_record`` (e.g. ``"iaf1_afw"`` → ``"iaf1"``).
- **``PerSimRecordNoiseSelection``** — tighter; one record per sim.
  Preserves intra-record temporal structure too.
- **``PerPairChannelMatchedNoiseSelection``** — speculative; ties
  synthetic bipolar pair index to IAFDB channel name (e.g.
  synthetic pair index 0 → CS12). The mapping is arbitrary on a 2D
  patch so the value is unclear; might be more meaningful once 3D
  anatomy lands.

Build cost: small (one Protocol + concretes + a
``_build_index_by_source_record`` helper in the noise bank wrapper).
Trigger to actually do it: when the classifier moves to multi-pair
input, OR when we want noise-stratified eval metrics.

YAML config gain: ``noise_selection.type: uniform_random`` block
inside ``mix:``, mirroring how substrate / activation / label_policy
discriminate on ``type`` today.

Tracked 2026-06-18 after a design discussion about whether to do
this in v0.2.0; deferred because the v1 per-trace classifier doesn't
consume cross-pair signal so the cost outweighs the benefit until
the classifier architecture changes.

### Additional label policies

- ``FibroticTypeLabel`` — multi-class. Requires the multi-type
  ``HeterogeneousMix`` substrate to be meaningful.
- ``NeighborhoodCompositionLabel`` — fractional composition per
  fibrosis type in the bipolar pair's neighborhood, then thresholded.
- ``DistanceToNearestFibroticLabel`` — minimum distance from the
  pair's midpoint to a fibrotic node; useful as a continuous
  regression target.

Each is a new class in `simulate/label_policy.py`. No Protocol
changes; no schema changes (labels are still int + a labels_dict).

### Phase 2 — Courtemanche

Swap the cell model from Aliev-Panfilov to Courtemanche 1998 (human
atrial ionic model). Lands as a new model class inside
`backends/finitewave/backend.py` — the strategy specs and
`RawSimulationResult` shape stay the same, only the AP-specific
helpers (currently `_configure_anisotropy_2d` reading
`model.D_al`/`D_ac`) need a Courtemanche-aware variant. Likely a
`_configure_anisotropy_2d_courtemanche` sibling or a small dispatch
on a new `cell_model` knob in `RunConfig`.

### Phase 5 — 3D atrial geometry

The most likely trigger for the Option A → Option B migration in
`architecture.md`. New concretes:

- `AtrialMesh3D` (GeometrySpec) carrying a path to a `.pts/.elem/.lon`
  triple or an equivalent unstructured-mesh reference.
- `EndocardialSurface3D` (ElectrodePlacement) — electrodes sampled or
  projected onto the endocardial surface.

New backend likely too — Finitewave is finite-difference on regular
grids; 3D unstructured meshes are TorchCor's wheelhouse. Add a
`backends/torchcor/` subdirectory then; the strategy Protocols stay
identical (Guardrail 2 holds), the backend interface stays one
method (Option A still in force).

## v1.0 — what graduating Phase 1 means

Promoting to v1.0 needs:

- Two backends in the wild (Finitewave + at least one other) so the
  migration cost story stops being theoretical.
- White-paper figures locked in; the producer emits everything those
  figures need without follow-up CLI hacks.
- Schema versions stable across a release cycle without breaking
  changes.

Until then, every minor bump may break the on-disk format; consumers
pin exact versions.

## Won't-do (out of scope, but documented to save the question)

- **No HDF5 I/O inside this repo.** Every writer lives in
  ``myocard-egm-data``. If this repo's source tree grows an h5py
  import, it's a smell.
- **No DSP primitives inside this repo.** Filters, threshold strategies,
  calibration scaffolding live in ``myocard-egm-signal``. The mixer's
  bandpass is the only place egm-signal gets used today; future
  shared signal-processing math lands there, not here.
- **No model code.** No torch dependency, no nn.Module. egm-classifier
  consumes our output; we don't consume its model.
- **No real-data preprocessing.** IAFDB download / calibration /
  healthy-segment extraction lives in ``myocard-iafdb-pipeline``. We
  consume that repo's noise_bank in the mixer but never call into it.

## Open architectural questions for later

- **Should `LabelPolicy.apply` take the substrate mask explicitly
  rather than reading it off `SimulationResult`?** Today the result
  bundles the mask in; that means the result type knows about
  labeling, which is borderline cross-cutting. If a third use of the
  mask emerges (e.g. visualization library reading it directly), the
  pattern self-justifies. If only LabelPolicy uses it, refactoring
  to pass it explicitly might be cleaner.
- **Should the producer ship its own visualization helpers
  (`viz.py`)?** Today notebooks plot from `SimulationResult` ad-hoc.
  If the paper figures become reproducibility artifacts the producer
  emits, a `viz/` subpackage with matplotlib helpers may be worth
  it. Lower priority than the validator and the per-radius density
  question above.
- **Should `RunConfig` grow a `cell_model` discriminator?** Today the
  backend hard-codes `AlievPanfilov2D` from Phase 1; Courtemanche
  (Phase 2) means either swapping in a different model class
  internally (one backend, dispatches on RunConfig) or having
  separate backends per model. Decide when Phase 2 starts.
