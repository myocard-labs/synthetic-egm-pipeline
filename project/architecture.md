# synthetic-egm-pipeline — architecture and design rationale

Internal design doc for people building / maintaining the synthetic
producer. The public surface is documented in `docs/usage.md`; this doc
explains the why.

> **For future contributors (human or agent):** read this document
> *before* any non-trivial code change in this repo. The architecture
> is designed so an eventual swap of the simulator backend
> (Finitewave → openCARP / TorchCor) is a contained refactor; that
> property depends on the three guardrails in the [Guardrails](#guardrails)
> section staying intact. Don't violate them without coming back here
> and updating the doc.

## Where the package sits

```
┌────────────────────────┐  ┌───────────────────────┐  ┌────────────────────────┐
│ myocard-egm-contracts  │  │  myocard-egm-signal   │  │   myocard-egm-data     │
│   (format schemas)     │  │  (numpy + scipy DSP)  │  │  (HDF5 I/O, datasets)  │
└──────────┬─────────────┘  └──────────┬────────────┘  └──────────┬─────────────┘
           │                           │                          │
           │                           │                          │
           │ schema-version contract   │ bandpass / etc.          │ write_classifier_bank
           │ via egm-data writers      │                          │ write_synthetic_bank
           │                           │                          │ read_noise_bank_hdf5
           │                           │                          │
           └─────────────┬─────────────┴────────────┬─────────────┘
                         │                          │
                         ▼                          ▼
                ┌────────────────────────────────────────┐
                │      synthetic-egm-pipeline            │
                │           (this repo)                  │
                │                                        │
                │  simulate/specs       — Protocols      │
                │  simulate/result      — public types   │
                │  simulate/runner      — orchestrator   │
                │  simulate/pseudo_egm  — Okenov forward │
                │  simulate/label_policy— LabelPolicy    │
                │  simulate/dataset     — N-sim orchest. │
                │  simulate/builders    — pure-mem bank  │
                │                          assembly      │
                │  simulate/storage     — write wrappers │
                │                          (build → disk)│
                │  backends/finitewave  — concrete impl  │
                │  mixer/               — noise overlay  │
                │  cli/_config + 2 CLIs — YAML-driven    │
                └────────────────┬───────────────────────┘
                                 │
                                 ▼
                         on-disk artifacts (default):
                           <name>.classifier.h5    ← labeled, primary
                         optional:
                           <name>.synthetic.h5     ← rich per-trace metadata
                                 │
                                 ▼
                 ┌─────────────────────────────────────┐
                 │   downstream consumers              │
                 │     - egm-classifier (training)     │
                 │     - egm-viewer (Inspection tab)   │
                 │     - notebook analysis             │
                 └─────────────────────────────────────┘
```

synthetic-egm-pipeline is a **producer**, like iafdb-pipeline. It owns the
Finitewave-driven simulation, the Okenov pseudo-EGM forward calculation,
the LabelPolicy mechanism, and the additive-noise mixer. It pulls
shared DSP from `egm-signal`, writes through `egm-data`, and stamps
schema versions from `egm-contracts`.

## Folder layout

```
src/myocard_synthetic_egm_pipeline/
├── __init__.py
├── constants.py                  ← Phase-1 numeric defaults
├── ids.py                        ← stable cross-artifact id derive/validate/resolve
├── simulate/
│   ├── __init__.py               ← re-exports public API
│   ├── specs.py                  ← 4 strategy Protocols + concretes (pure data)
│   ├── result.py                 ← RawSimulationResult, SimulationResult
│   ├── pseudo_egm.py             ← Okenov 2024 forward + bipolar pairing
│   ├── label_policy.py           ← LabelPolicy Protocol + concretes
│   ├── runner.py                 ← run_single orchestrator
│   ├── dataset.py                ← generate_dataset (N-sim)
│   ├── builders.py               ← pure-in-memory bank assembly
│   │                                (DatasetResult → ClassifierBank / SyntheticBank)
│   └── storage.py                ← thin write wrappers (build → egm-data writer)
├── backends/
│   ├── __init__.py               ← SimulationBackend Protocol
│   └── finitewave/
│       ├── __init__.py
│       └── backend.py            ← FinitewaveBackend + private adapters
├── mixer/
│   ├── __init__.py
│   ├── mixing.py
│   └── storage.py
└── cli/
    ├── __init__.py
    ├── generate_dataset_cmd.py
    └── mix_cmd.py
```

The `backends/` subdirectory is where every line that imports `finitewave`
lives. Everywhere else in the tree, the only thing you'll find is plain
numpy, scipy, and the strategy Protocols from `simulate/specs.py`.

## The five Protocols

`simulate/specs.py` ships four strategy Protocols. `backends/__init__.py`
ships one Backend Protocol. Together they form the contract.

### `GeometrySpec` (Protocol)

Describes the tissue domain. Phase 1 ships `Patch2DGeometry` (2D
rectangular patch). Future: `AtrialMesh3D` (path to an unstructured mesh
file), `Cylinder3D`, etc.

### `SubstrateStrategy` (Protocol)

Describes how to populate the tissue with fibrosis or other substrate
variation. Phase 1 ships `UniformRandomFibrosis(density)`. Future:
`InterstitialFibrosis`, `PatchyFibrosis`, `CompactFibrosis`,
`HeterogeneousMix` (multi-type).

### `ActivationSource` (Protocol)

Describes how to initiate the wave. Phase 1 ships
`PlanarEdgeStimulus(edge)`. Future: `PointStimulus(position)`,
`PacingTrain(period, n_beats)`, `S1S2Protocol(s1_interval, s2_interval)`.

### `ElectrodePlacement` (Protocol)

Describes where the recording electrodes sit and which pairs form
bipolar leads. Phase 1 ships `CenteredGrid2D(rows, cols, spacing_mm,
height_mm_range)`. Future: `SubregionGrid2D`, `EndocardialSurface3D`,
`PentaRay`, `LassoCatheter`.

### `SimulationBackend` (Protocol)

Takes the four strategy specs plus a `RunConfig` and returns a
`RawSimulationResult` carrying per-electrode unipolar traces +
substrate mask + electrode positions + provenance metadata. Phase 1
ships `FinitewaveBackend`. Future: `OpenCARPBackend`,
`TorchCorBackend`.

The backend Protocol is small — one `simulate()` method. Internally
the backend does whatever it needs to (build tissue, install fibrosis
pattern, attach stimulus, run the AP, compute φ_e per electrode).
Backend-internal types stay backend-internal — they're not exposed in
the Protocol surface.

## The two result types

```python
@dataclass(frozen=True)
class RawSimulationResult:
    """What a backend returns — per-electrode unipolar traces, pre-bipolar."""
    unipolar_traces: np.ndarray          # (T_capture, n_electrodes) at fs_capture_hz
    fs_capture_hz: float                 # backend's native capture rate
    substrate_mask: np.ndarray           # (n_i, n_j) — 1 healthy, 2 fibrotic
    substrate_mask_dr_mm: float          # spatial resolution of the mask
    electrode_positions_mm: np.ndarray   # (n_electrodes, 3) — Cartesian mm
    bipolar_pairs: tuple[tuple[int, int], ...]  # (a_idx, b_idx) per pair
    substrate_realization_metadata: dict[str, Any]
    backend_metadata: dict[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    """What the runner exposes — bipolar traces ready for labeling."""
    bipolar_traces: np.ndarray           # (n_pairs, T_samples_at_output_fs)
    fs_hz: float                         # output sample rate (typically 1 kHz)
    bipolar_pair_midpoints_mm: np.ndarray # (n_pairs, 3) — for locality calcs
    substrate_mask: np.ndarray           # (n_i, n_j) — for locality calcs
    substrate_mask_dr_mm: float          # spatial resolution of the mask
    electrode_positions_mm: np.ndarray   # (n_electrodes, 3) — for locality calcs
    bipolar_pairs: tuple[tuple[int, int], ...]
    substrate_realization_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
```

The runner consumes `RawSimulationResult.unipolar_traces`, forms
bipolar pairs via `pseudo_egm.bipolar_from_unipolar`, downsamples to
the output sample rate, and produces `SimulationResult`. The
V_m → φ_e forward calc lives **inside the backend** rather than in the
runner — Finitewave's built-in `ECG2DTracker` already implements the
Okenov/Plonsey pseudo-EGM formula, so forcing the runner to redo it
would waste a forward pass. Future backends that don't ship native
pseudo-EGM (openCARP's "phie recovery" is listed as under-testing
upstream) import the shared
:func:`~myocard_synthetic_egm_pipeline.simulate.pseudo_egm.compute_phi_e`
helper to do the same calc on their V_m field. The substrate mask is
forwarded so a `LabelPolicy` can compute neighborhood statistics
without re-simulating.

`SimulationResult` is the **public output type** of the producer
pipeline. Every `LabelPolicy`, every storage writer, and every offline
analysis script operates on this type.

## Labeling

Labels are computed **inside the producer**, not downstream. A
`LabelPolicy.apply(result)` takes a `SimulationResult` and returns
`(labels: np.ndarray, labels_dict: dict[int, str])` — the same shape
egm-data's existing `label_fn` takes.

```python
class LabelPolicy(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def name(self) -> str: ...

    def apply(self, result: SimulationResult) -> tuple[np.ndarray, dict[int, str]]:
        ...
```

(See the [Protocol attribute style](#protocol-attribute-style) note
below for why `type` and `name` are declared as read-only `@property`
rather than mutable attrs.)

Concrete policies shipped Phase 1:

- `GlobalDensityLabel(threshold)` — global fibrotic density > threshold
  → 1, else 0. Preserves v0.1.0 behavior.
- `LocalDensityLabel(radius_mm, threshold)` — for each bipolar pair,
  count fibrotic vs healthy nodes within `radius_mm` of the pair's
  midpoint; label 1 if density > threshold, else 0. Addresses the
  global-density failure mode at low overall density where bipolar
  signal sees only the local neighborhood.

Future policies (`FibroticTypeLabel`, `NeighborhoodCompositionLabel`,
etc.) drop into `simulate/label_policy.py` without orchestrator
changes.

## Two outputs, different purposes, joined by `simulation_id`

> **Phase 1.5 / v0.4.0.** Through v0.3.0 the SyntheticBank was an
> optional sibling behind an `also_emit_synthetic_bank` flag that
> defaulted off. The `synthetic_bank` v2.0 restructure changes what the
> artifact *is*, and with it the decision. Recorded here as the settled
> design; the flag retires when SEP12 lands.

A synthetic generation run writes **both** banks. They are not a
primary and an optional extra — they are two artifacts with different
jobs, keyed to each other by `simulation_id`:

- **`ClassifierBank`** — the **ML artifact**, deliberately
  *source-agnostic*: signal, label, and the `simulation_id` key.
  Nothing about *how* the signal was generated. That is what lets
  egm-classifier consume synthetic and IAFDB banks through one code
  path, and it is why θ does **not** go here.
- **`synthetic_bank`** — the **θ / provenance artifact**: the per-simulation
  typed per-function config (geometry · cell_model · substrate ·
  activation · electrodes · backend · label_policy) plus the bank-scoped
  `generation_params` θ-spec. This is what egm-studio's STU1 / STU4 /
  STU5 read to recover θ.

Crucially the synthetic bank is **not the ClassifierBank's source
bank** — it is a parallel record of the same run, not an upstream
artifact it was derived from. Consumers join the two on
`simulation_id`.

**Why the label still runs inside the producer.** Unchanged from
Phase 1, and still the reason the ClassifierBank can stay this thin:
the `LabelPolicy` runs at write time with full in-memory access to the
simulation state, so a compact integer label crosses the boundary
instead of the metadata a downstream re-computation would need. Future
label policies (locality at arbitrary radii, multi-type
classification) drop in as new `LabelPolicy` classes without a schema
bump on either bank.

**Accepted cost: the trace signal is stored twice.** Both banks carry
the same waveforms. This was weighed and accepted — the banks are
small and gitignored, and de-duplicating them would mean making one
bank reference the other's storage, which reintroduces exactly the
source-relationship the split exists to avoid. Tracked as **FB-11** in
`intracardiac-platform/project/feature_backlog.md`; de-dup deferred.

**Flag mechanics (this repo's call).** `output.also_emit_synthetic_bank`
is **retired** rather than defaulted to true. A flag that can turn the
θ artifact off is a flag that can silently break the T4 thread — and,
in a project whose whole narrative is per-component change control,
"generate a bank with no recoverable generation provenance" is not an
option worth offering. A config still carrying the key raises a
`ConfigError` naming the change rather than being silently ignored.
`output.synthetic_bank` may be omitted; it then derives as a sibling of
`output.classifier_bank`.

**Standalone `synthegm-mix` is the one exception.** Run against a bare
ClassifierBank on disk it has no access to the generation config, so it
cannot write a v2.0 synthetic bank and does not try — it is a
post-process on an existing bank, not a synthetic *run*. Asking it for
one is a config error, not a degraded output.

## Stable cross-artifact IDs

Since v0.3.0 (egm-contracts v0.5.0 / egm-data v0.4.0) every bank the
producer writes carries a stable cross-artifact ID — an egm-contracts
`ArtifactId` the intracardiac-platform phase manifests + future
provenance graph key on. `ids.py` owns the derive / validate / resolve
logic; it imports no backend code (Guardrail 1).

- **Clean path.** `build_classifier_bank_from_dataset` (and the
  SyntheticBank builder) derive `tbank_synthetic_<cell_model>_<date>`
  from the run's cell model and stamp it on the ClassifierBank's own
  `id`, its source entry, and every trace. Synthetic banks are always
  labeled training banks, so the role prefix is always `tbank_`.
- **Noise-mixed path.** `mix_classifier_bank` gives the noise-mixed bank a
  `_noise_mixed` variant id. The mixed traces keep the **clean** source id
  (additive noise → the clean synthetic is the primary source). The
  appended "noise source" provenance entry carries the **noise bank's
  own** id, which the mixer reads from the iafdb noise run-record sidecar
  (`<noise>_run_record.json`) — full provenance fidelity, with a derived
  `nbank_iafdb_<date>` fallback. This is the one place the producer
  reaches across a repo boundary to resolve another producer's id, and it
  does so by file convention, not by importing iafdb-pipeline.
- **Single-sourced pattern + override.** The id *pattern* lives once in
  egm-contracts (`common.ArtifactId`); `ids.py` only composes + validates
  strings (the validation idiom mirrors egm-data's ClassifierBank-id
  check). Every id is overridable via the CLI config (`output.bank_id`,
  `mix.noise_bank_id`) or an orchestrator kwarg.

This is pure provenance plumbing — it does not touch the four strategy
Protocols, the `SimulationResult` type, or the Backend boundary, so none
of the three guardrails are affected.

## The Backend boundary

```python
class SimulationBackend(Protocol):
    name: str

    def simulate(
        self,
        geometry: GeometrySpec,
        substrate: SubstrateStrategy,
        activation: ActivationSource,
        electrodes: ElectrodePlacement,
        config: RunConfig,
        rng: np.random.Generator,
    ) -> RawSimulationResult: ...
```

The backend's job is "given four pure-data specs, run a simulation and
return per-electrode unipolar traces + spatial metadata." The runner
does everything else: form bipolar pairs, downsample, then later the
dataset orchestrator runs the LabelPolicy and hands to the writer.

`FinitewaveBackend.simulate()` internally:

1. Builds a `fw.CardiacTissue2D` from the `GeometrySpec` via
   `_build_tissue_2d`.
2. Configures anisotropy on the model from `GeometrySpec.anisotropy_ratio`
   via `_configure_anisotropy_2d` (sets `D_al` / `D_ac` so the realized
   CV ratio matches).
3. Realizes the `SubstrateStrategy` on the tissue mesh via
   `_apply_substrate_2d` → `_apply_uniform_random_fibrosis_2d` →
   `fw.DiffusePattern`.
4. Installs the `ActivationSource` via `_install_activation_2d` →
   `_build_planar_edge_stimulus_2d` → `fw.StimVoltageCoord2D`.
5. Installs `ElectrodePlacement` positions on an `fw.ECG2DTracker`,
   which computes per-electrode unipolar pseudo-EGMs internally
   (Finitewave's tracker implements the same Okenov/Plonsey formula
   our `pseudo_egm.compute_phi_e` does — for Finitewave we leverage
   the built-in; for backends without that infra we'd use our helper).
6. Runs the AP solver, returns `RawSimulationResult` with the unipolar
   traces from the tracker.

The `_2d` suffix on helpers is intentional: any helper touching a
2D-specific Finitewave type (`fw.CardiacTissue2D`, `Patch2DGeometry`,
2D mesh_shape) is suffixed so 3D parallels (`_build_tissue_3d`,
`_configure_anisotropy_3d`, ...) can land beside them later without
naming collisions.

Future `OpenCARPBackend` does steps 1-6 in its own way against
openCARP's API; if it doesn't ship a native pseudo-EGM tracker it
imports `compute_phi_e` from `simulate/pseudo_egm.py` to do the V_m → φ_e
step itself, then returns the same `RawSimulationResult` shape.

### Protocol attribute style

The four strategy Protocols + `LabelPolicy` declare their
discriminator attributes (`type` and, for LabelPolicy, `name`) as
read-only `@property`:

```python
@runtime_checkable
class GeometrySpec(Protocol):
    @property
    def type(self) -> str: ...
```

rather than as mutable class attrs (`type: str`). The choice is
deliberate: the Phase-1 concretes are all `@dataclass(frozen=True)`,
which makes the attribute read-only at runtime. A bare `type: str`
Protocol declaration means "must be a mutable attribute," which
frozen dataclasses don't satisfy under strict mypy. The `@property`
form means "must be readable," which any reasonable concrete (frozen
dataclass, plain class with a class attribute, normal property) does
satisfy.

This is a **Protocol widening** relative to the bare-attribute form —
any pre-existing satisfier under the bare form also satisfies the
`@property` form. So changing the Protocol surface this way is in
scope for Guardrail 2 (which locks against *breaking* changes, not
permissive widenings). When future concretes land, they can keep
using class attributes / Literals / `@property` — whatever is
ergonomic — and the Protocol won't reject them.

## Guardrails

These three rules are what make the eventual Option B refactor cheap.
Future contributors changing code in this repo MUST not violate them:

### Guardrail 1: zero `finitewave` imports outside `backends/finitewave/`

`simulate/specs.py`, `simulate/result.py`, `simulate/pseudo_egm.py`,
`simulate/label_policy.py`, `simulate/runner.py`, `simulate/dataset.py`,
`simulate/storage.py`, `mixer/*` — none of these should import
finitewave. Run `grep -rn "import finitewave\|from finitewave" src/`;
the only matches should be under `backends/finitewave/`.

Strategy classes (`UniformRandomFibrosis`, `PlanarEdgeStimulus`, etc.)
are pure-data specs. They hold their parameters as plain Python types
and expose them via attributes / properties. They do NOT take a
backend-native tissue object or model as a parameter. The backend
holds the adapter logic that turns a strategy into a backend-native
representation.

**Why it matters:** if a strategy imports finitewave, swapping the
backend means changing every strategy. The whole point of Option A's
boundary is that strategies stay the same when the backend changes.

### Guardrail 2: `SimulationResult` and the four strategy Protocols are public + concrete

`SimulationResult` is a frozen dataclass with explicit named fields.
It is NOT a Protocol. Backends produce it; the runner threads it
through `LabelPolicy` and the writer; every consumer (egm-viewer,
notebook analysis) reads it.

The four strategy Protocols (`GeometrySpec`, `SubstrateStrategy`,
`ActivationSource`, `ElectrodePlacement`) are public, named, and
locked. Adding new fields to a Protocol breaks every concrete
implementation; adding new concretes that satisfy the existing
Protocol is fine.

**Why it matters:** if the result type or the Protocol surface keeps
changing, the migration to Option B requires rebuilding the public
interfaces. Locking them now means Option B is a backend-internal
refactor, not an API-breaking redesign.

#### Widening: `SimulationResult.specs` (Phase 1.5, SEP12)

`SimulationResult` gained one field — `specs: SimulationSpecs`, a frozen
bundle of the four strategy specs the simulation actually ran with.

This is a **widening**, in the same category as the `@property`
Protocol change above, and in scope for Guardrail 2 for the same reason:
the guardrail locks against *breaking* changes. Every existing reader of
`SimulationResult` keeps working untouched, and the only producer of the
type — `run_single` — is the only thing that changed. The four strategy
Protocols were not modified at all.

**Why it was needed.** `generate_dataset` samples a fibrosis density, an
activation edge and an electrode height per simulation, builds
`UniformRandomFibrosis` / `PlanarEdgeStimulus` / `CenteredGrid2D` from
them, hands those to `run_single` — and then dropped them, keeping only
a handful of duck-typed scalars in `run_metadata`
(`fibrosis_density_requested`, `stim_edge`, `electrode_height_mm`, …).
That was sufficient while `synthetic_bank` 1.1 stored generation
parameters as flat per-trace columns. Schema 2.0 stores each generation
*function* as a typed, `type`-discriminated object once per simulation,
so the objects themselves have to survive the call. `run_metadata`'s
scalars remain as a lossy view of the same facts for existing readers.

Keeping the specs as objects rather than widening `run_metadata` further
is the point: a dict of scalars cannot express a point stimulus's
coordinate, an S1–S2 protocol's timings, or a cell model's conductance
scalings, and those are exactly what Phase 1.5+ adds.

### Guardrail 3: backend-internal types stay backend-internal

`fw.CardiacTissue2D`, `fw.AlievPanfilov2D`, `fw.ECG2DTracker`, etc.
never appear in the runner's signature, in `SimulationResult`, in any
strategy class, or in any public function exported from
`myocard_synthetic_egm_pipeline`. They live entirely inside
`backends/finitewave/`. If the runner ever needs to know about them,
that's a signal the Backend Protocol is missing a method — extend the
Protocol instead.

**Why it matters:** when Option B's capability split happens, the
"backend-internal types" are exactly what gets exposed as the
intermediate types of the capability methods. Keeping them confined to
the backend means the migration is a contained rewrite of one
package, not a sprawling type-leak fix.

## Option B migration plan

Option B is the same shape with the backend boundary refined into a
sequence of capability methods:

```python
class SimulationBackend(Protocol):
    def build_tissue(self, geometry: GeometrySpec) -> Tissue: ...
    def apply_substrate(self, t: Tissue, s: SubstrateStrategy, rng) -> SubstrateRealization: ...
    def install_recorder(self, t: Tissue, p: ElectrodePlacement) -> Recorder: ...
    def install_stimulus(self, t: Tissue, a: ActivationSource) -> None: ...
    def run(self, t: Tissue, r: Recorder, cfg: RunConfig) -> RawSimulationResult: ...
```

When to migrate: when a second backend concretely lands and the
Finitewave impl reveals a need to introspect intermediate state
(e.g. save V_m for visualization between substrate realization and
stimulus install, run multiple stimuli against one realized substrate
without rebuilding, swap recorder mid-experiment).

Expected migration cost, assuming the three guardrails have held:

1. Split `FinitewaveBackend.simulate()` body into five capability
   methods at its natural breakpoints. **0.5-1 day, mechanical.**
2. Define the four opaque intermediate types (`Tissue`,
   `SubstrateRealization`, `Recorder`, `RawSimulationResult` already
   exists). **Design work** — 0.5-1 day to pick what each type
   exposes.
3. Update the runner: one call becomes five. **0.5 day.**
4. Strategy code, label policies, the writer, the schema, the mixer,
   the CLIs: untouched.
5. Second backend implements the five-method Protocol from scratch.
   **Independent work.**

Total: 1-2 days of synthetic-egm-pipeline-side work, plus the second
backend's own implementation cost. Comparable to the egm-signal
extraction we did out of iafdb-pipeline — mechanical when the
boundaries are right.

What would kill this estimate: any drift on the three guardrails
during the lifetime of Option A. That's why the workflow rule at the
top of this doc exists.

## Module-by-module rationale

### `simulate/specs.py`

All four strategy Protocols + their Phase-1 concrete implementations
live in one file. Rationale: each concrete is small (~30-50 lines),
the Protocols themselves are smaller, and grouping them keeps the
Option A boundary visible — you can read the whole strategy surface
in one window.

When Phase 3 adds `InterstitialFibrosis` etc., they go in this same
file. If the file grows past ~500 lines, split by strategy family
(substrate.py, activation.py, electrode_placement.py, geometry.py).
Don't preemptively split.

### `simulate/pseudo_egm.py`

The Okenov 2024 pseudo-EGM forward calculation lives at the top of
the package (not inside any backend) because the math is
backend-agnostic. Given V_m on a grid + electrode positions, the
formula is the same regardless of which solver produced V_m.

It is a **helper for backends that don't ship a native pseudo-EGM
tracker**, not a step the runner unconditionally runs. The Phase-1
`FinitewaveBackend` uses Finitewave's built-in `ECG2DTracker`
(which implements the same Okenov/Plonsey formula in C-extension
code while the solver iterates) and produces `unipolar_traces` in
`RawSimulationResult` directly. The runner consumes those unipolar
traces, forms bipolar pairs (`bipolar_from_unipolar`), and
downsamples (`downsample`) — `compute_phi_e` doesn't appear in the
Finitewave hot path at all.

A future backend that runs a monodomain solver without a native ECG
tracker (e.g. an openCARP backend) imports `compute_phi_e` from this
module, runs it against the V_m field it captures, and stuffs the
resulting per-electrode traces into `RawSimulationResult.unipolar_traces`.
From the runner's perspective the boundary is the same regardless of
whether the backend used a tracker or our helper.

### `simulate/runner.py`

Thin orchestrator. Five steps after the backend returns:

```python
def run_single(
    *,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    backend: SimulationBackend,
    config: RunConfig,
    rng: np.random.Generator,
) -> SimulationResult:
    raw = backend.simulate(
        geometry=geometry, substrate=substrate, activation=activation,
        electrodes=electrodes, config=config, rng=rng,
    )
    # 1. Bipolar pairing at the capture rate.
    bipolar_capture = bipolar_from_unipolar(raw.unipolar_traces, raw.bipolar_pairs)
    # 2. Downsample to the output rate.
    bipolar_target = downsample(
        bipolar_capture,
        source_fs_hz=raw.fs_capture_hz,
        target_fs_hz=config.output_fs_hz,
    )
    # 3. Truncate/pad to exact target samples; reshape to (n_pairs, T) float32.
    # 4. Compute per-pair midpoints in physical mm.
    # 5. Stamp run_metadata with duck-typed scalars from the strategy specs.
    return SimulationResult(...)
```

`compute_phi_e` is NOT called here — the backend's
`RawSimulationResult.unipolar_traces` is the entry point for the
runner regardless of whether the backend used a native tracker (Finitewave)
or our helper.

### `simulate/dataset.py`

N-simulation orchestrator. Samples per-sim density from the density
distribution, picks a random edge stimulus, samples per-sim electrode
height, calls `run_single`, accumulates `SimulationResult` instances,
applies the `LabelPolicy` once at the end (after concatenation), hands
to the storage layer.

### `simulate/builders.py`

Pure in-memory bank assembly. Three public builders:

- `build_classifier_bank_from_dataset(dataset_result, config, bank_path, description)`
  → `ClassifierBank` (in-memory).
- `build_synthetic_bank_from_dataset(dataset_result, config, description)`
  → Pydantic `SyntheticBank` (in-memory).
- `build_synthetic_bank_from_classifier(noise_mixed_bank, description)`
  → Pydantic `SyntheticBank` (for the noise-mixed post-mixer case; reads
  mixer audit fields from `trace_metadata`).

Plus the shared per-trace metadata helper (`build_clean_trace_metadata`)
and the canonical bank-level provenance helper
(`build_bank_metadata_for_classifier_bank`).

No I/O — pair each builder with the matching `myocard-egm-data` writer
to land the result on disk. The storage layer (below) is thin wrappers
around build + write; the CLI's inline-mixer path calls
`build_classifier_bank_from_dataset` directly to get a bank in memory,
runs `mix_classifier_bank`, and writes the noise-mixed bank as the primary
output.

### `simulate/storage.py`

Thin write wrappers — each builds via `builders.py` then hands to an
`egm-data` writer. `write_classifier_bank_from_dataset` is the
default path; `write_synthetic_bank_from_dataset` is the optional
sibling-output path. Both go through `egm-data` writers; no `h5py`
imports in this repo.

### `simulate/label_policy.py`

`LabelPolicy` Protocol + Phase-1 concretes (`GlobalDensityLabel`,
`LocalDensityLabel`). Future policies drop in without changing the
runner or storage. Locality computation reads the `substrate_mask` +
`bipolar_pair_midpoints_mm` from `SimulationResult` — no schema
fields, no metadata packing.

### `backends/finitewave/backend.py`

The only file in the repo that imports `finitewave`. Holds
`FinitewaveBackend` + private adapter helpers
(`_build_tissue_2d`, `_configure_anisotropy_2d`, `_apply_substrate_2d`,
`_apply_uniform_random_fibrosis_2d`, `_install_activation_2d`,
`_build_planar_edge_stimulus_2d`, `_pick_capture_step`) that translate
the four strategy types into Finitewave's native API.

The `_2d` suffix marks adapters that touch 2D-specific Finitewave
types. `_pick_capture_step` is pure time/rate math and stays
unsuffixed. Future 3D backends drop `_3d` parallels alongside; the
public `FinitewaveBackend.simulate()` method dispatches on
`geometry.type` to pick the right helper family.

Private helpers prefixed with `_` to discourage downstream imports —
the public surface of this subpackage is `FinitewaveBackend` only.

### `mixer/`

Reads a noise_bank.h5 produced by iafdb-pipeline, samples noise per
trace at a target SNR, adds it to a clean ClassifierBank's signal
column, emits a noise-mixed ClassifierBank. Uses egm-signal's `bandpass`
for the band-pass-domain mixing step.

The mixer is its own orchestrator path — it can run standalone against
a previously-written clean ClassifierBank, or it can run inline at
dataset generation time (the `generate_dataset` CLI takes an optional
mixer config block).

### `cli/`

Two YAML-config-driven CLIs (matching iafdb-pipeline's convention):

- `synthegm-generate-dataset CONFIG.yaml [--overwrite] [--no-progress]`
- `synthegm-mix CONFIG.yaml [--overwrite] [--no-progress]`

The YAML configs specify which `LabelPolicy`, which
`SubstrateStrategy`, etc. by name; `_config.py`'s build functions
translate to the typed strategy instances.

#### Why YAML and not argparse

Phase 1 already has eight to ten substantive knobs per run (geometry,
substrate type + params, activation type + params, electrode placement
type + params, label policy type + params, backend choice, run config,
output paths, optional mixer block). Phase 2+ adds at minimum
multi-beat pacing protocols and the second substrate-type composition
API. Argparse would be a wall of `--flag` arguments that diff badly,
can't be checked into a repo cleanly, and force the user to remember
the order. YAML configs are checkable, diff-able, and let a single
file capture the entire experiment.

The other repos in the stack already use YAML for the same reason —
egm-classifier always has, iafdb-pipeline switched in `v0.2.0`. One
config format across the project.

#### Config shape

The actual YAML schema for both CLIs (every field, every default,
every accepted enum value) is documented in
[`docs/usage.md`](../docs/usage.md) and demonstrated in
[`examples/`](../examples/). The shape falls out of Option A's
four-strategy-Protocol design: each section (`geometry`, `substrate`,
`activation`, `electrodes`, `label_policy`, `backend`) carries a
`type:` discriminator plus per-strategy parameters; the typed config
dataclass in `cli/_config.py` validates each `type` at load time and
constructs the matching concrete from `simulate/specs.py`.

#### Strategy dispatch by name

Each `type` field selects which concrete strategy to instantiate;
unknown values fail loudly at config-load time (mirrors how
iafdb-pipeline's threshold-mode validation works). The typed config
dataclass in `_config.py` carries `Literal["finitewave", "opencarp", ...]`
fields and the build function maps each `type` + its peer keys to the
concrete strategy instance. Adding a new concrete strategy means
adding a new `type` value to the Literal, a new build case, and a new
strategy class — no Protocol changes.

## Open questions for future phases

- **Multi-pace protocols (Phase 2).** `ActivationSource` Protocol may
  need to grow `n_beats` or list-of-stimuli support. Likely an
  additive change — `PacingTrain(period, n_beats)` becomes a new
  concrete; older single-shot `PlanarEdgeStimulus` continues to work.
- **3D geometry (Phase 5).** `GeometrySpec` and `ElectrodePlacement`
  both grow 3D concretes. The
  `RawSimulationResult.unipolar_traces` field stays the same shape
  (per-electrode time series), but `substrate_mask` becomes
  3D-aware (and `LocalDensityLabel` grows a 3D-aware sibling that
  walks the mask as a tetrahedral neighborhood rather than a circle
  in mm). This is the most likely trigger for the Option B
  migration.
- **Multi-type substrate.** `SubstrateStrategy` may need to grow a
  composition API where a single simulation has multiple substrate
  types in different regions. Likely a new `HeterogeneousSubstrate`
  concrete that holds a list of `(region, strategy)` pairs.
- **Per-trace augmentation.** Augmentation today lives in `egm-data`'s
  `TraceTransform` at training time. If we ever want per-trace
  *simulation*-time augmentation (e.g. slight rotation, jitter on
  electrode positions), it's a `simulate/` concern, not a training
  concern.
