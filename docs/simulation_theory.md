# Simulation theory — what each step does to the signal

This document walks through what happens inside one
``synthegm-generate-dataset`` simulation, what each step represents
physically, and what it does to the signal that eventually lands in the
ClassifierBank. It is meant to be read by a developer who already knows
signal processing but is new to cardiac electrophysiology — every step
links back to the math and to a small number of papers worth reading
deeper.

The reading list at the bottom is annotated. Start with the three
**[anchor]** papers if you only read three.

## The pipeline at a glance

A single simulation goes through nine steps. Five live inside the
backend; four are backend-agnostic and live in the runner / storage
modules.

```
                    one simulation
   ┌──────────────────────────────────────────────────────┐
   │                                                      │
   │  1. Tissue + substrate setup    ┐                    │
   │  2. Activation source           │  inside backend    │
   │  3. AP solver runs              │  (Finitewave)      │
   │  4. Unipolar φ_e captured       │                    │
   │     at electrode positions      ┘                    │
   │  5. Pseudo-EGM theory (math)       (reference)       │
   │  6. Bipolar pairing             ┐                    │
   │  7. Downsample to 1 kHz         │  backend-agnostic  │
   │  8. Label assignment            │  runner            │
   │  9. Storage (ClassifierBank)    ┘                    │
   │                                                      │
   └──────────────────────────────────────────────────────┘

   what the classifier eventually sees:
     bipolar EGM trace  (n_pairs, T_samples)  at 1 kHz
     integer label      ∈ {healthy, fibrotic}
```

The figure below shows the same nine steps with the data type at each
hand-off:

```
Patch2DGeometry              ←──────┐
UniformRandomFibrosis        ←──────┤      strategy specs
PlanarEdgeStimulus           ←──────┤      (pure data)
CenteredGrid2D               ←──────┘
        │
        ▼
┌──────────────────────────────────┐
│ FinitewaveBackend                │
│   1. build 2D mesh               │
│   2. apply fibrosis              │
│   3. configure model             │
│      anisotropy + dt             │
│   4. install stimulus            │
│   5. install ECG2DTracker        │
│   6. run AP solver               │
│   7. capture per-electrode       │  φ_e(t, e) at fs_capture (~4 kHz)
│      unipolar φ_e from tracker   │
└──────────────┬───────────────────┘
             │
             ▼
        RawSimulationResult   (unipolar_traces, substrate_mask, ...)
             │
             ▼
┌──────────────────────────────────────┐
│ runner                                │
│   bipolar_from_unipolar(phi_e, pairs)│  → bipolar(t, p)
│   downsample(bipolar, 4kHz, 1kHz)    │  → bipolar(t', p)  at 1 kHz
└──────────────┬───────────────────────┘
               │
               ▼
        SimulationResult           (bipolar_traces, substrate_mask, midpoints, ...)
               │
               ▼
┌──────────────────────────────────┐
│ LabelPolicy.apply(result)         │  → int label per pair
└──────────────┬───────────────────┘
               │
               ▼
        ClassifierBank.h5 (egm-data writer)
```

## Step 1 — Tissue + substrate setup

**What the code does.** ``FinitewaveBackend`` reads the
``Patch2DGeometry`` spec (40 mm square at 0.25 mm spacing → 160×160 mesh
cells), builds a ``CardiacTissue2D`` with every interior cell marked
healthy (mesh value ``1``), sets a uniform fiber-orientation field, and
configures the model's diffusion tensor so the realized CV ratio
matches ``spec.anisotropy_ratio``. Then the ``UniformRandomFibrosis``
strategy walks every interior cell: for each cell, draw a uniform[0,1]
sample; mark the cell non-conductive (mesh value ``2``) if
``density > draw``.

**What this represents.** The 2D mesh is a flat patch of atrial wall.
Each cell is a small piece of tissue (0.25 mm × 0.25 mm) that the AP
solver will integrate forward in time. Mesh values are tissue type:

- ``0`` = boundary frame (Finitewave pads the array — these cells are
  not integrated)
- ``1`` = healthy myocardium — full AP cell model runs here, the cell
  is electrically excitable, currents flow into neighbours
- ``2`` = fibrotic, non-conductive — the AP cell model still runs but
  the cell does not transmit current to its neighbours

Anisotropy is the empirical fact that atrial myocardium conducts faster
along the fibre direction than across it. The published
along-to-across CV ratio for human atrium is roughly 2-3:1 (LA: see
[Hansson 1998], [Lemery 2007]). The ratio comes from the structure of
the syncytium: myocytes are elongated, gap junctions cluster at the
ends, so depolarising current spreads more efficiently along the cell's
long axis than across it. We bake this into the diffusion tensor with
``D_along = base · √ratio``, ``D_across = base / √ratio`` (since
$CV \propto \sqrt{D}$).

**What this does to the signal.** Two things, before anything else
happens:

1. **The fibrotic pattern carves "no-go" regions out of the mesh.** When
   a wave hits a fibrotic cell it cannot propagate further through that
   cell — the wave must go around. At low density (<10%) most of the
   wave finds an open path, the wavefront is slightly distorted but
   reaches every healthy region. At moderate density (~30%) the
   wavefront fractures into multiple smaller fronts that meet, collide,
   and create complex morphology. At high density (>60%) propagation
   fails: the wave cannot find a connected path of healthy cells across
   the patch ([Nezlobinsky 2021]).
2. **Anisotropy stretches the wavefront.** A wave moving along the
   fibre direction arrives at the far edge sooner than a wave moving
   across, so a planar stimulus from one edge produces an elongated
   ellipsoidal front rather than a circular one. Bipolar EGMs aligned
   along the fibre direction see sharper, faster activation; bipolars
   across the fibre see slower, broader activation.

The substrate mask itself is also kept around through every later
step. ``LocalDensityLabel`` reads it directly to compute per-pair local
density without re-running the simulation.

**Code:**
- Geometry data: `simulate/specs.py::Patch2DGeometry` (size, dr,
  fiber angle, anisotropy ratio).
- Substrate data: `simulate/specs.py::UniformRandomFibrosis` (density).
- Tissue construction: `backends/finitewave/backend.py::_build_tissue_2d`.
- Anisotropy plumbing: `backends/finitewave/backend.py::_configure_anisotropy_2d`.
- Substrate realisation: `backends/finitewave/backend.py::_apply_substrate_2d`
  → `_apply_uniform_random_fibrosis_2d`.
- Mask snapshot: `backends/finitewave/backend.py::FinitewaveBackend.simulate`
  (the `substrate_mask = tissue.mesh.astype(np.int8, copy=True)` line).

**Read deeper.**
- **[Nezlobinsky 2021]** — Diffuse fibrosis on a 2D patch; the
  density-vs-propagation regime map [anchor].
- **[Mitrea 2009]** — Optical mapping of fibrillation in fibrotic
  hearts. Real physiology to contrast with the idealised square patch.
- **[Hansson 1998]**, **[Lemery 2007]** — Empirical atrial CV
  anisotropy from human electroanatomic mapping; the source of the
  3:1 default.

## Step 2 — Activation source

**What the code does.** ``PlanarEdgeStimulus(edge)`` is a thin strip
of voltage applied to a chosen edge of the mesh at ``t=0``. The strip
is three cells thick by default. The CLI samples the edge uniformly
at random per simulation.

**What this represents.** The simplest possible model of a
propagating activation: an instantaneous, line-source pulse that raises
V_m above threshold on one strip of cells. In a real heart an
activation might come from sinus rhythm (from the SAN), from a focal
ectopic source (a small spot), or from re-entry (a self-sustaining
spiral). Phase 1 only models a planar wave from an edge because (a) it
is the simplest source that traverses the patch once cleanly, and (b)
the morphology of the resulting EGM is what we want the classifier to
learn from. Phase 2+ will swap this for ``PointStimulus``,
``PacingTrain``, and re-entry-protocol stimuli.

**What this does to the signal.** Sets the activation direction. A
planar wave from the ``left`` edge arrives at the electrodes from left
to right; a planar wave from the ``top`` edge arrives top to bottom.
Bipolar pairs aligned with the propagation direction see large,
biphasic morphology; pairs perpendicular to propagation see small,
fractionated morphology because both poles are excited near
simultaneously. By randomising the edge per simulation we prevent the
classifier from using activation direction as a shortcut feature —
fibrosis discrimination should generalise across propagation
directions, so we want directionally-balanced training data.

**Code:**
- Activation data: `simulate/specs.py::PlanarEdgeStimulus` (edge,
  voltage, strip thickness, time offset).
- Per-sim random edge sampling:
  `simulate/specs.py::random_edge` (called from
  `simulate/dataset.py::generate_dataset`).
- Stimulus install in the backend:
  `backends/finitewave/backend.py::_install_activation_2d`
  → `_build_planar_edge_stimulus_2d`.

**Read deeper.**
- **[Stewart 1999]** — Source-sink interpretation of why thin strips
  reliably initiate a propagating wave (the strip is wide enough to
  recruit downstream tissue but small enough not to behave like a
  point source).

## Step 3 — AP solver

**What the code does.** Finitewave runs the Aliev-Panfilov model
([Aliev 1996]) on the 160×160 mesh for ``trace_duration_ms /
AP_TIME_UNIT_MS`` model time units, with ``dt = 0.01`` model units per
integration step. AP_TIME_UNIT_MS = 1.97 ms is the calibration constant
that maps AP non-dimensional time → physical milliseconds (re-run the
calibration when ``dr`` or the model changes).

**What this represents.** The membrane potential of each cell evolves
in time according to a small system of ODEs (Aliev-Panfilov has two
state variables per cell: V and a recovery variable r), coupled
spatially via the diffusion term. This is the **monodomain**
reaction-diffusion equation:

$$
\frac{\partial V}{\partial t} = \nabla \cdot (D \nabla V) - I_{\text{ion}}(V, r)
$$

$$
\frac{\partial r}{\partial t} = G(V, r)
$$

In words:

- $I_{\text{ion}}$ is the cell model's ionic current — for AP this is
  a phenomenological polynomial that captures the basic excitable
  dynamics (resting → fast upstroke → plateau → recovery) without
  modeling individual ion channels.
- $\nabla \cdot (D \nabla V)$ is the diffusion term that couples
  neighbouring cells — depolarisation of one cell drives current into
  neighbours via gap junctions; we represent gap-junction conductivity
  with the diffusion tensor ``D``.
- For non-conductive cells (mesh ``= 2``) the diffusion term to/from
  that cell is zero — they are electrically isolated.

Phase 1 uses **Aliev-Panfilov** because:

- It is the simplest model that captures the excitable dynamics needed
  for wave propagation + collision behaviour.
- It runs fast on a laptop — a 4 cm patch for 200 ms takes seconds, not
  minutes.
- The pipeline shape (substrate → wave → V_m → EGM) does not depend
  on which cell model is plugged in; we can swap to a richer model
  later without redoing anything downstream.

The richer model for Phase 2 is **Courtemanche 1998**, a human atrial
ionic model with individually-modelled Na/K/Ca channels, gating
variables, and physiologically-tuned APD. It is far more expensive but
produces morphology that matches recorded atrial EGMs more faithfully.

**What this does to the signal.** Generates ``V_m(t, i, j)`` — a movie
of the transmembrane potential field across the patch. This is the
upstream signal that the pseudo-EGM step will project onto electrode
coordinates. Two things to notice about the V_m field:

1. **Each cell's V_m trace looks like a depolarisation-plateau-recovery
   sequence.** The fast upstroke of V_m at a cell happens when the
   activation wave arrives at that cell; the timing of the upstroke is
   the cell's local activation time. Bipolar EGM morphology is
   essentially the spatial derivative of the V_m field, captured by
   the pseudo-EGM forward calc — which is why the *timing* of the wave
   arriving at the two poles of a bipolar pair is what produces the
   sharp biphasic deflection.
2. **Fibrosis distorts the V_m field locally.** Near a fibrotic
   cluster, the wave has to go around. The arrival time at the
   downstream side of the cluster is delayed, the V_m field on either
   side of the cluster has visible asymmetry, and a bipolar pair
   straddling the cluster sees a fractionated waveform with multiple
   small deflections instead of one clean biphasic spike.

**Code:**
- Model construction + run:
  `backends/finitewave/backend.py::FinitewaveBackend.simulate`
  (steps 2-3 inside that method: `fw.AlievPanfilov2D()`,
  `model.t_max = ...`, `model.run()`).
- Per-run knobs:
  `backends/__init__.py::RunConfig` (trace duration, output rate,
  AP calibration constant, oversampling factor).
- AP solver internals: `finitewave.AlievPanfilov2D` (external lib).

**Read deeper.**
- **[Aliev 1996]** — The original Aliev-Panfilov phenomenological
  model [anchor].
- **[Courtemanche 1998]** — Human atrial ionic model; the
  Phase-2 substitute.
- **[Loewe 2014]** — AF-remodeled Courtemanche for the Phase-4
  rhythm work.
- **[Clayton 2011]** — A textbook-level review of computational
  electrophysiology models; useful to compare phenomenological vs
  ionic models.

## Step 4 — V_m → φ_e capture at electrode positions

**What the code does.** Finitewave's ``ECG2DTracker`` is instantiated
with the ``ElectrodePlacement``'s electrode coordinates (in mesh-index
units, ``coord_grid = coord_mm / dr_mm``). As the AP solver runs, the
tracker computes per-electrode unipolar pseudo-EGMs every
``tracker.step`` integration steps. The effective capture rate is
chosen so it's ~4× the output rate — i.e. ~4 kHz when the output bank
is at 1 kHz.

**What this represents.** Where the AP solver gives us the V_m field
on the mesh, what an electrode actually senses is the extracellular
potential φ_e at the electrode's position. Going from V_m (defined on
every mesh cell) to φ_e (defined at a few electrode positions) is the
**pseudo-EGM forward calculation** — the math is explained in detail
in Step 5 below.

For Finitewave-based simulations this step happens **inside the
backend**: ``ECG2DTracker`` implements the same Okenov/Plonsey formula
our :func:`compute_phi_e` does, but does it in C-extension code while
the solver is running, so it's both faster and cheaper than capturing
the full V_m field + running our own forward calc afterwards. The
backend returns per-electrode unipolar traces; ``compute_phi_e``
remains a public helper for *other* backends (TorchCor, openCARP) that
don't ship native pseudo-EGM.

**What this does to the signal.** Projects V_m (intracellular,
defined everywhere on the mesh) onto φ_e (extracellular, defined at
each electrode position). Time-discretises that projection at ~4 kHz so
the downstream downsampling step (#7) can use a clean integer-stride
decimation. The per-electrode unipolar trace at this point is what an
ideal point electrode would record at the chosen ``(x, y, z)``
position.

**Code:**
- Tracker install:
  `backends/finitewave/backend.py::FinitewaveBackend.simulate`
  (the `fw.ECG2DTracker(measure_coords=coords_grid)` block).
- Capture-rate selection:
  `backends/finitewave/backend.py::_pick_capture_step` (picks the
  integration-step stride so the achieved capture rate ≈
  `oversample × output_fs_hz`).
- Trace extraction at end of run:
  `backends/finitewave/backend.py::FinitewaveBackend.simulate`
  (the `unipolar = np.asarray(ecg_tracker.output, ...)` line).
- Electrode coords: `simulate/specs.py::CenteredGrid2D.sample` +
  `positions_mm` field.

**Read deeper.**
- **[Finitewave docs / source]** — Specifically ``ECG2DTracker``. The
  tracker takes ``measure_coords`` (electrode positions in mesh-index
  units) and a ``step`` attribute (how many integration steps between
  captures). Output is ``(T_capture, n_electrodes)``, ready to feed
  into bipolar pair subtraction.
- **[Okenov 2024]** supplement S2 — the formula the tracker is
  implementing.

## Step 5 — The pseudo-EGM forward calculation (theory)

**What the formula is.** The pseudo-EGM forward calculation is the
Plonsey/Barr extracellular-potential integral applied to a 2D mesh.
For each electrode at position $\vec r_e$, the unipolar
potential is

$$
\phi_e(\vec r_e, t) \;\propto\; \sum_{i, j \in \text{interior}}
                                 \frac{(D \nabla^2 V_m)_{i,j,t}}{r_{e, i, j}}
$$

where $r_{e, i, j}$ is the 3D Euclidean distance from electrode
$e$ to mesh node $(i, j)$ (with the electrode at height $z = h$
above the patch). Finitewave does this inside ``ECG2DTracker`` while
the AP solver runs (Step 4). Our `compute_phi_e` helper
implements the same formula in pure numpy for backends that don't ship
a tracker.

**Code:**
- Backend-agnostic helper:
  `simulate/pseudo_egm.py::compute_phi_e` (pure numpy, used by future
  backends without native pseudo-EGM; not called for Finitewave
  because ECG2DTracker already does the same math in C).
- Per-electrode trace consumption:
  `simulate/runner.py::run_single` (consumes
  `RawSimulationResult.unipolar_traces` regardless of which backend
  produced it).

**What this represents — the pseudo-bidomain approximation.**
The full **bidomain** model represents the heart as two co-located
continua — intracellular (where V_m lives) and extracellular (where
φ_e lives, and where the electrodes actually sense). The two are
coupled by the conductivity tensors of each domain.

A monodomain solver like Finitewave only integrates the intracellular
domain — it gives you V_m, not φ_e. To go from V_m to φ_e you either
solve a full bidomain problem (expensive) or use the **pseudo-bidomain
approximation** ([Plonsey 1969], [Plonsey 1988]): treat the
extracellular domain as an infinite, homogeneous, isotropic volume
conductor. Then the extracellular potential at any point ``r_e`` is the
solid-angle integral of the transmembrane current source ``I_m`` from
the intracellular domain:

$$
\phi_e(\vec r_e, t) = \frac{1}{4\pi \sigma_e}
                       \int \frac{I_m(\vec r, t)}{|\vec r_e - \vec r|} \, dV
$$

For us:

- The integral becomes a sum over mesh cells (we work on a discrete
  mesh).
- ``I_m`` is approximated by the discrete Laplacian of V_m (consistent
  with what Finitewave's own ECG tracker does internally; the diffusion
  step's change in V_m at each cell is the discrete equivalent of
  $D \nabla^2 V_m$).
- We deliberately drop the ``1 / (4π σ_e)`` normalisation because the
  classifier cares about *relative* amplitude across pairs (which
  carries morphology), not the absolute mV scale. If we ever need
  absolute mV we can scale this output downstream.

The Okenov 2024 paper applies exactly this forward calc on a similar
2D Finitewave-driven fibrosis substrate, then trains a CNN on the
resulting EGMs. Their formulation lives in the supplement, equation
S2.

**What this does to the signal.** Converts V_m (intracellular,
defined everywhere on the mesh) into φ_e (extracellular, defined at
electrode positions). Three properties of φ_e to keep in mind:

1. **φ_e is dominated by the cells closest to the electrode.** The
   ``1 / r`` kernel falls off with distance, so a 0.5 mm-high electrode
   primarily sees the ``I_m`` of the dozen-ish cells directly under
   it. This is exactly the receptive-field issue that motivates
   ``LocalDensityLabel`` — a bipolar pair "sees" only a small
   neighbourhood, so its label should be set by that neighbourhood, not
   by the global density.
2. **φ_e tracks the rate of change of V_m, not V_m itself.** The
   Laplacian kernel ensures φ_e responds to the *moving wavefront*
   rather than to a static depolarised region. This is why an EGM
   shows a brief spike at activation rather than a slow plateau.
3. **The electrode height matters a lot.** A 1 mm-high electrode sees
   a much broader (and smaller-amplitude) signal than a 0.2 mm-high
   electrode does. By sampling height uniformly in [0.2, 1.0] mm we
   produce a realistic distribution of contact quality across the
   training set.

**Read deeper.**
- **[Okenov 2024]** — Same forward formulation we use; trains a CNN
  to classify atrial fibrosis from simulated bipolar EGMs [anchor].
- **[Plonsey 1969]** — Foundational paper on the extracellular
  potential formulation; the volume-conductor integral that underlies
  the pseudo-bidomain approximation.
- **[Plonsey 1988]** — Specifically on the pseudo-bidomain
  approximation and when it's valid.
- **[Henriquez 1993]** — A clear and accessible derivation of the
  full bidomain equations; read this before reading Plonsey if the
  notation is unfamiliar.

## Step 6 — Bipolar pairing

**What the code does.** ``pseudo_egm.bipolar_from_unipolar(phi_e,
pairs)`` differences the unipolar φ_e traces between the two
electrodes of each pair: ``bipolar(t, p) = φ_e(t, a_p) - φ_e(t, b_p)``.

**What this represents.** Clinically, an EP catheter records bipolar
EGMs to suppress far-field signals (atrial activations recorded on a
ventricular catheter, distant chamber activity, etc.). The bipolar
subtraction is essentially a spatial high-pass filter — sources far
from the pair contribute roughly the same amount to both poles and
cancel in the difference; sources between the poles contribute
oppositely-signed to each pole and add to the difference.

**What this does to the signal.** Three effects:

1. **Suppresses far-field common-mode signal.** Activity at distant
   parts of the patch (say, a wavefront still propagating in the
   opposite corner) shows up equally at both poles of a small bipolar
   pair, so the difference is roughly zero. Local activation between
   the poles shows up oppositely on each, so the difference is large.
2. **Sharpens the timing of the activation deflection.** A unipolar
   φ_e at one electrode has a broad asymmetric morphology; the
   difference of two adjacent unipolars produces the characteristic
   biphasic sharp-peak-sharp-trough shape that EP physicians look for.
3. **Sensitive to wavefront orientation.** A wavefront parallel to the
   pair (both poles activated simultaneously) produces minimal
   bipolar signal; a wavefront perpendicular to the pair produces
   maximal signal. This is part of why we have 5 bipolar pairs per
   row — different pairs are at different orientations relative to
   the wave, so the classifier sees morphology across orientations.

**Code:**
- Pair definition: `simulate/specs.py::CenteredGrid2D.bipolar_pairs`
  (tuple of (a, b) electrode index pairs, within-row consecutive).
- Subtraction: `simulate/pseudo_egm.py::bipolar_from_unipolar`,
  called from `simulate/runner.py::run_single`.

**Read deeper.**
- **[Stevenson 2005]** — Practical introduction to EP signal
  interpretation; explains why bipolar > unipolar for local activation
  timing.

## Step 7 — Downsampling

**What the code does.** ``pseudo_egm.downsample(bipolar, source_fs_hz,
target_fs_hz)`` resamples each bipolar trace from the capture rate
(~4 kHz) to the output rate (1 kHz). Integer-stride fast path when
the ratio is exact; linear-interpolation slow path otherwise.

**What this represents.** Match the output rate to IAFDB (1 kHz) so
synthetic and real bipolar EGMs are length-comparable for the mixer
and the classifier. The synthetic side runs at higher rates internally
because the AP integration is most stable at small ``dt``; output at
1 kHz is plenty for bipolar EGM bandwidth (clinical bandpass for
bipolar atrial EGM is typically 30-300 Hz, so 1 kHz output respects
Nyquist by ~3×).

**What this does to the signal.** Reduces sample count without
distorting the morphology. We omit an anti-alias filter at this step
(Phase 1) because:

- AP membrane dynamics have minimal energy above a few hundred Hz.
- The source rate is 4× the target.

For Phase 2 / Courtemanche (faster wavefronts, sharper activation
deflections), we will swap the linear interpolation for
``scipy.signal.resample_poly`` with a proper polyphase filter; the
existing function-signature stays the same.

**Code:**
- Resample function: `simulate/pseudo_egm.py::downsample` (integer
  stride + linear-interp fallback), called from
  `simulate/runner.py::run_single` immediately after
  `bipolar_from_unipolar`.
- Exact-T pinning: `simulate/runner.py::run_single` (the
  truncate/pad block after the downsample, so every trace in a
  dataset shares T).

## Step 8 — Label assignment

**What the code does.** ``LabelPolicy.apply(simulation_result)`` is
called once per simulation. ``GlobalDensityLabel`` reads the realised
fibrosis density from the substrate metadata and labels every
bipolar pair in the simulation the same way. ``LocalDensityLabel``
walks each bipolar pair, samples the substrate mask in a circle of
radius ``radius_mm`` around the pair's midpoint, counts fibrotic vs
healthy interior cells, and labels each pair independently.

**What this represents.** The ground-truth label the classifier learns
to predict. The two policies represent two different research
questions:

- **GlobalDensityLabel** asks "Is the *whole patch* fibrotic?" Useful
  if your downstream clinical question is dichotomising patients into
  healthy vs. diffuse-fibrosis (e.g. for ablation triage).
- **LocalDensityLabel** asks "Is the *bipolar pair's neighbourhood*
  fibrotic?" Useful if your downstream clinical question is identifying
  which mapping points sit over scar tissue (e.g. for targeted
  ablation site selection — the original portfolio project goal).

The receptive-field mismatch in ``GlobalDensityLabel`` is the failure
mode we hit on the v1 training run: at low global density (~10%) a
bipolar pair's signal is dominated by what's within a few mm of the
pair, not by the whole-patch density. Traces from a low-density sim
with no fibrosis near the pair look healthy but are labeled fibrotic
under GlobalDensityLabel. ``LocalDensityLabel`` resolves this by
labelling each pair from its own neighbourhood.

**What this does to the signal.** Nothing — labels are metadata, not
modifications to the signal. They land in the ClassifierBank
alongside the trace.

**Code:**
- Protocol + concretes: `simulate/label_policy.py::LabelPolicy`,
  `GlobalDensityLabel`, `LocalDensityLabel`.
- Application loop:
  `simulate/dataset.py::generate_dataset` (calls
  `config.label_policy.apply(result)` per sim, accumulates flat
  labels + asserts label-dict consistency).

**Read deeper.**
- **[Marchlinski 2000]**, **[Sanders 2003]** — The clinical voltage
  thresholds used for ablation site selection; these define the
  ground-truth criteria that a clinician would use to decide "this
  electrode is over scar".

## Step 9 — Storage

**What the code does.** ``simulate/storage.py`` collects all
``SimulationResult`` instances + ``LabelPolicy.apply()`` outputs across
the N simulations, builds a Pydantic ``ClassifierBank`` model from
egm-contracts, and calls ``myocard_egm_data.banks.write_classifier_bank``.
Optionally also writes a SyntheticBank if the config sets
``also_emit_synthetic_bank: true``.

**What this represents.** The producer's contract with downstream
consumers. ClassifierBank is the unified, labelled, classifier-ready
format that egm-classifier, egm-viewer, and any future consumer
agree on. SyntheticBank is the older "rich per-trace metadata" format
preserved for offline analysis.

**What this does to the signal.** Nothing material — float32 traces
land in HDF5 alongside metadata. The schema-versioned writer makes the
file self-describing, so a consumer at the same egm-contracts pin can
validate the file on read.

**Code:**
- Default ClassifierBank writer:
  `simulate/storage.py::write_classifier_bank_from_dataset` (builds
  ClassifierBankMetaData + per-trace ClassifierTrace list, hands to
  `myocard_egm_data.banks.write_classifier_bank`).
- Optional Pydantic SyntheticBank writer:
  `simulate/storage.py::write_synthetic_bank_from_dataset` (builds
  the Pydantic model with all 12 per-trace columns and the top-level
  provenance fields, hands to `myocard_egm_data.banks.write_synthetic_bank`).
- For the hybrid (post-mixer) case, the equivalent
  SyntheticBank writer is
  `mixer/storage.py::write_hybrid_synthetic_bank_from_classifier`
  — it reads mixer audit fields from `trace_metadata` instead of
  emitting `NaN`/`""`.

## What the classifier eventually sees

After nine steps, a single training example looks like:

| Field | Type | Where it came from |
|---|---|---|
| ``signal`` | ``(T_samples,)`` float32 at 1 kHz | bipolar pair from one electrode row of one simulation |
| ``label_truth`` | int (0 = healthy, 1 = fibrotic) | LabelPolicy.apply() |
| ``trace_metadata.sim_id`` | int | which simulation produced this trace |
| ``trace_metadata.pair_index`` | int | which bipolar pair within that simulation |
| ``trace_metadata.electrode_row`` | int | which mesh row of the 5×5 grid |
| ``trace_metadata.fibrosis_density_realized`` | float | per-sim global density |
| ``trace_metadata.electrode_height_mm`` | float | per-sim sampled height |
| ``trace_metadata.stim_edge`` | str | per-sim edge |

With ~100 simulations × 20 bipolar pairs = ~2000 traces, sampled
across density, edge, electrode height. The classifier sees the
bipolar morphology; the metadata supports analysis but is not part of
the model input.

## Annotated reading list

The three **[anchor]** papers are the highest leverage if you only
read three. Group them as: physics, formulation, downstream
classification.

**Physics**
- **[Aliev 1996]** — Aliev R, Panfilov AV. *A simple two-variable
  model of cardiac excitation.* Chaos Solitons Fractals. The
  Aliev-Panfilov phenomenological cell model — Phase 1's cell model
  [anchor].
- **[Courtemanche 1998]** — Courtemanche M, Ramirez RJ, Nattel S.
  *Ionic mechanisms underlying human atrial action potential
  properties.* Am J Physiol. The human atrial ionic model — Phase 2's
  cell model.
- **[Clayton 2011]** — Clayton RH, Bernus O, Cherry EM, et al.
  *Models of cardiac tissue electrophysiology: progress, challenges,
  and open questions.* Prog Biophys Mol Biol. A textbook-level review
  comparing phenomenological vs ionic models, monodomain vs bidomain,
  fitted vs first-principles parameterisation.
- **[Loewe 2014]** — Loewe A, Wilhelms M, Schmid J, et al. *Parameter
  estimation of ionic models for human atrial myocytes.* Comput
  Cardiol. AF-remodelled Courtemanche; useful for Phase 4.

**Forward formulation (V_m → φ_e)**
- **[Plonsey 1969]** — Plonsey R. *Bioelectric phenomena.* The
  textbook treatment of extracellular potentials from intracellular
  current sources.
- **[Plonsey 1988]** — Plonsey R, Barr RC. *Bioelectricity: A
  quantitative approach.* The textbook on the pseudo-bidomain
  approximation; chapter on volume-conductor integrals.
- **[Henriquez 1993]** — Henriquez CS. *Simulating the electrical
  behavior of cardiac tissue using the bidomain model.* Crit Rev
  Biomed Eng. Clean derivation of the bidomain equations.
- **[Okenov 2024]** — Okenov AO, Nezlobinsky T, Vandersickel N,
  Panfilov AV. *Atrial fibrosis identification with bipolar
  electrograms using a deep learning approach.* PLOS ONE. Most
  directly analogous prior work: same backend (Finitewave), same
  fibrosis substrate, same pseudo-EGM forward calc, CNN classifier
  [anchor]. Supplement S2 has the explicit formula.

**Substrate + propagation**
- **[Nezlobinsky 2021]** — Nezlobinsky T, Solovyova O, Panfilov AV.
  *Anisotropy of conduction velocity in atrial fibrosis: implications
  for tissue heterogeneity assessment.* Front Physiol. The
  density-vs-propagation regime map for diffuse atrial fibrosis on a
  2D patch [anchor].
- **[Mitrea 2009]** — Mitrea BG, Caldwell BJ, Pertsov AM. *Imaging
  electrical excitation inside the myocardial wall.* Biomed Eng
  Online. Optical mapping of fibrillation in fibrotic hearts — real
  physiology to compare against the idealised square patch.
- **[Hansson 1998]** — Hansson A, et al. *Right atrial free wall
  conduction velocity and degree of anisotropy in patients with
  stable sinus rhythm studied during open heart surgery.* Eur Heart
  J. The empirical atrial CV anisotropy that informs our default
  3:1 ratio.
- **[Lemery 2007]** — Lemery R, Birnie D, Tang ASL, et al. *Normal
  atrial activation and voltage during sinus rhythm in the human
  heart: an endocardial and epicardial mapping study in patients with
  a history of atrial fibrillation.* J Cardiovasc Electrophysiol.
  Empirical baseline atrial activation maps.

**Bipolar EGM interpretation + clinical anchors**
- **[Stevenson 2005]** — Stevenson WG, Soejima K. *Recording
  techniques for clinical electrophysiology.* J Cardiovasc
  Electrophysiol. Why bipolar over unipolar for activation timing.
- **[Marchlinski 2000]** — Marchlinski FE, Callans DJ, Gottlieb CD,
  Zado E. *Linear ablation lesions for control of unmappable
  ventricular tachycardia in patients with ischemic and nonischemic
  cardiomyopathy.* Circulation. The original < 0.5 mV bipolar
  threshold for scar.
- **[Sanders 2003]** — Sanders P, Morton JB, Kistler PM, et al.
  *Electrophysiological and electroanatomic characterization of the
  atria in patients with atrial flutter.* Circulation. Atrial-specific
  voltage thresholds (the "electrically silent" tier at < 0.05 mV).

**Synthetic-side ML precedent**
- **[Sánchez 2021]** — Sánchez J, et al. *Influence of left atrial
  ostial regions on the morphology of the left atrial appendage
  during atrial fibrillation: A computational and electroanatomic
  mapping study.* Front Physiol. Synthetic + real mixing for atrial
  EGM classification; the methodological ancestor of this pipeline.
- **[Stewart 1999]** — Stewart MS. *Source-sink mismatch and
  termination of conducted action potentials.* Numerical study of
  why thin-strip stimulation reliably initiates a propagating wave.
