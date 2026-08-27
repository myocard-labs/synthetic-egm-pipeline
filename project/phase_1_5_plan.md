# synthetic-egm-pipeline — Phase 1.5 implementation plan

**Repo:** synthetic-egm-pipeline · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** in progress · **Progress:** 32/48 steps done — **Wave 1 complete; Wave 2 underway**
(S38 split into S38a/S38b/S38c, S16 into S16a/S16b and S18 into S18a/S18b/S18c, hence 46, plus S41 and S42 as local steps = 48; **S17 absorbed by S38c**, which
does not reduce the total — it is a step accounted for, not a step deleted)
**Next:** **S20** — the Courtemanche wall-clock characterisation, rescoped: the timing must be
taken at **`dr` = 0.1 mm**, the pitch its cards are actually solved at, not the 0.25 the entry was
written against. That number is the one input still missing from the backend re-evaluation and the
AP/Courtemanche tiering decision, and it has twice had to be recorded as an estimate.
S41 · S42 · S18c · S19 shipped 2026-08-25.
The pseudo-EGM / calibration cluster
(S37 · S39 · S40 · S38a · S38b · S38c) is **complete and round-trip verified**, and S16a/S16b
shipped 2026-08-16; CL-167's detection-curve unification still rides with a later step — S16a made
the curve *selectable*, which is the precondition for measuring whether unifying it matters, not the
unification itself.
**Repo estimate:** **85.5–152 h** active (40 complexity points; cold-start ranges — the
`estimation_ledger.csv` is empty, so every estimate here is by analogy against the §8 reference
anchors, not `points × measured rate`)

---

## Scope — what this plan covers

Eleven §3 core issues, the two single-repo §4 backlog items, and one defect carried into Wave 2 (CL-143). Wave numbers
are from design §7: Wave 1 is the schema migration (no new behavior), Wave 2 is features. **Rows are in
execution order**, matching the Steps section below.

| Phase item | Wave | What it needs from this repo | Cx | Estimate | Steps |
|---|---|---|---|---|---|
| SEP12 | 1 | Write `synthetic_bank` v2.0 with today's behavior — per-sim typed per-function config, collapsed `traces/`, int label, trivial θ-spec, both banks always emitted | L (5) | 16.5–31 h *(planned 12–24; S9 was added mid-wave from review)* | S0–S11 |
| SEP2 | 2 | Controlled-position crop — call SIG1's `SingleActivationWindower`; **this repo owns the sizing response + wiring + provenance**, not the crop math | S (2) | 7–12 h | S12 · S14–S15 |
| CL-143 **+ B13** | 2 | **One ClassifierBank, one θ bank** — the clean intermediate gets its own id base (B13) and its own θ bank, so the join stops refusing it | S (2) | 2–4 h | S13 |
| SEP10 | 2 | Anchoring opt-in flag — `UniformPositionGenerator(p, p)` is the fixed arm; config-only | XS (1) | *(absorbed)* | S15 |
| SEP13 **+ CL-167 half** | 2 | Positional-sensitivity probe bank — one sim, crop offset swept on a grid, morphology/seed held constant. **Plus the configurable detection curve** folded in 2026-08-16: `crop_traces` always accepted a `preprocessor` the runner never passed, so the seam was unreachable from config | M (3) | 4–8 h | S16a · S16b |
| SEP5 | 2 | Courtemanche 1998 human-atrial cell model alongside Aliev–Panfilov. **S17 absorbed by S38c**; S18 split into S18a (spec-carried cell model, pure refactor) + S18b (the model itself) | L (5) | 11–20 h | S18a · S18b · S18c · S19 · S20 |
| B12 | 2 | Resource / CPU cap on a generation run | S (2) | 2–4 h | S21 |
| SEP11 | 2 | θ-sweep harness + `generation_params` writer + pluggable sampler (OAT first, then LHS/grid) — one capability, run twice per §8.2 | L (5) | 12–20 h | S22–S25 |
| SEP1 | 2 | Band-pass out of the mixer into a general, independently-runnable post-processing stage | M (3) | 6–9 h | S26–S28 |
| SEP8 | 2 | Pluggable `NoiseSelectionStrategy` (uniform default + per-sim patient / record) | M (3) | 6–9 h | S29–S31 |
| SEP3 | 2 | Absolute noise floor in the mixer (today's SNR is purely relative) | S (2) | 3–6 h | S32 |
| SEP6 | 2 | Multi-edge `planar_edge` activation variant | S (2) | 3–6 h | S33 |
| SEP7 | 2 | `point` + `s1s2` activation variants | M (3) | 6–11 h | S34–S35 |
| CL-169/170 · CL-166 | 2 | **Pseudo-EGM correctness** — vendor our own EGM kernel, fix the axis transpose, restore `1/r`. *Local steps, not §3 issues.* | M (3) | 6–11 h | S37 · S39–S40 |
| CL-166/167 | 2 | **CV recalibration + one detection curve.** *Local step.* | S (2) | 3–5 h | S38 |
| *(phase exit)* | — | **examples/ config-set rework** · roadmap trim · CHANGELOG · architecture.md reconciliation | — | 2–4 h | S36 |

**Cross-repo prerequisites — all met (checked 2026-08-11).** Wave 1 is done here (SEP12 shipped).
**SIG1 landed as egm-signal v0.4.0**, so SEP2 is unblocked. The constellation has moved on since the
Wave-1 re-pin, so **Wave 2 opens with a re-pin**: contracts `v0.6.0 → v0.6.1`, data `v0.6.0 → v0.6.2`,
signal `v0.2.0 → v0.4.0` (design §7 Wave 2). Nothing else blocks.

## Design notes

Eight decisions, all settled. D1, D2, D3, D6 and D7 are repo-internal (mine); D4 and D8 were
escalated and settled by the project-lead / Daniel; D5 is external input.

### D1 — `SimulationResult` must carry the realized per-sim specs *(repo-internal)*

`generate_dataset` constructs the per-sim `UniformRandomFibrosis(density=…)`,
`PlanarEdgeStimulus(edge=…)` and `CenteredGrid2D.sample(…)` objects, hands them to `run_single`, and
then **throws them away** — only duck-typed scalars survive, in `run_metadata`
(`fibrosis_density_requested`, `stim_edge`, `electrode_height_mm`, …). `synthetic_bank` v2.0 needs
the *realized objects* per simulation to serialize `geometry` / `substrate` / `activation` /
`electrodes` as typed polymorphic JSON. So `SimulationResult` grows one field — a frozen bundle of
the four (soon five, see D2) specs actually used for that sim.

Guardrail 2 locks `SimulationResult` as a public concrete frozen dataclass and locks the four
strategy *Protocols*. Adding a field to the concrete result is a **widening**: every existing reader
keeps working, and only the runner (the sole producer) changes. The Protocols themselves are
untouched. This is the same category as the `@property` widening already recorded in
architecture.md, so it is in scope — but it must be **written into architecture.md**, not left in
this ephemeral plan.

### D2 — cell model as a fifth strategy spec, not a `RunConfig` knob *(repo-internal)*

roadmap.md's open question ("should `RunConfig` grow a `cell_model` discriminator? Decide when
Courtemanche starts") comes due at SEP5. The schema design treats `cell_model` as one of the
**stable functions** with its own typed polymorphic object, parallel to geometry / substrate /
activation / electrodes. Matching that in code means a fifth spec Protocol (`CellModelSpec`, with
`AlievPanfilov` and `Courtemanche` concretes) rather than a scalar on `RunConfig`.

The cost: `SimulationBackend.simulate()` gains a `cell_model` parameter. That is a change to the
Backend Protocol, which Guardrail 3 explicitly says is the right move when the runner needs
something the Protocol doesn't carry ("extend the Protocol instead"). Only `FinitewaveBackend` and
the test `MockBackend` implement it today, so the blast radius is two classes. The alternative — a
`RunConfig.cell_model` string — would leave the schema's typed `cell_model` object with no
corresponding spec to serialize from, and would put model-specific parameters (Courtemanche
conductance scalings, which SEP11 wants to sweep by `path`) into a config object that has no place
for them. Decision: **fifth spec.** Record in architecture.md at S17, and note that
`RunConfig.ap_time_unit_ms` becomes Aliev–Panfilov-specific (Courtemanche runs in real ms) and
should move onto the `AlievPanfilov` concrete.

### D3 — the noise-mixed `synthetic_bank` path loses its reconstruction route *(repo-internal)*

`build_synthetic_bank_from_classifier` rebuilds a SyntheticBank from a **noise-mixed
ClassifierBank** by reading per-trace scalars back out of `trace_metadata`. Under v2.0 the per-sim
typed config is not in `trace_metadata` and cannot be reconstructed from it, so that route breaks.

Plan: the **inline** mixer path (`synthegm-generate-dataset` with a `mix:` block) builds the v2.0
bank from the `DatasetResult` it already holds in memory, mixing only the trace signals. The
**standalone** `synthegm-mix` path, run against a bare ClassifierBank on disk, then cannot emit a
v2.0 `synthetic_bank` — it has no access to the generation config. That is an acceptable, documented
limitation (standalone mix keeps emitting a noise-mixed ClassifierBank, which is what it is
principally for), not a silent degradation: it should raise a clear error if
`also_emit_synthetic_bank` is set on the standalone path. Confirm at S4.

### D4 — both banks are emitted, joined by `simulation_id` *(RESOLVED — project-lead, 2026-07-28)*

**Settled for Phase 1.5.** The ClassifierBank stays a **source-agnostic ML artifact** — signal,
label, `simulation_id` key — and θ does **not** go on it. The `synthetic_bank` carries θ + the
per-sim config as a **parallel artifact**, explicitly *not* the ClassifierBank's source bank. A
synthetic run emits both, joined on `simulation_id`, and the synthetic bank is **required** for T4,
not opt-in-off. Accepted known cost: the two banks duplicate the trace signal — logged as **FB-11**,
de-dup deferred (banks are small + gitignored). Detail in the schema investigation §12 + FB-11.

**Flag mechanics were left to this repo. Decision: retire the flag, don't default it to true.** A
flag that can switch the θ artifact off is a flag that can silently break the T4 thread — which is
exactly the failure this escalation caught — and "generate a bank with no recoverable generation
provenance" is not a mode worth offering in a project whose narrative is per-component change
control. So `output.also_emit_synthetic_bank` is removed; a config still carrying the key raises a
`ConfigError` naming the change rather than being silently ignored (the seven `examples/` configs and
Daniel's four local `configs/` files all set it, so silent-ignore would strand eleven files);
`output.synthetic_bank` becomes optional and derives as a sibling of `output.classifier_bank` when
omitted. Standalone `synthegm-mix` is the one exception and cannot emit a synthetic bank — see D3.
Rewritten into `project/architecture.md` ("Two outputs, different purposes, joined by
`simulation_id`", replacing "Why not SyntheticBank by default").

### D9 — no back-bounding; window synthetic like IAFDB *(SETTLED — Daniel, 2026-08-12)*

**Decision: we do not constrain the position range to keep the activation early. Synthetic traces are
windowed the same way IAFDB traces are** — same mechanism, same kind of range — with one expected
difference: a synthetic trace holds a **single** activation where an IAFDB window is cut from a
multi-beat record. Nothing else about the framing should differ, because differing framing is itself a
sim-to-real gap.

**Why the back-bounding argument is dead.** It was reasoned back from a measurement that turned out to
be an artifact. The chain was:

1. the front-budget probe reported activations at index 54–150 on a 40 mm patch, too early for a
   centred window needing 96 samples of lead-in;
2. so the front budget looked physically unbuyable;
3. research then explained *why* it could not be bought — bipolar rejects far field, so lead-in is
   inherently flat — and recommended holding `p` low.

Step 3's physics is sound and step 1's number was **not an activation**. CL-167 showed those
detections cluster on two fractions across all 40 traces: the synchronous stimulus launch and the
far-boundary extinction. The transpose bug (CL-169) had cancelled the near field, so the detector had
nothing local to find and locked onto global events. **The constraint was derived from the artifact it
was meant to work around.**

So back-bounding is a fix for a problem we have not actually observed. It is also expensive: it would
have put synthetic at `[0.1, 0.35]` against iafdb-pipeline's shipped `[0.4, 0.6]`, leaving the two
corpora **disjoint in position** — a clean "which corpus is this" cue, which is the sim-to-real gap T1
exists to close, not widen.

**What replaces it: an empirical check, not a preemptive constraint.** After S37 (axis conventions)
and S38 (CV recalibration) land, re-run the probe and measure where the *real* local activation
arrives. Then either a centred range fits at the production geometry, or it does not — and if it does
not we deal with the actual number rather than a predicted one. Note the fixes push in **opposite**
directions: the transpose fix should move detections to genuine, later local arrivals, while a correct
faster CV moves arrivals earlier. Which wins is measurable, not arguable.

**Standing rule from this:** do not introduce a constraint on `p` unless a measurement on
correctly-simulated traces demands it. If one ever does, the fix is a shared decision with
iafdb-pipeline so the ranges keep overlapping — never a synthetic-only narrowing.

**Cleanup this implies** (do before committing S14+S15):

- `simulate/sizing.py`'s module docstring argues the back-bounded case as settled rationale. Rewrite:
  the front/back asymmetry is real, but the conclusion drawn from it is not.
- the example configs' `activation_position: {low: 0.25, high: 0.75}` should line up with whatever
  iafdb-pipeline uses (`[0.4, 0.6]` today) rather than being chosen independently here.

### D8 — one ClassifierBank, one θ bank *(SETTLED — Daniel, 2026-08-11)*

egm-studio hit this against real Wave-1 banks (CL-143): a **clean intermediate** ClassifierBank and its
noise-mixed sibling both name the *same* θ file but claim **different ids** for it, so egm-data's
`join_traces_with_simulations` refuses the clean one. The refusal is correct — simulation ids restart at
0 in every bank, so a permissive join would pair traces with the wrong config.

**Mechanism, from my own code.** In the inline-mix path `generate_dataset_cmd` builds the clean bank
**without passing `bank_id`**, so it derives a cell-model id and `theta_bank_id_from` turns *that* into
a companion id — while the single θ file actually written belongs to the *mixed* run, under the
configured stem. Two call sites deriving independently, one file. The `theta_bank_id_from` docstring
predicted the failure and mis-stated the cause: it needs no second override, just two derivations.

**Decision: emit a clean θ bank alongside the clean ClassifierBank. Every ClassifierBank has exactly
one θ partner.**

I had leaned the other way (have the companion entry *record* the θ file's real id, one artifact, no
duplication). Daniel's reasoning is better, and one argument is decisive:

- **The 1:1 invariant is worth more than the saved bytes.** "Every ClassifierBank has exactly one θ
  partner whose id it names" is checkable in one line and has no special cases. Recording the real id
  instead would make it *many-to-one*, and every consumer would then need to know that a θ bank may be
  shared.
- **A θ bank stores signals, not just config** — and this is what I got wrong in the first draft of
  this note. I argued the sharing was harmless because "only the signals differ". But `traces/signal`
  *is* part of the θ bank, so under the shared-file design a clean ClassifierBank's θ partner would
  contain the **mixed** waveforms. A consumer joining on `simulation_id`/`pair_index` would get
  different signals from the two artifacts for the same trace, with nothing flagging it. That is the
  same failure shape as the `bank_path` bug D7 fixed: a pointer whose target doesn't contain what the
  pointer implies.
- **The duplication is transient, not standing.** Once §8.3 / STU5 settle whether noise-mixed or clean
  features sit closer to IAFDB, production runs generate one or the other — not both. The dual-output
  case is a Phase-1.5 comparison artefact, so the FB-11 cost I was protecting is paid rarely and
  deliberately, on exactly the runs whose purpose is to compare the two.

**Consequence to document:** an inline-mix run with `output.clean_intermediate` set now writes **four**
files — mixed ClassifierBank + mixed θ, clean ClassifierBank + clean θ.

**This absorbs B13.** The root cause is that the clean intermediate has no id base of its own; B13
("custom bank id for the clean bank in noise-mix runs") is the same gap approached from the config
side. Giving the clean bank a proper id base is now a *prerequisite* of the fix rather than a separate
nicety, so the two land together in **S13** and B13's own step is retired.

### D7 — keep both builders; do **not** derive the ClassifierBank via egm-data's converter *(repo-internal, confirmed with Daniel 2026-08-01)*

egm-data ships `synthetic_bank_to_classifier`, and deriving the ClassifierBank from the SyntheticBank
would delete ~90 lines here and make D4's two artifacts agree by construction. I proposed exactly that
in CL-103. **It is not worth it in 1.5**, for one concrete reason: the converter hardcodes
`amp_type="mv"`, and synthetic traces are not in millivolts — the pseudo-EGM forward calc drops the
`4π/σ_e` normalisation, which is why this repo stamps `"synthetic_au"`. `synthetic_bank` 2.0 records
no amplitude convention at all, so the converter *cannot* do better without a schema change.

Verified 2026-08-01 that this is a latent trap and not a live bug: nothing in any repo's `src/` calls
the converter (the single caller is one egm-classifier test already being reworked). Adopting it is
what would have created the exposure.

**So: the producer keeps building both banks directly.** The proper fix — the bank stating its own
amplitude convention, required on write, with the converter reading it — is **FB-17**, batched with
FB-15 / FB-16 into the Phase-2 contracts bump. The cost of keeping two builders is drift risk, which
**S3** covers with a cross-check test.

### D6 — the probe sweeps by exact shift, not by re-detection *(repo-internal)*

SEP13 is not "call SEP2's crop N times." SEP2 places the activation by running SIG1's detector and
cropping to the sampled `p`, which carries detector jitter — fine when `p` is a training-augmentation
draw, wrong for a probe. STU8 plots model output **against** activation offset; if the offset axis is
itself noisy, the jitter blurs the curve the study exists to measure, and the anchored-vs-varied
difference gets harder to see for a reason that has nothing to do with the models.

So the probe detects **once** on the source trace and then places the activation by exact integer
shift per grid point. The x-axis carries no detector jitter, and it is cheaper (one detection, not N).
The verification is correspondingly stricter: assert the emitted `activation_position` column
*reproduces the grid*, not that it approximates it.

**Amended 2026-08-16 — "exact" needs the grid to live on the sample lattice.** The original wording
("realized position equals requested position by construction") is unachievable for a grid stated in
fractions, and the arithmetic says so plainly. The crop places the window at
`s = round(t_a − p(T−1))` and reports `realized = (t_a − s)/(T−1)`, so realized equals requested
**iff `p(T−1)` is an integer**. At `T = 192`, `T − 1 = 191`, which is **prime** — the only
representable positions are `j/191`, and a natural grid lands on none of them:

| requested | 0.1 | 0.2 | 0.3 | … | 0.9 |
|---|---|---|---|---|---|
| `k = round(p·191)` | 19 | 38 | 57 | | 172 |
| realized | 0.099476 | 0.198953 | 0.298429 | | 0.900524 |

Worst-case error is half a sample — sub-sample by construction, i.e. below what the axis can
represent at all.

**Resolution (Daniel, 2026-08-16): fractions in, snapped grid of record.** The config states the grid
in fractions, matching the `activation_position: low/high` idiom already shipped; each point is
immediately snapped to `k = round(p(T−1))`, and **the snapped value is the grid** — what the sweep
requests, what the bank stores, and what the verify asserts against. So D6's guarantee survives in
the form that was always the real one: *the probe's x-axis is a sample lattice, and every point on it
is hit exactly.* The requested-vs-snapped difference is reported once at config time, not per trace.

Two consequences worth stating, because both are places this could go quietly wrong:

- **Snapping can collide.** A grid finer than the lattice maps two requested fractions to one `k`,
  which would emit the same crop twice under two different labels. That **errors**, naming the
  colliding pair — it is a mis-specified study, not something to silently deduplicate. `T = 192`
  admits at most 192 distinct points.
- **The column is `float32`.** `k/191` is exact in float64 on both sides (same integer division), but
  the bank stores float32. The equality assert therefore compares **in float32** — `np.float32(k/191)`
  against the column — rather than comparing a float64 grid to a float32 round-trip and discovering
  a tolerance is needed.

### D5 — `T` is now fixed at **192 ms**; `𝒫` still comes from study §8.1 *(external input)*

**`T` settled (CL-024 §4, 2026-07-28): 192 ms at 1 kHz.** The constraint is `T ≡ 0 (mod 64 samples)`
— CLF3's MobileViT needs it — so §8.1 optimises on the 64-sample grid and the 150–250 ms window
rounds to 192 ms (rounding to the grid was chosen over relaxing the architecture or padding). Two
consequences here: every shipped config's `run.trace_duration_ms: 200.0` becomes **192.0**, and
S35's sim-sizing derives from 192 ms rather than a placeholder.

**Also fixed: synthetic and IAFDB share a sample rate** (catch22 lag features depend on it), so
`run.output_fs_hz` is not independently tunable — any rate change is both-sides-or-neither and goes
through the project-lead.

**Still open:** the fraction range `𝒫` for SEP2 / SEP10, which study §8.1 sets alongside the yield
trade-off. The plan builds the structure — `𝒫` as a `[0,1]` fraction range with the "collapsed to a
point = fixed" idiom from A4 — and takes the number when the study lands. No code waits on it; only
the shipped config values do.

**`𝒫_synth ⊇ 𝒫_iafdb` is deliberate and was re-affirmed — don't "fix" it.** CL-066/CL-067 asked
whether the ranges should be matched now that both corpora store the position, and research said no:
`activation_position` is an **augmentation axis, not a realism axis**, so it is excluded from the STU4
objective and the STU5 distance rather than having the ranges constrained. The inequality exists for
T1's positional-shortcut augmentation. **SEP2 is unchanged by that whole exchange** — recorded here
because the range looks like an inconsistency to a later reader and it is not.

**Watch item: `T` could still move to 256 ms.** CL-072's §8.1 review adds a feasibility go/no-go — if
the longest fractionated `W_act` approaches 192 ms, the feasible `p` range collapses and multi-beat
drop rates spike, which triggers widening `T`. S35's sizing derives from `T`, so it should read the
value from config rather than assume 192.

---

## Steps

Each step is one focused commit, ends green (its own tests + `ruff format` + `ruff check` + `mypy`),
and states its verification. ☐ todo · 🔨 wip · ✅ done.

**Step ids are flat `S<n>` in execution order**, matching the other repos' plans; the §3 issue each
step serves is named in its title and in the Scope table's *Steps* column. **S0–S11 are Wave 1**
(shipped); **S17 onward is Wave 2**. Renumbered from the earlier per-issue scheme on 2026-08-11 for
fleet consistency — which also surfaced one dependency that had been recorded from chronology rather
than necessity (now S7).

### Wave 1 — SEP12: `synthetic_bank` v2.0, current behavior

> The Wave-1 gate (design §7) is: a bank regenerated with today's config round-trips through v2.0
> and yields the **same traces + labels** as before the restructure. No new generation behavior in
> any of these steps.

### S0 — Per-sim specs on `SimulationResult` + rename the join key (SEP12) ✅ (2–4 h)
- **Change:** (a) add a frozen `SimulationSpecs` bundle (geometry · substrate · activation ·
  electrodes, plus `label_policy` at the dataset level) to `simulate/result.py`; populate it in
  `run_single`; keep `run_metadata`'s existing duck-typed scalars untouched so nothing downstream
  breaks yet. Record the D1 widening in `project/architecture.md` under Guardrail 2.
  (b) **`sim_id` → `simulation_id`** in `build_clean_trace_metadata` (CL-024 §3, resolving my
  CL-008 in favour of option 1): the producer's direct-write path now uses the same key as
  egm-data's converter, the schema-pinned column, and the T4 join. `run_metadata["sim_id"]` in
  `dataset.py` and the local fixtures follow.
- **Verify:** `tests/test_runner.py` asserts the bundle holds the exact objects passed in;
  `grep -rn '"sim_id"' src/ tests/` returns nothing; existing runner + builder tests green.
- **Depends on:** none — **touches no schema, so it is startable before contracts v0.6.0 tags.**
  Deliberately the earliest step: egm-studio has fixtures on `sim_id` (CL-024 §3) and egm-data adds
  a writer check on the key name, so landing the rename first unblocks both rather than making them
  wait on the whole Wave-1 cascade.

### S1 — Spec → contracts per-function model mapping (SEP12) ✅ (2–4 h)
- **Change:** new `simulate/bank_config.py` — pure functions mapping each spec concrete to its typed
  egm-contracts model (`geometry`, `cell_model`, `substrate`, `activation`, `electrodes` incl. the
  realized `pairs` list, `backend`, `label_policy`, `label_names`, `substrate_summary`). No h5py, no
  finitewave (Guardrails 1 + the no-HDF5 rule).
- **Verify:** unit tests per mapper — round-trip each Phase-1.5 concrete through its contracts model
  and back to equal field values.
- **Depends on:** S0; **egm-contracts v0.6.0 tagged**.

### S2 — Re-pin + rewrite the clean SyntheticBank builder (SEP12) ✅ (2–4 h)
- **Change:** re-pin `myocard-egm-contracts@v0.6.0` + `myocard-egm-data@v0.5.x` in `pyproject.toml`;
  rewrite `build_synthetic_bank_from_dataset` to emit the `simulations/` group + collapsed `traces/`
  (`signal`, `simulation_id`, `pair_index`, `label` as int, `snr_db`, `noise_record`,
  `noise_channel`). Drop `StimEdgeEnum` and the `fibrosis_density*` / `electrode_row` /
  `electrode_height_mm` / `stim_edge` trace columns.
  If CL-060 lands, the collapsed `traces/` also carries a nullable **`activation_position`** column,
  written absent here (Wave 1 has no controlled crop) and populated at S23.
- **Verify:** `tests/test_builders.py` — a two-sim `DatasetResult` produces two `simulations/`
  entries with the right FK join, and per-trace labels match `DatasetResult.labels` exactly.
- **Depends on:** S1; **egm-data v0.5.x tagged**.

### S3 — Cross-check the two banks agree (SEP12) ✅ (0.5–1 h)
- **Change:** a test asserting that the ClassifierBank and the `synthetic_bank` emitted from **one
  run** agree per trace on `label` and `simulation_id`. The producer builds the two independently
  (D7), so nothing structural forces them to match; this is the guard that replaces the
  by-construction guarantee deriving one from the other would have given.
- **Verify:** the test fails if either builder's ordering or labelling drifts.
- **Depends on:** S2.

### S4 — Noise-mixed path onto v2.0 (SEP12 · D3) ✅ (2–3 h)
- **Change:** inline-mix path builds the v2.0 bank from the in-memory `DatasetResult` + mixed
  signals; `build_synthetic_bank_from_classifier` is retired or reduced to the ClassifierBank-only
  case; standalone `synthegm-mix` raises a clear `ConfigError` if asked for a synthetic bank.
- **Verify:** inline noise-mixed run emits a v2.0 bank whose `snr_db` / `noise_record` /
  `noise_channel` columns are populated and whose `simulations/` config matches the clean run;
  standalone-mix error path asserted in `tests/test_cli_config.py`.
- **Depends on:** S2.

### S5 — Always emit both banks; retire the flag (SEP12 · D4) ✅ (1–2 h)
- **Change:** remove `output.also_emit_synthetic_bank` from `cli/_config.py` (both the
  generate-dataset and mix config builders); every synthetic run writes the ClassifierBank **and**
  the `synthetic_bank`. `output.synthetic_bank` becomes optional, deriving as a sibling of
  `output.classifier_bank`. A config still carrying the retired key raises a `ConfigError` naming the
  change. Update the seven `examples/*.yaml` that set it.
- **Verify:** `tests/test_cli_config.py` covers the retired-key error + the derived path; a plain
  generation run produces both banks with matching `simulation_id` sets. Daniel's four local
  `configs/phase_1_5_*.yaml` also carry the key — flag them, they're untracked and his to edit.
- **Depends on:** S4.

### S6 — Trivial θ-spec + bank-root fields (SEP12) ✅ (1–2 h)
- **Change:** write `generation_params_json` as `{regime: {…type discriminators…}, knobs: []}` — the
  regime populated from the run's fixed type discriminators, the knob list empty until SEP11.
- **Verify:** a generated bank's θ-spec validates against the `TunedParam` schema with zero knobs;
  regime matches the config's declared types.
- **Depends on:** S2.

### S7 — Bank-reference semantics (SEP12) ✅ (2–3 h)
- **Change:** `<local>` sentinel for the origin entry; `companion_path` for
  relative-inside-the-tree / absolute-outside companion paths; a
  `synthetic_generation_params` companion entry linking the ClassifierBank to its
  `synthetic_bank` on `simulation_id`.
- **Verify:** end-to-end clean + noise-mixed runs show no entry claiming to hold traces it doesn't;
  companion resolves to a bare filename for siblings and absolute for distant targets.
- **Depends on:** S5 — the companion entry describes the *second* bank, so it needs both banks being
  written, plus the distinct θ id that landed with them. (The old plan listed the docs step here; that
  was chronology, not a prerequisite, and it showed up as a forward dependency once the steps were
  renumbered into execution order.)

### S8 — De-duplicate the ClassifierBank's metadata (SEP12) ✅ (1–2 h)
- **Change:** strip generation parameters from `trace_metadata` (6 keys) and `bank_metadata`
  (~14 keys) now that the θ link makes them reachable per-simulation. Keep identity
  (`simulation_id` / `pair_index` / `patient_id`), the label, `producer` / `producer_version`
  (reproducibility — 2.0 has nowhere else for them), `description`, `trace_duration_ms`, and the
  `label_policy` **identity**. Noise fields stay **absent** on a clean bank.
- **Verify:** named-key tests assert each removed field is gone (so a re-add fails loudly); a real
  Finitewave run shows every removed value still recoverable from the synthetic bank.
- **Depends on:** S7.

### S9 — Fix four bank-reference defects found in review (SEP12) ✅ (1–2 h)
- **Change:** derive the noise-mixed id from the clean bank's **id** (`noise_mixed_id_from`) instead
  of re-deriving from `cell_model`; re-point the θ companion entry at the *mixed* run's θ bank when
  mixing; relax the relative-path rule to "share a real common ancestor"; pass `output_bank_path`
  from `synthegm-mix`.
- **Verify:** real Finitewave run — companion id equals the θ bank's id on both the clean and mixed
  paths, no `unknown` in any id, noise path renders `../noise/…` across sibling directories.
- **Depends on:** S8.

### S10 — Wave-1 equivalence check (SEP12) ✅ (1–2 h)
- **Change:** no source change — a regeneration + comparison. Run an existing example config on the
  pre-migration tag and on `development`, and diff traces + labels.
- **Verify:** identical trace arrays (bitwise, same seed) and identical label vectors; full suite
  green; `grep -rn "import finitewave\|from finitewave" src/` still matches only
  `backends/finitewave/`. This is the design §7 Wave-1 gate for this repo.
- **Depends on:** S5, S6.

### S11 — Docs for the restructure (SEP12) ✅ (1–2 h)
- **Change:** `docs/usage.md` (the new bank layout, what moved, and the retired
  `also_emit_synthetic_bank` — it appears in the config table twice and in two config samples) and
  `docs/simulation_theory.md` (one stale `also_emit_synthetic_bank: true` reference);
  `project/architecture.md` gets the per-sim config section + the D1/D3 notes (the D4 rewrite already
  landed 2026-07-28); `CHANGELOG.md` Unreleased entry.
- **Verify:** `grep -rn also_emit docs/ examples/ src/` returns nothing; the pre-PR run in
  `intracardiac-platform/project/pr_checklist.md` passes.
- **Depends on:** S10.

### Wave 2 — features

**Execution is serial, and the order is settled here rather than at run time** (Daniel,
2026-08-11). The earlier *"these threads can proceed in any order"* framing assumed parallel chats
writing code. That assumption is dead: the long pole is **review attention**, not machine time, so
steps land one at a time — which makes ordering a decision to make **once, in advance**, where it
can be reasoned about, instead of a question re-answered under pressure at each step boundary.

**No step in this plan generates a bank** (Daniel, 2026-08-11). Bank generation is not code work:
it happens **after the code in every repo is finished**, driven entirely from config files, so that
one consistent set of repo versions is stamped across every bank a study uses. Anything a generated
bank needs to have set is therefore a **config value, never a hardcoded one** — a parameter baked
into code cannot be varied at generation time and cannot be recorded as provenance.

That rules out a whole class of ordering argument. "Land X before the banks that would otherwise
need regenerating" is vacuous here, because nothing is generated until the end. What is left:

1. **Real code dependencies.** Only one crosses issues: S22's `TunedParam` path resolution needs
   S18's Courtemanche parameter paths, which puts **SEP5 before SEP11**. Everything else is
   within-issue.
2. **The re-pin goes first (S12).** A dependency bump wants to be bisect-isolable and underneath
   everything built on it — and it carries the three queued chores, including the stale
   `producer_version` that CL-117 flagged.
3. **Cheap defect fixes early (S13).** Not because anything would inherit the CL-143 θ-id
   divergence — nothing is generated yet — but because it is 2–4 h and it is the one known-broken
   behavior in the repo.
4. **Review flow for the rest.** With generation moved to the end, the remaining order is a
   review-sequencing choice, not a correctness one: SEP2 → SEP13 → SEP5 → B12 → SEP11 front-loads
   the work with cross-repo consumers (SIG1's windower, STU8's probe bank, STU7's θ-spec) and leaves
   the self-contained mixer and activation-variant work to the tail.

**Consequence for SEP11 and STU7.** S24's hand-off is a **code** hand-off — STU7 gets the capability
to read a θ-spec bank, not a bank. The §8.2 sequence (OAT banks → STU7 screening → θ-spec membership
→ design sweep → STU4) runs in the data-generation stage after all repos land, and it still works
there: STU7's screening result reaches SEP11 as **config input** to S25, which is why S25 was already
scoped as "config, not new harness code."

**Step sizing (Daniel, 2026-08-11).** Steps target a small-to-medium commit. The first draft split
Wave 2 into 37 steps and several were commit-shaped only on paper — a trailing "config" step, then a
trailing "docs" step, each landing a handful of lines. Two rules apply from here:

1. **A feature's config dispatch and its docs ship in the feature's own commit.** They are how the
   feature is reached and explained; separating them leaves a middle commit that adds a capability
   nothing can call, and a final commit that is pure prose.
2. **Split only when the split buys a distinct verification.** S17 stays apart from S18 because
   "no numeric change" is only meaningful before the new cell model exists, and S12 stays apart from
   everything because a bisect over a dependency bump has to be able to isolate it.

### S12 — Wave-2 re-pin + three queued chores (SEP2) ✅ (1–2 h)
- **Change:** re-pin **contracts v0.6.1 · data v0.6.2 · signal v0.4.0** (design §7 Wave 2). Then the
  three items the project-lead routed here for "SEP2's next touch":
  **CL-117** — derive `__version__` from `importlib.metadata` instead of the hardcoded constant.
  Not cosmetic: it currently reads **`"0.2.0"` against a `v0.3.0` tag**, so every bank written this
  phase has stamped a **wrong `producer_version`** — the reproducibility field CL-109 argued to keep.
  **CL-118** — cap `numpy>=1.26,<2.5` (7/8 repos already have it; numpy 2.5's PEP-695 stubs break mypy
  at `python_version = "3.10"`).
  **CL-112** — `trace_duration_ms` default 200 → **192** (§8.1's 64-grid value; 200 makes MobileViT
  fail outright).
- **Verify:** full suite + bare mypy green on the new pins; a generated bank stamps the real package
  version; the default config yields T = 192 samples at 1 kHz.
- **Left small on purpose:** a dependency bump belongs in its own commit. Fold it into a feature and
  a later bisect can no longer separate "the new pin broke it" from "the feature broke it" — which
  is the one question a re-pin commit exists to answer.
- **Depends on:** none — first Wave-2 step, and it unblocks the rest.
- **Done 2026-08-11.** Scope grew by one thing worth recording: **all five `examples/` configs set
  `trace_duration_ms: 200.0` explicitly**, so changing the constant alone would have left every
  example generating banks MobileViT rejects — the config is what a run actually reads, and per
  Daniel's *"parameters for a generated bank live in config files"* rule that is where the value has
  to move. Test fixtures now derive their length from the same constant (they had 200 baked into
  four coupled places), so the suite exercises the shipped value and won't trip S14's `T % 64` guard.
  Verified end-to-end on a real two-sim run, not just at import: `producer_version` 0.3.0 (was
  stamping a stale `0.2.0`), `trace_duration_ms` 192.0, signal `(40, 192)`.

### S13 — One ClassifierBank, one θ bank (CL-143 + B13 · D8) ✅ (2–4 h)
- **Change:** give the clean intermediate its **own id base** — B13's ask, and the root cause of
  CL-143: the clean bank is built with no `bank_id`, derives a cell-model id, and its companion entry
  then names the mixed run's θ file. Then **write a clean θ bank** beside it. After this, every
  ClassifierBank has exactly one θ partner whose id it names.
- **Verify:** `join_traces_with_simulations` accepts **both** the clean intermediate and the mixed bank
  against their own θ files — the case egm-studio's S4 refused. Assert the clean θ bank holds the
  **clean** signals: the shared-θ design D8 rejected would surface here as mixed waveforms. Regression
  test over a full inline-mix run, not a hand-built fixture — this defect lived in the CLI wiring, and
  every builder-level test passed while it was present.
- **Depends on:** S12 (the re-pin).
- **Done 2026-08-11.** One defect found beyond the brief, in the *other* CLI:
  `_rewrite_theta_companion` rewrote the θ entry's **id** but kept the clean
  bank's **path**. That was invisible while the two banks shared one θ file —
  the very arrangement D8 removed — so fixing the clean side would have aimed
  the mixed bank at the clean θ artifact. Both fields now move together, and
  standalone `synthegm-mix`, which *cannot* write a θ bank at all, now **drops**
  the entry instead of leaving a mixed-derived id on the clean run's file: the
  same id-versus-file divergence as CL-143, one command over.
  Tests live in a new `tests/test_cli_inline_mix.py` driving `main()` with the
  backend swapped for `MockBackend` — the plan called for a full-run regression
  precisely because every builder-level test passed while the defect shipped.

### S14 — Position config + size the simulation to the widest `p` (SEP2) ✅ (3–5 h)
- **Change:** thread SIG1's `UniformPositionGenerator(low, high)` and `SingleActivationWindower`
  through an `activation_position:` config block. **Much smaller than planned**: SIG1 owns the
  detection curve, the `argmax g` detection, the fractional→index conversion and the crop; this repo
  owns only the config surface and the wiring. `𝒫` is *not* a local type — the generator is
  SIG1's and is **stateful** (it holds an rng), so it is constructed per run from the master seed,
  not shared.
  Then the actual synthetic-side *response*, and the part SIG1 cannot do: derive the required
  capture duration from `T` (192 ms) and the widest sampled `p` so the crop never runs off the back —
  extend for back-overhang, flat front. **Delete the zero-pad fallback** in `run_single` and replace
  it with a hard error: under correct sizing a short capture is a bug, and padding zeros into a trace
  manufactures exactly the positional regularity T1 exists to remove (the same argument egm-classifier
  makes in CL-112 for cropping rather than padding at the dataset boundary).
  Guard `T ≡ 0 (mod 64)` here — cheapest place to catch it, and CL-112 shows what it costs downstream.
- **Verify:** a run with a collapsed range yields one realized position; a widened range spans it; a
  config at the widest `p` produces full-length traces with no padding; an artificially short
  capture raises; a `T` off the 64-grid is rejected at config load.
- **Depends on:** S12.
- **⏸ FOLDED INTO S15 (Daniel, 2026-08-11). Written, verified, uncommitted — do not commit alone.**
  Daniel tested the output banks in egm-studio and saw no windowing, correctly. Measured: with and
  without an `activation_position` block the banks are **byte-identical**, `(4, 192)` either way.
  Alone this step ships a config surface that reads as if it controls cropping, changes nothing
  observable, and makes the solver run ~1.7x longer to discard the extra samples.
  **This was a bad split by our own rule.** The step-size pass said a feature's config dispatch ships
  in the feature's own commit, or you get "a commit adding a capability nothing can reach". I applied
  that at issue *boundaries* and missed it *inside* SEP2, where it is the same shape: S14 is the
  config, S15 is the feature. The sizing arithmetic having honest verification of its own is what made
  the split look defensible — and that is not sufficient. **Added clause: a split is also wrong when
  the earlier half ships a user-facing surface whose effect only arrives in the later half.**
  Next session starts by wiring the windower (S15) and the two land as one commit; merge them into a
  single plan step at that point.
- Two deviations from the step text, both deliberate.
  (1) **The windower is constructed at its call site in S15, not here.** The step
  said "thread `UniformPositionGenerator` *and* `SingleActivationWindower`", but a
  windower nobody calls is dead code, and its `preprocessor` (which detection
  curve) is a crop-time decision. The generator lands here because the *sizing*
  genuinely needs it. (2) **This step's "collapsed range yields one realized
  position" is asserted on the *sampled* positions**, not realized ones —
  realized positions only exist once a window is cut, which is S15. The plan
  listed the same check in both steps; the generator-level version belongs here
  and the bank-level one there.
  **`activation_position` is opt-in with no default.** Anchored would have made
  the arm T1 suspects of enabling a positional shortcut the thing a careless run
  falls into, and a varied range has no natural bounds to assume — so both arms
  name their range, and `examples/synthegm_v1_anchored.yaml` is the fixed one.

### S37 — Our own EGM kernel, numerically identical to the stock tracker (CL-169) ✅ (2–4 h)
- **Discovered mid-wave 2026-08-12**, so it takes the next free number and sits where it happens.
  **Local step — not raised to a §3 phase issue** (Daniel).
- **Change:** subclass Finitewave's `ECG2DTracker` and override `calc_ecg` with our own `njit`
  kernel in a new `backends/finitewave/egm_kernel.py`. **`egm`, not `ecg`** — the tracker computes
  φ_e at intracardiac electrode positions, which is an *electrogram*; "ECG" is upstream's framing for
  surface leads and it quietly misled this investigation for two days.
  **This step changes no physics.** The kernel reproduces stock 0.9.3 exactly: same `1/r²`, same
  axis handling, same everything. It exists to prove the plumbing before the physics moves.
- **Why a separate step at all** — the two defects it enables fixing are each one line, so bundling
  would be tempting. But the verification here is *"the banks are byte-identical"*, and that check
  stops meaning anything the moment a real change rides along. Same argument as S17's `CellModelSpec`
  refactor and the Wave-1 equivalence gate; both earned their keep.
- **Why vendor rather than the alternatives** — upstream's `solvers` branch already fixes the
  weighting, but it is unreleased (PyPI stops at 0.9.3) and restructures the package around a
  numba/jax/mlx backend abstraction, so adopting it is a port, not a bump. Switching to our
  `compute_phi_e` instead would need the whole V_m history in memory (~504 MB per simulation at the
  production geometry) — which is exactly why the streaming tracker was chosen in the first place.
- **Verify:** regenerate a bank with the vendored kernel and assert the signals are **byte-identical**
  to one generated with the stock tracker. Nothing else.
- **Depends on:** S13.
- **Done 2026-08-12.** `EGMTracker` overrides **`initialize` only** — upstream's `calc_ecg`
  dispatches through `self._compute`, so reassigning it there swaps the arithmetic while inheriting
  the diffusion-kernel call, step scheduling and output accumulation. Overriding `calc_ecg` would
  have meant copying real logic we have no reason to own. 3D **raises** rather than falling back to
  upstream's kernel, which is how a future 3D geometry would otherwise end up on arithmetic that
  never had this review.
  **The byte-identity check needed two guards to mean anything.** Identical code trivially gives
  identical output, so what it actually tests is the *plumbing*; a subclass that silently failed to
  install its kernel would inherit upstream's and match every byte. Hence an explicit assertion that
  `_compute is egm_kernel_2d`, and a second that the traces are not two matching piles of zeros.

### S39 — One axis convention across stimulus, electrodes and fibres (CL-169 + CL-170) ✅ (3–5 h)
- **Change:** fix the transpose *inside our kernel* — the repo holds **three** axis maps and they
  disagree. Ours is `x = j` (axis-1), `y = i` (axis-0), per `simulate/pseudo_egm.py` and the edge
  definitions in `specs.py`. Finitewave's kernel differences coordinate column 0 against axis-0, so
  feeding it `positions_mm / dr` reflects the electrode grid across the diagonal. Every pair ends up
  **perpendicular** to a `left` wavefront, both poles fire together, and the **near** field cancels —
  the `1.35e-6` "dead healthy tissue" blob.
  Then audit the **fibre tensor** the same way (`_build_tissue_2d` sets `fibers[...,0] = cos θ`): if
  component 0 is axis-0, `fiber_angle_rad = 0` runs fibres along our `y`, not the `x` a reader
  assumes — same root cause, third surface.
- **Verify:** regression test — a plane wave launched **parallel** to a known pair gives a large
  biphasic deflection; **perpendicular** gives ≈ 0. That is textbook bipolar directional sensitivity
  and it is the assertion that would have caught this. Plus: on a uniform density-0 patch the healthy
  EGM shows a **real local activation**, and per-pair `activation_position` **spreads** across the
  grid rather than clustering (CL-167's check that the pseudo-EGM is locally dominated). Anisotropy:
  a wave ∥ fibres is √3 ≈ 1.73x faster than ⊥ fibres, on the intended axis.
- **Blocks:** S38 — CV must be measured on a correctly-identified axis.
- **Depends on:** S37.
- **Done 2026-08-12.** Confirmed from Finitewave's source that it is **internally consistent** and
  simply opposite to us — "x" means axis-0 in `_compute_ecg_2d`, in `compute_weights` (`d_xx` acts on
  the `(i-1, j)` neighbour), and in `StimVoltageCoord.stimulate` (`mesh[x1:x2, y1:y2]`). So two of our
  three surfaces needed swapping and one did not.
  **The stimulus was immune, and the reason generalises:** it was named by *index* (`top` = a strip at
  low `i`) rather than by axis, so there was no physical `x`/`y` to mistranslate. That is now recorded
  in `architecture.md` as a convention for future code — **name geometric things by the index they act
  on**, and the error cannot be written.
  **Measured before/after** on a clean 40 mm patch, changing only the stimulus edge: `left` (along the
  pairs) went 1.35e-06 → **1.08e-01**, `top` (across) went 1.61e-01 → 1.34e-06. Exactly mirrored,
  which retires CL-168's "planar waves over uniform tissue are degenerate" hypothesis outright.
  **The fibre swap mattered more than a label:** `fibers[...,0]` acts on axis-0 = our `y`, so
  `fiber_angle_rad = 0` ran fibres along `+y` while every doc said `+x`. At `anisotropy_ratio = 3`
  that put the **fast conduction axis 90° from the intended one**, corrupting any CV measured along it.
  **S37's byte-identity gate was sharpened, not deleted:** our kernel given `(x, y, z)` must equal the
  stock kernel given `(y, x, z)` exactly, pinning the change as *precisely* a transpose. It moved to
  the fast suite by calling both kernels directly, on a non-square mesh so an `i`/`j` mix-up cannot
  hide behind symmetry.

### S40 — Restore the 1/r weighting (CL-166) ✅ (1–2 h)
- **Change:** one `sqrt`. The stock kernel divides by `d`, the **squared** grid distance, giving an
  effective `1/r²`; restoring `sqrt(d)` gives the `1/r` of the Laplacian form we compute the source
  for. Expose `distance_power` (default 1) as upstream's branch does, so the pre-fix `1/r²` banks
  remain reproducible for comparison — that is worth having when everything gets regenerated and the
  question is "how much did this actually change?". Add the missing `1/(4πσ_e)` prefactor to both
  this kernel and `compute_phi_e`; a constant scale, but it makes our output comparable with anything
  else computing a pseudo-EGM.
- **Why this is not a judgement call.** `1/r²` is correct when paired with the **gradient** ∇V_m,
  which is what openCARP does. Our source term is the **Laplacian** (the diffusion increment), so it
  pairs with `1/r`. Upstream's `solvers` branch reached the same conclusion independently — `sqrt`
  restored, `distance_power` defaulting to 1 — and Finitewave's own class docstring said *"the
  inverse of the distance"* all along.
- **Verify — on isotropic clean tissue only** (Daniel, 2026-08-12, option 1 of three). The vendored
  kernel and `compute_phi_e` agree numerically on the same V_m field, at `anisotropy_ratio = 1` and
  `density = 0`. **This is the step where `compute_phi_e` stops being decorative**: it has been
  unit-tested in isolation and never called in production, which is precisely how two kernels came to
  disagree on the physics unnoticed.
  **Why the tissue is constrained, and it is not a fudge.** The two compute *different Laplacians*:
  `compute_phi_e` applies a plain isotropic 5-point stencil with no tissue mask, while our kernel
  uses Finitewave's `diffusion_kernel(u_tr, u, model.weights, myo_indexes)` — the **anisotropic**
  stencil, masked to myocardium. At production settings (ratio 3, density up to 0.6) they must
  disagree, and a naive `allclose` would fail for entirely correct reasons.
  Constraining the tissue keeps the cross-check pointed at what it is *for*: arithmetic errors in the
  **weighting** and the **axes**, both of which are visible on the simplest possible tissue. The
  anisotropy and the mask are the solver's business and are already covered by S37's byte-identity
  gate against upstream.
  **Rejected alternatives.** Teaching `compute_phi_e` the tensor and the mask would make it a true
  production-settings reference, but the two implementations then converge toward each other and the
  independence that makes the check worth having erodes. Retiring `compute_phi_e` gives up the
  independent check entirely, and its non-Finitewave-backend justification — openCARP skipped,
  TorchCor Phase 2+ — is distant rather than dead.
  **The trap this avoids** is the one that created the situation: `compute_phi_e` was thoroughly
  tested on inputs that never resembled production. Promoting it to a reference without stating what
  it can and cannot reference would repeat exactly that.
- **Measured, 2026-08-13.** Cross-check agrees to `rtol = 1e-10` on a random field over a
  non-square grid, with a companion test proving the check fails at `distance_power = 2` — a
  reference comparison insensitive to the exponent would prove nothing about the weighting.
  Effect on a 20 mm clean-tissue run: amplitude **4.1× smaller** (median peak-to-peak
  `1.08e-1 → 2.62e-2`, most of it the `1/(4πσ_e)` prefactor), waveform correlation old-vs-new
  **0.961**, detected activation indices unchanged (10–15 both ways). So this is a scale and a
  mild re-weighting toward near nodes, not a new morphology — consistent with a near-field
  measurement where the nearest sources already dominate. It does **not** rescue the far-field
  problem on its own, which is why S38 still stands.
- **Depends on:** S39.

### S38 — Model calibration + one detection curve across corpora (CL-166 + CL-167 + CL-172/173/174) ☐ (6–10 h)
- **Re-scoped 2026-08-13/14; NOT blocked.** The premise below was wrong and measuring it first is
  what found that out. Full write-up, with derivations, measured knob laws, literature and figures:
  `intracardiac-platform/project/investigations/ap_model_calibration.md`.
  **What research gates, and it is less than it looks (CL-174, investigation §6.4).** Because the
  deliverable is a `calibrate(targets) → parameters` routine, the physiological numbers are
  *arguments*, not inputs to the code. **Both implementation steps are unblocked.** Research gates
  only the single call that writes the shipped default model file — the APD90 target (~180 ms
  AF-remodelled vs ~250 ms sinus), the anisotropy ratio (3.0 or 2.0), and ideally the
  `100u − 80` mV voltage mapping, which wants to ride the *same* regeneration wave rather than
  force a second one. That call is the last action before bank generation, which was already
  sequenced after all code in all repos — so the dependency resolves on its own.
  **Split accordingly:** S38a ✅ (the anisotropy fix) and S38b ✅ (the routine, config exposure,
  model-file plumbing, round-trip test, shipped values).

#### S38b — `calibrate(targets)`, model cards, config exposure, ρ 2.0 ✅ (6–9 h) — 2026-08-15
- **Shipped as ONE step at Daniel's call**, against my proposed split: *"if we don't put the hooks
  into the code I can't test the whole processing chain."* That is the S14 lesson restated — a
  surface whose effect only arrives later is untestable — and it applies here more than it did
  there, because the deliverable is a config path.
- **`simulate/calibration.py`** — `ModelTargets → calibrate() → SolvedParameters`, analytic.
  `eps` is *held* at the published 0.002, not solved: it is the free direction in a
  4-unknowns-from-3-targets system, and spending it on staying citable is the cheapest option
  (largest APD* ⇒ smallest K ⇒ smallest D ⇒ largest dt). Solving away from the measured `eps`
  raises rather than silently using constants measured elsewhere.
- **`simulate/model_cards.py`** — three-block cards (`targets` / `solved` / `measured`), resolved
  by shipped name or by path beside the config. **`verify_solved` re-runs the solve on every
  load** and raises on drift; the solve is analytic *precisely so* that guard is affordable
  per-load rather than CI-only. **No partial overrides** — restating a card-owned key under `run:`
  is an error, since a name that means two things is worse than no name.
- **`backends/finitewave/measure.py`** — CV and APD90 measured off V_m via the activation-time and
  action-potential trackers, not off the EGM: routing a conduction measurement through the
  electrode model is how a transpose masqueraded as a physics problem for two days.
- **RunConfig** gains `diffusion`, `membrane_eps`, `dt_model_units`, `dr_model_units` and the
  resolved `model_card`, with a **stability-bound check that raises** — an over-large `dt` does not
  crash, it writes a well-formed bank full of a diverged field. Defaults reproduce pre-S38b
  behaviour so existing constructions still mean what they meant.
- **Shipped card `af_remodelled_220ms`**: CV 80 cm/s, APD90 220 ms, ρ 2.0 → K 5.7098,
  D 7.8264, dt 0.001797. Roughly **1.9× the integration steps** of the old settings.
- **All six example configs restated `ap_time_unit_ms` and `anisotropy_ratio`**, so changing the
  constants alone would have been cosmetic — the identical trap to S12's `trace_duration_ms`.
  They now name the card and omit the ratio.
- **Verify:** round-trip (solve → simulate → measure → assert targets); realized anisotropy at
  2.0; **no second deflection inside the CROPPED window at the smallest `p`** (CL-178 — the
  original 51 ms finding was measured on the raw capture and left the window arithmetic implicit);
  plus card drift, rounding tolerance, unstable-`dt` refusal and card/RunConfig disagreement.
- **⚠ ABSTRACTION AUDIT (Daniel, 2026-08-15) — S38b put cell-model parameters on the wrong
  objects, and D2 had already ruled on it.** Written up below because the mapping matters more than
  the fix: Courtemanche arrives at SEP5, and a parameter in the wrong home is a migration later.

  **The rule: a parameter belongs to the *narrowest* thing that can change it independently.**
  Four axes vary independently in this repo, so there are four homes:

  | Axis | Varies when… | Home | Examples |
  |---|---|---|---|
  | **Physiology** | never — it is what we are trying to reproduce | `simulate/` targets, backend- and model-agnostic | `conduction_velocity_cm_s`, `apd90_ms` |
  | **Tissue structure** | the patch changes | `GeometrySpec` | `anisotropy_ratio`, `fiber_angle_rad`, `dr_mm` |
  | **Cell model** | AP → Courtemanche | **`CellModelSpec`** (D2) | `eps`, `time_unit_ms`, the AP measured constants, the AP solve |
  | **Backend / scheme** | Finitewave → TorchCor | `RunConfig` + backend | `dt`, `dr_model_units`, `capture_oversample` |

  **Why each of those boundaries is real, not taxonomic:**
  - *Physiology vs cell model.* "APD90 = 220 ms" is a fact about atrium. `eps = 0.002` is an
    artifact of one phenomenological model — Courtemanche has no `eps` at all, and being
    **dimensional** it has no `time_unit_ms` either. A target survives a model swap; a knob does
    not. That is the test for which side of the line something sits on.
  - *Cell model vs backend.* Courtemanche runs on Finitewave **and** on TorchCor, so cell model is
    orthogonal to backend rather than nested under it. Hence `simulate/cell_models.py`, importing no
    backend library, and **not** `backends/finitewave/`.
  - *Tissue vs cell model.* Anisotropy is fibre architecture, not membrane kinetics. It already
    lives on `Patch2DGeometry`.
  - *Backend.* `dr_model_units` presumes a **regular grid**; a TorchCor FEM backend has no such
    quantity. `dt` is the price of an explicit scheme, not a property of tissue.

  **What S38b got wrong, measured against that table:**
  1. `RunConfig.membrane_eps`, `.diffusion`, `.dt_model_units` — cell-model parameters on the
     **shared** backend config. D2 forbids exactly this: *"would put model-specific parameters …
     into a config object that has no place for them."* A Courtemanche run would carry an `eps`
     field that means nothing.
  2. `MODEL_UNIT_APD90`, `MODEL_UNIT_CV`, `AP_EPS_PUBLISHED`, `SolvedParameters` and `calibrate()`
     in `simulate/calibration.py` — AP-specific measurements and an AP-specific solve, sitting in a
     module whose name claims universality. `calibrate`'s arithmetic (`K = APD/APD*`) only exists
     because AP is dimensionless; for Courtemanche the same function is a different equation.
  3. `ModelTargets.anisotropy_ratio` — **a duplicate** of `Patch2DGeometry.anisotropy_ratio`, with
     nothing reconciling the two, and `calibrate()` never reads it. Two homes for one number.
  4. `RunConfig.model_card` — the card names a *cell-model* parameterisation, so it should hang off
     the cell-model spec.

  **What is correctly placed, and why:** `ModelTargets` / `MeasuredValues` / `ModelCard` as
  *concepts* are genuinely universal — every cell model targets a conduction velocity and an APD,
  and every one of them benefits from a named, verified parameterisation. Only the **payload** of
  `solved` is model-specific, so the card stays generic over it. `measure.py` is correctly under
  `backends/finitewave/` because measuring requires integrating (Guardrail 1) — although what it
  measures is universal, so a second backend gets its own `measure` and the round-trip test becomes
  backend-parameterised.

  **Fixed in S38c ✅ (2026-08-15)** — D2's `CellModelSpec` implemented, pulling part
  of SEP5 forward because it was cheaper than migrating after banks exist.
  - `simulate/cell_models.py` — `CellModelSpec` Protocol + `AlievPanfilovCellModel`,
    carrying `time_unit_ms`, `diffusion`, `eps`, `dt_model_units`, plus the AP
    measured constants and `calibrate_aliev_panfilov`. **Not** under `backends/`:
    Courtemanche runs on Finitewave *and* TorchCor, so cell model is orthogonal to
    backend, and this module imports no solver.
  - **`ms_to_model_time()` is the seam that lets Courtemanche join.** Callers wrote
    `duration_ms / ap_time_unit_ms` — an Aliev-Panfilov idiom leaking into the
    runner, and simply wrong for a dimensional model. Asking the model instead
    means Courtemanche answers *the same number* and nothing upstream changes.
  - `RunConfig` keeps only trace/capture timing and `dr_model_units`; the Protocol
    takes `cell_model` as the fifth spec, and `FinitewaveBackend` **refuses an
    unfamiliar model by name** — duck-typing into one would return numbers instead
    of an error. The stability check moved to the backend, the one place where the
    model's `diffusion` and the backend's `dr` are both in hand.
  - `ModelTargets` lost `anisotropy_ratio` (a duplicate of the geometry's, never
    read by the solve).
- **Verified end-to-end 2026-08-15, first clean run:** targets CV 80 cm/s /
  APD90 220 ms → **measured 83.3 cm/s, 219.8 ms**, now recorded in the card's
  `measured:` block. **APD lands 0.09 % off**, which is the separability the
  two-target solve depends on — had `eps` and `K` been entangled, this is where it
  would have shown. **CV runs 4.1 % high**, the discretization excess predicted as
  "~1.5 % at D≈5, 8 % at D≈10"; the shipped `D` is 7.83, so the prediction held.
  83.3 also sits inside Hansson's 88 ± 9 cm/s, so the *achieved* value is arguably
  closer to atrium than the target asked for.
- **Cost of getting there, worth recording for the retrospective.** S38c took
  **six review round-trips**, almost all import errors, stale `type: ignore`
  comments and lint ordering — the class a local `pytest`/`ruff`/`mypy` catches in
  seconds. It was written blind in the Cowork sandbox, whose 45 s-per-call ceiling
  cannot run the solver. Claude Code was authenticated on 2026-08-15 and closed it
  in one pass. **The write-run-fix loop belongs there; design and investigation
  stay in Cowork.** `CLAUDE.md` now carries the conventions so it starts warm.
- **Left for later, deliberately:** per-simulation APD *sampling* over 200–260 ms. At 220 > T = 192
  the marker is already outside the window, so sampling buys physiological variation rather than
  correctness — and it needs a fifth per-simulation strategy spec through the backend Protocol,
  which is its own step. Same for `fiber_angle_rad` sampling, which CL-176 wants **after** the
  point stimulus lands so the distributions can be re-measured first.

#### S38a — make `anisotropy_ratio` operative ✅ (2 h) — done 2026-08-14
- **Change:** `_configure_anisotropy_2d` builds an `AsymmetricStencil2D` with `D_al = 1.0`,
  `D_ac = 1/ratio²` and assigns it to `model.stencil`. `base_diffusion` removed — absolute scale is
  `model.D_model`'s job, and the API already has three multipliers on one coefficient.
  Docstring rewritten; the v0.2.0 "now prescriptive" claim retracted in `constants.py` and
  `specs.py`; the S39 test's magnitude reasoning corrected (it requested ratio 9 and passed on the
  stencil default — the *direction* it asserts was always real, so the transpose fix stayed
  verified, but a test that would not fail if its own input were ignored was proving less than it
  claimed).
- **Which axis is held fixed — a deliberate change of meaning.** `D_al` pinned, ratio into `D_ac`,
  so anisotropy never disturbs the along-fibre velocity. The old docstring's geometric-mean
  invariant would have made this knob move the axis we calibrate CV on.
- **Verified 2026-08-14.** Suite + ruff + mypy green. End-to-end via four configs
  (`configs/aniso_{across,along}_r{1,3}.yaml`) and `configs/compare_banks.py`: the ACROSS pair
  **differs** (`max|Δ|/max|A| = 0.988`) where before the fix it was byte-identical; the ALONG pair
  on clean tissue is **exactly identical** (`0.000e+00`), confirming the pinning.
- **Two things the verification itself taught, both worth carrying:**
  **(1) The along-fibre invariance is a plane-wave property.** The first draft of the ALONG configs
  used fibrotic tissue and legitimately failed — the wave diffracts around holes and samples the
  transverse tensor entry continuously. Measured `1.075` there, *larger* than the across-fibre
  case. **So ρ = 2.0 in S38b changes every bank, not only transversely-propagating ones.**
  **(2) `compare_banks.py` shipped a vacuous green** — a missing-bank skip counted as a pass and it
  reported "all checks passed" against an empty directory. Fixed to return "nothing checked".
  Precisely the failure mode this whole thread exists to remove, found in the tool built to find it.
  **(a) CV needs no fix.** Along the fibre axis it measures **81.8 cm/s**; across, 26.5 cm/s. The
  20–23 cm/s below was the **across**-fibre axis — precisely the confound CL-170 warned about and
  the reason this step said "measure on the along-fibre axis, only identifiable after S37." Acting
  on it would have driven the along-fibre velocity to ~245 cm/s.
  **(b) APD90 is 51 ms** against 200–300 ms, because the 2026-06-10 calibration bought CV by
  shrinking `ap_time_unit_ms` 12.9 → 1.97 and `APD ∝ K`. **Not cosmetic**: the repolarisation
  deflection lands *inside* the 192 ms window at **51.2 ± 1.0 ms** after every activation, at 9–21 %
  of the activation amplitude — a fixed-offset marker present in all synthetic traces and no real
  ones, i.e. a shortcut feature. That makes this a correctness fix.
  **(c) `anisotropy_ratio` is a no-op.** `_configure_anisotropy_2d` writes `model.D_al`/`D_ac`;
  Finitewave reads them off the **stencil**. Realized ratio is 3.093 for requested 1.0, 3.0 and 6.0
  alike. Since our config always asked for 3.0, **no bank is corrupted** — a documented knob has
  simply been inert, and the v0.2.0 changelog claim that it was fixed needs retracting.
  **(d) A fourth knob: `eps`.** Moves APD 1.9× over a 10× range while CV shifts under 2 %, which is
  what makes both targets reachable at once. Measured candidate: `eps = 0.002` (the published AP
  value), `K = 4.672`, `D_model = 5.24`, `dt = 0.0028` → **81.2 cm/s, 180.7 ms**, ~1.6× the steps.
  **ANSWERED by research, CL-176 + CL-178 (2026-08-14):**
  - **APD90 = ~220 ms, SAMPLED 200–260 per simulation** — not the 180 I proposed. My 180 **failed
    the rule the fix exists to satisfy**: repolarisation leaves the window iff `APD > T(1−p)`, so
    `APD ≥ T = 192` is the unconditional bound and 180 needs `p > 0.0625`. Franz *JACC* 1997 gives
    219–245 ms at CL 800 for AF/flutter; the ~150–180 figures are short-CL rates we do not
    simulate. **Sample, don't fix — the defect is the *fixed* offset, not repolarisation itself.**
  - **ρ = 2.0** (Hansson *Eur Heart J* 1998: RA free wall 88 ± 9 cm/s, only weakly
    direction-dependent; high ratios belong to bundles, not working myocardium). Puts transverse at
    41 cm/s inside 30–50 where 3.0 forces 26.5, and validates our 81.8 cm/s along-fibre directly.
  - **Voltage scaling: DON'T** — it is a pure ×100 gain (the −80 cancels in any Laplacian-like
    difference) and yields no real mV. **Leaves the pre-generation gate entirely**, since a pure
    scale can be applied post-hoc.
  - **STU4 searches `(CV_∥, ρ)` only, holding APD and `eps` fixed** — once the shortcut is fixed,
    APD is unobservable in the window *by construction*, so searching it adds a direction the data
    cannot speak to. This corrects my §5b.3 proposal.
  - **CL-178 — assert on the CROPPED trace.** The sim runs far longer than `T` and the window is
    cut afterwards, so the round-trip "no second deflection" check must run on the cropped
    192-sample trace at **smallest `p` + largest patch**, not the raw capture.
  - **New: fibre angle is a degeneracy a point stimulus will NOT fix.** At `fiber_angle_rad = 0`
    every bipole is exactly parallel to the fibres, and max bipolar delay differs by a factor of ρ
    between parallel and perpendicular. **Sample `fiber_angle_rad` uniform on `[0, π)`** — after
    the point stimulus lands, then re-measure.
  **Still open (not gating):** fibrosis as insulating holes vs graded conductivity (interstitial ⇒
  graded; an SR item, not Phase 1.5) and a synthetic-vs-real corpus-difference audit (endorsed;
  wants an FB entry).
  **Shape once unblocked:** two steps, not one — (1) the anisotropy fix, provably inert at our
  default and therefore a clean low-risk commit; (2) the recalibration, which changes every future
  bank. Daniel leaned toward folding them together pending this investigation; the split is
  recommended because mixing a no-op change with a regenerate-everything change makes the second
  harder to review. Detection-curve unification (CL-167) is unaffected and can ride with either.
  **(e) The calibration knobs are hardcoded and must move to config as part of this (CL-173).**
  Only `ap_time_unit_ms` is config-reachable today; `D_model`, `eps`, `dt`, `dr` and
  `tissue.conductivity` are module constants or untouched library defaults. Since `D_model` and
  `eps` change every sample of every trace, the no-hardcoding rule already requires it — and it is
  what lets STU4 treat them as estimation parameters later. Tiered: physics exposed freely;
  **`dt`/`dr` derived from the stability bound and hard-errored on violation**, because exceeding
  it does not raise, it writes a well-formed bank full of a diverged field; AP reaction parameters
  behind an opt-in that stamps `backend_metadata`. Everything exposed gets stamped, or banks stop
  being reproducible. See `ap_model_calibration.md` §5b.
  **(f) The deliverable is a routine, not four constants (CL-174).** Settled with Daniel
  2026-08-14: parameterise by **physical targets** (CV, APD, anisotropy) and ship
  `calibrate(targets) → parameters`, called **once** at the agreed targets. In target space the
  solve is four unknowns from three targets, so it has one free direction, and spending it on
  minimum cost makes the old "Option A" the argmin rather than a judgement call — Options B and C
  stop being options at all (B is a worse point on A's family; C is a *constraint*, since the
  space-scale constant is fixed by cells-per-electrode-spacing). Verified by **round-trip**: solve,
  simulate, measure, assert the targets come back — the test that would have caught the 2026-06-10
  error. The routine is also what STU4 needs per proposal, so it is reusable rather than throwaway.
  **(g) AP parameters live in their own model file** referenced from the main config via the
  existing `_resolve_path` convention, with standard parameterisations shipped in-package behind
  short names. Banks record the **resolved contents**, not the path; **no partial overrides**.
  See `ap_model_calibration.md` §5b.4.
- **Discovered mid-wave 2026-08-12.** Local step, as S37.
- ~~**Change:** conduction velocity measures **20–23 cm/s** against ~50–100 cm/s for human atrium.~~
  *(superseded — see the block above; retained so the reasoning trail stays readable)*
  Aliev–Panfilov is dimensionless, so physical CV is entirely our `D` / `dr_mm` / `ap_time_unit_ms`
  choice: `CV ∝ √D`, `CV ∝ dr_mm`, `CV ∝ 1/ap_time_unit_ms` — but **`APD ∝ ap_time_unit_ms`**, so
  raise `D` (or coarsen `dr_mm`) and **leave `ap_time_unit_ms` alone**, or APD rescales with it.
  Watch the explicit-scheme stability bound `dt ≤ dr²/(4D)`; a 9x `D` needs ~9x smaller `dt` unless
  `dr` coarsens, which relaxes it. Measure on the **along-fibre** axis, which is only identifiable
  after S37.
  Second half: **unify the detection curve with IAFDB** (CL-167). `cropping.py` currently hardcodes
  `RectifiedDerivative` with a docstring arguing it is fine for synthetic; CL-167 corrects that —
  `activation_position` must be the *same measurand* on both corpora, so the curve is a shared
  choice, not a per-corpus convenience. iafdb-pipeline dispatches all three curves from config, so
  this needs a decision on which, not just a code change.
- **Verify:** plane-wave sweep of `D` lands CV in the 50–100 cm/s band on the correct axis; APD
  unchanged from before the sweep; the solver stays stable; both corpora name the same curve.
- **Depends on:** S40 (CV must be measured on a correctly-identified axis, and after the
  weighting settles — a `1/r` kernel changes what the detector sees).

### S15 — Crop per trace, record the realized position, anchoring flag (SEP2 + SEP10) ✅ (3–5 h)
- **Change:** call the windower per bipolar trace. **Per trace, not per simulation** — the wave sweeps
  the grid so pairs activate at different times; one per-sim offset would leave the position
  uncontrolled for most pairs. Synthetic goes through `window_train` as a **train of one** (CL-134),
  the same function IAFDB uses, which is what keeps T1 a detected-vs-detected comparison.
  Write `AnchoredWindow.realized_position` into the `synthetic_bank` column that has been waiting for
  it since S2 — **realized, never requested**; they differ whenever `p·(T−1)` is not an integer, and
  SIG1's docstring is explicit that only the realized value is meaningful downstream.
  **SEP10 rides along here**, and that is the point: SIG1's `UniformPositionGenerator` is
  **point-collapsible** (`(0.5, 0.5)` *is* the fixed arm, `is_fixed()` reports it), so both arms of the
  §8.9 A/B are one class and the "flag" is a config value, not a code path — and it only becomes
  *testable* once cropping runs, so it has no meaningful commit of its own. Needs **no
  egm-classifier change**: both arms are just two banks.
  `docs/simulation_theory.md` gains the sizing derivation, why the front is flat and the back
  overhangs, and that the synthetic side **detects** (the stimulus time is a coarse prior, not the
  anchor — CL-130/CL-147). Cross-reference `activation_splitting_method.md`; CHANGELOG.
- **Verify:** realized position lands within tolerance of the requested fraction for every pair; the
  column is populated; a Wave-1 bank (absent column) still reads; two configs (anchored / varied)
  produce banks whose realized-position distributions are a point mass and a spread respectively;
  pr_checklist passes.
- **Depends on:** S14. **Absorbs S14** — they committed together.
- **Done 2026-08-12, after two rounds of the same mistake.** The step raised at the shipped geometry
  and stayed blocked for a day while CL-163→170 tracked the cause. Two corrections worth carrying
  forward, because they are the *same* error twice:
  **(1) The front budget.** S14 sized the back and argued the front away, from activation indices
  that were detection artifacts of the transpose bug (CL-169). The fix is a **stimulus delay**
  `D = k(p_hi)`, which guarantees the front for any geometry because travel time is non-negative —
  no CV term, so it survives S38's recalibration untouched.
  **(2) The travel allowance.** `V` was first set to `T`, justified by a **clean-tissue** conduction
  velocity. Daniel's 30–60 % fibrosis run broke it (activation at 329, allowance 192): fibrosis
  slows conduction, so clean tissue is the *fastest* case and the wrong end from which to bound a
  maximum. Now `2T`, overridable via `run.travel_allowance_ms`, and stress-tested by Daniel on a
  wide position range at high density.
  **Both were bounding a worst case with a best-case number.** `V` is still an assumption rather
  than a bound; deriving it from patch size and a measured CV becomes possible after S38.
- **⏸ BLOCKED 2026-08-12 on CL-163 — the front budget.** Code is written and works; it raises at the
  geometry we ship. A window with the activation at `p` needs `round(p·(T−1))` samples **before** it —
  96 at `p=0.5`. Measured activation arrival on a clean 40 mm patch, left-edge stim, 5×5 grid:
  **index 54–150**, so the earliest pair has 54 and needs 96. At 10 mm it is 2–34. Not a test artifact.
  **S14's sizing reasoning was wrong on the front.** It sized the back carefully and assumed the front
  away ("the front is flat, rely on natural lead-in"). It is not enough at any patch size we run.
  **A stimulus delay does not fix it** (my first proposal; Daniel caught it). `φ_e ∝ Σ I_m,i/r_i` sums
  over the whole mesh, so while nothing depolarises the lead-in is flat — a zero-pad the solver
  generated, reintroducing the regularity S14 deleted the zero-pad to remove.
  **Daniel's alternative:** enlarge the patch, hold 2 mm spacing — the wave crosses more tissue, so
  activation lands later *and* the lead-in holds a real approaching wavefront. Open wrinkle raised to
  research: **bipolar subtraction rejects far field**, so the approach may be weak in bipolar even on a
  big patch, which would make both options ways of buying flat lead-in and turn this into a study-design
  question about how back-bounded `p` may be. Cost note: 40→60 mm is ~3x per simulation, a §8 input.

### S16 — Positional-sensitivity probe bank + configurable detection curve (SEP13) ✅ (4–8 h)

**Done 2026-08-16** as S16a (curve) then S16b (probe), four commits. Gate green, and Daniel
generated a probe bank and **opened it in egm-studio** — the end-to-end check that matters, since
egm-studio is the consumer whose uniqueness rule the first attempt violated. 22 probe tests +
19 curve tests; every verify bullet below has one.

**S16b was reworked once, and the lesson is worth more than the step.** The first cut tiled the
trace axis to `(pair × grid point)` inside one simulation and added a `pair_index_per_trace` field
to keep it straight. It shipped a bank with `pair_index` running 0–59 against 20 real pairs.
I had reviewed that design twice and written a careful audit table for keeping the three array
lengths consistent — which made the wrong shape *look* rigorous. **The audit was real; it was
auditing a structure that should not have existed.** Daniel found it from the generated data in
about a minute by asking whether `pair_index` still meant what it says. Checking that a structure
is internally consistent is not the same as checking it is the right structure.

**One scope reduction, recorded because it was not agreed:** `grid.pair_indices` — the optional
pair subset Daniel selected when choosing the probe shape — **did not ship**. It is rejected by
name with a pointer, which is the right mechanism, and the workaround (select pairs when analysing
the bank) is real. But a subset would not reintroduce the tiling; it would need the result to carry
a *subset index*, which stays unique and legal. So the knob is deferrable, not impossible, and the
shipped error message states the reason more strongly than the facts support. Revisit if a sweep
ever gets expensive enough that cropping 20 pairs instead of 3 matters.
- **Change:** a probe generation mode — run **one** simulation, then emit one trace per grid point by
  cropping that same simulation output at each offset, morphology and seed held constant. Reuses
  S14's sizing rule (the sim must reach the widest grid point) and SEP12's writer; **no schema
  change** — the grid value lands in the existing `activation_position` column. Ships with its
  config block + CLI wiring, and a `docs/usage.md` probe section explaining what the bank is *for*
  (it is not training data); CHANGELOG.

**Where the grid lives — settled 2026-08-16 (Daniel), and it moved.** The entry originally put the
grid in a **top-level `probe:` block, mutually exclusive with `activation_position:`**. S16a
invalidated that: the detection curve now lives *inside* `activation_position:`, so a probe run that
excluded that block would have nowhere to read its curve from, while this entry also requires the
probe to inherit the run's curve. Both could not hold.

The grid is therefore an **optional sub-block of `activation_position:`**, mutually exclusive with
`low`/`high`:

```yaml
activation_position:
  detection:                 # shared by both modes, for free
    curve: botteron_envelope
  grid:                      # OR low/high, never both
    low: 0.2
    high: 0.8
    n_points: 13
    pair_indices: [0, 5, 10]   # optional; default is every pair
```

One home for *how the activation is positioned in the window* — a probe grid is a **deterministic**
position policy where `low`/`high` is a random one. The mutual exclusion becomes structural instead
of a cross-block check, and the curve is shared without duplicating the detection schema. This is
the same argument that moved `detection:` in S16a; applying it once rather than repeating the error
one step later.

**Exactly one example config, and it is new (Daniel).** A probe bank is an occasional diagnostic,
not part of a normal run, so the grid must **not** be sprinkled through the existing examples —
they stay untouched. One new `examples/synthegm_probe.yaml` is the single place it is demonstrated.
That also lands part of S36's deliberate example set ("one probe (SEP13)") early; **fold that row
of S36 into this step rather than leaving it to be redone.**

**Folded in 2026-08-16 (Daniel): make the detection curve configurable.** `crop_traces` has always
taken a `preprocessor`, and `run_single` has never passed one — so every synthetic bank in the
project's history was windowed with `RectifiedDerivative` and no config could say otherwise. Same
family as the `anisotropy_ratio` no-op, though milder: that one silently ignored a value the config
*asked for*; this one cannot be asked at all, so no bank is wrong — the seam is simply unreachable.

**Why it belongs in this step rather than its own.** The probe detects too. Shipping S16 with the
curve hardcoded and making it configurable later means writing the probe's detection call twice — and
worse, the probe must inherit **the run's** curve rather than defaulting independently, or a sweep
would characterise a bank it does not share a detector with. One wiring, done once.

**What it unblocks:** train a model on synthetic windowed with curve X, evaluate against an IAFDB
bank windowed with curve Y, and measure whether matched-vs-mismatched windowing moves the result.
That is the evidence CL-167's unification ruling currently lacks — it was decided on the argument
that `activation_position` *must* be the same measurand on both corpora, never measured.

**Match iafdb-pipeline's vocabulary exactly — this is the load-bearing detail.** It already
dispatches all three curves through `build_preprocessor(cfg, *, fs_hz)`:

```yaml
activation:
  detection:
    curve: botteron_envelope     # | rectified_derivative | teager_kaiser
    botteron_band_hz:    [40.0, 250.0]
    botteron_lowpass_hz: 20.0
```

Same block, same curve names, same parameter names. A different spelling on the synthetic side would
put a translation step inside every cross-corpus comparison, which is where the mistakes go.
**Default stays `rectified_derivative`**, so every shipped config and every existing test is
unchanged and the absence of the block still means today's behaviour.

**Copy the curve subset, NOT iafdb's whole `DetectionConfig`.** Theirs also carries
`threshold_rule` / `threshold_c` / `threshold_lam` / `threshold_q` / `min_prominence` /
`refractory_ms` / `refine_curve` / `refine_radius_ms`, because IAFDB runs
`detect_activation_train` — preprocess → threshold → select → suppress → refine. The synthetic side
runs `detect_activation`, which is `argmax g` on a trace with **exactly one activation by
construction**: no threshold, no candidate selection, no refractory rule. Accepting those keys here
would add config surface that provably does nothing — which is the same unreachable-seam defect this
step exists to remove, freshly minted. Take `curve`, `botteron_band_hz`, `botteron_lowpass_hz`, and
**reject the rest by name** with a message saying they belong to multi-activation detection.

**`BotteronEnvelope(fs=...)` takes `output_fs_hz`, not the capture rate.** The runner downsamples
(step 2) before it crops (step 5), so the trace the detector sees is already at the output rate.
Passing `fs_capture_hz` would mis-scale the band and low-pass by the oversample factor and still run
without complaint — a wrong number, not an error.

**Provenance is deliberately deferred — FB-35.** The curve changes *where the window is cut*, so it
changes the stored waveform, and neither bank schema has anywhere to record it. Recording it properly
is an egm-contracts change flowing into most of the constellation, which per the schema-migration
rule ships as its own wave. **Daniel's call (2026-08-16): not now.** Phase 1.5 has grown a lot during
implementation and getting to the studies outranks complete provenance. **Until FB-35 lands, the
bank's `description` is the record of which curve produced it, maintained by hand.** `docs/usage.md`
must say so plainly, next to the config block, rather than leaving a reader to discover it.
**Do not route the curve into `simulations/backend` as a workaround** — that object is the
simulator's capture knobs, and a wrong home reads as authoritative in a way that no home does not.

*Small assist, veto if it is scope creep:* `_format_result` already prints a run summary, so add the
resolved curve to it. Zero schema impact, and it puts the string on screen at the moment the
description is being written.
- **Crop exactly, don't re-detect (see D6):** detect the activation **once** per pair on the source
  trace, then place it by exact integer shift per grid point. Grid stated in fractions, **snapped to
  the sample lattice**, snapped value is the grid of record — see D6's 2026-08-16 amendment for why
  a fractional grid cannot be hit exactly and what "exact" means instead.

**egm-signal already carries the seam.** `window_train(signal, activation_train, *, ...)` is public
and its docstring names this exact caller — *"a probe sweeping crop offsets over one simulation,
which must detect once on the source rather than re-detect per crop."* So the probe calls
`detect_activation` once per pair and then `window_train` per grid point with a point-collapsed
`UniformPositionGenerator(p, p)`. **Not a hand-rolled slice**: routing through the shared path is
what keeps probe windows and training windows the same geometry (CL-134), and it hands back
`in_bounds` and the bounds diagnostics for free rather than reimplementing them.

**No egm-signal change is needed** — checked, not assumed: `window_train`, `detect_activation` and
`UniformPositionGenerator` are all in `myocard_egm_signal.__all__`, and `detect_activation(signal, *,
preprocessor)` is the same call `SingleActivationWindower._detect` makes internally. Handing it
`cropping.default_preprocessor()` makes probe detection identical to training detection by
construction rather than by matching two configurations. **S16 is a single-repo step.**

**REWORKED 2026-08-16 after the first probe bank — one grid point is one `simulation_id`.**

The first implementation made the trace axis `(pair × grid point)` inside a single simulation, tiled
the per-trace arrays, and added a `pair_index_per_trace` field to carry the mapping. Daniel generated
a bank and found `pair_index` running **0–59 against 20 real pairs**. His reading was the right one:
`pair_index` had been shoehorned into doing *trace-identity* work it was never defined for.

**It is not a matter of taste — the invariant is enforced downstream.**
`egm-studio/loaders/synthetic_bank.py` matches ClassifierBank traces to their θ companion on
`(simulation_id, pair_index)` "rather than by position", and **raises if either side's key is
non-unique**. So `(simulation_id, pair_index)` is the de-facto composite primary key of the bank
pair, and the tiled probe bank is unloadable by the one consumer it exists to serve (STU8).

**The fix — and it deletes code rather than adding it.** A probe emits **N logical simulations that
differ only in crop offset**, computed efficiently by reusing one solve. The single solve is an
implementation detail, not the unit of identity. Consequences, all simplifications:

- every `SimulationResult` has `n_pairs` traces again, so `pair_index` means 0–19 and nothing tiles;
- `bipolar_pair_midpoints_mm`, `activation_positions` and the label array go back to per-pair length,
  so `LabelPolicy.apply` needs no thought at all;
- **`pair_index_per_trace` is deleted.** It was the wrong shape — a widening bought to support a
  layout that should not have existed;
- the sweep is identified by the **shared `seed`**, which the schema already carries per simulation
  and does not require to be unique.
- **Cost, to be stated in `docs/usage.md`:** the bank reports N simulations where the config asked
  for one, and patient-aware splitting would treat grid points as separate patients. Harmless — a
  probe bank is never training data — but it must be written down, not discovered.

**Collapse the duplicate `pair_index` computation while here.** The 0–59 values came from
`builders.py:238` building the ClassifierBank with `for pair_idx in range(result.n_pairs)` while the
θ path used `dataset_result.pair_indices`: **one quantity, two computations, only one updated.**
After this rework `range(result.n_pairs)` becomes correct again — which means the symptom disappears
on its own and the landmine stays armed. Make both banks read the *same* array.
- **`run_single` needs the capture, not the trace.** Without a position generator it truncates to the
  leading `T` samples, so the probe cannot reuse that path as-is — the sweep needs the full sized
  capture to cut from. The grid goes in through the same parameter slot as the position generator,
  which is what makes them mutually exclusive by construction: two ways to set one trace's position
  is the two-sources-for-one-number trap `_build_run_config` already refuses for the model card.
  A grid with `n_simulations > 1` errors rather than quietly running one — a probe answers a
  question about *one* substrate, so a second simulation is a mis-specified study, not extra data.
- **Sizing needs no new arithmetic.** Front is bought by `D = k(p_hi)` and back by
  `N = D + V + T − k(p_lo)`; feeding the grid's snapped max/min in as `p_hi`/`p_lo` is exactly the
  existing call. The grid is fixed rather than sampled, which makes the bound tighter, not different.
- **Verify — the curve (S16a):** setting `detection.curve` changes **the stored signal**, asserted on
  a bank-to-bank diff, not on the config object; the three curve names all dispatch and an unknown
  name errors listing the valid ones; the Botteron parameters reach the constructed preprocessor;
  **omitting the block reproduces the existing bank byte-for-byte**, which is what makes the default
  claim checkable rather than asserted.
- **Verify — the probe (S16b):** the `activation_position` column equals the snapped grid **exactly,
  compared in float32**, one value per logical simulation; colliding grid points error, naming the
  pair; a grid point outside the sim's sizing errors with the existing front/back diagnostic rather
  than clipping; **`(simulation_id, pair_index)` is unique across the bank** — asserted directly,
  because egm-studio raises on a duplicate and this repo cannot import it to find out; every trace in
  one sweep carries the identical `sim_seed` while `simulation_id` runs `0…n_grid−1`; `pair_index`
  stays within `[0, n_pairs)` **and the ClassifierBank and θ bank agree on it trace for trace** — the
  defect that prompted the rework was the two disagreeing, which no single-bank assert would catch;
  the signal at two grid points is the same waveform at different offsets (**cross-correlation peak
  at exactly the expected sample lag**, not eyeball, not "looks similar"); a probe run inherits the
  run's configured curve rather than the default; the example probe config runs end-to-end and yields
  a bank STU8 can read.
- **Guard against the vacuous pass.** Three tests in this repo have passed for the wrong reason, all
  proxies that stopped tracking their referent. Two to watch here, and both are the *natural* way to
  write the test:
  - a cross-correlation assert comparing a trace **to itself** passes at lag 0 whatever the sweep
    did. The lag must be asserted to *be the grid difference*, non-zero, read from a genuinely
    different grid point.
  - a curve test that asserts `cfg.preprocessor.name == "botteron_envelope"` passes **with the
    runner still ignoring it** — which is the exact bug being fixed. The assert has to be on the
    emitted signal.
- **Split into S16a (curve) then S16b (probe) — reconsidered after folding.** The original entry
  argued one step, because a probe sweep with no config path cannot run end to end. Adding the curve
  seam changes that: it is independently observable — a Botteron-windowed bank generated end to end —
  and has its own distinct verification, so it now passes the split test that the probe halves did
  not. Two further reasons to put the **curve first**: the probe must inherit an already-configurable
  curve, so building in this order writes that wiring once; and the curve seam is the half that
  unblocks the windowing study, so shipping it first reaches a study sooner, which is the stated
  priority. Collapse them into one commit if the review overhead is not worth it.
- **Depends on:** S15. Single-repo throughout; no egm-contracts change (FB-35).

### S17 — `CellModelSpec` Protocol + `AlievPanfilov` concrete (SEP5 · D2) ✅ **absorbed by S38c**
- **Shipped 2026-08-15 as part of S38c**, ahead of schedule and for a different reason: S38b had put
  `eps` / `diffusion` / `dt_model_units` / `ap_time_unit_ms` on `RunConfig` — exactly what D2 rules
  against — and the correction *is* this step. Every element landed: the `CellModelSpec` Protocol,
  the `AlievPanfilovCellModel` concrete, `SimulationBackend.simulate()` gaining `cell_model`, both
  implementors adopting it, and `RunConfig` losing `ap_time_unit_ms`.
- **One deviation from the entry as written:** the Protocol lives in `simulate/cell_models.py`, not
  `simulate/specs.py`. A cell model carries a *solve* and measured model-unit constants, which the
  other four specs do not; keeping it beside them would have pulled that arithmetic into the module
  that is otherwise pure data. The Protocol shape is unchanged, so the D2 decision stands.
- **Its "nothing changed numerically" verification was NOT available**, and that cost is worth
  recording: the entry argued for a separate commit precisely so the refactor could be checked
  against unchanged fixtures. Arriving inside S38c, the refactor landed in the same wave as a
  recalibration that moved every number deliberately, so the check had to be replaced by the
  round-trip test. **S18 therefore opens without the clean baseline this step was meant to leave
  behind** — the Courtemanche numbers get compared against published human-atrial ranges, not
  against a known-good prior fixture.
- **Depends on:** S10. **Total stays 43** — this is a step accounted for, not a step deleted.

### S18a — `SimulationSpecs` carries the cell model; delete the string-sniffing ✅ (1–2 h)

**Done 2026-08-15.** `SimulationSpecs` grew `cell_model` (the fifth spec, D2), the runner passes
the object it already had, `cell_model_model()` takes the spec, and both string-sniffs are gone —
`bank_config`'s identity dispatch *and* `builders._cell_model_from_backend_meta`, which named every
bank this project has written after finitewave's class. Full gate green (296 fast + 11 slow).

**The "nothing moved" check, and the one place the entry's wording could not be met.** A bank
regenerated before and after is **not byte-comparable as a file**: `created_utc` is stamped at write
time and HDF5 object headers carry `track_times`, so *two runs of unchanged code* already differ on
disk — verified, not assumed. The check is therefore exact **content** comparison — every group,
dataset, attribute, dtype and shape, floats compared bit-for-bit — with `created_utc` excluded and
nothing else. Both controls were run before trusting it: unchanged code twice → identical, and a
one-ULP float / one-character string edit → caught. Two configs, since an explicit `output.bank_id`
bypasses the derived-id path that also changed: **identical on both, both banks, all 118 keys.** The
"before" for the derived-id run came from `git archive HEAD` into a scratch tree shadowed by
`PYTHONPATH`, so it is genuinely the old code, not a reconstruction of it.

- **`ap_time_unit_ms` in `backend_metadata`: removed, after grepping every reader.** The only
  consumers were the sniffing mapper and `backend_model`'s exclusion list, which existed to keep it
  *out* of `params` — so it reached disk through no path and its removal moves no number. The
  backend's own `dt_ms = dt_model_units * ap_time_unit_ms` is untouched: it reads the cell model
  directly, never the metadata. The exclusion entry stays as a rule about where the fact lives.
- **`model_class` is still emitted, and no longer read by anything.** It is honest provenance about
  which finitewave class integrated the run; it was never in `params` either. Two tests pin the
  change: metadata that *lies* (`model_class="TotallyDifferentModel3D"`, `ap_time_unit_ms` × 3) must
  not disturb the bank. Both fail on HEAD for the stated reason — checked by running HEAD's own
  functions against the same lever: `"totally_different_model3_d"` and `5.91`.
- **Not done here:** the Courtemanche branch that the old mapper carried for a `model_class` string
  no backend emits is gone rather than re-keyed on a spec that does not exist yet. `cell_model_model`
  now refuses an unwired spec by name, which is the extension point S18b lands in.

**A behaviour-preserving refactor, and it exists to recover a check S17 lost.** S17's entry argued
a pure refactor earns its own commit *because* its whole verification is "nothing changed
numerically" — then S17 was absorbed into S38c, which moved every number on purpose, so that check
was never available. This is the same category of change and it can still be verified that way, so
take it **before** Courtemanche rather than folding it in.

- **Change:** `SimulationSpecs` bundles four specs and the cell model is not one of them, so
  `bank_config.cell_model_model()` recovers the model identity by **string-sniffing**
  `backend_metadata["model_class"]` and reading `ap_time_unit_ms` back out of backend metadata. Its
  own docstring says "Phase 1.5 has no `CellModelSpec` yet … when SEP5 lands, this takes the spec
  directly and the string sniffing goes" — S38c landed the spec and left the sniffing. Add
  `cell_model` to `SimulationSpecs`, have `bank_config` take the spec, delete the sniffing and the
  `ap_time_unit_ms` round-trip through `backend_metadata` that only exists to feed it.
- **Why it blocks Courtemanche:** that function is the dispatch point where a Courtemanche bank
  would be recognised. Building on a string-sniff means the new model is identified by the class
  name finitewave happens to use, which is a dependency on a third party's naming.
- **Verify:** **every existing fixture numerically unchanged** — that is the whole point; a
  regenerated AP bank must be byte-identical. `model_class` no longer read anywhere for identity.
- **Depends on:** S16.

### S18b — `Courtemanche` cell model, card and backend dispatch (SEP5) ✅ (5–9 h)

**Done 2026-08-15.** `CourtemancheCellModel`, `calibrate_courtemanche`, both card branches, the
`_build_model_2d` dispatch and the `courtemanche_control` card. Full gate green (324 fast + 14 slow, up from 296 + 11).

**The published vector reproduces, at the pinned protocol** — 50 beats at BCL 1 s, single cell,
20 mV/ms × 2 ms stimulus, `dt = 0.02 ms`. Against Wilhelms Table 1 C: amplitude **−3.5 %**, RMP
**+0.5 %**, APD50 **+2.7 %**, APD90 **−0.8 %**, dV/dt max **+14.2 %**. The last one is the outlier
and the stimulus is why — the maximum falls *inside* the 2 ms stimulus window, so it carries the
amplitude with it (measured 165 / 195 / 218 / 227 V/s at 12 / 15 / 21 / 30 mV/ms), and Wilhelms does
not state what it used. The test's tolerance on that quantity is 20 % and says so; the other four
are 3–15 %. Workman 2001's experimental 203 ± 11 V/s brackets our 213.

**Trap 3 is real, and it is now measured rather than argued.** CV was measured at three diffusions
spanning 16×: `D = 0.0385 → 24.13`, `0.154 → 57.40`, `0.616 → 121.66` cm/s. A pure `sqrt(D)` law
anchored at the middle point predicts 28.70 and 114.80 — so **−15.9 % and +6.0 %**, i.e. CV ~ `D^0.58`
rather than `D^0.50` at `dr = 0.25 mm`. The law is exact physics; the mesh is what bends it, and the
direction is right for under-resolution (smaller `D` ⇒ narrower wavefront ⇒ fewer cells across it).
The solved point is only 1.94× from the anchor, so the residual is small: solved for 80 cm/s,
**measured 82.9** (+3.6 %), recorded on the card the way S38b recorded Aliev-Panfilov's. **This is
the number S18c's convergence sweep should watch** — the exponent returning toward 0.50 is what
"converged" looks like.

**Two things landed that the entry did not call for, both because the alternative was a silent
no-op.** (1) `ModelTargets.apd90_ms` became `float | None` as the entry required, which raised the
question of what a *stated* APD target means on a card that cannot solve for one; it is checked
against the card's own `measured:` block (`MEASURED_APD_RTOL`, 5 %) rather than ignored, because
this repo has been bitten three times by a knob that quietly did nothing. S18c's matched card is
what will exercise it. (2) The conductance-scaling names are an explicit backend registry, and a
name outside it is refused — which surfaced the finding below at authoring time rather than at sweep
time.

**S18c blocker found early: `g_Kur_scale` cannot be applied by assignment.** finitewave 0.9.3
computes I_Kur's conductance *inside* the kernel as a function of voltage
(`gkur = 0.005 + 0.05 / (1 + exp(-(u - 15) / 13))`); there is no `gkur` parameter. Three of CL-180's
four cAF scalings (`I_to`, `I_CaL`, `I_K1`) are plain attributes and work; the fourth needs a kernel
change or a fork. The backend refuses the name rather than accepting it into a dead attribute, so
the severity sweep cannot silently move three currents and report four.

**One thing Courtemanche makes live that was filed under "Phase 2":** the capture→output downsample
has **no anti-alias filter**, justified in `docs/simulation_theory.md` by Aliev-Panfilov having
"minimal energy above a few hundred Hz". A 0.59 ms upstroke does not obviously satisfy that at a
1 kHz output. How much survives the pseudo-EGM's `1/r` spatial integration, which smooths heavily,
is **not measured** — the doc now says so rather than carrying the phase-number justification.
Worth a spectrum of a Courtemanche bank before S20's runtime characterisation.

**Not done here, deliberately:** `dt` and `dr` were both authored at what could be *measured*, not
chosen. `CRN_MAX_DT_MS = 0.02` comes from a convergence check (0.02 vs 0.01 moves every property by
≤ 1 %); `dr` stays at 0.25 mm because S18c owns that measurement, and the card records the pitch it
was solved at so `calibrate_courtemanche` and load-time verification both refuse another one.

**Reconciled 2026-08-16 — the original entry predated S38 and was wrong in two ways.** It called for
"the `cell_model:` config block and its dispatch in `cli/_config.py`". There is no such block: S38b
made the cell model come from a **model card** named by `backend.model`, and adding a parallel block
would restore precisely the two-sources-for-one-number failure the card exists to prevent.
Courtemanche arrives as a **card type**. It also placed the spec in `specs.py`; the fifth spec lives
in `cell_models.py`. The extension points are already cut and named — `parse_model_card` refuses an
unregistered type, and `verify_targets_against_solve` says "add a branch here when the model gains
a solve."

**Card shape: partial targets — CV solved, APD measured (Daniel, 2026-08-16).** The card was built
around a dimensionless model that needs a solve; Courtemanche is dimensional and mostly does not.
The physics splits cleanly and the card follows it:

- **CV stays solvable.** `CV ∝ √D` is a property of diffusion, not of the membrane, so a
  `conduction_velocity_cm_s` target inverts to `diffusion` and `dt` exactly as it does for
  Aliev–Panfilov. Load-time verification keeps working on this half.
- **APD does not.** There is no time-unit constant; APD90 falls out of the ionic equations and the
  conductance scalings, with no closed-form inverse. It is recorded as **measured**, never solved.
- **Conductance scalings are `params`** — chosen, not derived. The contract already fixed this half:
  `CourtemancheCellModel` is `type` plus an open `params` dict, deliberately open because "which
  conductances are worth varying is an experimental question", and the θ-spec points into it by
  `path` when SEP11 sweeps one.

So `targets` becomes **per-model partial** rather than a fixed pair. That is the cost, and it is
the right one: it keeps an AP bank and a Courtemanche bank able to state that they aimed at the
same conduction velocity, which is what makes the A/B between them interpretable at all.

**Research answered in CL-180 (2026-08-16), and it splits this step in two.** Source throughout is
**Wilhelms et al., *Front Physiol* 2012;3:487**, an independent reimplementation with a stated
protocol — *not* CRN 1998's own table, which research could not retrieve. That is arguably the
better fixture (matching someone else's implementation validates ours, and it gives a five-element
vector rather than one number) but it is a **different claim**, and the card must say which.

**CRN control, BCL 1 s** — Amplitude 110.11 mV · RMP −81.04 mV · APD50 165.16 ms ·
**APD90 294.83 ms** · **dV/dt max 186.58 V/s**. Control **clears `T = 192` with margin**, which is
what makes it authorable now.

**Full cAF remodelling does NOT clear `T`** — `I_to` −65 %, `I_CaL` −65 %, `I_Kur` −49 %,
`I_K1` +110 % (van Wagoner 1997 · Bosch 1999 · Dobrev 2001) gives **APD90 143.87 ms < 192**, so the
shortcut CL-176 removed would come straight back. The fix is **partial** remodelling along the same
published axis via a severity scalar `s ∈ [0,1]`, swept once until measured APD90 = 220 ms
(`s ≈ 0.4–0.6`). That sweep is a **measurement**, so it gets its own step.

**Three traps, all from CL-180, all capable of producing a wrong number that looks right:**

1. **CRN never reaches steady state.** APD90 falls to 83 % of first-beat over 16 min; APD50 falls
   42 % over 20 min at BCL 1 s. "CRN's APD90" is meaningless without a beat count — the same model
   legitimately reads 295 ms or ~245 ms depending when you look. **Pin BCL *and* number of beats in
   the card and the test** (Wilhelms paces 50 s at BCL 1 s). Most likely cause of a flaky failure.
2. **Do not also apply Wilhelms' 30 % intracellular-conductivity reduction.** Our card *solves* CV
   from diffusion, so applying both double-counts: the solve would simply raise `diffusion` to
   cancel it — a no-op with extra steps, leaving a `diffusion` that means nothing physical. The
   gap-junction effect belongs **inside** the CV target; say so on the card.
3. **`dr = 0.25 mm` is under-resolved for CRN, and the failure is silent.** CRN's upstroke is
   ≈ 0.59 ms ⇒ a wavefront ≈ 0.47 mm wide ⇒ **1.9 cells** at today's `dr`. Monodomain practice wants
   5–10. **A CV-solve will absorb the discretisation error into `diffusion` and hit the target
   anyway**, leaving a physical-looking number that is not — the same shape as S38b's measured
   "CV ran 8 % high at D ≈ 10". Cost is **f⁴**, not f²: refining by f costs f² nodes *and* f²
   timesteps, since holding CV forces `D ∝ f²` which tightens the stability bound equally.

   | `dr_mm` | grid | cells across the upstroke | mesh cost |
   |---|---|---|---|
   | 0.25 (today) | 160² | 1.9 | 1× |
   | 0.10 | 400² | 4.7 | **39×** |
   | 0.05 | 800² | 9.4 | 625× |

   With CRN's per-node cost over AP (21 state variables and gating exponentials vs 2) this is
   **roughly 400–1200× an AP simulation** at `dr = 0.1`. **Daniel's call, 2026-08-16: proceed
   anyway** — a desktop and an overnight run cover it, and a capability that is expensive to run
   beats not having it. SEP5 needs *one good* CRN bank for a comparison, not a corpus.
   **Note the useful coupling already in place:** a card records the `dr` it was solved at and
   `verify_targets_against_solve` re-solves on every load, so a card authored at `dr = 0.1` and
   loaded against `geometry.dr_mm: 0.25` **raises**. Under-resolution cannot happen silently through
   the card path; it is caught by machinery that already exists.

- **Change:** `CourtemancheCellModel` in `cell_models.py` — `ms_to_model_time` returns its argument
  unchanged (the seam `CellModelSpec` was cut for), `to_metadata`, and its own stability limit;
  a `calibrate_courtemanche` solving CV → `diffusion`/`dt`; the `parse_model_card` and
  `verify_targets_against_solve` branches; `_build_model_2d` dispatch in
  `backends/finitewave/backend.py` and removal of the AP-only refusal. **Ships the `courtemanche_control`
  card** — fully sourced from Wilhelms Table 1, clears `T` with margin, and gives the end-to-end
  path a real target. The AF-matched card waits for S18c.
  **Availability confirmed 2026-07-28** (see the decisions log): finitewave 0.9.3 ships
  `fw.Courtemanche` with a `fw.Courtemanche2D` back-compat alias, so the dispatch is the same idiom
  as today's `fw.AlievPanfilov2D()`. Conductances (`gna`, `gk1`, `gto`, `gkr`, `gks`, `gcal`, …) are
  plain instance attributes read at kernel-run time, so scalings are set by assignment — no patching
  of the model, and SEP11 can address them by `path`. **Confirmed on landing, with one exception:
  `gkur` is not among them** — see the note above the change list.
- **Verify:** a short Courtemanche sim produces a physiologically plausible AP upstroke + plateau —
  **assert upstroke velocity and APD90 fall in published human-atrial ranges, not just "it runs"**;
  the solved diffusion reaches the measured CV target within the same tolerance S38b accepted;
  an AP card still loads and verifies unchanged; a card naming an unregistered model is refused by
  name; `ms_to_model_time` is the identity for Courtemanche and is *asserted* to be, since a silent
  division by a time unit that does not exist is the failure this seam was cut to prevent.
- **No known-good prior fixture, and that is a consequence worth stating.** S17 was meant to leave a
  clean numerical baseline for exactly this step; absorbed into S38c, it did not. So Courtemanche is
  validated against **Wilhelms' published vector** rather than against a previous run of our own.
  That is weaker, and it is why the five values are asserted rather than eyeballed.
- **Depends on:** S18a.

### S41 — Process identifiers out of `src/`, `docs/` and `tests/` ✅ (2–4 h)
- **Daniel, 2026-08-25**, on finding a step id in a `cell_models.py` comment: comments may cite
  papers and physics but must not reference project phases, coordination-log entries, or
  implementation steps. Confirmed to cover **both `src/` and `docs/`**; `project/` keeps everything.
- **Why:** these packages are going to PyPI. `CL-176` means nothing to an external reader and never
  will, and once a phase is archived the number is noise internally too. A citation stays verifiable
  forever; a process id rots. It slots into the existing `project/` (internal) / `docs/` (external)
  split rather than inventing a new rule.
- **Scale, measured after the S18 commit:** **127 in `src/` (32 files) · 21 in `docs/` (4 files) ·
  79 in `tests/` (17 files) = ~227 sites.** `tests/` included on Daniel's call 2026-08-25: a test
  docstring's job is to say what the test proves and why it matters, and *"Both halves of CL-180's
  trap 3"* fails that for anyone who cannot open CL-180 — which by phase end includes us.
- **This is NOT a find-and-delete, and treating it as one will make the codebase worse.** A regex
  over 227 sites would leave 227 bare assertions — the scripted-bulk-edit trap CLAUDE.md names.
  Four distinct cases, and they need different handling:

  | case | example | treatment |
  |---|---|---|
  | **trailing citation, reasoning already present** | *"…was always the stencil's built-in 3.09 no matter what was requested (CL-172)."* | delete the tag; the sentence stands |
  | **tag stands IN PLACE of the reasoning** | *"off-grid T fails outright downstream (CL-112)"* | **write the reason**: the classifier halves the sequence six times, so a length off the 64-sample grid fails at the first ragged stage |
  | **tag used as a feature NAME** | *"as before SEP2"*, *"the probe sweep (SEP13)"* | replace with the English name — *"before controlled-position cropping"* — the sentence needs a subject, not a deletion |
  | **tag used as a version marker** | *"**2.0 since S38b**, down from 3.0"* | delete the marker; git records when. Keep the citation that justifies the value |

  The second case is the dangerous one: `_config.py`'s *"which is the root cause of CL-143"* becomes
  "the root cause of" nothing at all once stripped.
- **Own commit, not folded into S18.** Half-converting the nine files S18 touches while leaving the
  other eleven is worse than either end state, and this needs real review attention precisely
  because it *looks* mechanical and is not.
- **Also update `CLAUDE.md`** — done 2026-08-25; it previously documented the opposite convention.
- **Verify:** zero matches for `CL-[0-9]`, `SEP[0-9]`, `FB-[0-9]`, step ids and `design note D[0-9]`
  under `src/`, `docs/` and `tests/`; `project/` untouched; full gate green.
- **The verification that actually matters is not greppable:** *every site that lost a tag gained a
  reason.* One mechanical proxy is worth applying though — **the diff should be net-positive in
  lines.** Promoting reasoning makes comments longer; a net-negative diff is direct evidence the
  pass deleted rather than rewrote, and is grounds to reject it without reading further.
- **Sibling repos are unchecked** and almost certainly carry the same pattern, since it was authored
  consistently across the constellation. Out of scope here (one repo at a time); worth a backlog
  entry or a note to the project-lead.
- **Depends on:** S18b committed.

### S42 — Anti-alias the decimation, and the two-number upstroke protocol ✅ (3–5 h)

**Done 2026-08-25.** Both corrections landed. Full gate green (347 fast + 15 slow, up from 324 + 14).

**(a) The filter, and the measurement that justifies its shape.** `band_limit` low-passes the
capture at 0.8 × the output Nyquist (400 Hz), zero-phase, delegating to **egm-signal's `lowpass`**
rather than reimplementing — one `sosfiltfilt` across the constellation is one place for a phase bug
to live. `decimate` there could not be used for the reason the entry predicted of scipy's: it takes
an integer factor. Order 8 rather than egm-signal's default 4, measured rather than assumed:
Butterworth is maximally flat, so a higher order steepens the skirt *and* flattens the passband —
order 8 both rejects better (0.000036 % left above Nyquist against 0.0021 %) and keeps more passband
(98.55 % against 98.20 %).

**The committed spectrum, on a fibrotic 16 mm patch, as a percentage of trace power:**

| band | AP capture | AP output | CRN capture | CRN output, unfiltered | CRN output, filtered |
|---|---|---|---|---|---|
| 0–100 Hz | 98.031 | 97.960 | 65.985 | 65.690 | 66.752 |
| 100–250 Hz | 1.968 | 2.038 | 30.314 | 30.819 | 30.904 |
| 250–400 Hz | 0.0016 | 0.0020 | 2.500 | 2.764 | 2.314 |
| 400–500 Hz | 0.0000 | 0.0000 | 0.460 | **0.692** | **0.031** |
| above 500 Hz | 0.0000 | — | **0.741** | — | — |

The Courtemanche 400–500 Hz output band reads **0.692 % unfiltered against 0.460 % in the capture
itself** — the excess is the folded energy arriving, which is the aliasing made visible. Filtered it
reads 0.031 %. Aliev-Panfilov is unchanged to three decimals in every band, so the old docstring's
claim was true *for the model it was written about*, and this is entirely a consequence of the ionic
one landing.

**Zero-phase, verified at the level that matters.** The regression test asserts the detected
activation index on the **finished output trace**, because that is where `activation_position` is
taken. The control beside it applies the same Butterworth forwards only and shows a **2-sample
shift** at the output rate — 0.010 in realized position, on every trace, with nothing in the output
that looks wrong. Without that control the zero-phase assertion would only be saying the filter is
gentle.

**The fixture this broke, and why the repair is the interesting part.** `ambiguous_complex` split
the three detection curves with a **one-sample biphasic spike** — a delta, flat past Nyquist. The
filter removes exactly the content that made it the steepest feature, two curves then agreed, and
three tests failed. The obvious repair does not work either: rectified-derivative peaks at the
steepest *carrier crossing* and Teager-Kaiser at the *envelope* maximum, so for a single burst they
sit a quarter carrier period apart — at most one sample at 1 kHz, and it rounds to zero for a
quarter of all capture lengths (measured: 150 failures over 601 lengths). The fixture now gives
**each curve its own feature**, chosen by the property that curve measures. Verified over every
capture length from 400 to 1200 in both the paths it is used in: **zero collapses, minimum index gap
60 samples.** The old fixture was asserting on content no bank could contain.

**(b) The protocol, pinned — and the outlier turns out not to be an implementation difference.**
The capture threshold is now **measured** by bisection (`measure_capture_threshold`), not guessed:
**10.906 mV/ms** at the 2 ms duration, stable to 0.3 % between a rested cell and one 49 beats into
the train and independent of the conditioning amplitude. The stimulus is 2× that.

| property | Wilhelms | at 20 mV/ms (before) | at 2× threshold (now) |
|---|---|---|---|
| Amplitude | 110.11 mV | −3.5 % | **−1.8 %** |
| RMP | −81.04 mV | +0.5 % | +0.6 % |
| APD50 | 165.16 ms | +2.7 % | **+0.9 %** |
| APD90 | 294.83 ms | −0.8 % | −1.3 % |
| dV/dt max | 186.58 V/s | +14.2 % | +15.3 % |

Pinning the stated protocol **halved three of the four small residuals**. It did not fix the fifth,
and the reason is now measured rather than speculated: the upstroke happens *while the stimulus is
still on* — the maximum falls at 1.86 ms of a 2 ms stimulus — so the measured value contains the
stimulus. Raising the stimulus 1.82 mV/ms raised the measured maximum 2.13 V/s. **Wilhelms states
"twice threshold" but not the duration, and threshold scales with duration**, so the ambiguity is
irreducible from their side:

| duration | threshold | 2× | dV/dt max | vs Wilhelms |
|---|---|---|---|---|
| 1 ms | 21.343 | 42.687 | 200.46 V/s | +7.4 % |
| **2 ms** | **10.906** | **21.812** | **215.16 V/s** | **+15.3 %** |
| 5 ms | 4.535 | 9.070 | 195.39 V/s | +4.7 % |
| 10 ms | 2.416 | 4.832 | 179.57 V/s | −3.8 % |

2 ms is kept because it is the conventional single-cell duration and because the textbook 2 nA
stimulus sits at 1.83× our measured threshold, which cross-validates the threshold — **not** because
it matches; it is the worst of the four. Subtracting the stimulus's own contribution gives 193.3 at
2 ms and 186.3 at 5 ms against Wilhelms' 186.58, which points at protocol rather than membrane; that
is recorded as an observation, not asserted on.

**Tolerances, built from measured variation rather than from the miss.** Three things can
legitimately move a number now that the amplitude is pinned, and all three were measured: read-point
in the train (±10 beats moves APD50 0.9 %, everything else ≤0.25 %), timestep (halving it moves
0.2–1.0 %), and the unstated stimulus duration (4.5 % on amplitude, 19 % on dV/dt max). Summing per
property: **amplitude 10 → 7 %, APD50 15 → 6 %, APD90 10 → 5 %, RMP 3 % unchanged, dV/dt max
20 → 18 %.** Four tighten substantially; dV/dt max barely moves, and that is the honest answer — a
tolerance under 16 % would be asserting that Wilhelms used our duration.

**Two upstrokes on the card, labelled, and they differ by 39 %.** `measured.upstroke_v_s` is the
**propagated** figure — 131.1 V/s at the patch centre, 20 mm from the stimulus — and the validation
block carries the **stimulated** single-cell 215.2. An isolated cell puts all its sodium current
into its own membrane; a cell in tissue spends much of it charging the cells ahead. The propagated
one is the card's physical claim, because no electrode in a real bank sits on a stimulus site.

**Scheduling consequence, now in `docs/usage.md`:** this is a regeneration event. Banks either side
of it are not comparable, and which side a bank is on is readable from
`run_metadata.antialias_cutoff_hz`, absent on every bank written before.

**Not folded in, as the entry required:** matching IAFDB's acquisition band.
`run.antialias_cutoff_fraction` is the knob that step would turn.


Two corrections to the **measurement and signal path**, grouped because both change numbers we
already record and both are protocol errors rather than defects.

**(a) The capture→output decimation has no anti-alias filter.** `downsample` takes an integer-ratio
**stride** — bare decimation. Its own docstring predicted this coming due: *"acceptable for Phase 1
because the AP membrane potential has no spectral energy above a few hundred Hz … for a faster
wavefront, swap for a polyphase filter."* The ionic model landed, so the condition the docstring
named has been met.

**Do it rather than measure first**, on three arguments, the middle one being the real one:

1. **Aliasing is irreversible and undetectable after the fact.** Unlike every other artifact this
   phase has chased, you cannot look at an output trace and tell whether it is aliased — so
   "measure and decide later" leaves every bank already generated unauditable.
2. **Band-limiting before sampling is physically correct, not a distortion.** Real electrophysiology
   front-ends apply analog anti-alias filtering ahead of the ADC, and the IAFDB records came through
   one. **Unfiltered synthetic traces therefore carry spectral content no real recording can
   contain** — which is a corpus difference, i.e. exactly the shortcut-feature class this project
   keeps finding, except this time we would be manufacturing it knowingly.
3. It is cheap — but **not the one-call swap it was described as**, see below.

**Measured 2026-08-25, and it changes the implementation: the stride path never runs.**
`_pick_capture_step` sets `achieved_capture_fs_hz = 1000 / (step · dt_ms)` with `step` an integer,
so the achieved rate is a *consequence* of the integration step rather than a chosen multiple of the
output rate. At both shipped cards the ratio is **not** an integer —

| card | `dt` (ms) | step | achieved capture | ratio |
|---|---|---|---|---|
| `af_remodelled_220ms` | 0.010261 | 24 | 4060.9 Hz | **4.0609** |
| `courtemanche_control` | 0.020000 | 12 | 4166.7 Hz | **4.1667** |

— so `downsample` has been taking its **linear-interpolation** branch all along, not the stride
branch its docstring justifies. Two consequences:

- **`scipy.signal.decimate` cannot be dropped in**: it takes an integer decimation factor.
- **Prefer band-limit-then-resample**: apply a zero-phase low-pass to the capture, then leave the
  existing rate conversion alone. It works at any ratio, keeps filtering and resampling as separate
  inspectable concerns, and is far easier to defend in a methods section than a polyphase resample
  at a rational approximation of the rate. (`resample_poly` with a `Fraction.limit_denominator`
  ratio is the alternative; it folds a small rate error into the fix and is harder to state.)
- **The filter MUST be zero-phase** — `sosfiltfilt`, not `sosfilt`. A causal filter has group delay,
  which shifts the **detected activation index**, which shifts `activation_position` — a stored
  column, an asserted value, and the axis the whole controlled-position crop is built on. This is
  the single most likely way to get this change subtly wrong.
- Linear interpolation *is* a crude low-pass, which is presumably why this went unnoticed; it is
  simply a bad one, with poor rejection above Nyquist and a signal-dependent response.

- **This is a regeneration event, and that is the scheduling constraint.** Every bank to date was
  produced by striding; adding the filter changes **every trace**, so banks either side stop being
  comparable. **It must land before study banks are generated, not after.** Whether existing banks
  are regenerated or simply retired is Daniel's call and should be stated in `docs/`.
- **Model cards are unaffected** — calibration measures conduction velocity and duration from the
  membrane potential via trackers, never from the electrogram. No card needs re-solving.
- **Still measure the spectrum either side of the decimation** — to *document* the choice, not to
  make it. Commit the measurement.
- **Deliberately NOT folded in:** matching IAFDB's acquisition band. It is a larger realism
  question, it is not required for correctness here, and bundling it would make this change
  impossible to attribute.

**(b) `dV/dt max` is measured under an unstated stimulus, and the reference protocol is known.**
Wilhelms §2 states it: *twice the threshold amplitude* for single cell, *20 % above threshold* for
tissue. Ours used an unstated amplitude and read 213.03 V/s against their 186.58 — a **protocol
mismatch, not an implementation error**. Measured across amplitudes we get 165 / 195 / 218 / 227 V/s
at 12 / 15 / 21 / 30 mV/ms, so a threshold near 6–7 mV/ms puts 2× squarely on their figure.

- Determine the threshold, stimulate at 2×, re-measure, and **tighten the tolerance from 20 %** —
  which only becomes defensible once the protocol is pinned.
- **Record two numbers, labelled.** The **stimulated single-cell** value validates against Wilhelms.
  The **propagated** upstroke — measured a distance from the stimulus — is the card's physical
  claim, because no electrode in a real bank sits on the stimulus site. They answer different
  questions and conflating them is what produced the outlier.
- **Why this is not cosmetic:** with conduction velocity and duration matched across both cards,
  upstroke morphology is the only variable left between them, and electrogram amplitude scales with
  it. This is the number the model comparison rests on.

- **Verify:** a signal with energy above the output Nyquist comes back **attenuated, not folded** —
  and note the vacuous version of that test is feeding it a signal that was already band-limited,
  which passes whatever the code does; the committed spectrum either side of the decimation; a
  regenerated bank differs from a pre-filter one **in the high band specifically**, which is what
  proves the filter is live rather than merely present; `dV/dt max` at 2× threshold lands inside the
  tightened tolerance; both upstroke numbers recorded and labelled distinctly.
- **Depends on:** S41.

### S18c — Mesh convergence, the cAF severity sweep, and the matched card (SEP5) ✅ (4–7 h)

**Done 2026-08-26.** Both sweeps, both cards, and
`investigations/courtemanche_calibration.md`. Full gate green (353 fast + 18 slow, up from 347 + 15).

**The rig, and it is the reason this step was affordable.** A **1-D cable** — 20 mm, three mesh rows
so exactly one is interior — validated against the 40 mm patch *before* being trusted: propagated
upstroke agrees to **0.004 %**, first-beat APD90 to **0.09 %**, because a plane wave in a sheet has
no transverse gradient. Conduction velocity differs by 1.3 % (the fit windows sit at different
distances from the stimulus), so **CV is measured on the patch and the other two on the cable** in
both cards. The seven-pitch sweep took **84 seconds**; on the patch it would have been hours.

**One confound found and removed, and it nearly went the other way.** `strip_thickness` counts
*cells*, so refining the mesh was shrinking the physical stimulus — 0.75 mm at `dr = 0.25` down to
0.225 mm at 0.075. It bites hardest at **high** diffusion (a larger `D` drains the stimulated region
into its neighbours faster), and at `dr = 0.075`, `D = 0.616` the wave stopped launching and the
sweep died with *"0 of 106 nodes activated"*. Loud, and lucky: a *weak* launch would have returned
numbers and blamed the mesh for the stimulus. Stimulus now pinned at 0.75 mm of tissue; whole sweep
re-run.

**(a) The convergence curve** — cable, `D` fixed at the operating point, exponent fitted across the
same three diffusions the patch used:

| `dr` (mm) | cells | CV at fixed `D` | propagated dV/dt max | `CV ∝ Dⁿ` |
|---|---|---|---|---|
| 0.250 | 80 | 81.854 | 131.139 | 0.5835 |
| 0.200 | 100 | 83.074 | 124.301 | 0.5611 |
| 0.150 | 133 | 84.616 | 120.245 | 0.5430 |
| 0.125 | 160 | 85.758 | 119.699 | 0.5351 |
| **0.100** | **200** | **86.680** | **119.230** | **0.5278** |
| 0.075 | 267 | 87.412 | 118.878 | 0.5210 |
| 0.050 | 400 | 87.935 | 118.672 | 0.5126 |

At `dr = 0.25` the cable reproduces the patch's exponents to four figures — an independent check
that the rig measures the same thing.

**Expectation 1 held. Expectation 2 did not, and the direction matters.** The exponent falls
monotonically toward 0.500, near-exactly first-order in the pitch (`n − 0.5 ≈ 0.28·dr`). The
propagated upstroke **falls** 131.1 → 118.7 rather than rising. The mechanism is the opposite of the
one assumed: a coarse mesh does not smear the upstroke, it *compresses the wavefront into fewer
cells*, so each node's transition is steeper in time and refining relaxes it. **The 131.1 V/s S42
committed to the control card was a mesh artifact, 10.5 % high in the one observable an ionic model
was added to get right.** It also *widens* the propagated-versus-stimulated gap rather than
narrowing it, so S42's reading of that gap is strengthened.

**The reproducibility demonstration:** at a **fixed** `D = 0.299142`, with no physics changed, the
same tissue conducts at **81.854 cm/s on a 0.25 mm mesh and 87.935 on a 0.05 mm one — 7.4 % apart.**
A slow test asserts that gap stays *large*, so a bug making the mesh inoperative fails rather than
passes.

**Pitch chosen: `dr = 0.10`**, on a stated criterion — *the coarsest pitch at which the propagated
upstroke is within 0.5 % of the finest measured* (119.230 against 118.672, 0.47 %). **The exponent
has NOT plateaued** — 0.528 here, still 0.513 at 0.05 — so the solved diffusion remains partly a
mesh-compensation quantity. What refining bought is that the compensation is small: measured CV
overshoots its 80 cm/s target by **+1.56 %** against **+3.6 %** at 0.25. Refining further costs
`1/dr³`.

**(b) The severity sweep** — three currents, bisected on the cable to a propagated tissue APD90 of
220 ms: **`s = 0.56`, measured 219.49 ms**, scalings `g_to ×0.636 · g_CaL ×0.636 · g_K1 ×1.616`.
Two properties recorded: **APD90 is diffusion-independent** (219.00 / 218.98 / 218.97 ms at
`D` = 0.9× / 1.0× / 1.1×, so the sweep did not need to know the diffusion in advance), and **CV
moves 2.2 % and the upstroke 1.1 % across the whole `s` range** — the remodelling is close to CV-
and upstroke-neutral, so the severity choice does not confound the comparison.

**The measurand differs from the entry's wording, deliberately.** The sweep is on the **tissue first
beat**, not the single cell. `af_remodelled_220ms`'s own 220 ms is a tissue, first-beat, centre-node
APD90 — matching two cards requires matching the *measurand*, not merely the number, and sweeping
the single cell to 220 would have given a tissue APD near 244. The cable makes the correct measurand
free, so there was no trade.

**The expectation on `s`, checked on the axis it was stated for.** CL-180's 0.4–0.6 is a
**single-cell paced** interpolation. On that axis our three-current set reads 290.98 ms at `s = 0`
and 185.70 at `s = 0.56`, reaching 220 at **`s ≈ 0.38`** — below 0.49, as predicted. The shipped
0.56 is larger only because the tissue axis starts at 321.8 rather than 291.0. Reading the two as
one axis is the mistake to avoid.

**(c) `af_remodelled_crn_220ms` ships**, declaring the omission in the terms the entry required —
*three of the four cAF conductance changes collected in Wilhelms 2012; `I_Kur` (−49 %) omitted
because the solver inlines its voltage-dependent conductance*. Never "the published cAF
parameterisation". A limitations-register entry is in the investigation doc. **No compensation
through `gkr`/`gks`**, and a test asserts their *absence* from the card's params — the honest hole,
mechanically enforced.

**One piece of machinery this needed:** a **reference registry**. The velocity anchor is a property
of the conductance set *and* the mesh — the AF scalings move CV at fixed diffusion by 2.3 % — so
`calibrate_courtemanche` now looks up `(params, dr)` and refuses a combination it has no measurement
for, rather than reaching for the nearest one.

**Open, and outside this step:** `af_remodelled_220ms` is still authored at `dr = 0.25`. A
model-versus-model comparison needs everything but the membrane matched, and the mesh is not
currently matched. Flagged rather than decided.

**Follow-up, 2026-08-26 — three fixes, fast gate only** (parameter-passing refactor, an error
message, a config example and docs; nothing that can move a number, per the cadence table).

1. **`_build_tissue_2d` took a geometry plus a shape override that contradicted it** — a signature
   callable two ways meaning one of them. It now takes `shape` and `fiber_angle_rad`, which are the
   only two things it read. Both a patch and the cable still route through it, which is what keeps
   one fibre-convention implementation.
2. **The missing conduction-velocity anchor warns and proceeds** on the nearest registered anchor
   instead of raising. Blocking made `backend.model` unreadable without `geometry.dr_mm`; cross-
   section rules get their own tool (FB-37) rather than accumulating in whichever loader notices
   first. Nearest is two-tier — matching conductances first, then closest pitch — and the warning
   names the anchor, both pitches, and the direction of the resulting error (finer than the anchor
   ⇒ realised CV lands **above** target). **The substitution is written into the bank**:
   `model_anchor_name` and `model_anchor_dr_mm` always, plus `model_anchor_substituted`,
   `model_anchor_run_dr_mm` and `model_anchor_conductances_differ` when it happened. Verified end to
   end into a bank's `params`. A hard error survives only for the genuine impossibility — an empty
   registry, where there is nothing to substitute.

   **One thing the downgrade needed that the brief did not name.** With the anchor check downgraded
   the load still failed, on the *drift* check: `dt` is derived from the mesh through the CFL bound,
   so at another pitch a card's step and a fresh solve's step are **supposed** to differ. That is a
   pitch mismatch reported as calibration drift. `dt_model_units` is now excluded from the drift
   comparison when the anchor was substituted; `diffusion` — the field the check exists to protect,
   and the one the anchor actually sets — is still compared. The card's own step stays *conservative*
   at a coarser pitch rather than unstable, because the bound loosens as the mesh coarsens.
3. **`examples/synthegm_courtemanche.yaml` ships**, identical to the baseline in every block but the
   membrane, with `dr_mm: 0.1` and `run.dr_model_units: 0.1`. A fast test asserts the example loads
   with **no substitution warning**, which is what pins the example's pitch to the card's anchor.
   `usage.md` states both halves of the coupling — mechanical (the anchor was measured at that
   pitch, and a solve elsewhere absorbs the mismatch into `diffusion` and hits its target anyway)
   and physical (at 0.25 the upstroke spans 1.9 cells against ~5 and reads 10.5 % high). The second
   is why the first matters: a borrowed anchor makes the run *succeed*, at a velocity that matches
   its target, with an upstroke that is 10 % wrong.

   **The 39x cost figure in the brief is wrong and the docs say 14.5x.** 39x assumes `dt ∝ dr²`
   throughout. It does not: at `dr = 0.25` the step was set by the sodium current's 0.02 ms ceiling,
   not by the diffusion bound's 0.0597, so refining pays the quadratic penalty over only part of the
   range. Measured from the two solves: 6.25x the nodes, a 2.33x smaller step, **≈14.5x** the work.

- **Change:** the two **measurements** CL-180 asks for, then the card they produce.
  **(a) Mesh convergence** — sweep `dr` on a **1D strip, not the 40 mm patch**; CV and dV/dt max
  versus `dr`, continued until both plateau. Pick the coarsest `dr` on the plateau and record the
  curve. A strip is cheap and answers the same question, so the 39–625× patch cost is paid once for
  the shipped card rather than once per candidate `dr`.
  **(b) The cAF severity sweep** — single cell, no mesh at all, so it is nearly free: sweep
  `s ∈ [0,1]` over **three** conductances until measured APD90 = 220 ms —

  `g_to ×(1−0.65s)` · `g_CaL ×(1−0.65s)` · `g_K1 ×(1+1.10s)`

  **`I_Kur` is the fourth published change and it is NOT in that list — it cannot be.** Its
  conductance is voltage-dependent and inlined into the solver's kernel, so no parameter for it
  exists anywhere in the object graph. Expect `s` **below** CL-180's interpolated 0.4–0.6: `I_Kur`
  is repolarising, so its −49 % *prolongs* APD and opposes the shortening, and omitting it should
  shorten more per unit `s`. **Treat that as expected, not guaranteed** — `I_Kur` block is
  non-monotonic in APD, because prolonging the plateau recruits more `I_Kr`/`I_Ks` (CL-182). The
  sweep is empirical and settles it.
  **(c)** Author `af_remodelled_crn_220ms` — named to mirror `af_remodelled_220ms` so the shared
  target is visible in the filename — recording `s`, the resulting scalings, the pinned BCL and beat
  count, and the `dr` it was solved at. Write `investigations/courtemanche_calibration.md` beside
  `ap_model_calibration.md` (research left that file to us deliberately).
- **Why partial remodelling is defensible — RETRACTED AND REPLACED (CL-182).** The original argument
  here was that partial remodelling stays *on the published parameter axis*, moving less far along
  van Wagoner / Bosch / Dobrev rather than in a new direction. **Research withdrew that on our
  challenge: dropping a current is a different direction, not a shorter distance**, and the same
  authors' 1999 AF-remodelling paper also identifies `I_Kur`, so there is no published three-current
  alternative to fall back on. Do not ship the old wording.

  **The replacement is stronger, and it is about what the traces actually contain.** Ask what the
  card has to be true *for*. The claim is not that we reproduce cAF myocyte electrophysiology; it is
  that we generate bipolar electrograms with realistic **activation morphology** at a matched
  **conduction velocity**, with an APD90 that keeps repolarisation **outside the stored window**.
  Against those three observables:

  - the **upstroke** — the entire scientific content of the model comparison — is `I_Na`-dominated,
    and `gna` is settable. **`I_Kur` contributes nothing to `dV/dt` max.**
  - `I_Kur` shapes the **plateau and APD90**, and we have deliberately engineered APD so that
    repolarisation lies *outside* the cropped trace. **The omission is confined to the one part of
    the action potential our data does not contain.**
  - APD is a **measured, targeted** observable, so the sweep absorbs the omission: `s` is tuned
    until APD90 = 220 ms either way. The net observable matches; only the ionic decomposition
    behind it differs.

  It also preserves the architecture: **APD stays measured, never solved** — the sweep is a one-time
  authoring step, exactly as the analytic solve is run once for Aliev–Panfilov.

- **Declare the omission; do not disguise it.** The card must **not** claim "the published cAF
  parameterisation". Wording along the lines of: *partial AF-remodelling calibrated to a measured
  APD90 of 220 ms, applying three of the four cAF conductance changes collected in Wilhelms 2012;
  `I_Kur` (−49 %) is omitted because the solver inlines its voltage-dependent conductance into the
  kernel.* Then a **limitations-register entry**, given the same treatment as the seven Sánchez
  deviations. A declared, mechanically-forced omission in a feature the traces exclude is a
  defensible limitation; a three-of-four sweep reported as four would not be.
- **Do NOT compensate through `gkr`/`gks`.** They are settable, so `I_Kur`'s net APD effect could be
  absorbed into another repolarising current. Research explicitly recommends against it and the
  reasoning is right: it buys nothing observable, since the `s` sweep already lands APD on target,
  and it converts a **forced, declarable** deviation into a **chosen, fabricated** one — strictly
  harder to defend. Prefer the honest hole.
- **The modality tension is real and must be stated, not smoothed.** In-vivo MAP gives 219–245 ms
  for AF patients (Franz, CL-176) while isolated cAF myocytes give 95–144 ms — a ~2× disagreement.
  They are different measurements at different remodelling stages. **We simulate in-vivo tissue
  during a mapping procedure, so the MAP line is the matched modality**, which is why the AP card is
  at 220 ms and why CRN follows it. Christ 2008's 287 ± 16 ms puts 220 inside the published cAF
  range regardless.
- **Two convergence checks, not one — and they move in opposite directions.** The mesh sweep has a
  stated expectation on each, which is what makes it a test rather than a plot:
  1. **The CV exponent should fall toward 0.500.** Measured 0.625 then 0.542 at `dr = 0.25`.
  2. **The propagated upstroke should RISE.** S42 measured 131.1 V/s propagated against ~187
     stimulated. Some of that gap is real — axial current from neighbours is gentler than a direct
     stimulus — but an under-resolved mesh smears precisely this quantity, so refining should
     recover part of it. Two quantities converging from opposite directions is far stronger evidence
     than either alone.
- **The reproducibility test that makes the problem concrete (CL-182):** at the shipped operating
  point, conduction velocity is **mesh-dependent** — the solve has absorbed discretisation error into
  `diffusion`, so **changing `dr` or patch size will move CV at fixed `D`.** Run one `D` at two `dr`
  values and watch it move. That is a direct hazard to a methods section and it is cheap to
  demonstrate.
- **Verify:** measured APD90 within tolerance of 220 ms at the pinned protocol; measured CV matched
  to the AP card's, so **upstroke morphology is the only free variable left** — that is the actual
  scientific payoff of the comparison, since AP's upstroke is smooth and broad while CRN's is far
  steeper and electrogram amplitude scales with `dV/dt`; the convergence **curve** is committed, not
  just its conclusion; the `I_Kur` omission appears in the card text and the limitations register,
  and the sweep reports three currents rather than four.
- **If the sweep cannot reach 220 ms** while staying monotone and stable, CL-180 offers control CRN
  (294.83 ms) as the fallback. **Take that only after escalating**, because it silently breaks
  CL-180's own Q4 answer: two cards at different APDs describe different atria, and SEP5 stops being
  model-vs-model. Prefer reopening the shared *target* over reopening the *matching*. **Do not
  reopen `T`** — it is fixed by the classifier's multiple-of-64 constraint and by IAFDB extraction.
- **Depends on:** S18b.

### S19 — Verify anisotropy on the ionic model (SEP5) ✅ (1–3 h)

**Rescoped 2026-08-25 — the code half is absorbed and writing it now would be a regression.** The
entry called for a `_configure_anisotropy_2d_courtemanche` sibling and a bypass of the time-unit
translation. Both arrived by other routes:

- **Anisotropy is already model-agnostic.** S38a moved `D_al`/`D_ac` onto the **stencil**, which is a
  separate object from the cell model, so the helper writes a pure tensor *shape* that any model
  inherits. There is one call site and no sibling. **Do not write one** — two code paths for one
  concept is exactly what the stencil fix removed, and the second path is where they drift.
- **The time base is handled.** `ms_to_model_time` returns its argument unchanged for Courtemanche,
  which is the seam the fifth spec was cut for. Nothing to bypass.
- The note about differing diffusion defaults is moot: both models now take diffusion from a solved
  card, not from a package default.

**What is left is verification, and it is the part that matters.** `measure_anisotropy_ratio` is
**the last measurement function still typed to `AlievPanfilovCellModel`** — `measure`,
`measure_conduction_velocity` and the internal helpers all took `CellModelSpec` during S18b; only
this one was missed. So the realized anisotropy of the ionic model has never been measured, and **as
typed it cannot be.**

That matters more here than it would anywhere else in this codebase. **`anisotropy_ratio` is the knob
that silently did nothing for the entire life of the project**, because the code plainly appeared to
set it and nobody measured the result. "Model-agnostic by construction" is the same species of claim
as "the helper sets D_al" — true about the code, and previously insufficient.

- **Change:** widen `measure_anisotropy_ratio` from `AlievPanfilovCellModel` to `CellModelSpec`;
  measure the realized along/across ratio for Courtemanche and assert it against the requested one.
- **Verify:** realized ratio matches `geometry.anisotropy_ratio` within tolerance **for the ionic
  model**, measured from the activation map rather than inferred from the stencil values — reading
  back the numbers we wrote would restate the assignment, not test it. Aliev–Panfilov's existing
  measurement stays green and unchanged.
- **Cost, and it is a slow test:** two solver runs at `dr = 0.1 mm` (along and across), which is the
  fine-mesh regime. Mark it slow, and consider a smaller patch — the ratio is a property of the
  tensor and the mesh pitch, not of the patch extent, so it does not need 40 mm to be measurable.
- **Depends on:** S18c.

### S20 — Courtemanche runtime characterization + docs (SEP5) ☐ (3–5 h)

**Rescoped 2026-08-25. Measuring what this entry originally asked for would produce a number that
describes a configuration nobody can use.** It says "at the v1 geometry (40 mm, dr 0.25 mm →
160×160)" — but S18c moved Courtemanche's calibration mesh to **0.1 mm**, and its cards refuse to
solve honestly at 0.25. A like-for-like 0.25-vs-0.25 comparison would understate the real cost by
roughly the 39× mesh factor and would be quietly meaningless.

**The comparison that matters is each model at the pitch it is actually run at:**
Aliev–Panfilov at `dr = 0.25` (160×160) against Courtemanche at `dr = 0.1` (400×400). That is not
an apples-to-apples measure of the membrane cost, and it should not pretend to be — it is the
*operational* cost, which is the question being asked. **Report both decompositions**: the mesh
factor and the per-node membrane factor separately, so a later reader can tell how much of the total
is the ionic model and how much is the resolution its upstroke demands.

**This is the number the project has been estimating and should stop estimating.** It has been
recorded as an explicit estimate twice — the "roughly 400–1200× an AP simulation" figure carries an
UNVERIFIED marker in the Courtemanche investigation, because the per-node factor was never measured
and AP wall-clock was never taken. **Three open decisions consume it:** the simulator-backend
re-evaluation, the AP/Courtemanche tiering question (Aliev–Panfilov for routine generation,
Courtemanche only for studies that need it), and the §8 data plan. It is worth more now than when
the entry was written.

- **Change:** measure wall-clock per simulation for both models at their operating pitches; write
  the timing table into `docs/simulation_theory.md`. Project it: what a 100-sim and a 1000-sim CRN
  bank cost in hours, on the desktop as well as the laptop, since that is the machine an overnight
  run would use.
- **Already done, do not redo:** the example config shipped with the S18c follow-up
  (`examples/synthegm_courtemanche.yaml`). The time-base explanation and the reason only conduction
  velocity inverts are in `simulation_theory.md`'s calibration section. The ionic formulation and
  the conductance scalings are covered in the platform investigation. What is genuinely missing is
  the **timing**, and a short pointer from the theory doc to that investigation.
- **Verify:** the timing table is present and states the machine, the pitch and the patch size for
  every row — a wall-clock number without its hardware is not a measurement; the projected
  bank-generation figures are derived from measured per-sim time rather than assumed; the estimate
  in the investigation is **replaced** by the measured value and its UNVERIFIED marker removed.
- **Raise to the project-lead if the projected time is impractical** — that is a §8 data-plan input
  and it also feeds the tiering decision, which is not this repo's to make.
- **Depends on:** S19.

### S21 — Resource / CPU cap (B12) ☐ (2–4 h)
- **Change:** a `resources:` config block capping worker / BLAS thread counts so a long generation
  run doesn't redline the laptop. Pairs with S20 — Courtemanche runs are exactly when this bites.
- **Verify:** thread caps observably applied; a capped run completes with the same output as an
  uncapped one.
- **Depends on:** S10.

> **SEP11 is one issue, not two** (project-lead, 2026-07-28). It is the θ-sweep **harness** + the
> `generation_params` writer + a **pluggable sampler** — θ-spec-agnostic, depending only on SEP12.
> The same capability is *run twice* in the post-code §8 data stage: an OAT config produces the screening
> banks, egm-studio's **STU7** screening then sets the θ-spec membership, and a calibrated config
> drives the design sweep for STU4. That ordering is **§8.2 run-ordering — data flow, not a code
> dependency**; nothing here waits on STU7 to compile. The promotion test is "distinct code
> capability?", and OAT vs LHS/grid is a swapped sampler over a shared harness. Steps are ordered so
> **the harness + OAT sampler land first and unblock STU7**, with the LHS/grid sampler after.
>
> **Escape hatch:** if the design sweep turns out to need substantial harness code beyond a sampler
> swap — the GP-emulator design-bank layout, per-cell N-tagging, real LHS infrastructure — that is
> the signal to flag it and split into SEP11a/SEP11b. Default stays single.

### S22 — `TunedParam` path resolution (SEP11) ☐ (2–4 h)
- **Change:** a resolver that reads and writes a config value by its θ-spec `path`
  (`substrate.density`, `cell_model.courtemanche.params.gcal`, `backend.diffusion`, `mixer.snr_db`)
  against the spec objects. This is what makes θ membership a per-sweep choice with zero contract
  churn — and what keeps the harness θ-spec-agnostic.
- **Verify:** get/set round-trip for every 1.5-reachable path; a clear error on an unknown path.
- **Depends on:** S10 (+ S18 for the cell-model paths).

### S23 — Sweep harness + pluggable sampler + OAT sampler + `sweep:` config (SEP11) ☐ (4–6 h)
- **Change:** the harness — a `Sampler` Protocol producing a design matrix over the knob list, and
  `generate_dataset` driven from that matrix instead of today's independent per-axis sampling
  (density uniform, edge uniform, height uniform). Ship `OATSampler` (per knob, a few levels with
  everything else at nominal) as the first concrete, **and the `sweep:` config block that drives
  them** — a harness reachable only from Python is a harness nobody can run a bank from. The
  no-sweep path must still reproduce today's sampling exactly.
- **Verify:** an OAT design varies exactly one knob per cell with the rest pinned at `nominal`; the
  no-sweep path is bit-identical to pre-SEP11 output at the same master seed; config tests cover the
  block + an unknown-sampler error.
- **Depends on:** S22.

### S24 — `generation_params` writer + OAT example + docs — **unblocks STU7** (SEP11) ☐ (3–5 h)
- **Change:** populate the bank-root `{regime, knobs:[TunedParam]}` stubbed at S5 — regime from
  the fixed type discriminators, knobs from the sweep definition (bounds, transform, role, nominal).
  The OAT banks need this too: STU7 has to know which knob moved in which cell. Ships with the OAT
  example config the §8.2 screening banks will be generated from in the post-code data stage — so
  getting the example right matters more than usual — plus a `docs/usage.md` sweep section and the θ / regime framing in
  `docs/simulation_theory.md`.
- **Verify:** a swept bank's θ-spec validates; each knob's per-sim values recovered via its `path`
  match the design matrix; the OAT example runs end-to-end and produces a bank whose θ-spec +
  per-sim values are readable by an egm-studio consumer. **Code hand-off to STU7 here** — the
  capability, not a bank; the screening banks are generated later, in the data stage.
- **Depends on:** S23.

### S25 — LHS + grid samplers for the design sweep (SEP11) ☐ (3–5 h)
- **Change:** `LatinHypercubeSampler` + `GridSampler` concretes behind the same Protocol, M cells × N
  traces per cell, for the calibrated design run that feeds STU4. Membership of the θ-spec comes from
  STU7's screening result, so this is a *config* input, not new harness code — **if it turns out
  otherwise, invoke the escape hatch above rather than growing this step.**
- **Verify:** a small LHS sweep produces the requested cell count with each knob's marginal covering
  its bounds; an LHS example config runs; pr_checklist passes.
- **Depends on:** S24.

### S26 — `PostProcessStage` protocol + `BandpassStage` (SEP1) ☐ (2–3 h)
- **Change:** new `postprocess/` subpackage — a small stage Protocol + `BandpassStage` wrapping
  egm-signal's `bandpass`. Pure functions on trace arrays; no bank awareness.
- **Verify:** unit tests showing the stage reproduces the mixer's current `bandpass_clean` output
  bit-for-bit on the same input.
- **Depends on:** S10.

### S27 — Wire the stage into the producer + double-filter guard (SEP1) ☐ (2–3 h)
- **Change:** `postprocess:` config block applied after the dataset run; **guard** — enabling a
  band-pass post-process stage while `mix.bandpass_clean` is true raises a `ConfigError` naming both
  fields, rather than silently filtering twice.
- **Verify:** guard test asserts the error; a raw / band-passed-clean / noise-mixed triple generated
  from one config set, with the band-passed-clean bank distinguishable from the raw one.
- **Depends on:** S26.

### S28 — Standalone post-process entry point + docs (SEP1) ☐ (2–3 h)
- **Change:** make the stage independently runnable per the §3 sanity-pass resolution — a
  `synthegm-postprocess` CLI taking a ClassifierBank in and writing one out. `docs/mixer_theory.md`
  + `docs/usage.md` then explain that band-passing is now a producer stage and `mix.bandpass_clean`
  is the legacy in-mixer path; CHANGELOG.
- **Verify:** `--help` renders; a round-trip on a small fixture bank; pr_checklist passes.
- **Depends on:** S27.

### S29 — `NoiseSelectionStrategy` Protocol + uniform default (SEP8) ☐ (2–3 h)
- **Change:** fifth mixer-side Protocol (parallel to the simulation strategies) with
  `UniformRandomNoiseSelection` wrapping today's `sample_noise_for_length`. Behavior-preserving.
- **Verify:** existing `tests/test_mixer.py` passes unchanged with the same master seed — the
  refactor must not perturb the RNG stream.
- **Depends on:** S10.

### S30 — Per-sim patient + record selection (SEP8) ☐ (2–3 h)
- **Change:** `PerSimPatientNoiseSelection` (each simulation draws one IAFDB patient; the
  highest-leverage variant per roadmap) and `PerSimRecordNoiseSelection`. Both need the trace's
  `simulation_id` and a grouping over the noise bank's `source_record`. **Parse confirmed
  2026-07-28** against `banks/iafdb_noise_v1.h5`: records are `<patient>_<placement>`, so
  `split("_")[0]` gives a clean 8-way partition (iaf1–iaf8). The `noise_bank` schema is slim
  (`source_record` + `source_channel` only, no `patient_id`), so parsing is the only route — put the
  parse in one helper with a clear error if a record ever fails the convention, rather than inlining
  `split` at three call sites.
- **Sampling decision inside this step:** segment counts per patient are very uneven (iaf1 3,227 →
  iaf4 15,548, ~4.8×). Drawing a patient uniformly and drawing a segment uniformly give materially
  different noise distributions. Pick one deliberately, document which, and record it in
  `docs/mixer_theory.md`.
- **Verify:** a mixed bank where all traces of one sim share a patient / record, asserted directly
  from the per-trace `noise_record` audit field.
- **Depends on:** S29.

### S31 — Noise-selection config dispatch + docs (SEP8) ☐ (2–3 h)
- **Change:** `noise_selection.type` block inside `mix:` + strategy dispatch by name in
  `cli/_config.py`, following the existing dispatch idiom. `docs/mixer_theory.md` then explains why
  uniform sampling destroys within-patient noise correlation and what each strategy preserves;
  CHANGELOG.
- **Verify:** `tests/test_cli_config.py` covers each type + an unknown-type error; pr_checklist
  passes.
- **Depends on:** S30.

### S32 — Absolute noise floor + config + provenance (SEP3) ☐ (3–6 h)
- **Change:** an absolute floor term in `MixerConfig` so noise no longer scales down without limit
  as the clean trace weakens (today `snr_scale` is purely relative — confirmed in the §3 sanity
  pass). Effective scale becomes the larger of the relative-SNR scale and the floor scale. **Write
  the formulation into `docs/mixer_theory.md` first** — this is math, and the doc is where it gets
  reviewed. Full height-coupled SNR stays deferred (SR1). Ships with its `mix.noise_floor` config
  and the floor recorded in the mixer provenance entry — a floor no config can reach is untestable
  from the outside, so the two halves belong in one commit.
- **Verify:** a weak-signal trace receives the floor while a strong-signal trace keeps its relative
  SNR; the realized per-trace SNR audit field reflects which regime applied; config tests;
  provenance round-trip; CHANGELOG.
- **Depends on:** S10.

### S33 — Multi-edge `planar_edge` + config + docs (SEP6) ☐ (3–6 h)
- **Change:** `PlanarEdgeStimulus.edge` → `edges: tuple[Edge, ...]` (the schema's `planar_edge` is
  already plural); backend installs one strip per edge. Sánchez stimulates three sides — the point is
  propagation-direction variety, so the dataset sampler needs to sample edge *sets*, not one edge.
  `activation.edges` / edge-count sampling in the config; docs + CHANGELOG.
- **Verify:** a two-edge sim shows wavefronts entering from both edges in the activation map; the
  single-edge case is unchanged bit-for-bit; config tests; example config runs.
- **Depends on:** S10.

### S34 — `PointStimulus` + config + docs (SEP7) ☐ (3–5 h)
- **Change:** spec + `_build_point_stimulus_2d` adapter. Focal source for the spiral-wave studies.
  `activation.type` dispatch extended to `point`; docs + CHANGELOG.
- **Verify:** activation map shows radial spread from the configured position; config tests for the
  variant + an unknown-type error; example config runs.
- **Depends on:** S10.

### S35 — `S1S2Protocol` + config + docs (SEP7) ☐ (3–6 h)
- **Change:** spec + adapter installing two timed stimuli. **`T` interaction:** S1–S2 puts two
  activations inside the capture window, which collides with SEP2's single-activation crop — the
  crop must either target the S2 activation explicitly or the sim must capture long enough to crop
  around it. Resolve when both are in; note it here so it isn't discovered late.
  `activation.type` dispatch extended to `s1s2`; docs + CHANGELOG.
- **Verify:** both activations appear at the configured intervals; the S2 wave shows the expected
  conduction slowing at short coupling intervals; config tests; example config runs.
- **Depends on:** S34.

### S36 — Docs + phase exit ☐ (3–5 h)

**`docs/simulation_theory.md` carries retired numbers — found 2026-08-16 during the S16a doc pass.**
The whole `docs/` tree fell out of step with S15 and S38 and nobody noticed until a config key
(`run.travel_allowance_ms`) turned out to be undocumented. `usage.md` and `README.md` were corrected
in S16a; **the theory doc was not**, because it needs re-derivation rather than a find-and-replace.
Both of its worked examples are computed with **`V = T`** (retired in S15, now `2T`) and
**`K = 1.97`** (retired in S38b, now `5.709836`), so every downstream number is wrong. Corrected
values, computed so the fix is mechanical:

| Location | Doc says | Correct now |
|---|---|---|
| §"four lengths" preamble (~L96) | `ap_time_unit_ms: 1.97` listed among "shipped production settings" | not a config key at all — it is `backend.model` → the card's solved `time_unit_ms` |
| table row 2, capture (~L103) | 335 ms | **671 ms** (`D=143`, `V=2T=384`, `k(0.25)=48`) — 335 is `D=0, V=T`, i.e. pre-S15 |
| table row 3, solver time | 170.05 model units | **117.52** |
| table row 4, capture samples | 1309 @ 3905 Hz | **2684 @ 4000 Hz** |
| second worked example (~L1135) | `N = 115 + 192 + 192 − 76 = 423` | **`N = 115 + 384 + 192 − 76 = 615`** |
| its solver clock (~L1139) | `423 / 1.97 = 214.7` | **`615 / 5.709836 = 107.7`** |

**Do NOT bulk-replace `anisotropy_ratio = 3` or `ap_time_unit_ms` across this file.** Several
occurrences are **historical narrative** — L257's account of the fibre-transpose bug reasons about
the ratio that was in force *at the time*, and rewriting it to 2.0 would make the story wrong.
Same class of error as the regex that ate a constructor argument in S38c. Fix the *worked examples*;
leave the *history* alone, adding a forward pointer where a reader might mistake one for the other.

**The habit this exposes:** a step that adds, retires or recalibrates a config key is not done until
`docs/` moves with it. S38a/b/c changed the config surface substantially and shipped no doc change.

**`project/architecture.md` says "four strategy Protocols" and there are five — found 2026-08-16.**
`CellModelSpec` landed in S38c (as absorbed S17, whose entry explicitly required "record the Protocol
extension + its Guardrail-3 rationale in architecture.md" — that half did not ship). The phrase
recurs **nine times**, including in Guardrail 2's own heading. S16b corrected the module map and
added the trace-key invariant; **this reconciliation was left for here on purpose.**

**It is not a find-and-replace, and treating it as one will introduce errors.** The occurrences do
not all mean the same thing:

- `SimulationSpecs` genuinely bundles **four** — geometry, substrate, activation, electrodes.
  `CellModelSpec` is *not* a member. That "four" is **correct**; changing it breaks the doc.
- `specs.py` genuinely ships four Protocols; the fifth lives in `cell_models.py`. Technically true,
  actively misleading, so it needs rewording rather than a number swap.
- `run_single` takes **five** specs now. That one is simply wrong.
- The `## The five Protocols` heading counts four strategy specs **plus** `SimulationBackend`, so it
  is now six on its own terms — the two counts in this file were never the same count.

Read each in context. Same class as the regex that ate a constructor argument in S38c, and the
same class as the `anisotropy_ratio = 3` passages in `simulation_theory.md` above.

- **Change:** **rework the `examples/` config set** — it has drifted (Daniel, 2026-08-01: new configs
  added ad hoc during the phase) and is no longer a good cross-section of the runs we actually want
  to demonstrate. Decide the set deliberately — one clean baseline, one noise-mixed, one calibration,
  one sweep (SEP11), one probe (SEP13 — **already shipped as `examples/synthegm_probe.yaml` in
  S16b**, so only the remaining four are open) — rather than accreting one per experiment. Also trim
  `roadmap.md` of everything shipped (the Phase 1.5 cluster, the polymorphic
  `stimulation` entry, and the `RunConfig.cell_model` open question closed by D2); finalize
  `CHANGELOG.md`; make sure `project/architecture.md` reflects the fifth spec, the `SimulationResult`
  widening, and the post-process stage.
- **Also enforce "config, not hardcoded" (Daniel, 2026-08-11).** Audit run on that date came back
  clean — every generation parameter already lives on a config-backed field, and the in-code
  constants are physics/math (`1/r` weighting, `10^(dB/10)`), not study values. **One real soft
  spot:** `_build_label_policy` reads `label_policy:` with `default={}`, so a generate config that
  omits the block silently gets `global_density` at `threshold = 0.1` — and that unchosen value is
  then stamped into the bank's provenance as though it were a decision. Make the block **required**
  for a generate run. (The three `synthegm_mix*.yaml` configs correctly have no label policy: the
  standalone mixer preserves its input bank's labels and never labels anything — checked, not
  assumed.)
- **Verify:** a generate config missing `label_policy:` raises; no config example relies on a code
  default for any value stamped into bank provenance; full pre-PR run in
  `intracardiac-platform/project/pr_checklist.md`; the release
  checklist's plan-deletion step happens at Cleanup, **after** the Effort section below is rolled up.
- **Depends on:** all prior steps.

## Effort tracking

> **Not tracked for Phase 1.5** (Daniel, 2026-07-29). Flow-down process, organization, and effort
> tracking are being reworked: Daniel + the project-lead will design the method after this phase's
> flow-down completes, and it applies from **Phase 2 onward**. So this phase records **estimates
> only** — the `Estimate` column below is still the flow-down deliverable the project-lead reads into
> design §6; `Actual` / `Elapsed` / `Sessions` stay empty by decision, not by oversight.
>
> **Two consequences worth stating plainly.** (1) Phase 1.5 contributes **no rows** to
> `estimation_ledger.csv`, so Phase 2's estimates are still cold-start by analogy — the ledger's first
> real data arrives a phase later than the method assumed. (2) The release checklist's "roll up the
> Effort section before deleting the plan" gate is **moot** for 1.5; there is nothing to roll up, and
> cleanup should not block waiting for it.
>
> Method for reference (dormant this phase):
> `intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md` §7 (mechanism) + §8
> (complexity rubric).

### Effort by issue

> _Estimates only this phase. The `Active` / `Elapsed` / `Sessions` columns are kept so the table
> shape still matches the template for when tracking resumes in Phase 2._

| Issue | Task-type | Cx | Estimate | Active | Elapsed | Sessions |
|---|---|---|---|---|---|---|
| SEP12 | schema-migration | 5 | 16.5–31 h | | | |
| SEP2 | pipeline (algorithm) | 2 | 7–12 h | | | |
| CL-143 + B13 | schema-plumbing | 2 | 2–4 h | | | |
| SEP10 | pipeline | 1 | *(absorbed into S15)* | | | |
| SEP13 | pipeline | 2 | 3–6 h | | | |
| SEP5 | pipeline (physics) | 5 | 11–20 h | | | |
| B12 | pipeline | 2 | 2–4 h | | | |
| SEP11 | pipeline | 5 | 12–20 h | | | |
| SEP1 | pipeline (refactor) | 3 | 6–9 h | | | |
| SEP8 | pipeline | 3 | 6–9 h | | | |
| SEP3 | pipeline (math) | 2 | 3–6 h | | | |
| SEP6 | pipeline | 2 | 3–6 h | | | |
| SEP7 | pipeline | 3 | 6–11 h | | | |
| *(phase exit)* | docs | — | 2–4 h | | | |
| CL-169/170 + CL-166/167 | simulator (physics) | 3 | 6–10 h | | | |
| **Repo total** | | **40** | **85.5–152 h** | | | |

**Estimate basis.** The ledger is empty, so these are reference-class-by-analogy, not
`points × measured rate`. Anchors used: SEP12 is *the* rubric's L anchor; SEP5 is sized equal to it
(comparable surface, higher physics novelty, lower coordination); SEP11 equal again (rewrites the
sampling loop and gates STU4). The XS/S items are anchored on the `noise_bank` `bank_id` add (S).
Ranges are deliberately wide — roughly 2× low-to-high — because there is nothing yet to narrow them
with. First cleanup replaces all of this with measured rates.

**Reconciled 2026-08-11 with the step-size compression pass.** This table and the Scope table had
drifted apart from the steps: SEP12 still carried the 12–24 h planned before S9 was added mid-wave,
and SEP2 read 4–8 h against steps summing to 7–10 h. Both now equal the sum of their own steps, and
the repo total moved 71.5–134 h → **79.5–142 h** — **no scope was added**; the tables simply stopped
disagreeing with the plan they summarize. **Complexity points are unchanged at 37**, which is the
tell that this was a bookkeeping fix: Cx is assigned per §3 issue, and compressing steps does not
move an issue's difficulty. Anyone reading the two numbers together should see hours up, points
flat, and conclude estimate error — not growth.

## Notes / decisions log

- 2026-07-28 — Plan drafted from design §3 (ten SEP issues) + §4 (B12, B13). Five design notes
  recorded: D1 (`SimulationResult` widening), D2 (cell model as a fifth spec), D3 (noise-mixed
  reconstruction route), D4 (**escalation** — is `synthetic_bank` still opt-in, given STU1/4/5 read θ
  from it?), D5 (`T` and `𝒫` come from study §8.1).
- 2026-07-28 — Two verification-worthy unknowns flagged inside steps rather than assumed away:
  whether the pinned Finitewave ships a Courtemanche 2D model (S18), and whether an IAFDB patient
  is parseable from the noise bank's `source_record` string (S32). **Both investigated the same
  day; both resolved yes** — neither becomes an escalation.
- 2026-07-28 — **Finitewave Courtemanche: available.** The pinned `finitewave>=0.9` resolves to
  0.9.3, whose model tree is `finitewave/cpuwave/model/` with `aliev_panfilov.py` and
  `courtemanche.py` side by side; `Courtemanche(CardiacModel)` is dimension-agnostic and selects a 2D
  or 3D stencil from `cardiac_tissue.dimensions`. 0.9.3 dropped the dimension-suffixed class names in
  favour of bare ones but **re-exports back-compat aliases** (`AlievPanfilov2D = AlievPanfilov`,
  `CardiacTissue2D = CardiacTissue`, `ECG2DTracker = ECGTracker`, `StimVoltageCoord2D =
  StimVoltageCoord`, `Courtemanche2D = Courtemanche`) — which is why the existing backend still runs
  unchanged. Conductances come from `courtemanche/ops.py::get_parameters()`, are installed as
  instance attributes, and are read back via `getattr` at `run_ionic_kernel` time, so a curated
  scaling is a plain assignment before `run()` — SEP11's `path` addressing needs no model patching.
  Six other ionic models ship alongside (TenTusscherPanfilov2006, BuenoOrovio, FentonKarma,
  LuoRudy91, MitchellSchaeffer, Barkley) — relevant to later phases, not 1.5.
  **Latent risk, not acted on:** the repo is written entirely against the deprecated alias names. If
  a future finitewave drops them, every `fw.*2D` call site breaks at once. Cheap hedge — move to the
  bare names as part of S17, since that step already touches the backend's model construction.
- 2026-08-11 — **Bank generation is not a plan step** (Daniel, correcting me). *"Generating banks
  shouldn't be a code implementation step. I will just generate each of the banks after the code
  itself is done... which means bank generation happens after all the code for all the repos has
  been finished. That ensures we are consistent with repo version numbers for the generated banks."*
  Plus the rule that falls out of it: **any parameter a generated bank needs is a config value, not
  a hardcoded one.**
  **This invalidated two of the four ordering constraints I had just written** (constraints 2 and 3:
  *defects before the banks that inherit them*, *trace-shape changes before study-bank generation*).
  Both assumed banks get generated as implementation proceeds. They don't — so "land X before the
  banks that would otherwise need regenerating" is vacuous, and the ordering rests on real code
  dependencies (exactly one crosses issues: SEP5 → SEP11), the re-pin going first, and review flow.
  **The order itself did not change** — it was defensible on the surviving grounds — but the stated
  reasons did, and a plan whose reasons are wrong will be re-derived wrongly the next time someone
  reorders it.
  Also demoted: **S24 "unblocks STU7" is a *code* hand-off**, not a data one. STU7 gets the ability
  to read a θ-spec bank; the screening banks it ranks are generated in the post-code data stage. The
  §8.2 chain still holds there, because STU7's result reaches SEP11 as **config input** to S25 —
  which S25 was already scoped for.
  **[CL-161](../../intracardiac-platform/phases/phase_1_5/coordination_log.md) withdrawn by
  [CL-162](../../intracardiac-platform/phases/phase_1_5/coordination_log.md)** — it asked which
  features must precede bank generation at S24, a question that only exists if generation happens
  during implementation.
- 2026-08-11 — **Wave 2 re-ordered into execution order and renumbered** (Daniel). Two changes, one
  cause. The plan had said the Wave-2 threads *"can proceed in any order"* — a leftover from when the
  assumption was several chats writing code in parallel. In practice **the long pole is Daniel's
  review time, not machine time**, so implementation is serial, and "any order" stops being a freedom
  and becomes an unmade decision that would get re-litigated at every step boundary. The order is now
  settled up front with its four constraints written down: the re-pin first (it moves
  `trace_duration_ms` to 192, so banks either side of it are not comparable), then the CL-143 defect
  before anything can inherit it, then SEP2's crop before any §8.2 study bank is generated, then the
  STU7 handoff as early as its dependencies allow. **Two of those four were corrected hours later —
  see the next entry.**
  **Renumbered rather than living with out-of-order ids.** The cost was checked, not assumed: no file
  outside this plan references a synthetic-egm Wave-2 step (S-numbers are repo-local — the hits
  elsewhere are other repos' own), and none of Wave 2 has shipped, so the ids were still free to move.
  The deciding argument is that the forward-dependency check is literally `dep_number < step_number` —
  it works only while numbers *are* execution order, and it is what caught S7→S11 during the flat
  renumber. Letting the two diverge would have spent that property immediately. iafdb-pipeline's plan
  does run out of order (S0–S6, S9–S11, S7, S8, S12), but that divergence was **earned**: those steps
  were discovered mid-execution, when renumbering was no longer available. Hence the standing rule —
  **numbers are assigned in execution order at plan time; steps discovered during execution take the
  next free number and sit in the document where they happened.**
  Both tables now read in execution order too. **SEP2 is deliberately non-contiguous** (S12 ·
  S14–S15): the CL-143 fix belongs between its re-pin and its crop work, and keeping an issue
  contiguous would have meant ordering by paperwork instead of by dependency.
- 2026-08-11 — **Step-size compression pass** (Daniel): Wave 2 went **37 steps → 25** (repo total
  49 → 37). Prompted by the same problem surfacing in iafdb-pipeline — steps had shrunk to where
  some carried no code at all, or a handful of lines. The recurring shape here was a per-issue tail
  of *"config"* then *"docs"*, which is a decomposition of the **plan**, not of the **work**: it
  yields a commit adding a capability nothing can reach, followed by a commit of pure prose.
  Config dispatch and docs now ship inside the feature's own step. Two splits were kept against the
  rule and the reason recorded in the step itself — S17 (a behavior-preserving refactor whose entire
  verification is *"nothing changed numerically"*, which stops meaning anything once Courtemanche is
  in the same diff) and S12 (a dependency re-pin, which has to stay bisect-isolable). SEP10 lost its
  standalone step: it is a config value on SEP2's block and is not testable until cropping runs, so
  it now rides in S15.
  **The pass also caught two stale estimates** that the per-issue numbering had hidden: SEP2's row
  read 4–8 h while its own steps summed to 7–10 h, and SEP12's row still read the 12–24 h planned
  before S9 was added mid-wave from review findings (actual planned total 16.5–31 h). The repo
  estimate moves 71.5–134 h → **79.5–142 h** — no new work, just the table finally agreeing with
  the steps underneath it.
- 2026-08-11 — **Steps renumbered to the fleet's flat `S<n>` scheme** (Daniel), matching egm-signal and
  iafdb-pipeline: ids are sequential in *execution* order rather than per-issue, the §3 issue moved into
  each step's title, and the Scope table's *Steps* column now gives ranges (`S33–S24`). S0–S11 are the
  shipped Wave 1; S17+ is Wave 2.
  **The renumber earned its keep immediately:** ordering by execution exposed a **forward dependency** —
  the bank-reference step listed the *docs* step as its prerequisite, which was chronology (it happened
  to be written afterwards) rather than necessity. Corrected to the real prerequisite, both-banks-emitted.
  A per-issue numbering hides that class of error, because nothing about `SEP12.9 depends on SEP12.8`
  looks wrong.
- 2026-08-11 — **Wave-2 spin-up pass over the plan** (28 log entries landed since the last check).
  **SIG1 shipped (egm-signal v0.4.0)**, and it delivers more than the plan assumed: `SingleActivationWindower`
  is explicitly "the synthetic case" (argmax, no threshold), `UniformPositionGenerator` is
  **point-collapsible** so SEP10's two arms are one class, and `AnchoredWindow.realized_position` is
  documented as the value to store. **SEP2 shrinks M(3)/7–12 h → S(2)/4–8 h** and **SEP10 → 0.5–1 h**:
  this repo owns the *sizing response*, wiring and provenance, not the crop math. Repo total 74–137 → **72.5–136 h**.
  **New work:** **S13** for the CL-143 clean-bank θ-id divergence (design §7 Wave 2 assigns it
  here), with **D8** recommending *record the θ file's real id* over *emit a second clean θ bank* —
  one question for Daniel inside it. **S33** added as the wave-opening step: the re-pin
  (contracts 0.6.0→0.6.1, data 0.6.0→0.6.2, signal 0.2.0→0.4.0) plus the three chores the PL routed to
  "SEP2's next touch" — CL-117, CL-118, CL-112.
  **CL-117 is worse than a chore:** `__version__` is hardcoded `"0.2.0"` against tag `v0.3.0`, so every
  bank written this phase stamped a **wrong `producer_version`** — the reproducibility field CL-109
  argued to keep. Fixing it early means the phase's real banks carry a true value.
  **Also noted:** design §3 SEP2/SEP13 were rewritten after CL-130 (synthetic **detects**; probe sweeps
  by exact shift) — both already match this plan's D6 and step text, no change needed. §7 says SEP13
  must not be scheduled last in the wave, since STU8 sits behind it.
- 2026-08-01 — **S9: four defects Daniel found reviewing a generated bank.** Three he
  reported, plus one his first report exposed:
  1. **The mixed ClassifierBank's id read `synthetic_unknown_noise_mixed`** — a **regression I
     introduced in S8**. `_resolve_noise_mixed_id` re-derived the id from `cell_model` read out
     of the clean bank's `bank_metadata`; cleaning generation params off the ClassifierBank an hour
     earlier removed that key, so the lookup fell back to `"unknown"` and produced a
     *wrong-but-valid* id. The tests missed it because they all pass explicit ids. Now derived from
     the clean bank's **id**, so one string anchors the whole family; an id-less clean bank raises
     instead of fabricating one (egm-data refuses to write such a bank anyway).
  2. **The θ companion entry didn't name the θ bank written beside it** on the mixed path — the mixer
     copied the clean bank's entries forward verbatim, so a mixed bank pointed at the *clean* run's θ
     bank. The companion is not shared history; it is now re-pointed at the mixed run's own θ bank.
  3. **Relative paths gave up too early.** My "any `..` → absolute" rule was over-conservative: banks
     routinely sit in sibling directories under one project root that moves as a unit, so
     `../noise/x.h5` is portable and useful. Rule is now "relative iff the two share a real
     (non-root) common ancestor"; only genuinely unrelated trees go absolute.
  4. **`synthegm-mix` still wrote an absolute noise path** — I wired `output_bank_path` into the
     inline path at S7 and missed the standalone CLI.
  **Pattern worth noting:** (1) is the second time this session a *removal* broke a silent
  dependency at distance (cf. the `_config.py` key that stopped being read). Both were caught by
  Daniel exercising the real artifact, not by the suite — the tests specify behaviour and the
  fixtures avoid the defaults where these bugs live.
- 2026-08-01 — **S8: ClassifierBank metadata de-duplicated** (Daniel spotted it; his rule —
  "needed for training stays, otherwise if it's in the synthetic bank, clean it out" — is verbatim
  `synthetic_bank_source_of_truth.md` §12). **Audit:** my direct writer emitted **23 bank keys + 9
  trace keys**; egm-data's converter, which already implements §12, emits **6 + 6**. Every one of my
  six extra trace keys was a generation parameter — i.e. the same flat per-trace columns the 2.0
  restructure removed from the *synthetic* bank, still sitting in the *ClassifierBank*. The
  restructure had moved them out of one artifact and left the copy in the other.
  **Third instance this session of two writers of one artifact disagreeing** (after
  `sim_id`/`simulation_id` and `amp_type`); the converter is the reference implementation and my
  direct writer predated the decision.
  **Safe to remove — checked first:** zero references to any of the six keys across egm-classifier /
  egm-studio / egm-features `src/`; only `patient_id` is read (the patient-aware split). egm-studio
  flattens `trace_metadata` generically, so columns vanish without breaking code — and per CL-102 it
  now reads θ from the typed SyntheticBank, which is the point.
  **Two Daniel calls:** keep `producer` / `producer_version` (reproducibility, and 2.0 has nowhere to
  record them — dropping would *lose* the fact rather than de-duplicate it); and do **not** write
  `snr_db` / `noise_record` / `noise_channel` on a clean bank, since a NaN reads as "mixed, SNR
  unknown". That is a deliberate divergence from egm-data's converter, which always writes them.
- 2026-08-01 — **S7 added + done: bank-reference semantics** (Daniel's review of a generated
  bank). Three findings, all confirmed by audit: banks used **absolute** paths; a clean run's entry
  named its own not-yet-written output; a noise-mixed run's entry named a clean bank whose traces
  differ from the file's. Root cause is semantic — `bank_path` means "where the source bank was
  *loaded from*", and a producer has no source file, so the field was filled with whatever was handy.
  **Fix (agreed with Daniel):** three entry kinds — **origin** (`<local>`), **companion** (a path to
  an artifact that describes these traces: noise bank, θ bank), **source** (a real upstream bank; not
  produced here today). `<local>` over `""` because a blank must stay available as a *bug* signal;
  angle brackets because `<`/`>` are illegal in Windows filenames, so the sentinel cannot collide with
  a real path — including the bare relative filenames companions now carry. Companion paths are
  relative **only when the target is inside the bank's directory tree** (the case where relative
  actually buys portability), absolute otherwise; the first cut used a `..`-count threshold, which was
  arbitrary — the tree test is the principled version.
  **Audit results:** nothing reads `bank_path`, so none of this is breaking; **B14 does not cover it**
  (that item is egm-classifier's `training_run_record`), so producer bank paths were a genuine gap;
  and the θ companion entry needs no contracts/data change — `bank_type` is a free string, there is no
  1:1 assumption between entries and traces, and it round-trips. It only works because the θ bank got
  a **distinct id** earlier today; before that it would have collided with the origin entry under
  `concat`'s dedup-by-id.
- 2026-08-01 — **Bank-id collision found in review and fixed (Daniel).** Both banks were taking the
  **identical** `ArtifactId`, derived *and* overridden — fine while the synthetic bank was an optional
  sibling view of the same data, a real collision once D4 made them parallel artifacts, since the phase
  manifest keys on stable ID. **Not covered by B13**, which is clean-vs-noise-mixed ClassifierBank ids.
  Chosen shape (Daniel): **one base id + a known marker** — ClassifierBank keeps the base, the
  `synthetic_bank` gets `theta` inserted *before* any trailing date so the result stays inside the
  `ArtifactId` grammar. One override names both. Rejected two independent overrides because setting
  only one silently leaves the other derived, producing a pair that looks unrelated with nothing to
  detect it. Also dropped the redundant `clean_intermediate: null` lines for the same reason as
  `synthetic_bank: null` — `null` and absent are identical to the loader.
- 2026-08-01 — **Example-config cleanup (Daniel's review).** My flag removal left
  `synthetic_bank: null` in six of eight examples with a three-line comment bolted on — but `null` and
  *absent* are identical to the loader, so the line documented nothing and read as though it did
  something. Deleted from all six; the field is instead **demonstrated once**, with a real custom path,
  in `synthegm_calibration.yaml` where naming the file actually helps. An example should show a
  feature being used, not hint that a key exists.
  **The review also surfaced a bug I introduced:** `build_mix_config` stopped reading
  `output.synthetic_bank` when I dropped the field, so a mix config setting it would have been
  *silently ignored* — the same failure mode I had just argued against for the retired flag. Standalone
  mix now rejects **both** synthetic-bank keys.
- 2026-08-01 — **SEP12 complete (.3b/.5/.6/.7/.8).** Flag retired (error, not silent ignore —
  rejected even when `false`, since accepting it implies the writer still honours it); both banks
  always written with a derived sibling path; seven example configs updated. **143 tests green**
  (+1 `slow`), ruff + bare mypy clean.
  **S10 ran for real.** finitewave installs fine in the sandbox, so the equivalence gate is an
  actual two-simulation Finitewave run, not an assertion about mocks: signals byte-identical to
  simulator output in both banks, labels identical to the `DatasetResult`, and every value 1.1 stored
  per-trace (density, edge, height, electrode row) recovered exactly from the 2.0 per-simulation
  config. Kept as a `slow`-marked test (`pytest -m slow`) so the default MockBackend suite stays ~1 s.
  **That real run earned its keep immediately:** it passed while the fast equivalent failed, which
  exposed that the `SimulationSpecs` fixture I wrote at S0 was internally inconsistent — it
  declared a 1×5 electrode grid while using bipolar pairs that need 8 electrodes, so the per-pair
  `electrode_row` disagreed with what the runner computes. Production code was right; the mock was
  wrong. Second time this session a too-convenient fixture nearly hid a real check (cf. the thin
  `backend_metadata`).
- 2026-08-01 — **S1 + .3 + .4 landed together — the split was artificial** (Daniel's review).
  S0 was scoped to avoid the tag so it could start early, but the whole point of the step is the
  migration, and re-pinning proves the split can't hold: `builders.py` imports `StimEdgeEnum`, which
  v0.6.0 removes, so the repo does not *import* between the re-pin and the builder rewrite. .4 came
  along because the two CLIs call the retired reconstruction builder. Lesson recorded for the phase
  retro: **a migration step is not divisible below "the repo imports again"** — wave planning should
  size steps by that boundary, not by what can be made to look independent.
  Landed: re-pin contracts+data **v0.6.0**; new `simulate/bank_config.py` (spec → typed per-function
  models); `build_synthetic_bank_from_dataset` rewritten to 2.0 (per-sim `simulations/` group,
  collapsed `traces/`, int label, θ-spec regime with empty knobs — the schema **requires**
  `generation_params`, so S6's trivial writer folded in too);
  `build_synthetic_bank_from_classifier` retired per D3 with the inline mix path now building from
  the `DatasetResult`; standalone `synthegm-mix` raises a clear `ConfigError`. **132 tests green,
  ruff + bare mypy clean**, including an end-to-end round-trip through egm-data's real writer/reader
  and a check that the bank we emit is convertible by egm-data.
  **mypy earned its keep again:** the generated models wrap constrained scalars in `RootModel`s
  (`Threshold`, `PairIndexItem`, `LabelItem`, `ElectrodeIndice`, `PositionMm`) and `Edge` is an Enum,
  not the producer's `Literal`. Pydantic coerces all of these at runtime, so the tests passed while
  the static types were wrong — exactly the class of thing that silently rots.
- 2026-08-01 — **CL-103 downgraded -> D7 + FB-17.** Verified no `src/` in any repo calls
  `synthetic_bank_to_classifier`, so the `amp_type="mv"` hardcode is a latent trap, not a live bug —
  adopting the converter (my own CL-103 suggestion) is what would have created the exposure.
  Withdrawn. Producer keeps both builders; proper fix (bank states its amplitude convention,
  required-on-write, converter reads it) backlogged as **FB-17** for the Phase-2 contracts bump with
  FB-15/FB-16. Added **S3** — cross-check that the two banks agree on label + `simulation_id`
  per trace — to cover the drift risk two builders reintroduce.
- 2026-08-01 — **S0 done.** `SimulationSpecs` added to `simulate/result.py` and populated in
  `run_single`; `sim_id` -> `simulation_id` fleet-wide in this repo; architecture.md records the
  Guardrail-2 widening. 125 tests green, ruff + bare `mypy` clean. Also landed **CL-100** here
  (`[tool.mypy] files = ["src","tests"]`) since the step already touched tests — it surfaced exactly
  the sloppiness CL-100 predicted: a `parametrize` widened `Edge` to `str`, and a test read
  `size_mm` through the `GeometrySpec` Protocol, which by design exposes only `type`. Both fixed
  properly (narrow with `isinstance`, parametrize over `EDGES`) rather than ignored.
  **Sandbox note:** the local siblings are already v0.6.0, so verifying S0 against the *current*
  v0.5.3/v0.5.0 pins needed `git archive` exports of those tags — editable siblings would have
  silently tested the post-re-pin world (the insulation CL-102 mentions).
- 2026-07-30 — **CL-075: new issue SEP13** (positional-sensitivity probe bank), from the §8.9 code
  audit — research promoted the probe from optional to **core** because the anchored-vs-varied A/B
  alone can't separate "positional shortcut" from "generic augmentation" (CL-071). Scored **S (2),
  3-6 h, 2 steps**; repo total 35 -> **37 pts, 74-135 h**. Added design note **D6**: the probe sweeps
  by *exact shift* off a single detection, not by re-running SEP2's detector per grid point --
  otherwise detector jitter blurs the very axis STU8 plots against. Reuses `synthetic_bank` +
  `activation_position`; no schema change. **STU8 depends on it, so it must not be scheduled last in
  Wave 2.**
- 2026-07-30 — **`activation_position` field approved (CL-062)** — S23 is no longer conditional.
  egm-contracts defines it once in `common` with both banks `$ref`ing it (CL-064); egm-data carries
  it on the typed reader and keeps it off the ClassifierBank with a negative test (CL-063).
- 2026-07-30 — **SEP2 unchanged by the CL-065->CL-068 exchange.** `P_synth` superset of `P_iafdb` was
  questioned and re-affirmed: `activation_position` is an augmentation axis, not a realism axis, so it
  is excluded from the STU4 objective and STU5 distance rather than the ranges being matched
  ("augmentation axes are not realism axes"). Recorded in D5 so the inequality isn't later mistaken
  for an inconsistency. Also noted there: CL-072's §8.1 go/no-go could still widen `T` 192 -> 256, so
  S35 must read `T` from config rather than hardcode it.
- 2026-07-29 — **CL-059 answered (CL-060): the crop `p` has nowhere to be persisted.** Design §5
  assumed it "rides the per-sim config", which fails twice — v2.0's collapsed `traces/` has no
  position column, and the *realized* position is inherently per-trace (the wave sweeps the grid, so
  each pair activates at a different time; detection jitter + `round(frac·(T−1))` add more). Asked
  the project-lead for a nullable per-trace `activation_position` on CON1, mirroring the
  `iafdb_bank` field IAF3 just added — written absent in Wave 1 (S2), populated Wave 2
  (S23). Negligible cost, no §6 change. **S23 is conditional on that decision.**
- 2026-07-29 — **CI ruff pinned** to `ruff==0.15.17` in `.github/workflows/ci.yml` per CL-041 /
  CL-024 §1 (fleet-wide pin, option (a) at a single version). No 0.16 reformat run — the pin makes it
  moot, and 0.15.17 leaves Markdown alone, which is what protects the shape-annotation alignment I
  argued for in CL-007. Verified: `ruff check .` passes, `ruff format --check .` reports 31 files
  already formatted. `.pre-commit-config.yaml` was already `v0.15.17`, so no pre-commit change here.
- 2026-07-29 — **CL-024 §3 + §4 folded in.** Join key: my CL-008 resolved as option 1 — producer
  renames `sim_id` → `simulation_id`, landed in **S0** (earliest step, no schema dependency, so
  egm-studio's fixture flip and egm-data's writer check aren't held behind Wave 1). `T` = **192 ms**
  at 1 kHz on a 64-sample grid, so D5's `T` half is closed, configs move 200.0 → 192.0, and S35
  gains a grid guard. `𝒫` still pending §8.1.
- 2026-07-29 — **Effort tracking off for Phase 1.5** (Daniel). *(Note: CL-024 §5b records the
  opposite rule — a repo chat's flow-down session counts toward its issues' `Actual`. Daniel's
  2026-07-29 decision is the later one and governs 1.5; he is raising the process rework with the
  project-lead.)* Flow-down process, organization, and
  effort tracking go back to the drawing board with the project-lead once 1.5 flow-down completes;
  the new method applies from Phase 2. This plan therefore carries estimates but no actuals, 1.5
  contributes no ledger rows, and the release checklist's roll-up-before-delete gate is moot here.
  No transcript reconstruction attempted — the decision is to not track, not to backfill.
- 2026-07-28 — **D4 resolved** (project-lead): both banks emitted for synthetic runs, joined by
  `simulation_id`; ClassifierBank stays source-agnostic (no θ); `synthetic_bank` is a **parallel**
  artifact, not the ClassifierBank's source; required for T4. Duplicated trace signal accepted →
  FB-11. Flag mechanics delegated to this repo → **retire `also_emit_synthetic_bank`** rather than
  default it true, with a `ConfigError` on the legacy key. New step **S5**; SEP12 estimate
  11–21 h → **12–23 h**, repo total **71–129 h**. `project/architecture.md`'s "Why not SyntheticBank
  by default" rewritten to "Two outputs, different purposes, joined by `simulation_id`" — the doc
  marks it as Phase-1.5/v0.4.0 design, since the flag doesn't actually retire until SEP12 lands.
  `docs/` still describes v0.3.0 behavior correctly and is deliberately **not** pre-updated; it moves
  at S11.
- 2026-07-28 — **SEP11 stays one issue** (project-lead). Screening circularity resolved by promoting
  egm-studio's screening to its own issue (**STU7**); SEP11 is one capability run twice, not
  SEP11a/SEP11b. Promotion test = "distinct code capability?" — OAT vs LHS/grid is a swapped sampler
  over a shared harness (one capability); screening (`screening.py`) is genuinely different code (its
  own issue). Steps re-ordered so the harness + OAT sampler land first (S23–S24, handing off to
  STU7) and the LHS/grid sampler follows (S25); the run-ordering between them is §8.2 data flow,
  **not** a code dependency. Escape hatch recorded on the issue. Estimate 10–20 h → **12–21 h**
  (the sampler Protocol boundary + the OAT/LHS split are real added structure); Cx unchanged at L.
- 2026-07-28 — **IAFDB patient parse: confirmed.** `banks/iafdb_noise_v1.h5` (schema 1.0, 66,355
  traces × 200 samples @ 1 kHz) has 31 distinct `source_record` values, all
  `<patient>_<placement>` — iaf1–iaf8 × {svc, ivc, tva, afw}, minus `iaf6_ivc` which is absent.
  Channels are CS12/34/56/78/90. Matches
  `intracardiac-platform/references/iafdb_dataset_primer.md` §2.
