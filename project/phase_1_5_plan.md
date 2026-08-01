# synthetic-egm-pipeline — Phase 1.5 implementation plan

**Repo:** synthetic-egm-pipeline · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** in progress · **Progress:** 4/45 steps done
**Repo estimate:** **74–136 h** active (37 complexity points; cold-start ranges — the
`estimation_ledger.csv` is empty, so every estimate here is by analogy against the §8 reference
anchors, not `points × measured rate`)

---

## Scope — what this plan covers

Eleven §3 core issues plus the two single-repo §4 backlog items assigned to this repo. Wave numbers
are from design §7: Wave 1 is the schema migration (no new behavior), Wave 2 is features.

| Phase item | Wave | What it needs from this repo | Cx | Estimate | Steps |
|---|---|---|---|---|---|
| SEP12 | 1 | Write `synthetic_bank` v2.0 with today's behavior — per-sim typed per-function config, collapsed `traces/`, int label, trivial θ-spec, both banks always emitted | L (5) | 12–24 h | SEP12.1–.8 |
| SEP5 | 2 | Courtemanche 1998 human-atrial cell model alongside Aliev–Panfilov | L (5) | 10–20 h | SEP5.1–.5 |
| SEP1 | 2 | Band-pass out of the mixer into a general, independently-runnable post-processing stage | M (3) | 6–9 h | SEP1.1–.4 |
| SEP8 | 2 | Pluggable `NoiseSelectionStrategy` (uniform default + per-sim patient / record) | M (3) | 6–9 h | SEP8.1–.4 |
| SEP3 | 2 | Absolute noise floor in the mixer (today's SNR is purely relative) | S (2) | 3–6 h | SEP3.1–.2 |
| SEP2 | 2 | Controlled-position crop — activation at fractional `p ∼ 𝒫`, sim sized to the widest `p`, no padding | M (3) | 7–12 h | SEP2.1–.5 |
| SEP10 | 2 | Anchoring opt-in flag — fixed `p` vs the `𝒫` range | XS (1) | 1–2 h | SEP10.1 |
| SEP13 | 2 | Positional-sensitivity probe bank — one sim, crop offset swept on a grid, morphology/seed held constant | S (2) | 3–6 h | SEP13.1–.2 |
| SEP6 | 2 | Multi-edge `planar_edge` activation variant | S (2) | 3–6 h | SEP6.1–.2 |
| SEP7 | 2 | `point` + `s1s2` activation variants | M (3) | 6–12 h | SEP7.1–.3 |
| SEP11 | 2 | θ-sweep harness + `generation_params` writer + pluggable sampler (OAT first, then LHS/grid) — one capability, run twice per §8.2 | L (5) | 12–21 h | SEP11.1–.5 |
| B12 | 2 | Resource / CPU cap on a generation run | S (2) | 2–4 h | B12.1 |
| B13 | 2 | Custom bank id for the clean bank in noise-mix runs | XS (1) | 1–2 h | B13.1 |
| *(phase exit)* | — | Roadmap trim · CHANGELOG · architecture.md reconciliation | — | 2–3 h | X.1 |

**Cross-repo prerequisites.** SEP12 cannot start until **egm-contracts v0.6.0** and **egm-data
v0.5.x** are merged + tagged (design §7 Wave 1, re-pin cascade); we are pinned to contracts v0.5.3 /
data v0.5.0 today. SEP2 depends on **SIG1** (`extraction.activation_based`) landing in egm-signal.
Everything else is unblocked once SEP12 is in.

## Design notes

Seven decisions. D1, D2, D3, D6 and D7 are repo-internal (mine); D4 was escalated and is now
settled by the project-lead; D5 is external input, half of it still pending.

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
for them. Decision: **fifth spec.** Record in architecture.md at SEP5.1, and note that
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
`also_emit_synthetic_bank` is set on the standalone path. Confirm at SEP12.4.

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
**SEP12.3b** covers with a cross-check test.

### D6 — the probe sweeps by exact shift, not by re-detection *(repo-internal)*

SEP13 is not "call SEP2's crop N times." SEP2 places the activation by running SIG1's detector and
cropping to the sampled `p`, which carries detector jitter — fine when `p` is a training-augmentation
draw, wrong for a probe. STU8 plots model output **against** activation offset; if the offset axis is
itself noisy, the jitter blurs the curve the study exists to measure, and the anchored-vs-varied
difference gets harder to see for a reason that has nothing to do with the models.

So the probe detects **once** on the source trace and then places the activation by exact integer
shift per grid point. Realized position equals requested position by construction, the x-axis is
exact, and it is cheaper (one detection, not N). The verification is correspondingly stricter: assert
the emitted `activation_position` column *reproduces the grid*, not that it approximates it.

### D5 — `T` is now fixed at **192 ms**; `𝒫` still comes from study §8.1 *(external input)*

**`T` settled (CL-024 §4, 2026-07-28): 192 ms at 1 kHz.** The constraint is `T ≡ 0 (mod 64 samples)`
— CLF3's MobileViT needs it — so §8.1 optimises on the 64-sample grid and the 150–250 ms window
rounds to 192 ms (rounding to the grid was chosen over relaxing the architecture or padding). Two
consequences here: every shipped config's `run.trace_duration_ms: 200.0` becomes **192.0**, and
SEP2.2's sim-sizing derives from 192 ms rather than a placeholder.

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
drop rates spike, which triggers widening `T`. SEP2.2's sizing derives from `T`, so it should read the
value from config rather than assume 192.

---

## Steps

Each step is one focused commit, ends green (its own tests + `ruff format` + `ruff check` + `mypy`),
and states its verification. ☐ todo · 🔨 wip · ✅ done.

### Wave 1 — SEP12: `synthetic_bank` v2.0, current behavior

> The Wave-1 gate (design §7) is: a bank regenerated with today's config round-trips through v2.0
> and yields the **same traces + labels** as before the restructure. No new generation behavior in
> any of these steps.

#### SEP12.1 — Per-sim specs on `SimulationResult` + rename the join key ✅ (2–4 h)
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

#### SEP12.2 — Spec → contracts per-function model mapping ✅ (2–4 h)
- **Change:** new `simulate/bank_config.py` — pure functions mapping each spec concrete to its typed
  egm-contracts model (`geometry`, `cell_model`, `substrate`, `activation`, `electrodes` incl. the
  realized `pairs` list, `backend`, `label_policy`, `label_names`, `substrate_summary`). No h5py, no
  finitewave (Guardrails 1 + the no-HDF5 rule).
- **Verify:** unit tests per mapper — round-trip each Phase-1.5 concrete through its contracts model
  and back to equal field values.
- **Depends on:** SEP12.1; **egm-contracts v0.6.0 tagged**.

#### SEP12.3 — Re-pin + rewrite the clean SyntheticBank builder ✅ (2–4 h)
- **Change:** re-pin `myocard-egm-contracts@v0.6.0` + `myocard-egm-data@v0.5.x` in `pyproject.toml`;
  rewrite `build_synthetic_bank_from_dataset` to emit the `simulations/` group + collapsed `traces/`
  (`signal`, `simulation_id`, `pair_index`, `label` as int, `snr_db`, `noise_record`,
  `noise_channel`). Drop `StimEdgeEnum` and the `fibrosis_density*` / `electrode_row` /
  `electrode_height_mm` / `stim_edge` trace columns.
  If CL-060 lands, the collapsed `traces/` also carries a nullable **`activation_position`** column,
  written absent here (Wave 1 has no controlled crop) and populated at SEP2.4.
- **Verify:** `tests/test_builders.py` — a two-sim `DatasetResult` produces two `simulations/`
  entries with the right FK join, and per-trace labels match `DatasetResult.labels` exactly.
- **Depends on:** SEP12.2; **egm-data v0.5.x tagged**.

#### SEP12.3b — Cross-check the two banks agree ☐ (0.5–1 h)
- **Change:** a test asserting that the ClassifierBank and the `synthetic_bank` emitted from **one
  run** agree per trace on `label` and `simulation_id`. The producer builds the two independently
  (D7), so nothing structural forces them to match; this is the guard that replaces the
  by-construction guarantee deriving one from the other would have given.
- **Verify:** the test fails if either builder's ordering or labelling drifts.
- **Depends on:** SEP12.3.

#### SEP12.4 — Noise-mixed path onto v2.0 (D3) ✅ (2–3 h)
- **Change:** inline-mix path builds the v2.0 bank from the in-memory `DatasetResult` + mixed
  signals; `build_synthetic_bank_from_classifier` is retired or reduced to the ClassifierBank-only
  case; standalone `synthegm-mix` raises a clear `ConfigError` if asked for a synthetic bank.
- **Verify:** inline noise-mixed run emits a v2.0 bank whose `snr_db` / `noise_record` /
  `noise_channel` columns are populated and whose `simulations/` config matches the clean run;
  standalone-mix error path asserted in `tests/test_cli_config.py`.
- **Depends on:** SEP12.3.

#### SEP12.5 — Always emit both banks; retire the flag (D4) ☐ (1–2 h)
- **Change:** remove `output.also_emit_synthetic_bank` from `cli/_config.py` (both the
  generate-dataset and mix config builders); every synthetic run writes the ClassifierBank **and**
  the `synthetic_bank`. `output.synthetic_bank` becomes optional, deriving as a sibling of
  `output.classifier_bank`. A config still carrying the retired key raises a `ConfigError` naming the
  change. Update the seven `examples/*.yaml` that set it.
- **Verify:** `tests/test_cli_config.py` covers the retired-key error + the derived path; a plain
  generation run produces both banks with matching `simulation_id` sets. Daniel's four local
  `configs/phase_1_5_*.yaml` also carry the key — flag them, they're untracked and his to edit.
- **Depends on:** SEP12.4.

#### SEP12.6 — Trivial θ-spec + bank-root fields ☐ (1–2 h)
- **Change:** write `generation_params_json` as `{regime: {…type discriminators…}, knobs: []}` — the
  regime populated from the run's fixed type discriminators, the knob list empty until SEP11.
- **Verify:** a generated bank's θ-spec validates against the `TunedParam` schema with zero knobs;
  regime matches the config's declared types.
- **Depends on:** SEP12.3.

#### SEP12.7 — Wave-1 equivalence check ☐ (1–2 h)
- **Change:** no source change — a regeneration + comparison. Run an existing example config on the
  pre-migration tag and on `development`, and diff traces + labels.
- **Verify:** identical trace arrays (bitwise, same seed) and identical label vectors; full suite
  green; `grep -rn "import finitewave\|from finitewave" src/` still matches only
  `backends/finitewave/`. This is the design §7 Wave-1 gate for this repo.
- **Depends on:** SEP12.5, SEP12.6.

#### SEP12.8 — Docs for the restructure ☐ (1–2 h)
- **Change:** `docs/usage.md` (the new bank layout, what moved, and the retired
  `also_emit_synthetic_bank` — it appears in the config table twice and in two config samples) and
  `docs/simulation_theory.md` (one stale `also_emit_synthetic_bank: true` reference);
  `project/architecture.md` gets the per-sim config section + the D1/D3 notes (the D4 rewrite already
  landed 2026-07-28); `CHANGELOG.md` Unreleased entry.
- **Verify:** `grep -rn also_emit docs/ examples/ src/` returns nothing; the pre-PR run in
  `intracardiac-platform/project/pr_checklist.md` passes.
- **Depends on:** SEP12.7.

### Wave 2 — features

Once SEP12 lands, the four independent threads below can proceed in any order; SEP6 / SEP7 / SEP11
are the ones that build on the migrated schema.

#### SEP5.1 — `CellModelSpec` Protocol + `AlievPanfilov` concrete (D2) ☐ (2–4 h)
- **Change:** fifth spec Protocol in `simulate/specs.py` + the `AlievPanfilov` concrete carrying
  `ap_time_unit_ms`; `SimulationBackend.simulate()` gains `cell_model`; `FinitewaveBackend` and the
  test `MockBackend` adopt it. Pure refactor — Aliev–Panfilov behavior unchanged. Record the
  Protocol extension + its Guardrail-3 rationale in architecture.md.
- **Verify:** full suite green with no numeric change to any existing test fixture; `RunConfig` no
  longer carries `ap_time_unit_ms`.
- **Depends on:** SEP12.7.

#### SEP5.2 — `Courtemanche` concrete + backend dispatch ☐ (3–6 h)
- **Change:** `Courtemanche` spec (curated conductance scalings as a params dict, so SEP11 can
  address them by `path`) + `_build_model_2d` dispatch in `backends/finitewave/backend.py`.
  **Availability confirmed 2026-07-28** (see the decisions log): finitewave 0.9.3 ships
  `fw.Courtemanche` with a `fw.Courtemanche2D` back-compat alias, so the dispatch is the same idiom
  as today's `fw.AlievPanfilov2D()`. Conductances (`gna`, `gk1`, `gto`, `gkr`, `gks`, `gcal`, …) are
  plain instance attributes read at kernel-run time, so scalings are set by assignment — no patching
  of the model, and SEP11 can address them by `path`.
- **Verify:** a short Courtemanche sim produces a physiologically plausible AP upstroke + plateau
  (assert upstroke velocity and APD90 fall in published human-atrial ranges, not just "runs").
- **Depends on:** SEP5.1.

#### SEP5.3 — Anisotropy + time-base for an ionic model ☐ (2–4 h)
- **Change:** `_configure_anisotropy_2d_courtemanche` sibling; Courtemanche runs in real ms so the
  AP time-unit translation is bypassed — `trace_duration_ms` maps directly. Note the two models also
  ship different diffusion defaults (`D_model` = 1.0 for Aliev–Panfilov, 0.154 for Courtemanche), so
  the anisotropy helper cannot reuse AP's scaling constants.
- **Verify:** realized conduction-velocity ratio along/across fibers matches
  `geometry.anisotropy_ratio` within tolerance, measured from the activation map.
- **Depends on:** SEP5.2.

#### SEP5.4 — Runtime characterization ☐ (1–3 h)
- **Change:** no feature change — measure and document wall-clock per sim for Courtemanche vs
  Aliev–Panfilov at the v1 geometry (40 mm, dr 0.25 mm → 160×160), and record it in
  `docs/simulation_theory.md`. An ionic model is far more expensive than AP; if a 1000-sim bank
  becomes infeasible on the laptop, that is a §8 data-plan input the project-lead needs.
- **Verify:** timing table in the docs; a note raised to the project-lead if the projected bank
  generation time is impractical. Pairs with B12.
- **Depends on:** SEP5.3.

#### SEP5.5 — Courtemanche docs + config ☐ (2–3 h)
- **Change:** `cell_model:` config block + dispatch in `cli/_config.py`; an example config;
  `docs/simulation_theory.md` gains the Courtemanche section (ionic formulation, what the curated
  scalings mean, why the time base differs); CHANGELOG.
- **Verify:** example config runs end-to-end; pr_checklist passes.
- **Depends on:** SEP5.4.

#### SEP1.1 — `PostProcessStage` protocol + `BandpassStage` ☐ (2–3 h)
- **Change:** new `postprocess/` subpackage — a small stage Protocol + `BandpassStage` wrapping
  egm-signal's `bandpass`. Pure functions on trace arrays; no bank awareness.
- **Verify:** unit tests showing the stage reproduces the mixer's current `bandpass_clean` output
  bit-for-bit on the same input.
- **Depends on:** SEP12.7.

#### SEP1.2 — Wire the stage into the producer + double-filter guard ☐ (2–3 h)
- **Change:** `postprocess:` config block applied after the dataset run; **guard** — enabling a
  band-pass post-process stage while `mix.bandpass_clean` is true raises a `ConfigError` naming both
  fields, rather than silently filtering twice.
- **Verify:** guard test asserts the error; a raw / band-passed-clean / noise-mixed triple generated
  from one config set, with the band-passed-clean bank distinguishable from the raw one.
- **Depends on:** SEP1.1.

#### SEP1.3 — Standalone post-process entry point ☐ (1–2 h)
- **Change:** make the stage independently runnable per the §3 sanity-pass resolution — a
  `synthegm-postprocess` CLI taking a ClassifierBank in and writing one out.
- **Verify:** `--help` renders; a round-trip on a small fixture bank.
- **Depends on:** SEP1.2.

#### SEP1.4 — Mixer band-pass deprecation + docs ☐ (1 h)
- **Change:** `docs/mixer_theory.md` + `docs/usage.md` explain that band-passing is now a producer
  stage and `mix.bandpass_clean` is the legacy in-mixer path; CHANGELOG.
- **Verify:** pr_checklist passes.
- **Depends on:** SEP1.3.

#### SEP8.1 — `NoiseSelectionStrategy` Protocol + uniform default ☐ (2–3 h)
- **Change:** fifth mixer-side Protocol (parallel to the simulation strategies) with
  `UniformRandomNoiseSelection` wrapping today's `sample_noise_for_length`. Behavior-preserving.
- **Verify:** existing `tests/test_mixer.py` passes unchanged with the same master seed — the
  refactor must not perturb the RNG stream.
- **Depends on:** SEP12.7.

#### SEP8.2 — Per-sim patient + record selection ☐ (2–3 h)
- **Change:** `PerSimPatientNoiseSelection` (each simulation draws one IAFDB patient; the
  highest-leverage variant per roadmap) and `PerSimRecordNoiseSelection`. Both need the trace's
  `sim_id` and a grouping over the noise bank's `source_record`. **Parse confirmed 2026-07-28**
  against `banks/iafdb_noise_v1.h5`: records are `<patient>_<placement>`, so `split("_")[0]` gives a
  clean 8-way partition (iaf1–iaf8). The `noise_bank` schema is slim (`source_record` +
  `source_channel` only, no `patient_id`), so parsing is the only route — put the parse in one
  helper with a clear error if a record ever fails the convention, rather than inlining `split` at
  three call sites.
- **Sampling decision inside this step:** segment counts per patient are very uneven (iaf1 3,227 →
  iaf4 15,548, ~4.8×). Drawing a patient uniformly and drawing a segment uniformly give materially
  different noise distributions. Pick one deliberately, document which, and record it in
  `docs/mixer_theory.md`.
- **Verify:** a mixed bank where all traces of one sim share a patient / record, asserted directly
  from the per-trace `noise_record` audit field.
- **Depends on:** SEP8.1.

#### SEP8.3 — Config dispatch ☐ (1–2 h)
- **Change:** `noise_selection.type` block inside `mix:` + strategy dispatch by name in
  `cli/_config.py`, following the existing dispatch idiom.
- **Verify:** `tests/test_cli_config.py` covers each type + an unknown-type error.
- **Depends on:** SEP8.2.

#### SEP8.4 — Noise-selection docs ☐ (1 h)
- **Change:** `docs/mixer_theory.md` — why uniform sampling destroys within-patient noise
  correlation and what each strategy preserves; CHANGELOG.
- **Verify:** pr_checklist passes.
- **Depends on:** SEP8.3.

#### SEP3.1 — Absolute noise floor ☐ (2–4 h)
- **Change:** an absolute floor term in `MixerConfig` so noise no longer scales down without limit
  as the clean trace weakens (today `snr_scale` is purely relative — confirmed in the §3 sanity
  pass). Effective scale becomes the larger of the relative-SNR scale and the floor scale. **Write
  the formulation into `docs/mixer_theory.md` first** — this is math, and the doc is where it gets
  reviewed. Full height-coupled SNR stays deferred (SR1).
- **Verify:** a weak-signal trace receives the floor while a strong-signal trace keeps its relative
  SNR; the realized per-trace SNR audit field reflects which regime applied.
- **Depends on:** SEP12.7.

#### SEP3.2 — Floor config + provenance ☐ (1–2 h)
- **Change:** `mix.noise_floor` config + the floor recorded in the mixer provenance entry.
- **Verify:** config tests; provenance round-trip; CHANGELOG.
- **Depends on:** SEP3.1.

#### SEP2.1 — Activation-position policy ☐ (1–2 h)
- **Change:** a position spec in `simulate/` — `𝒫` as a `[0,1]` fraction range with
  `idx = round(frac·(T−1))` at point of use. Per the A4 resolution a fixed position is the range
  collapsed to a point, mirroring the mixer's `snr_db_range=(X,X)` idiom.
- **Verify:** unit tests on the fraction→index conversion at both endpoints and on the collapsed
  range.
- **Depends on:** SEP12.7.

#### SEP2.2 — Size the simulation to the widest `p` ☐ (2–3 h)
- **Change:** derive the required capture duration from `T` (**192 ms**, CL-024 §4) and the widest
  sampled `p` so the crop never runs off the back of the simulation — extend for back-overhang, flat
  front. **Delete the zero-pad fallback** in `run_single` step 3 and replace it with a hard error:
  under the new sizing, a short capture is a bug, and silently padding zeros into a trace is exactly
  the kind of artifact the classifier could key on. Guard the 64-sample-grid constraint here too — a
  `T` that isn't a multiple of 64 samples breaks CLF3 downstream, and this is the cheapest place to
  catch it.
- **Verify:** a config at the widest `p` produces full-length traces with no padding; an artificially
  short backend capture raises rather than pads.
- **Depends on:** SEP2.1.

#### SEP2.3 — Detect + crop per trace ☐ (2–4 h)
- **Change:** use egm-signal's SIG1 `extraction.activation_based` primitives to locate each bipolar
  trace's activation, then crop to length `T` with the activation at the sampled `p`. Note this is
  **per trace, not per simulation** — the wave sweeps across the grid, so pairs activate at different
  times; a single per-sim crop offset would leave the position uncontrolled for most pairs.
- **Verify:** on a synthetic fixture, the detected activation lands within tolerance of the requested
  fraction for every pair; the shared-`p` case gives every trace the same index.
- **Depends on:** SEP2.2; **SIG1 shipped in egm-signal**.

#### SEP2.4 — Config + realized-position provenance ☐ (1–2 h)
- **Change:** `activation_position:` config block; populate the per-trace **`activation_position`**
  column (float `[0,1]`, same name + convention as `iafdb_bank`'s) with the *realized* fraction, so
  STU5 compares stored-vs-stored and the §8.9 A/B is checkable. The column itself ships nullable in
  Wave 1 (SEP12.3); this fills it. **Field approved (CL-062)** — egm-contracts defines it once in
  `common` and both banks `$ref` it (CL-064); egm-data carries it on the typed reader and keeps it
  *off* the ClassifierBank via the converter, with a negative test (CL-063). No longer conditional.
- **Verify:** config tests; the realized fractions of a range run span `𝒫`; a Wave-1 bank still reads
  with the column absent.
- **Depends on:** SEP2.3 · the CL-060 field decision.

#### SEP2.5 — Cropping docs ☐ (1 h)
- **Change:** `docs/simulation_theory.md` — the sizing derivation and why the front is flat and the
  back overhangs; cross-reference `activation_splitting_method.md` for the shared math; CHANGELOG.
- **Verify:** pr_checklist passes.
- **Depends on:** SEP2.4.

#### SEP13.1 — Probe-mode generator: one sim, `p` swept on a grid ☐ (2–4 h)
- **Change:** a probe generation mode — run **one** simulation, then emit one trace per grid point by
  cropping that same simulation output at each offset, morphology and seed held constant. Reuses
  SEP2.2's sizing rule (the sim must reach the widest grid point) and SEP12's writer; **no schema
  change** — the grid value lands in the existing `activation_position` column.
- **Crop exactly, don't re-detect (see D6):** detect the activation **once** on the source trace, then
  place it by exact integer shift per grid point, so the realized position equals the requested one.
- **Verify:** a probe bank's `activation_position` column reproduces the requested grid exactly (not
  approximately); every trace in one sweep has identical `simulation_id` + `seed`; the signal at two
  grid points is the same waveform at different offsets (assert by cross-correlation peak, not eyeball).
- **Depends on:** SEP2.4 · SEP10.1.

#### SEP13.2 — Probe config + docs ☐ (1–2 h)
- **Change:** a `probe:` config block (grid bounds, n points, which pairs) + CLI wiring; `docs/usage.md`
  gains a probe section explaining what the bank is *for* (it is not training data); CHANGELOG.
- **Verify:** example probe config runs end-to-end and yields a bank STU8 can read; config tests cover
  a grid that would exceed the sim's sizing (must error, not silently clip).
- **Depends on:** SEP13.1.

#### SEP10.1 — Anchoring opt-in flag ☐ (1–2 h)
- **Change:** the fixed-vs-range flag on SEP2's position config. Note-level per design §3 — the
  anchored-vs-varied comparison is study §8.9 and needs **no egm-classifier change**; both arms are
  just two training banks.
- **Verify:** two configs (anchored / varied) produce banks whose realized-position distributions are
  a point mass and a spread respectively.
- **Depends on:** SEP2.4.

#### SEP6.1 — Multi-edge `planar_edge` ☐ (2–4 h)
- **Change:** `PlanarEdgeStimulus.edge` → `edges: tuple[Edge, ...]` (the schema's `planar_edge` is
  already plural); backend installs one strip per edge. Sánchez stimulates three sides — the point is
  propagation-direction variety, so the dataset sampler needs to sample edge *sets*, not one edge.
- **Verify:** a two-edge sim shows wavefronts entering from both edges in the activation map; the
  single-edge case is unchanged bit-for-bit.
- **Depends on:** SEP12.7.

#### SEP6.2 — Multi-edge config + docs ☐ (1–2 h)
- **Change:** `activation.edges` / edge-count sampling in the config; docs + CHANGELOG.
- **Verify:** config tests; example config runs.
- **Depends on:** SEP6.1.

#### SEP7.1 — `PointStimulus` ☐ (2–4 h)
- **Change:** spec + `_build_point_stimulus_2d` adapter. Focal source for the spiral-wave studies.
- **Verify:** activation map shows radial spread from the configured position.
- **Depends on:** SEP12.7.

#### SEP7.2 — `S1S2Protocol` ☐ (2–5 h)
- **Change:** spec + adapter installing two timed stimuli. **`T` interaction:** S1–S2 puts two
  activations inside the capture window, which collides with SEP2's single-activation crop — the
  crop must either target the S2 activation explicitly or the sim must capture long enough to crop
  around it. Resolve when both are in; note it here so it isn't discovered late.
- **Verify:** both activations appear at the configured intervals; the S2 wave shows the expected
  conduction slowing at short coupling intervals.
- **Depends on:** SEP7.1.

#### SEP7.3 — Activation-variant config + docs ☐ (2–3 h)
- **Change:** `activation.type` dispatch extended to `point` / `s1s2`; docs + CHANGELOG.
- **Verify:** config tests per variant + unknown-type error; example configs run.
- **Depends on:** SEP7.2.

> **SEP11 is one issue, not two** (project-lead, 2026-07-28). It is the θ-sweep **harness** + the
> `generation_params` writer + a **pluggable sampler** — θ-spec-agnostic, depending only on SEP12.
> The same capability is *run twice* as §8 data-generation: an OAT config produces the screening
> banks, egm-studio's **STU7** screening then sets the θ-spec membership, and a calibrated config
> drives the design sweep for STU4. That ordering is **§8.2 run-ordering — data flow, not a code
> dependency**; nothing here waits on STU7 to compile. The promotion test is "distinct code
> capability?", and OAT vs LHS/grid is a swapped sampler over a shared harness. Steps are ordered so
> **the harness + OAT sampler land first and unblock STU7**, with the LHS/grid sampler after.
>
> **Escape hatch:** if the design sweep turns out to need substantial harness code beyond a sampler
> swap — the GP-emulator design-bank layout, per-cell N-tagging, real LHS infrastructure — that is
> the signal to flag it and split into SEP11a/SEP11b. Default stays single.

#### SEP11.1 — `TunedParam` path resolution ☐ (2–4 h)
- **Change:** a resolver that reads and writes a config value by its θ-spec `path`
  (`substrate.density`, `cell_model.courtemanche.params.gcal`, `backend.diffusion`, `mixer.snr_db`)
  against the spec objects. This is what makes θ membership a per-sweep choice with zero contract
  churn — and what keeps the harness θ-spec-agnostic.
- **Verify:** get/set round-trip for every 1.5-reachable path; a clear error on an unknown path.
- **Depends on:** SEP12.7 (+ SEP5.2 for the cell-model paths).

#### SEP11.2 — Sweep harness + pluggable sampler + OAT sampler ☐ (3–5 h)
- **Change:** the harness — a `Sampler` Protocol producing a design matrix over the knob list, and
  `generate_dataset` driven from that matrix instead of today's independent per-axis sampling
  (density uniform, edge uniform, height uniform). Ship `OATSampler` (per knob, a few levels with
  everything else at nominal) as the first concrete. The no-sweep path must still reproduce today's
  sampling exactly.
- **Verify:** an OAT design varies exactly one knob per cell with the rest pinned at `nominal`; the
  no-sweep path is bit-identical to pre-SEP11 output at the same master seed.
- **Depends on:** SEP11.1.

#### SEP11.3 — `generation_params` writer ☐ (2–4 h)
- **Change:** populate the bank-root `{regime, knobs:[TunedParam]}` stubbed at SEP12.5 — regime from
  the fixed type discriminators, knobs from the sweep definition (bounds, transform, role, nominal).
  The OAT banks need this too: STU7 has to know which knob moved in which cell.
- **Verify:** a swept bank's θ-spec validates; each knob's per-sim values recovered via its `path`
  match the design matrix.
- **Depends on:** SEP11.2.

#### SEP11.4 — OAT config + docs — **unblocks STU7** ☐ (2–3 h)
- **Change:** `sweep:` config block + an OAT example config; `docs/usage.md` gains a sweep section;
  `docs/simulation_theory.md` gets the θ / regime framing. This is the step the §8.2 screening banks
  are generated from, so getting the example right matters more than usual.
- **Verify:** the OAT example config runs end-to-end and produces a bank whose θ-spec + per-sim
  values are readable by an egm-studio consumer. **Hand off to STU7 here.**
- **Depends on:** SEP11.3.

#### SEP11.5 — LHS + grid samplers for the design sweep ☐ (3–5 h)
- **Change:** `LatinHypercubeSampler` + `GridSampler` concretes behind the same Protocol, M cells × N
  traces per cell, for the calibrated design run that feeds STU4. Membership of the θ-spec comes from
  STU7's screening result, so this is a *config* input, not new harness code — **if it turns out
  otherwise, invoke the escape hatch above rather than growing this step.**
- **Verify:** a small LHS sweep produces the requested cell count with each knob's marginal covering
  its bounds; an LHS example config runs; pr_checklist passes.
- **Depends on:** SEP11.4.

#### B12.1 — Resource / CPU cap ☐ (2–4 h)
- **Change:** a `resources:` config block capping worker / BLAS thread counts so a long generation
  run doesn't redline the laptop. Pairs with SEP5.4 — Courtemanche runs are exactly when this bites.
- **Verify:** thread caps observably applied; a capped run completes with the same output as an
  uncapped one.
- **Depends on:** SEP12.7.

#### B13.1 — Custom bank id for the clean bank ☐ (1–2 h)
- **Change:** `output.clean_bank_id`. Today `output.bank_id` names the *primary* output — which in a
  noise-mix run is the noise-mixed bank — and the clean intermediate only ever gets a derived id.
- **Verify:** `tests/test_bank_ids.py` covers the override + its validation; a noise-mix run with
  both ids set stamps each bank correctly.
- **Depends on:** SEP12.7.

### Phase-exit

#### X.1 — Docs + phase exit ☐ (2–3 h)
- **Change:** trim `roadmap.md` of everything shipped (the Phase 1.5 cluster, the polymorphic
  `stimulation` entry, and the `RunConfig.cell_model` open question closed by D2); finalize
  `CHANGELOG.md`; make sure `project/architecture.md` reflects the fifth spec, the `SimulationResult`
  widening, and the post-process stage.
- **Verify:** full pre-PR run in `intracardiac-platform/project/pr_checklist.md`; the release
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
| SEP12 | schema-migration | 5 | 12–24 h | | | |
| SEP5 | pipeline (physics) | 5 | 10–20 h | | | |
| SEP1 | pipeline (refactor) | 3 | 6–9 h | | | |
| SEP8 | pipeline | 3 | 6–9 h | | | |
| SEP3 | pipeline (math) | 2 | 3–6 h | | | |
| SEP2 | pipeline (algorithm) | 3 | 7–12 h | | | |
| SEP10 | pipeline | 1 | 1–2 h | | | |
| SEP13 | pipeline | 2 | 3–6 h | | | |
| SEP6 | pipeline | 2 | 3–6 h | | | |
| SEP7 | pipeline | 3 | 6–12 h | | | |
| SEP11 | pipeline | 5 | 12–21 h | | | |
| B12 | pipeline | 2 | 2–4 h | | | |
| B13 | pipeline | 1 | 1–2 h | | | |
| *(phase exit)* | docs | — | 2–3 h | | | |
| **Repo total** | | **37** | **74–136 h** | | | |

**Estimate basis.** The ledger is empty, so these are reference-class-by-analogy, not
`points × measured rate`. Anchors used: SEP12 is *the* rubric's L anchor; SEP5 is sized equal to it
(comparable surface, higher physics novelty, lower coordination); SEP11 equal again (rewrites the
sampling loop and gates STU4). The XS/S items are anchored on the `noise_bank` `bank_id` add (S).
Ranges are deliberately wide — roughly 2× low-to-high — because there is nothing yet to narrow them
with. First cleanup replaces all of this with measured rates.

## Notes / decisions log

- 2026-07-28 — Plan drafted from design §3 (ten SEP issues) + §4 (B12, B13). Five design notes
  recorded: D1 (`SimulationResult` widening), D2 (cell model as a fifth spec), D3 (noise-mixed
  reconstruction route), D4 (**escalation** — is `synthetic_bank` still opt-in, given STU1/4/5 read θ
  from it?), D5 (`T` and `𝒫` come from study §8.1).
- 2026-07-28 — Two verification-worthy unknowns flagged inside steps rather than assumed away:
  whether the pinned Finitewave ships a Courtemanche 2D model (SEP5.2), and whether an IAFDB patient
  is parseable from the noise bank's `source_record` string (SEP8.2). **Both investigated the same
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
  bare names as part of SEP5.1, since that step already touches the backend's model construction.
- 2026-08-01 — **SEP12.2 + .3 + .4 landed together — the split was artificial** (Daniel's review).
  SEP12.1 was scoped to avoid the tag so it could start early, but the whole point of the step is the
  migration, and re-pinning proves the split can't hold: `builders.py` imports `StimEdgeEnum`, which
  v0.6.0 removes, so the repo does not *import* between the re-pin and the builder rewrite. .4 came
  along because the two CLIs call the retired reconstruction builder. Lesson recorded for the phase
  retro: **a migration step is not divisible below "the repo imports again"** — wave planning should
  size steps by that boundary, not by what can be made to look independent.
  Landed: re-pin contracts+data **v0.6.0**; new `simulate/bank_config.py` (spec → typed per-function
  models); `build_synthetic_bank_from_dataset` rewritten to 2.0 (per-sim `simulations/` group,
  collapsed `traces/`, int label, θ-spec regime with empty knobs — the schema **requires**
  `generation_params`, so SEP12.6's trivial writer folded in too);
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
  FB-15/FB-16. Added **SEP12.3b** — cross-check that the two banks agree on label + `simulation_id`
  per trace — to cover the drift risk two builders reintroduce.
- 2026-08-01 — **SEP12.1 done.** `SimulationSpecs` added to `simulate/result.py` and populated in
  `run_single`; `sim_id` -> `simulation_id` fleet-wide in this repo; architecture.md records the
  Guardrail-2 widening. 125 tests green, ruff + bare `mypy` clean. Also landed **CL-100** here
  (`[tool.mypy] files = ["src","tests"]`) since the step already touched tests — it surfaced exactly
  the sloppiness CL-100 predicted: a `parametrize` widened `Edge` to `str`, and a test read
  `size_mm` through the `GeometrySpec` Protocol, which by design exposes only `type`. Both fixed
  properly (narrow with `isinstance`, parametrize over `EDGES`) rather than ignored.
  **Sandbox note:** the local siblings are already v0.6.0, so verifying SEP12.1 against the *current*
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
- 2026-07-30 — **`activation_position` field approved (CL-062)** — SEP2.4 is no longer conditional.
  egm-contracts defines it once in `common` with both banks `$ref`ing it (CL-064); egm-data carries
  it on the typed reader and keeps it off the ClassifierBank with a negative test (CL-063).
- 2026-07-30 — **SEP2 unchanged by the CL-065->CL-068 exchange.** `P_synth` superset of `P_iafdb` was
  questioned and re-affirmed: `activation_position` is an augmentation axis, not a realism axis, so it
  is excluded from the STU4 objective and STU5 distance rather than the ranges being matched
  ("augmentation axes are not realism axes"). Recorded in D5 so the inequality isn't later mistaken
  for an inconsistency. Also noted there: CL-072's §8.1 go/no-go could still widen `T` 192 -> 256, so
  SEP2.2 must read `T` from config rather than hardcode it.
- 2026-07-29 — **CL-059 answered (CL-060): the crop `p` has nowhere to be persisted.** Design §5
  assumed it "rides the per-sim config", which fails twice — v2.0's collapsed `traces/` has no
  position column, and the *realized* position is inherently per-trace (the wave sweeps the grid, so
  each pair activates at a different time; detection jitter + `round(frac·(T−1))` add more). Asked
  the project-lead for a nullable per-trace `activation_position` on CON1, mirroring the
  `iafdb_bank` field IAF3 just added — written absent in Wave 1 (SEP12.3), populated Wave 2
  (SEP2.4). Negligible cost, no §6 change. **SEP2.4 is conditional on that decision.**
- 2026-07-29 — **CI ruff pinned** to `ruff==0.15.17` in `.github/workflows/ci.yml` per CL-041 /
  CL-024 §1 (fleet-wide pin, option (a) at a single version). No 0.16 reformat run — the pin makes it
  moot, and 0.15.17 leaves Markdown alone, which is what protects the shape-annotation alignment I
  argued for in CL-007. Verified: `ruff check .` passes, `ruff format --check .` reports 31 files
  already formatted. `.pre-commit-config.yaml` was already `v0.15.17`, so no pre-commit change here.
- 2026-07-29 — **CL-024 §3 + §4 folded in.** Join key: my CL-008 resolved as option 1 — producer
  renames `sim_id` → `simulation_id`, landed in **SEP12.1** (earliest step, no schema dependency, so
  egm-studio's fixture flip and egm-data's writer check aren't held behind Wave 1). `T` = **192 ms**
  at 1 kHz on a 64-sample grid, so D5's `T` half is closed, configs move 200.0 → 192.0, and SEP2.2
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
  default it true, with a `ConfigError` on the legacy key. New step **SEP12.5**; SEP12 estimate
  11–21 h → **12–23 h**, repo total **71–129 h**. `project/architecture.md`'s "Why not SyntheticBank
  by default" rewritten to "Two outputs, different purposes, joined by `simulation_id`" — the doc
  marks it as Phase-1.5/v0.4.0 design, since the flag doesn't actually retire until SEP12 lands.
  `docs/` still describes v0.3.0 behavior correctly and is deliberately **not** pre-updated; it moves
  at SEP12.8.
- 2026-07-28 — **SEP11 stays one issue** (project-lead). Screening circularity resolved by promoting
  egm-studio's screening to its own issue (**STU7**); SEP11 is one capability run twice, not
  SEP11a/SEP11b. Promotion test = "distinct code capability?" — OAT vs LHS/grid is a swapped sampler
  over a shared harness (one capability); screening (`screening.py`) is genuinely different code (its
  own issue). Steps re-ordered so the harness + OAT sampler land first (SEP11.2–.4, handing off to
  STU7) and the LHS/grid sampler follows (SEP11.5); the run-ordering between them is §8.2 data flow,
  **not** a code dependency. Escape hatch recorded on the issue. Estimate 10–20 h → **12–21 h**
  (the sampler Protocol boundary + the OAT/LHS split are real added structure); Cx unchanged at L.
- 2026-07-28 — **IAFDB patient parse: confirmed.** `banks/iafdb_noise_v1.h5` (schema 1.0, 66,355
  traces × 200 samples @ 1 kHz) has 31 distinct `source_record` values, all
  `<patient>_<placement>` — iaf1–iaf8 × {svc, ivc, tva, afw}, minus `iaf6_ivc` which is absent.
  Channels are CS12/34/56/78/90. Matches
  `intracardiac-platform/references/iafdb_dataset_primer.md` §2.
