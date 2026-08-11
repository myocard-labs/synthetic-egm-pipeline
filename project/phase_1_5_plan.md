# synthetic-egm-pipeline — Phase 1.5 implementation plan

**Repo:** synthetic-egm-pipeline · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** in progress · **Progress:** 13/37 steps done — **Wave 1 complete; Wave 2 underway**
**Repo estimate:** **79.5–142 h** active (37 complexity points; cold-start ranges — the
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
| SEP13 | 2 | Positional-sensitivity probe bank — one sim, crop offset swept on a grid, morphology/seed held constant | S (2) | 3–6 h | S16 |
| SEP5 | 2 | Courtemanche 1998 human-atrial cell model alongside Aliev–Panfilov | L (5) | 11–20 h | S17–S20 |
| B12 | 2 | Resource / CPU cap on a generation run | S (2) | 2–4 h | S21 |
| SEP11 | 2 | θ-sweep harness + `generation_params` writer + pluggable sampler (OAT first, then LHS/grid) — one capability, run twice per §8.2 | L (5) | 12–20 h | S22–S25 |
| SEP1 | 2 | Band-pass out of the mixer into a general, independently-runnable post-processing stage | M (3) | 6–9 h | S26–S28 |
| SEP8 | 2 | Pluggable `NoiseSelectionStrategy` (uniform default + per-sim patient / record) | M (3) | 6–9 h | S29–S31 |
| SEP3 | 2 | Absolute noise floor in the mixer (today's SNR is purely relative) | S (2) | 3–6 h | S32 |
| SEP6 | 2 | Multi-edge `planar_edge` activation variant | S (2) | 3–6 h | S33 |
| SEP7 | 2 | `point` + `s1s2` activation variants | M (3) | 6–11 h | S34–S35 |
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

### D7 — keep both builders### D7 — keep both builders; do **not** derive the ClassifierBank via egm-data's converter *(repo-internal, confirmed with Daniel 2026-08-01)*

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
shift per grid point. Realized position equals requested position by construction, the x-axis is
exact, and it is cheaper (one detection, not N). The verification is correspondingly stricter: assert
the emitted `activation_position` column *reproduces the grid*, not that it approximates it.

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

### S13 — One ClassifierBank, one θ bank (CL-143 + B13 · D8) ☐ (2–4 h)
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

### S14 — Position config + size the simulation to the widest `p` (SEP2) ☐ (3–5 h)
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

### S15 — Crop per trace, record the realized position, anchoring flag (SEP2 + SEP10) ☐ (3–5 h)
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
- **Depends on:** S14.

### S16 — Positional-sensitivity probe bank (SEP13) ☐ (3–6 h)
- **Change:** a probe generation mode — run **one** simulation, then emit one trace per grid point by
  cropping that same simulation output at each offset, morphology and seed held constant. Reuses
  S14's sizing rule (the sim must reach the widest grid point) and SEP12's writer; **no schema
  change** — the grid value lands in the existing `activation_position` column. Ships with its
  `probe:` config block (grid bounds, n points, which pairs) + CLI wiring, and a `docs/usage.md`
  probe section explaining what the bank is *for* (it is not training data); CHANGELOG.
- **Crop exactly, don't re-detect (see D6):** detect the activation **once** on the source trace, then
  place it by exact integer shift per grid point, so the realized position equals the requested one.
- **Verify:** a probe bank's `activation_position` column reproduces the requested grid exactly (not
  approximately); every trace in one sweep has identical `simulation_id` + `seed`; the signal at two
  grid points is the same waveform at different offsets (assert by cross-correlation peak, not
  eyeball); the example probe config runs end-to-end and yields a bank STU8 can read; config tests
  cover a grid that would exceed the sim's sizing (must error, not silently clip).
- **Depends on:** S15.

### S17 — `CellModelSpec` Protocol + `AlievPanfilov` concrete (SEP5 · D2) ☐ (2–4 h)
- **Change:** fifth spec Protocol in `simulate/specs.py` + the `AlievPanfilov` concrete carrying
  `ap_time_unit_ms`; `SimulationBackend.simulate()` gains `cell_model`; `FinitewaveBackend` and the
  test `MockBackend` adopt it. Pure refactor — Aliev–Panfilov behavior unchanged. Record the
  Protocol extension + its Guardrail-3 rationale in architecture.md.
- **Verify:** full suite green with no numeric change to any existing test fixture; `RunConfig` no
  longer carries `ap_time_unit_ms`.
- **Not merged into S18 on purpose:** a behavior-preserving refactor earns its own commit precisely
  because its whole verification is *"nothing changed numerically"*. Fold the new cell model in and
  that check stops being meaningful — every fixture delta becomes ambiguous between the refactor and
  Courtemanche.
- **Depends on:** S10.

### S18 — `Courtemanche` concrete + backend dispatch + config block (SEP5) ☐ (4–7 h)
- **Change:** `Courtemanche` spec (curated conductance scalings as a params dict, so SEP11 can
  address them by `path`) + `_build_model_2d` dispatch in `backends/finitewave/backend.py`, **plus
  the `cell_model:` config block and its dispatch in `cli/_config.py`** so the model is reachable
  from a config file in the same commit that introduces it.
  **Availability confirmed 2026-07-28** (see the decisions log): finitewave 0.9.3 ships
  `fw.Courtemanche` with a `fw.Courtemanche2D` back-compat alias, so the dispatch is the same idiom
  as today's `fw.AlievPanfilov2D()`. Conductances (`gna`, `gk1`, `gto`, `gkr`, `gks`, `gcal`, …) are
  plain instance attributes read at kernel-run time, so scalings are set by assignment — no patching
  of the model, and SEP11 can address them by `path`.
- **Verify:** a short Courtemanche sim produces a physiologically plausible AP upstroke + plateau
  (assert upstroke velocity and APD90 fall in published human-atrial ranges, not just "runs");
  config tests cover the new block + an unknown-model error.
- **Depends on:** S17.

### S19 — Anisotropy + time-base for an ionic model (SEP5) ☐ (2–4 h)
- **Change:** `_configure_anisotropy_2d_courtemanche` sibling; Courtemanche runs in real ms so the
  AP time-unit translation is bypassed — `trace_duration_ms` maps directly. Note the two models also
  ship different diffusion defaults (`D_model` = 1.0 for Aliev–Panfilov, 0.154 for Courtemanche), so
  the anisotropy helper cannot reuse AP's scaling constants.
- **Verify:** realized conduction-velocity ratio along/across fibers matches
  `geometry.anisotropy_ratio` within tolerance, measured from the activation map.
- **Depends on:** S18.

### S20 — Courtemanche runtime characterization + docs (SEP5) ☐ (3–5 h)
- **Change:** measure wall-clock per sim for Courtemanche vs Aliev–Panfilov at the v1 geometry
  (40 mm, dr 0.25 mm → 160×160) and write the timing table into `docs/simulation_theory.md`,
  alongside the Courtemanche section proper (ionic formulation, what the curated scalings mean, why
  the time base differs). An ionic model is far more expensive than AP; if a 1000-sim bank becomes
  infeasible on the laptop, that is a §8 data-plan input the project-lead needs. An example config
  + CHANGELOG close the issue out.
- **Verify:** example config runs end-to-end; timing table present; a note raised to the
  project-lead if the projected bank generation time is impractical; pr_checklist passes. Pairs
  with B12 (S21).
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

### S36 — Docs + phase exit ☐ (2–4 h)
- **Change:** **rework the `examples/` config set** — it has drifted (Daniel, 2026-08-01: new configs
  added ad hoc during the phase) and is no longer a good cross-section of the runs we actually want
  to demonstrate. Decide the set deliberately — one clean baseline, one noise-mixed, one calibration,
  one sweep (SEP11), one probe (SEP13) — rather than accreting one per experiment. Also trim
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
| **Repo total** | | **37** | **79.5–142 h** | | | |

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
