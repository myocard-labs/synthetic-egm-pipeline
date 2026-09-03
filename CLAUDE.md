# synthetic-egm-pipeline — working agreements

Generates synthetic intracardiac EGMs (Finitewave / Aliev–Panfilov) for a CNN that
flags fibrotic ablation targets in AF. The deliverable is a **white paper**, so
reproducibility and honest provenance outrank convenience everywhere below.

## Verify before claiming

**While iterating, run the test modules covering what you changed.**

```
pytest -q tests/test_<module>.py && ruff check . && mypy
```

Test modules are named after the modules they cover — `probe.py` → `test_probe.py` — so the
selection is mechanical, not a judgement call. When a change touches a shared type
(`SimulationResult`, a spec Protocol, a card), the blast radius is wider than the filename
suggests: name the modules you ran and why.

**Before handing work back, run the whole fast gate.**

```
pytest -q && ruff check . && ruff format --check . && mypy
```

**Why this is now two tiers rather than one.** The fast suite is 361 cases across 19 files and is
heavy enough to redline a laptop, which made running it on every edit a real drag on the work —
`pyproject.toml` still describes it as finishing "in ~1 s", which was true once and stopped being
true without anyone noticing. Targeted selection is the stopgap. **It is not a licence to skip the
full gate**, which is the only thing that catches cross-module breakage, and which no amount of
per-module confidence substitutes for.

**The slow suite is a different decision, because it costs 30+ minutes.**

```
pytest -q -m slow
```

It runs the real solver, so it is the only thing that can catch a physics
regression — but running it on every change is a measurable drag on the work and
most changes cannot possibly affect it. **Run it when the change can move a
number, and say which case applies when reporting:**

| run the slow suite | skip it |
|---|---|
| solver, backend, or cell-model code | docs, comments, error-message text |
| calibration, model cards, anchors | signature refactors with no behaviour change |
| the signal path — filtering, resampling, cropping | config parsing and validation |
| anything whose diff changes a recorded value | example configs, test names |

**When in doubt, run it** — but say so rather than running it reflexively. The
step brief will normally state which is expected; if it does not, decide and
state the reason.

**A green fast suite is not a green suite.** That is why the table exists rather
than a blanket permission to skip.
- `ruff format`, not just `ruff check`. A pre-commit hook otherwise reformats and
  fails the commit.
- Generation runs: `synthegm-generate-dataset configs/<name>.yaml --overwrite`.

**Measure, don't restate.** Before designing around any inherited claim about
behaviour or performance, measure it. This project has twice built work on a
number that turned out to be measured through a broken layer — a conduction
velocity read off the wrong axis, and an activation index that was a detection
artifact. A quantity measured through an unverified layer is evidence about the
layer, not the quantity.

**Never filter verification output to where you expect errors.** Read all of it.

## Tests must fail for the reason they claim

Three tests in this repo have passed or failed for reasons unrelated to their
stated purpose. Each time the cause was a **proxy that stopped tracking the thing
it stood for**:

- an anisotropy test that requested ratio 9 and passed on a stencil default,
  because the knob was a silent no-op;
- a "wave clears the mesh sooner" proxy that saturated once APD lengthened;
- a bank-comparison script that reported "all checks passed" against an **empty
  directory**, because a skip counted as a pass.

When writing a test, ask what would happen if its own input were ignored. If it
would still pass, it is measuring something else.

## Architecture

Five strategy specs — geometry, substrate, activation, electrodes, **cell model**
— are pure data with a `type` discriminator the backend dispatches on. Declare
`type` as a read-only `@property` in the Protocol; the bare `type: str` form
demands a settable attribute and frozen dataclasses then fail to satisfy it.

**A parameter belongs to the narrowest thing that can change it independently:**

| Axis | Home | Examples |
|---|---|---|
| Physiology (survives every swap) | `simulate/calibration.py` targets | `conduction_velocity_cm_s`, `apd90_ms` |
| Tissue structure | `GeometrySpec` | `anisotropy_ratio`, `fiber_angle_rad`, `dr_mm` |
| Cell model (AP → Courtemanche) | `simulate/cell_models.py` | `eps`, `time_unit_ms`, `diffusion`, `dt` |
| Backend / scheme | `RunConfig` | `dr_model_units`, `capture_oversample` |

The test is *does this survive a cell-model swap?* `apd90_ms` does; `eps` does not
(Courtemanche has none), and `time_unit_ms` does not (only a dimensionless model
needs a mapping to ms).

**Guardrail 1:** `backends/` is the only place that may import a solver library.
**Guardrail 3:** if the runner needs something the backend Protocol lacks, extend
the Protocol rather than smuggling it through `RunConfig`.

**No hardcoded generation parameters.** Anything a generated bank depends on lives
in config or a model card. Membrane knobs come from a named card
(`src/.../models/*.yaml`), verified against a fresh solve on every load.

## Working style

- **One step at a time, one repo at a time.** Stop after each step for review.
  Review attention is the bottleneck, not typing.
- **Do not hand over git commands until asked.** Review first.
- Commits: `[Type] Subject` + `*` bullets. Separate commits for feature / docs /
  tests. **No backticks, `$`, backslash or `!` in the message** — it is passed as
  a double-quoted shell string and backticks open command substitution.
- Daniel opens and merges PRs in the GitHub web UI. Supply plain `git` commands,
  no `gh` CLI, one command per code block, no `cd` prefix.
- `GIT_OPTIONAL_LOCKS=0` on any git read; never `stash`/`add` from an agent.
- Every repo has `project/` (internal) and `docs/` (external). `/configs/` is
  gitignored scratch; shared configs live in `examples/`.

## Traps that have already cost time

- **Scripted bulk edits.** A regex stripping a keyword from one constructor ate
  the same keyword inside another. A `replace()` that silently matched nothing
  left two definitions of one constant. **Verify the result after every scripted
  edit**, not when something downstream breaks.
- **Moving a symbol between modules.** Grep *every* reference immediately,
  including imports buried inside function bodies — fixing only the one in the
  traceback cost three round-trips.
- **Example configs restate defaults.** Changing a constant in `constants.py` is
  often cosmetic because all six examples set the value explicitly. Check them.
- Finitewave's `D_al`/`D_ac` live on the **stencil**, not the model; assigning to
  the model creates a dead attribute and fails silently.

## Where the reasoning lives

`project/phase_1_5_plan.md` is the implementation plan (steps, status, design
notes D1–D9). Cross-repo decisions and their evidence live in
`../intracardiac-platform/project/investigations/` and the phase coordination log.

**Process identifiers stay in `project/`. Never in `src/` or `docs/`.**

| | may reference |
|---|---|
| `src/`, `docs/` | papers, physics, measured numbers — anything externally verifiable |
| `project/` | all of the above **plus** `CL-NNN`, `SEP-NN`, `FB-NN`, step ids (`S18c`), design notes (`D6`), phase names |

This package is going to PyPI. `CL-176` means nothing to an external reader and
never will, and once the phase is archived the number is noise even internally.
A citation stays verifiable forever; a process id rots.

**The rule is not "delete the tag" — it is "write the reason instead."** Most of
these pointers stand in *place of* the reasoning rather than beside it. A comment
reading "APD must exceed the trace duration (CL-176)" becomes an unjustified
assertion the moment the tag is stripped. Promote the content:

> must exceed the trace duration, because otherwise the repolarisation deflection
> lands inside every cropped window at a fixed offset — a marker present in all
> synthetic traces and no real ones.

Self-contained, and better than the pointer was. A regex that removes tags
without promoting the reasoning leaves the codebase worse than it found it.
