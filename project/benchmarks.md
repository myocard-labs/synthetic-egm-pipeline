# Runtime benchmarks — recorded results

**What a simulation costs, measured rather than estimated.** Three open
decisions consume this number: whether to keep this solver, whether to reserve
the ionic model for studies that need it rather than routine generation, and how
large a bank the data plan can ask for. Until 2026-08-27 the figure feeding all
three was an estimate that no code in the repository could re-derive.

The harness is `tests/test_benchmarks.py`. This file is its logbook: what was
run, on what, when, and what came out.

## How to run it

```
pytest -m benchmark -s
```

`-s` is not optional — the recorded numbers are printed, and a benchmark whose
output you cannot see has measured nothing.

**These tests never run in CI, and are excluded twice over.** The default suite
selects `-m "not slow and not benchmark"` and CI's slow job selects
`-m "slow and not benchmark"`. No benchmark test carries the `slow` marker
either: double-marking is the leak path, because a test carrying both would ride
into CI on the marker it was not excluded by. A wall-clock assertion on a shared
runner fails for reasons that have nothing to do with the change under test.

## Machines

| label | machine | cores | notes |
|---|---|---|---|
| **laptop** | Surface Pro 7, Intel i5-1035G4 @ 1.10 GHz | 4 physical / 8 logical | 15 GB. Measured **under interactive load** — browser and desktop app running. Not a quiet machine. |
| **desktop** | — | — | **NOT MEASURED.** This is the machine an overnight generation run would actually use, so every projection below is a lower bound on what it can do and no substitute for running the harness there. |

Rows are labelled because a wall-clock number without its hardware is not a
measurement.

---

## Methodology — what each test actually does

Four tests in `tests/test_benchmarks.py`. Three are `benchmark`-marked; one is
not, and the reason it is not is part of the design.

### 1. Mesh factor — `test_the_mesh_factor_between_the_two_operating_points`

**Unmarked, so it runs in the fast suite.** It starts no solver and holds no
stopwatch: it reads the two shipped cards, takes their node counts from the patch
size and the pitch each was solved at, takes their integration steps from the
`solved` blocks, and multiplies. Arithmetic over committed values, in
milliseconds — there is no reason to hide a free check behind a marker, and every
reason to have it fail immediately if a card is re-solved and the ratio moves.

$$
\text{mesh factor} \;=\; \underbrace{\frac{(L/\Delta r_{\text{CRN}})^2}{(L/\Delta r_{\text{AP}})^2}}_{\text{nodes}} \times \underbrace{\frac{\Delta t_{\text{AP}}}{\Delta t_{\text{CRN}}}}_{\text{steps}}
$$

Both factors matter and they pull in opposite directions — Courtemanche needs
6.25× the nodes but takes a *larger* step than Aliev–Panfilov, so the step term
is close to unity rather than the penalty a naive reading expects.

### 2. Membrane factor — `test_the_membrane_factor_is_the_ionic_model_alone`

This is the one the whole exercise exists for, and it is arranged so **only the
membrane differs**: the same 160² mesh for both models, no trackers attached, no
electrogram computed, no bank written. Anything that would scale with node count
rather than with membrane complexity is removed rather than subtracted.

**Timed by slope between two run lengths, never by a single run.** `model.run()`
carries a fixed setup cost — thread-pool spin-up and array allocation — of about
429 ms on a 160² mesh. Dividing a single run by its step count charges that
overhead to the model:

$$
\text{ns per node-step} \;=\; \frac{t_{\text{long}} - t_{\text{short}}}{(\text{steps}_{\text{long}} - \text{steps}_{\text{short}}) \times \text{nodes}}
$$

The difference cancels the fixed term exactly, for either model, without needing
to know its size. See the corrections section below for what this cost us before
it was noticed.

**Run lengths are sized to a target duration, not to a step count.** A fixed step
count makes the cheap model's run short and the expensive model's run long, so
the two are measured under different amounts of scheduler noise. Sizing to
duration puts both in the same regime. Aliev–Panfilov's pair spans 10× rather
than 4× because its kernel is cheap enough that per-step overhead is a larger
share of it.

### 3–4. Operational recorders — `test_record_*_operational_cost`

The full production path for one simulation: solver, electrogram tracking,
bipolar pairing, band-limiting, decimation, cropping. **These record and assert
nothing about time.** They are how a per-simulation figure and the projections
below are obtained, and they are the reason the machine table exists.

### What is asserted and what is only recorded

| quantity | treatment | why |
|---|---|---|
| ns per node-step, seconds per simulation | **recorded** | a fact about a machine as much as about this code; an assertion on one fails on any hardware it was not written on, and would then be widened until it meant nothing |
| membrane factor, mesh factor | **asserted** | properties of the code and the cards rather than of the CPU, so they survive being carried to another machine |

The membrane assertion band is **12–40×**, against measurements of 21.3×, 23.2× and 29.0× across
three sessions. That is wide on purpose — see the precision note below; the scatter is ~35 % —
and its limits should be stated rather than assumed: it catches a kernel that
became 3× slower (≈65×) or a Courtemanche that stopped being an ionic model
(≈1×). **It would not catch a 50 % regression.** It is a gross-regression guard,
not a precision instrument.

### Reproducing it

```
pytest -m benchmark -s
```

`-s` because the recorders print their tables. Nothing here runs in CI by
design — see the marker note in `pyproject.toml`.

---

## Operational cost — what a whole simulation costs

Each model **at the pitch it is actually run at**. This is deliberately *not*
a like-for-like membrane comparison: Courtemanche's cards are solved at 0.1 mm,
and running them at 0.25 mm borrows a conduction-velocity anchor and reads the
upstroke about 10 % high. A 0.25-against-0.25 table would compare two membranes
over a configuration nobody should generate from. What is measured here is the
**operational** cost — what each model costs to run the way the project runs it.

Includes everything a bank pays for: substrate realisation, the solve, the
electrogram tracker, bipolar pairing, band-limiting and the crop.

| machine | model | pitch | patch | grid | dt (ms) | capture (ms) | per sim |
|---|---|---|---|---|---|---|---|
| laptop | Aliev–Panfilov | 0.25 mm | 40 mm | 160² | 0.0102606 | 615 | **16.8 s** (16.8 / 16.8 / 23.3) |
| laptop | Courtemanche control | 0.10 mm | 40 mm | 400² | 0.0086 | 615 | **2538 s ≈ 42 min** (2538 / 2713) |
| desktop | either | — | — | — | — | — | **NOT MEASURED** |

**Operational ratio: 151×.** Below the solver-side product (7.46 × 29.0 ≈ 216×), because the
electrogram tracking and setup scale with node count but not with step count — so that overhead grows
6.25× for Courtemanche while the solve grows far more. Direction as predicted; magnitude was not.

**What it costs to actually generate.** Sequential loop, laptop:

| | Aliev–Panfilov | Courtemanche |
|---|---|---|
| one simulation | 16.8 s | **42 min** |
| 100-sim bank | 0.5 h | **70.5 h ≈ 2.9 days** |
| a 5-knob × 3-level OAT screen (11 cells × 100 sims) | **5.1 h** | **776 h ≈ 32 days** |

> **⚠ PROVISIONAL — laptop only.** The desktop is the machine a generation run would actually use
> and has not been measured. Treat every figure here as an upper bound on time and draw no
> conclusion about what Courtemanche can be used for until it has been run there.

### Measurement precision — worse than the digits suggest

The membrane factor has been measured three times on this machine: **21.3×, 23.2×, 29.0×** — a 35 %
spread. The raw runs show why: three repetitions of the same 2000-step Courtemanche solve gave
**55.03, 62.20 and 11.22 s**, a 5.5× spread within one measurement. Best-of-three recovers a usable
slope, but **this hardware cannot resolve the ratio better than roughly ±30 %**, and any figure
quoted from it to three significant figures is false precision. The 12–40× assertion band was
correctly wide rather than lazily wide.

Fastest of the repetitions is quoted, with all repetitions beside it. Other
processes can only ever add time, so on a machine someone is also using the
smallest observation is the closest one to the cost of the code — and showing
the spread is the reader's only signal that the machine was quiet enough to
believe the number.

## The two factors behind it

The total is reported as two separately measured pieces, so a later reader can
tell how much of the cost is the ionic model itself and how much is the mesh
resolution its upstroke demands. They are different kinds of quantity and only
one of them needs a stopwatch.

### Mesh factor — arithmetic, asserted in the fast suite

Nodes × timesteps between the two operating points. Both example configs capture
for the same 615 ms, so this is nodes and timestep alone.

| | vs Aliev–Panfilov |
|---|---|
| `courtemanche_control` | **7.46×** |
| `af_remodelled_crn_220ms` | **7.87×** |

The two differ only through `dt` — the AF card's smaller diffusion-bound step.

**This is not the 39× the project carried for a while.** Two separate things are
wrong with that figure. It assumed `dt ∝ dr²` all the way down, which fails
because at 0.25 mm Courtemanche's step is capped by the sodium current's ionic
ceiling rather than by the diffusion bound — so refining only pays the quadratic
penalty over part of the range. And it is a *different comparison*: 39× was
Courtemanche against **itself** at a coarser pitch, not against Aliev–Panfilov.
Three numbers are easy to confuse here and are worth keeping apart:

| quantity | value | comparison |
|---|---|---|
| mesh factor | 7.46× | AP at 0.25 mm vs CRN at 0.10 mm |
| CRN mesh refinement | 14.5× | CRN at 0.25 mm vs CRN at 0.10 mm |
| the superseded figure | 39× | the same refinement, computed as if `dt ∝ dr²` held throughout |

Because it costs nothing to check, this one runs in the **ordinary fast suite**
rather than behind the benchmark marker, and guards the cards on every change.

### Membrane factor — 21 state variables against 2

Both models on one 160² mesh, no trackers, so only the membrane differs.

| machine | date | AP (ns/node-step) | CRN (ns/node-step) | factor |
|---|---|---|---|---|
| laptop | 2026-08-27 | 8.9 | 206.5 | 23.2× |
| laptop | 2026-08-27 | 8.4 | 178.8 | **21.3×** |

Two runs of the same measurement, agreeing within 9 %. Call it **21–23×**, and
see the uncertainty note below before quoting a tighter figure.

Asserted at `12 ≤ factor ≤ 40`. Wide, and honestly so: vectorisation and cache
behaviour genuinely differ between CPUs, and the run-to-run scatter on this
laptop reached 20 % even after outlier rejection. It is still not unbounded — an
ionic kernel that got 3× slower lands near 65, and a Courtemanche model that
quietly stopped being one lands near 1.

---

## Two methodology corrections, both of which moved the answer

**These are recorded because the first version of each produced a number that
was reported before it was checked.**

### A single-length timing measured the setup, not the model

`model.run()` carries a fixed cost per call — thread-pool spin-up and array
allocation — measured at **~429 ms on a 160² mesh**. Dividing one short run by
its own step count therefore charges that fixed cost to the steps.

At 2000 steps it was **57 % of the Aliev–Panfilov measurement**. That method
reported 8.9 ns for a model whose marginal cost is 8.4 — tolerable — but had
earlier reported **12.6 ns**, and the membrane factor built on it came out at
**18.1×** against a true 21–23×. Courtemanche never showed the bias, because its
runs were already tens of seconds long, which is exactly why the error survived
a sanity check: the number that was wrong was the one nobody suspected.

The harness now takes the **slope** between two run lengths, which cancels the
fixed term exactly, for either model, without needing to know its size:

```
ns/node-step = (t_long − t_short) / (steps_long − steps_short) / nodes
```

### A short run on a busy laptop measures the laptop

Even with the slope method, Aliev–Panfilov measured over 2000–8000 steps gave
**6.3, 18.6 and 8.9 ns** on three runs with no code change between them — a 3×
swing. Its kernel is cheap enough that per-step threading overhead is a large
share of each step, and on four physical cores that someone is also using, the
scheduler dominates. Courtemanche never swung more than ±10 %.

Fixed by sizing each model's run to a *duration* rather than to a shared step
count, and by spanning 10× for Aliev–Panfilov so the slope is large compared
with the scatter around it. After that change AP gave 8.9 and 8.4 ns on
successive runs.

Outliers are still real and still visible in the logs — a 400 000-step AP run
took 214 s against 85 s twice, and a 2000-step CRN run took 125 s against 10 s
twice. Best-of-three rejects them, which is what `min` is for here.

## Uncertainty, stated rather than implied

The laptop was under interactive load throughout. Run-to-run scatter reached
20 % on the ionic model and, before the fix above, 3× on the phenomenological
one. **The figures here are good to roughly ±20 %, not better.**

Anything that needs a tighter number should re-run the harness on the desktop
with the machine otherwise idle. That is also the machine an overnight
generation run would use, so it is the one the data plan should actually be
built on.

## What is not claimed

- **Not a membrane-cost comparison at matched resolution.** The operational
  table compares two operating points, not two membranes; the membrane factor
  is the row that isolates the membrane, and it is measured on one shared mesh.
- **Not a projection for any machine but this one.** The desktop rows are blank
  because they were not measured, not because they were assumed equal.
- **Not a statement about scaling across cores.** Every figure is single-process
  with whatever threading numba chose by default; capping that is a separate
  question.
