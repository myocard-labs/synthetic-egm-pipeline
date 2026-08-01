# Mixer theory — what each step does to the signal

This document is the mixer's counterpart to
[`simulation_theory.md`](simulation_theory.md). It walks through the
math the mixer applies to each clean synthetic trace + each sampled
noise segment, what each step represents physically, and what it does
to the bipolar EGM the classifier eventually sees.

Each step cites the file + function in this repo where it's
implemented; if you want to jump from a concept to the code, follow
the **Code:** line.

The reading list at the bottom is annotated. Start with the two
**[anchor]** papers if you only read two.

## The pipeline at a glance

Each trace in a clean ClassifierBank goes through six steps. Three are
per-trace math (1, 2, 3); two are bookkeeping (4, 5); one is
provenance (6).

```
                    one clean trace
   ┌──────────────────────────────────────────────────────┐
   │                                                      │
   │  1. Band-pass the clean trace                        │
   │     (sig → sig_bp at 30-300 Hz)                      │
   │  2. Sample a noise segment                           │
   │     (random pick + tile/crop to T samples)           │
   │  3. Compute the SNR scale alpha                      │
   │     such that sig_bp + alpha*noise has target SNR   │
   │  4. Additive mix in the bandpass domain              │
   │     (mixed = sig_bp + alpha * noise)                 │
   │  5. Per-trace SNR sampled from a range               │
   │     (every trace gets a different target SNR)        │
   │  6. Stamp audit fields onto trace_metadata           │
   │     + append a "mixer" entry to bank.banks           │
   │                                                      │
   └──────────────────────────────────────────────────────┘

  what the classifier ends up training on:
    bipolar EGM trace (n_pairs, T_samples) at 1 kHz,
    one IAFDB-noise component per trace, SNR sampled from
    the configured range (default 10-25 dB).
```

The whole thing is driven by `mix_classifier_bank`, which iterates
the clean bank's traces and applies steps 1-6 per trace.

**Code:** `mixer/mixing.py::mix_classifier_bank`

## Step 1 — Band-pass the clean trace

**What the code does.** Apply a zero-phase Butterworth band-pass to
the clean trace at the clinical bipolar EGM band (default ~ 30 - 300
Hz). The filter is delegated to `myocard-egm-signal`'s
backend-agnostic `bandpass` primitive so the producer, the mixer, and
any future analyser characterise signals in the same band.

**What this represents.** The clean synthetic trace, straight out of
the pseudo-EGM forward calc, has spectral energy at *all* frequencies
the AP simulator produced. Real bipolar EGM recordings, by contrast,
are band-passed at acquisition time (the EP recording system's
analog filters) to suppress baseline wander (sub-1 Hz) and high-
frequency electronic noise (above a few hundred Hz). The IAFDB noise
bank's segments were already band-passed at extraction time by
`iafdb-pipeline`. To make the additive mix meaningful, the clean
side has to live in the same band — otherwise sub-band energy in
the clean signal that the classifier will never see leaks into the
SNR calculation and biases the scaling.

**What this does to the signal.** Sharp out-of-band content is
attenuated; in-band morphology (the activation spike, post-activation
ripple) is preserved. The bandpass is zero-phase (forward-backward
filtfilt under the hood) so peak timing isn't shifted — important
because the activation deflection's timing carries the morphology
information.

**Code:**
- Orchestration: `mixer/mixing.py::mix_classifier_bank` (the
  `if cfg.bandpass_clean: sig_bp = bandpass(...)` block).
- Primitive: `myocard_egm_signal::bandpass` (shared, axis-0,
  zero-phase Butterworth).
- Config knob: `mixer/mixing.py::MixerConfig.bandpass_clean` (True
  by default; turn off only for ablation studies that need raw
  un-bandpassed clean traces).

**Read deeper.**
- **[Sánchez 2021]** — Uses the same band (30-300 Hz) for hybrid
  synthetic+real atrial EGM training. The methodological ancestor
  of this mixer [anchor].
- **[Unger 2019]**, **[Deno 2017]** — Independent confirmations of
  the clinical bipolar-EGM band convention.

## Step 2 — Sample a noise segment

**What the code does.** Pick one segment uniformly at random from the
noise bank. If the segment's length matches the clean trace's length,
return it as-is. If it's longer, crop a random sub-window of the
exact length needed. If it's shorter, tile (repeat) and crop to the
exact length needed.

**What this represents.** The IAFDB noise bank is a collection of
windows from real recordings that `iafdb-pipeline`'s threshold
strategy flagged as low-amplitude (under Sanders 2003's
"electrically silent" tier or a per-record percentile cutoff). Each
window carries a (`source_record`, `source_channel`) tag — when you
look at a noise-mixed trace later, you can trace its noise component back
to the specific IAFDB record + bipolar channel it came from.

For each clean trace we want a noise segment of the same length so
the additive mix is well-defined. The bank's segments are typically
200 ms windows (matching iafdb-pipeline's noise-extraction default),
but the clean traces are also typically 200 ms — so most of the
time the lengths match. The tile/crop logic handles the edge cases:

- **Length match**: pass the segment through. Most common path.
- **Bank segment longer than trace**: crop a random sub-window. The
  random offset means two clean traces drawing from the same noise
  segment can still see different noise.
- **Bank segment shorter than trace**: tile (repeat) until covered;
  crop to exact length. For Phase 1 we don't cross-fade at tile
  joins — noise is broadband so the joint discontinuity sits well
  below physiological amplitude.

**What this does to the signal.** Produces a `(T_samples,)` noise
array ready to be added to the clean trace. The accompanying
`(source_record, source_channel)` audit fields propagate to the
noise-mixed trace's `trace_metadata`.

**Code:** `mixer/mixing.py::sample_noise_for_length` — handles the
three branches (match / crop / tile) and returns the audit fields
along with the signal.

**Read deeper.**
- See `iafdb-pipeline`'s `docs/usage.md` for how the noise bank is
  produced (threshold strategies, calibration policy, the
  intentional minimalism of the `noise_bank` schema).

## Step 3 — Compute the SNR scale

**What the code does.** Given the bandpassed clean signal, the
sampled noise, and a target SNR in decibels, compute the scalar α to
multiply the noise by so the post-mix SNR matches the target.

**What this represents.** SNR (signal-to-noise ratio) is the
canonical way to characterise the relative power of two additive
components. The decibel form is

$$
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\left(\frac{P_s}{P_n}\right)
$$

where $P_s$ is the mean-square power of the clean signal and $P_n$ is
the mean-square power of the noise *as it ends up in the mix*. We
want to set α so that

$$
10 \log_{10}\!\left(\frac{P_s}{\alpha^2 \, P_n}\right) \;=\; \mathrm{SNR}_{\mathrm{dB}}^{\mathrm{target}}
$$

Solving for α:

$$
\alpha \;=\; \sqrt{\frac{P_s}{10^{\mathrm{SNR}/10} \cdot P_n}}
$$

The function computes $P_s = \frac{1}{T}\sum_t s_t^2$ and $P_n =
\frac{1}{T}\sum_t n_t^2$ directly from the arrays (mean-square = RMS
squared), then evaluates the formula. Two degenerate cases are
handled defensively:

- **Zero-power noise** ($P_n = 0$): return α = 0 (no-op mix; the
  trace stays clean). This shouldn't happen for IAFDB noise — every
  segment has at least some baseline activity — but the noise bank
  could in principle contain a flat-line segment if the extraction
  was misconfigured.
- **Zero-power clean** ($P_s = 0$): return α = 1 (the mix is just
  the raw noise). Avoids dividing by zero in the formula.

**What this does to the signal.** Produces a single scalar α. The
mix in Step 4 is `sig + alpha * noise`; α governs how loud the noise
is relative to the signal. A larger α = louder noise = lower SNR
(harder for the classifier).

**Code:** `mixer/mixing.py::snr_scale` — pure math, no side effects.
Exposed in the public API so notebooks can verify SNR calibration
independently.

**Read deeper.**
- **[Vaseghi 2008]** — Standard textbook treatment of additive noise
  models and SNR. Chapter 1.

## Step 4 — Additive mix in the bandpass domain

**What the code does.** Add the SNR-scaled noise to the bandpassed
clean signal:

```python
mixed = sig_bp + alpha * noise
```

That's it. One line.

**What this represents.** The choice of *additive* over
*multiplicative* (or convolutive) noise is deliberate. EP recording
systems contribute ambient noise — thermal noise in the amplifier,
60 Hz line interference, electrode-tissue interface noise — that
sums into the recorded signal in the same physical band as the
bipolar EGM. The signal we're trying to recover and the noise we're
trying to be robust against are co-additive in the recorded V or mV
trace.

Multiplicative noise would model amplitude jitter (the recording
system's gain wobbling per-sample). Convolutive noise would model
filter mismatch or unmodelled IR. Neither matches the dominant
real-world failure mode for atrial bipolar EGM, which is additive
ambient + electrode contact noise.

**What this does to the signal.** Produces the final noise-mixed bipolar
trace — same shape as the clean trace, same units, with noise added
at the configured SNR. This is what the classifier trains on.

**Code:** `mixer/mixing.py::mix_classifier_bank`, the
`mixed = sig_bp + alpha * noise` line right after the `snr_scale`
call.

**Read deeper.**
- **[Goodfellow 2014]** §7.5 — Why adding noise to training data is
  a regulariser; the variance-reduction story when the noise is
  zero-mean and the loss is mean-squared-error-like.
- **[Sánchez 2021]** — Sections on hybrid corpus construction for
  atrial-EGM ML. Closest direct precedent for what we're doing here.

## Step 5 — Per-trace SNR variability

**What the code does.** Sample the per-trace target SNR uniformly
from a configured range `(snr_lo, snr_hi)`. Default range is
`(10.0, 25.0)` dB.

**What this represents.** The classifier needs to be robust to the
range of SNRs it will see in real deployment. Real EP recordings vary
substantially in SNR — a firm-contact mapping point in a clean
electrically-quiet area sits at maybe 25 dB SNR; a glancing-contact
point in a noisy area might be at 5-10 dB. Training on a single
fixed SNR teaches the classifier exactly one operating point, which
generalises poorly. Training on a *distribution* of SNRs is the
straightforward regularisation answer.

A uniform distribution over `(snr_lo, snr_hi)` is the simplest
choice. A log-uniform distribution (uniform in dB-space, which we
already have for free since `snr_db` is the parameter we sample) is
correct — perceived SNR is logarithmic in linear power, so uniform
spacing in dB corresponds to uniform spacing in the perceptually
relevant axis.

For ablation studies that want every trace to see the same SNR, set
`snr_db_range=(X, X)` for a fixed X — the uniform sample collapses
to a point.

**What this does to the signal.** Different traces in the same
training run see different noise loudness. Across the full bank
(~2000 traces in the v1 spec), the SNR distribution is approximately
uniform in dB-space across the configured range, which makes the
classifier robust to deployment-time SNR variation.

**Code:** `mixer/mixing.py::mix_classifier_bank`, the
`target_snr_db = float(rng.uniform(snr_lo, snr_hi))` line. Range
configured via `MixerConfig.snr_db_range`.

**Read deeper.**
- **[Cubuk 2019]** (RandAugment) — The general principle of
  parameter-distribution augmentation rather than fixed
  augmentation. Different domain (vision) but the same regularisation
  story.

## Step 6 — Provenance and audit trail

**What the code does.** For each noise-mixed trace, stamp three audit
fields into `trace_metadata`:

- `snr_db` — the realised per-trace target SNR (dB).
- `noise_record` — source dataset record (e.g. `"iaf1_afw"`).
- `noise_channel` — source bipolar channel (e.g. `"CS12"`).

Plus, append one `ClassifierBankMetaData` entry to the bank's
`banks` list with `bank_type="mixer"` carrying the mixer config (SNR
range, band, bandpass flag, master seed, noise bank source). The
clean source bank's entry stays in place — both layers of provenance
coexist.

**What this represents.** Reproducibility + debuggability. A
classifier prediction can be traced back to:

1. The clean source bank entry → which simulator config produced
   the underlying clean trace.
2. The "mixer" entry → which mixer config + which noise bank was
   overlaid.
3. The trace_metadata → which noise record + channel + SNR went
   into *this* trace specifically.

That's enough information to recreate the exact noise-mixed trace from
the source banks if needed (modulo the random seed for noise
selection — saved on the mixer entry's `master_seed` field).

**What this does to the signal.** Nothing — provenance is metadata,
not modifications to the signal. They land in the ClassifierBank
alongside the trace and are available to any downstream consumer
(egm-studio, eval-time stratification by SNR or source record, etc.).
They are present **only on a mixed bank** — a clean bank omits them
rather than writing NaN, which would read as "mixed, SNR unknown".

**Code:**
- Per-trace audit fields:
  `mixer/mixing.py::mix_classifier_bank`, the
  `mixed_metadata["snr_db"] = ...` block.
- Bank-level mixer entry: `mixer/mixing.py::mix_classifier_bank`,
  the `mixer_entry = ClassifierBankMetaData(...)` construction.
- Noise-mixed `synthetic_bank`: written by the **inline** path
  (`synthegm-generate-dataset` with a `mix:` block) from the in-memory
  `DatasetResult`, passing the mixed signals plus the three noise
  columns. Schema 2.0's per-simulation generation config cannot be
  reconstructed from a mixed ClassifierBank, so the previous
  rebuild-from-ClassifierBank route is gone; standalone `synthegm-mix`
  therefore writes only a ClassifierBank.

## What the classifier sees after mixing

After the six steps, each noise-mixed training example looks like:

| Field | Type | Origin |
|---|---|---|
| `signal` | `(T_samples,)` float32 at 1 kHz | bandpassed clean + α·noise |
| `label_truth` | int (0 = healthy, 1 = fibrotic) | set during clean dataset gen by LabelPolicy |
| `trace_metadata.simulation_id` | int | join key into the `synthetic_bank`'s per-sim config |
| `trace_metadata.pair_index` | int | which bipolar pair of that simulation |
| `trace_metadata.patient_id` | str | `= simulation_id`; the patient-aware split unit |
| `trace_metadata.snr_db` | float | this mixer's SNR sample |
| `trace_metadata.noise_record` | str | IAFDB source record |
| `trace_metadata.noise_channel` | str | IAFDB source channel |

Generation parameters — density, stimulus edge, electrode height — are
**not** here. They live once per simulation on the `synthetic_bank` and
are reached through `simulation_id`; carrying a copy per trace is the
duplication schema 2.0 exists to remove.

With ~100 simulations × 20 bipolar pairs = ~2000 traces, each at a
different per-trace SNR sampled from the configured range, each
overlaid with one random IAFDB noise segment. The classifier sees
bipolar morphology under a realistic noise profile; the per-trace
SNR distribution makes it robust to deployment-time SNR variation.

## Annotated reading list

The two **[anchor]** papers are the highest leverage if you only
read two. Group them as: domain precedent + math foundation.

**Synthetic-side ML precedent**
- **[Sánchez 2021]** — Sánchez J, et al. *Influence of left atrial
  ostial regions on the morphology of the left atrial appendage
  during atrial fibrillation: A computational and electroanatomic
  mapping study.* Front Physiol. Synthetic + real mixing for atrial
  EGM classification; the methodological ancestor of this mixer
  [anchor]. SNR range, band choice, additive-noise rationale all
  trace back here.

**SNR + additive noise foundations**
- **[Vaseghi 2008]** — Vaseghi SV. *Advanced Digital Signal
  Processing and Noise Reduction.* 4th ed., Wiley. Chapter 1
  textbook treatment of additive-noise models and SNR; the
  derivation of α we use is straight out of here [anchor].
- **[Proakis 2014]** — Proakis JG, Manolakis DG. *Digital Signal
  Processing.* Pearson. Chapter on noise and signal characterisation;
  alternative textbook with the same material.

**Augmentation + robustness theory**
- **[Goodfellow 2014]** — Goodfellow I, Bengio Y, Courville A.
  *Deep Learning.* MIT Press. §7.5 "Noise Robustness" derives the
  variance-reduction story for additive-input-noise training.
- **[Cubuk 2019]** — Cubuk ED, et al. *RandAugment: Practical
  automated data augmentation with a reduced search space.* CVPR
  Workshops. Argues for parameter-distribution augmentation over
  fixed augmentation. We don't search the SNR distribution
  automatically (we sample uniformly from a hand-set range), but the
  motivation is the same.

**Clinical EP signal interpretation (band + noise convention)**
- **[Unger 2019]** — Unger LA, et al. *Cycle length statistics
  during human atrial fibrillation reveal refractory properties of
  the underlying tissue: a combined in silico and in vivo study.*
  Europace. Confirms the 30-300 Hz band convention for atrial
  bipolar EGMs.
- **[Deno 2017]** — Deno DC, et al. *Orientation-independent
  catheter-based characterization of myocardial activation.* IEEE
  Trans Biomed Eng. Additional confirmation of the bandpass
  convention; also discusses orientation effects on bipolar EGM
  morphology that are upstream of any mixer concern.

**Source data + noise extraction**
- See `myocard-iafdb-pipeline` for the IAFDB noise extraction
  pipeline. The `noise_bank` schema (in `myocard-egm-contracts`) and
  its `noise_bank_run_record.json` sidecar document the provenance
  of the noise that the mixer consumes.
