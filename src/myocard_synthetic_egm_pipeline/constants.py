"""Defaults shared across the synthetic EGM pipeline.

Simulator-side constants only. Noise-side constants (bipolar band, window
sizes, IAFDB channel layout) live in ``myocard-iafdb-pipeline`` and
``myocard-egm-signal`` — the synthetic side reads them as imports rather
than redefining.

Schema version stamps come from ``myocard_egm_contracts.schema_info``
rather than being hard-coded here; the producer always stamps the version
it was linked against.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tissue (2D atrial patch) defaults
# ---------------------------------------------------------------------------

DEFAULT_PATCH_SIZE_MM: float = 40.0
"""2D patch edge length (mm). Matches the Sánchez 2021 patch scale."""

DEFAULT_PATCH_DR_MM: float = 0.25
"""Spatial step (mm). Finitewave default for phenomenological models like AP."""

DEFAULT_ANISOTROPY_RATIO: float = 2.0
"""Conduction velocity along:across ratio.

Atrial physiology sits in the 2-3:1 range; 3.0 is the project default.

**Prescriptive since 2026-08-14, and only since then.** v0.2.0 claimed to have
made this value prescriptive, but the helper wrote the tensor components to the
*model* while Finitewave reads them off the **stencil**, so the realized ratio
was always the stencil's built-in 3.09 no matter what was requested (CL-172).
The claim is retracted; the fix is in
``backends.finitewave.backend._configure_anisotropy_2d``.

Because the config has always requested 3.0 and the accidental value was 3.09,
**no bank generated before the fix is wrong** — the knob was inoperative, not
mis-set.

**2.0 since S38b** (CL-176), down from 3.0. Hansson *Eur Heart J* 1998;19:293
measured right-atrial free wall intra-operatively in sinus rhythm at 88 ± 9 cm/s
and only weakly direction-dependent — 74-81 cm/s across four propagation
directions. The high ratios belong to **bundles** (crista terminalis, pectinate
muscles), not to working myocardium, where 2:1 is standard, and our patch is
generic working myocardium. At 2.0 the transverse velocity lands near 41 cm/s,
inside the usual 30-50 range; at 3.0 it was forced to 26.5, below it.

The value stayed 3.0 through S38a deliberately: that step made the knob
*operative* and had to be bit-identical at the shipped setting, so changing the
number waited for the recalibration that regenerates everything anyway.
"""

# ---------------------------------------------------------------------------
# Time-domain output defaults
# ---------------------------------------------------------------------------

DEFAULT_TRACE_DURATION_MS: float = 192.0
"""Per-trace duration after downsampling (ms). Captures one full activation.

**192, not 200** (CL-112, design §8.1). At :data:`DEFAULT_OUTPUT_FS_HZ` this is
T = 192 samples, and T must be a multiple of 64: egm-classifier's 1D MobileViT
downsamples by 2 six times, so a length off that grid does not merely degrade
the model — it fails outright at the first ragged stage. 200 is off the grid.

192 is the largest multiple of 64 at or below the 200 ms this defaulted to, so
the change costs 8 ms of capture and no activation morphology.
"""

DEFAULT_OUTPUT_FS_HZ: float = 1000.0
"""Per-trace sampling rate (Hz). Matches IAFDB so synthetic and real are
mixable without resampling."""

# Aliev-Panfilov is non-dimensional. We map model time units → physical ms
# using a single calibration constant. Calibrated 2026-06-10 against an
# atrial longitudinal CV target of 80 cm/s using the calibration config
# pair in ``configs/`` and the analysis script in the (legacy)
# documentation tree. Re-run the calibration when anything upstream
# changes (dr, AP diffusion coef, model swap to Courtemanche).
AP_TIME_UNIT_MS: float = 1.97
"""**Legacy fallback only. Superseded by model cards (S38b).**

The operative value now comes from a model card's ``solved.time_unit_ms``,
derived from physiological targets by ``simulate.calibration.calibrate``. This
constant survives as the default for a ``RunConfig`` built without a card, so
pre-S38b code keeps meaning what it meant.

**It is also a cautionary tale, which is why the history stays here.** Set
2026-06-10 (from 12.9, the AP 1996 canine fit) by scaling to hit an 80 cm/s
longitudinal CV target. The arithmetic was right and the result was wrong: this
constant appears in the CV expression and the APD expression in *opposite*
senses, so buying velocity with it sold action potential duration — APD fell
from ~334 ms to 51 ms, putting a repolarisation deflection inside the analysis
window at a fixed 51 ms offset after every activation. Two months, and nothing
could contradict it, because the only artifact was four numbers and a comment.

The measurement that revealed it was itself taken through a broken layer: the
"12.2 cm/s" above was the *transverse* velocity read under a longitudinal label,
because the fibre field was transposed (CL-170).
"""

# ---------------------------------------------------------------------------
# Electrode grid defaults
# ---------------------------------------------------------------------------

DEFAULT_ELECTRODE_GRID_ROWS: int = 5
DEFAULT_ELECTRODE_GRID_COLS: int = 5
DEFAULT_ELECTRODE_SPACING_MM: float = 2.0
"""5x5 electrode grid, 2 mm intra-row spacing (matches IAFDB CS pairs / PentaRay)."""

DEFAULT_ELECTRODE_HEIGHT_MM_RANGE: tuple[float, float] = (0.2, 1.0)
"""Per-simulation electrode height sampled uniformly from this range.

Models the realistic range of catheter-tissue contact quality (firm
contact ↔ slight floating). Lower bound of 0.2 mm is set by the
dr=0.25 mm mesh discretization — need ≥ ~1 cell off-surface to avoid
the 1/r² singularity in the pseudo-EGM formula.
"""

# ---------------------------------------------------------------------------
# Fibrosis defaults
# ---------------------------------------------------------------------------

DEFAULT_FIBROSIS_DENSITY_RANGE: tuple[float, float] = (0.0, 0.5)
"""Per-simulation fibrosis density sampled uniformly from this range.

Capped at 0.5 to stay safely below the ~0.6 propagation-failure regime
documented in Nezlobinsky 2021.
"""

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

BANK_SOURCE: str = "synthetic_egm_pipeline"
"""Provenance tag stamped into every synthetic_bank's root source attr."""

THETA_BANK_SOURCE: str = "synthetic_generation_params"
"""``bank_type`` of the ClassifierBank entry pointing at its ``synthetic_bank``.

A synthetic run writes two artifacts joined on ``simulation_id``, and until
today nothing in the ClassifierBank said which ``synthetic_bank`` was its
partner — pairing them was a fact that lived only in someone's head. This
entry records it.

**Phase-1.5 scope.** It is unambiguous only while a ClassifierBank carries
traces from one run. Concatenate two such banks and you get two source
entries and two companion entries with no way to pair them, because traces
reference their *source* entry only. We do not concatenate in Phase 1.5; the
general fix needs a real relationship field on the entry and is Phase-2 work.
"""

LOCAL_BANK_PATH: str = "<local>"
r"""``bank_path`` sentinel meaning "the traces are in **this** file".

``ClassifierBankMetaData.bank_path`` is documented as the path a source bank
was *loaded from*, which presumes the traces came from somewhere else. A
producer **originates** its traces: there is no source file, and writing one
in anyway is how the field ended up naming a bank that is never written (the
clean path) or one whose traces differ from the ones in the file (the
noise-mixed path).

Why a sentinel rather than an empty string: ``""`` is indistinguishable from
"nobody filled this in", so it cannot mean *deliberately local* and *missing*
at once. Reserving it for the latter turns a blank into a bug signal.

Why angle brackets specifically: ``<`` and ``>`` are **illegal in Windows
filenames** (with ``: " / \ | ? *``), so this string can never collide with a
real portable path — including a bare relative filename such as
``run7.synthetic.h5``, which is what companion entries carry. The convention
is the same one Python uses for ``<stdin>`` / ``<string>``.
"""
