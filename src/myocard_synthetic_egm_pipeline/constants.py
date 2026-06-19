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

DEFAULT_ANISOTROPY_RATIO: float = 3.0
"""Conduction velocity along:across ratio.

Atrial physiology sits in the 2-3:1 range; 3.0 is the project default.
The value is now prescriptive (vs. the v0.1.0 behavior where it was
metadata only) — see ``simulate.tissue.configure_anisotropy`` for the
helper that plumbs the ratio through to Finitewave's diffusion tensor.
"""

# ---------------------------------------------------------------------------
# Time-domain output defaults
# ---------------------------------------------------------------------------

DEFAULT_TRACE_DURATION_MS: float = 200.0
"""Per-trace duration after downsampling (ms). Captures one full activation."""

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
"""Aliev-Panfilov model time-unit → physical ms conversion. Calibrated
2026-06-10 (was 12.9, the AP 1996 canine fit). Target was 80 cm/s
longitudinal CV; observed under-K=12.9 was 12.2 cm/s, scaled by
0.122/0.80 to give K = 1.97. Transverse calibration confirmed the same
K within rounding."""

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
