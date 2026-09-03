"""Strategy Protocols + Phase-1 concrete implementations.

This module is the **public strategy surface** of the producer. Four
Protocols describe what can be plugged in (geometry, substrate,
activation, electrode placement), and the Phase-1 concretes are the
default implementations the CLIs construct from YAML.

Per ``project/architecture.md`` Guardrail 1: this module imports
**no backend code**. Concretes are pure-data specs — they describe what
the caller wants without knowing how a backend realizes it. The backend
(``backends/finitewave/backend.py``) holds the adapter logic that turns
each concrete strategy into its backend-native representation.

Protocols
---------
``GeometrySpec``         — tissue domain (2D patch today; 3D atrial mesh later)
``SubstrateStrategy``    — fibrosis pattern (uniform-random today; interstitial / patchy later)
``ActivationSource``     — wave initiation (planar edge today; point / pacing-train later)
``ElectrodePlacement``   — electrode positions + bipolar pair list

Phase-1 concretes
-----------------
``Patch2DGeometry``      — 2D square patch with anisotropy ratio
``UniformRandomFibrosis``— i.i.d. nodal fibrosis at a given density
``PlanarEdgeStimulus``   — thin strip of voltage on one of the four edges
``CenteredGrid2D``       — 5x5 grid centered on a 2D patch with sampled height
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from myocard_synthetic_egm_pipeline.constants import (
    DEFAULT_ANISOTROPY_RATIO,
    DEFAULT_ELECTRODE_GRID_COLS,
    DEFAULT_ELECTRODE_GRID_ROWS,
    DEFAULT_ELECTRODE_HEIGHT_MM_RANGE,
    DEFAULT_ELECTRODE_SPACING_MM,
    DEFAULT_PATCH_DR_MM,
    DEFAULT_PATCH_SIZE_MM,
)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@runtime_checkable
class GeometrySpec(Protocol):
    """A backend-agnostic description of the tissue domain.

    Concretes name themselves so a backend can dispatch on ``type``
    without isinstance walls. They expose enough geometric structure for
    the runner and the electrode-placement layer to lay out pair
    midpoints in physical units; backend-specific mesh details (cell
    count, boundary handling, etc.) are computed by the backend's
    adapter from these specs.

    The ``type`` attribute is declared as a read-only ``@property`` so
    frozen-dataclass concretes (Phase 1) satisfy it cleanly. The bare
    ``type: str`` form would require the concretes to be mutable.
    """

    @property
    def type(self) -> str:
        """Discriminator value matching the YAML ``geometry.type`` field."""
        ...


@dataclass(frozen=True)
class Patch2DGeometry:
    """A 2D square tissue patch.

    Phase-1 spec per ``simulator_v1_spec``: 40 mm square, dr = 0.25 mm
    (Finitewave default for phenomenological models), uniform fiber
    orientation along +x at ``anisotropy_ratio`` between the along- and
    across-fiber CV.

    Attributes
    ----------
    size_mm
        Physical edge length of the patch in millimetres.
    dr_mm
        Spatial step (mm). Mesh size is ``round(size_mm / dr_mm)`` per
        edge.
    fiber_angle_rad
        Fiber orientation in radians (0 = along +x, π/2 = along +y).
    anisotropy_ratio
        Conduction-velocity ratio CV_along / CV_across. Prescriptive: the
        backend's adapter shapes the **stencil's** diffusion tensor so the
        realized CV ratio matches (since CV ∝ √D, this means
        D_along / D_across = anisotropy_ratio²). Raising it holds the
        along-fibre velocity fixed and slows the transverse one, so it does
        not disturb a conduction-velocity calibration.

        Prescriptive **since 2026-08-14** — before that the adapter wrote to
        the model rather than the stencil and the knob did nothing at all.
    """

    size_mm: float = DEFAULT_PATCH_SIZE_MM
    dr_mm: float = DEFAULT_PATCH_DR_MM
    fiber_angle_rad: float = 0.0
    anisotropy_ratio: float = DEFAULT_ANISOTROPY_RATIO
    type: Literal["patch_2d"] = "patch_2d"

    def __post_init__(self) -> None:
        if self.size_mm <= 0:
            raise ValueError("size_mm must be positive.")
        if self.dr_mm <= 0:
            raise ValueError("dr_mm must be positive.")
        if self.anisotropy_ratio < 1.0:
            raise ValueError("anisotropy_ratio must be >= 1 (along/across; equal = isotropic).")

    @property
    def n_cells_per_edge(self) -> int:
        """Number of mesh cells per edge."""
        return round(self.size_mm / self.dr_mm)

    @property
    def shape(self) -> tuple[int, int]:
        """Mesh shape (n_i, n_j)."""
        return (self.n_cells_per_edge, self.n_cells_per_edge)


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------


@runtime_checkable
class SubstrateStrategy(Protocol):
    """Describes a fibrosis (or other substrate) pattern as pure data.

    A concrete strategy carries the parameters of the pattern (density,
    patch sizes, type distributions). The backend's adapter realizes the
    pattern on its native mesh representation; the strategy itself
    doesn't know about meshes.

    The realization-time metadata (realized density, count of fibrotic
    nodes, etc.) returns from the backend's adapter, NOT from the
    strategy — strategies are pure data and don't mutate state.
    """

    @property
    def type(self) -> str: ...


@dataclass(frozen=True)
class UniformRandomFibrosis:
    """I.i.d. nodal fibrosis at a configured density.

    For each interior mesh node, draw uniform[0, 1]; mark non-conductive
    if ``density > draw``. Matches Nezlobinsky 2021 / Okenov 2024.

    Attributes
    ----------
    density
        Target fibrotic-node fraction in [0, 1). 0.0 is exactly healthy;
        values above ~0.6 cause propagation failure (Nezlobinsky 2021).
        Phase 1 samples in [0, 0.5].
    """

    density: float
    type: Literal["uniform_random_fibrosis"] = "uniform_random_fibrosis"

    def __post_init__(self) -> None:
        if not 0.0 <= self.density < 1.0:
            raise ValueError("density must be in [0, 1).")


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


@runtime_checkable
class ActivationSource(Protocol):
    """Describes how to initiate the activation wave."""

    @property
    def type(self) -> str: ...


# Edge naming follows Finitewave's mesh convention and is the value
# persisted to the ``synthetic_bank`` schema's ``traces/stim_edge`` field
# (enum ``["top", "bottom", "left", "right"]``). Renaming requires a
# schema bump — see ``project/architecture.md`` for the convention notes.
Edge = Literal["top", "bottom", "left", "right"]
EDGES: tuple[Edge, ...] = ("top", "bottom", "left", "right")

_DEFAULT_VOLT_VALUE: float = 1.0
_DEFAULT_STRIP_THICKNESS: int = 3


@dataclass(frozen=True)
class PlanarEdgeStimulus:
    """Thin strip of voltage applied at one mesh edge.

    The wave propagates inward across the patch.

    Naming
    ------
    ``top`` / ``bottom`` / ``left`` / ``right`` follow Finitewave's mesh
    convention (axis 0 = "rows" = i, axis 1 = "cols" = j):

    - ``top``    → smallest ``i`` (axis-0 = 0 strip)
    - ``bottom`` → largest  ``i`` (axis-0 = n-1 strip)
    - ``left``   → smallest ``j`` (axis-1 = 0 strip)
    - ``right``  → largest  ``j`` (axis-1 = n-1 strip)

    The electrode coordinates use the same axis convention. The names
    are the values written into ``synthetic_bank``'s ``stim_edge``
    field, so the contract is owned by ``myocard-egm-contracts``.

    Attributes
    ----------
    edge
        Which edge fires.
    voltage
        Voltage value in AP model units. ~1.0 reliably triggers
        propagation in Aliev-Panfilov.
    time_model_units
        Time at which the stimulus fires, in model time units.
    strip_thickness
        Strip thickness in mesh cells.
    """

    edge: Edge
    voltage: float = _DEFAULT_VOLT_VALUE
    time_model_units: float = 0.0
    strip_thickness: int = _DEFAULT_STRIP_THICKNESS
    type: Literal["planar_edge"] = "planar_edge"

    def __post_init__(self) -> None:
        if self.edge not in EDGES:
            raise ValueError(f"edge must be one of {EDGES}, got {self.edge!r}.")
        if self.voltage <= 0:
            raise ValueError("voltage must be positive.")
        if self.strip_thickness < 1:
            raise ValueError("strip_thickness must be >= 1.")


def random_edge(rng: np.random.Generator) -> Edge:
    """Uniform-random sample of one of the four edges."""
    return EDGES[int(rng.integers(0, len(EDGES)))]


# ---------------------------------------------------------------------------
# Electrode placement
# ---------------------------------------------------------------------------


@runtime_checkable
class ElectrodePlacement(Protocol):
    """Describes electrode positions + bipolar pair list for one simulation.

    Concretes must expose ``positions_mm`` (electrode positions in
    millimetres relative to the geometry's origin) and ``bipolar_pairs``
    (list of (a_index, b_index) tuples). The backend's adapter converts
    ``positions_mm`` to its native coordinate system (e.g. Finitewave
    grid index units = mm / dr_mm).

    The Protocol describes **one placement instance** with one resolved
    set of positions. The dataset orchestrator constructs a fresh
    placement per simulation via the concrete's ``.sample(...)``
    classmethod, so per-sim varying positions are the expected pattern,
    not a static layout. For Phase-1 :class:`CenteredGrid2D`, X/Y are
    deterministic (centered grid) but Z (height) is sampled per sim
    from ``height_mm_range``. A future ``EndocardialSurface3D``
    placement would sample X/Y/Z together from anatomical regions of a
    3D mesh; the Protocol surface doesn't change, just the concrete's
    ``.sample(...)`` semantics.
    """

    @property
    def type(self) -> str: ...

    @property
    def positions_mm(self) -> npt.NDArray[np.float64]:
        """``(n_electrodes, 3)`` float64 array in physical millimetres."""
        ...

    @property
    def bipolar_pairs(self) -> tuple[tuple[int, int], ...]:
        """Indices into ``positions_mm`` defining each bipolar pair."""
        ...


@dataclass(frozen=True)
class CenteredGrid2D:
    """5x5 grid (default) centred on a 2D patch with sampled per-sim height.

    Within-row consecutive electrodes form bipolar pairs, so a 5x5 grid
    produces ``n_rows * (n_cols - 1) = 20`` bipolar traces.

    Heights are sampled per-simulation, so this dataclass is constructed
    via :meth:`sample` rather than directly. The result is a frozen
    snapshot of the realized placement for one simulation.

    Attributes
    ----------
    n_rows, n_cols
        Grid shape.
    spacing_mm
        Intra-row and inter-row electrode spacing.
    height_mm
        Per-instance electrode height above the tissue surface
        (sampled, not a configurable knob on the placement itself).
    positions_mm
        Resolved ``(n_electrodes, 3)`` electrode positions.
    bipolar_pairs
        Resolved tuple of ``(a, b)`` index pairs.
    """

    n_rows: int
    n_cols: int
    spacing_mm: float
    height_mm: float
    positions_mm: npt.NDArray[np.float64] = field(repr=False)
    bipolar_pairs: tuple[tuple[int, int], ...] = field(repr=False)
    type: Literal["centered_grid_2d"] = "centered_grid_2d"

    @classmethod
    def sample(
        cls,
        *,
        geometry: Patch2DGeometry,
        n_rows: int = DEFAULT_ELECTRODE_GRID_ROWS,
        n_cols: int = DEFAULT_ELECTRODE_GRID_COLS,
        spacing_mm: float = DEFAULT_ELECTRODE_SPACING_MM,
        height_mm_range: tuple[float, float] = DEFAULT_ELECTRODE_HEIGHT_MM_RANGE,
        rng: np.random.Generator | None = None,
    ) -> CenteredGrid2D:
        """Build a one-shot grid placement for a 2D patch with a sampled height.

        Parameters
        ----------
        geometry
            The 2D patch to centre the grid on.
        n_rows, n_cols, spacing_mm
            Grid layout (defaults from :mod:`constants`).
        height_mm_range
            ``(lo, hi)`` — per-call height sampled uniformly.
        rng
            Random generator. Defaults to fresh entropy.
        """
        if n_rows < 1 or n_cols < 2:
            raise ValueError("Need n_rows >= 1 and n_cols >= 2 to form bipolar pairs.")
        if spacing_mm <= 0:
            raise ValueError("spacing_mm must be positive.")
        lo, hi = height_mm_range
        if not 0 < lo <= hi:
            raise ValueError("height_mm_range must satisfy 0 < lo <= hi.")

        rng = rng if rng is not None else np.random.default_rng()
        height_mm = float(rng.uniform(lo, hi))
        return cls.at(
            geometry=geometry,
            n_rows=n_rows,
            n_cols=n_cols,
            spacing_mm=spacing_mm,
            height_mm=height_mm,
        )

    @classmethod
    def at(
        cls,
        *,
        geometry: Patch2DGeometry,
        n_rows: int = DEFAULT_ELECTRODE_GRID_ROWS,
        n_cols: int = DEFAULT_ELECTRODE_GRID_COLS,
        spacing_mm: float = DEFAULT_ELECTRODE_SPACING_MM,
        height_mm: float = DEFAULT_ELECTRODE_HEIGHT_MM_RANGE[0],
    ) -> CenteredGrid2D:
        """Build a grid at an **explicit** height, drawing nothing.

        :meth:`sample` is this with the height drawn first. Split apart because
        ``positions_mm`` is the only thing the backend actually reads —
        ``height_mm`` is a record of what produced it — so anything that changes
        the layout has to rebuild the positions rather than edit the scalar
        beside them. ``dataclasses.replace(grid, height_mm=...)`` produces an
        object whose recorded height and actual electrode geometry disagree, and
        nothing downstream would notice.

        That is why this is a constructor rather than a setter: there is no
        supported way to change one of these fields without recomputing the
        others, so the only offered route recomputes them all.
        """
        if n_rows < 1 or n_cols < 2:
            raise ValueError("Need n_rows >= 1 and n_cols >= 2 to form bipolar pairs.")
        if spacing_mm <= 0:
            raise ValueError("spacing_mm must be positive.")
        if height_mm <= 0:
            raise ValueError("height_mm must be positive.")

        # Centre the grid on the patch in (x, y) mm.
        grid_w_mm = (n_cols - 1) * spacing_mm
        grid_h_mm = (n_rows - 1) * spacing_mm
        x0_mm = (geometry.size_mm - grid_w_mm) / 2.0
        y0_mm = (geometry.size_mm - grid_h_mm) / 2.0
        if x0_mm < 0 or y0_mm < 0:
            raise ValueError(
                f"Electrode grid does not fit in patch: "
                f"grid ({grid_w_mm:.1f}x{grid_h_mm:.1f} mm) > patch "
                f"({geometry.size_mm} mm)."
            )

        n_electrodes = n_rows * n_cols
        positions = np.empty((n_electrodes, 3), dtype=np.float64)
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                positions[idx, 0] = x0_mm + c * spacing_mm  # x in mm
                positions[idx, 1] = y0_mm + r * spacing_mm  # y in mm
                positions[idx, 2] = height_mm  # z in mm

        pairs: list[tuple[int, int]] = []
        for r in range(n_rows):
            for c in range(n_cols - 1):
                a = r * n_cols + c
                b = r * n_cols + c + 1
                pairs.append((a, b))

        return cls(
            n_rows=n_rows,
            n_cols=n_cols,
            spacing_mm=spacing_mm,
            height_mm=height_mm,
            positions_mm=positions,
            bipolar_pairs=tuple(pairs),
        )

    @property
    def n_electrodes(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_bipolar_pairs(self) -> int:
        return len(self.bipolar_pairs)

    def bipolar_pair_midpoints_mm(self) -> npt.NDArray[np.float64]:
        """Per-pair midpoint coordinates ``(n_pairs, 3)`` in mm.

        Used by ``LocalDensityLabel`` and any future locality-aware
        analysis to query the substrate mask in a neighborhood of each
        bipolar pair without re-deriving the geometry.
        """
        midpoints = np.empty((self.n_bipolar_pairs, 3), dtype=np.float64)
        for pair_idx, (a, b) in enumerate(self.bipolar_pairs):
            midpoints[pair_idx] = 0.5 * (self.positions_mm[a] + self.positions_mm[b])
        return midpoints

    def row_col(self, electrode_index: int) -> tuple[int, int]:
        """Return (row, col) for a flat electrode index."""
        return divmod(electrode_index, self.n_cols)
