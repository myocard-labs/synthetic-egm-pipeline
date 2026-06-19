"""LabelPolicy Protocol + Phase-1 concrete policies.

Labels are computed inside the producer with full in-memory access to
the simulation state (substrate mask, bipolar pair midpoints, electrode
positions). The resulting integer labels land in a
:class:`~myocard_egm_data.banks.ClassifierBank` directly — no schema
fields to invent, no metadata packing across the producer/consumer
boundary.

Per ``project/architecture.md``: this module imports no backend code
and operates on :class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationResult`
only.

The two Phase-1 policies bracket the failure mode we hit on the
original v1 training run:

- :class:`GlobalDensityLabel` is the v0.1.0 baseline — global density >
  threshold ⇒ 1. Loses signal at low density (~10%) because a bipolar
  pair only "sees" a few mm around itself, not the whole patch.
- :class:`LocalDensityLabel` computes per-pair fibrotic density within
  a configurable radius of the pair midpoint, then thresholds. Matches
  the electrode's actual receptive field.

Adding new policies (multi-type substrate, neighborhood composition,
per-electrode distance-to-nearest-fibrotic) means dropping a new class
into this module — no Protocol changes, no schema changes, no
consumer-side migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult


@runtime_checkable
class LabelPolicy(Protocol):
    """A policy that turns a simulation result into per-trace labels.

    Implementations consume one :class:`SimulationResult` at a time
    (one simulation = many bipolar traces) and return a flat int64
    array of length ``result.n_pairs`` plus the global ``labels_dict``
    that names the integer codes.

    The labels_dict must be the same for every simulation in a dataset
    run — :class:`~myocard_synthetic_egm_pipeline.simulate.dataset.generate_dataset`
    asserts this and refuses to assemble a bank from disagreeing
    label-dict policies.
    """

    @property
    def type(self) -> str: ...

    @property
    def name(self) -> str: ...

    def apply(self, result: SimulationResult) -> tuple[npt.NDArray[np.int64], dict[int, str]]:
        """Compute labels for one simulation.

        Returns
        -------
        labels
            ``(n_pairs,)`` int64 array — one label per bipolar trace
            in ``result.bipolar_traces``.
        labels_dict
            Integer-to-name mapping. Constant across all simulations in
            a single dataset run.
        """
        ...


# ---------------------------------------------------------------------------
# Global density (v0.1.0 baseline behavior)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalDensityLabel:
    """Binary label from global fibrotic density.

    Every trace from the simulation gets the same label: 1 if the
    simulation's realized fibrotic density exceeds ``threshold``, else
    0. This preserves the v0.1.0 producer's label semantics.

    Known failure mode (see ``project/architecture.md`` for full
    context): at low global density (~10%) a bipolar pair's signal is
    dominated by what's within a few mm of the pair, not by the global
    density. Traces from a low-density sim with no fibrosis in the
    pair's neighborhood look like healthy traces but are labeled 1. Use
    :class:`LocalDensityLabel` to fix this.

    Attributes
    ----------
    threshold
        Density above which a simulation is labeled fibrotic. Default
        0.1 matches the v1.5 investigation's threshold.
    healthy_name, fibrotic_name
        Names for the two integer codes — surfaces in the
        ClassifierBank's ``labels`` dict.
    """

    threshold: float = 0.1
    healthy_name: str = "healthy"
    fibrotic_name: str = "fibrotic"
    name: str = "global_density"
    type: Literal["global_density"] = "global_density"

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError("threshold must be in [0, 1).")

    def apply(self, result: SimulationResult) -> tuple[npt.NDArray[np.int64], dict[int, str]]:
        realized = float(result.substrate_realization_metadata.get("density_realized", 0.0))
        label = 1 if realized > self.threshold else 0
        labels = np.full(result.n_pairs, label, dtype=np.int64)
        return labels, {0: self.healthy_name, 1: self.fibrotic_name}


# ---------------------------------------------------------------------------
# Local density (the actual receptive-field-aware label)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalDensityLabel:
    """Binary label from per-pair fibrotic density in a local neighborhood.

    For each bipolar pair, sample the substrate mask in a circle of
    radius ``radius_mm`` around the pair's midpoint, count fibrotic vs
    healthy interior nodes, and label 1 if the local density exceeds
    ``threshold``, else 0.

    This addresses the global-density failure mode: a low-density sim
    can produce a mix of "fibrotic-neighborhood" pairs (high local
    density) and "healthy-neighborhood" pairs (low local density). The
    bipolar signal sees only its local neighborhood, so the per-pair
    label is the right one for the classifier to learn.

    Attributes
    ----------
    radius_mm
        Circle radius around each pair midpoint, in physical mm.
        Default 2.0 mm matches the typical electrode-electrode spacing
        and the rough receptive field of a clinical bipolar pair.
    threshold
        Local density above which a pair is labeled fibrotic.
    healthy_name, fibrotic_name
        Names for the two integer codes.
    """

    radius_mm: float = 2.0
    threshold: float = 0.1
    healthy_name: str = "healthy"
    fibrotic_name: str = "fibrotic"
    name: str = "local_density"
    type: Literal["local_density"] = "local_density"

    def __post_init__(self) -> None:
        if self.radius_mm <= 0:
            raise ValueError("radius_mm must be positive.")
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError("threshold must be in [0, 1).")

    def apply(self, result: SimulationResult) -> tuple[npt.NDArray[np.int64], dict[int, str]]:
        mask = result.substrate_mask
        if mask.ndim != 2:
            raise ValueError(
                "LocalDensityLabel requires a 2-D substrate mask; "
                f"got {mask.ndim}-D. Use a 3D-aware label policy with 3D geometries."
            )
        dr_mm = float(result.substrate_mask_dr_mm)
        midpoints = result.bipolar_pair_midpoints_mm  # (n_pairs, 3)

        # Build a coordinate grid over the substrate mask in mm.
        # Axis 0 (i) → y in mm; axis 1 (j) → x in mm (matches the
        # CenteredGrid2D position convention in specs.py).
        n_i, n_j = mask.shape
        j_idx, i_idx = np.meshgrid(np.arange(n_j), np.arange(n_i), indexing="xy")
        node_x_mm = j_idx.astype(np.float64) * dr_mm
        node_y_mm = i_idx.astype(np.float64) * dr_mm

        r2_max = self.radius_mm * self.radius_mm

        labels = np.empty(result.n_pairs, dtype=np.int64)
        for pair_idx in range(result.n_pairs):
            mx, my, _mz = midpoints[pair_idx]
            dx = node_x_mm - mx
            dy = node_y_mm - my
            in_circle = (dx * dx + dy * dy) <= r2_max
            # Restrict to interior cells (mask != 0).
            local = mask[in_circle & (mask != 0)]
            if local.size == 0:
                # Circle missed every interior cell — defensive zero
                # label. This shouldn't happen with the Phase-1 5x5
                # centered grid + 40 mm patch, but a smaller radius or
                # off-center grid could trigger it.
                labels[pair_idx] = 0
                continue
            n_fibrotic = int(np.count_nonzero(local == 2))
            local_density = n_fibrotic / local.size
            labels[pair_idx] = 1 if local_density > self.threshold else 0

        return labels, {0: self.healthy_name, 1: self.fibrotic_name}
