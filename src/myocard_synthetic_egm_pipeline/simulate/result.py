"""Public result dataclasses passed between the backend, runner, and labeler.

Two types per ``project/architecture.md`` Guardrail 2 — both public,
both concrete, both stable across the Option A→B migration:

- :class:`RawSimulationResult` is what a backend produces. It carries
  per-electrode unipolar EGMs at the backend's native time grid plus
  enough spatial metadata for the runner to form bipolar pairs and for
  label policies to compute neighborhood statistics. Backends are
  responsible for the V_m → φ_e forward calc internally — Finitewave
  uses its built-in ECG tracker; future backends without that infra
  can import :func:`~myocard_synthetic_egm_pipeline.simulate.pseudo_egm.compute_phi_e`
  as a shared helper. Backend-internal types (Finitewave's
  ``CardiacTissue2D``, openCARP's mesh objects, etc.) stay inside the
  backend — none of them leak into this dataclass.

- :class:`SimulationResult` is what the runner exposes. It carries the
  finished bipolar traces at the output sample rate plus the substrate
  mask, electrode positions, and bipolar-pair midpoints needed for any
  reasonable :class:`~myocard_synthetic_egm_pipeline.simulate.label_policy.LabelPolicy`
  to do its job without re-running the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from myocard_synthetic_egm_pipeline.simulate.specs import (
        ActivationSource,
        ElectrodePlacement,
        GeometrySpec,
        SubstrateStrategy,
    )


@dataclass(frozen=True)
class SimulationSpecs:
    """The strategy specs a single simulation was **actually run with**.

    Every field holds the concrete, *realized* spec object — the one the
    backend received — not the configured range it was drawn from. For
    Phase 1 that distinction is the whole point of this type: the
    dataset orchestrator samples a fibrosis density, an activation edge
    and an electrode height per simulation, builds
    :class:`~myocard_synthetic_egm_pipeline.simulate.specs.UniformRandomFibrosis`
    / ``PlanarEdgeStimulus`` / ``CenteredGrid2D`` from them, and those
    objects are the only record of what that simulation actually was.

    Before ``synthetic_bank`` 2.0 they were discarded after
    :func:`~myocard_synthetic_egm_pipeline.simulate.runner.run_single`
    returned, and a handful of duck-typed scalars were copied into
    ``run_metadata`` in their place. 2.0 serializes the specs themselves
    as typed per-function objects, one set per simulation, so the
    objects have to survive.

    **Guardrail 2 note.** This adds a field to the concrete, public
    :class:`SimulationResult` — a *widening*: every existing reader
    keeps working untouched and only the runner (the sole producer of
    the type) changes. The four strategy Protocols are not modified. See
    ``project/architecture.md`` → Guardrails.
    """

    geometry: GeometrySpec
    substrate: SubstrateStrategy
    activation: ActivationSource
    electrodes: ElectrodePlacement


@dataclass(frozen=True)
class RawSimulationResult:
    """Backend output, pre-bipolar pairing.

    Attributes
    ----------
    unipolar_traces
        ``(T_capture, n_electrodes)`` float64 — per-electrode unipolar
        pseudo-EGMs at the backend's native capture rate. Backends own
        the V_m → φ_e forward calc internally; the runner consumes
        these traces to form bipolar pairs and downsample to the
        output rate.
    fs_capture_hz
        Time resolution of ``unipolar_traces`` (capture rate before
        downsampling to the output rate).
    substrate_mask
        ``(n_i, n_j)`` int8 array. ``1`` = healthy interior node,
        ``2`` = fibrotic / non-conductive, ``0`` = boundary frame.
        Forwarded through to ``SimulationResult`` so a label policy
        can compute neighborhood statistics without re-simulating.
    substrate_mask_dr_mm
        Spatial resolution of ``substrate_mask`` (mm per cell). Comes
        from the geometry spec; needed for any locality query in mm.
    electrode_positions_mm
        ``(n_electrodes, 3)`` float64 — electrode positions in physical
        mm, NOT in backend-native grid units. Kept for downstream label
        policies and offline analysis.
    bipolar_pairs
        Tuple of ``(a_idx, b_idx)`` index pairs into
        ``electrode_positions_mm``. Comes from the
        :class:`~myocard_synthetic_egm_pipeline.simulate.specs.ElectrodePlacement`
        used to build the recorder.
    substrate_realization_metadata
        Per-strategy realization metadata returned by the backend's
        substrate adapter (realized density, fibrotic node count, etc.).
        Pass-through for downstream labelling and provenance.
    backend_metadata
        Backend-specific provenance — version, solver settings, capture
        rate, anything the backend wants stamped into the bank's
        metadata blob.
    """

    unipolar_traces: npt.NDArray[np.float64]
    fs_capture_hz: float
    substrate_mask: npt.NDArray[np.int8]
    substrate_mask_dr_mm: float
    electrode_positions_mm: npt.NDArray[np.float64]
    bipolar_pairs: tuple[tuple[int, int], ...]
    substrate_realization_metadata: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    """Runner output — bipolar traces ready for labeling and storage.

    The runner consumes :class:`RawSimulationResult` from the backend,
    applies the Okenov pseudo-EGM forward calc, forms bipolar pairs,
    downsamples to the output sample rate, and emits this. Every
    :class:`~myocard_synthetic_egm_pipeline.simulate.label_policy.LabelPolicy`
    and every storage writer operates on this type.

    Attributes
    ----------
    bipolar_traces
        ``(n_pairs, T_samples)`` float32 — finished bipolar traces at
        ``fs_hz``. Pair order matches ``bipolar_pair_midpoints_mm``.
    fs_hz
        Output sample rate of ``bipolar_traces`` (typically 1000.0).
    trace_duration_ms
        Realized per-trace duration (= target duration after
        downsampling, modulo dt rounding).
    bipolar_pair_midpoints_mm
        ``(n_pairs, 3)`` float64 — midpoint of each bipolar pair in
        physical mm. Used by locality-aware label policies (e.g.
        ``LocalDensityLabel``) to query the substrate mask in a
        neighborhood without re-deriving the geometry.
    substrate_mask
        ``(n_i, n_j)`` int8 — forwarded from
        :class:`RawSimulationResult` so a label policy can compute
        any neighborhood statistics it needs.
    substrate_mask_dr_mm
        Spatial resolution of ``substrate_mask`` (mm per cell).
    electrode_positions_mm
        ``(n_electrodes, 3)`` float64 — kept for offline analysis and
        for any future placement-aware label policy.
    bipolar_pairs
        Pass-through of ``RawSimulationResult.bipolar_pairs``.
    specs
        The realized :class:`SimulationSpecs` this simulation ran with —
        the typed source of truth that ``synthetic_bank`` 2.0's
        per-simulation config is serialized from. ``run_metadata``'s
        duck-typed scalars are a flattened, lossy view of the same
        facts, kept for backwards compatibility with existing readers.
    substrate_realization_metadata
        Pass-through — realized fibrosis metadata.
    run_metadata
        Run-level provenance (stim edge, electrode height range, seed,
        AP calibration constant, fiber angle, etc.) stamped by the
        runner. Distinct from ``backend_metadata`` (backend-internal
        knobs) which the runner forwards into the same blob via the
        storage layer.
    """

    bipolar_traces: npt.NDArray[np.float32]
    fs_hz: float
    trace_duration_ms: float
    bipolar_pair_midpoints_mm: npt.NDArray[np.float64]
    substrate_mask: npt.NDArray[np.int8]
    substrate_mask_dr_mm: float
    electrode_positions_mm: npt.NDArray[np.float64]
    bipolar_pairs: tuple[tuple[int, int], ...]
    specs: SimulationSpecs
    substrate_realization_metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.bipolar_traces.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.bipolar_traces.shape[1])
