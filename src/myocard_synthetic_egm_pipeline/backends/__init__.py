"""Simulator backends.

Every concrete simulator (Finitewave today, openCARP / TorchCor in
future phases) implements the :class:`SimulationBackend` Protocol. The
Protocol is intentionally small — one ``simulate()`` method that takes
the four strategy specs + a :class:`RunConfig` and returns a
:class:`~myocard_synthetic_egm_pipeline.simulate.result.RawSimulationResult`.

This subpackage is the **only place in the repo that imports backend
libraries** (Guardrail 1 in ``project/architecture.md``). Everything
else operates on the public strategy + result types.

To add a new backend:

1. Drop a subdirectory under ``backends/`` (e.g. ``backends/torchcor/``).
2. Implement ``simulate(...)`` returning a ``RawSimulationResult``.
3. Wire its ``type`` discriminator into the CLI's backend dispatch in
   ``cli/_config.py``.

The new backend's strategy adapters (substrate → mesh, activation →
stimulus, electrode placement → recorder) live entirely inside its
subdirectory. The four strategy Protocols, the runner, and the storage
layer don't change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from myocard_synthetic_egm_pipeline.simulate.result import RawSimulationResult
from myocard_synthetic_egm_pipeline.simulate.specs import (
    ActivationSource,
    ElectrodePlacement,
    GeometrySpec,
    SubstrateStrategy,
)


@dataclass(frozen=True)
class RunConfig:
    """Per-run knobs the backend needs but that aren't in the strategy specs.

    Attributes
    ----------
    trace_duration_ms
        Physical duration of the captured trace, in milliseconds.
        Phase 1 default 200 ms (one activation pass).
    output_fs_hz
        Target on-disk sample rate after downsampling. Phase 1 default
        1000 Hz to match IAFDB. The backend captures at
        ``capture_oversample`` times this rate; the runner then
        downsamples.
    ap_time_unit_ms
        Aliev-Panfilov model time-unit → physical ms calibration
        constant. Used by Finitewave-based backends to translate
        ``trace_duration_ms`` into model time units. Future backends
        with non-AP models may ignore this.
    capture_oversample
        Backend captures at ``capture_oversample * output_fs_hz``;
        runner downsamples to ``output_fs_hz``. Phase 1 default 4 —
        4x oversampling is enough margin to avoid aliasing without an
        explicit anti-alias filter at the AP membrane bandwidth.
    """

    trace_duration_ms: float
    output_fs_hz: float
    ap_time_unit_ms: float
    capture_oversample: int = 4
    capture_duration_ms: float | None = None

    def __post_init__(self) -> None:
        if self.trace_duration_ms <= 0:
            raise ValueError("trace_duration_ms must be positive.")
        if self.output_fs_hz <= 0:
            raise ValueError("output_fs_hz must be positive.")
        if self.ap_time_unit_ms <= 0:
            raise ValueError("ap_time_unit_ms must be positive.")
        if self.capture_oversample < 1:
            raise ValueError("capture_oversample must be >= 1.")
        if self.capture_duration_ms is not None:
            if self.capture_duration_ms <= 0:
                raise ValueError("capture_duration_ms must be positive.")
            if self.capture_duration_ms < self.trace_duration_ms:
                raise ValueError(
                    "capture_duration_ms "
                    f"({self.capture_duration_ms}) is shorter than trace_duration_ms "
                    f"({self.trace_duration_ms}); the capture cannot be shorter than "
                    "the trace cut from it."
                )

    @property
    def effective_capture_duration_ms(self) -> float:
        """How long the backend actually simulates.

        Separate from :attr:`trace_duration_ms`, which is what lands on disk.
        The two were one number until controlled-position cropping (SEP2): a
        window placed around the activation needs signal *after* it, so the
        solver has to run past the end of the trace it will eventually yield.
        ``None`` means "no cropping configured" and keeps the historical
        behaviour of capturing exactly the trace.
        """
        return (
            self.capture_duration_ms
            if self.capture_duration_ms is not None
            else self.trace_duration_ms
        )


@runtime_checkable
class SimulationBackend(Protocol):
    """Run one simulation; return a RawSimulationResult.

    A backend translates the four pure-data strategy specs into its
    native API, runs the AP solver, captures per-electrode unipolar
    pseudo-EGMs (using its built-in tracker when available, our
    :func:`~myocard_synthetic_egm_pipeline.simulate.pseudo_egm.compute_phi_e`
    helper otherwise), and returns the unipolar traces + spatial
    metadata as a :class:`RawSimulationResult`.
    """

    name: str
    """Discriminator value matching the YAML ``backend.type`` field."""

    def simulate(
        self,
        *,
        geometry: GeometrySpec,
        substrate: SubstrateStrategy,
        activation: ActivationSource,
        electrodes: ElectrodePlacement,
        config: RunConfig,
        rng: np.random.Generator,
    ) -> RawSimulationResult:
        """Run one simulation and return the raw result.

        The backend is responsible for:

        - Realizing the substrate on its native mesh representation.
        - Configuring anisotropy from ``geometry.anisotropy_ratio``.
        - Installing the activation source.
        - Computing per-electrode unipolar φ_e at the capture rate.
        - Returning the substrate mask and electrode positions
          unchanged (the runner forwards them into
          :class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationResult`
          so label policies can use them).
        """
        ...
