"""Finitewave-based concrete backend.

The only subpackage in the repo that imports the ``finitewave``
library (Guardrail 1 in ``project/architecture.md``). Implements
:class:`SimulationBackend` against Finitewave's
:class:`AlievPanfilov2D` solver + :class:`ECG2DTracker` for the
Phase-1 spec.
"""

from __future__ import annotations

from myocard_synthetic_egm_pipeline.backends.finitewave.backend import FinitewaveBackend

__all__ = ["FinitewaveBackend"]
