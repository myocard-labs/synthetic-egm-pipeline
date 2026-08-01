"""Noise-mixed-bank write helpers.

The mixer's default output is a noise-mixed
:class:`~myocard_egm_data.banks.ClassifierBank` with the mixer metadata
stamped into each trace's ``trace_metadata`` and a "mixer" provenance
entry appended to ``banks``. That bank goes straight through
:func:`myocard_egm_data.banks.write_classifier_bank` — no new writer
needed, and that is what this module still supports.

**What changed at ``synthetic_bank`` 2.0.** There used to be a second
path here that rebuilt a Pydantic ``SyntheticBank`` *from a noise-mixed
ClassifierBank*, reading the per-trace scalars (``stim_edge``,
``electrode_height_mm``, ``fibrosis_density_requested``, …) back out of
``trace_metadata``. Schema 2.0 moves the generation config into a typed
per-simulation group, and none of it is recoverable from per-trace
metadata — so that reconstruction is not merely harder, it is
impossible, and pretending otherwise would emit a bank whose config did
not describe the simulations that produced it.

The replacement is upstream: the **inline mixer path** builds the
noise-mixed ``synthetic_bank`` from the in-memory
:class:`~myocard_synthetic_egm_pipeline.simulate.dataset.DatasetResult`
it already holds, passing the mixed signals and per-trace noise columns
to
:func:`~myocard_synthetic_egm_pipeline.simulate.builders.build_synthetic_bank_from_dataset`.

**Consequence, deliberately not worked around.** Standalone
``synthegm-mix``, run against a bare ClassifierBank on disk, has no
access to the generation config and therefore cannot emit a
``synthetic_bank`` at all. It is a post-process over an existing bank,
not a synthetic *run*. Asking it for one is a configuration error rather
than a degraded output — see ``project/architecture.md``.

This module imports no backend code (Guardrail 1).
"""

from __future__ import annotations

__all__: list[str] = []
