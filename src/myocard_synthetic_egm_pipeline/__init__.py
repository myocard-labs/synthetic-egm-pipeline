"""myocard-synthetic-egm-pipeline — Finitewave-driven synthetic EGM producer.

Subpackages:

- :mod:`~myocard_synthetic_egm_pipeline.simulate` — clean EGM generation:
  strategy Protocols (specs), result dataclasses, pseudo-EGM forward
  calc, label policies, runner, dataset orchestrator, storage writer.
- :mod:`~myocard_synthetic_egm_pipeline.backends` — concrete simulator
  backends (Finitewave today, additional backends in the future). The
  only place in the repo that imports backend-specific libraries.
- :mod:`~myocard_synthetic_egm_pipeline.mixer` — additive bandpass-domain
  noise overlay. Reads a noise_bank.h5 (produced by iafdb-pipeline),
  samples noise per trace at a target SNR, emits a hybrid ClassifierBank.

CLIs are wired in ``[project.scripts]``; see ``docs/usage.md`` for the
end-user surface and ``project/architecture.md`` for the design.
"""

from __future__ import annotations

__version__ = "0.2.0"
