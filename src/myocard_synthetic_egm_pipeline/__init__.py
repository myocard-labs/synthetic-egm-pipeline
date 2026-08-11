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
  samples noise per trace at a target SNR, emits a noise-mixed ClassifierBank.

CLIs are wired in ``[project.scripts]``; see ``docs/usage.md`` for the
end-user surface and ``project/architecture.md`` for the design.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Package version, read from installed distribution metadata (CL-117) and so
# single-sourced from pyproject.toml's [project] version rather than restated
# here. The hardcoded literal this replaced had drifted to "0.2.0" against a
# v0.3.0 tag — and since this value is stamped into every bank as
# `producer_version` (the reproducibility field CL-109 argued to keep), every
# artifact written before this fix claims a version that never produced it.
#
# The fallback is deliberately an obviously-wrong sentinel rather than a
# plausible number: an uninstalled source tree should be *visibly* unversioned
# in provenance, not quietly mislabeled — that mislabeling is the bug here.
try:
    __version__ = _pkg_version("myocard-synthetic-egm-pipeline")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"
