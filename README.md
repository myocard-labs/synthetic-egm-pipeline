# myocard-synthetic-egm-pipeline

> Finitewave-driven synthetic intracardiac-EGM producer for the myocard-labs stack. Aliev-Panfilov on a 2D atrial patch, pseudo-EGM forward calc through a 5x5 electrode grid, optional IAFDB noise overlay.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal-processing toolkit.

---

## Why

Real intracardiac EGM data with ground-truth fibrosis labels is hard to come by. Catheter-based mapping gives you bipolar voltage but not a verified substrate map; explanted-heart studies have the substrate but not the EGMs. The standard workaround is **simulation**: solve cardiac electrophysiology on a synthetic substrate with known fibrosis content, project the resulting V_m field onto electrode positions via a pseudo-EGM forward calc, and produce labelled training data at any scale.

`myocard-synthetic-egm-pipeline` is the Phase-1 implementation of that pipeline for the myocard-labs project. It uses [Finitewave](https://github.com/TiNeZ-Lab/Finitewave) to run the Aliev-Panfilov model on a 2 mm-anisotropic 4 cm square patch with uniform-random fibrosis substrate, captures bipolar EGMs through a centered 5x5 electrode grid (20 bipolar pairs per simulation), and labels each pair via a configurable `LabelPolicy` (global density or local-density-around-pair). The mixer then overlays IAFDB noise segments from [`myocard-iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline) at a target SNR distribution, producing noise-mixed synthetic training data — synthetic EGMs carrying real-recorded IAFDB noise (not a synthetic+real *hybrid* dataset in the Sánchez sense; the electrograms are still 100% synthetic).

What this repo does NOT do: the AP solver itself (Finitewave owns that), the HDF5 writers (egm-data), the schema definitions (egm-contracts), the shared DSP primitives like bandpass (egm-signal), the IAFDB noise extraction (iafdb-pipeline), or the classifier (egm-classifier). The split lets the producer stay focused on one job — converting simulator output into bank-format artifacts that satisfy the project-wide contracts.

The architecture is designed so that swapping the simulator backend (Finitewave → openCARP / TorchCor for 3D unstructured atrial meshes) becomes a contained refactor when 3D anatomy becomes load-bearing. See [`project/architecture.md`](project/architecture.md) for the design rationale and the three guardrails that protect that property.

---

## Install

From source during pre-1.0 iteration:

```bash
pip install "myocard-synthetic-egm-pipeline @ git+https://github.com/myocard-labs/synthetic-egm-pipeline.git"
```

Editable install for development:

```bash
git clone https://github.com/myocard-labs/synthetic-egm-pipeline.git
cd synthetic-egm-pipeline
pip install -e ".[dev]"
pre-commit install
```

The runtime deps (`myocard-egm-contracts`, `myocard-egm-data`, `myocard-egm-signal`, `pydantic`, `numpy`, `scipy`, `finitewave`, `tqdm`, `pyyaml`) come in transitively. The three myocard siblings are pinned to git tags during pre-1.0; drop the direct references once they publish to PyPI.

---

## Quick start

End-to-end: generate a clean dataset, then overlay IAFDB noise.

```bash
# 1. Generate 100-sim clean dataset (Phase 1 default)
synthegm-generate-dataset examples/synthegm_v1_baseline.yaml

# 2. Mix in IAFDB noise at the default 10-25 dB SNR range
synthegm-mix examples/synthegm_mix.yaml
```

Or in one step:

```bash
synthegm-generate-dataset examples/synthegm_v1_noise_mixed.yaml
```

Two console scripts are installed:

| Command | Purpose |
|---|---|
| `synthegm-generate-dataset` | Run an N-simulation Finitewave dataset; optionally overlay noise inline. |
| `synthegm-mix` | Standalone mixer — overlay noise on an already-written clean ClassifierBank. |

Both are YAML-config-driven; the YAML schema and per-CLI walkthrough live in [`docs/usage.md`](docs/usage.md). Pre-written configs covering the Phase 1 baseline, the noise-mixed path, a calibration run, and a fixed-SNR ablation live under [`examples/`](examples/).

**A generation run always writes two banks:** the `ClassifierBank` you train on and the `synthetic_bank` carrying θ plus the per-simulation generation config. They are parallel artifacts joined on `simulation_id`, and the ClassifierBank names its partner in its `banks` list — there is no flag to disable either. `synthegm-mix` is the exception: it post-processes an existing ClassifierBank and has no access to the generation config, so it writes only a ClassifierBank.

The mixer consumes a `noise_bank.h5` produced by [`myocard-iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline)'s `iafdb-export-noise-bank` CLI — install + produce a noise bank from that repo before running the noise-mixed path.

---

## Stable bank IDs

Every bank carries a stable cross-artifact ID (an egm-contracts `ArtifactId`) so the intracardiac-platform phase manifests and provenance tracking can refer to it.

A run writes **two** banks, and they take **distinct** IDs from one base — the phase manifest keys on stable IDs, so two files sharing one would collide. The base is derived from the cell model; the ClassifierBank keeps it, the `synthetic_bank` gets a `theta` marker, and a noise-mixed run inserts `noise_mixed`. All markers go **before** the date, so the result stays inside the `ArtifactId` grammar:

| Run | ClassifierBank | `synthetic_bank` |
|---|---|---|
| clean | `tbank_synthetic_aliev_panfilov_2026-06-27` | `tbank_synthetic_aliev_panfilov_theta_2026-06-27` |
| noise-mixed | `…_noise_mixed_2026-06-27` | `…_noise_mixed_theta_2026-06-27` |

`output.bank_id` overrides the **base**, so one setting names the whole family and keeps it in step. The mixer's "noise source" provenance entry carries the noise bank's own ID, read from the iafdb noise run-record sidecar. See [`docs/usage.md`](docs/usage.md) for the full reference.

---

## Programmatic usage

The orchestrators are importable when you'd rather drive the producer from Python:

```python
from pathlib import Path
import numpy as np

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
from myocard_synthetic_egm_pipeline.simulate import (
    DatasetConfig, GlobalDensityLabel, Patch2DGeometry,
    generate_dataset, write_classifier_bank_from_dataset,
)

config = DatasetConfig(
    n_simulations=10,
    geometry=Patch2DGeometry(),
    label_policy=GlobalDensityLabel(threshold=0.1),
    run_config=RunConfig(trace_duration_ms=200.0, output_fs_hz=1000.0, ap_time_unit_ms=1.97),
    fibrosis_density_range=(0.0, 0.5),
    fraction_healthy=0.3,
    master_seed=42,
    show_progress=True,
)
result = generate_dataset(config=config, backend=FinitewaveBackend())
write_classifier_bank_from_dataset(
    dataset_result=result, config=config,
    output_path=Path("out/synthegm_dev.classifier.h5"), overwrite=True,
)
```

See [`docs/usage.md`](docs/usage.md) for the mixer orchestrator, the strategy spec reference, and the consumer-side bank-reading recipe.

---

## Tests

```bash
pytest                  # default suite — MockBackend only, ~1 s
pytest -m slow          # the real-Finitewave tests (deselected by default)
pytest -m ""            # everything, both groups
pytest --cov            # with coverage
ruff check .            # lint
ruff format --check .   # format check
mypy                    # type check (src + tests)
```

**Two groups.** The default run uses a `MockBackend` for the orchestrator tests, so it finishes in
about a second without starting a solver — which is what makes it cheap enough to run on every save.
A handful of tests marked `slow` spin up the **real Finitewave backend**; they are deselected by
default (`addopts = ["-m", "not slow"]` in `pyproject.toml`) and you will see them reported as
`deselected`. They are the ones that check the pipeline end to end rather than the plumbing — most
notably the schema-migration equivalence check — so **run `pytest -m slow` before opening a PR**, not
just the fast suite.

CI runs the same checks on Python 3.10, 3.11, and 3.12 — see `.github/workflows/ci.yml`.

---

## Project status

This package is part of the in-progress [myocard-labs](https://github.com/myocard-labs) refactor. Pre-1.0 — expect breaking changes across minor versions until the schemas + strategy Protocols stabilise. The current release is `v0.3.0`; `development` pins `egm-contracts v0.6.0`, `egm-data v0.6.0`, and `egm-signal v0.2.0`. See [`project/roadmap.md`](project/roadmap.md) for what's planned (NoiseSelectionStrategy, multi-substrate-type composition, compatibility validator) and [`project/architecture.md`](project/architecture.md) for the design.

---

## Citation

If you use this software in academic work, please cite both the Finitewave simulator + the project's Phase-1 method:

```bibtex
@software{finitewave,
  author = {Nezlobinsky, Timur and Vandersickel, Nele and Panfilov, Alexander V.},
  title  = {Finitewave: a finite-difference cardiac electrophysiology simulator},
  url    = {https://github.com/TiNeZ-Lab/Finitewave},
}

@software{klein_myocard_synthetic_egm_pipeline_2026,
  author  = {Klein, Daniel},
  title   = {myocard-synthetic-egm-pipeline: Finitewave-driven synthetic intracardiac-EGM producer for the myocard-labs stack},
  year    = {2026},
  url     = {https://github.com/myocard-labs/synthetic-egm-pipeline},
}
```

The Phase 1 substrate + EGM-forward-calc approach is closest to Okenov et al. 2024 (PLOS ONE); the additive-noise mixing approach is closest to Sánchez et al. 2021 (Frontiers in Physiology). Full reading list in [`docs/simulation_theory.md`](docs/simulation_theory.md) and [`docs/mixer_theory.md`](docs/mixer_theory.md).

---

## License

MIT — see [LICENSE](LICENSE). Attribution requirements for Finitewave + the IAFDB dataset (when consumed via the mixer) are listed in [NOTICE](NOTICE).
