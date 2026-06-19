"""Clean synthetic EGM generation.

Public API re-exports from the backend-agnostic modules of the
simulator layer:

- :mod:`specs` — strategy Protocols + Phase-1 concretes.
- :mod:`result` — RawSimulationResult, SimulationResult.
- :mod:`pseudo_egm` — Okenov forward calc + bipolar pairing + downsample.
- :mod:`label_policy` — LabelPolicy Protocol + concrete policies.
- :mod:`runner` — :func:`run_single` per-simulation orchestrator.
- :mod:`dataset` — :func:`generate_dataset` N-sim orchestrator.
- :mod:`storage` — ClassifierBank (default) + optional SyntheticBank writers.
"""

from __future__ import annotations

from myocard_synthetic_egm_pipeline.simulate.builders import (
    AMP_TYPE,
    build_classifier_bank_from_dataset,
    build_clean_trace_metadata,
    build_synthetic_bank_from_classifier,
    build_synthetic_bank_from_dataset,
)
from myocard_synthetic_egm_pipeline.simulate.dataset import (
    DatasetConfig,
    DatasetResult,
    generate_dataset,
)
from myocard_synthetic_egm_pipeline.simulate.label_policy import (
    GlobalDensityLabel,
    LabelPolicy,
    LocalDensityLabel,
)
from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import (
    bipolar_from_unipolar,
    compute_phi_e,
    downsample,
)
from myocard_synthetic_egm_pipeline.simulate.result import (
    RawSimulationResult,
    SimulationResult,
)
from myocard_synthetic_egm_pipeline.simulate.runner import run_single
from myocard_synthetic_egm_pipeline.simulate.specs import (
    EDGES,
    ActivationSource,
    CenteredGrid2D,
    Edge,
    ElectrodePlacement,
    GeometrySpec,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    SubstrateStrategy,
    UniformRandomFibrosis,
    random_edge,
)
from myocard_synthetic_egm_pipeline.simulate.storage import (
    write_classifier_bank_from_dataset,
    write_synthetic_bank_from_dataset,
)

__all__ = [
    "AMP_TYPE",
    "EDGES",
    "ActivationSource",
    "CenteredGrid2D",
    "DatasetConfig",
    "DatasetResult",
    "Edge",
    "ElectrodePlacement",
    "GeometrySpec",
    "GlobalDensityLabel",
    "LabelPolicy",
    "LocalDensityLabel",
    "Patch2DGeometry",
    "PlanarEdgeStimulus",
    "RawSimulationResult",
    "SimulationResult",
    "SubstrateStrategy",
    "UniformRandomFibrosis",
    "bipolar_from_unipolar",
    "build_classifier_bank_from_dataset",
    "build_clean_trace_metadata",
    "build_synthetic_bank_from_classifier",
    "build_synthetic_bank_from_dataset",
    "compute_phi_e",
    "downsample",
    "generate_dataset",
    "random_edge",
    "run_single",
    "write_classifier_bank_from_dataset",
    "write_synthetic_bank_from_dataset",
]
