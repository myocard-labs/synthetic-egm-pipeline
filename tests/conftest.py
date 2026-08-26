"""Shared fixtures for the synthetic-egm-pipeline test suite.

Hand-built lightweight fixtures — no Finitewave, no disk I/O. The
substrate_mask + simulation_result fixtures use a small 16x16 mesh
(physical 4 mm patch at dr = 0.25 mm) with a concentrated fibrotic
patch so locality-aware tests have deterministic ground truth.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_contracts._generated.python.noise_bank import (
    SchemaVersion as NoiseSchemaVersion,
)
from myocard_egm_contracts._generated.python.noise_bank import Traces as NoiseTraces
from myocard_egm_data.banks import ClassifierBank, ClassifierBankMetaData, ClassifierTrace

from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend
from myocard_synthetic_egm_pipeline.constants import DEFAULT_TRACE_DURATION_MS
from myocard_synthetic_egm_pipeline.simulate import (
    CenteredGrid2D,
    DatasetConfig,
    DatasetResult,
    GlobalDensityLabel,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    RawSimulationResult,
    SimulationResult,
    SimulationSpecs,
    UniformRandomFibrosis,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import AlievPanfilovCellModel
from myocard_synthetic_egm_pipeline.simulate.specs import Edge

# ---------------------------------------------------------------------------
# Simulation specs
# ---------------------------------------------------------------------------


#: The cell model the hand-built fixtures below ran with. Its numbers match
#: what those fixtures already claim in ``backend_metadata``
#: (``ap_time_unit_ms`` 1.97, ``ap_dt_model_units`` 0.01) rather than the
#: shipped card's, so the fixture stays internally consistent — a fixture that
#: disagreed with itself about the model is how a previous one went wrong.
FIXTURE_CELL_MODEL = AlievPanfilovCellModel(
    time_unit_ms=1.97,
    diffusion=1.0,
    eps=0.002,
    dt_model_units=0.01,
)


#: Trace length used by every fixture below, in samples at 1 kHz. Tied to the
#: shipped default so the suite exercises the value real runs use — a fixture
#: on a different T would still pass while hiding a T-dependent defect, and
#: would fall foul of the `T % 64 == 0` guard the config loader applies.
TRACE_SAMPLES: int = round(DEFAULT_TRACE_DURATION_MS)


@pytest.fixture
def minimal_specs() -> SimulationSpecs:
    """A throwaway :class:`SimulationSpecs` for tests that don't care about it.

    ``SimulationResult.specs`` is required (it is what ``synthetic_bank``
    2.0's per-simulation config serializes from), but a label-policy test
    exercises the mask and the midpoints only. Deliberately has no default
    on the dataclass itself: a result that silently omits its specs is
    exactly the bug the field exists to prevent.
    """
    return SimulationSpecs(
        geometry=Patch2DGeometry(size_mm=4.0, dr_mm=0.25),
        substrate=UniformRandomFibrosis(density=0.0),
        activation=PlanarEdgeStimulus(edge="top"),
        electrodes=CenteredGrid2D(
            n_rows=1,
            n_cols=5,
            spacing_mm=2.0,
            height_mm=0.5,
            positions_mm=np.zeros((8, 3), dtype=np.float64),
            bipolar_pairs=tuple((i, i + 1) for i in range(4)),
        ),
        cell_model=FIXTURE_CELL_MODEL,
    )


# ---------------------------------------------------------------------------
# Small mesh + spatial fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def substrate_mask_with_patch() -> npt.NDArray[np.int8]:
    """16x16 substrate mask with a concentrated fibrotic patch in the
    upper-left interior.

    At dr_mm = 0.25, the mesh is 4 mm physical. Fibrotic interior
    cells sit at i,j in [2, 5] → physical y,x in [0.5, 1.25] mm. The
    rest of the interior is healthy. The single-row boundary frame
    (mask == 0) sits at i,j in {0, 15}.
    """
    mask = np.ones((16, 16), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    mask[2:6, 2:6] = 2
    return mask


@pytest.fixture
def healthy_substrate_mask() -> npt.NDArray[np.int8]:
    """16x16 substrate mask with NO fibrotic nodes (all interior healthy)."""
    mask = np.ones((16, 16), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    return mask


@pytest.fixture
def four_pair_midpoints() -> npt.NDArray[np.float64]:
    """4 bipolar pair midpoints — two over the fibrotic patch, two far away.

    Designed to interact deterministically with
    `substrate_mask_with_patch`:
    - pair 0: (1.0, 1.0) mm — squarely over the fibrotic patch
    - pair 1: (0.75, 1.0) mm — adjacent to the patch
    - pair 2: (3.0, 3.0) mm — far from the patch
    - pair 3: (3.5, 0.5) mm — far from the patch in the opposite corner
    Z = 0.5 mm for all (catheter height).
    """
    return np.array(
        [
            [1.0, 1.0, 0.5],
            [0.75, 1.0, 0.5],
            [3.0, 3.0, 0.5],
            [3.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# SimulationResult / DatasetResult / ClassifierBank fixtures
# ---------------------------------------------------------------------------


def _make_simulation_result(
    *,
    bipolar_traces: npt.NDArray[np.float32],
    fs_hz: float,
    midpoints: npt.NDArray[np.float64],
    mask: npt.NDArray[np.int8],
    dr_mm: float,
    density_realized: float,
    simulation_id: int = 0,
    sim_seed: int = 42,
    stim_edge: str = "top",
    electrode_height_mm: float = 0.5,
    fibrosis_density_requested: float = 0.3,
) -> SimulationResult:
    """Build a SimulationResult with all the per-trace metadata fields
    a downstream builder/labeller expects."""
    n_pairs = bipolar_traces.shape[0]
    # Electrode positions are not used by label policies but the runner
    # forwards them; populate with anything sensible.
    electrode_positions = np.zeros((n_pairs * 2, 3), dtype=np.float64)
    bipolar_pairs = tuple((i * 2, i * 2 + 1) for i in range(n_pairs))
    # The grid must describe the SAME electrodes as `bipolar_pairs` above:
    # pairs are (0,1), (2,3), ... so one pair per row of a 2-column grid.
    # An n_rows=1 grid here would disagree with its own pair list, and the
    # per-pair electrode_row written into the bank would silently differ
    # from the one the runner computes.
    geometry = Patch2DGeometry(size_mm=40.0, dr_mm=dr_mm)
    specs = SimulationSpecs(
        geometry=geometry,
        substrate=UniformRandomFibrosis(density=fibrosis_density_requested),
        activation=PlanarEdgeStimulus(edge=cast(Edge, stim_edge)),
        electrodes=CenteredGrid2D(
            n_rows=n_pairs,
            n_cols=2,
            spacing_mm=2.0,
            height_mm=electrode_height_mm,
            positions_mm=electrode_positions,
            bipolar_pairs=bipolar_pairs,
        ),
        cell_model=FIXTURE_CELL_MODEL,
    )
    return SimulationResult(
        bipolar_traces=bipolar_traces,
        fs_hz=fs_hz,
        trace_duration_ms=float(bipolar_traces.shape[1] * 1000.0 / fs_hz),
        bipolar_pair_midpoints_mm=midpoints,
        substrate_mask=mask,
        substrate_mask_dr_mm=dr_mm,
        electrode_positions_mm=electrode_positions,
        bipolar_pairs=bipolar_pairs,
        specs=specs,
        substrate_realization_metadata={
            "density_realized": density_realized,
            "n_fibrotic_nodes": int(density_realized * 1000),
        },
        run_metadata={
            "simulation_id": simulation_id,
            "sim_seed": sim_seed,
            "stim_edge": stim_edge,
            "electrode_height_mm": electrode_height_mm,
            "fibrosis_density_requested": fibrosis_density_requested,
            # Matches the 2-column grid above: one pair per row.
            "electrode_row_per_pair": list(range(n_pairs)),
            "backend_metadata": {
                "backend_name": "mock",
                "finitewave_version_pin": "0.9.3",
                "ap_dt_model_units": 0.01,
                # Neither of these is read for identity any more: the
                # cell model comes from `specs.cell_model`. They stay here on
                # purpose — this fixture is the one producer that still emits
                # them, so `backend_model`'s promise to keep them out of
                # `params` is tested against something rather than nothing.
                "ap_time_unit_ms": 1.97,
                "model_class": "AlievPanfilov2D",
            },
        },
    )


@pytest.fixture
def four_pair_simulation_result(
    substrate_mask_with_patch: npt.NDArray[np.int8],
    four_pair_midpoints: npt.NDArray[np.float64],
) -> SimulationResult:
    """SimulationResult sized for the locality-aware label-policy tests."""
    bipolar = np.random.default_rng(0).standard_normal((4, TRACE_SAMPLES)).astype(np.float32)
    return _make_simulation_result(
        bipolar_traces=bipolar,
        fs_hz=1000.0,
        midpoints=four_pair_midpoints,
        mask=substrate_mask_with_patch,
        dr_mm=0.25,
        density_realized=0.15,
    )


@pytest.fixture
def small_dataset_result(
    substrate_mask_with_patch: npt.NDArray[np.int8],
    four_pair_midpoints: npt.NDArray[np.float64],
) -> DatasetResult:
    """3-sim DatasetResult, 4 pairs per sim, 192-sample traces at 1 kHz."""
    rng = np.random.default_rng(0)
    results = [
        _make_simulation_result(
            bipolar_traces=rng.standard_normal((4, TRACE_SAMPLES)).astype(np.float32),
            fs_hz=1000.0,
            midpoints=four_pair_midpoints,
            mask=substrate_mask_with_patch,
            dr_mm=0.25,
            density_realized=0.05 * (i + 1),
            simulation_id=i,
            sim_seed=42 + i,
            stim_edge=["top", "bottom", "left"][i],
            fibrosis_density_requested=0.05 * (i + 1),
        )
        for i in range(3)
    ]
    # Per-sim labels: sim 0 → all 0, sim 1 → all 1, sim 2 → all 0.
    labels_per_sim = [
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
    ]
    labels = np.concatenate(labels_per_sim)
    simulation_ids = np.concatenate([np.full(4, i, dtype=np.int64) for i in range(3)])
    pair_indices = np.concatenate([np.arange(4, dtype=np.int64) for _ in range(3)])
    seeds = np.array([42, 43, 44], dtype=np.int64)
    return DatasetResult(
        results=results,
        labels=labels,
        labels_dict={0: "healthy", 1: "fibrotic"},
        simulation_ids=simulation_ids,
        pair_indices=pair_indices,
        seeds=seeds,
    )


@pytest.fixture
def small_dataset_config() -> DatasetConfig:
    """DatasetConfig matching small_dataset_result's per-sim parameters."""
    return DatasetConfig(
        n_simulations=3,
        geometry=Patch2DGeometry(size_mm=4.0, dr_mm=0.25),
        label_policy=GlobalDensityLabel(threshold=0.1),
        run_config=RunConfig(
            trace_duration_ms=DEFAULT_TRACE_DURATION_MS,
            output_fs_hz=1000.0,
        ),
        fibrosis_density_range=(0.0, 0.5),
        fraction_healthy=0.3,
        master_seed=0,
        show_progress=False,
    )


_FIXTURE_BANK_ID = "tbank_synthetic_aliev_panfilov_2026-06-27"
"""Stable id on the fixture bank (egm-data v0.4.0+ requires a string id)."""


@pytest.fixture
def small_classifier_bank() -> ClassifierBank:
    """A tiny ClassifierBank ready for mixer tests.

    4 traces, all 192 samples at 1 kHz, two healthy + two fibrotic
    labels. Per-trace metadata mirrors what the producer's builder
    stamps, so the mixer + the noise_mixed SyntheticBank builder both
    accept it.
    """
    rng = np.random.default_rng(0)
    traces = []
    for i in range(4):
        simulation_id = i // 2  # two pairs per sim
        pair_idx = i % 2
        label = 0 if simulation_id == 0 else 1
        traces.append(
            ClassifierTrace(
                bank_id=_FIXTURE_BANK_ID,
                signal=rng.standard_normal(TRACE_SAMPLES).astype(np.float32),
                freq_hz=1000.0,
                amp_type="synthetic_au",
                split=None,
                label_truth=label,
                prediction=None,
                trace_metadata={
                    "simulation_id": simulation_id,
                    "pair_index": pair_idx,
                    "electrode_row": 0,
                    "fibrosis_density_requested": 0.1 * simulation_id,
                    "fibrosis_density_realized": 0.1 * simulation_id,
                    "electrode_height_mm": 0.5,
                    "stim_edge": "top",
                    "sim_seed": 42 + simulation_id,
                    "patient_id": str(simulation_id),
                },
            )
        )
    bank_meta = ClassifierBankMetaData(
        bank_id=_FIXTURE_BANK_ID,
        bank_type="synthetic_egm_pipeline",
        bank_path="<test fixture>",
        bank_metadata={
            "producer": "synthetic_egm_pipeline",
            "backend": "mock",
            "cell_model": "aliev_panfilov",
            "n_simulations": 2,
            "label_policy_name": "global_density",
            "label_policy_type": "global_density",
            "fibrosis_density_range": [0.0, 0.5],
            "fraction_healthy": 0.0,
            "geometry_size_mm": 4.0,
            "geometry_dr_mm": 0.25,
            "electrode_n_rows": 5,
            "electrode_n_cols": 5,
            "electrode_spacing_mm": 2.0,
            "electrode_height_mm_range": [0.2, 1.0],
            "ap_time_unit_ms": 1.97,
        },
    )
    return ClassifierBank(
        # A bank with no id is not a writable artifact — egm-data refuses
        # one — so the fixture carries the id a real clean bank would.
        id="tbank_synthetic_aliev_panfilov_2026-06-27",
        banks=[bank_meta],
        traces=traces,
        labels={0: "healthy", 1: "fibrotic"},
    )


# ---------------------------------------------------------------------------
# Noise bank fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def small_noise_bank() -> NoiseBank:
    """A tiny Pydantic NoiseBank with 8 segments of 200 samples at 1 kHz.

    Each segment is uncorrelated Gaussian noise; provenance fields
    track per-segment source. Matches the egm-contracts v0.2.0
    noise_bank schema.
    """
    rng = np.random.default_rng(1)
    n_segments = 8
    signal_rows = [rng.standard_normal(200).astype(np.float32).tolist() for _ in range(n_segments)]
    source_records = [f"iaf{(i % 4) + 1}_afw" for i in range(n_segments)]
    source_channels = ["CS12", "CS34", "CS56", "CS78"] * 2
    from datetime import datetime, timezone

    return NoiseBank(
        # noise_bank 1.1 (egm-contracts v0.6.0) added the root bank_id.
        schema_version=NoiseSchemaVersion.field_1_1,
        bank_id="nbank_test_fixture",
        created_utc=datetime.now(timezone.utc),
        source="test fixture",
        fs_hz=1000.0,
        traces=NoiseTraces(
            signal=signal_rows,
            source_record=source_records,
            source_channel=source_channels,
        ),
    )


# ---------------------------------------------------------------------------
# Mock backend fixture
# ---------------------------------------------------------------------------


class _MockBackend:
    """Returns a deterministic canned RawSimulationResult per call.

    Doesn't actually simulate anything — feeds the runner enough
    structure to exercise its post-processing (bipolar pairing,
    downsampling, midpoint computation, run_metadata stamping).
    """

    name: str = "mock"

    def simulate(
        self,
        *,
        geometry: Any,
        substrate: Any,
        activation: Any,
        electrodes: Any,
        cell_model: Any,
        config: Any,
        rng: np.random.Generator,
    ) -> RawSimulationResult:
        # Mirrors the real backend: it simulates the *capture* duration, which
        # exceeds the trace duration whenever a position policy is configured.
        # A mock that captured only the trace would make the sizing look
        # untested — every crop would fit because nothing extra was ever asked
        # for.
        n_capture = round(config.effective_capture_duration_ms) * config.capture_oversample
        n_electrodes = electrodes.positions_mm.shape[0]
        unipolar = rng.standard_normal((n_capture, n_electrodes)).astype(np.float64)
        return RawSimulationResult(
            unipolar_traces=unipolar,
            fs_capture_hz=float(config.capture_oversample * config.output_fs_hz),
            substrate_mask=np.ones(geometry.shape, dtype=np.int8),
            substrate_mask_dr_mm=geometry.dr_mm,
            electrode_positions_mm=electrodes.positions_mm.copy(),
            bipolar_pairs=electrodes.bipolar_pairs,
            substrate_realization_metadata={
                "density_requested": substrate.density,
                "density_realized": substrate.density,
                "n_fibrotic_nodes": 0,
                "strategy_type": substrate.type,
            },
            # Mirrors the key set FinitewaveBackend actually emits. A
            # fixture thinner than the real backend lets a mapping gap
            # pass tests — which is exactly how `version` and
            # `dt_model_units` were silently left unset.
            backend_metadata={
                "backend_name": self.name,
                "finitewave_version_pin": "0.9.3",
                "ap_dt_model_units": float(cell_model.dt_model_units),
                "ap_dr_model_units": float(config.dr_model_units),
                "capture_step_integration": 4,
                "fs_capture_hz": float(config.output_fs_hz * config.capture_oversample),
                "model_class": "AlievPanfilov2D",
            },
        )


@pytest.fixture
def mock_backend() -> SimulationBackend:
    """A no-op backend that returns canned RawSimulationResults."""
    return _MockBackend()


# ---------------------------------------------------------------------------
# A capture the detection curves disagree about
# ---------------------------------------------------------------------------


def ambiguous_complex(n_samples: int, fs_hz: float = 1000.0) -> npt.NDArray[np.float64]:
    """One trace holding three candidate activations, one per detection curve.

    Every feature is written in **milliseconds** and sampled at ``fs_hz``, so
    the waveform is the same shape whatever rate it is built at. That matters
    more than it looks: this trace goes through the runner's anti-alias filter
    and rate conversion, and a shape defined in *samples* would mean something
    different either side of them.

    Each curve anchors on a **different feature**, by the property that curve
    measures:

    - a monophasic Gaussian pulse at ``0.25 n``, sigma 1 ms — no carrier, so
      ``x * x''`` is strongly negative at its peak and **Teager-Kaiser**
      (``x'^2 - x x''``) lands here;
    - a 250 Hz burst at ``0.40 n``, sigma 2 ms — the steepest carrier crossing
      in the trace, so **rectified-derivative** lands here;
    - a broad 100 Hz burst at ``0.625 n``, sigma 6 ms — squarely inside
      Botteron's 40-250 Hz band and carrying by far the most in-band energy, so
      the **smoothed envelope** lands here.

    All three sit in the middle 25-63 % of the capture on purpose: a probe
    sweeps windows across a range of activation positions, and an anchor near
    either end leaves no room for the window at one end of that range.

    Three curves, three indices: that is what makes a bank-to-bank diff mean
    "the curve reached the crop" rather than "the seed differed".

    **The separation is structural, not sub-sample.** Measured over every
    capture length from 400 to 1200 samples, in both the paths this fixture is
    used in — built at 1 kHz and handed straight to a curve, and built at the
    capture rate and taken through the runner's band-limit plus rate conversion
    — the three indices stay on their own features with a **minimum gap of 60
    samples**, and never once collapse.

    That robustness is the point of the three-feature shape, and it was learned
    the hard way. **The first version separated the curves with a one-sample
    biphasic spike**, which worked until the anti-alias filter arrived: a
    one-sample spike at 1 kHz is a delta, its spectrum is flat past Nyquist,
    and band-limiting removes precisely the content that made it the steepest
    feature. Two curves then agreed and the fixture silently stopped testing
    what it claimed. The obvious repair — a fast band-limited burst instead of
    the delta — does not hold either: rectified-derivative peaks at the
    steepest *carrier crossing* while Teager-Kaiser peaks at the *envelope*
    maximum, and for a single burst those sit a quarter carrier period apart,
    which at a 1 kHz output is at most one sample and rounds to zero for a
    quarter of all capture lengths. Giving each curve its own feature is what
    makes the fixture depend on what the curves measure rather than on where
    the sample grid happens to fall.
    """
    t_ms = np.arange(n_samples, dtype=np.float64) * 1000.0 / fs_hz
    pulse_at = t_ms[round(0.250 * n_samples)]
    fast_at = t_ms[round(0.400 * n_samples)]
    broad_at = t_ms[round(0.625 * n_samples)]

    def envelope(centre_ms: float, sigma_ms: float) -> npt.NDArray[np.float64]:
        return np.asarray(np.exp(-0.5 * ((t_ms - centre_ms) / sigma_ms) ** 2), dtype=np.float64)

    def burst(centre_ms: float, sigma_ms: float, freq_hz: float) -> npt.NDArray[np.float64]:
        carrier = np.sin(2.0 * np.pi * freq_hz * (t_ms - centre_ms) / 1000.0)
        return np.asarray(envelope(centre_ms, sigma_ms) * carrier, dtype=np.float64)

    return np.asarray(
        2.0 * envelope(pulse_at, 1.0)
        + 1.5 * burst(fast_at, 2.0, 250.0)
        + 2.0 * burst(broad_at, 6.0, 100.0),
        dtype=np.float64,
    )


class _AmbiguousComplexBackend:
    """The mock backend with its Gaussian capture replaced by the complex.

    Wraps rather than reimplements, so the ``backend_metadata`` the runner
    forwards stays exactly what the real backend emits — a thinner fixture is
    how a mapping gap gets to pass.

    Every electrode sees the same waveform at a different amplitude, so each
    bipolar difference is a scaled copy of it. All three curves take an
    ``argmax`` of a non-negative, homogeneous transform, so scale and sign
    cannot move the detected index: every pair in the run detects the same
    sample, and the only thing that can move a window is the curve.
    """

    name = "ambiguous_complex"

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def simulate(self, **kwargs: Any) -> Any:
        raw = self._inner.simulate(**kwargs)
        n_capture, n_electrodes = raw.unipolar_traces.shape
        # Built directly at the capture rate rather than sample-and-held from
        # the output rate. The hold that used to be here produced a staircase,
        # whose images above the output Nyquist are exactly what the runner's
        # anti-alias filter now removes — so the decimated trace would no
        # longer be the waveform this fixture claims to place.
        waveform = ambiguous_complex(n_capture, fs_hz=raw.fs_capture_hz)
        amplitudes = np.arange(1, n_electrodes + 1, dtype=np.float64)
        return replace(raw, unipolar_traces=waveform[:, None] * amplitudes[None, :])


def ambiguous_complex_backend_for(inner: SimulationBackend) -> SimulationBackend:
    """Wrap a backend so its capture carries the ambiguous complex.

    A function rather than only a fixture because several tests build one per
    run inside a loop, and because the wrapper is what a test hands to
    ``generate_dataset`` directly.
    """
    return _AmbiguousComplexBackend(inner)
