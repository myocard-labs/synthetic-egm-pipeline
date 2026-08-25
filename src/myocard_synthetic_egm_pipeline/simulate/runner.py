"""Thin per-simulation orchestrator.

:func:`run_single` is the public entry point that wires the four
strategy specs into a chosen backend, then post-processes the
backend's ``RawSimulationResult`` into the runner-side
:class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationResult`
that label policies and storage consume.

:func:`run_probe_sweep` is its sibling for the positional-sensitivity probe
(SEP13): one solve, **one result per crop offset**. It returns a list rather
than being a mode on ``run_single``, because the return type genuinely differs
and a conditional one would push the branch onto every caller. Both share the
capture and the result-building below, so the two cannot drift on what a
simulation's post-processing is.

The orchestration is intentionally small:

1. Call ``backend.simulate(...)`` — backend owns the V_m → φ_e step
   internally (see ``project/architecture.md`` for why).
2. Form bipolar pairs from the per-electrode unipolar traces at the
   capture rate.
3. Downsample to the output rate (typically 1 kHz).
4. Truncate / pad to the exact target sample count.
5. Compute per-pair midpoints in physical mm so locality-aware label
   policies don't have to re-derive them.
6. Stamp per-run provenance (geometry / substrate / activation /
   electrode type, plus per-sim sampled scalars like ``stim_edge`` and
   ``electrode_height_mm``) into ``run_metadata``.

This module imports no backend code (Guardrail 1) — it operates on the
:class:`~myocard_synthetic_egm_pipeline.backends.SimulationBackend`
Protocol and the public strategy types only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # Backend types are used only in annotations; importing them at
    # runtime would close the cycle backends → simulate.result →
    # simulate → simulate.runner → backends. ``from __future__ import
    # annotations`` makes all annotations strings so this is safe.
    from myocard_egm_signal import ActivationPositionGenerator, DetectionPreprocessor

    from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend

from myocard_synthetic_egm_pipeline.simulate.cell_models import CellModelSpec
from myocard_synthetic_egm_pipeline.simulate.cropping import crop_traces
from myocard_synthetic_egm_pipeline.simulate.probe import ProbeGrid, sweep_capture
from myocard_synthetic_egm_pipeline.simulate.pseudo_egm import (
    bipolar_from_unipolar,
    downsample,
)
from myocard_synthetic_egm_pipeline.simulate.result import SimulationResult, SimulationSpecs
from myocard_synthetic_egm_pipeline.simulate.specs import (
    ActivationSource,
    ElectrodePlacement,
    GeometrySpec,
    SubstrateStrategy,
)


def run_single(
    *,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    cell_model: CellModelSpec,
    backend: SimulationBackend,
    config: RunConfig,
    rng: np.random.Generator,
    position_generator: ActivationPositionGenerator | None = None,
    detection_preprocessor: DetectionPreprocessor | None = None,
) -> SimulationResult:
    """Run one simulation and return a finished :class:`SimulationResult`.

    Parameters
    ----------
    geometry, substrate, activation, electrodes
        Four of the five strategy specs; the backend's adapters translate
        them into the backend's native representation.
    cell_model
        The fifth spec (design note D2) — the membrane kinetics each node
        runs. Handed to the backend *and* recorded on the result's
        :class:`~myocard_synthetic_egm_pipeline.simulate.result.SimulationSpecs`,
        which is where the bank serializer reads the model's identity
        from (S18a). It used to read the class name the backend reported
        instead.
    backend
        Concrete :class:`SimulationBackend`. The runner is generic over
        which backend ran.
    config
        Per-run knobs (duration, output rate, AP calibration constant,
        oversampling factor).
    rng
        Reproducible source of randomness for any per-sim sampling the
        backend's strategy adapters do (e.g. fibrosis pattern draw).
    position_generator
        egm-signal's position generator (SEP2). When supplied, each bipolar
        trace is cropped to a ``T``-sample window with its activation at a
        sampled fractional position; when ``None`` the trace is the first
        ``T`` samples of the capture, as before cropping existed.

        Deliberately **not** on :class:`RunConfig`: the generator is stateful
        (it owns an rng) while ``RunConfig`` is a frozen value object handed to
        the backend, so putting it there would let two runs sharing a config
        silently share one random stream.
    detection_preprocessor
        The detection curve the crop anchors on (S16a);
        :func:`~myocard_synthetic_egm_pipeline.simulate.cropping.default_preprocessor`
        when ``None``. Read only when ``position_generator`` is set — with no
        crop there is nothing to detect for.

        ``crop_traces`` has always taken this and the runner never passed it,
        so every bank in the project's history was windowed with
        ``RectifiedDerivative`` and no config could say otherwise.

        **Not recorded in either bank — FB-35.** The curve decides *where the
        window is cut*, so it changes the stored waveform, and neither schema
        has anywhere to put it. Until FB-35 lands the bank's ``description`` is
        the record, maintained by hand; it is deliberately not smuggled into
        ``backend_metadata``, which describes the simulator's capture and would
        read as authoritative about something it does not know.
    """
    captured, raw = _capture_bipolar(
        geometry=geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        cell_model=cell_model,
        backend=backend,
        config=config,
        rng=rng,
    )
    target_samples = _target_samples(config)

    # --- 5. Place the window (SEP2) -------------------------------------
    # With a position policy the trace is a T-sample window cut around each
    # pair's *detected* activation; without one it is the leading T samples,
    # which is what the producer did before cropping existed.
    activation_positions: npt.NDArray[np.float64] | None = None
    if position_generator is None:
        bipolar_traces = captured[:, :target_samples]
    else:
        cropped = crop_traces(
            traces=captured,
            position_generator=position_generator,
            window_length_samples=target_samples,
            preprocessor=detection_preprocessor,
        )
        bipolar_traces = cropped.signals
        activation_positions = cropped.realized_positions

    return _build_result(
        raw=raw,
        geometry=geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        cell_model=cell_model,
        config=config,
        bipolar_traces=bipolar_traces,
        activation_positions=activation_positions,
    )


def run_probe_sweep(
    *,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    cell_model: CellModelSpec,
    backend: SimulationBackend,
    config: RunConfig,
    rng: np.random.Generator,
    grid: ProbeGrid,
    detection_preprocessor: DetectionPreprocessor | None = None,
) -> list[SimulationResult]:
    """Solve once, then return **one result per grid point** (SEP13 / S16b).

    A probe emits N simulations that differ only in where the window was cut.
    Reusing a single solve is what makes that cheap; it is not what makes them
    one simulation. Each returned result is an ordinary
    :class:`SimulationResult` with ``n_pairs`` traces, so ``pair_index`` means
    what the schema says it means and nothing downstream needs a probe-shaped
    branch.

    A separate function rather than a mode on :func:`run_single`: the return
    type genuinely differs, and a conditional ``SimulationResult |
    list[SimulationResult]`` would push that branch onto every caller.

    The caller gives every returned result the **same** ``sim_seed`` — they came
    from one solve — and consecutive ``simulation_id`` values. The schema
    carries ``seed`` per simulation and does not require it to be unique, so the
    shared seed is what identifies a sweep.
    """
    captured, raw = _capture_bipolar(
        geometry=geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        cell_model=cell_model,
        backend=backend,
        config=config,
        rng=rng,
    )
    _target_samples(config)  # same short-capture guard the single path applies

    swept = sweep_capture(
        traces=captured,
        grid=grid,
        preprocessor=detection_preprocessor,
    )
    n_pairs = captured.shape[0]
    return [
        _build_result(
            raw=raw,
            geometry=geometry,
            substrate=substrate,
            activation=activation,
            electrodes=electrodes,
            cell_model=cell_model,
            config=config,
            bipolar_traces=swept.signals[point_index],
            # One value per logical simulation, repeated across its pairs
            # because the column is per trace. Not a tiling of a longer axis:
            # every trace in this simulation genuinely sits at this offset.
            activation_positions=np.full(n_pairs, position, dtype=np.float64),
        )
        for point_index, position in enumerate(swept.positions)
    ]


def _capture_bipolar(
    *,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    cell_model: CellModelSpec,
    backend: SimulationBackend,
    config: RunConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], Any]:
    """Steps 1-4: solve, pair, downsample, and hand back ``(n_pairs, n_capture)``.

    Shared by :func:`run_single` and :func:`run_probe_sweep` so the two cannot
    drift on what "the capture" means.
    """
    raw = backend.simulate(
        geometry=geometry,
        substrate=substrate,
        activation=activation,
        electrodes=electrodes,
        cell_model=cell_model,
        config=config,
        rng=rng,
    )

    # --- 1. Bipolar pairing at the capture rate -------------------------
    bipolar_capture = bipolar_from_unipolar(raw.unipolar_traces, raw.bipolar_pairs)

    # --- 2. Downsample to the output rate -------------------------------
    bipolar_target = downsample(
        bipolar_capture,
        source_fs_hz=raw.fs_capture_hz,
        target_fs_hz=config.output_fs_hz,
    )

    # --- 3. Truncate to exact target samples ----------------------------
    # The downsample step's sample count is approximate (integer rounding
    # of duration_ms * fs_hz). We pin to the exact target so every trace
    # in a dataset shares T.
    #
    # A capture that comes in SHORT is a hard error, not something to pad.
    # This used to zero-pad "so the bank stays uniform", which is uniform in
    # the worst way: zeros are perfectly flat and always at the tail, so a
    # padded trace carries a positional regularity a classifier can key on —
    # exactly the shortcut controlled-position cropping (T1) exists to
    # remove. It is the same argument egm-classifier makes in CL-112 for
    # cropping rather than padding at the dataset boundary. Under correct
    # sizing (simulate.sizing) this branch is unreachable; if it fires, the
    # sizing is wrong and silently manufacturing data would hide that.
    target_samples = _target_samples(config)
    n = bipolar_target.shape[0]
    if n < target_samples:
        raise ValueError(
            f"Capture produced {n} samples at {config.output_fs_hz} Hz but the trace "
            f"needs {target_samples} (trace_duration_ms={config.trace_duration_ms}). "
            f"The backend simulated {config.effective_capture_duration_ms} ms. "
            "Increase capture_duration_ms (or the position range's lower bound, "
            "which drives it) rather than accepting a short trace."
        )

    # --- 4. Reshape (n_samples, n_pairs) → (n_pairs, n_samples) + float32 ---
    captured: npt.NDArray[np.float32] = bipolar_target.T.astype(np.float32, copy=False)
    return captured, raw


def _target_samples(config: RunConfig) -> int:
    """``T`` in samples for this run."""
    return round(config.trace_duration_ms * 1e-3 * config.output_fs_hz)


def _build_result(
    *,
    raw: Any,
    geometry: GeometrySpec,
    substrate: SubstrateStrategy,
    activation: ActivationSource,
    electrodes: ElectrodePlacement,
    cell_model: CellModelSpec,
    config: RunConfig,
    bipolar_traces: npt.NDArray[np.float32],
    activation_positions: npt.NDArray[np.float64] | None,
) -> SimulationResult:
    """Steps 5-7: midpoints, run-level provenance, and the finished result."""
    # --- 5. Per-pair midpoints (physical mm) ----------------------------
    midpoints = _compute_pair_midpoints_mm(
        positions_mm=raw.electrode_positions_mm,
        bipolar_pairs=raw.bipolar_pairs,
    )

    # --- 6. Run-level provenance ----------------------------------------
    # Duck-typed extraction of per-sim scalars from the strategy specs.
    # Concretes that don't expose the field get None; downstream
    # consumers (ClassifierBank trace_metadata, SyntheticBank columns)
    # handle the None case.
    # Per-pair row index for downstream per-trace metadata. For
    # placements with a row_col helper (CenteredGrid2D) we use it;
    # other placements get None per pair.
    electrode_row_per_pair: list[int | None] = _electrode_row_per_pair(
        electrodes=electrodes, bipolar_pairs=raw.bipolar_pairs
    )

    run_metadata: dict[str, Any] = {
        "geometry_type": geometry.type,
        "substrate_type": substrate.type,
        "activation_type": activation.type,
        "electrode_type": electrodes.type,
        "electrode_height_mm": getattr(electrodes, "height_mm", None),
        "electrode_n_rows": getattr(electrodes, "n_rows", None),
        "electrode_n_cols": getattr(electrodes, "n_cols", None),
        "electrode_spacing_mm": getattr(electrodes, "spacing_mm", None),
        "electrode_row_per_pair": electrode_row_per_pair,
        "stim_edge": getattr(activation, "edge", None),
        "fiber_angle_rad": getattr(geometry, "fiber_angle_rad", None),
        "anisotropy_ratio": getattr(geometry, "anisotropy_ratio", None),
        "patch_size_mm": getattr(geometry, "size_mm", None),
        "patch_dr_mm": getattr(geometry, "dr_mm", None),
        "fibrosis_density_requested": getattr(substrate, "density", None),
        "backend_metadata": dict(raw.backend_metadata),
    }

    return SimulationResult(
        bipolar_traces=bipolar_traces,
        fs_hz=config.output_fs_hz,
        trace_duration_ms=config.trace_duration_ms,
        bipolar_pair_midpoints_mm=midpoints,
        substrate_mask=raw.substrate_mask,
        substrate_mask_dr_mm=raw.substrate_mask_dr_mm,
        electrode_positions_mm=raw.electrode_positions_mm,
        bipolar_pairs=raw.bipolar_pairs,
        # The realized specs, kept as objects rather than flattened.
        # ``run_metadata`` above is a lossy view of the same facts;
        # ``synthetic_bank`` 2.0's per-simulation config serializes from
        # these (see result.SimulationSpecs).
        specs=SimulationSpecs(
            geometry=geometry,
            substrate=substrate,
            activation=activation,
            electrodes=electrodes,
            cell_model=cell_model,
        ),
        activation_positions=activation_positions,
        substrate_realization_metadata=dict(raw.substrate_realization_metadata),
        run_metadata=run_metadata,
    )


def _electrode_row_per_pair(
    *,
    electrodes: ElectrodePlacement,
    bipolar_pairs: tuple[tuple[int, int], ...],
) -> list[int | None]:
    """Best-effort row lookup per bipolar pair.

    Placements that expose a ``row_col(electrode_index) -> (row, col)``
    helper (Phase 1's :class:`CenteredGrid2D`) get a concrete row index
    per pair (taken from the pair's first electrode). Future placements
    that don't define rows (e.g. an endocardial-surface 3D placement)
    get ``None`` per pair; downstream consumers fall back to
    ``electrode_positions_mm`` for spatial context.
    """
    row_col = getattr(electrodes, "row_col", None)
    if row_col is None:
        return [None] * len(bipolar_pairs)
    return [int(row_col(a)[0]) for a, _b in bipolar_pairs]


def _compute_pair_midpoints_mm(
    *,
    positions_mm: npt.NDArray[np.float64],
    bipolar_pairs: tuple[tuple[int, int], ...],
) -> npt.NDArray[np.float64]:
    """``(n_pairs, 3)`` array of midpoint coordinates per bipolar pair.

    Phase 1 assumes Cartesian midpoints (linear average of the two
    pole positions). 3D placements on a curved surface (Phase 5) may
    want a geodesic midpoint; when that lands, the
    :class:`ElectrodePlacement` Protocol grows a
    ``bipolar_pair_midpoints_mm()`` method and concretes override.
    """
    n_pairs = len(bipolar_pairs)
    midpoints = np.empty((n_pairs, 3), dtype=np.float64)
    for pair_idx, (a, b) in enumerate(bipolar_pairs):
        midpoints[pair_idx] = 0.5 * (positions_mm[a] + positions_mm[b])
    return midpoints
