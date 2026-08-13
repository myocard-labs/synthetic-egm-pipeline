"""FinitewaveBackend — Aliev-Panfilov 2D solver + our EGM tracker forward calc.

This file is the only place in the repo that imports ``finitewave``
(Guardrail 1 in ``project/architecture.md``). It translates the four
public strategy specs into Finitewave's native API:

- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.Patch2DGeometry`
  → :class:`finitewave.CardiacTissue2D` with anisotropic diffusion
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.UniformRandomFibrosis`
  → :class:`finitewave.DiffusePattern` applied to the tissue mesh
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.PlanarEdgeStimulus`
  → :class:`finitewave.StimVoltageCoord2D` wrapped in a
  :class:`finitewave.StimSequence`
- :class:`~myocard_synthetic_egm_pipeline.simulate.specs.CenteredGrid2D`
  → :class:`~myocard_synthetic_egm_pipeline.backends.finitewave.egm_kernel.EGMTracker`,
  our subclass of Finitewave's tracker. It streams the pseudo-EGM during
  the AP integration, as the stock tracker does, but computes it with a
  kernel we own — see that module for why vendoring was necessary.

Each translation lives in a private ``_apply_*`` helper. The public
:meth:`FinitewaveBackend.simulate` method wires them together, runs
the solver, and returns a
:class:`~myocard_synthetic_egm_pipeline.simulate.result.RawSimulationResult`.

Phase-1 limitations:

- Only ``Patch2DGeometry`` is supported. ``AtrialMesh3D`` would need
  a different Finitewave model (``CardiacTissue3D``) and is out of
  scope for v0.2.0.
- Only ``UniformRandomFibrosis`` is supported. Future substrate
  strategies wire into ``_apply_substrate_2d``.
- Only ``PlanarEdgeStimulus`` is supported. Future activation sources
  wire into ``_install_activation_2d``.
"""

from __future__ import annotations

from typing import Any

import finitewave as fw
import numpy as np
import numpy.typing as npt

from myocard_synthetic_egm_pipeline.backends import RunConfig, SimulationBackend
from myocard_synthetic_egm_pipeline.backends.finitewave.egm_kernel import EGMTracker
from myocard_synthetic_egm_pipeline.simulate.result import RawSimulationResult
from myocard_synthetic_egm_pipeline.simulate.specs import (
    ActivationSource,
    CenteredGrid2D,
    ElectrodePlacement,
    GeometrySpec,
    Patch2DGeometry,
    PlanarEdgeStimulus,
    SubstrateStrategy,
    UniformRandomFibrosis,
)

# Aliev-Panfilov stability defaults (Finitewave's recommended pair).
_AP_DT_MODEL_UNITS: float = 0.01
"""Integration step in AP model time units. 0.01 is Finitewave's
recommended pairing with ``dr = 0.25`` for stable propagation."""

_AP_DR_MODEL_UNITS: float = 0.25
"""AP model's internal dr (non-dimensional). NOT the same as
PatchSpec.dr_mm — they're independent: dr_mm is the physical mesh
resolution, this is the solver's discretization scale."""

_FINITEWAVE_VERSION_TAG: str = "0.9"
"""Pinned via pyproject (``finitewave>=0.9``). Stamped into
backend_metadata so the bank records which backend it came from."""


class FinitewaveBackend(SimulationBackend):
    """Concrete backend wrapping Finitewave's AlievPanfilov2D solver.

    Instantiate once; reuse across simulations — the backend itself
    holds no per-simulation state.
    """

    name: str = "finitewave"

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
        # --- 1. Dispatch ----------------------------------------------------
        if geometry.type != "patch_2d":
            raise ValueError(
                f"FinitewaveBackend supports only 'patch_2d' geometry; got {geometry.type!r}."
            )
        if electrodes.type != "centered_grid_2d":
            raise ValueError(
                "FinitewaveBackend supports only 'centered_grid_2d' electrode placement; "
                f"got {electrodes.type!r}."
            )

        # Strategy types are checked at runtime via the dispatch in the
        # private helpers; isinstance narrows the typing here so the
        # rest of this method sees concrete types.
        assert isinstance(geometry, Patch2DGeometry)
        assert isinstance(electrodes, CenteredGrid2D)

        # --- 2. Build the AP model and the tissue ---------------------------
        model = fw.AlievPanfilov2D()
        model.dt = _AP_DT_MODEL_UNITS
        model.dr = _AP_DR_MODEL_UNITS

        tissue = _build_tissue_2d(geometry)
        _configure_anisotropy_2d(model, geometry)

        # --- 3. Apply substrate (mutates tissue.mesh in place) --------------
        substrate_meta = _apply_substrate_2d(tissue=tissue, strategy=substrate, rng=rng)

        # Snapshot the realised substrate before the solver runs. The
        # mask survives into SimulationResult so label policies can
        # query it (mesh int8: 0=boundary, 1=healthy, 2=fibrotic).
        substrate_mask: npt.NDArray[np.int8] = tissue.mesh.astype(np.int8, copy=True)

        # Attach the tissue to the model. From here on, model.cardiac_tissue
        # is the source of truth for what the solver sees.
        model.cardiac_tissue = tissue

        # --- 4. Install activation ------------------------------------------
        _install_activation_2d(model=model, source=activation, tissue=tissue)

        # --- 5. Set t_max and install the EGM tracker -----------------------
        # The capture, not the trace: with cropping configured the solver must
        # run past the end of the trace so a window placed around the
        # activation has signal behind it (see simulate.sizing).
        t_max_model_units = config.effective_capture_duration_ms / config.ap_time_unit_ms
        model.t_max = t_max_model_units

        capture_step, fs_capture_hz = _pick_capture_step(
            dt_model_units=_AP_DT_MODEL_UNITS,
            output_fs_hz=config.output_fs_hz,
            oversample=config.capture_oversample,
            ap_time_unit_ms=config.ap_time_unit_ms,
        )

        # The tracker takes electrode positions in mesh-index units
        # (coords_grid = coord_mm / dr_mm). We subclass Finitewave's rather
        # than replacing it wholesale: streaming during the solve is why it
        # was chosen, and computing phi_e ourselves would mean holding the
        # whole V_m history (~504 MB per simulation at the production
        # geometry). Only the kernel arithmetic is ours.
        #
        # NOTE: these coords are still in OUR (x, y, z) order, which the
        # kernel differences against (i, j) — the transpose of our convention.
        # That mismatch is deliberate at this step so the vendored kernel
        # reproduces the stock tracker byte-for-byte; S39 fixes it.
        coords_grid = electrodes.positions_mm / geometry.dr_mm
        egm_tracker = EGMTracker(measure_coords=coords_grid)
        egm_tracker.step = capture_step
        tracker_seq = fw.TrackerSequence()
        tracker_seq.add_tracker(egm_tracker)
        model.tracker_sequence = tracker_seq

        # --- 6. Run ---------------------------------------------------------
        model.run()

        # --- 7. Collect unipolar traces ------------------------------------
        # tracker.output shape: (n_capture, n_electrodes).
        unipolar = np.asarray(egm_tracker.output, dtype=np.float64)

        # --- 8. Assemble result --------------------------------------------
        backend_metadata: dict[str, Any] = {
            "backend_name": self.name,
            "finitewave_version_pin": _FINITEWAVE_VERSION_TAG,
            "ap_dt_model_units": float(_AP_DT_MODEL_UNITS),
            "ap_dr_model_units": float(_AP_DR_MODEL_UNITS),
            "ap_time_unit_ms": float(config.ap_time_unit_ms),
            "t_max_model_units": float(t_max_model_units),
            "capture_step_integration": int(capture_step),
            "fs_capture_hz": float(fs_capture_hz),
            "model_class": "AlievPanfilov2D",
        }

        return RawSimulationResult(
            unipolar_traces=unipolar,
            fs_capture_hz=fs_capture_hz,
            substrate_mask=substrate_mask,
            substrate_mask_dr_mm=geometry.dr_mm,
            electrode_positions_mm=electrodes.positions_mm.copy(),
            bipolar_pairs=electrodes.bipolar_pairs,
            substrate_realization_metadata=substrate_meta,
            backend_metadata=backend_metadata,
        )


# ---------------------------------------------------------------------------
# Private adapters — strategy spec → Finitewave native API
# ---------------------------------------------------------------------------


def _build_tissue_2d(geometry: Patch2DGeometry) -> fw.CardiacTissue2D:
    """Build a healthy 2D tissue patch with a uniform fiber field."""
    import math

    tissue = fw.CardiacTissue2D(shape=geometry.shape)
    n_i, n_j = tissue.mesh.shape
    fibers = np.zeros((n_i, n_j, 2), dtype=np.float64)
    fibers[:, :, 0] = math.cos(geometry.fiber_angle_rad)
    fibers[:, :, 1] = math.sin(geometry.fiber_angle_rad)
    tissue.fibers = fibers
    return tissue


def _configure_anisotropy_2d(
    model: Any, geometry: Patch2DGeometry, base_diffusion: float = 1.0
) -> None:
    """Set the AP model's along/across diffusion so CV ratio matches geometry.

    For Aliev-Panfilov in Finitewave's asymmetric stencil, the
    diffusion-tensor knobs are ``D_al`` (along-fiber) and ``D_ac``
    (across-fiber). Since :math:`CV \\propto \\sqrt{D}`, setting
    ``D_al / D_ac = ratio^2`` yields ``CV_al / CV_ac = ratio``. The
    geometric mean is held at ``base_diffusion`` so the absolute CV
    magnitude is preserved across different ratios.

    Earlier (v0.1.0) behavior was to leave the stencil at Finitewave's
    internal defaults, which produced a measured ratio of ~3.05
    regardless of ``geometry.anisotropy_ratio``. This helper closes
    that gap.
    """
    import math

    ratio = geometry.anisotropy_ratio
    sqrt_ratio = math.sqrt(ratio)
    model.D_al = float(base_diffusion * sqrt_ratio)
    model.D_ac = float(base_diffusion / sqrt_ratio)


def _apply_substrate_2d(
    *,
    tissue: fw.CardiacTissue2D,
    strategy: SubstrateStrategy,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Dispatch on strategy type; mutate ``tissue.mesh`` in place.

    Returns realization metadata (requested density, realised density,
    fibrotic node count, etc.) for downstream provenance + labelling.
    """
    if strategy.type == "uniform_random_fibrosis":
        assert isinstance(strategy, UniformRandomFibrosis)
        return _apply_uniform_random_fibrosis_2d(tissue=tissue, strategy=strategy, rng=rng)
    raise ValueError(f"FinitewaveBackend has no adapter for substrate type {strategy.type!r}.")


def _apply_uniform_random_fibrosis_2d(
    *,
    tissue: fw.CardiacTissue2D,
    strategy: UniformRandomFibrosis,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Realize UniformRandomFibrosis via Finitewave's DiffusePattern."""
    if strategy.density == 0.0:
        return {
            "density_requested": 0.0,
            "density_realized": 0.0,
            "n_fibrotic_nodes": 0,
            "strategy_type": strategy.type,
        }

    n_i, n_j = tissue.mesh.shape
    pattern = fw.DiffusePattern(
        density=strategy.density,
        x1=1,
        x2=n_i - 1,
        y1=1,
        y2=n_j - 1,
    )
    # Finitewave's DiffusePattern uses the global ``np.random`` state.
    # Seed it from our reproducible RNG and restore the prior state so
    # we don't bleed determinism into the caller.
    seed = int(rng.integers(0, 2**31 - 1))
    prior_state = np.random.get_state()
    try:
        np.random.seed(seed)
        tissue.add_pattern(pattern)
    finally:
        np.random.set_state(prior_state)

    n_fibrotic = int(np.count_nonzero(tissue.mesh == 2))
    n_interior = int(np.count_nonzero(tissue.mesh != 0))
    density_realized = n_fibrotic / max(n_interior, 1)
    return {
        "density_requested": float(strategy.density),
        "density_realized": float(density_realized),
        "n_fibrotic_nodes": n_fibrotic,
        "strategy_type": strategy.type,
    }


def _install_activation_2d(
    *,
    model: Any,
    source: ActivationSource,
    tissue: fw.CardiacTissue2D,
) -> None:
    """Dispatch on activation type; attach a StimSequence to the model."""
    if source.type == "planar_edge":
        assert isinstance(source, PlanarEdgeStimulus)
        stim = _build_planar_edge_stimulus_2d(source=source, mesh_shape=tissue.mesh.shape)
        stim_seq = fw.StimSequence()
        stim_seq.add_stim(stim)
        model.stim_sequence = stim_seq
        return
    raise ValueError(f"FinitewaveBackend has no adapter for activation type {source.type!r}.")


def _build_planar_edge_stimulus_2d(
    *,
    source: PlanarEdgeStimulus,
    mesh_shape: tuple[int, int],
) -> fw.StimVoltageCoord2D:
    """Build the Finitewave stimulus for the given mesh shape.

    Edge naming (top / bottom / left / right) follows Finitewave's mesh
    convention. See ``specs.PlanarEdgeStimulus`` docstring for the axis
    convention; the geometry is unchanged from the legacy code.
    """
    n_i, n_j = mesh_shape
    t = source.strip_thickness
    if source.edge == "top":
        x1, x2, y1, y2 = 1, t + 1, 1, n_j - 1
    elif source.edge == "bottom":
        x1, x2, y1, y2 = n_i - 1 - t, n_i - 1, 1, n_j - 1
    elif source.edge == "left":
        x1, x2, y1, y2 = 1, n_i - 1, 1, t + 1
    else:  # "right"
        x1, x2, y1, y2 = 1, n_i - 1, n_j - 1 - t, n_j - 1
    return fw.StimVoltageCoord2D(
        time=source.time_model_units,
        volt_value=source.voltage,
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
    )


def _pick_capture_step(
    *,
    dt_model_units: float,
    output_fs_hz: float,
    oversample: int,
    ap_time_unit_ms: float,
) -> tuple[int, float]:
    """Choose the tracker's integration-step stride and report the achieved capture rate.

    The tracker captures every ``step`` integration steps. We choose
    ``step`` so the capture rate is approximately
    ``oversample * output_fs_hz``; the exact achieved rate is reported
    back so the runner's downsample step uses it.

    Returns
    -------
    (step, achieved_capture_fs_hz)
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    target_capture_fs_hz = oversample * output_fs_hz
    # Time per integration step in ms.
    dt_ms = dt_model_units * ap_time_unit_ms
    step_f = (1000.0 / target_capture_fs_hz) / dt_ms
    step = max(1, round(step_f))
    achieved_capture_fs_hz = 1000.0 / (step * dt_ms)
    return step, achieved_capture_fs_hz
