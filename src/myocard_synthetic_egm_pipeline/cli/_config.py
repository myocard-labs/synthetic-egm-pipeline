"""YAML config loading + per-CLI typed config builders.

Two CLIs share this module:

- ``synthegm-generate-dataset`` consumes a YAML config that pins every
  knob of one N-simulation run (geometry, substrate, activation,
  electrodes, label policy, backend, output, optional inline mixer).
- ``synthegm-mix`` consumes a smaller YAML that pins one mixing run
  against an already-written clean ClassifierBank.

The shared helpers (``ConfigError``, ``load_yaml``, ``_required``,
``_optional``, ``_resolve_path``) mirror the iafdb-pipeline convention:
every path is resolved against the config file's directory; missing
required fields raise a precise error; unknown ``type`` enum values
fail loudly at load time rather than crashing the runner.

Schema for both YAMLs is documented in the example configs under
``examples/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

# egm-signal owns the bandpass default (mixer block), the position policy
# (SEP2) and the detection curves (S16a). The generator is SIG1's type, not a
# local one — see _build_position_generator for why it is constructed rather
# than wrapped.
from myocard_egm_signal import (
    DEFAULT_BIPOLAR_BAND_HZ,
    DEFAULT_BOTTERON_BAND_HZ,
    DEFAULT_BOTTERON_LOWPASS_HZ,
    DetectionPreprocessor,
    UniformPositionGenerator,
)

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.constants import (
    DEFAULT_ANISOTROPY_RATIO,
    DEFAULT_ELECTRODE_GRID_COLS,
    DEFAULT_ELECTRODE_GRID_ROWS,
    DEFAULT_ELECTRODE_HEIGHT_MM_RANGE,
    DEFAULT_ELECTRODE_SPACING_MM,
    DEFAULT_FIBROSIS_DENSITY_RANGE,
    DEFAULT_OUTPUT_FS_HZ,
    DEFAULT_PATCH_DR_MM,
    DEFAULT_PATCH_SIZE_MM,
    DEFAULT_TRACE_DURATION_MS,
)
from myocard_synthetic_egm_pipeline.ids import validate_artifact_id
from myocard_synthetic_egm_pipeline.mixer import DEFAULT_SNR_DB_RANGE, MixerConfig
from myocard_synthetic_egm_pipeline.simulate import (
    EDGES,
    Edge,
    GeometrySpec,
    GlobalDensityLabel,
    LabelPolicy,
    LocalDensityLabel,
    Patch2DGeometry,
)
from myocard_synthetic_egm_pipeline.simulate.calibration import ModelCard
from myocard_synthetic_egm_pipeline.simulate.cell_models import CellModelSpec
from myocard_synthetic_egm_pipeline.simulate.cropping import (
    DETECTION_CURVES,
    build_preprocessor,
    default_preprocessor,
)
from myocard_synthetic_egm_pipeline.simulate.model_cards import (
    ModelCardError,
    load_model_card,
)
from myocard_synthetic_egm_pipeline.simulate.probe import ProbeGrid
from myocard_synthetic_egm_pipeline.simulate.sizing import (
    WINDOW_LENGTH_MULTIPLE,
    required_capture_duration_ms,
    required_stimulus_delay_ms,
    window_length_samples,
)

DEFAULT_MODEL_CARD = "af_remodelled_220ms"
"""Model card used when a config names none.

Defaulted rather than optional: a config that omits ``backend.model`` still gets
the calibrated parameterisation. Falling back to bare constants would silently
generate uncalibrated physics, which is the failure S38 exists to remove, and
"the user left out a line" is not a reason to do it.
"""


class ConfigError(ValueError):
    """Raised when a config file is malformed or missing required keys."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Resolves the parent path into the returned dict under
    ``_config_dir`` so subsequent relative paths in the doc resolve
    against the YAML's directory (same convention iafdb-pipeline and
    egm-classifier use).
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config {p} did not parse as a YAML mapping at the top level.")
    loaded["_config_dir"] = p.parent.resolve()
    return loaded


def _required(doc: dict[str, Any], *path: str) -> Any:
    """Walk ``path`` into the nested dict; raise on any missing key."""
    node: Any = doc
    for k in path:
        if not isinstance(node, dict) or k not in node:
            dotted = ".".join(path)
            raise ConfigError(f"Required config field missing: {dotted}")
        node = node[k]
    return node


def _optional(doc: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Walk ``path`` into the nested dict; return ``default`` if missing."""
    node: Any = doc
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _resolve_path(value: str | None, config_dir: Path) -> Path | None:
    """Resolve a YAML-supplied path against the config file's dir.

    ``None`` / empty string returns ``None``. Absolute paths pass
    through; relative paths resolve against ``config_dir``.
    """
    if value is None or value == "":
        return None
    p = Path(value)
    return p if p.is_absolute() else (config_dir / p).resolve()


def _required_path(doc: dict[str, Any], *path: str) -> Path:
    """Variant of :func:`_required` that resolves the result as a path."""
    cfg_dir: Path = doc["_config_dir"]
    value = _required(doc, *path)
    resolved = _resolve_path(str(value), cfg_dir)
    if resolved is None:
        dotted = ".".join(path)
        raise ConfigError(f"Required path field is empty: {dotted}")
    return resolved


def _expect_range(value: Any, *, field_path: str) -> tuple[float, float]:
    """Validate that ``value`` is a 2-element numeric list ``[lo, hi]``."""
    if not (isinstance(value, list | tuple) and len(value) == 2):
        raise ConfigError(f"{field_path} must be a two-element list [lo, hi]; got {value!r}.")
    return float(value[0]), float(value[1])


def _validated_id(value: Any, *, field_path: str) -> str | None:
    """Validate an optional config-supplied ArtifactId override at load time.

    Returns the id string unchanged (``None`` passes through). Raises
    :class:`ConfigError` with the field path on a malformed id, so a bad
    override fails fast at config-load — before an expensive simulation or
    mix run rather than at the bank write at the very end.
    """
    if value is None:
        return None
    try:
        return validate_artifact_id(str(value))
    except ValueError as exc:
        raise ConfigError(f"{field_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Strategy / policy / backend builders
# ---------------------------------------------------------------------------


def _build_geometry(doc: dict[str, Any]) -> GeometrySpec:
    """Construct a :class:`GeometrySpec` from the ``geometry:`` block."""
    block = _optional(doc, "geometry", default={}) or {}
    g_type = str(_optional(block, "type", default="patch_2d"))
    if g_type != "patch_2d":
        raise ConfigError(
            f"geometry.type must be 'patch_2d' for Phase 1; got {g_type!r}. "
            "Future 3D geometries land alongside patch_2d in a later release."
        )
    return Patch2DGeometry(
        size_mm=float(_optional(block, "size_mm", default=DEFAULT_PATCH_SIZE_MM)),
        dr_mm=float(_optional(block, "dr_mm", default=DEFAULT_PATCH_DR_MM)),
        fiber_angle_rad=float(_optional(block, "fiber_angle_rad", default=0.0)),
        anisotropy_ratio=float(
            _optional(block, "anisotropy_ratio", default=DEFAULT_ANISOTROPY_RATIO)
        ),
    )


def _build_label_policy(doc: dict[str, Any]) -> LabelPolicy:
    """Construct a :class:`LabelPolicy` from the ``label_policy:`` block."""
    block = _optional(doc, "label_policy", default={}) or {}
    p_type = str(_optional(block, "type", default="global_density"))
    if p_type == "global_density":
        return GlobalDensityLabel(
            threshold=float(_optional(block, "threshold", default=0.1)),
            healthy_name=str(_optional(block, "healthy_name", default="healthy")),
            fibrotic_name=str(_optional(block, "fibrotic_name", default="fibrotic")),
        )
    if p_type == "local_density":
        return LocalDensityLabel(
            radius_mm=float(_optional(block, "radius_mm", default=2.0)),
            threshold=float(_optional(block, "threshold", default=0.1)),
            healthy_name=str(_optional(block, "healthy_name", default="healthy")),
            fibrotic_name=str(_optional(block, "fibrotic_name", default="fibrotic")),
        )
    raise ConfigError(
        f"label_policy.type must be 'global_density' or 'local_density'; got {p_type!r}."
    )


def _build_position_generator(doc: dict[str, Any]) -> UniformPositionGenerator | None:
    """Construct the position generator from the ``activation_position:`` block.

    ``None`` when the block is absent, which means **no cropping** — the trace
    is the first ``T`` samples of the capture, as before SEP2. Cropping is
    opt-in rather than defaulted because the position policy decides the
    positional structure of every bank a run writes, and there is no default
    that is right by accident: anchored is the arm T1 suspects of enabling a
    positional shortcut, and a varied range has no natural bounds to assume.
    Both arms of the §8.9 A/B therefore name their range explicitly.

    The generator is **stateful** — it owns an rng — so it is constructed once
    per run and seeded from the run's ``master_seed``, keeping a run
    reproducible without threading a generator through every call.
    """
    block = _optional(doc, "activation_position", default=None)
    if block is None:
        return None
    if "grid" in block:
        if "low" in block or "high" in block:
            raise ConfigError(
                "activation_position sets both 'grid' and 'low'/'high', and each "
                "decides where the activation sits in the stored trace. A probe "
                "sweeps a fixed grid of offsets; low/high draw them at random per "
                "trace. Keep one: nest low/high inside 'grid' for a sweep, or drop "
                "the grid block for an ordinary run."
            )
        # The sweep supplies its own offsets, so there is no random stream.
        return None
    if "low" not in block or "high" not in block:
        raise ConfigError(
            "activation_position must set both 'low' and 'high' (use the same "
            "value twice for the fixed/anchored arm, e.g. low: 0.5, high: 0.5), "
            "or a 'grid' block for a positional-sensitivity probe."
        )
    low = float(block["low"])
    high = float(block["high"])
    if not 0.0 <= low <= high <= 1.0:
        raise ConfigError(
            f"activation_position must satisfy 0 <= low <= high <= 1; got low={low}, high={high}."
        )
    seed = _optional(block, "seed", default=None)
    master_seed = int(_optional(doc, "dataset", "master_seed", default=0))
    return UniformPositionGenerator(
        low=low, high=high, seed=master_seed if seed is None else int(seed)
    )


def _window_length_samples(doc: dict[str, Any]) -> int:
    """``T`` in samples, from the ``run:`` block's duration and rate.

    Read before ``RunConfig`` exists because the probe grid snaps against ``T``
    and the capture is then sized from the *snapped* bounds. Both callers go
    through :func:`~myocard_synthetic_egm_pipeline.simulate.sizing.window_length_samples`
    so the number cannot drift between them.
    """
    block = _optional(doc, "run", default={}) or {}
    return window_length_samples(
        trace_duration_ms=float(
            _optional(block, "trace_duration_ms", default=DEFAULT_TRACE_DURATION_MS)
        ),
        output_fs_hz=float(_optional(block, "output_fs_hz", default=DEFAULT_OUTPUT_FS_HZ)),
    )


def _build_probe_grid(doc: dict[str, Any]) -> ProbeGrid | None:
    """Construct the probe sweep from ``activation_position.grid:`` (SEP13).

    ``None`` for an ordinary run. The block nests inside ``activation_position``
    for the same reason ``detection`` does — it decides where the activation
    sits in the stored trace — and is mutually exclusive with ``low``/``high``,
    which :func:`_build_position_generator` enforces.

    The fractions are snapped here, at config load, so the run reports what it
    will actually sweep before spending a simulation on it. The snapped values
    are the grid of record (D6).
    """
    block = _optional(doc, "activation_position", "grid", default=None)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ConfigError(f"activation_position.grid must be a mapping of keys; got {block!r}.")
    missing = [key for key in ("low", "high", "n_points") if key not in block]
    if missing:
        raise ConfigError(
            f"activation_position.grid must set {', '.join(missing)}. A sweep needs a "
            "range and a point count: low, high, n_points."
        )

    n_simulations = int(_optional(doc, "dataset", "n_simulations", default=1))
    if n_simulations != 1:
        raise ConfigError(
            f"activation_position.grid runs exactly one simulation, but "
            f"dataset.n_simulations is {n_simulations}. The sweep holds morphology, "
            "substrate and seed constant so the only thing varying across its traces "
            "is the crop offset; a second simulation would vary the substrate too. "
            "Set dataset.n_simulations to 1."
        )

    if "pair_indices" in block:
        raise ConfigError(
            "activation_position.grid.pair_indices is not supported: a probe sweeps "
            "every bipolar pair. One grid point is one logical simulation, so its "
            "traces ARE that simulation's pairs and pair_index means 0..n_pairs-1. "
            "Sweeping a subset would need a per-trace pair mapping, which is the "
            "thing that made an earlier probe bank unloadable by egm-studio (it "
            "joins the bank pair on (simulation_id, pair_index) and raises on a "
            "non-unique key). Drop the key and select pairs when you analyse the "
            "bank instead."
        )

    try:
        return ProbeGrid.snapped(
            low=float(block["low"]),
            high=float(block["high"]),
            n_points=int(block["n_points"]),
            window_length_samples=_window_length_samples(doc),
        )
    except ValueError as exc:
        raise ConfigError(f"activation_position.grid: {exc}") from exc


#: Keys iafdb-pipeline's ``DetectionConfig`` carries that this side refuses, in
#: both the nested (``threshold: {rule: ...}``) and flat (``threshold_rule``)
#: spellings, so a block copied across fails whichever way it was written.
#:
#: They parameterise ``detect_activation_train`` — preprocess, threshold,
#: select, suppress, refine — over a multi-beat record. A synthetic trace holds
#: exactly one activation by construction and is detected with
#: ``detect_activation``, which is ``argmax g``: no threshold, no candidate
#: selection, no refractory rule. Accepting them would add config surface that
#: provably does nothing, which is the unreachable-seam defect S16a exists to
#: remove, freshly minted.
_MULTI_ACTIVATION_DETECTION_KEYS: tuple[str, ...] = (
    "threshold",
    "threshold_rule",
    "threshold_c",
    "threshold_lam",
    "threshold_q",
    "min_prominence",
    "refractory_ms",
    "refine",
    "refine_curve",
    "refine_radius_ms",
)


def _reject_retired_activation_detection_block(doc: dict[str, Any]) -> None:
    """Refuse a config carrying ``activation.detection``.

    S16a shipped the block one level too high, under the ``activation:`` spec.
    In this repo ``activation:`` is the :class:`ActivationSource` — *how the
    wave is launched* — while detection is part of the **crop**, so the block
    now lives beside the position range it is used with.

    Erroring rather than ignoring: the old path silently falls back to the
    default curve, which is precisely the class of failure S16a existed to
    remove. A config written against the first spelling would keep running and
    keep producing ``rectified_derivative`` banks while reading as though it had
    asked for something else.
    """
    if _optional(doc, "activation", "detection", default=None) is not None:
        raise ConfigError(
            "activation.detection has moved to activation_position.detection. "
            "In this repo `activation:` is the activation *source* — how the wave "
            "is launched — and the detection curve is part of the crop: "
            "crop_traces builds one windower out of the preprocessor and the "
            "position generator, so they belong in one block. (iafdb-pipeline "
            "nests detection under `activation:` because there that block means "
            "activation-based windowing; this side matches its curve names and "
            "parameter names, not its block path.) Move the block down one level: "
            "activation_position: {low: ..., high: ..., detection: {curve: ...}}."
        )


def _build_detection_preprocessor(
    doc: dict[str, Any],
    *,
    crops: bool,
    output_fs_hz: float,
) -> DetectionPreprocessor | None:
    """Construct the detection curve from ``activation_position.detection:``.

    Nested inside the position block rather than beside it, because the two are
    one decision: :func:`~myocard_synthetic_egm_pipeline.simulate.cropping.crop_traces`
    builds a single ``SingleActivationWindower`` out of the preprocessor and the
    position generator. Nesting also makes the mismatch unrepresentable — a
    curve configured for a run that never crops cannot be written down, where
    the first spelling accepted it and ignored it.

    Same curve names and same parameter names as iafdb-pipeline
    (``cli/_config.py::DetectionConfig`` there), which is the part that has to
    match: a different spelling would put a translation step inside every
    cross-corpus comparison, and comparing the two corpora on a curve each was
    windowed with is the point of making it configurable at all (CL-167). The
    *path* deliberately differs — their ``activation:`` block means
    activation-based windowing, ours means the activation source.

    ``crops`` is whether this run cuts windows at all — a position range or a
    probe grid. Both detect, and **the probe must share the run's curve**: a
    sweep that characterised a bank through a different detector would be
    measuring two things at once.

    ``None`` when the run does not crop: nothing is detected, so there is no
    curve to resolve. An absent ``detection`` sub-block inside a present
    ``activation_position`` gives
    :func:`~myocard_synthetic_egm_pipeline.simulate.cropping.default_preprocessor`,
    so every config written before the knob existed keeps its meaning.

    ``output_fs_hz``, **not** the capture rate: the runner downsamples before it
    crops, so the trace the detector sees is already at the output rate. Passing
    ``fs_capture_hz`` would mis-scale Botteron's band and low-pass by the
    oversample factor and still run without complaint.
    """
    if not crops:
        return None
    block = _optional(doc, "activation_position", "detection", default=None)
    if block is None:
        return default_preprocessor()
    if not isinstance(block, dict):
        raise ConfigError(
            f"activation_position.detection must be a mapping of keys; got {block!r}."
        )

    refused = [key for key in _MULTI_ACTIVATION_DETECTION_KEYS if key in block]
    if refused:
        raise ConfigError(
            f"activation_position.detection does not accept {', '.join(refused)}: those keys "
            "belong to multi-activation detection. iafdb-pipeline needs them because "
            "an IAFDB window is cut from a multi-beat record, so its chain is "
            "preprocess -> threshold -> select -> suppress -> refine. A synthetic "
            "trace holds exactly one activation by construction and is detected with "
            "argmax over the curve, so they would have no effect here. This side "
            "accepts curve, botteron_band_hz and botteron_lowpass_hz only."
        )

    curve = str(_optional(block, "curve", default="rectified_derivative"))
    if curve not in DETECTION_CURVES:
        raise ConfigError(
            f"activation_position.detection.curve must be one of {DETECTION_CURVES}; got {curve!r}."
        )

    # Botteron's band and low-pass describe that curve alone. Accepting them
    # beside another curve would silently ignore a deliberate setting.
    botteron_keys = [k for k in ("botteron_band_hz", "botteron_lowpass_hz") if k in block]
    if curve != "botteron_envelope" and botteron_keys:
        raise ConfigError(
            f"activation_position.detection sets {', '.join(botteron_keys)} alongside "
            f"curve={curve!r}, but those apply only to curve='botteron_envelope'. "
            "The other two curves are pure sample-domain arithmetic and read no "
            "frequencies, so the setting would be silently ignored."
        )

    band_hz = _expect_range(
        _optional(block, "botteron_band_hz", default=list(DEFAULT_BOTTERON_BAND_HZ)),
        field_path="activation_position.detection.botteron_band_hz",
    )
    lowpass_hz = float(_optional(block, "botteron_lowpass_hz", default=DEFAULT_BOTTERON_LOWPASS_HZ))
    try:
        return build_preprocessor(
            curve=curve,
            fs_hz=output_fs_hz,
            botteron_band_hz=band_hz,
            botteron_lowpass_hz=lowpass_hz,
        )
    except ValueError as exc:
        # egm-signal validates the band against itself (0 < low < high) and the
        # low-pass; surface that at config load rather than mid-run.
        raise ConfigError(f"activation_position.detection: {exc}") from exc


def _stimulus_delay_ms(doc: dict[str, Any], *, position_high: float | None) -> float:
    """``activation.stimulus_delay_ms`` — explicit, else derived, else 0.

    Three cases, in precedence order:

    1. **explicit** — the config sets it; used as given;
    2. **derived** — the run crops but sets no delay: the smallest delay that
       guarantees the front budget, ``k(p_high)``. Cropping does not work without
       one, so defaulting to zero here would mean every shipped cropping config
       raises;
    3. **zero** — no crop, so nothing needs positioning and the stimulus fires
       at ``t = 0`` as it always has.

    ``position_high`` is the largest position any window will be placed at —
    the position range's upper bound, or the probe grid's largest **snapped**
    offset. ``None`` means the run does not crop.

    **What the delay is for, and what it is not for.** It positions the
    activation far enough into the capture that a window can be placed at any
    ``p`` — the same property an IAFDB record has for free by being long. It is
    *not* a way to buy pre-activation morphology: ``phi_e ∝ Σ I_m,i / r_i`` sums
    over the whole mesh, so while nothing is depolarising the lead-in is flat.
    That flatness is a realism question, deliberately not gated on here (D9).
    """
    value = _optional(doc, "activation", "stimulus_delay_ms", default=None)
    if value is not None:
        delay = float(value)
        if delay < 0:
            raise ConfigError(f"activation.stimulus_delay_ms must be >= 0; got {delay}.")
        return delay
    if position_high is None:
        return 0.0
    # Cropping configured but no explicit delay: derive the smallest delay that
    # guarantees a window at p_high has signal in front of it. Without this the
    # crop raises on every trace whose activation arrives before k(p_high) —
    # which, at any realistic conduction velocity, is all of them.
    block = _optional(doc, "run", default={}) or {}
    return required_stimulus_delay_ms(
        trace_duration_ms=float(
            _optional(block, "trace_duration_ms", default=DEFAULT_TRACE_DURATION_MS)
        ),
        output_fs_hz=float(_optional(block, "output_fs_hz", default=DEFAULT_OUTPUT_FS_HZ)),
        position_high=position_high,
    )


def _build_run_config(
    doc: dict[str, Any],
    *,
    position_low: float | None,
    stimulus_delay_ms: float = 0.0,
    model_card: ModelCard | None = None,
) -> RunConfig:
    """Construct a :class:`RunConfig` from the ``run:`` block.

    When the run crops, the capture is sized from ``position_low`` — the
    position range's lower bound, or the probe grid's smallest **snapped**
    offset — so the solver runs past the end of the trace and a window placed
    around the activation has signal behind it (see
    :mod:`~myocard_synthetic_egm_pipeline.simulate.sizing`). A probe needs no
    new arithmetic here: a fixed grid is a tighter range, not a different one.

    When a **model card** is given, the membrane knobs come from its solved
    block and the ``run:`` block may not restate them. Two sources for one
    number is how a config ends up disagreeing with the bank it produced; the
    card wins because it is the thing that carries an identity and a
    verification (:mod:`~myocard_synthetic_egm_pipeline.simulate.model_cards`).
    """
    block = _optional(doc, "run", default={}) or {}
    trace_duration_ms = float(
        _optional(block, "trace_duration_ms", default=DEFAULT_TRACE_DURATION_MS)
    )
    # V — assumed upper bound on stimulus-to-pair travel. Overridable because
    # the right value depends on conduction velocity and the substrate, and a
    # heavily fibrotic run conducts far slower than the default was sized for.
    travel_raw = _optional(block, "travel_allowance_ms", default=None)
    travel_allowance_ms = None if travel_raw is None else float(travel_raw)
    if travel_allowance_ms is not None and travel_allowance_ms <= 0:
        raise ConfigError(f"run.travel_allowance_ms must be > 0; got {travel_allowance_ms}.")
    output_fs_hz = float(_optional(block, "output_fs_hz", default=DEFAULT_OUTPUT_FS_HZ))

    t_samples = window_length_samples(
        trace_duration_ms=trace_duration_ms, output_fs_hz=output_fs_hz
    )
    if t_samples % WINDOW_LENGTH_MULTIPLE != 0:
        raise ConfigError(
            f"run.trace_duration_ms={trace_duration_ms} at {output_fs_hz} Hz gives "
            f"T={t_samples} samples, which is not a multiple of {WINDOW_LENGTH_MULTIPLE}. "
            "egm-classifier's 1D MobileViT halves the sequence six times, so an "
            f"off-grid T fails outright downstream (CL-112). Nearest valid: "
            f"{t_samples - t_samples % WINDOW_LENGTH_MULTIPLE} or "
            f"{t_samples + WINDOW_LENGTH_MULTIPLE - t_samples % WINDOW_LENGTH_MULTIPLE} samples."
        )

    capture_duration_ms = (
        required_capture_duration_ms(
            trace_duration_ms=trace_duration_ms,
            output_fs_hz=output_fs_hz,
            position_low=position_low,
            stimulus_delay_ms=stimulus_delay_ms,
            travel_allowance_ms=travel_allowance_ms,
        )
        if position_low is not None
        else None
    )

    # With no position policy the capture is just the trace, but an explicit
    # delay still has to be paid for or the wave is truncated at the far end.
    if capture_duration_ms is None and stimulus_delay_ms > 0:
        capture_duration_ms = trace_duration_ms + stimulus_delay_ms

    # A model card owns every membrane knob, so restating one in `run:` is
    # refused rather than silently overridden — the whole point of a named
    # parameterisation is that two banks bearing the name mean the same thing.
    retired = [
        key
        for key in ("ap_time_unit_ms", "diffusion", "membrane_eps", "dt_model_units")
        if key in block
    ]
    if retired:
        raise ConfigError(
            f"run.{', run.'.join(retired)} is retired. Membrane parameters are now "
            "derived from physiological targets and live on a model card named by "
            f"backend.model (default {DEFAULT_MODEL_CARD!r}) — see "
            "src/myocard_synthetic_egm_pipeline/models/. Setting them here would "
            "let two banks claim one parameterisation name with different physics, "
            "which is the failure the card exists to prevent. To use different "
            "values, write your own card and point backend.model at it."
        )

    return RunConfig(
        trace_duration_ms=trace_duration_ms,
        output_fs_hz=output_fs_hz,
        capture_oversample=int(_optional(block, "capture_oversample", default=4)),
        capture_duration_ms=capture_duration_ms,
        dr_model_units=float(_optional(block, "dr_model_units", default=0.25)),
        model_card=model_card,
    )


# ---------------------------------------------------------------------------
# generate-dataset config
# ---------------------------------------------------------------------------


BackendType = Literal["finitewave"]


@dataclass(frozen=True)
class InlineMixConfig:
    """Inline mixer block inside a generate-dataset config.

    The mixer runs against the just-generated clean ClassifierBank
    in memory, then the dataset CLI writes the noise-mixed bank to
    ``output.classifier_bank``. Set ``output.clean_intermediate``
    to write the pre-mix clean bank as a sibling artifact.
    """

    noise_bank_path: Path
    mixer_config: MixerConfig
    noise_bank_id: str | None


@dataclass(frozen=True)
class GenerateDatasetCLIConfig:
    """Typed config for ``synthegm-generate-dataset``.

    Holds the fully-constructed strategy / policy / backend / mixer
    objects ready to feed into
    :func:`~myocard_synthetic_egm_pipeline.simulate.generate_dataset`
    and the storage layer.
    """

    # Dataset orchestrator
    n_simulations: int
    master_seed: int
    show_progress: bool

    # Backend choice
    backend_type: BackendType

    # Strategy specs / per-sim sampling
    geometry: GeometrySpec
    fibrosis_density_range: tuple[float, float]
    fraction_healthy: float
    fixed_stim_edge: Edge | None
    stimulus_delay_ms: float
    electrode_n_rows: int
    electrode_n_cols: int
    electrode_spacing_mm: float
    electrode_height_mm_range: tuple[float, float]

    # Label policy + run config
    label_policy: LabelPolicy
    run_config: RunConfig
    cell_model: CellModelSpec
    """Resolved from ``backend.model``. Held here rather than on ``RunConfig``
    because it is a strategy spec (D2), not a per-run knob."""
    # Position policy for controlled-position cropping (SEP2). ``None`` means
    # no cropping was configured. Held here rather than folded into RunConfig
    # because it is stateful (it owns an rng) and RunConfig is a frozen value
    # object the backend receives — a generator on it would make two runs
    # sharing a RunConfig silently share a random stream.
    position_generator: UniformPositionGenerator | None
    probe_grid: ProbeGrid | None
    """Positional-sensitivity sweep, from ``activation_position.grid`` (S16b).

    ``None`` for an ordinary run. Mutually exclusive with
    ``position_generator``: exactly one of the two is ever set, because each
    decides where the activation sits in the stored trace.
    """
    detection_preprocessor: DetectionPreprocessor | None
    """Detection curve the crop anchors on, from ``activation_position.detection`` (S16a).

    ``None`` exactly when ``position_generator`` is ``None``: the curve lives
    *inside* the position block, so "a curve for a run that never crops" is not
    a config that can be written. Present, it is always a concrete
    preprocessor — an absent ``detection`` sub-block resolves to
    ``RectifiedDerivative`` here rather than being left for a downstream default,
    so the CLI can report which curve produced a bank without a "defaulted"
    branch. That report matters more than usual: **FB-35 leaves the curve out of
    both bank schemas**, so until it lands the run summary and the hand-written
    ``output.description`` are the only record of it.
    """

    # Output
    classifier_bank_output: Path
    clean_intermediate_output: Path | None
    # Always resolved: explicit if the config sets output.synthetic_bank,
    # otherwise derived as a sibling of the ClassifierBank. Both banks are
    # always written, so there is no 'no path' case.
    synthetic_bank_output: Path
    # The clean intermediate's own theta bank (D8: one ClassifierBank, one
    # theta partner). Resolved iff a clean intermediate is requested — with no
    # clean ClassifierBank on disk there is nothing to partner, and writing one
    # anyway would leave an orphan theta file no artifact names.
    clean_theta_output: Path | None
    description: str
    bank_id: str | None
    # Id base for the clean intermediate pair (B13). The mixed pair takes
    # `bank_id`; without a separate base the clean bank fell back to a
    # cell-model-derived id, which is the root cause of CL-143.
    clean_intermediate_bank_id: str | None

    # Optional inline mixing
    mix: InlineMixConfig | None


def _sibling_synthetic_path(classifier_bank_path: Path) -> Path:
    """``<name>.classifier.h5`` -> ``<name>.synthetic.h5``.

    ``output.synthetic_bank`` is optional; when omitted the synthetic
    bank lands beside its ClassifierBank so the pair a run produces is
    obvious on disk without the config in hand.
    """
    name = classifier_bank_path.name
    stem = (
        name[: -len(".classifier.h5")]
        if name.endswith(".classifier.h5")
        else classifier_bank_path.stem
    )
    return classifier_bank_path.with_name(f"{stem}.synthetic.h5")


def _reject_retired_synthetic_bank_flag(doc: dict[str, Any]) -> None:
    """Refuse a config still carrying ``output.also_emit_synthetic_bank``.

    The flag is retired: a synthetic run now always writes **both** the
    ClassifierBank (the source-agnostic ML artifact) and the
    ``synthetic_bank`` (theta + per-simulation provenance), joined on
    ``simulation_id``. A switch that could turn the theta artifact off
    was a switch that could silently leave the parameter-estimation work
    with nothing to read.

    Erroring rather than ignoring the key is deliberate. A retired option
    that silently does nothing is worse than one that fails loudly —
    especially here, where the failure would only surface much later, as
    a missing artifact at analysis time.
    """
    if _optional(doc, "output", "also_emit_synthetic_bank", default=None) is not None:
        raise ConfigError(
            "output.also_emit_synthetic_bank was retired in synthetic-egm-pipeline "
            "v0.4.0. Both banks are now always written for a synthetic run: the "
            "ClassifierBank for training and the synthetic_bank for theta and "
            "per-simulation provenance, joined on simulation_id. Remove the key; "
            "set output.synthetic_bank to control the path (it defaults to a "
            "sibling of output.classifier_bank)."
        )


def build_generate_dataset_config(doc: dict[str, Any]) -> GenerateDatasetCLIConfig:
    """Translate a parsed YAML dict into a typed generate-dataset config."""
    # --- dataset block --------------------------------------------------
    n_simulations = int(_required(doc, "dataset", "n_simulations"))
    if n_simulations < 1:
        raise ConfigError("dataset.n_simulations must be >= 1.")
    master_seed = int(_optional(doc, "dataset", "master_seed", default=0))
    show_progress = bool(_optional(doc, "dataset", "show_progress", default=True))

    # --- backend block --------------------------------------------------
    backend_type_raw = str(_optional(doc, "backend", "type", default="finitewave"))
    if backend_type_raw != "finitewave":
        raise ConfigError(
            f"backend.type must be 'finitewave' for Phase 1; got {backend_type_raw!r}."
        )
    backend_type: BackendType = "finitewave"

    # --- strategies + policy --------------------------------------------
    geometry = _build_geometry(doc)
    label_policy = _build_label_policy(doc)
    # Checked before the model card is solved, so a config written against the
    # first spelling fails in milliseconds rather than after the calibration.
    _reject_retired_activation_detection_block(doc)
    position_generator = _build_position_generator(doc)
    probe_grid = _build_probe_grid(doc)
    # One pair of bounds, whichever mode set them: the sizing arithmetic is the
    # same either way (a fixed grid is a tighter range, not a different one), so
    # it reads floats rather than branching on which object supplied them. The
    # probe's are the SNAPPED extremes — sizing the capture for the fractions
    # asked for would be sizing it for offsets the sweep never cuts at.
    position_low: float | None
    position_high: float | None
    if probe_grid is not None:
        position_low, position_high = probe_grid.low, probe_grid.high
    elif position_generator is not None:
        position_low, position_high = position_generator.low, position_generator.high
    else:
        position_low = position_high = None
    stimulus_delay_ms = _stimulus_delay_ms(doc, position_high=position_high)

    # The card is resolved against the geometry, because verify_solved re-runs
    # the calibration and the solve depends on the mesh pitch. A card is only
    # valid for the mesh it was solved for.
    # Defaulted, not optional. A config that names no card still gets the
    # calibrated parameterisation rather than falling back to bare constants —
    # silently generating uncalibrated physics is exactly the failure S38 exists
    # to remove, and "the user forgot a line" is not a reason to do it.
    model_reference = _optional(doc, "backend", "model", default=DEFAULT_MODEL_CARD)
    assert isinstance(geometry, Patch2DGeometry)
    try:
        model_card = load_model_card(
            str(model_reference),
            config_dir=doc.get("_config_dir"),
            dr_mm=geometry.dr_mm,
            dr_model_units=float(_optional(doc, "run", "dr_model_units", default=0.25)),
        )
    except ModelCardError as exc:
        raise ConfigError(str(exc)) from exc

    run_config = _build_run_config(
        doc,
        position_low=position_low,
        stimulus_delay_ms=stimulus_delay_ms,
        model_card=model_card,
    )
    # Built from the resolved RunConfig rather than by re-reading `run:`, so
    # there is one source for the rate the detector is told about. Botteron
    # turns it into filter coefficients, and the trace it sees is already
    # downsampled — see _build_detection_preprocessor. The position generator
    # goes in for the same reason: it is what decides whether this run crops at
    # all, and asking it beats re-deriving the answer from the doc.
    detection_preprocessor = _build_detection_preprocessor(
        doc,
        crops=position_low is not None,
        output_fs_hz=run_config.output_fs_hz,
    )

    # --- substrate block ------------------------------------------------
    substrate_block = _optional(doc, "substrate", default={}) or {}
    substrate_type = str(_optional(substrate_block, "type", default="uniform_random_fibrosis"))
    if substrate_type != "uniform_random_fibrosis":
        raise ConfigError(
            f"substrate.type must be 'uniform_random_fibrosis' for Phase 1; got {substrate_type!r}."
        )
    density_range = _expect_range(
        _optional(substrate_block, "density_range", default=list(DEFAULT_FIBROSIS_DENSITY_RANGE)),
        field_path="substrate.density_range",
    )
    fraction_healthy = float(_optional(substrate_block, "fraction_healthy", default=0.0))

    # --- activation block -----------------------------------------------
    activation_block = _optional(doc, "activation", default={}) or {}
    activation_type = str(_optional(activation_block, "type", default="planar_edge"))
    if activation_type != "planar_edge":
        raise ConfigError(
            f"activation.type must be 'planar_edge' for Phase 1; got {activation_type!r}."
        )
    fixed_edge_raw = _optional(activation_block, "fixed_edge", default=None)
    fixed_stim_edge: Edge | None
    if fixed_edge_raw is None:
        fixed_stim_edge = None
    else:
        fixed_edge_str = str(fixed_edge_raw)
        if fixed_edge_str not in EDGES:
            raise ConfigError(
                f"activation.fixed_edge must be None or one of {EDGES}; got {fixed_edge_str!r}."
            )
        # Membership check above narrows fixed_edge_str to Edge.
        fixed_stim_edge = fixed_edge_str

    # --- electrodes block -----------------------------------------------
    electrodes_block = _optional(doc, "electrodes", default={}) or {}
    electrodes_type = str(_optional(electrodes_block, "type", default="centered_grid_2d"))
    if electrodes_type != "centered_grid_2d":
        raise ConfigError(
            f"electrodes.type must be 'centered_grid_2d' for Phase 1; got {electrodes_type!r}."
        )
    electrode_n_rows = int(
        _optional(electrodes_block, "n_rows", default=DEFAULT_ELECTRODE_GRID_ROWS)
    )
    electrode_n_cols = int(
        _optional(electrodes_block, "n_cols", default=DEFAULT_ELECTRODE_GRID_COLS)
    )
    electrode_spacing_mm = float(
        _optional(electrodes_block, "spacing_mm", default=DEFAULT_ELECTRODE_SPACING_MM)
    )
    electrode_height_mm_range = _expect_range(
        _optional(
            electrodes_block,
            "height_mm_range",
            default=list(DEFAULT_ELECTRODE_HEIGHT_MM_RANGE),
        ),
        field_path="electrodes.height_mm_range",
    )

    # --- output block ---------------------------------------------------
    classifier_bank_output = _required_path(doc, "output", "classifier_bank")
    clean_intermediate_output = _resolve_path(
        _optional(doc, "output", "clean_intermediate", default=None),
        doc["_config_dir"],
    )
    _reject_retired_synthetic_bank_flag(doc)
    synthetic_bank_output = _resolve_path(
        _optional(doc, "output", "synthetic_bank", default=None),
        doc["_config_dir"],
    )
    if synthetic_bank_output is None:
        synthetic_bank_output = _sibling_synthetic_path(classifier_bank_output)
    clean_theta_output = (
        _sibling_synthetic_path(clean_intermediate_output)
        if clean_intermediate_output is not None
        else None
    )
    description = str(_optional(doc, "output", "description", default=""))
    # Optional explicit stable ids; None -> derived. Validated at load so a
    # malformed override fails before the N-sim run rather than after it.
    bank_id = _validated_id(
        _optional(doc, "output", "bank_id", default=None), field_path="output.bank_id"
    )
    clean_intermediate_bank_id = _validated_id(
        _optional(doc, "output", "clean_intermediate_bank_id", default=None),
        field_path="output.clean_intermediate_bank_id",
    )
    if clean_intermediate_bank_id is not None and clean_intermediate_output is None:
        raise ConfigError(
            "output.clean_intermediate_bank_id names a bank that is not being "
            "written; set output.clean_intermediate too, or drop the id."
        )

    # --- optional mix block --------------------------------------------
    mix: InlineMixConfig | None = None
    mix_block = _optional(doc, "mix", default=None)
    if mix_block is not None:
        mix = _build_inline_mix(mix_block, doc["_config_dir"])

    return GenerateDatasetCLIConfig(
        n_simulations=n_simulations,
        master_seed=master_seed,
        show_progress=show_progress,
        backend_type=backend_type,
        geometry=geometry,
        fibrosis_density_range=density_range,
        fraction_healthy=fraction_healthy,
        fixed_stim_edge=fixed_stim_edge,
        stimulus_delay_ms=stimulus_delay_ms,
        electrode_n_rows=electrode_n_rows,
        electrode_n_cols=electrode_n_cols,
        electrode_spacing_mm=electrode_spacing_mm,
        electrode_height_mm_range=electrode_height_mm_range,
        label_policy=label_policy,
        run_config=run_config,
        cell_model=model_card.solved,
        position_generator=position_generator,
        probe_grid=probe_grid,
        detection_preprocessor=detection_preprocessor,
        classifier_bank_output=classifier_bank_output,
        clean_intermediate_output=clean_intermediate_output,
        synthetic_bank_output=synthetic_bank_output,
        clean_theta_output=clean_theta_output,
        description=description,
        bank_id=bank_id,
        clean_intermediate_bank_id=clean_intermediate_bank_id,
        mix=mix,
    )


# ---------------------------------------------------------------------------
# Standalone mix config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixCLIConfig:
    """Typed config for ``synthegm-mix``."""

    input_classifier_bank: Path
    noise_bank_path: Path
    output_classifier_bank: Path
    mixer_config: MixerConfig
    noise_bank_id: str | None
    bank_id: str | None


def build_mix_config(doc: dict[str, Any]) -> MixCLIConfig:
    """Translate a parsed YAML dict into a typed standalone-mix config."""

    input_classifier_bank = _required_path(doc, "input", "classifier_bank")
    noise_bank_path = _required_path(doc, "input", "noise_bank")

    output_classifier_bank = _required_path(doc, "output", "classifier_bank")
    # synthegm-mix cannot write a synthetic_bank at all, so BOTH of the
    # keys that would ask it to are rejected rather than ignored. Silently
    # accepting either would leave a config that reads as though it
    # requested a theta artifact which never appears on disk.
    for retired_key in ("also_emit_synthetic_bank", "synthetic_bank"):
        if _optional(doc, "output", retired_key, default=None) is not None:
            raise ConfigError(
                f"output.{retired_key} is not supported by synthegm-mix. "
                "Standalone mixing is a post-process over a ClassifierBank on disk, and "
                "synthetic_bank 2.0's per-simulation generation config cannot be "
                "recovered from it -- a bank written here would carry a config that does "
                "not describe the simulations behind its traces. Use "
                "synthegm-generate-dataset with a `mix:` block, which still holds the "
                "DatasetResult and writes both banks."
            )

    mixer_block = _optional(doc, "mixer", default={}) or {}
    mixer_config = _build_mixer_config(mixer_block)

    # Optional explicit ids: the noise reference (overrides the sidecar
    # read) and the noise-mixed output bank's own id.
    # Validated at load so a malformed override fails before the mix run.
    noise_bank_id = _validated_id(
        _optional(doc, "input", "noise_bank_id", default=None), field_path="input.noise_bank_id"
    )
    bank_id = _validated_id(
        _optional(doc, "output", "bank_id", default=None), field_path="output.bank_id"
    )

    return MixCLIConfig(
        input_classifier_bank=input_classifier_bank,
        noise_bank_path=noise_bank_path,
        output_classifier_bank=output_classifier_bank,
        mixer_config=mixer_config,
        noise_bank_id=noise_bank_id,
        bank_id=bank_id,
    )


# ---------------------------------------------------------------------------
# Shared mixer-block builders
# ---------------------------------------------------------------------------


def _build_mixer_config(block: dict[str, Any]) -> MixerConfig:
    """Construct a :class:`MixerConfig` from a ``mixer:`` block."""
    snr_range = _expect_range(
        _optional(block, "snr_db_range", default=list(DEFAULT_SNR_DB_RANGE)),
        field_path="mixer.snr_db_range",
    )
    band_hz = _expect_range(
        _optional(block, "band_hz", default=list(DEFAULT_BIPOLAR_BAND_HZ)),
        field_path="mixer.band_hz",
    )
    return MixerConfig(
        snr_db_range=snr_range,
        bandpass_clean=bool(_optional(block, "bandpass_clean", default=True)),
        band_hz=band_hz,
        master_seed=int(_optional(block, "master_seed", default=0)),
        show_progress=bool(_optional(block, "show_progress", default=True)),
        description=str(_optional(block, "description", default="")),
    )


def _build_inline_mix(block: dict[str, Any], config_dir: Path) -> InlineMixConfig:
    """Construct an :class:`InlineMixConfig` from a ``mix:`` block.

    The ``mix:`` block has the same mixer knobs as the standalone
    ``mixer:`` block (delegated to :func:`_build_mixer_config`) plus
    the noise bank path it wraps.
    """
    noise_raw = _optional(block, "noise_bank", default=None)
    noise_bank_path = _resolve_path(str(noise_raw) if noise_raw is not None else "", config_dir)
    if noise_bank_path is None:
        raise ConfigError("mix.noise_bank must be set when the 'mix:' block is present.")
    noise_bank_id = _validated_id(
        _optional(block, "noise_bank_id", default=None), field_path="mix.noise_bank_id"
    )
    return InlineMixConfig(
        noise_bank_path=noise_bank_path,
        mixer_config=_build_mixer_config(block),
        noise_bank_id=noise_bank_id,
    )
