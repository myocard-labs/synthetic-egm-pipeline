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

# egm-signal owns the bandpass default; importing for the mixer block.
from myocard_egm_signal import DEFAULT_BIPOLAR_BAND_HZ

from myocard_synthetic_egm_pipeline.backends import RunConfig
from myocard_synthetic_egm_pipeline.constants import (
    AP_TIME_UNIT_MS,
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


def _build_run_config(doc: dict[str, Any]) -> RunConfig:
    """Construct a :class:`RunConfig` from the ``run:`` block."""
    block = _optional(doc, "run", default={}) or {}
    return RunConfig(
        trace_duration_ms=float(
            _optional(block, "trace_duration_ms", default=DEFAULT_TRACE_DURATION_MS)
        ),
        output_fs_hz=float(_optional(block, "output_fs_hz", default=DEFAULT_OUTPUT_FS_HZ)),
        ap_time_unit_ms=float(_optional(block, "ap_time_unit_ms", default=AP_TIME_UNIT_MS)),
        capture_oversample=int(_optional(block, "capture_oversample", default=4)),
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
    electrode_n_rows: int
    electrode_n_cols: int
    electrode_spacing_mm: float
    electrode_height_mm_range: tuple[float, float]

    # Label policy + run config
    label_policy: LabelPolicy
    run_config: RunConfig

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
    run_config = _build_run_config(doc)

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
        electrode_n_rows=electrode_n_rows,
        electrode_n_cols=electrode_n_cols,
        electrode_spacing_mm=electrode_spacing_mm,
        electrode_height_mm_range=electrode_height_mm_range,
        label_policy=label_policy,
        run_config=run_config,
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
