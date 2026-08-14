"""Load and resolve named model cards.

A **model card** is a small YAML file naming one tissue parameterisation: the
physiological targets, the knobs :func:`~...calibration.calibrate` solved for
them, and what a simulation measured. It is a file rather than a block inside a
generation config for three reasons:

1. **Identity.** A named file is something the methods section can cite and a
   diff can show. Eight numbers repeated across a dozen configs is not, and it
   drifts silently between them.
2. **Reuse.** A handful of standard parameterisations, referenced by name, keeps
   generation configs about geometry and output rather than about membrane
   kinetics.
3. **The STU4 lifecycle.** The estimator's output is a *region* of parameters,
   not a point. Sampling it means writing many parameterisations, which is
   natural if a parameterisation is a file.

Resolution order
----------------
``model:`` in a config is either a **shipped name** (``af_remodelled_220ms``) or
a **path** (``models/mine.yaml``, resolved against the config file's own
directory, per the existing ``_resolve_path`` convention). Names resolve to
files inside this package, so a config that uses one is reproducible on any
checkout; paths are the escape hatch for experiments.

No partial overrides, deliberately
----------------------------------
A generation config may **not** patch individual fields of a card. It is
convenient and it destroys the identity the card exists to provide — two banks
both claiming ``af_remodelled_220ms`` with different ``eps`` is worse than no
name at all. Point at a shipped name, or write your own file.

What a bank records
-------------------
The **resolved contents**, not the path. A path is a pointer into a mutable
filesystem; six months from now it may name a different file, or none. The card
name is kept alongside as a human-readable label, but the values are the record.
Same reasoning that made bank references record a relative path *and* an id
rather than either alone (S13 / B13).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from myocard_synthetic_egm_pipeline.simulate.calibration import (
    MeasuredValues,
    ModelCard,
    ModelTargets,
    verify_solved,
)
from myocard_synthetic_egm_pipeline.simulate.cell_models import (
    AlievPanfilovCellModel,
    CellModelSpec,
)

#: Package subdirectory holding the shipped cards.
_SHIPPED_PACKAGE = "myocard_synthetic_egm_pipeline.models"


class ModelCardError(ValueError):
    """A model card is missing, malformed, or no longer agrees with the solver."""


def _require(block: dict[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise ModelCardError(
            f"model card is missing {where}.{key!r}. A card must state its targets "
            "and the solved knobs; the targets alone would leave the physics "
            "dependent on whichever version of calibrate() happened to run."
        )
    return block[key]


def parse_model_card(doc: dict[str, Any], *, source: str) -> ModelCard:
    """Build a :class:`ModelCard` from parsed YAML. No filesystem access."""
    if not isinstance(doc, dict) or "model" not in doc:
        raise ModelCardError(f"{source}: expected a top-level 'model:' block.")
    block = doc["model"]
    if not isinstance(block, dict):
        raise ModelCardError(f"{source}: 'model:' must be a mapping.")

    # `type` selects which cell model the `solved` block describes. Dispatch
    # rather than assume: Courtemanche joins by adding a branch, and an unknown
    # type is refused by name instead of being mis-parsed as Aliev-Panfilov.
    model_type = str(block.get("type", "aliev_panfilov"))
    if model_type != "aliev_panfilov":
        raise ModelCardError(
            f"{source}: no card parser is registered for cell model {model_type!r}. "
            "A new membrane model needs its own measured calibration constants "
            "before a card can be written for it."
        )

    targets_block = _require(block, "targets", "model")
    solved_block = _require(block, "solved", "model")

    try:
        targets = ModelTargets(
            conduction_velocity_cm_s=float(
                _require(targets_block, "conduction_velocity_cm_s", "model.targets")
            ),
            apd90_ms=float(_require(targets_block, "apd90_ms", "model.targets")),
        )
        solved: CellModelSpec = AlievPanfilovCellModel(
            time_unit_ms=float(_require(solved_block, "time_unit_ms", "model.solved")),
            diffusion=float(_require(solved_block, "diffusion", "model.solved")),
            eps=float(_require(solved_block, "eps", "model.solved")),
            dt_model_units=float(_require(solved_block, "dt_model_units", "model.solved")),
        )
    except (TypeError, ValueError) as exc:
        raise ModelCardError(f"{source}: {exc}") from exc

    # `measured` is optional and may be present-but-empty: a card can be written
    # from the solve before anyone has run the simulation that fills it in. That
    # is a real state, not a malformed file, so it parses to None rather than
    # raising — and the round-trip test reports the values to paste back.
    measured = None
    measured_block = block.get("measured") or {}
    if measured_block.get("conduction_velocity_cm_s") is not None:
        measured = MeasuredValues(
            conduction_velocity_cm_s=float(measured_block["conduction_velocity_cm_s"]),
            apd90_ms=float(_require(measured_block, "apd90_ms", "model.measured")),
        )

    return ModelCard(
        name=str(block.get("name", source)),
        targets=targets,
        solved=solved,
        measured=measured,
    )


def shipped_card_names() -> list[str]:
    """Names of the parameterisations that ship inside the package."""
    files = resources.files(_SHIPPED_PACKAGE)
    return sorted(
        entry.name.removesuffix(".yaml")
        for entry in files.iterdir()
        if entry.name.endswith(".yaml")
    )


def load_model_card(
    reference: str,
    *,
    config_dir: Path | None = None,
    dr_mm: float,
    dr_model_units: float,
) -> ModelCard:
    """Resolve ``reference`` to a card and verify it still agrees with the solver.

    ``reference`` is a shipped name or a path. The verification is not optional
    and not a warning: a card whose recorded knobs no longer match a fresh solve
    means the calibration routine changed under it, and generating from it would
    put two different physics under one parameterisation name.

    Parameters
    ----------
    config_dir
        Directory of the referencing config, for resolving relative paths —
        the same convention every other path in a config follows.
    dr_mm, dr_model_units
        Needed by the verification solve, and they are properties of the
        *geometry and solver*, not of the card. A card is therefore only valid
        in the mesh context it was solved for, which is why they are required
        here rather than defaulted.
    """
    text: str
    source: str

    if reference in shipped_card_names():
        source = f"shipped card {reference!r}"
        text = resources.files(_SHIPPED_PACKAGE).joinpath(f"{reference}.yaml").read_text()
    else:
        path = Path(reference)
        if not path.is_absolute() and config_dir is not None:
            path = config_dir / path
        if not path.exists():
            available = ", ".join(shipped_card_names()) or "(none)"
            raise ModelCardError(
                f"model card {reference!r} is neither a shipped name nor an existing "
                f"file (looked at {path}). Shipped names: {available}."
            )
        source = str(path)
        text = path.read_text()

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ModelCardError(f"{source}: could not parse YAML: {exc}") from exc

    card = parse_model_card(doc, source=source)

    try:
        verify_solved(card, dr_mm=dr_mm, dr_model_units=dr_model_units)
    except ValueError as exc:
        raise ModelCardError(f"{source}: {exc}") from exc

    return card


def card_provenance(card: ModelCard) -> dict[str, Any]:
    """Flatten a card for ``backend_metadata`` — resolved values, not a path."""
    # The universal half is written here; the model-specific half is the cell
    # model's own business, so it is asked rather than introspected.
    provenance: dict[str, Any] = {
        "model_card_name": card.name,
        "model_target_conduction_velocity_cm_s": float(card.targets.conduction_velocity_cm_s),
        "model_target_apd90_ms": float(card.targets.apd90_ms),
        **card.solved.to_metadata(),
    }
    if card.measured is not None:
        provenance["model_measured_conduction_velocity_cm_s"] = float(
            card.measured.conduction_velocity_cm_s
        )
        provenance["model_measured_apd90_ms"] = float(card.measured.apd90_ms)
    return provenance


__all__ = [
    "ModelCardError",
    "card_provenance",
    "load_model_card",
    "parse_model_card",
    "shipped_card_names",
]
