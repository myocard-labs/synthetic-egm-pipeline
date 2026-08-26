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
3. **Parameter estimation.** An estimator fitted against real recordings
   returns a *region* of parameters, not a point. Sampling that region means
   writing many parameterisations, which is natural if a parameterisation is a
   file and unbearable if it is a block inside a config.

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
Same reasoning that makes a bank reference record a relative path *and* an
id rather than either alone: the path locates the file today, the id says which
artifact it was.
"""

from __future__ import annotations

from collections.abc import Callable
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
    CourtemancheCellModel,
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


def _parse_aliev_panfilov_solved(block: dict[str, Any]) -> CellModelSpec:
    """The four knobs ``calibrate_aliev_panfilov`` returns."""
    return AlievPanfilovCellModel(
        time_unit_ms=float(_require(block, "time_unit_ms", "model.solved")),
        diffusion=float(_require(block, "diffusion", "model.solved")),
        eps=float(_require(block, "eps", "model.solved")),
        dt_model_units=float(_require(block, "dt_model_units", "model.solved")),
    )


def _parse_courtemanche_solved(block: dict[str, Any]) -> CellModelSpec:
    """The two knobs ``calibrate_courtemanche`` returns, plus chosen scalings.

    ``dt_ms`` rather than ``dt_model_units`` in the file: Courtemanche's model
    time unit *is* the millisecond, and a card is read by people. The spec field
    keeps the shared name because the backend asks every model the same
    question.

    ``params`` is optional and empty means control — the shipped card. Its
    values are **chosen**, not solved, so unlike the two fields above they are
    carried through verification rather than re-derived.
    """
    params_block = block.get("params") or {}
    if not isinstance(params_block, dict):
        raise ValueError("model.solved.params must be a mapping of scaling names to numbers.")
    return CourtemancheCellModel(
        diffusion=float(_require(block, "diffusion", "model.solved")),
        dt_model_units=float(_require(block, "dt_ms", "model.solved")),
        params={str(key): float(value) for key, value in params_block.items()},
    )


#: Card ``type`` → the parser for its ``solved:`` block.
#:
#: A table rather than an if-chain because the *set of keys* differs per model —
#: Aliev-Panfilov has ``eps`` and ``time_unit_ms``, Courtemanche has neither and
#: has ``params`` instead — so there is nothing shared to factor out, and a
#: missing entry is what refuses an unknown model by name.
_CARD_PARSERS: dict[str, Callable[[dict[str, Any]], CellModelSpec]] = {
    "aliev_panfilov": _parse_aliev_panfilov_solved,
    "courtemanche": _parse_courtemanche_solved,
}


def parse_model_card(doc: dict[str, Any], *, source: str) -> ModelCard:
    """Build a :class:`ModelCard` from parsed YAML. No filesystem access."""
    if not isinstance(doc, dict) or "model" not in doc:
        raise ModelCardError(f"{source}: expected a top-level 'model:' block.")
    block = doc["model"]
    if not isinstance(block, dict):
        raise ModelCardError(f"{source}: 'model:' must be a mapping.")

    # `type` selects which cell model the `solved` block describes. Dispatch
    # rather than assume: an unknown type is refused by name instead of being
    # mis-parsed as Aliev-Panfilov, which would read a `diffusion` in mm^2/ms as
    # a dimensionless one and differ by three orders of magnitude in silence.
    model_type = str(block.get("type", "aliev_panfilov"))
    if model_type not in _CARD_PARSERS:
        raise ModelCardError(
            f"{source}: no card parser is registered for cell model {model_type!r}. "
            f"Known: {', '.join(sorted(_CARD_PARSERS))}. A new membrane model needs "
            "its own measured calibration constants before a card can be written "
            "for it."
        )

    targets_block = _require(block, "targets", "model")
    solved_block = _require(block, "solved", "model")

    try:
        targets = ModelTargets(
            conduction_velocity_cm_s=float(
                _require(targets_block, "conduction_velocity_cm_s", "model.targets")
            ),
            # Optional, and per model: Courtemanche measures APD rather than
            # solving it, so a card for it may state no APD target at all. The
            # solve-side branch refuses a *missing* target for a model that
            # needs one, so absence cannot slip past as a default.
            apd90_ms=(
                None if targets_block.get("apd90_ms") is None else float(targets_block["apd90_ms"])
            ),
        )
        solved: CellModelSpec = _CARD_PARSERS[model_type](solved_block)
    except (TypeError, ValueError) as exc:
        raise ModelCardError(f"{source}: {exc}") from exc

    # `measured` is optional and may be present-but-empty: a card can be written
    # from the solve before anyone has run the simulation that fills it in. That
    # is a real state, not a malformed file, so it parses to None rather than
    # raising — and the round-trip test reports the values to paste back.
    measured = None
    measured_block = block.get("measured") or {}
    if measured_block.get("conduction_velocity_cm_s") is not None:
        # `upstroke_v_s` is optional even here: it is only meaningful for a
        # model whose potential is in millivolts, so an Aliev-Panfilov card
        # leaves it out rather than writing a number in units it does not have.
        upstroke = measured_block.get("upstroke_v_s")
        measured = MeasuredValues(
            conduction_velocity_cm_s=float(measured_block["conduction_velocity_cm_s"]),
            apd90_ms=float(_require(measured_block, "apd90_ms", "model.measured")),
            upstroke_v_s=None if upstroke is None else float(upstroke),
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
        **card.solved.to_metadata(),
    }
    # Omitted rather than written as a null when the card states no APD target:
    # backend_metadata lands in HDF5 attributes, which have no null, so the
    # alternative is a sentinel number that reads as a target.
    if card.targets.apd90_ms is not None:
        provenance["model_target_apd90_ms"] = float(card.targets.apd90_ms)
    if card.measured is not None:
        provenance["model_measured_conduction_velocity_cm_s"] = float(
            card.measured.conduction_velocity_cm_s
        )
        provenance["model_measured_apd90_ms"] = float(card.measured.apd90_ms)
        if card.measured.upstroke_v_s is not None:
            # Named `propagated` on disk, because the distinction from a
            # stimulated single-cell upstroke is the whole point of recording
            # it and a bare `upstroke` would invite reading it as the other one.
            provenance["model_measured_propagated_upstroke_v_s"] = float(card.measured.upstroke_v_s)
    return provenance


__all__ = [
    "ModelCardError",
    "card_provenance",
    "load_model_card",
    "parse_model_card",
    "shipped_card_names",
]
