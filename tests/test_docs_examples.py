"""Execute the code examples in docs/usage.md and README.md.

A documented example that no longer runs is invisible: nothing imports it,
nothing lints it, and a reader discovers the breakage instead of the author.
Both files had rotted once already — ``RunConfig`` no longer takes
``ap_time_unit_ms`` (``TypeError``), ``trace_duration_ms: 200.0`` is refused
because ``T`` must be a multiple of 64 (``ConfigError``), and the YAML schema
reference still documented ``run.ap_time_unit_ms``, which is now a hard error.
Every one of those is a **construction** failure, which is why this file is
cheap: building the objects catches them in milliseconds.

**What this checks and what it does not.** The examples are executed with the
Finitewave backend swapped for the test mock, so what is verified is that they
*construct and wire* — imports resolve, keyword arguments exist, config
constraints hold, writer and builder signatures match. The physics is not
exercised and is not the point; a doc test that ran ten real simulations would
be a doc test nobody runs. The patch is a plain ``setattr``, so a rename of
``FinitewaveBackend`` itself still fails here rather than being papered over.

Extraction is by fence, from the shipped files. There is deliberately no copy
of the examples in this module: a test asserting against its own transcription
of the docs would keep passing while the docs rotted, which is the failure it
exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from myocard_egm_contracts._generated.python.noise_bank import NoiseBank
from myocard_egm_data.banks import write_noise_bank

import myocard_synthetic_egm_pipeline.backends.finitewave as finitewave_module
from myocard_synthetic_egm_pipeline.cli._config import (
    build_generate_dataset_config,
    build_mix_config,
    load_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
USAGE_MD = REPO_ROOT / "docs" / "usage.md"
README_MD = REPO_ROOT / "README.md"

#: Where the mixer example expects to find an iafdb noise bank. Read from the
#: doc rather than assumed: the example is the specification here.
NOISE_BANK_IN_EXAMPLE = Path("banks/iafdb_noise_v1.h5")


def _fenced_blocks(path: Path, language: str) -> list[str]:
    """Every ```<language> block in ``path``, in document order."""
    blocks: list[str] = []
    current: list[str] | None = None
    opening = f"```{language}"
    for line in path.read_text(encoding="utf-8").splitlines():
        if current is not None:
            if line.strip() == "```":
                blocks.append("\n".join(current))
                current = None
            else:
                current.append(line)
        elif line.strip() == opening:
            current = []
    assert current is None, f"unclosed {opening} block in {path.name}"
    return blocks


def _run_example(source: str, *, name: str) -> None:
    """Execute one extracted block as a module-level script."""
    exec(compile(source, f"<{name}>", "exec"), {"__name__": "__doc_example__"})


@pytest.fixture
def doc_sandbox(
    tmp_path: Path,
    mock_backend: Any,
    small_noise_bank: NoiseBank,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """A working directory the examples' relative paths resolve inside.

    The examples write to ``out/`` and read ``banks/iafdb_noise_v1.h5``. Only
    the noise bank is created here — ``out/`` is left to the writer on purpose,
    so a reader following the doc on a fresh checkout is doing exactly what this
    test does.

    ``raising`` is left at its default on the backend patch: if
    ``FinitewaveBackend`` were renamed, this fails on the missing attribute
    rather than silently installing a mock under a name the doc no longer uses.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finitewave_module, "FinitewaveBackend", lambda: mock_backend)
    write_noise_bank(small_noise_bank, tmp_path / NOISE_BANK_IN_EXAMPLE)
    return tmp_path


# ---------------------------------------------------------------------------
# Python examples
# ---------------------------------------------------------------------------


def test_usage_python_examples_run(doc_sandbox: Path) -> None:
    """docs/usage.md's "Programmatic use" examples, in document order.

    In order and in one sandbox because they are one narrative: the mixer
    example opens the bank the generator example just wrote ("The mixer is
    similar"). Running them independently would need a bank invented here, and
    an invented input is the thing most likely to differ from the one the doc
    actually produces.
    """
    examples = _fenced_blocks(USAGE_MD, "python")

    # An extractor that quietly found nothing would report "all examples pass"
    # against an empty list — the exact shape of a check this repo has already
    # been burned by.
    assert examples, f"no ```python blocks found in {USAGE_MD.name}"

    for index, source in enumerate(examples):
        _run_example(source, name=f"{USAGE_MD.name}[python {index}]")


def test_readme_python_example_runs(doc_sandbox: Path) -> None:
    """The README's programmatic-usage example.

    Its own sandbox: the README stands alone, and a reader lands on it without
    having run anything from usage.md first.
    """
    examples = _fenced_blocks(README_MD, "python")

    assert examples, f"no ```python blocks found in {README_MD.name}"

    for index, source in enumerate(examples):
        _run_example(source, name=f"{README_MD.name}[python {index}]")


# ---------------------------------------------------------------------------
# YAML schema reference
# ---------------------------------------------------------------------------

#: Which builder a documented config belongs to, keyed on the top-level block
#: that discriminates the two CLIs. Dispatching on content rather than on
#: position means a block added to the reference is checked automatically
#: instead of being silently skipped by an index that no longer lines up.
_YAML_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "dataset": build_generate_dataset_config,
    "input": build_mix_config,
}


def test_documented_yaml_configs_load(tmp_path: Path) -> None:
    """Every YAML config in the schema reference builds a typed config.

    Loaded through ``load_yaml`` from a real file, so path resolution against
    the config's directory is exercised the way a user's config is. The
    placeholder paths are never opened — the loader resolves them and the
    builder validates them, which is where the documented keys are checked.

    This is the check the reference block most needed: it documented
    ``run.ap_time_unit_ms`` long after that key became an error, so the
    canonical example of a working config was one that could not be loaded.
    """
    blocks = _fenced_blocks(USAGE_MD, "yaml")

    assert blocks, f"no ```yaml blocks found in {USAGE_MD.name}"

    for index, source in enumerate(blocks):
        path = tmp_path / f"documented_config_{index}.yaml"
        path.write_text(source, encoding="utf-8")
        doc = load_yaml(path)

        builders = [build for key, build in _YAML_BUILDERS.items() if key in doc]
        if not builders:
            pytest.fail(
                f"{USAGE_MD.name} yaml block {index} matches no known config shape "
                f"(expected a top-level {' or '.join(_YAML_BUILDERS)} key). Teach "
                "_YAML_BUILDERS which builder validates it — a documented config "
                "nothing loads is how this reference came to document a key that "
                "had become an error."
            )

        for build in builders:
            build(doc)


# ---------------------------------------------------------------------------
# Every shipped example, found by globbing rather than by being named
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_every_shipped_example_config_loads() -> None:
    """Each file in ``examples/`` builds a typed config.

    **Globbed, not listed.** Until this existed every example was loaded only by
    a test that named it, so a newly shipped example got no coverage at all
    until somebody remembered to add one — and an example that cannot load is
    worse than no example, because it is the first thing a new user copies.

    Dispatch is by top-level block, the same way the documented-config test
    picks a builder, so a mix config and a generate config are each checked
    against the builder that owns them.
    """
    configs = sorted(EXAMPLES_DIR.glob("*.yaml"))
    assert configs, f"no example configs found in {EXAMPLES_DIR}"

    for path in configs:
        doc = load_yaml(path)
        builders = [build for key, build in _YAML_BUILDERS.items() if key in doc]
        assert builders, (
            f"{path.name} matches no known config shape (expected a top-level "
            f"{' or '.join(_YAML_BUILDERS)} key)."
        )
        for build in builders:
            build(doc)


def test_the_sweep_example_screens_the_two_knobs_it_documents() -> None:
    """The shipped screen is the design it says it is.

    Asserted on the built design rather than on the YAML text: a comment
    claiming five cells is worth nothing if the sampler produces four.
    """
    doc = load_yaml(EXAMPLES_DIR / "synthegm_sweep.yaml")
    cfg = build_generate_dataset_config(doc)

    assert cfg.sweep is not None
    assert [k.path for k in cfg.sweep.knobs] == [
        "substrate.density_range.high",
        "mix.snr_db_range.low",
    ]
    assert all(k.role == "nuisance" for k in cfg.sweep.knobs)

    cells = cfg.sweep.design()
    assert len(cells) == 5, "the header's cell count no longer matches the design"
    assert sum(1 for c in cells if c.varied is None) == 1, "no all-nominal baseline"
