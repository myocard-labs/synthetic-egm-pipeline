"""Thread caps: applied through the library's runtime API, verified through its getter.

**Every assertion here reads the effect, never the request.** That is the whole
design of the module and the whole design of this file, because the obvious
implementation of a thread cap — writing ``OMP_NUM_THREADS`` into
``os.environ`` from a parsed config — is a silent no-op. The library read that
variable when it loaded, on ``import numpy``, long before the config existed.
The assignment succeeds, the variable really is set, and a test that reads it
back is green while nothing at all has been capped.

This project has shipped that failure once already, in an anisotropy knob that
assigned two attributes to an object nothing consulted. So the question these
tests ask is not "was the number recorded" but "does the library report doing
it".
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from myocard_synthetic_egm_pipeline.resources import (
    AppliedLimits,
    ResourceLimits,
    ThreadCapWarning,
    apply_thread_limits,
)

# numba is importable but leaves these three untyped, so the suppressions are
# gathered here rather than scattered over every assertion. Thin on purpose:
# each one is a single call, so nothing can hide behind the annotation.


def numba_threads() -> int:
    """What numba says it is currently using."""
    import numba

    return int(numba.get_num_threads())  # type: ignore[no-untyped-call]


def numba_max_threads() -> int:
    """The ceiling fixed when numba initialised."""
    import numba

    return int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]


def set_numba_threads(count: int) -> None:
    import numba

    numba.set_num_threads(count)  # type: ignore[no-untyped-call]


@pytest.fixture(autouse=True)
def restore_thread_count() -> Iterator[None]:
    """Put the thread count back, whatever a test did to it.

    ``set_num_threads`` is process-global, so a test that lowered it would
    otherwise slow down every test that ran afterwards — and, far worse, would
    make the benchmark suite report a capped machine as an uncapped one.
    """
    before = numba_threads()
    yield
    set_numba_threads(before)


# ---------------------------------------------------------------------------
# The cap is observable
# ---------------------------------------------------------------------------


def test_a_cap_is_visible_through_numbas_own_getter() -> None:
    """The assertion that distinguishes this from an environment-variable no-op.

    ``numba.get_num_threads()`` is the library reporting its own state. Nothing
    this package wrote is being read back.
    """
    if numba_max_threads() < 2:
        pytest.skip("a single-thread machine cannot demonstrate a cap")

    applied = apply_thread_limits(ResourceLimits(max_threads=1))

    assert numba_threads() == 1, (
        "numba still reports its previous thread count, so the cap was not applied"
    )
    assert applied.effective == 1


def test_the_reported_effective_count_is_read_back_not_computed() -> None:
    """``AppliedLimits.effective`` must come from the library, not from arithmetic.

    Constructed by hand with a contradictory pair, the dataclass keeps both:
    it is a record of *what was asked* and *what happened*, and collapsing those
    into one field is exactly how a cap that silently failed would come to look
    like a cap that worked.
    """
    record = AppliedLimits(requested=4, effective=8, maximum=8)
    assert record.requested == 4
    assert record.effective == 8


def test_no_cap_leaves_the_thread_count_alone() -> None:
    """The default, and the reason adding this block changed no existing run."""
    before = numba_threads()
    applied = apply_thread_limits(ResourceLimits())

    assert numba_threads() == before
    assert applied.requested is None
    assert applied.effective == before


def test_asking_for_more_threads_than_exist_warns_and_uses_the_maximum() -> None:
    """The ceiling is fixed when numba initialises and cannot be raised here.

    Warns rather than raises: discovering it at the top of a multi-hour
    generation run should not throw the run away, and the number it settles on
    is the best available rather than a wrong one.
    """
    maximum = numba_max_threads()

    with pytest.warns(ThreadCapWarning, match="exceeds this process's maximum"):
        applied = apply_thread_limits(ResourceLimits(max_threads=maximum + 5))

    assert applied.clamped
    assert applied.effective == maximum
    assert numba_threads() == maximum


def test_zero_or_negative_threads_is_refused_at_construction() -> None:
    """Refused where it is written, not where it would take effect."""
    with pytest.raises(ValueError, match="at least 1"):
        ResourceLimits(max_threads=0)


# ---------------------------------------------------------------------------
# The cap must not change what is produced
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_capped_run_produces_the_same_traces_as_an_uncapped_one() -> None:
    """Thread count is a performance knob, and must be nothing else.

    Runs the *same* simulation — same seed, same specs, same geometry — twice,
    once on one thread and once on as many as the machine has, and requires the
    electrograms to agree **exactly**. Not approximately: the solver is an
    explicit scheme over a fixed grid with no reduction whose order depends on
    the schedule, so identical inputs must give identical floats. A tolerance
    here would hide precisely the bug worth finding.

    If this ever fails, the cap is not the problem — a solver whose output
    depends on how many threads happened to be available is, and no bank
    produced before the failure can be trusted either.
    """
    import dataclasses

    if numba_max_threads() < 2:
        pytest.skip("a single-thread machine cannot show a difference either way")

    from pathlib import Path

    import yaml

    from myocard_synthetic_egm_pipeline.backends.finitewave import FinitewaveBackend
    from myocard_synthetic_egm_pipeline.cli._config import build_generate_dataset_config
    from myocard_synthetic_egm_pipeline.cli.generate_dataset_cmd import _build_dataset_config
    from myocard_synthetic_egm_pipeline.simulate import Patch2DGeometry
    from myocard_synthetic_egm_pipeline.simulate.dataset import _sample_specs
    from myocard_synthetic_egm_pipeline.simulate.runner import run_single

    path = Path("examples/synthegm_v1_baseline.yaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["_config_dir"] = path.parent
    cfg = _build_dataset_config(build_generate_dataset_config(doc), show_progress=False)
    assert isinstance(cfg.geometry, Patch2DGeometry)

    # A short capture on a small patch: this test is about thread-independence,
    # not about the production geometry, and the property either holds for every
    # mesh or is broken for all of them.
    cfg = dataclasses.replace(
        cfg,
        geometry=dataclasses.replace(cfg.geometry, size_mm=12.0),
        run_config=dataclasses.replace(
            cfg.run_config, trace_duration_ms=64.0, capture_duration_ms=None
        ),
    )

    def traces(threads: int) -> npt.NDArray[np.float64]:
        apply_thread_limits(ResourceLimits(max_threads=threads))
        assert numba_threads() == threads
        rng = np.random.default_rng(20260903)
        substrate, activation, electrodes = _sample_specs(config=cfg, sim_rng=rng)
        result = run_single(
            geometry=cfg.geometry,
            substrate=substrate,
            activation=activation,
            electrodes=electrodes,
            backend=FinitewaveBackend(),
            cell_model=cfg.cell_model,
            config=cfg.run_config,
            rng=rng,
            position_generator=None,
            detection_preprocessor=cfg.detection_preprocessor,
        )
        return np.asarray(result.bipolar_traces, dtype=np.float64)

    one_thread = traces(1)
    many_threads = traces(numba_max_threads())

    assert one_thread.shape == many_threads.shape
    np.testing.assert_array_equal(
        one_thread,
        many_threads,
        err_msg=(
            "the electrograms differ between a 1-thread and a "
            f"{numba_max_threads()}-thread run of the same seed. The "
            "thread count is changing the physics, which invalidates every bank "
            "generated at any other thread count."
        ),
    )


# ---------------------------------------------------------------------------
# Sizing the pool before numba loads — the half that actually frees the machine
# ---------------------------------------------------------------------------


def test_the_launcher_reads_the_cap_out_of_a_config(tmp_path: Path) -> None:
    """The launcher must find the cap using nothing that imports numba."""
    from myocard_synthetic_egm_pipeline.cli._launch import size_thread_pool_from_config

    config = tmp_path / "run.yaml"
    config.write_text("resources:\n  max_threads: 3\n", encoding="utf-8")

    saved = os.environ.pop("NUMBA_NUM_THREADS", None)
    try:
        assert size_thread_pool_from_config([str(config)]) == 3
        assert os.environ["NUMBA_NUM_THREADS"] == "3"
    finally:
        os.environ.pop("NUMBA_NUM_THREADS", None)
        if saved is not None:
            os.environ["NUMBA_NUM_THREADS"] = saved


def test_an_explicit_environment_variable_beats_the_config(tmp_path: Path) -> None:
    """Someone who exported it meant it, and is talking about this process."""
    from myocard_synthetic_egm_pipeline.cli._launch import size_thread_pool_from_config

    config = tmp_path / "run.yaml"
    config.write_text("resources:\n  max_threads: 3\n", encoding="utf-8")

    saved = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = "7"
    try:
        assert size_thread_pool_from_config([str(config)]) is None
        assert os.environ["NUMBA_NUM_THREADS"] == "7"
    finally:
        os.environ.pop("NUMBA_NUM_THREADS", None)
        if saved is not None:
            os.environ["NUMBA_NUM_THREADS"] = saved


def test_a_config_without_the_block_sets_nothing(tmp_path: Path) -> None:
    """Absent block, unreadable file and missing argument all mean 'no cap'.

    Swallowed rather than raised: the real config loader runs a moment later and
    reports properly, and an exception thrown from here would replace a good
    error message with one from a frame the user has no context for.
    """
    from myocard_synthetic_egm_pipeline.cli._launch import size_thread_pool_from_config

    empty = tmp_path / "run.yaml"
    empty.write_text("dataset:\n  n_simulations: 1\n", encoding="utf-8")
    broken = tmp_path / "broken.yaml"
    broken.write_text("resources: [this is not a mapping\n", encoding="utf-8")

    saved = os.environ.pop("NUMBA_NUM_THREADS", None)
    try:
        assert size_thread_pool_from_config([str(empty)]) is None
        assert size_thread_pool_from_config([str(broken)]) is None
        assert size_thread_pool_from_config([str(tmp_path / "absent.yaml")]) is None
        assert size_thread_pool_from_config([]) is None
        assert size_thread_pool_from_config(["--no-progress"]) is None
        assert "NUMBA_NUM_THREADS" not in os.environ
    finally:
        if saved is not None:
            os.environ["NUMBA_NUM_THREADS"] = saved


def test_the_launcher_module_imports_nothing_that_loads_numba() -> None:
    """The constraint the whole design rests on, asserted rather than trusted.

    ``_launch`` runs before numba so that it can size the pool. If a future
    edit adds a numpy or finitewave import at its module scope — directly, or
    through a package ``__init__`` — the pool is already built by the time the
    config is read, the cap silently stops working, and every other test here
    still passes because they all exercise the function rather than the import.

    Checked in a subprocess because this one has numba loaded already.
    """
    import subprocess

    probe = (
        "import sys, importlib;"
        "importlib.import_module('myocard_synthetic_egm_pipeline.cli._launch');"
        "loaded = [m for m in ('numba', 'numpy', 'finitewave') if m in sys.modules];"
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", (
        f"importing the launcher pulled in {result.stdout.strip()}, so numba's "
        "thread pool is sized before the config is read and resources.max_threads "
        "has silently stopped working"
    )


def test_the_launcher_actually_shrinks_numbas_pool(tmp_path: Path) -> None:
    """The decisive observable, and the one ``get_num_threads`` cannot give.

    ``numba.config.NUMBA_NUM_THREADS`` is the **pool size**, fixed at import and
    unchangeable afterwards. ``get_num_threads()`` is how many of that pool get
    work. Only the first one corresponds to how much of the machine is being
    consumed, because under the OpenMP layer the unused threads spin rather than
    sleep — capping work assignment alone measured 6.4 cores of load on an
    8-thread box while the getter reported 1.

    So this test asserts the pool shrank. In a subprocess, because the pool in
    *this* process was fixed when the suite started and no test can change it.
    """
    import subprocess

    config = tmp_path / "run.yaml"
    config.write_text("resources:\n  max_threads: 2\n", encoding="utf-8")

    probe = (
        "import sys;"
        "from myocard_synthetic_egm_pipeline.cli._launch import size_thread_pool_from_config;"
        f"size_thread_pool_from_config([{str(config)!r}]);"
        "import numba;"
        "print(numba.config.NUMBA_NUM_THREADS)"
    )
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_NUM_THREADS"}
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )

    assert result.stdout.strip() == "2", (
        f"numba built a pool of {result.stdout.strip()} threads despite a cap of 2; "
        "the config was read too late to size it"
    )
