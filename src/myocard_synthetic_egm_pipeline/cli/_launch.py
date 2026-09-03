"""The console-script entry point, and the only place a thread cap can work.

**This module must not import numpy, numba, finitewave, or anything that pulls
them.** That is not a style preference — it is the entire reason the module
exists, and an import added here silently disables the cap for every run.

Why a shim at all
-----------------
numba fixes the size of its thread pool **when it initialises**, from the
``NUMBA_NUM_THREADS`` environment variable or, failing that, the core count.
After that the pool is what it is. ``numba.set_num_threads`` can select how many
of those threads are given work, but the rest continue to exist — and under the
OpenMP threading layer they *spin-wait*, consuming a core each while doing
nothing.

Measured on a 160² Courtemanche mesh, 2000 steps, on an 8-thread laptop:

===============================  =========  =========  ==========
condition                        wall       CPU        CPU / wall
===============================  =========  =========  ==========
uncapped                         10.4 s     67.3 s     6.49
``set_num_threads(1)`` at runtime 15.4 s     98.8 s     6.42
``NUMBA_NUM_THREADS=2`` pre-import 14.5 s    26.9 s     1.86
``OMP_NUM_THREADS=2`` pre-import  11.3 s     73.3 s     6.46
===============================  =========  =========  ==========

The middle row is the trap in full: ``numba.get_num_threads()`` returns 1, so
every obvious assertion passes, and the machine is burning 6.4 cores exactly as
before. **The getter is not a sufficient observable** — it reports the work
assignment, not the load. Only the third row actually gives the machine back,
and note the fourth: numba ignores ``OMP_NUM_THREADS`` when sizing its own pool,
so the variable most people would reach for does nothing here either.

Since the pool is sized at import, a cap read from a config file can only work
if the config is read **first**. So the console script points here: this module
peeks at the one key it needs using nothing heavier than PyYAML, sets the
environment variable, and only then imports the real command.

What this does not do
---------------------
It does not help the programmatic path. Anyone calling ``generate_dataset``
from their own script has already imported the solver by the time they build a
config, and no library code can undo that. They get
:func:`~myocard_synthetic_egm_pipeline.resources.apply_thread_limits`, which
still correctly limits *work assignment*, and they can set
``NUMBA_NUM_THREADS`` themselves before importing. The docstring in
``resources.py`` says so rather than letting the API imply a guarantee it
cannot keep.
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: The variable numba reads when sizing its pool. Deliberately not
#: ``OMP_NUM_THREADS``: numba ignores that one for pool sizing, as measured
#: above, so setting it would be a no-op that looks like a cap.
_POOL_SIZE_VAR = "NUMBA_NUM_THREADS"


def _config_path_from(argv: list[str]) -> str | None:
    """First non-flag argument, which is where the config path lives.

    Deliberately not argparse: the real parser lives in the command module,
    which cannot be imported yet. This only has to be right often enough to
    find a path, and wrong is not dangerous — an unreadable or unexpected
    argument just means no cap is applied and the run proceeds as before.
    """
    for arg in argv:
        if not arg.startswith("-"):
            return arg
    return None


def _requested_max_threads(config_path: str) -> int | None:
    """Read ``resources.max_threads`` and nothing else.

    Every failure is swallowed on purpose. This runs before the real config
    loader, which validates properly and reports properly; raising here would
    replace a good error message with a worse one thrown from a stack frame the
    user has no context for. A malformed config still fails, a moment later,
    with the message it should have.
    """
    try:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            doc: Any = yaml.safe_load(handle)
        value = (doc or {}).get("resources", {}).get("max_threads")
        return None if value is None else int(value)
    except Exception:
        return None


def size_thread_pool_from_config(argv: list[str]) -> int | None:
    """Set ``NUMBA_NUM_THREADS`` from the config, before numba can load.

    Returns the value it set, or ``None`` if it set nothing.

    An existing ``NUMBA_NUM_THREADS`` in the environment wins. Someone who
    exported it meant it, and it is the more specific instruction: they are
    talking about this process, whereas the config is talking about every
    machine that ever runs it.
    """
    if _POOL_SIZE_VAR in os.environ:
        return None

    config_path = _config_path_from(argv)
    if config_path is None:
        return None

    requested = _requested_max_threads(config_path)
    if requested is None or requested < 1:
        return None

    os.environ[_POOL_SIZE_VAR] = str(requested)
    return requested


def main(argv: list[str] | None = None) -> int:
    """Size the pool, then hand off to the real command."""
    args = sys.argv[1:] if argv is None else argv
    size_thread_pool_from_config(args)

    # Imported here, not at module scope. This line is what loads numba, and
    # everything above had to happen first.
    from myocard_synthetic_egm_pipeline.cli.generate_dataset_cmd import main as _main

    return _main(args)
