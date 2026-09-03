"""Thread caps, so a long generation run does not redline the machine.

A Courtemanche bank is tens of minutes per simulation and the loop is
sequential, so a thousand-simulation run occupies the machine for days. Left
uncapped, the solver takes every core it can see and the machine becomes
unusable for anything else. This module is the knob that gives some back.

**Environment variables cannot do this job, and the way they fail is the
dangerous kind.**
--------------------------------------------------------------------------
``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``OPENBLAS_NUM_THREADS`` are read
by the threading library **when it loads**, which happens on ``import numpy`` —
long before any configuration file has been opened. Setting them from a parsed
config is therefore too late by construction::

    os.environ["OMP_NUM_THREADS"] = "4"   # does nothing; the library has loaded

It does not raise. It does not warn. The variable really is set, so a test that
reads it back passes, and the cap that was never applied looks applied. That is
the same shape of failure as the anisotropy knob that assigned to the wrong
object for the life of the project: the code plainly appeared to do the thing,
nothing errored, and only a measurement of the *effect* would have caught it.

Which is why :func:`apply_thread_limits` reports what it achieved by asking
**numba's own getter**, not by echoing back what it was told.

What is capped, and what deliberately is not
--------------------------------------------
**numba is capped, in two places, because one of them is not enough.**
Finitewave's kernels are ``njit`` with ``prange``, so numba's threading is what
governs the solver — but numba has two separate notions of thread count and
only one of them corresponds to load on the machine:

- the **pool size**, fixed when numba is imported from ``NUMBA_NUM_THREADS``;
- the **work assignment**, which ``set_num_threads`` selects at runtime.

``set_num_threads`` alone is not a cap on the machine. The threads it stops
assigning work to still exist, and under the OpenMP threading layer they
*spin-wait* rather than sleep. Measured on a 160² Courtemanche mesh: capping to
one thread left ``get_num_threads()`` reporting 1 and the process consuming
**6.4 cores**, indistinguishable from uncapped. Sizing the pool before import
took CPU time from 67 s to 27 s for the same work.

So the pool is sized by
:mod:`~myocard_synthetic_egm_pipeline.cli._launch`, which runs before numba
loads, and this module handles work assignment on top. Both, not either.

**This function cannot size the pool**, and callers should know it. By the time
any library code runs, numba has been imported and its pool built. Reaching
:func:`apply_thread_limits` from a script gives correct work assignment and no
reduction in load; to get the second, set ``NUMBA_NUM_THREADS`` in the
environment before importing this package.

**BLAS is not capped, and no dependency was added for it.** Capping BLAS would
need ``threadpoolctl``, which this package does not depend on. Before adding it,
the question was measured rather than assumed: instrumenting every
BLAS-dispatching entry point in numpy and scipy and running a full simulation —
solver, electrogram tracking, bipolar pairing, band-limiting and cropping —
recorded **zero calls**. The structure agrees: the only such call anywhere in
this package is a degree-1 ``polyfit`` in the calibration measurement path,
which no generation run touches.

Zero calls is a stronger result than a timing comparison would have been. "No
measurable difference" is also what you would see if BLAS ran and happened not
to matter on that day's machine; "never called" cannot be a coincidence of
hardware. A dependency is carried forever, so it needs better than a plausible
story, and here the story turned out to be that there is nothing to cap.

If a future change puts real linear algebra on the generation path — a bidomain
solve, an implicit scheme, an eigendecomposition anywhere — this reasoning
expires and ``threadpoolctl`` becomes worth its weight. The measurement above is
the thing to re-run, not this paragraph to re-read.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AppliedLimits",
    "ResourceLimits",
    "ThreadCapWarning",
    "apply_thread_limits",
]


class ThreadCapWarning(UserWarning):
    """A requested thread cap could not be honoured exactly."""


@dataclass(frozen=True)
class ResourceLimits:
    """What the ``resources:`` config block asks for.

    ``max_threads`` of ``None`` means "do not touch it" — which is not the same
    as "use one thread", and is the default so that adding this block changed no
    existing run.
    """

    max_threads: int | None = None

    def __post_init__(self) -> None:
        if self.max_threads is not None and self.max_threads < 1:
            raise ValueError(
                f"resources.max_threads must be at least 1, got {self.max_threads}. "
                "Omit the key entirely to leave the thread count alone."
            )


@dataclass(frozen=True)
class AppliedLimits:
    """What was actually achieved, read back from the library.

    ``effective`` is **not** computed from ``requested``. It is whatever
    ``numba.get_num_threads()`` reports after the attempt, because the entire
    point of this type is to distinguish a cap that was applied from one that
    was merely requested.
    """

    requested: int | None
    effective: int
    maximum: int

    @property
    def clamped(self) -> bool:
        """True when more threads were asked for than the process can ever use."""
        return self.requested is not None and self.requested > self.maximum

    def describe(self) -> str:
        if self.requested is None:
            return f"threads: {self.effective} (uncapped; process maximum {self.maximum})"
        return (
            f"threads: {self.effective} (requested {self.requested}, "
            f"process maximum {self.maximum})"
        )


def apply_thread_limits(limits: ResourceLimits) -> AppliedLimits:
    """Apply ``limits`` to the current process and report what took effect.

    The launch-time maximum is a hard ceiling: numba fixes its thread pool size
    when it initialises, from ``NUMBA_NUM_THREADS`` or the core count, and
    ``set_num_threads`` can only select a number at or below it. Asking for more
    is a configuration mistake worth surfacing — but not worth aborting a
    multi-hour run over, so it warns and uses the maximum.
    """
    import warnings

    # Imported here rather than at module scope so that parsing a config, or
    # importing this package at all, does not drag in the compiler runtime.
    import numba

    maximum = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]

    if limits.max_threads is None:
        # Report the status quo without touching it, so a run that configured
        # nothing still records what it had.
        return AppliedLimits(
            requested=None,
            effective=int(numba.get_num_threads()),  # type: ignore[no-untyped-call]
            maximum=maximum,
        )

    target = min(limits.max_threads, maximum)
    if limits.max_threads > maximum:
        warnings.warn(
            f"resources.max_threads={limits.max_threads} exceeds this process's "
            f"maximum of {maximum} threads, which is fixed when numba initialises "
            f"and cannot be raised from here. Using {maximum}. To go higher, set "
            f"the NUMBA_NUM_THREADS environment variable before starting python.",
            ThreadCapWarning,
            stacklevel=2,
        )

    numba.set_num_threads(target)  # type: ignore[no-untyped-call]

    # Read back rather than assume. `target` is what we asked for; this is what
    # the library says it is doing, and only the second one is evidence.
    return AppliedLimits(
        requested=limits.max_threads,
        effective=int(numba.get_num_threads()),  # type: ignore[no-untyped-call]
        maximum=maximum,
    )
