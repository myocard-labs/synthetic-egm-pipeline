"""What a simulation costs, and the two factors behind it.

**Runtime cost is a deliverable here, not a nicety.** Three open decisions
consume it — whether to keep this solver, whether to reserve the ionic model for
studies that need it, and how large a bank the data plan can ask for — and until
now the figure feeding them was an estimate nobody could re-derive. This module
is what makes it re-derivable.

What is asserted, and what is only recorded
-------------------------------------------
**Absolute wall-clock is RECORDED, never asserted.** A per-simulation time is a
fact about a machine as much as about this code, so an assertion on one fails on
whatever hardware it was not written on — and would then be "fixed" by widening
it until it meant nothing. Every recorded number therefore carries its machine,
its mesh pitch and its patch size; a wall-clock number without those is not a
measurement.

**Ratios ARE asserted**, because they are properties of the code and the cards
rather than of the CPU, and they survive being carried to another machine. Two
of them:

- the **membrane factor** — Courtemanche's per-node-step cost over
  Aliev-Panfilov's, measured on one mesh with one step count and no trackers, so
  that only the membrane differs;
- the **mesh factor** — nodes times timesteps between the two operating points,
  which is arithmetic over the shipped cards and configs.

Between them they answer "did this change move the benchmarks": an ionic kernel
that became three times slower moves the first, and a card or config that
changed pitch or timestep moves the second.

Why the mesh factor is *not* benchmark-marked
---------------------------------------------
It runs no solver and takes no timing — it is arithmetic over two YAML files and
two cards. There is no reason to hide a free check behind a marker somebody has
to remember to pass, so it runs in the ordinary fast suite and guards the cards
on every change. Only the two that need a stopwatch are marked.

Running them
------------
``pytest -m benchmark -s`` — ``-s`` because the recorded numbers are printed,
and a benchmark whose output you cannot see has measured nothing. They are
excluded from the default suite *and* from CI's slow suite; see
``project/benchmarks.md`` for the committed results and the reasoning.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from myocard_synthetic_egm_pipeline.simulate import Patch2DGeometry
from myocard_synthetic_egm_pipeline.simulate.cell_models import CellModelSpec
from myocard_synthetic_egm_pipeline.simulate.model_cards import load_model_card

# --- the two operating points, each at the pitch it is actually run at -------
#
# NOT a like-for-like membrane comparison, and it must not be read as one.
# Courtemanche's cards are solved at 0.1 mm and a 0.25 mm run borrows an anchor
# and reads its upstroke ~10 % high, so a 0.25-against-0.25 table would compare
# two membranes over a configuration nobody should generate from. What these
# rows measure is the OPERATIONAL cost: what each model costs to run as the
# project actually runs it.

AP_CONFIG = Path("examples/synthegm_v1_baseline.yaml")
CRN_CONFIG = Path("examples/synthegm_courtemanche.yaml")

AP_CARD = "af_remodelled_220ms"
AP_DR_MM = 0.25
CRN_CARDS = ("courtemanche_control", "af_remodelled_crn_220ms")
CRN_DR_MM = 0.10

#: Mesh for the membrane-factor control. Both models run on the same node
#: count for the same step counts, so the ratio isolates the membrane.
MEMBRANE_GRID = 160

#: Two step counts, because the cost of a run is **not** proportional to its
#: length and pretending otherwise gave a wrong answer once already.
#:
#: ``model.run()`` carries a fixed setup cost — thread-pool spin-up and array
#: allocation — that a short run divides over very few steps. Measured on a
#: 160^2 mesh it is ~429 ms, which against Aliev-Panfilov's 0.75 s at 2000
#: steps is **57 % of the measurement**. Reading ns/node-step off a single
#: short run therefore reported 12.6 ns for a model whose marginal cost is
#: 6.3, and the membrane factor built on that came out near 18x rather than
#: its true value.
#:
#: Taking the difference between two lengths cancels the fixed term exactly,
#: for either model, without needing to know what it is::
#:
#:     ns/node-step = (t_long - t_short) / (steps_long - steps_short) / nodes
#:
#: This is why the constant is a pair. A single length can only be made
#: trustworthy by growing it until the overhead is negligible, which for the
#: ionic model would mean ten-minute repetitions.
#: Chosen **per model**, so that every timed run lasts tens of seconds.
#:
#: The step counts differ between the two and that is deliberate: the quantity
#: is normalised by nodes *and* steps, so it is comparable across different
#: lengths, while a shared count would make one of the two runs far too short.
#: At 2000 steps Aliev-Panfilov finishes in well under a second, and on a
#: 4-core laptop that someone is also using, thread scheduling then dominates
#: — repeating that measurement gave 6.3 ns once and 18.6 ns an hour later,
#: a 3x swing with no code change between them. Courtemanche never showed it,
#: because its runs were already tens of seconds long.
#: The Aliev-Panfilov pair spans 10x rather than 4x for the same reason its
#: runs are long: its kernel is cheap enough that per-step threading overhead
#: is a large share of each step, so only a wide baseline makes the slope
#: large compared with the scatter around it.
MEMBRANE_STEPS: dict[str, tuple[int, int]] = {
    "aliev_panfilov": (40_000, 400_000),
    "courtemanche": (2_000, 8_000),
}

#: Repetitions of each timed run; the **minimum** is kept, not the mean.
#: Interference from other processes can only ever add time, so on a machine
#: someone is also using, the smallest observation is the closest one to the
#: cost of the code. A mean would let a browser tab into the measurement.
MEMBRANE_REPEATS = 3


@dataclass(frozen=True)
class Machine:
    """Who took the measurement. Recorded beside every wall-clock number."""

    product: str
    cpu: str
    cores: str
    python: str

    @classmethod
    def here(cls) -> Machine:
        def read(path: str, fallback: str) -> str:
            try:
                return Path(path).read_text(encoding="utf-8").strip()
            except OSError:
                return fallback

        return cls(
            product=read("/sys/devices/virtual/dmi/id/product_name", platform.node()),
            cpu=platform.processor() or platform.machine(),
            cores=str(os_cpu_count()),
            python=platform.python_version(),
        )

    def __str__(self) -> str:
        return f"{self.product} / {self.cpu} / {self.cores} logical cores / py{self.python}"


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def _dt_ms(solved: CellModelSpec) -> float:
    """The integration step in **milliseconds**, whatever the model's time base.

    Aliev-Panfilov's ``dt_model_units`` is dimensionless and needs its
    ``time_unit_ms``; Courtemanche's already is a millisecond. Asking through
    the spec rather than branching on the class keeps this correct for a third
    model that has not been written.
    """
    return float(solved.dt_model_units) * float(getattr(solved, "time_unit_ms", 1.0))


def _load_config(path: Path) -> Any:
    from myocard_synthetic_egm_pipeline.cli._config import build_generate_dataset_config

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["_config_dir"] = path.parent
    return build_generate_dataset_config(doc)


def _node_steps(*, grid: int, capture_ms: float, dt_ms: float) -> float:
    """Nodes times timesteps — the solver work one simulation asks for."""
    return (grid * grid) * (capture_ms / dt_ms)


# ---------------------------------------------------------------------------
# The mesh factor — arithmetic, so it runs in the fast suite
# ---------------------------------------------------------------------------


def test_the_mesh_factor_between_the_two_operating_points() -> None:
    """How much more solver work the ionic operating point asks for, before any
    per-node cost is counted.

    **Not 39x**, which is what the project carried for a while and which this
    test exists partly to keep from coming back. That figure assumed
    ``dt`` scales as ``dr**2`` all the way down, and it does not: at 0.25 mm
    Courtemanche's step is capped by the sodium current's ionic ceiling rather
    than by the diffusion bound, so refining the mesh only pays the quadratic
    penalty over part of the range. It is also a different comparison — 39x was
    Courtemanche against *itself* at a coarser pitch, whereas the number that
    matters operationally is Courtemanche against Aliev-Panfilov, each at the
    pitch its own card was solved for.

    Both configs capture for the same duration, so the ratio is nodes and
    timestep alone with nothing folded in from trace length.
    """
    ap_cfg = _load_config(AP_CONFIG)
    crn_cfg = _load_config(CRN_CONFIG)

    ap_capture = ap_cfg.run_config.effective_capture_duration_ms
    crn_capture = crn_cfg.run_config.effective_capture_duration_ms
    assert ap_capture == pytest.approx(crn_capture), (
        "the two example configs no longer capture for the same duration, so a "
        "mesh factor computed from them would fold in trace length as well"
    )

    ap_work = _node_steps(
        grid=ap_cfg.geometry.n_cells_per_edge,
        capture_ms=ap_capture,
        dt_ms=_dt_ms(ap_cfg.cell_model),
    )

    factors = {}
    for card_name in CRN_CARDS:
        card = load_model_card(card_name, dr_mm=CRN_DR_MM, dr_model_units=CRN_DR_MM)
        crn_work = _node_steps(
            grid=crn_cfg.geometry.n_cells_per_edge,
            capture_ms=crn_capture,
            dt_ms=_dt_ms(card.solved),
        )
        factors[card_name] = crn_work / ap_work

    # Measured 2026-08-27: 7.46 for the control card, 7.87 for the AF one. They
    # differ only through dt -- the AF card's smaller diffusion-bound step --
    # and 2 % admits a card re-solve that moves the last digit while refusing
    # anything that moved the pitch or the timestep rule.
    assert factors["courtemanche_control"] == pytest.approx(7.46, rel=0.02), factors
    assert factors["af_remodelled_crn_220ms"] == pytest.approx(7.87, rel=0.02), factors


# ---------------------------------------------------------------------------
# The membrane factor — needs a stopwatch, so it is benchmark-marked
# ---------------------------------------------------------------------------


def _bare_solve_seconds(*, solved: CellModelSpec, dr_mm: float, grid: int, steps: int) -> float:
    """Integrate ``steps`` steps on a ``grid`` square mesh with **no trackers**.

    No trackers on purpose. An activation tracker compares the whole mesh on
    every step and an electrogram tracker reduces it per electrode; both cost
    the same for either model, so leaving them in would dilute exactly the
    difference this function exists to isolate.
    """
    import finitewave as fw

    from myocard_synthetic_egm_pipeline.backends.finitewave import backend as bk
    from myocard_synthetic_egm_pipeline.simulate import (
        PlanarEdgeStimulus,
        UniformRandomFibrosis,
    )

    geometry = Patch2DGeometry(size_mm=grid * dr_mm, dr_mm=dr_mm)
    model = bk._build_model_2d(cell_model=solved, geometry=geometry, dr_model_units=dr_mm).model
    tissue = bk._build_tissue_2d(shape=geometry.shape, fiber_angle_rad=geometry.fiber_angle_rad)
    bk._apply_substrate_2d(
        tissue=tissue,
        strategy=UniformRandomFibrosis(density=0.0),
        rng=np.random.default_rng(0),
    )
    model.cardiac_tissue = tissue
    bk._install_activation_2d(model=model, source=PlanarEdgeStimulus(edge="left"), tissue=tissue)
    model.t_max = steps * solved.dt_model_units
    model.tracker_sequence = fw.TrackerSequence()

    start = time.perf_counter()
    model.run()
    return time.perf_counter() - start


def _ns_per_node_step(*, name: str, solved: CellModelSpec, dr_mm: float) -> float:
    """Marginal cost of one node advanced one step, in nanoseconds.

    The *slope* of seconds against step count, not a single run divided by its
    own length — see :data:`MEMBRANE_STEPS` for why that difference is the
    whole measurement. Best-of-N at each length, because interference from
    other processes can only ever add time, so the smallest observation is the
    one closest to the cost of the code.

    Prints every repetition rather than only the minimum it keeps. A benchmark
    that shows one number cannot be told apart from a benchmark that got the
    same number twice by luck, and the spread here is the reader's only signal
    that the machine was quiet enough to believe.
    """
    # Numba compiles on first call; a short throwaway run pays that so it is not
    # billed to the first timed repetition.
    _bare_solve_seconds(solved=solved, dr_mm=dr_mm, grid=MEMBRANE_GRID, steps=50)
    short_steps, long_steps = MEMBRANE_STEPS[name]

    def best(steps: int) -> float:
        runs = [
            _bare_solve_seconds(solved=solved, dr_mm=dr_mm, grid=MEMBRANE_GRID, steps=steps)
            for _ in range(MEMBRANE_REPEATS)
        ]
        print(f"    {name} {steps:>7d} steps: {', '.join(f'{r:.2f} s' for r in runs)}")
        return min(runs)

    slope = (best(long_steps) - best(short_steps)) / (long_steps - short_steps)
    return 1e9 * slope / (MEMBRANE_GRID * MEMBRANE_GRID)


@pytest.mark.benchmark
def test_the_membrane_factor_is_the_ionic_model_alone() -> None:
    """Courtemanche's per-node cost over Aliev-Panfilov's: 21 state variables
    against 2, with the mesh held identical so nothing else can contribute.

    The portable half of the cost story. Wall-clock moves with the machine, but
    the *ratio* of two kernels timed back to back on one mesh does not move
    much — which is what lets it be asserted at all, and what makes it the
    thing that would catch an ionic kernel becoming three times slower.
    """
    machine = Machine.here()
    ap = load_model_card(AP_CARD, dr_mm=AP_DR_MM, dr_model_units=AP_DR_MM).solved
    crn = load_model_card(CRN_CARDS[0], dr_mm=CRN_DR_MM, dr_model_units=CRN_DR_MM).solved

    ap_ns = _ns_per_node_step(name="aliev_panfilov", solved=ap, dr_mm=AP_DR_MM)
    crn_ns = _ns_per_node_step(name="courtemanche", solved=crn, dr_mm=CRN_DR_MM)
    factor = crn_ns / ap_ns

    print(f"\n--- membrane factor, bare solve, no trackers --- {machine}")
    print(f"  mesh {MEMBRANE_GRID}^2, slope of two lengths, best of {MEMBRANE_REPEATS} at each")
    print(f"  aliev_panfilov  {ap_ns:7.1f} ns/node-step")
    print(f"  courtemanche    {crn_ns:7.1f} ns/node-step")
    print(f"  membrane factor {factor:7.2f}x")

    # Measured 21.3x and 23.2x on two runs on a Surface Pro 7, with
    # Aliev-Panfilov at 8.4-8.9 ns and Courtemanche at 179-207.
    #
    # The band is wide, and honestly so. Two independent things widen it:
    # vectorisation and cache behaviour genuinely differ between CPUs, and this
    # measurement was taken on an interactive laptop where run-to-run scatter
    # reached 20 % even after outlier rejection. A tighter band would be a
    # claim the measurement does not support, and would fail on the desktop for
    # reasons having nothing to do with the code.
    #
    # It is still not unbounded, which is the point: an ionic kernel that got
    # 3x slower lands near 65, and a Courtemanche model that quietly stopped
    # being one lands near 1. Both are refused.
    assert 12.0 <= factor <= 40.0, (
        f"membrane factor {factor:.2f}x is outside the band 21-23x was measured "
        f"in ({ap_ns:.1f} ns AP against {crn_ns:.1f} ns CRN)"
    )


# ---------------------------------------------------------------------------
# Operational wall-clock — recorded only
# ---------------------------------------------------------------------------


def _operational_seconds(*, config_path: Path, n_sims: int) -> dict[str, Any]:
    """Time whole simulations through the path a generation run actually takes.

    Not a bare solve: this includes substrate realisation, the electrogram
    tracker, bipolar pairing, band-limiting and the crop — everything a bank
    pays for. That is the number a data plan needs, and it is meaningfully
    larger than the solver alone.
    """
    import dataclasses

    from myocard_synthetic_egm_pipeline.backends.finitewave.backend import FinitewaveBackend
    from myocard_synthetic_egm_pipeline.cli.generate_dataset_cmd import _build_dataset_config
    from myocard_synthetic_egm_pipeline.simulate.dataset import _sample_specs
    from myocard_synthetic_egm_pipeline.simulate.runner import run_single

    cli_cfg = _load_config(config_path)
    cfg = _build_dataset_config(cli_cfg, show_progress=False)
    # Narrowed rather than assumed: the geometry field is the Protocol, and a
    # patch is the only shape whose node count these numbers mean anything for.
    assert isinstance(cfg.geometry, Patch2DGeometry)
    backend = FinitewaveBackend()

    def one(config: Any, geometry: Any, rng: np.random.Generator, position: Any) -> float:
        substrate, activation, electrodes = _sample_specs(config=config, sim_rng=rng)
        start = time.perf_counter()
        run_single(
            geometry=geometry,
            substrate=substrate,
            activation=activation,
            electrodes=electrodes,
            backend=backend,
            cell_model=config.cell_model,
            config=config.run_config,
            rng=rng,
            position_generator=position,
            detection_preprocessor=config.detection_preprocessor,
        )
        return time.perf_counter() - start

    # Compile the kernels on a small patch first. 12 mm is the floor that still
    # fits the 8 mm electrode grid, so the same specs are sampled; the capture
    # is shortened and cropping dropped because none of that needs to be real
    # to make numba compile.
    warm_cfg = dataclasses.replace(
        cfg,
        geometry=dataclasses.replace(cfg.geometry, size_mm=12.0),
        run_config=dataclasses.replace(
            cfg.run_config, trace_duration_ms=64.0, capture_duration_ms=None
        ),
    )
    one(warm_cfg, warm_cfg.geometry, np.random.default_rng(0), None)

    master = np.random.default_rng(cfg.master_seed)
    times = []
    for _ in range(n_sims):
        rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        times.append(one(cfg, cfg.geometry, rng, cli_cfg.position_generator))

    capture_ms = cfg.run_config.effective_capture_duration_ms
    grid = cfg.geometry.n_cells_per_edge
    work = _node_steps(grid=grid, capture_ms=capture_ms, dt_ms=_dt_ms(cfg.cell_model))
    return {
        "dr_mm": cfg.geometry.dr_mm,
        "size_mm": cfg.geometry.size_mm,
        "grid": grid,
        "dt_ms": _dt_ms(cfg.cell_model),
        "capture_ms": capture_ms,
        "node_steps": work,
        "times_s": times,
        "fastest_s": min(times),
        "ns_per_node_step": 1e9 * min(times) / work,
    }


def _report(name: str, result: dict[str, Any], machine: Machine) -> None:
    print(f"\n--- operational: {name} --- {machine}")
    print(
        f"  dr {result['dr_mm']} mm, {result['size_mm']} mm patch, "
        f"{result['grid']}^2 nodes, dt {result['dt_ms']:.6g} ms, "
        f"capture {result['capture_ms']:g} ms"
    )
    print(f"  per sim: {', '.join(f'{t:.1f} s' for t in result['times_s'])}")
    print(
        f"  fastest: {result['fastest_s']:.1f} s  ({result['ns_per_node_step']:.1f} ns/node-step)"
    )
    for n in (100, 1000):
        print(f"  {n}-sim bank: {n * result['fastest_s'] / 3600:.1f} h (loop is sequential)")


@pytest.mark.benchmark
def test_record_aliev_panfilov_operational_cost() -> None:
    """Wall-clock per simulation at the Aliev-Panfilov operating point.

    **Records; asserts nothing about the duration.** The only assertion is that
    a simulation happened at the geometry claimed, because a recorded cost for
    the wrong mesh is worse than no record.
    """
    machine = Machine.here()
    result = _operational_seconds(config_path=AP_CONFIG, n_sims=3)
    _report("aliev_panfilov", result, machine)

    assert result["grid"] == 160
    assert result["dr_mm"] == pytest.approx(AP_DR_MM)


@pytest.mark.benchmark
def test_record_courtemanche_operational_cost() -> None:
    """Wall-clock per simulation at the Courtemanche operating point.

    **Tens of minutes per simulation**, which is the finding rather than an
    inconvenience: it is what a bank-size decision has to be made against. Two
    repetitions rather than three for that reason.
    """
    machine = Machine.here()
    result = _operational_seconds(config_path=CRN_CONFIG, n_sims=2)
    _report("courtemanche", result, machine)

    assert result["grid"] == 400
    assert result["dr_mm"] == pytest.approx(CRN_DR_MM)
