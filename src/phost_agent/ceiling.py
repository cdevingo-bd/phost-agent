"""Tier 0: how much a replacement could possibly gain, asked before anything is proposed or compiled.

The lab times the host's own version of each stage as a piece of the step (``opportunities``). What an infinitely fast
replacement would give is ``1 / (1 - share)``; a stage that is a few percent of the step is not worth an attempt, however
clever the candidate, and an agent that is told so up front does not spend a GPU on it. For a stage the lab could only time
inside a larger phase the share is an upper bound and so is the figure: if even the bound is small the refusal stands.

Stages that are not a speed swap (they change the physics or only observe) have no ceiling. For the ``speed`` goal they are
refused; for the ``behaviour`` goal the gate passes them, because the loop is then after a candidate that passes its
checks, not one that is fast. A stage the lab has no native timing for is passed with its ceiling marked unknown.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import phost

from .journal import Journal
from .lease import Lease
from .memo import key_for
from .task import Task, World, resolve

SPEED_CLASSES = ("exact", "approximate")


@dataclass(frozen=True)
class Gate:
    ok: bool
    reason: str
    scene: str
    stage: str
    swap_class: str
    share: float | None = None          # fraction of the step the stage takes
    ceiling: float | None = None        # the most a replacement could gain (an upper bound when ``enclosing``)
    native_ms: float | None = None
    step_ms: float | None = None
    enclosing: str | None = None
    worlds: int | None = None           # the world count the figure is for (the most favourable one)
    basis: str = "measured"             # measured | memo | supplied
    min_ceiling: float = 1.05

    @property
    def bounded(self) -> bool:
        return self.ceiling is not None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _bound(entry: Mapping[str, Any]) -> float | None:
    """The ceiling the lab gave, else the bound its share implies (a stage timed inside a larger phase has only a bound)."""
    if entry.get("ceiling") is not None:
        return float(entry["ceiling"])
    share = entry.get("share")
    if share is None:
        return None
    return math.inf if share >= 1.0 else 1.0 / (1.0 - float(share))


def gate_from_report(report: Mapping[str, Any], scene: str, stage: str, *, swap_class: str, goal: str = "speed",
                     min_ceiling: float = 1.05, basis: str = "measured") -> Gate:
    """Judge a lab ``opportunities`` report (``{"cells": [{"worlds", "step_ms", "stages": {...}}]}``) for one stage."""
    common = dict(scene=scene, stage=stage, swap_class=swap_class, basis=basis, min_ceiling=min_ceiling)
    cells = [c for c in report.get("cells", ()) if stage in c.get("stages", {})]
    if not cells:
        return Gate(False, f"the scene's host does not offer stage {stage!r}", **common)
    applied = [c for c in cells if c["stages"][stage].get("applies", True)]
    if not applied:
        return Gate(False, f"the host refuses this stage on this scene: {cells[0]['stages'][stage].get('reason')}", **common)
    if goal == "speed" and swap_class not in SPEED_CLASSES:
        return Gate(False, f"{stage} is a {swap_class} stage: it has no speed to gain, so a speed search cannot rank a "
                           f"swap of it (use the behaviour goal)", **common)
    best = max(applied, key=lambda c: (_bound(c["stages"][stage]) or 0.0, c["worlds"]))
    entry = best["stages"][stage]
    bound = _bound(entry)
    figures = dict(share=entry.get("share"), ceiling=bound, native_ms=entry.get("native_ms"), step_ms=best.get("step_ms"),
                   enclosing=entry.get("enclosing"), worlds=best["worlds"])
    if goal == "behaviour":
        return Gate(True, "behaviour goal: no speed ceiling applies", **common, **figures)
    if bound is None:
        return Gate(True, "the host has no native timing for this stage, so its ceiling is unknown (not refused)", **common, **figures)
    if bound < min_ceiling:
        what = "at most" if entry.get("enclosing") else "even a free replacement gains"
        return Gate(False, f"{what} {bound:.3f}x: the stage is {100 * (entry.get('share') or 0):.1f}% of the step, below the "
                           f"{min_ceiling:g}x needed to be worth an attempt", **common, **figures)
    return Gate(True, f"up to {bound:.2f}x available ({100 * (entry.get('share') or 0):.0f}% of the step)", **common, **figures)


Measure = Callable[[Task, World], Mapping[str, Any]]


def measure_opportunities(task: Task, world: World) -> Mapping[str, Any]:
    """Run the lab's opportunities report for the task's scene in a worker process. Returns its report, or ``{"error": ...}``."""
    from phost_lab.graph import NATIVE
    from phost_lab.isolate import run_isolated
    from phost_lab.runner import RunConfig

    result = run_isolated(NATIVE, task.scene, RunConfig(worlds=task.worlds), timeout=task.run_timeout, plugins=task.plugins,
                          scene_modules=task.scene_modules, scene_dirs=task.scene_dirs,
                          report=dict(kinds=["opportunities"], worlds=list(task.worlds)))
    if "error" in result:
        return dict(error=result["error"])
    return result["opportunities"]


def _identity(task: Task, world: World) -> dict[str, Any]:
    return dict(what="ceiling", scene=task.scene, engine=world.engine_name, worlds=list(task.worlds), phost=phost.__version__)


def check_ceiling(task: Task, journal: Journal, *, registry: phost.Registry | None = None, measure: Measure | None = None,
                  lease: Lease | None = None, refresh: bool = False, max_age_seconds: float = 24 * 3600.0,
                  now: Callable[[], float] = time.time) -> tuple[Gate, float]:
    """The gate for ``task`` and the device seconds measuring it cost (0 when a recent measurement was reused).

    The measurement is the whole scene's report, kept in the journal under a content key, so every stage's gate on this
    scene costs one measurement, and so does asking again within ``max_age_seconds``.
    """
    world = resolve(task, registry)
    swap_class = world.registry.contract(task.stage).swap_class.value
    if world.unavailable:
        return Gate(False, f"the scene's engine is not usable here: {world.unavailable}", scene=task.scene, stage=task.stage,
                    swap_class=swap_class, min_ceiling=task.min_ceiling), 0.0
    key = key_for(_identity(task, world))
    report, basis, seconds = None, "measured", 0.0
    if not refresh:
        for entry in reversed(journal.records("ceiling")):
            if entry.get("key") == key and now() - entry.get("measured_at", 0.0) <= max_age_seconds:
                report, basis = entry["report"], "memo"
                break
    if report is None:
        lease = lease or Lease(world.lease_device)
        started = time.monotonic()
        with lease.hold(f"ceiling {task.scene}", timeout=task.run_timeout):
            report = measure(task, world) if measure else measure_opportunities(task, world)
        seconds = time.monotonic() - started
        if "error" in report:
            return Gate(False, f"the ceiling could not be measured: {report['error']}", scene=task.scene, stage=task.stage,
                        swap_class=swap_class, min_ceiling=task.min_ceiling), seconds
        journal.append("ceiling", key=key, scene=task.scene, engine=world.engine_name, worlds=list(task.worlds),
                       measured_at=now(), seconds=round(seconds, 3), report=report)
    return gate_from_report(report, task.scene, task.stage, swap_class=swap_class, goal=task.goal,
                            min_ceiling=task.min_ceiling, basis=basis), seconds


def recorded_gate(task: Task, journal: Journal, *, registry: phost.Registry | None = None) -> Gate | None:
    """The gate from the journal's latest measurement of this scene, or None if it was never measured. Nothing is run."""
    world = resolve(task, registry)
    key = key_for(_identity(task, world))
    swap_class = world.registry.contract(task.stage).swap_class.value
    for entry in reversed(journal.records("ceiling")):
        if entry.get("key") == key:
            return gate_from_report(entry["report"], task.scene, task.stage, swap_class=swap_class, goal=task.goal,
                                    min_ceiling=task.min_ceiling, basis="memo")
    return None
