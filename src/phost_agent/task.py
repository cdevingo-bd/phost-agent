"""One search: which stage to improve, on which scene, how it is measured and what counts as a win."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import phost
from phost_lab.bootstrap import load_registry, load_scenes
from phost_lab.engine import engine_for, get_engine, unavailable
from phost_lab.scene import Scene


@dataclass(frozen=True)
class Task:
    scene: str
    stage: str
    worlds: tuple[int, ...] = (1, 8)       # world counts a timed run measures
    steps: int = 20                         # steps replayed for the accuracy comparison
    launches: int = 5
    repeats: int = 11
    driver: str = "graph"                   # how the host is stepped while timed (phost_lab.driver): "graph", "eager", or a plugin's
    min_ceiling: float = 1.05               # tier 0: refuse a stage no replacement could speed up by this much
    min_speedup: float = 1.0                # a candidate must beat this, on every world count, to count as a win
    goal: str = "speed"                     # "speed": rank swaps on speedup. "behaviour": model-changing and observer stages, which
                                            # have no ceiling (the funnel still checks them; a win is just a candidate that passes)
    plugins: tuple[str, ...] = ()           # modules exposing ``register(registry)``
    scene_modules: tuple[str, ...] = ()
    scene_dirs: tuple[str, ...] = ()
    label: str = ""
    run_timeout: float = 900.0
    check_timeout: float = 300.0

    def __post_init__(self):
        object.__setattr__(self, "worlds", tuple(int(w) for w in self.worlds))
        for name in ("plugins", "scene_modules", "scene_dirs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.goal not in ("speed", "behaviour"):
            raise ValueError(f"goal is 'speed' or 'behaviour', not {self.goal!r}")
        if not self.worlds or min(self.worlds) < 1:
            raise ValueError("worlds must be positive counts")
        if self.min_ceiling < 1.0:
            raise ValueError("min_ceiling is a speedup bound, at least 1")

    @property
    def id(self) -> str:
        text = repr((self.scene, self.stage, self.worlds, self.steps, self.goal, self.driver, self.plugins, self.scene_modules, self.scene_dirs))
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return dict(id=self.id, scene=self.scene, stage=self.stage, worlds=list(self.worlds), steps=self.steps,
                    launches=self.launches, repeats=self.repeats, driver=self.driver, min_ceiling=self.min_ceiling,
                    min_speedup=self.min_speedup, goal=self.goal, plugins=list(self.plugins),
                    scene_modules=list(self.scene_modules), scene_dirs=list(self.scene_dirs), label=self.label)


@dataclass
class World:
    """What a task resolves to on this machine."""

    registry: phost.Registry
    scene: Scene
    engine_name: str
    device_kind: str                        # "cpu", "cuda", or "unknown" (treated as exclusive)
    unavailable: str | None
    graph_host: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def cpu_only(self) -> bool:
        return self.device_kind == "cpu"

    @property
    def lease_device(self) -> str:
        return "cpu" if self.cpu_only else "cuda:0"


def resolve(task: Task, registry: phost.Registry | None = None) -> World:
    """The registry (stage contracts, built-in implementations, the task's plugins) and the scene the task names."""
    registry = load_registry(task.plugins, registry)
    scene = load_scenes(task.scene_modules, task.scene_dirs).get(task.scene)
    engine = engine_for(scene)
    contract = registry.contract(task.stage)             # raises ContractError for a stage that does not exist
    del contract
    return World(registry, scene, engine.name, getattr(engine, "device_kind", "unknown"), unavailable(scene), engine.name)


__all__ = ["Task", "World", "resolve", "get_engine"]
