"""Shared pieces: a private state directory per test, and lab results and check results built by hand."""

import pytest

from phost_agent import CheckResult, Journal, Task

STAGE = "effective.solve"
SCENE = "chain32:pluck"

GOOD_SOLVER = '''
"""A float64 dense solve of the effective system."""
import numpy as np


class Runtime:
    def __init__(self, host, params):
        self.host = host

    def solve(self, s):
        nv = self.host.facts().nv
        h, rhs = np.asarray(s.h, np.float64), np.asarray(s.rhs, np.float64)
        for w in range(h.shape[0]):
            upper = np.triu(h[w, :nv, :nv])
            s.qacc[w, :nv] = np.linalg.solve(upper + np.triu(upper, 1).T, rhs[w, :nv])

    def metrics(self):
        return {"status": 0}
'''

WRONG_SOLVER = GOOD_SOLVER.replace("s.qacc[w, :nv] = np.linalg.solve(upper + np.triu(upper, 1).T, rhs[w, :nv])",
                                   "s.qacc[w, :nv] = 1.01 * np.linalg.solve(upper + np.triu(upper, 1).T, rhs[w, :nv])")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test keeps its journal, ledger, leases and candidates under its own directory."""
    monkeypatch.setenv("PHOST_HOME", str(tmp_path / "phost"))
    return tmp_path / "phost"


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.jsonl")


@pytest.fixture
def task():
    return Task(scene=SCENE, stage=STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)


def passing(host="chain"):
    return CheckResult("pass", host=host, metrics={"max_relative_residual": 1e-9})


def make_result(graph, speedup=2.0, scene=SCENE, klass="exact", other=0, spread=0.01, accuracy=1.0, worlds=(1, 2)):
    """What ``phost_lab.runner.run_graph`` returns, built by hand."""
    stage = next(iter(graph.stages))
    cells = [dict(worlds=w, step_ms=1.0, native_ms=speedup, speedup=speedup, native_spread=spread, rms_dqpos=1e-9,
                  control_rms_dqpos=0.0, accuracy_ratio=accuracy, stage_metrics={stage: dict(calls=5)}, advisories=[],
                  other_gpu_processes=other,
                  verdict=dict(ok=True, problems=[], notes=[], gates=dict(finite=True, accuracy=True, stage=True)))
             for w in worlds]
    return dict(graph=graph.to_dict(), scene=scene, scene_hidden=False, stage_classes={stage: klass},
                config=dict(worlds=list(worlds), steps=5, driver="eager"),
                meta=dict(device="cpu", nv=32, timestep=0.002, date="2026-01-01", repeatable=True), cells=cells,
                comparable_to_native=True, **{"pass": True})
