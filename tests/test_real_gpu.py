"""The layer on a real GPU scene (MuJoCo Warp). Skipped without mujoco_warp, phost-coupledchol and a CUDA device."""

import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("phost_coupledchol")
wp = pytest.importorskip("warp")
try:
    wp.init()
    HAVE_CUDA = wp.is_cuda_available()
except Exception:                                   # pragma: no cover
    HAVE_CUDA = False
pytestmark = pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device")

from phost_agent import MEASURED, ACCEPTED, REJECTED, Funnel, Lease, Proposal, Task, check_ceiling
from phost_score import Ledger

PLUGIN = "phost_coupledchol.phost_impl"
STAGE = "newton.linear_solve"
SCENE = "trampoline8:impact"
TASK = Task(SCENE, STAGE, worlds=(8,), steps=10, launches=3, repeats=5, plugins=(PLUGIN,))


def funnel(tmp_path, journal, **options):
    return Funnel(TASK, journal, ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cuda:0", tmp_path / "leases"), **options)


def test_the_ceiling_is_the_labs_measurement_of_the_stage_on_this_gpu(tmp_path, journal):
    gate, seconds = check_ceiling(TASK, journal, lease=Lease("cuda:0", tmp_path / "leases"))
    assert gate.ok and 0.0 < gate.share < 1.0 and gate.ceiling > 1.0 and seconds > 0 and gate.basis == "measured"


def judged(f, proposal, tries=4):
    """The outcome of a timed run the scorer accepted as a measurement. The accuracy ratio on a GPU is against the native
    control's own noise and sometimes lands over its gate for a candidate that is fine; that is not remembered, so ask again."""
    for attempt in range(1, tries + 1):
        outcome = f.evaluate(proposal, attempt)
        if not outcome.reason.startswith("accuracy ratio"):
            return outcome
    pytest.fail(f"the accuracy ratio was over its gate in {tries} runs in a row: {outcome.reason}")


def test_a_registered_implementation_is_checked_on_the_gpu_scene_under_the_lease_and_then_timed(tmp_path, journal):
    f = funnel(tmp_path, journal)
    assert f.world.engine_name == "mjwarp" and not f.world.cpu_only
    proposal = Proposal(STAGE, impl="newton.linear_solve/coupled")
    outcome = judged(f, proposal)
    assert outcome.reached == "run" and outcome.verdict in (MEASURED, ACCEPTED) and outcome.speedup > 0, outcome.reason
    assert f.spend.gpu_seconds > 0 and f.spend.timed_runs >= 1
    tiers = {e["tier"]: e["ok"] for e in journal.records("tier")}
    assert tiers == {"static": True, "cheap": True, "real": True, "run": True}
    assert Ledger(tmp_path / "ledger.jsonl").verify().ok and f.evaluate(proposal, 9).memo


def test_a_candidate_that_is_wrong_on_the_scene_is_stopped_by_the_real_check_not_the_timed_run(tmp_path, journal):
    wrong = '''
import numpy as np
from phost.testing.runtimes import ReferenceRuntime


class Runtime(ReferenceRuntime):
    def solve(self, s):
        super().solve(s)
        if hasattr(s.x, "assign"):
            s.x.assign(s.x.numpy() * 1.5)
        else:
            s.x *= 1.5
'''
    f = funnel(tmp_path, journal, allow_unsandboxed=True, timed=lambda *a: pytest.fail("a wrong candidate must not be timed"))
    outcome = f.evaluate(Proposal(STAGE, source=wrong), 1)
    assert outcome.verdict == REJECTED and outcome.reached in ("cheap", "real") and f.spend.timed_runs == 0
