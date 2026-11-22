"""The layer on the JAX engine's scene. Needs jax (the lab's own JAX tests run in a separate environment for the same reason)."""

import pytest

pytest.importorskip("jax")

from phost_agent import ACCEPTED, Funnel, Journal, Lease, Proposal, Task, check_ceiling
from phost_score import Ledger

SCENE = "jaxchain32:pluck"
STAGE = "dof.passive_force"
LAW = Proposal(STAGE, impl="dof.passive_force/spring-damper", params=dict(k=3.0, c=0.4, q_ref=0.1))


def test_the_jax_scene_runs_on_a_cpu_build_so_it_is_checked_and_timed_like_the_chain(tmp_path, journal):
    task = Task(SCENE, STAGE, worlds=(1,), goal="behaviour", steps=3, launches=1, repeats=2, driver="eager")
    funnel = Funnel(task, journal, ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cpu", tmp_path / "leases"))
    assert funnel.world.engine_name == "jaxchain" and funnel.world.cpu_only
    outcome = funnel.evaluate(LAW, 1)
    assert outcome.verdict == ACCEPTED and outcome.reached == "run" and outcome.klass == "model_changing"
    assert funnel.spend.timed_runs == 1 and funnel.spend.gpu_seconds == 0.0


def test_a_candidate_is_checked_on_the_jax_host_with_the_same_contract(tmp_path, journal):
    task = Task(SCENE, STAGE, worlds=(1,), goal="behaviour")
    funnel = Funnel(task, journal, ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cpu", tmp_path / "leases"),
                    timed=lambda graph, t, c: pytest.fail("this test stops before the timed run"))
    prepared = funnel.prepare(LAW)
    assert prepared.rejection is None and prepared.passed == {"static", "cheap", "real"}
    hosts = {e["tier"]: e.get("host") for e in journal.records("tier") if e["tier"] in ("cheap", "real")}
    assert hosts["real"] == "jaxchain"
