"""The layer against real simulators: worker processes, the contract checker, the lab's timed runs. CPU only: the chain engine."""

import json

import pytest

from phost_agent import (ACCEPTED, MEASURED, REJECTED, Budget, Funnel, Journal, Lease, Proposal, ScriptedProposer, Task, run_check,
                         run_loop, write_candidate)
from phost_agent import checker
from phost_lab.bootstrap import load_registry
from phost_score import Ledger
from conftest import GOOD_SOLVER, SCENE, STAGE, WRONG_SOLVER


def request(stage=STAGE, impl=f"{STAGE}/native", tier="cheap", params=None, plugins=()):
    return dict(stage=stage, impl=impl, params=params or {}, tier=tier, scene=SCENE, worlds=2, plugins=list(plugins),
                scene_modules=[], scene_dirs=[])


def source_request(source, tier="cheap", tmp_path=None):
    registry = load_registry()
    proposal = Proposal(STAGE, source=source)
    candidate = write_candidate(proposal, registry, tmp_path / "cand")
    return request(impl=proposal.impl_id, tier=tier, plugins=[candidate.module]), candidate


# ---- the worker -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("tier", ["cheap", "real"])
def test_a_builtin_law_passes_its_contract_in_a_worker_process(tier):
    done = run_check(request("dof.passive_force", "dof.passive_force/spring-damper", tier, params=dict(k=3.0, c=0.4, q_ref=0.2)))
    assert done.ok, done.message()
    assert done.host == ("chain" if tier == "real" else done.host) and done.metrics["max_law_error"] < 1e-6 and done.sandbox == "none"


def test_a_stage_no_cheap_host_offers_is_skipped_not_failed():
    from phost_agent.worker import cheap_host
    assert cheap_host("no.such.stage") is None and cheap_host(STAGE)[1] in ("fake", "chain")
    done = run_check(request("no.such.stage", "no.such.stage/x"))
    assert done.status == "skipped" and "no simulator-free host" in done.reason and not done.ok


def test_a_correct_solver_passes_both_hosts_and_a_wrong_one_is_caught_by_the_residual(tmp_path):
    for tier in ("cheap", "real"):
        req, cand = source_request(GOOD_SOLVER, tier, tmp_path)
        good = run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory)
        assert good.ok, good.message()
        req, cand = source_request(WRONG_SOLVER, tier, tmp_path)
        bad = run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory)
        assert bad.status == "fail" and "max_relative_residual" in bad.message()


def test_source_that_cannot_be_imported_is_an_error_naming_the_cause(tmp_path):
    req, cand = source_request("def broken(:\n", tmp_path=tmp_path)
    done = run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory)
    assert done.status == "error" and "SyntaxError" in done.reason and not done.transient


def test_a_check_that_runs_too_long_is_stopped_and_not_blamed_on_the_candidate(tmp_path):
    source = GOOD_SOLVER.replace("    def solve(self, s):\n", "    def solve(self, s):\n        import time\n        time.sleep(60)\n")
    req, cand = source_request(source, tmp_path=tmp_path)
    done = run_check(req, untrusted=True, allow_unsandboxed=True, timeout=3.0, candidate_dir=cand.directory)
    assert done.status == "error" and "timed out after 3 s" in done.reason and done.transient


def test_source_is_refused_when_no_sandbox_is_available_and_nobody_said_it_may_run_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "sandbox_available", lambda: False)
    req, cand = source_request(GOOD_SOLVER, tmp_path=tmp_path)
    done = run_check(req, untrusted=True, candidate_dir=cand.directory)
    assert done.status == "error" and "sandbox" in done.reason and done.transient
    assert run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory).ok


@pytest.mark.skipif(not checker.sandbox_available(), reason="needs phost-author and bwrap")
def test_source_runs_in_the_sandbox_when_there_is_one_and_cannot_read_the_users_home(tmp_path):
    import os
    secret = os.path.expanduser("~/.phost_agent_test_secret")
    snoop = GOOD_SOLVER.replace("    def solve(self, s):\n", f"    def solve(self, s):\n        open({secret!r}).read()\n")
    try:
        with open(secret, "w") as handle:
            handle.write("x")
        req, cand = source_request(GOOD_SOLVER, tmp_path=tmp_path)
        inside = run_check(req, untrusted=True, candidate_dir=cand.directory)
        assert inside.sandbox == "bwrap" and inside.ok, inside.message()
        req, cand = source_request(snoop, tmp_path=tmp_path)
        done = run_check(req, untrusted=True, candidate_dir=cand.directory)
        assert done.sandbox == "bwrap" and done.status in ("fail", "error") and "FileNotFoundError" in (done.message() + str(done.reason))
    finally:
        os.unlink(secret)


# ---- the funnel and the loop on the chain engine -------------------------------------------------------------

def funnel_for(tmp_path, journal, **options):
    task = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3, plugins=())
    return Funnel(task, journal, ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cpu", tmp_path / "leases"),
                  candidates_dir=tmp_path / "cand", allow_unsandboxed=True, **options)


def test_a_wrong_candidate_never_earns_a_timed_run(tmp_path, journal):
    funnel = funnel_for(tmp_path, journal)
    outcome = funnel.evaluate(Proposal(STAGE, source=WRONG_SOLVER), 1)
    assert outcome.verdict == REJECTED and outcome.reached == "cheap" and "max_relative_residual" in outcome.reason
    assert funnel.spend.timed_runs == 0 and not (tmp_path / "ledger.jsonl").exists()


def test_a_correct_candidate_is_timed_by_the_lab_and_judged_and_recorded_for_real(tmp_path, journal):
    funnel = funnel_for(tmp_path, journal)
    outcome = funnel.evaluate(Proposal(STAGE, source=GOOD_SOLVER), 1)
    assert outcome.verdict in (MEASURED, ACCEPTED) and outcome.reached == "run" and outcome.klass == "exact"
    assert outcome.speedup is not None and outcome.speedup > 0 and len(outcome.entries) == 2
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.verify().ok
    entries = [r.entry for r in ledger]
    assert {e["scene"] for e in entries} == {SCENE} and {e["worlds"] for e in entries} == {1, 2}
    assert all(e["eligible"] and e["stages"] == [STAGE] for e in entries)
    assert funnel.spend.timed_runs == 1 and funnel.spend.cpu_seconds > 0 and funnel.spend.gpu_seconds == 0
    assert funnel.evaluate(Proposal(STAGE, source=GOOD_SOLVER), 2).memo       # and now it is known


def test_the_ceiling_the_loop_asks_for_is_the_labs_own_measurement(tmp_path, journal):
    from phost_agent import check_ceiling
    gate, seconds = check_ceiling(Task("chain256:ringing", STAGE, worlds=(1,)), journal, lease=Lease("cpu", tmp_path))
    assert gate.ok and gate.basis == "measured" and gate.share > 0.5 and gate.ceiling > 2 and seconds > 0
    again, spent = check_ceiling(Task("chain256:ringing", STAGE, worlds=(1,)), journal, lease=Lease("cpu", tmp_path))
    assert again.basis == "memo" and spent == 0.0 and again.ceiling == gate.ceiling


def test_a_stage_that_is_a_sliver_of_the_step_is_refused_by_the_labs_numbers(tmp_path, journal):
    from phost_agent import check_ceiling
    task = Task(SCENE, "dof.passive_force", worlds=(1,), min_ceiling=1.05)
    gate, _ = check_ceiling(task, journal, lease=Lease("cpu", tmp_path))
    assert not gate.ok and "model_changing" in gate.reason                   # no speed to gain on a law that changes the physics


def test_a_whole_search_on_the_chain_engine_wrong_then_right_then_known(tmp_path, journal):
    task = Task("chain256:ringing", STAGE, worlds=(1,), steps=5, launches=2, repeats=3)
    proposals = [Proposal(STAGE, source=WRONG_SOLVER), Proposal(STAGE, source=GOOD_SOLVER), Proposal(STAGE, source=GOOD_SOLVER)]
    ledger = Ledger(tmp_path / "ledger.jsonl")
    report = run_loop(task, ScriptedProposer(proposals), budget=Budget(stall=9), journal=journal, ledger=ledger,
                      lease=Lease("cpu", tmp_path / "leases"), candidates_dir=tmp_path / "cand", allow_unsandboxed=True)
    assert report.status == "finished" and report.gate.ok
    assert [o.verdict for o in report.outcomes][0] == REJECTED and report.outcomes[1].reached == "run"
    assert report.outcomes[2].memo and report.outcomes[2].speedup == report.outcomes[1].speedup
    assert report.funnel()["proposed"] == 3 and report.spend.timed_runs == 1 and report.spend.memo_hits == 1
    assert journal.verify().ok and ledger.verify().ok and len(list(ledger)) == 1


# ---- what a candidate prints --------------------------------------------------------------------------------

PRINTING_SOLVER = GOOD_SOLVER.replace("        nv = self.host.facts().nv\n",
                                      "        nv = self.host.facts().nv\n        print('W=%d h=%s' % (s.h.shape[0], s.h.dtype))\n")


@pytest.mark.parametrize("tier", ["cheap", "real"])
def test_a_candidate_that_prints_gets_its_output_back_with_a_passing_check(tier, tmp_path):
    req, cand = source_request(PRINTING_SOLVER, tier, tmp_path)
    done = run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory)
    assert done.ok, done.message()
    assert done.output.startswith("W=") and "float" in done.output and done.to_dict()["output"] == done.output


def test_what_a_candidate_printed_comes_back_when_the_check_fails_or_the_code_raises(tmp_path):
    wrong = PRINTING_SOLVER.replace("solve(upper", "solve(1.01 * upper")
    req, cand = source_request(wrong, "cheap", tmp_path)
    failed = run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory)
    assert failed.status == "fail" and failed.output.startswith("W=")
    raising = PRINTING_SOLVER.replace("        nv = self.host.facts().nv\n", "        nv = self.host.facts().nv\n        raise RuntimeError('PROBE')\n", 1)
    req, cand = source_request(raising, "cheap", tmp_path)
    errored = run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory)
    assert errored.status == "error" and "PROBE" in errored.reason


def test_a_candidate_that_prints_nothing_returns_no_output(tmp_path):
    req, cand = source_request(GOOD_SOLVER, "cheap", tmp_path)
    assert run_check(req, untrusted=True, allow_unsandboxed=True, candidate_dir=cand.directory).output == ""


# ---- what a check's inputs are, measured by a probe the arena writes ----------------------------------------------------------------

@pytest.mark.parametrize("scene", ["chain32:pluck", "chain256:ringing"])
def test_a_probe_written_from_the_card_prints_what_a_check_hands_a_runtime_and_stops(tmp_path, journal, scene):
    from phost_agent.cards import stage_card
    from phost_agent.facts import probe_source
    card = stage_card(load_registry(), STAGE)
    funnel = Funnel(Task(scene=scene, stage=STAGE, worlds=(1,), steps=5, launches=2, repeats=3), journal,
                    lease=Lease("cpu", tmp_path / "leases"), allow_unsandboxed=True, candidates_dir=tmp_path / "cand")
    result = funnel.diagnose(Proposal(STAGE, source=probe_source(card)), "real")
    assert result.status == "error" and "probe: stopped on purpose" in result.reason
    out = result.output
    assert "host.facts(): ModelFacts(" in out and "installed: numpy=yes" in out and "cpus available" in out
    for port in card["ports"]:
        assert f"system.{port['name']}: " in out
    assert "dtype=float" in out and "shape=(" in out and "c_contiguous=" in out
