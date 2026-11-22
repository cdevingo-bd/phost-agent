"""The loop's stopping rules, bookkeeping and fan-out, over fakes. One real run is in test_real.py."""

import threading
import time

import pytest

import phost
from phost import Impl, ImplKind, ParamSpec
from phost_agent import (ACCEPTED, MEASURED, REJECTED, Budget, CheckResult, Funnel, Journal, Lease, Proposal, ProposerError,
                         ScriptedProposer, Task, run_loop, summarize)
from phost_lab.bootstrap import load_registry
from phost_score import Ledger
from conftest import SCENE, STAGE, make_result, passing

FAST = f"{STAGE}/fast"


@pytest.fixture(scope="module")
def registry():
    reg = load_registry(registry=phost.Registry())
    reg.add_impl(Impl(id=FAST, stage=STAGE, stage_version=1, version="1", title="fast", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: None, params=(ParamSpec("gain", "float", 1.0, min=0.0, max=100.0),)))
    return reg


def gains(*values):
    return [Proposal(STAGE, impl=FAST, params={"gain": float(g)}) for g in values]


def gate_report(share=0.5):
    return dict(scene=SCENE, cells=[dict(worlds=1, nv=32, step_ms=2.0, stages={STAGE: dict(
        swap_class="exact", applies=True, reason=None, share=share, ceiling=1 / (1 - share), native_ms=1.0, enclosing=None)})])


def run(tmp_path, registry, proposer, speedup_of=lambda gain: 1.0 + gain, budget=Budget(), measure=None, task=None, check=None,
        **kwargs):
    task = task or Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)
    journal = Journal(tmp_path / "journal.jsonl")

    def timed(graph, t, candidate):
        return make_result(graph, speedup=speedup_of(graph.stages[STAGE].params["gain"]))

    report = run_loop(task, proposer, budget=budget, journal=journal, registry=registry,
                      measure=measure or (lambda t, w: gate_report()), ledger=Ledger(tmp_path / "ledger.jsonl"),
                      lease=Lease("cpu", tmp_path / "leases"), check=check or (lambda request, **kw: passing()), timed=timed,
                      candidates_dir=tmp_path / "candidates", **kwargs)
    return report, journal


def test_a_stage_with_no_room_to_gain_ends_the_search_before_anything_is_proposed(tmp_path, registry):
    proposed = []

    class Spy:
        name = "spy"

        def propose(self, brief, history):
            proposed.append(1)

    report, journal = run(tmp_path, registry, Spy(), measure=lambda t, w: gate_report(share=0.01))
    assert report.status == "refused" and "below the 1.05x needed" in report.reason and proposed == [] and report.outcomes == []
    assert [e["status"] for e in journal.records("task") if e["phase"] == "end"] == ["refused"]
    assert report.spend.attempts == 0 and report.spend.timed_runs == 0


def test_the_loop_runs_until_the_proposer_has_nothing_left(tmp_path, registry):
    report, _ = run(tmp_path, registry, ScriptedProposer(gains(1, 2, 3)))
    assert report.status == "finished" and len(report.outcomes) == 3 and report.best.proposal.params == {"gain": 3.0}
    assert report.best.speedup == 4.0 and report.spend.attempts == 3 and report.spend.timed_runs == 3


def test_the_best_is_the_fastest_accepted_not_the_latest(tmp_path, registry):
    report, _ = run(tmp_path, registry, ScriptedProposer(gains(5, 1, 2)))
    assert report.best.proposal.params == {"gain": 5.0} and report.best.speedup == 6.0


def test_a_stalled_search_stops_after_the_budgeted_attempts_without_a_new_best(tmp_path, registry):
    report, _ = run(tmp_path, registry, ScriptedProposer(gains(9, 1, 2, 3, 4, 5, 6)), budget=Budget(stall=3))
    assert report.status == "stalled" and len(report.outcomes) == 4 and report.spend.stalled == 3      # 1 best + 3 that did not beat it


def test_a_new_best_resets_the_stall_count(tmp_path, registry):
    report, _ = run(tmp_path, registry, ScriptedProposer(gains(1, 0, 2, 0, 3, 0)), budget=Budget(stall=2))
    assert report.status == "finished" and len(report.outcomes) == 6


def test_the_attempt_budget_stops_the_loop(tmp_path, registry):
    report, _ = run(tmp_path, registry, ScriptedProposer(gains(*range(1, 20))), budget=Budget(attempts=4, stall=99))
    assert report.status == "budget" and "attempt budget" in report.reason and report.spend.attempts == 4


def test_the_token_budget_counts_what_the_proposer_says_it_spent(tmp_path, registry):
    proposals = [Proposal(STAGE, impl=FAST, params={"gain": float(g)}, tokens=400) for g in range(1, 10)]
    report, _ = run(tmp_path, registry, ScriptedProposer(proposals), budget=Budget(tokens=1000, stall=99))
    assert report.status == "budget" and "token budget" in report.reason and report.spend.tokens == 1200 and len(report.outcomes) == 3


def test_the_device_time_budget_stops_the_loop(tmp_path, registry):
    task = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)

    def slow(graph, t, candidate):
        time.sleep(0.15)
        return make_result(graph)

    journal = Journal(tmp_path / "j.jsonl")
    report = run_loop(task, ScriptedProposer(gains(*range(1, 10))), budget=Budget(cpu_seconds=0.2, stall=99), journal=journal,
                      registry=registry, measure=lambda t, w: gate_report(), ledger=Ledger(tmp_path / "l.jsonl"),
                      lease=Lease("cpu", tmp_path / "leases"), check=lambda request, **kw: passing(), timed=slow)
    assert report.status == "budget" and "cpu-seconds" in report.reason and 1 <= len(report.outcomes) <= 3


def test_a_proposer_that_repeats_itself_is_answered_from_the_memo_and_stalls(tmp_path, registry):
    report, _ = run(tmp_path, registry, ScriptedProposer(gains(2, 2, 2, 2, 2, 2)), budget=Budget(stall=3))
    assert report.status == "stalled" and report.spend.memo_hits == 3 and report.spend.timed_runs == 1


def test_a_proposer_that_fails_costs_an_attempt_and_the_loop_goes_on(tmp_path, registry):
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def propose(self, brief, history):
            self.calls += 1
            if self.calls == 1:
                raise ProposerError("the command exited with code 2")
            return Proposal(STAGE, impl=FAST, params={"gain": 3.0}) if self.calls == 2 else None

    report, _ = run(tmp_path, registry, Flaky())
    assert [o.verdict for o in report.outcomes] == [REJECTED, ACCEPTED] and "exited with code 2" in report.outcomes[0].reason
    assert report.outcomes[0].reached == "none" and report.spend.attempts == 2


def test_rejected_candidates_are_counted_by_the_tier_that_stopped_them(tmp_path, registry):
    def check(request, **kw):
        if request["tier"] == "cheap" and request["params"]["gain"] == 2.0:
            return CheckResult("fail", failures=["residual"])
        if request["tier"] == "real" and request["params"]["gain"] == 3.0:
            return CheckResult("fail", failures=["reads the lower triangle"])
        return passing()

    proposals = gains(1, 2, 3) + [Proposal(STAGE, impl="effective.solve/native")]
    report, _ = run(tmp_path, registry, ScriptedProposer(proposals), check=check)
    assert report.funnel() == dict(proposed=4, static=3, cheap=2, real=1, run=1, accepted=1)
    assert report.spend.refused_before_run == 3 and report.spend.timed_runs == 1


def test_everything_the_loop_did_is_in_the_journal_and_the_chain_is_whole(tmp_path, registry):
    report, journal = run(tmp_path, registry, ScriptedProposer(gains(1, 2)))
    kinds = [e["kind"] for e in journal.records()]
    assert kinds[0] == "ceiling" and kinds.count("attempt") == 2 and kinds[-1] == "task" and journal.verify().ok
    start, end = [e for e in journal.records("task")]
    assert start["phase"] == "start" and start["proposer"] == "scripted" and start["gate"]["ok"]
    assert end["phase"] == "end" and end["status"] == "finished" and end["best"]["speedup"] == 3.0
    assert report.head == str(journal.head)
    attempts = journal.records("attempt")
    assert [a["improved"] for a in attempts] == [True, True] and attempts[0]["task_id"] == report.task.id


def test_a_second_search_on_the_same_scene_reuses_the_ceiling_and_every_result(tmp_path, registry):
    measured = []
    measure = lambda t, w: measured.append(1) or gate_report()
    run(tmp_path, registry, ScriptedProposer(gains(1, 2)), measure=measure)
    again, journal = run(tmp_path, registry, ScriptedProposer(gains(1, 2, 3)), measure=measure)
    assert len(measured) == 1 and again.spend.memo_hits == 2 and again.spend.timed_runs == 1
    assert again.gate.basis == "memo"


def test_the_summary_reads_the_funnel_the_memo_and_the_spend_off_the_journal(tmp_path, registry):
    run(tmp_path, registry, ScriptedProposer(gains(1, 2, 2)))
    summary = summarize(Journal(tmp_path / "journal.jsonl"))
    assert summary["intact"] and summary["attempts"] == 3 and summary["funnel"]["accepted"] == 3
    assert summary["memo"]["hits"] == 1 and summary["memo"]["rate"] == pytest.approx(1 / 3)
    assert summary["best"]["speedup"] == 3.0 and summary["best"]["scene"] == SCENE and summary["tasks"]["finished"] == 1


# ---- fan-out ------------------------------------------------------------------------------------------------

def test_one_worker_is_the_default_and_proposals_are_taken_one_at_a_time(tmp_path, registry):
    asked = []

    class Counting(ScriptedProposer):
        def propose(self, brief, history):
            asked.append(len(history))
            return super().propose(brief, history)

    run(tmp_path, registry, Counting(gains(1, 2, 3)))
    assert asked == [0, 1, 2, 3]                                       # each saw every earlier outcome


def test_more_workers_check_the_cpu_tiers_of_several_candidates_at_once(tmp_path, registry):
    live, peak, lock = [0], [0], threading.Lock()

    def check(request, **kw):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.1)
        with lock:
            live[0] -= 1
        return passing()

    asked = []

    class Counting(ScriptedProposer):
        def propose(self, brief, history):
            asked.append(len(history))
            return super().propose(brief, history)

    report, _ = run(tmp_path, registry, Counting(gains(1, 2, 3, 4)), budget=Budget(workers=4, cpu_seconds=60.0), check=check)
    assert peak[0] > 1 and asked[:4] == [0, 0, 0, 0]                    # four were asked for before any outcome was known
    assert [o.attempt for o in report.outcomes] == [1, 2, 3, 4] and report.best.proposal.params == {"gain": 4.0}
    assert report.spend.timed_runs == 4


def test_timed_runs_are_never_concurrent_even_with_workers(tmp_path, registry):
    live, peak, lock = [0], [0], threading.Lock()
    task = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)

    def timed(graph, t, candidate):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return make_result(graph)

    run_loop(task, ScriptedProposer(gains(1, 2, 3, 4)), budget=Budget(workers=4, cpu_seconds=60.0), journal=Journal(tmp_path / "j.jsonl"),
             registry=registry, measure=lambda t, w: gate_report(), ledger=Ledger(tmp_path / "l.jsonl"),
             lease=Lease("cpu", tmp_path / "leases"), check=lambda request, **kw: passing(), timed=timed)
    assert peak[0] == 1


def test_a_gpu_scene_checks_for_real_one_at_a_time_even_with_workers(tmp_path, registry):
    live, peak, lock = [0], [0], threading.Lock()
    real_calls = []

    def check(request, **kw):
        if request["tier"] == "real":
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
                real_calls.append(1)
            time.sleep(0.05)
            with lock:
                live[0] -= 1
        return passing()

    task = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)
    journal = Journal(tmp_path / "j.jsonl")
    funnel = Funnel(task, journal, registry=registry, ledger=Ledger(tmp_path / "l.jsonl"), lease=Lease("cuda:0", tmp_path / "leases"),
                    check=check, timed=lambda graph, t, c: make_result(graph))
    funnel.world.device_kind = "cuda"
    report = run_loop(task, ScriptedProposer(gains(1, 2, 3)), budget=Budget(workers=3, cpu_seconds=60.0), journal=journal,
                      registry=registry, funnel=funnel, measure=lambda t, w: gate_report())
    assert len(real_calls) == 3 and peak[0] == 1 and len(report.outcomes) == 3
    assert funnel.spend.gpu_seconds > 0


def test_fan_out_is_never_wider_than_the_attempts_left(tmp_path, registry):
    asked = []

    class Counting(ScriptedProposer):
        def propose(self, brief, history):
            asked.append(1)
            return super().propose(brief, history)

    report, _ = run(tmp_path, registry, Counting(gains(*range(1, 20))), budget=Budget(attempts=5, workers=4, cpu_seconds=60.0, stall=99))
    assert report.spend.attempts == 5 and len(report.outcomes) == 5 and len(asked) == 5


# ---- what a screen reads ----------------------------------------------------------------------------------------

def test_the_overview_lists_the_latest_attempts_and_the_finished_tasks_with_who_holds_each_device(tmp_path, registry):
    from phost_agent import overview
    run(tmp_path, registry, ScriptedProposer(gains(1, 2, 2)))
    with Lease("cpu", tmp_path / "leases").hold("a timed run"):
        seen = overview(Journal(tmp_path / "journal.jsonl"), lease_dir=tmp_path / "leases")
    assert seen["leases"]["cuda:0"] is None and seen["leases"]["cpu"]["holder"] == "a timed run"
    assert [a["attempt"] for a in seen["recent"]] == [3, 2, 1] and seen["recent"][0]["memo"] is True
    assert seen["recent"][0]["scene"] == SCENE and seen["recent"][0]["candidate"] == f"{FAST} gain=2.0"
    (task,) = seen["task_list"]
    assert task["status"] == "finished" and task["attempts"] == 3 and task["best_speedup"] == 3.0 and task["proposer"] == "scripted"
    assert len(overview(Journal(tmp_path / "journal.jsonl"), limit=2)["recent"]) == 2


def test_the_overview_of_a_damaged_journal_still_shows_what_parses_and_says_it_is_damaged(tmp_path, registry):
    import json
    from phost_agent import overview
    run(tmp_path, registry, ScriptedProposer(gains(1, 2)))
    path = tmp_path / "journal.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["entry"]["seconds"] = 99
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    seen = overview(Journal(path))
    assert seen["intact"] is False and "line 2" in seen["problems"][0] and seen["recent"]


def test_the_overview_of_an_empty_journal_is_empty_not_an_error(tmp_path):
    from phost_agent import overview
    seen = overview(Journal(tmp_path / "none.jsonl"), lease_dir=tmp_path)
    assert seen["intact"] and seen["recent"] == [] and seen["task_list"] == [] and seen["attempts"] == 0
    assert seen["memo"]["rate"] is None and seen["best"] is None
