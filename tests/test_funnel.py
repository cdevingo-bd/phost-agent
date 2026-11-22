"""The funnel's decisions, with the check and the timed run replaced by recorded fakes so each tier's logic is tested alone."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import phost
from phost import Impl, ImplKind, ParamSpec
from phost_agent import (ACCEPTED, MEASURED, REJECTED, CheckResult, Funnel, Lease, Proposal, Task)
from phost_lab.bootstrap import load_registry
from phost_score import Ledger
from conftest import GOOD_SOLVER, SCENE, STAGE, make_result, passing

FAST = f"{STAGE}/fast"


@pytest.fixture(scope="module")
def registry():
    reg = load_registry(registry=phost.Registry())
    reg.add_impl(Impl(id=FAST, stage=STAGE, stage_version=1, version="1", title="fast", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: None, params=(ParamSpec("gain", "float", 1.0, min=0.0, max=10.0),)))
    return reg


class Checks:
    """A stand-in for ``run_check`` that records each call and answers by tier."""

    def __init__(self, **by_tier):
        self.by_tier, self.calls = by_tier, []

    def __call__(self, request, **kwargs):
        self.calls.append(dict(request, **{f"_{k}": v for k, v in kwargs.items()}))
        answer = self.by_tier.get(request["tier"], passing())
        return answer(request) if callable(answer) else answer

    def tiers(self):
        return [c["tier"] for c in self.calls]


class Timed:
    def __init__(self, **kwargs):
        self.kwargs, self.calls = kwargs, []

    def __call__(self, graph, task, candidate):
        self.calls.append((graph, candidate))
        return make_result(graph, **self.kwargs)


def build(tmp_path, journal, registry, task=None, checks=None, timed=None, **options):
    task = task or Task(scene=SCENE, stage=STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)
    checks, timed = checks or Checks(), timed or Timed()
    funnel = Funnel(task, journal, registry=registry, ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cpu", tmp_path / "leases"),
                    check=checks, timed=timed, candidates_dir=tmp_path / "candidates", **options)
    return funnel, checks, timed


def impl(gain=1.0):
    return Proposal(STAGE, impl=FAST, params={"gain": gain})


# ---- tier 1 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("proposal,text", [
    (Proposal("control.law", impl="control.law/pd"), "the task is for stage 'effective.solve'"),
    (Proposal(STAGE, impl="effective.solve/nope"), "effective.solve/nope"),
    (Proposal(STAGE, impl="effective.solve/native"), "host's own version"),
    (Proposal(STAGE, impl=FAST, params={"gain": 99.0}), "maximum"),
    (Proposal(STAGE, impl=FAST, params={"speed": 1.0}), "unknown params"),
    (Proposal(STAGE, impl="dof.passive_force/spring-damper"), "not 'effective.solve'"),
    (Proposal(STAGE, source="def broken(:\n"), "does not parse"),
    (Proposal(STAGE, source="class Other:\n    pass\n"), "defines no class 'Runtime'"),
    (Proposal(STAGE, source="class Runtime:\n    def metrics(self):\n        return {}\n"), "no method 'solve'"),
    (Proposal(STAGE, source="class Runtime:\n    def solve(self, s):\n        pass\n"), "no method 'metrics'"),
])
def test_a_proposal_that_cannot_be_tried_is_refused_before_any_process_starts(tmp_path, journal, registry, proposal, text):
    funnel, checks, timed = build(tmp_path, journal, registry)
    outcome = funnel.evaluate(proposal, 1)
    assert outcome.verdict == REJECTED and outcome.reached == "static" and text in outcome.reason
    assert checks.calls == [] and timed.calls == []


def test_a_stage_that_judges_against_the_implementations_own_reference_gets_a_source_without_one_refused(tmp_path, journal, registry):
    task = Task(SCENE, "dof.passive_force", goal="behaviour")
    funnel, checks, _ = build(tmp_path, journal, registry, task)
    source = "class Runtime:\n    def apply(self, s):\n        pass\n    def metrics(self):\n        return {}\n"
    outcome = funnel.evaluate(Proposal("dof.passive_force", source=source), 1)
    assert "reference(inputs, params)" in outcome.reason and checks.calls == []


def test_a_subclassing_runtime_is_given_the_benefit_of_the_doubt(tmp_path, journal, registry):
    funnel, checks, _ = build(tmp_path, journal, registry)
    outcome = funnel.evaluate(Proposal(STAGE, source="from base import B\nclass Runtime(B):\n    pass\n"), 1)
    assert outcome.reached != "static" and checks.tiers()[0] == "cheap"      # it may inherit solve and metrics; the check will tell


# ---- the whole way ------------------------------------------------------------------------------------------

def test_a_candidate_that_passes_everything_is_timed_judged_recorded_and_accepted(tmp_path, journal, registry):
    funnel, checks, timed = build(tmp_path, journal, registry, timed=Timed(speedup=2.5))
    outcome = funnel.evaluate(impl(2.0), 1)
    assert outcome.verdict == ACCEPTED and outcome.reached == "run" and outcome.speedup == 2.5 and outcome.klass == "exact"
    assert checks.tiers() == ["cheap", "real"] and len(timed.calls) == 1
    assert len(outcome.entries) == 2                                   # one score entry per world count
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.verify().ok and [r.entry["objectives"]["speedup"] for r in ledger] == [2.5, 2.5]
    assert {r.entry["trust"] for r in ledger} == {"self-reported"}     # produced here, judged here: not a judge's run
    tiers = {e["tier"]: e["ok"] for e in journal.records("tier")}
    assert tiers == {"static": True, "cheap": True, "real": True, "run": True}
    assert funnel.spend.timed_runs == 1 and funnel.spend.cpu_seconds >= 0


def test_the_graph_that_is_timed_is_one_stage_on_the_scenes_host(tmp_path, journal, registry):
    funnel, _, timed = build(tmp_path, journal, registry)
    funnel.evaluate(impl(3.0), 1)
    graph, candidate = timed.calls[0]
    assert graph.host == "chain" and graph.stages[STAGE].impl == FAST and dict(graph.stages[STAGE].params) == {"gain": 3.0}
    assert candidate is None


def test_a_candidate_that_is_correct_but_not_faster_is_measured_and_still_recorded(tmp_path, journal, registry):
    funnel, _, _ = build(tmp_path, journal, registry, timed=Timed(speedup=0.8))
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.verdict == MEASURED and "0.800x at its worst world count" in outcome.reason
    assert len(list(Ledger(tmp_path / "ledger.jsonl"))) == 2          # a measurement is a fact whether or not it won


def test_the_speedup_that_counts_is_the_worst_over_the_world_counts(tmp_path, journal, registry):
    def uneven(graph, task, candidate):
        r = make_result(graph, speedup=3.0)
        r["cells"][1].update(speedup=0.9, native_ms=0.9)
        return r

    funnel, _, _ = build(tmp_path, journal, registry, timed=uneven)
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.verdict == MEASURED and outcome.speedup == pytest.approx(0.9)


def test_the_behaviour_goal_accepts_a_candidate_that_passes_whatever_its_speed(tmp_path, journal, registry):
    task = Task(SCENE, STAGE, worlds=(1,), goal="behaviour", steps=5, launches=2, repeats=3)
    funnel, _, _ = build(tmp_path, journal, registry, task, timed=Timed(speedup=0.5, worlds=(1,)))
    assert funnel.evaluate(impl(), 1).verdict == ACCEPTED


def test_the_behaviour_goal_does_not_care_how_noisy_the_timing_was(tmp_path, journal, registry):
    task = Task(SCENE, STAGE, worlds=(1,), goal="behaviour", steps=5, launches=2, repeats=3)
    funnel, _, _ = build(tmp_path, journal, registry, task, timed=Timed(speedup=0.5, worlds=(1,), other=3))
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.verdict == ACCEPTED and funnel.evaluate(impl(), 2).memo


def test_the_driver_is_part_of_what_was_measured(tmp_path, journal, registry):
    funnel, _, timed = build(tmp_path, journal, registry)
    funnel.evaluate(impl(), 1)
    eager = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3, driver="eager")
    other, _, _ = build(tmp_path, journal, registry, eager, timed=timed)
    assert not other.evaluate(impl(), 1).memo


# ---- the memo -----------------------------------------------------------------------------------------------

def test_a_candidate_seen_before_costs_nothing(tmp_path, journal, registry):
    funnel, checks, timed = build(tmp_path, journal, registry, timed=Timed(speedup=2.0))
    first = funnel.evaluate(impl(2.0), 1)
    n_checks, n_runs, n_entries = len(checks.calls), len(timed.calls), len(list(Ledger(tmp_path / "ledger.jsonl")))
    second = funnel.evaluate(impl(2.0), 2)
    assert second.memo and not first.memo and second.verdict == ACCEPTED and second.speedup == 2.0
    assert (len(checks.calls), len(timed.calls), len(list(Ledger(tmp_path / "ledger.jsonl")))) == (n_checks, n_runs, n_entries)
    assert second.entries == first.entries


def test_a_different_parameter_is_a_different_candidate(tmp_path, journal, registry):
    funnel, checks, timed = build(tmp_path, journal, registry)
    funnel.evaluate(impl(2.0), 1)
    assert not funnel.evaluate(impl(3.0), 2).memo and len(timed.calls) == 2


def test_a_default_spelled_out_is_the_same_candidate_as_the_default_left_out(tmp_path, journal, registry):
    funnel, _, timed = build(tmp_path, journal, registry)
    funnel.evaluate(Proposal(STAGE, impl=FAST), 1)
    assert funnel.evaluate(Proposal(STAGE, impl=FAST, params={"gain": 1.0}), 2).memo and len(timed.calls) == 1


def test_the_agents_note_and_token_count_are_not_part_of_what_a_candidate_is(tmp_path, journal, registry):
    funnel, _, timed = build(tmp_path, journal, registry)
    funnel.evaluate(Proposal(STAGE, impl=FAST, note="a hunch", tokens=900), 1)
    assert funnel.evaluate(Proposal(STAGE, impl=FAST, note="the same, said otherwise", tokens=1), 2).memo


def test_a_failure_is_remembered_with_its_reason(tmp_path, journal, registry):
    checks = Checks(cheap=CheckResult("fail", failures=["max_relative_residual = 0.4 exceeds 1e-05"]))
    funnel, _, timed = build(tmp_path, journal, registry, checks=checks)
    first, second = funnel.evaluate(impl(), 1), funnel.evaluate(impl(), 2)
    assert first.reached == second.reached == "cheap" and first.reason == second.reason and "0.4 exceeds" in first.reason
    assert not first.memo and second.memo and len(checks.calls) == 1 and timed.calls == []


def test_a_timeout_says_nothing_about_the_candidate_and_is_not_remembered(tmp_path, journal, registry):
    checks = Checks(cheap=CheckResult("error", reason="timed out after 300 s", transient=True))
    funnel, _, _ = build(tmp_path, journal, registry, checks=checks)
    funnel.evaluate(impl(), 1)
    funnel.evaluate(impl(), 2)
    assert len(checks.calls) == 2


def test_a_candidate_that_crashes_the_worker_the_same_way_every_time_is_remembered(tmp_path, journal, registry):
    checks = Checks(real=CheckResult("error", reason="LookupError: the source defines no 'Runtime'"))
    funnel, _, _ = build(tmp_path, journal, registry, checks=checks)
    funnel.evaluate(impl(), 1)
    again = funnel.evaluate(impl(), 2)
    assert again.memo and again.reached == "real" and checks.tiers() == ["cheap", "real"]


def test_a_tier_that_found_nothing_to_check_on_is_recorded_as_skipped_and_the_candidate_goes_on(tmp_path, journal, registry):
    checks = Checks(cheap=CheckResult("skipped", reason="no simulator-free host offers this stage"))
    funnel, _, timed = build(tmp_path, journal, registry, checks=checks)
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.verdict == ACCEPTED and len(timed.calls) == 1
    assert [e["reason"] for e in journal.records("tier") if e["tier"] == "cheap"] == ["skipped: no simulator-free host offers this stage"]


def test_refresh_ignores_the_memo(tmp_path, journal, registry):
    funnel, checks, timed = build(tmp_path, journal, registry)
    funnel.evaluate(impl(), 1)
    fresh, _, _ = build(tmp_path, journal, registry, checks=checks, timed=timed, refresh=True)
    assert not fresh.evaluate(impl(), 2).memo and len(timed.calls) == 2


def test_the_memo_is_per_scene(tmp_path, journal, registry):
    funnel, _, timed = build(tmp_path, journal, registry)
    funnel.evaluate(impl(), 1)
    other, _, _ = build(tmp_path, journal, registry, Task("chain256:ringing", STAGE, worlds=(1, 2), steps=5), timed=timed)
    assert not other.evaluate(impl(), 1).memo


# ---- tiers 3 to 5 -------------------------------------------------------------------------------------------

def test_a_failing_real_check_stops_a_candidate_that_passed_the_cheap_one_before_it_is_timed(tmp_path, journal, registry):
    checks = Checks(real=CheckResult("fail", failures=["inputs modified"]))
    funnel, _, timed = build(tmp_path, journal, registry, checks=checks)
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.reached == "real" and "inputs modified" in outcome.reason and timed.calls == []
    assert funnel.spend.refused_before_run == 1 and funnel.spend.timed_runs == 0


def test_a_run_that_failed_is_rejected_at_the_run_and_nothing_reaches_the_ledger(tmp_path, journal, registry):
    funnel, _, _ = build(tmp_path, journal, registry, timed=lambda g, t, c: dict(error="IndexError: boom", **{"pass": False}))
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.verdict == REJECTED and outcome.reached == "run" and "IndexError: boom" in outcome.reason
    assert not (tmp_path / "ledger.jsonl").exists()
    assert funnel.evaluate(impl(), 2).memo                              # a deterministic failure is remembered


@pytest.mark.parametrize("error", ["timed out after 900 s", "cancelled", "worker exited with code -9 and no result"])
def test_a_run_lost_to_the_machine_is_not_remembered_against_the_candidate(tmp_path, journal, registry, error):
    runs = []
    funnel, _, _ = build(tmp_path, journal, registry, timed=lambda g, t, c: runs.append(1) or dict(error=error, **{"pass": False}))
    funnel.evaluate(impl(), 1)
    assert not funnel.evaluate(impl(), 2).memo and len(runs) == 2


def test_a_run_the_scorer_calls_ineligible_is_rejected_with_the_scorers_reasons(tmp_path, journal, registry):
    def bad(graph, task, candidate):
        r = make_result(graph)
        for cell in r["cells"]:
            cell["verdict"] = dict(ok=False, problems=["accuracy ratio 40.00 exceeds 10.0"], notes=[],
                                   gates=dict(finite=True, accuracy=False, stage=True))
        return r

    funnel, _, _ = build(tmp_path, journal, registry, timed=bad)
    outcome = funnel.evaluate(impl(), 1)
    assert outcome.verdict == REJECTED and "accuracy ratio 40.00" in outcome.reason and outcome.reached == "run"


def test_an_accuracy_ratio_against_a_noisy_control_is_not_held_against_the_candidate_for_good(tmp_path, journal, registry):
    runs = []

    def over_the_gate(graph, task, candidate, repeatable):
        runs.append(1)
        r = make_result(graph)
        r["meta"]["repeatable"] = repeatable
        for cell in r["cells"]:
            cell["verdict"] = dict(ok=False, problems=["accuracy ratio 12.02 exceeds 10.0"], notes=[],
                                   gates=dict(finite=True, accuracy=False, stage=True))
        return r

    noisy, _, _ = build(tmp_path, journal, registry, timed=lambda g, t, c: over_the_gate(g, t, c, False))
    first = noisy.evaluate(impl(1.0), 1)
    assert first.verdict == REJECTED and "accuracy ratio 12.02" in first.reason
    assert not noisy.evaluate(impl(1.0), 2).memo and len(runs) == 2                # a GPU's ratio moves between runs: ask again
    exact, _, _ = build(tmp_path, journal, registry, timed=lambda g, t, c: over_the_gate(g, t, c, True))
    exact.evaluate(impl(2.0), 3)
    assert exact.evaluate(impl(2.0), 4).memo and len(runs) == 3                    # an engine that repeats exactly has no such excuse


def test_a_noisy_measurement_is_not_a_win_and_is_run_again_next_time(tmp_path, journal, registry):
    runs = []

    def noisy(graph, task, candidate):
        runs.append(1)
        return make_result(graph, speedup=5.0, other=2)

    funnel, _, _ = build(tmp_path, journal, registry, timed=noisy)
    first = funnel.evaluate(impl(), 1)
    assert first.verdict == MEASURED and "unreliable" in first.reason and first.speedup == 5.0
    assert not funnel.evaluate(impl(), 2).memo and len(runs) == 2       # the timing says nothing, so it is asked again


def test_every_run_holds_the_lease_and_a_cpu_scene_checks_without_it(tmp_path, journal, registry):
    holders = []
    leases = tmp_path / "leases"

    def timed(graph, task, candidate):
        holders.append(("run", Lease("cpu", leases).holder()))
        return make_result(graph)

    def check(request, **kwargs):
        holders.append((request["tier"], Lease("cpu", leases).holder()))
        return passing()

    funnel, _, _ = build(tmp_path, journal, registry, checks=check, timed=timed)
    funnel.evaluate(impl(), 1)
    by_tier = dict(holders)
    assert by_tier["run"]["holder"].startswith("run ") and by_tier["cheap"] is None and by_tier["real"] is None


def test_a_gpu_scene_checks_for_real_under_the_lease_and_counts_that_time_as_gpu_time(tmp_path, journal, registry, monkeypatch):
    funnel, _, _ = build(tmp_path, journal, registry)
    funnel.world.device_kind = "cuda"
    holders = []

    def check(request, **kwargs):
        holders.append((request["tier"], Lease("cuda:0", tmp_path / "other").holder()))
        return passing()

    funnel.check = check
    funnel.lease = Lease("cuda:0", tmp_path / "other")
    funnel.evaluate(impl(), 1)
    assert [h for t, h in holders if t == "cheap"] == [None] and [h["holder"] for t, h in holders if t == "real"][0].startswith("check ")
    assert funnel.spend.gpu_seconds >= 0 and funnel.spend.timed_runs == 1


# ---- source ------------------------------------------------------------------------------------------------

def test_a_source_candidate_is_written_out_checked_as_untrusted_and_timed_with_its_directory_importable(tmp_path, journal, registry):
    funnel, checks, timed = build(tmp_path, journal, registry, allow_unsandboxed=True)
    outcome = funnel.evaluate(Proposal(STAGE, source=GOOD_SOLVER), 1)
    assert outcome.verdict == ACCEPTED
    for call in checks.calls:
        assert call["_untrusted"] is True and call["_allow_unsandboxed"] is True
        assert call["impl"].startswith(f"{STAGE}/agent-") and call["plugins"][-1].startswith("phost_agent_cand_")
        assert call["_candidate_dir"] == tmp_path / "candidates"
    graph, candidate = timed.calls[0]
    assert graph.plugins == (candidate.module,) and graph.stages[STAGE].impl == candidate.impl_id
    assert (candidate.directory / f"{candidate.module}.py").exists()
    (entry,) = {r.entry["artifacts"]["source"] for r in Ledger(tmp_path / "ledger.jsonl")}
    assert entry == Proposal(STAGE, source=GOOD_SOLVER).source_hash   # the ledger records which text was measured


def test_a_registered_implementation_is_never_checked_as_untrusted(tmp_path, journal, registry):
    funnel, checks, _ = build(tmp_path, journal, registry)
    funnel.evaluate(impl(), 1)
    assert all(c["_untrusted"] is False and c["_candidate_dir"] is None for c in checks.calls)


# ---- threads -----------------------------------------------------------------------------------------------

def test_several_proposals_can_be_prepared_at_once_and_finish_in_order(tmp_path, journal, registry):
    funnel, checks, timed = build(tmp_path, journal, registry)
    proposals = [impl(g) for g in (1.0, 2.0, 3.0, 4.0)] + [Proposal(STAGE, impl="effective.solve/nope")]
    with ThreadPoolExecutor(4) as pool:
        prepared = list(pool.map(lambda p: funnel.prepare(p, include_real=True), proposals))
    outcomes = [funnel.finish(p, i + 1) for i, p in enumerate(prepared)]
    assert [o.verdict for o in outcomes] == [ACCEPTED] * 4 + [REJECTED]
    assert [o.attempt for o in outcomes] == [1, 2, 3, 4, 5] and len(timed.calls) == 4 and len(checks.calls) == 8
    assert journal.verify().ok                                          # concurrent appends kept the chain whole
