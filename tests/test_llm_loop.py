"""A model in the proposer's seat, through the real loop: the funnel judges what it proposes, the history comes back to it, and the
budgets and the fatal case end the search the way they do for any proposer. The checks and timing are the fakes of test_loop.py."""

import pytest

import phost
from phost import Impl, ImplKind, ParamSpec
from phost_agent import Budget, Funnel, Journal, Lease, Task, run_loop
from phost_agent.llm import LLMError, LLMFatal, LLMProposer, Reply
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


class Scripted:
    name, model = "fake", "fake-1"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def complete(self, system, messages, *, max_tokens):
        self.calls.append([dict(m) for m in messages])
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def gain(g, tokens=500):
    return Reply(f'{{"proposal": {{"impl": "{FAST}", "params": {{"gain": {g}}}, "note": "gain {g}"}}}}', tokens, 0)


def search(tmp_path, registry, model, budget=Budget(), **kwargs):
    task = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3)
    report = run_loop(task, LLMProposer(model, **kwargs), budget=budget, journal=Journal(tmp_path / "journal.jsonl"), registry=registry,
                      measure=lambda t, w: dict(scene=SCENE, cells=[dict(worlds=1, nv=32, step_ms=2.0, stages={STAGE: dict(
                          swap_class="exact", applies=True, reason=None, share=0.5, ceiling=2.0, native_ms=1.0, enclosing=None)})]),
                      ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cpu", tmp_path / "leases"),
                      check=lambda request, **kw: passing(),
                      timed=lambda graph, t, candidate: make_result(graph, speedup=1.0 + graph.stages[STAGE].params["gain"]),
                      candidates_dir=tmp_path / "candidates")
    return report


def test_a_model_proposes_sees_how_it_fared_and_stops_when_it_has_nothing_left(tmp_path, registry):
    model = Scripted(gain(1), gain(3), Reply('{"stop": "gains above 3 would exceed what the stage can give"}', 400, 30))
    report = search(tmp_path, registry, model)
    assert report.status == "finished" and report.reason.endswith("gains above 3 would exceed what the stage can give")
    assert [o.speedup for o in report.outcomes] == [2.0, 4.0] and report.best.proposal.params == {"gain": 3.0}
    assert report.outcomes[0].proposal.note == "gain 1"
    third = model.calls[2][0]["content"]
    assert third.startswith("Attempt 3 of 20.") and f"#1 {FAST}" in third and "Best accepted so far: #2 at 4x of native." in third


def test_the_tokens_of_every_call_are_charged_and_a_token_budget_ends_the_search(tmp_path, registry):
    model = Scripted(gain(1, 600), gain(2, 600), gain(3, 600), gain(4, 600))
    report = search(tmp_path, registry, model, budget=Budget(tokens=1000, stall=9))
    assert report.status == "budget" and "token budget used (1000)" in report.reason
    assert len(report.outcomes) == 2 and report.spend.tokens == 1200 and len(model.calls) == 2


def test_a_model_that_repeats_itself_is_answered_from_the_memo_and_still_spends_the_attempt(tmp_path, registry):
    model = Scripted(gain(2), gain(2), gain(2), Reply('{"stop": "x"}', 1, 1))
    report = search(tmp_path, registry, model, budget=Budget(stall=2))
    assert [o.memo for o in report.outcomes] == [False, True, True] and report.spend.memo_hits == 2 and report.spend.timed_runs == 1
    assert report.status == "stalled" and report.spend.attempts == 3


def test_a_failed_call_costs_an_attempt_and_its_tokens_and_the_model_is_asked_again(tmp_path, registry):
    model = Scripted(Reply("not json", 300, 10), Reply("still not json", 300, 10), gain(2), Reply('{"stop": "done"}', 5, 5))
    report = search(tmp_path, registry, model)
    assert [o.verdict for o in report.outcomes] == ["rejected", "accepted"] and report.outcomes[0].proposal.impl == "(proposer failed)"
    assert "could not be used" in report.outcomes[0].reason and report.spend.tokens == 620 + 500


def test_a_failure_that_would_repeat_ends_the_search_without_spending_an_attempt(tmp_path, registry):
    report = search(tmp_path, registry, Scripted(LLMFatal("HTTP 401 from openai: bad key")))
    assert report.status == "stopped" and "cannot continue: HTTP 401 from openai: bad key" in report.reason
    assert report.outcomes == [] and report.spend.attempts == 0


def test_a_failure_after_good_work_ends_the_search_and_keeps_the_best(tmp_path, registry):
    report = search(tmp_path, registry, Scripted(gain(2), LLMFatal("quota exhausted")))
    assert report.status == "stopped" and report.best is not None and report.best.speedup == 3.0 and len(report.outcomes) == 1


def test_the_journal_names_the_model_that_drove_the_search(tmp_path, registry):
    search(tmp_path, registry, Scripted(gain(1), Reply('{"stop": "x"}', 1, 1)))
    started = [e for e in Journal(tmp_path / "journal.jsonl").records("task") if e["phase"] == "start"]
    assert started[0]["proposer"] == "llm:fake:fake-1"
