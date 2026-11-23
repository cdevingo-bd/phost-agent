"""The arena over fakes: the rules it keeps, the board it shows, and what it leaves to the contestants. One real run is in test_cli_arena.py."""

import json

import pytest

import phost
from phost import Impl, ImplKind, ParamSpec
from phost_agent import CheckResult, Journal, Lease
from phost_agent.arena import Arena, ArenaTask, Contestant
from phost_agent.arena_text import parse_action
from phost_agent.llm import LLMError, LLMFatal, Reply
from phost_lab.bootstrap import load_registry
from phost_lab.graph import GraphSpec, Node
from phost_score import SELF_REPORTED, Ledger, make_entries
from conftest import GOOD_SOLVER, SCENE, STAGE, make_result, passing

FAST = f"{STAGE}/fast"
OTHER = "dof.passive_force"


@pytest.fixture(scope="module")
def registry():
    reg = load_registry(registry=phost.Registry())
    reg.add_impl(Impl(id=FAST, stage=STAGE, stage_version=1, version="1", title="fast", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: None, params=(ParamSpec("gain", "float", 1.0, min=0.0, max=100.0),)))
    return reg


class Model:
    name, model = "fake", "m"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def complete(self, system, messages, *, max_tokens):
        self.calls.append(dict(system=system, messages=[dict(m) for m in messages]))
        answer = self.answers.pop(0) if self.answers else Reply(json.dumps(dict(action="look", what="card", stage=STAGE)), 1, 1)
        if isinstance(answer, Exception):
            raise answer
        return answer


def act(**action):
    blocks = action.pop("_blocks", "")
    return Reply(json.dumps(action) + blocks, 100, 20)


def submit(gain, note="n"):
    return act(action="submit", graph={"stages": {STAGE: {"impl": FAST, "params": {"gain": gain}}}}, note=note)


REPORT = dict(scene=SCENE, cells=[dict(worlds=2, nv=32, step_ms=2.0, stages={
    STAGE: dict(swap_class="exact", applies=True, reason=None, share=0.5, ceiling=2.0, native_ms=1.0, enclosing=None),
    OTHER: dict(swap_class="model_changing", applies=False, reason="no passive forces here", share=0.0, ceiling=None,
                native_ms=0.0, enclosing=None)})])


def arena(tmp_path, registry, *models, check=None, timed=None, **kwargs):
    contestants = [Contestant(name, model) for name, model in models]
    calls = dict(timed=[], check=[])

    def default_check(request, **kw):
        calls["check"].append(request)
        return passing()

    def default_timed(graph, task, candidate):
        calls["timed"].append((graph, candidate))
        first = next(iter(graph.stages.values()))
        return make_result(graph, speedup=1.0 + first.params.get("gain", 0.0))

    kwargs.setdefault("turns", 10)
    kwargs.setdefault("submits", 3)
    kwargs.setdefault("runtime_facts", False)           # no probe runs unless a test is about it
    a = Arena(ArenaTask(SCENE, worlds=(1, 2), steps=5, launches=2, repeats=3), contestants, journal=Journal(tmp_path / "journal.jsonl"),
              ledger=Ledger(tmp_path / "ledger.jsonl"), lease=Lease("cpu", tmp_path / "leases"), registry=registry,
              measure=lambda t, w: REPORT, check=check or default_check, timed=timed or default_timed,
              candidates_dir=tmp_path / "candidates", **kwargs)
    a.calls, a.by_name = calls, {c.name: c for c in contestants}
    return a


# ---- what they are told ----------------------------------------------------------------------------------------------------------

def test_the_prompt_gives_facts_and_rules_and_no_advice_on_what_to_try(tmp_path, registry):
    a = arena(tmp_path, registry, ("a", Model()))
    a.run()
    system = a.system
    assert "Where the host's own step spends its time" in system and "50.0% of the step" in system and "at most 2.00x" in system
    assert f"- {STAGE} [exact]" in system and FAST in system and "does not apply here (no passive forces here)" in system
    assert "Nobody here tells you how" in system and "10 turns and 3 timed runs" in system and "visible to the other contestants" in system
    for coaching in ("How to search well", "one thing at a time", "Stop when", "Spend attempts"):
        assert coaching not in system
    assert all(word in system for word in ('"action": "look"', '"action": "check"', '"action": "submit"'))
    assert '"action": "done"' not in system and "there is no other way to leave" in system


def test_a_closed_contest_says_so(tmp_path, registry):
    a = arena(tmp_path, registry, ("a", Model()), open_code=False)
    a.run()
    assert "but not what they submitted" in a.system and "visible to the other contestants" not in a.system


# ---- the contest ------------------------------------------------------------------------------------------------------------------

def test_contestants_alternate_each_sees_the_others_entries_and_the_board_ranks_them(tmp_path, registry):
    a_model = Model(submit(1, "first try"))
    b_model = Model(submit(3, "better"))
    a = arena(tmp_path, registry, ("alpha", a_model), ("beta", b_model))
    report = a.run()
    assert "alpha: speedup 2x" in b_model.calls[0]["messages"][-1]["content"]         # beta moved second and saw alpha's entry
    alpha_second = a_model.calls[1]["messages"][-1]["content"]
    assert "You are #2 of 2 in exact at 2 worlds" not in alpha_second and "Timed and ranked" in alpha_second
    assert "alpha (you): speedup 2x" in alpha_second and "beta: speedup 4x" in alpha_second
    b_result = b_model.calls[1]["messages"][-1]["content"]
    assert "You are #1 of 2 in exact at 2 worlds" in b_result
    rows = report.board[0]["rows"]
    assert [r["label"] for r in rows] == ["beta", "alpha"]
    assert {c["name"]: c["best"]["speedup"] for c in report.contestants} == {"alpha": 2.0, "beta": 4.0}
    assert {r.entry["label"] for r in Ledger(tmp_path / "ledger.jsonl").records()} == {"alpha", "beta"}


def test_a_rival_can_read_an_entry_it_wants_to_build_on_unless_the_contest_is_closed(tmp_path, registry):
    a = arena(tmp_path, registry, ("alpha", Model(submit(1, "plain gain"))),
              ("beta", Model(act(action="look", what="card", stage=STAGE))))
    a.run()
    entry_id = next(r["id"] for g in a.groups() for r in g["rows"])
    looker = Model(act(action="look", what="entry", id=entry_id))
    a2 = arena(tmp_path, registry, ("gamma", looker))
    a2.run()
    seen = looker.calls[1]["messages"][-1]["content"]
    assert f"Entry {entry_id} by alpha" in seen and "plain gain" in seen and f"{STAGE}: {FAST}" in seen
    closed = Model(act(action="look", what="entry", id=entry_id))
    arena(tmp_path, registry, ("delta", closed), open_code=False).run()
    assert "this contest is closed" in closed.calls[1]["messages"][-1]["content"]
    card = a.by_name["beta"]
    assert "is now pinned at the top of this conversation" in card.pending or STAGE in card.pinned
    pinned = card.pinned[STAGE]
    assert "class Runtime:" in pinned and "def __init__(self, host, params)" in pinned     # the interface is a fact
    assert "def solve(self, system)" in pinned and "#   rhs" in pinned


def test_an_entry_on_the_board_from_outside_the_contest_can_be_looked_at_too(tmp_path, registry):
    """The board is the ledger, so a person's entry or an earlier run's is ranked beside the contestants. Showing its id and
    then refusing to open it ("not submitted in this contest") was a hole: it opens from what the ledger kept."""
    outside = GraphSpec(name="theirs", host="chain", note="a person's entry",
                        stages={STAGE: Node(FAST, {"gain": 7.0})})
    [entry] = make_entries(make_result(outside, speedup=3.5, worlds=(2,)), label="someone-else", trust=SELF_REPORTED)
    Ledger(tmp_path / "ledger.jsonl").append(entry.to_dict())
    looker = Model(act(action="look", what="entry", id=entry.id), act(action="look", what="entry", id="0123456789abcdef"))
    a = arena(tmp_path, registry, ("gamma", looker))
    a.run()
    assert entry.id in a.board_text("gamma") and "someone-else" in a.board_text("gamma")
    seen = looker.calls[1]["messages"][-1]["content"]
    assert f"Entry {entry.id} by someone-else, recorded outside this contest (self-reported)" in seen
    assert "3.5x" in seen and "a person's entry" in seen and f"{STAGE}: {FAST} {{'gain': 7.0}}" in seen
    assert "no entry '0123456789abcdef' is on this board" in looker.calls[2]["messages"][-1]["content"]


def test_an_entry_may_replace_several_stages_and_bring_its_own_source(tmp_path, registry):
    graph = {"stages": {STAGE: {"entry": "Runtime"}, OTHER: {"impl": "dof.passive_force/spring-damper"}}}
    model = Model(act(action="submit", graph=graph, _blocks=f"\n```python {STAGE}\n{GOOD_SOLVER}\n```"))
    a = arena(tmp_path, registry, ("alpha", model), allow_unsandboxed=True)
    a.run()
    (spec, candidate), = a.calls["timed"]
    assert set(spec.stages) == {STAGE, OTHER} and spec.stages[STAGE].impl.startswith(f"{STAGE}/agent-")
    assert spec.plugins == (candidate.module,)
    assert {r["stage"] for r in a.calls["check"]} == {STAGE, OTHER}
    stored = Journal(tmp_path / "journal.jsonl").records("submission")[0]
    assert "class Runtime" in stored["graph"]["stages"][STAGE]["source"] and stored["contestant"] == "alpha"


# ---- the feed: every check anyone ran ---------------------------------------------------------------------------------------------

FAILING = {"stages": {STAGE: {"impl": FAST}}}
PASSING = {"stages": {OTHER: {"impl": "dof.passive_force/spring-damper"}}}


def failing_check(request, **kw):
    return CheckResult("fail", host="chain", failures=["residual too big: 3e-2"]) if request["impl"] == FAST else passing()


def test_every_check_is_posted_for_the_others_who_ran_it_whether_it_passed_and_why_not(tmp_path, registry):
    alpha = Model(act(action="check", graph=FAILING), act(action="check", graph=FAILING))
    beta = Model(act(action="check", graph=PASSING))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), check=failing_check, turns=3)
    a.run()
    seen_by_beta = beta.calls[0]["messages"][-1]["content"]                        # beta moves second: alpha's check is already up
    assert "New checks by the others" in seen_by_beta and "alpha turn 1" in seen_by_beta
    assert f"{STAGE} ({FAST})" in seen_by_beta and "FAILED at the" in seen_by_beta and "residual too big" in seen_by_beta
    seen_by_alpha = alpha.calls[1]["messages"][-1]["content"]
    feed = seen_by_alpha.split("New checks by the others")[1]
    assert "beta turn 1" in feed and "passed the checks" in feed and "alpha turn" not in feed       # not your own: you have those
    again = alpha.calls[2]["messages"][-1]["content"]
    assert "This exact graph was already checked by you on turn 1 (attempt " in again and "FAILED at the" in again
    stored = a.attempts()
    assert [(r["contestant"], r["ok"]) for r in stored] == [("alpha", False), ("beta", True), ("alpha", False)]
    assert all(r["scene"] == SCENE and r["digest"] for r in stored)


def test_an_attempt_can_be_looked_at_for_its_source_and_the_whole_reason_it_failed(tmp_path, registry):
    long_reason = "x" * 400 + " the end of a long reason"
    model = Model(act(action="submit", graph={"stages": {STAGE: {"entry": "Runtime"}}}, _blocks=f"\n```python {STAGE}\n{GOOD_SOLVER}\n```"))

    def check(request, **kw):
        return CheckResult("fail", host="chain", failures=[long_reason])

    a = arena(tmp_path, registry, ("alpha", model), check=check, allow_unsandboxed=True, turns=1)
    a.run()
    [attempt] = a.attempts()
    assert attempt["action"] == "submit" and not attempt["ok"]                              # a refused submit is posted too
    looker = Model(act(action="look", what="attempt", id=attempt["id"]), act(action="look", what="attempt", id="nope"))
    arena(tmp_path, registry, ("gamma", looker), turns=3).run()
    text = looker.calls[1]["messages"][-1]["content"]
    assert f"Attempt {attempt['id']} by alpha on turn 1 (submit)" in text and "class Runtime" in text and "the end of a long reason" in text
    assert "no attempt 'nope' is recorded for this scene" in looker.calls[2]["messages"][-1]["content"]
    shown = looker.calls[0]["messages"][-1]["content"]
    assert "x" * 400 not in shown and "..." in shown                                       # the feed line is short; the attempt is whole


def test_a_closed_contest_posts_nothing_and_refuses_to_show_an_attempt(tmp_path, registry):
    alpha, beta = Model(act(action="check", graph=FAILING)), Model(act(action="look", what="attempt", id="whatever"))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), check=failing_check, open_code=False, turns=2)
    a.run()
    assert "Recent checks" not in beta.calls[0]["messages"][-1]["content"] and "every check anyone runs" not in a.system
    assert "this contest is closed" in beta.calls[1]["messages"][-1]["content"]
    assert len(a.attempts()) == 1                                                          # still on the record, for you
    off = Model(act(action="check", graph=FAILING))
    b = arena(tmp_path, registry, ("alpha", off), ("beta", Model()), check=failing_check, feed_rows=0, turns=1)
    b.run()
    assert "every check anyone runs" not in b.system


def test_the_dashboard_summary_of_a_contest_names_who_is_in_it_what_they_checked_and_what_they_entered(tmp_path, registry):
    from phost_agent import arena_overview
    assert arena_overview(Journal(tmp_path / "empty.jsonl")) is None
    alpha = Model(act(action="check", graph=FAILING), act(action="submit", graph=PASSING, note="a plain gain"))
    beta = Model(act(action="check", graph=PASSING))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), check=failing_check, turns=3)
    a.run()
    ov = arena_overview(a.journal)
    assert ov["status"] == "finished" and ov["scene"] == SCENE and ov["earlier_contests"] == 0 and ov["budget"]["turns"] == 3
    by = {c["name"]: c for c in ov["contestants"]}
    assert [c["name"] for c in ov["contestants"]] == ["alpha", "beta"] and by["alpha"]["turns"] == 3 and by["alpha"]["left"] == "turns"
    assert (by["alpha"]["checks"], by["alpha"]["passed"], by["alpha"]["entries"], by["alpha"]["best_speedup"]) == (1, 0, 1, 1.0)
    assert (by["beta"]["checks"], by["beta"]["passed"], by["beta"]["entries"], by["beta"]["best_speedup"]) == (1, 1, 0, None)
    assert [c["contestant"] for c in ov["checks"]] == ["beta", "alpha"]                         # newest first
    assert ov["checks"][0]["ok"] and "residual too big" in ov["checks"][1]["outcome"] and f"{STAGE} ({FAST})" in ov["checks"][1]["what"]
    [entry] = ov["entries"]
    assert entry["contestant"] == "alpha" and entry["speedup"] == 1.0 and entry["klass"] == "exact" and entry["note"] == "a plain gain"
    assert "dof.passive_force" in entry["what"]
    arena(tmp_path, registry, ("gamma", Model()), turns=1).run()                                # a later contest replaces it
    later = arena_overview(a.journal)
    assert [c["name"] for c in later["contestants"]] == ["gamma"] and later["earlier_contests"] == 1 and later["checks"] == []


def test_a_contest_with_no_end_record_is_shown_as_running_with_its_last_activity(tmp_path):
    from phost_agent import arena_overview
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("arena", phase="start", task=dict(scene="s", worlds=[1]), contestants=[dict(name="x", model="m:1")], budget=dict(turns=9))
    journal.append("arena", phase="turn", contestant="x", turn=1, action="look", tokens=40)
    ov = arena_overview(journal)
    assert ov["status"] == "running" and ov["ended"] is None and ov["last_activity"] and ov["contestants"][0]["last_action"] == "look"


# ---- the rules the arena keeps ---------------------------------------------------------------------------------------------------

def test_an_entry_that_fails_its_checks_is_not_timed_and_costs_no_timed_run(tmp_path, registry):
    def check(request, **kw):
        return CheckResult("fail", host="chain", failures=["residual too big"]) if request["impl"] == FAST else passing()

    model = Model(submit(1), act(action="check", graph={"stages": {STAGE: {"impl": FAST}}}))
    a = arena(tmp_path, registry, ("alpha", model), check=check)
    a.run()
    assert a.calls["timed"] == [] and a.by_name["alpha"].submits == 0
    assert "Not timed" in model.calls[1]["messages"][-1]["content"] and "residual too big" in model.calls[1]["messages"][-1]["content"]
    assert "The checks failed; nothing was timed." in model.calls[2]["messages"][-1]["content"]


def test_timed_runs_are_the_scarce_thing_and_the_last_one_ends_the_turn_for_that_contestant(tmp_path, registry):
    model = Model(submit(1), submit(2))
    a = arena(tmp_path, registry, ("alpha", model), submits=2)
    report = a.run()
    assert a.by_name["alpha"].left == "submits" and len(a.calls["timed"]) == 2 and len(model.calls) == 2
    assert report.contestants[0]["left"] == "submits"


def test_turns_and_tokens_are_budgeted_per_contestant(tmp_path, registry):
    spin = Model(*[act(action="look", what="card", stage=STAGE)] * 10)
    a = arena(tmp_path, registry, ("spin", spin), turns=3)
    a.run()
    assert a.by_name["spin"].left == "turns" and len(spin.calls) == 3
    heavy = Model(*[Reply(json.dumps(dict(action="look", what="card", stage=STAGE)), 900, 100)] * 5)
    b = arena(tmp_path, registry, ("heavy", heavy), tokens=1500)
    b.run()
    assert b.by_name["heavy"].left == "tokens" and len(heavy.calls) == 2


def test_the_arena_does_not_choose_what_is_tried_a_contestant_may_do_nothing_useful_at_all(tmp_path, registry):
    a = arena(tmp_path, registry, ("alpha", Model()))
    report = a.run()
    assert a.calls["timed"] == [] and all(not g["rows"] for g in report.board)


# ---- when a contestant misbehaves -------------------------------------------------------------------------------------------------

def test_an_unreadable_reply_and_an_unknown_action_and_a_bad_graph_each_get_an_explanation_and_the_turn_is_spent(tmp_path, registry):
    model = Model(Reply("I think I will try something clever.", 5, 5), act(action="fly"), act(action="check", graph={"stages": {}}),
                  act(action="check", graph={"stages": {"no.such": {"impl": "x"}}}))
    a = arena(tmp_path, registry, ("alpha", model))
    a.run()
    replies = [c["messages"][-1]["content"] for c in model.calls[1:]]
    assert "could not be read" in replies[0] and "Unknown action 'fly'" in replies[1]
    assert "at least one stage" in replies[2] and "no stage 'no.such'" in replies[3]


def test_a_failed_call_costs_a_turn_and_a_fatal_one_removes_only_that_contestant(tmp_path, registry):
    flaky = Model(LLMError("rate limited"), submit(1))
    broken = Model(LLMFatal("HTTP 401: bad key"))
    a = arena(tmp_path, registry, ("flaky", flaky), ("broken", broken))
    a.run()
    assert "did not go through (rate limited)" in flaky.calls[1]["messages"][-1]["content"]
    assert a.by_name["broken"].left == "failed: HTTP 401: bad key" and a.by_name["flaky"].submits == 1


def test_a_reply_cut_off_or_declined_is_reported_back(tmp_path, registry):
    model = Model(Reply("{", 10, 99, stop="length"), Reply("", 10, 1, stop="refusal"))
    a = arena(tmp_path, registry, ("alpha", model), max_tokens=99)
    a.run()
    assert "cut off at 99 tokens" in model.calls[1]["messages"][-1]["content"] and "declined" in model.calls[2]["messages"][-1]["content"]


def test_nothing_about_the_machine_is_sent_and_the_journal_records_the_contest(tmp_path, registry, monkeypatch):
    monkeypatch.setenv("HOME", "/home/zebedee_q")
    monkeypatch.setenv("LOGNAME", "zebedee")
    model = Model(act(action="submit", graph={"stages": {STAGE: {"impl": FAST}}}, note="see /home/zebedee_q/notes by zebedee"))
    a = arena(tmp_path, registry, ("alpha", model))
    a.run()
    sent = a.system + "".join(m["content"] for call in model.calls for m in call["messages"])
    assert "zebedee" not in sent and "/home/" not in sent
    kinds = [r["kind"] for r in Journal(tmp_path / "journal.jsonl").records()]
    assert kinds.count("arena") >= 4 and "submission" in kinds


def test_parse_action_reads_tagged_source_blocks_and_ignores_braces_inside_them():
    action, blocks = parse_action('ok {"action": "check", "graph": {"stages": {"s": {"entry": "Runtime"}}}}\n'
                                  '```python s\nclass Runtime:\n    t = {"a": {"b": 1}}\n```\n```python\nx = 1\n```')
    assert action["action"] == "check" and blocks[0][0] == "s" and "t = " in blocks[0][1] and blocks[1][0] == ""
    with pytest.raises(ValueError, match="action"):
        parse_action('{"proposal": {}}')


# ---- what a runtime prints while a check runs --------------------------------------------------------------------------------------

def printing_check(request, **kw):
    return CheckResult("pass", host="chain", metrics={"max_relative_residual": 1e-9},
                       output=f"{request['tier']}: W=2 h=(2, 4, 4) float32\nsolve called  (x3)")


def test_the_prompt_says_what_a_runtime_may_print_and_that_the_others_see_it_too(tmp_path, registry):
    a = arena(tmp_path, registry, ("a", Model()))
    a.run()
    assert "Whatever your runtime `print`s while they run comes back with the result" in a.system and "4000 characters" in a.system
    assert "with what it printed" in a.system


def test_a_check_returns_what_the_runtime_printed_whether_it_passed_or_not_and_the_others_can_read_it(tmp_path, registry):
    alpha = Model(act(action="check", graph=PASSING), act(action="check", graph=PASSING))
    beta = Model(act(action="look", what="attempt", id="__first__"))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), check=printing_check, turns=3)
    a.run()
    own = alpha.calls[1]["messages"][-1]["content"]                               # alpha's own answer to its first check
    assert "The contract checks pass for every stage" in own
    assert f"printed by {OTHER} at the cheap tier:\n    cheap: W=2 h=(2, 4, 4) float32\n    solve called  (x3)" in own
    assert f"printed by {OTHER} at the real tier:\n    real: W=2" in own
    seen = beta.calls[0]["messages"][-1]["content"]                                # beta sees a stub in the feed
    assert "passed the checks; its runtime printed: cheap: W=2 h=(2, 4, 4) float32 solve called (x3)" in seen
    [first, _] = a.attempts()
    assert [o["tier"] for o in first["output"]] == ["cheap", "real"] and first["output"][0]["stage"] == OTHER
    looker = Model(act(action="look", what="attempt", id=first["id"]))
    arena(tmp_path, registry, ("gamma", looker), turns=2).run()
    assert "- printed by %s at the cheap tier:\n    cheap: W=2 h=(2, 4, 4) float32" % OTHER in looker.calls[1]["messages"][-1]["content"]


def test_a_repeated_check_shows_what_was_printed_the_first_time_without_running_it_again(tmp_path, registry):
    alpha = Model(act(action="check", graph=PASSING), act(action="check", graph=PASSING))
    checks = []
    a = arena(tmp_path, registry, ("alpha", alpha), check=lambda request, **kw: checks.append(request["tier"]) or printing_check(request),
              turns=3)
    a.run()
    assert len(a.attempts()) == 2 and checks == ["cheap", "real"]                  # the second check ran nothing
    again = alpha.calls[2]["messages"][-1]["content"]
    assert "This exact graph was already checked by you on turn 1" in again and "cheap: W=2 h=(2, 4, 4) float32" in again


def test_a_failing_check_that_printed_keeps_the_reason_and_the_output(tmp_path, registry):
    def check(request, **kw):
        return CheckResult("fail", host="chain", failures=["residual too big: 3e-2"], output="W=2")

    alpha = Model(act(action="check", graph=FAILING))
    a = arena(tmp_path, registry, ("alpha", alpha), check=check, turns=1)
    a.run()
    [attempt] = a.attempts()
    assert attempt["output"] == [dict(stage=STAGE, tier="cheap", text="W=2")] and not attempt["ok"]
    assert "residual too big" in attempt["reasons"][0]["reason"]


# ---- state is shown once, history is a window, the rest is a notebook -------------------------------------------------------------

import re


class Dyn(Model):
    """A model whose answers may be functions of what it was sent, for a move that names an id it was told."""

    def complete(self, system, messages, *, max_tokens):
        self.calls.append(dict(system=system, messages=[dict(m) for m in messages]))
        answer = self.answers.pop(0) if self.answers else act(action="look", what="board")
        return answer(messages) if callable(answer) else answer


def attempt_id(messages):
    return re.findall(r"This is attempt ([0-9a-f]{8})", messages[-1]["content"])[-1]


def test_the_board_and_the_feed_are_shown_once_and_never_copied_into_the_history(tmp_path, registry):
    alpha = Model(act(action="check", graph=FAILING), act(action="check", graph=PASSING), act(action="look", what="board"))
    beta = Model(act(action="check", graph=PASSING))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), check=failing_check, turns=3)
    a.run()
    first = alpha.calls[0]["messages"][-1]["content"]
    assert "Board for" in first and "no entries yet" in first
    second = alpha.calls[1]["messages"][-1]["content"]
    assert "New checks by the others" in second and "beta turn 1" in second                  # beta's check, new to alpha
    third = alpha.calls[2]["messages"]
    assert "beta turn 1" not in third[-1]["content"] and "(the board has not changed)" in third[-1]["content"]
    assert not any("New checks by the others" in m["content"] or "Board for" in m["content"] for m in third[:-1])


def test_a_new_entry_arrives_as_a_delta_and_the_whole_board_is_there_for_the_asking(tmp_path, registry):
    alpha = Model(act(action="look", what="board"), act(action="look", what="board"), act(action="look", what="board"))
    beta = Model(submit(1, "plain gain"))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), turns=3)
    a.run()
    assert "Board for" in alpha.calls[0]["messages"][-1]["content"]
    second = alpha.calls[1]["messages"][-1]["content"]
    assert "New on the board since your last move" in second and "beta: speedup" in second and "now 1 of 1" in second
    assert "Board for" in second                                                           # the whole board, because alpha asked
    assert "(the board has not changed)" in alpha.calls[2]["messages"][-1]["content"]


def test_a_long_contest_sends_a_window_and_a_notebook_of_what_it_left_out(tmp_path, registry):
    answers = [act(action="check", graph=FAILING, note="the plain gain first"), act(action="check", graph=PASSING, note="then the law")]
    answers += [act(action="look", what="board")] * 5
    model = Model(*answers)
    a = arena(tmp_path, registry, ("alpha", model), check=failing_check, turns=7, keep_turns=1)
    a.run()
    early = model.calls[3]["messages"]                                                    # turn 4: all of it is still in the window
    assert len(early) == 7 and "Notebook" not in early[-1]["content"]
    late = model.calls[5]["messages"]                                                     # turn 6: more than keep + 3 went by, so one is kept
    assert len(late) == 3 and len(model.calls[6]["messages"]) == 5                        # and the window grows again until the next trim
    newest = late[-1]["content"]
    first, second = a.attempts()[:2]
    assert "Notebook (your own earlier work" in newest
    assert f"- {first['id']} turn 1 check" in newest and "note: the plain gain first" in newest and "FAILED at the cheap tier" in newest
    assert f"- {second['id']} turn 2 check" in newest and "note: then the law" in newest and "passed the checks" in newest
    assert "Notebook" not in late[0]["content"] and not any(first["id"] in m["content"] for m in late[:-1])
    journal_turns = [r for r in a.journal.records("arena") if r.get("phase") == "turn"]
    assert [t["sent"]["exchanges"] for t in journal_turns] == [0, 1, 2, 3, 4, 1, 2]
    assert "Notebook" in model.calls[6]["messages"][-1]["content"]


def test_the_stage_cards_that_are_ranked_start_pinned_and_a_card_that_is_looked_at_is_pinned_too(tmp_path, registry):
    model = Model(act(action="look", what="card", stage=STAGE), act(action="look", what="card", stage=OTHER),
                  act(action="look", what="card", stage=OTHER), act(action="look", what="board"))
    a = arena(tmp_path, registry, ("alpha", model), turns=4)
    a.run()
    start = model.calls[0]["messages"][0]["content"]
    assert start.startswith("Pinned for the whole contest") and "# stage effective.solve" in start and "# stage dof.passive_force" not in start
    assert a.preloaded == [STAGE]
    assert "already pinned" in model.calls[1]["messages"][-1]["content"]                       # it started pinned
    assert "is now pinned at the top" in model.calls[2]["messages"][-1]["content"]              # a card it asked for
    assert "already pinned" in model.calls[3]["messages"][-1]["content"]
    assert "# stage dof.passive_force" in model.calls[3]["messages"][0]["content"]              # ... and from then on it leads the prompt
    assert model.calls[3]["messages"][0]["content"].count("# stage effective.solve") == 1       # each card once
    assert "last 6 turns word for word" in a.system and "pinned at the top" in a.system


# ---- building on an earlier source: fork and patch ---------------------------------------------------------------------------------

SOURCE_REQUEST = f"\n```python {STAGE}\n{GOOD_SOLVER}\n```"
DOCSTRING = '"""A float64 dense solve of the effective system."""'


def edit_block(old, new, stage=STAGE):
    return f"\n```edit {stage}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n```"


def test_an_earlier_attempt_can_be_built_on_with_an_edit_instead_of_retyping_the_source(tmp_path, registry):
    model = Dyn(act(action="check", graph={"stages": {STAGE: {"entry": "Runtime"}}}, note="dense", _blocks=SOURCE_REQUEST),
                lambda m: act(action="check", graph={"stages": {STAGE: {"base": attempt_id(m)}}}, note="banded",
                              _blocks=edit_block(DOCSTRING, '"""A banded solve."""')),
                lambda m: act(action="check", graph={"stages": {STAGE: {"base": attempt_id(m)}}}))
    a = arena(tmp_path, registry, ("alpha", model), allow_unsandboxed=True, turns=3)
    a.run()
    first, second, third = a.attempts()
    assert '"""A banded solve."""' in second["graph"]["stages"][STAGE]["source"] and DOCSTRING not in second["graph"]["stages"][STAGE]["source"]
    assert second["graph"]["stages"][STAGE]["base"] == first["id"] and second["graph"]["stages"][STAGE]["edit"] == "+1/-1"
    assert third["graph"]["stages"][STAGE]["source"] == second["graph"]["stages"][STAGE]["source"]   # a base with no edit runs as it was
    assert third["graph"]["stages"][STAGE]["edit"] == "+0/-0" and third["digest"] == second["digest"]
    assert "from " + first["id"] + " +1/-1" in a._what(second) and second["note"] == "banded"


def test_an_edit_that_does_not_apply_costs_a_turn_and_says_where_to_look(tmp_path, registry):
    model = Dyn(act(action="check", graph={"stages": {STAGE: {"entry": "Runtime"}}}, _blocks=SOURCE_REQUEST),
                lambda m: act(action="check", graph={"stages": {STAGE: {"base": attempt_id(m)}}}, _blocks=edit_block("def solve(self, q):", "x")),
                lambda m: act(action="check", graph={"stages": {STAGE: {"base": "deadbeef"}}}))
    a = arena(tmp_path, registry, ("alpha", model), allow_unsandboxed=True, turns=3)
    a.run()
    assert len(a.attempts()) == 1
    assert "Not accepted: effective.solve: the edit to " in model.calls[2]["messages"][-1]["content"]
    assert "matches nowhere" in model.calls[2]["messages"][-1]["content"] and "nearest line is" in model.calls[2]["messages"][-1]["content"]
    assert "no attempt or entry 'deadbeef' is recorded" in a.by_name["alpha"].pending


def test_a_rivals_entry_can_be_forked_by_its_id_and_a_closed_contest_keeps_the_rivals_source_to_itself(tmp_path, registry):
    graph = {"stages": {STAGE: {"entry": "Runtime"}}}
    holder = {}
    alpha = Dyn(act(action="submit", graph=graph, note="dense", _blocks=SOURCE_REQUEST))
    a = arena(tmp_path, registry, ("alpha", alpha), allow_unsandboxed=True, turns=1)
    a.run()
    assert "Your entries: " in a.by_name["alpha"].pending and "build on it with base" in a.by_name["alpha"].pending
    holder["entry"] = next(r["id"] for g in a.groups() for r in g["rows"])
    beta2 = Dyn(act(action="check", graph={"stages": {STAGE: {"base": holder["entry"]}}}, _blocks=edit_block(DOCSTRING, '"""Mine."""')))
    a2 = arena(tmp_path, registry, ("beta", beta2), allow_unsandboxed=True, turns=1)
    a2.run()
    forked = [r for r in a2.attempts() if r["contestant"] == "beta"][-1]
    assert forked["graph"]["stages"][STAGE]["base"] == holder["entry"] and '"""Mine."""' in forked["graph"]["stages"][STAGE]["source"]
    closed = Dyn(act(action="check", graph={"stages": {STAGE: {"base": holder["entry"]}}}))
    a3 = arena(tmp_path, registry, ("gamma", closed), allow_unsandboxed=True, turns=1, open_code=False)
    a3.run()
    assert "this contest is closed: you can build on your own attempts and entries only" in a3.by_name["gamma"].pending


def test_a_base_with_no_source_for_the_stage_says_what_it_does_have(tmp_path, registry):
    model = Dyn(act(action="check", graph=FAILING),
                lambda m: act(action="check", graph={"stages": {STAGE: {"base": attempt_id(m)}}}))
    a = arena(tmp_path, registry, ("alpha", model), turns=2)
    a.run()
    assert f"has no source for {STAGE} (it replaced no stage with source)" in a.by_name["alpha"].pending


# ---- several actions in one turn ---------------------------------------------------------------------------------------------------

def test_a_turn_can_hold_several_actions_and_counts_once_against_the_turns(tmp_path, registry):
    graph = {"stages": {STAGE: {"entry": "Runtime"}}}
    other = GOOD_SOLVER.replace(DOCSTRING, '"""Another."""')
    reply = Reply(json.dumps(dict(actions=[dict(action="look", what="board"), dict(action="check", graph=graph, note="one"),
                                           dict(action="check", graph=graph, note="two")]))
                  + f"\n```python {STAGE}\n{GOOD_SOLVER}\n```\n```python {STAGE}\n{other}\n```", 100, 20)
    model = Model(reply, act(action="look", what="board"))
    a = arena(tmp_path, registry, ("alpha", model), allow_unsandboxed=True, turns=2)
    a.run()
    one, two = a.attempts()
    assert one["turn"] == two["turn"] == 1 and one["digest"] != two["digest"] and (one["note"], two["note"]) == ("one", "two")
    assert GOOD_SOLVER.strip() in one["graph"]["stages"][STAGE]["source"] and "Another." in two["graph"]["stages"][STAGE]["source"]
    told = model.calls[1]["messages"][-1]["content"]
    assert "[1 of 3] look" in told and "[2 of 3] check" in told and "[3 of 3] check" in told and told.count("This is attempt") == 2
    turns = [r for r in a.journal.records("arena") if r.get("phase") == "turn"]
    assert turns[0]["action"] == "look+check+check" and a.by_name["alpha"].turns == 2


def test_one_failing_action_does_not_stop_the_others_and_extra_actions_are_not_run(tmp_path, registry):
    reply = Reply(json.dumps(dict(actions=[dict(action="look", what="nothing")] + [dict(action="look", what="board")] * 7)), 100, 20)
    model = Model(reply)
    a = arena(tmp_path, registry, ("alpha", model), turns=1)
    a.run()
    told = a.by_name["alpha"].pending
    assert "[1 of 8] look\nNot accepted:" in told and "[2 of 8] look\nBoard for" in told and "[6 of 8]" in told and "[7 of 8]" not in told
    assert "Only the first 6 of your 8 actions were run." in told


# ---- what each turn cost ------------------------------------------------------------------------------------------------------------

def test_every_turn_records_what_was_sent_and_what_it_cost_in_input_output_and_cache(tmp_path, registry):
    model = Model(Reply(json.dumps(dict(action="look", what="board")), 1000, 40, 700), Reply(json.dumps(dict(action="look", what="board")), 1500, 60, 1200))
    a = arena(tmp_path, registry, ("alpha", model), turns=2)
    report = a.run()
    turns = [r for r in a.journal.records("arena") if r.get("phase") == "turn"]
    assert [(t["input_tokens"], t["output_tokens"], t["cached_tokens"], t["tokens"]) for t in turns] == [(1000, 40, 700, 1040), (1500, 60, 1200, 1560)]
    assert set(turns[0]["sent"]) == {"messages", "exchanges", "chars", "system", "pinned", "notebook", "state"} and turns[0]["sent"]["pinned"] > 0
    [summary] = report.contestants
    assert (summary["input_tokens"], summary["output_tokens"], summary["cached_tokens"], summary["tokens"]) == (2500, 100, 1900, 2600)


# ---- facts measured once ----------------------------------------------------------------------------------------------------------

def test_what_a_checks_inputs_look_like_is_measured_once_pinned_with_the_card_and_kept_for_later_contests(tmp_path, registry):
    calls = []

    def probe(request, **kw):
        calls.append(request)
        return CheckResult("error", reason="RuntimeError: probe: stopped on purpose",
                           output="host.facts(): ModelFacts(nv=32)\n  system.h: numpy.ndarray dtype=float32 shape=(2, 32, 32)")

    model = Model(act(action="look", what="board"))
    a = arena(tmp_path, registry, ("alpha", model), check=probe, runtime_facts=True, allow_unsandboxed=True, turns=1)
    a.run()
    assert len(calls) == 1 and calls[0]["tier"] == "real" and calls[0]["stage"] == STAGE
    pinned = model.calls[0]["messages"][0]["content"]
    assert "Measured on this scene at the real tier" in pinned and "system.h: numpy.ndarray dtype=float32 shape=(2, 32, 32)" in pinned
    assert [r["stage"] for r in a.journal.records("facts")] == [STAGE]
    again = Model(act(action="look", what="board"))
    b = arena(tmp_path, registry, ("alpha", again), check=probe, runtime_facts=True, allow_unsandboxed=True, turns=1)
    b.run()
    assert len(calls) == 1 and "shape=(2, 32, 32)" in again.calls[0]["messages"][0]["content"]       # read from the journal, not measured again


def test_a_call_that_does_not_go_through_loses_neither_the_last_result_nor_what_was_new(tmp_path, registry):
    alpha = Model(act(action="look", what="board"), LLMError("rate limited"), act(action="look", what="board"))
    beta = Model(submit(1, "plain gain"), act(action="look", what="board"), act(action="look", what="board"))
    a = arena(tmp_path, registry, ("alpha", alpha), ("beta", beta), turns=3)
    a.run()
    retry = alpha.calls[2]["messages"][-1]["content"]                         # the call after the one that failed
    assert "Your last reply did not go through (rate limited)" in retry
    assert "Board for" in retry                                               # the result of its last turn is still there
    assert "New on the board since your last move" in retry and "beta: speedup" in retry     # and beta's entry was not used up


def test_the_arena_says_what_is_taking_the_time_and_what_each_turn_cost(tmp_path, registry):
    events = []
    graph = {"stages": {STAGE: {"impl": FAST, "params": {"gain": 1}}}}
    reply = Reply(json.dumps(dict(actions=[dict(action="check", graph=graph), dict(action="submit", graph=graph, note="n")])), 1200, 340, 200)
    a = arena(tmp_path, registry, ("alpha", Model(reply)), turns=1, on_event=events.append)
    a.run()
    kinds = [(e["event"], e.get("what", "")[:24]) for e in events if e["event"] in ("work", "turn")]
    assert kinds == [("work", "asking m"), ("work", "running the checks"), ("work", "running the checks"),
                     ("work", "timed run at 1, 2 worlds"), ("turn", "")]
    turn = next(e for e in events if e["event"] == "turn")
    assert (turn["input_tokens"], turn["output_tokens"], turn["cached_tokens"]) == (1200, 340, 200)
    assert turn["seconds"] >= turn["model_seconds"] >= 0
    assert [e["turn"] for e in events if e["event"] == "work"] == [1, 1, 1, 1]
