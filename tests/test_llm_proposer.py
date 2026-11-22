"""What the LLM proposer sends, how it reads the answer, what it charges, and what it does when the model misbehaves."""

import json

import pytest

import phost
from phost import Impl, ImplKind, ParamSpec
from phost_agent import ACCEPTED, MEASURED, REJECTED, Outcome, Proposal, ProposerError, ProposerFatal, Task
from phost_agent.cards import task_brief
from phost_agent.llm import LLMError, LLMFatal, LLMProposer, Reply, outcome_line, parse_reply, system_prompt, turn_prompt
from phost_agent.task import resolve
from conftest import GOOD_SOLVER, SCENE, STAGE


class Scripted:
    """A model that answers from a list (a Reply, or an exception to raise) and keeps what it was asked."""

    name, model = "fake", "fake-1"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def complete(self, system, messages, *, max_tokens):
        self.calls.append(dict(system=system, messages=[dict(m) for m in messages], max_tokens=max_tokens))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def say(text, i=100, o=20, **kw):
    return Reply(text, i, o, **kw)


@pytest.fixture(scope="module")
def brief():
    task = Task(SCENE, STAGE, worlds=(1, 2), steps=5, launches=2, repeats=3, label="a label")
    return task_brief(task, resolve(task), None, None)


def outcome(attempt, proposal, verdict, reached="run", speedup=None, reason="", memo=False, accuracy=None):
    return Outcome(attempt, proposal, "k", reached, verdict, reason, speedup, accuracy, memo)


def impl(**params):
    return Proposal(STAGE, impl=f"{STAGE}/x", params=params)


# ---- reading an answer ------------------------------------------------------------------------------------------------------------------

def test_the_json_may_be_bare_fenced_or_surrounded_by_prose():
    for text in ('{"proposal": {"impl": "a/b", "params": {"n": 2}}}',
                 'Sure.\n```json\n{"proposal": {"impl": "a/b", "params": {"n": 2}}}\n```\nHope that helps.',
                 'I think {"proposal": {"impl": "a/b", "params": {"n": 2}}} is best.',
                 '{"stage": "s", "impl": "a/b", "params": {"n": 2}}'):
        assert parse_reply(text)["proposal"]["impl"] == "a/b"


def test_source_may_be_in_a_python_block_after_the_json_and_braces_in_it_do_not_confuse_the_reader():
    text = '{"proposal": {"entry": "Runtime", "note": "n"}}\n```python\nclass Runtime:\n    table = {"a": {"b": 1}}\n```'
    proposal = parse_reply(text)["proposal"]
    assert proposal["source"] == 'class Runtime:\n    table = {"a": {"b": 1}}\n\n'.replace("}}\n\n", "}}\n")
    inside = {"proposal": {"source": "class Runtime: pass"}}
    assert parse_reply(json.dumps(inside) + "\n```python\nOTHER\n```")["proposal"]["source"] == "class Runtime: pass"


def test_a_stop_is_read_a_claimed_cost_is_ignored_and_nonsense_is_refused_with_a_reason():
    assert parse_reply('{"stop": "nothing left to try"}') == {"stop": "nothing left to try"}
    assert "tokens" not in parse_reply('{"proposal": {"impl": "a/b", "tokens": 1}}')["proposal"]
    for text, why in (("", "empty"), ("I would try a bigger tolerance.", "no JSON object"), ('{"foo": 1}', "no JSON object"),
                      ('{"proposal": 3}', "neither")):
        with pytest.raises(ValueError, match=why):
            parse_reply(text)


# ---- one call, one proposal --------------------------------------------------------------------------------------------------------------

def test_a_proposal_defaults_the_stage_keeps_the_note_and_is_charged_every_token_of_the_call(brief):
    model = Scripted(say('{"proposal": {"impl": "effective.solve/x", "params": {"a": 1}, "note": "try a"}}', 900, 60))
    proposal = LLMProposer(model).propose(brief, [])
    assert (proposal.stage, proposal.impl, proposal.params, proposal.note, proposal.tokens) == (STAGE, "effective.solve/x", {"a": 1}, "try a", 960)


def test_a_source_proposal_comes_through_with_its_source(brief):
    proposal = LLMProposer(Scripted(say('{"proposal": {"entry": "Runtime"}}\n```python\n' + GOOD_SOLVER + '\n```'))).propose(brief, [])
    assert proposal.source is not None and "class Runtime" in proposal.source and proposal.impl is None


def test_a_stop_ends_the_search_and_remembers_why(brief):
    proposer = LLMProposer(Scripted(say('{"stop": "the ceiling is spent"}')))
    assert proposer.propose(brief, []) is None and proposer.stop_reason == "the ceiling is spent"


def test_the_system_prompt_is_the_instructions_and_the_brief_and_does_not_change_between_turns(brief):
    model = Scripted(say('{"proposal": {"impl": "a/b"}}'), say('{"proposal": {"impl": "a/c"}}'))
    proposer = LLMProposer(model)
    first = proposer.propose(brief, [])
    proposer.propose(brief, [outcome(1, first, REJECTED, "static", reason="no such impl")])
    assert model.calls[0]["system"] == model.calls[1]["system"] == system_prompt(brief)
    system = model.calls[0]["system"]
    assert "ONE candidate" in system and "Proposing it gains nothing" in system and "effective.solve" in system and "## ports" in system and "Ceiling" not in system
    assert model.calls[0]["messages"][0]["content"].startswith("Attempt 1")
    assert model.calls[1]["messages"][0]["content"].startswith("Attempt 2") and "no such impl" in model.calls[1]["messages"][0]["content"]
    assert model.calls[0]["max_tokens"] == 16000


# ---- when the model gets it wrong ------------------------------------------------------------------------------------------------------------

def test_an_unusable_answer_is_sent_back_once_with_the_reason_and_the_two_calls_are_one_attempt(brief):
    model = Scripted(say("Let me think about it.", 500, 10), say('{"proposal": {"impl": "a/b"}}', 700, 15))
    proposal = LLMProposer(model).propose(brief, [])
    assert proposal.impl == "a/b" and proposal.tokens == 500 + 10 + 700 + 15 and len(model.calls) == 2
    second = model.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"] and "no JSON object" in second[2]["content"]


def test_an_answer_with_an_unknown_key_is_repaired_the_same_way(brief):
    model = Scripted(say('{"proposal": {"impl": "a/b", "confidence": 0.9}}'), say('{"proposal": {"impl": "a/b"}}'))
    assert LLMProposer(model).propose(brief, []).impl == "a/b" and "unknown keys" in model.calls[1]["messages"][-1]["content"]


def test_two_unusable_answers_cost_the_attempt_and_what_they_cost_in_tokens(brief):
    with pytest.raises(ProposerError, match="could not be used") as caught:
        LLMProposer(Scripted(say("no", 100, 5), say("still no", 200, 5))).propose(brief, [])
    assert caught.value.tokens == 310 and not isinstance(caught.value, ProposerFatal)
    with pytest.raises(ProposerError):
        LLMProposer(Scripted(say("no")), repair=0).propose(brief, [])


def test_a_reply_cut_off_a_refusal_and_a_dropped_call_each_cost_an_attempt_not_the_search(brief):
    with pytest.raises(ProposerError, match="cut off at 16000 tokens") as cut:
        LLMProposer(Scripted(say("{", 10, 16000, stop="length"))).propose(brief, [])
    assert cut.value.tokens == 16010 and not isinstance(cut.value, ProposerFatal)
    with pytest.raises(ProposerError, match="declined"):
        LLMProposer(Scripted(say("", stop="refusal"))).propose(brief, [])
    with pytest.raises(LLMError, match="rate") as dropped:
        LLMProposer(Scripted(say("bad", 100, 10), LLMError("rate limited"))).propose(brief, [])
    assert dropped.value.tokens == 110 and not isinstance(dropped.value, ProposerFatal)      # the first call is still paid for


def test_a_failure_that_would_repeat_is_fatal(brief):
    with pytest.raises(ProposerFatal, match="no key"):
        LLMProposer(Scripted(LLMFatal("no key"))).propose(brief, [])


def test_usage_adds_up_across_calls_including_the_cached_share(brief):
    model = Scripted(say("x", 1000, 10, cached_tokens=0), say('{"proposal": {"impl": "a/b"}}', 1000, 20, cached_tokens=900, estimated=True))
    proposer = LLMProposer(model)
    proposer.propose(brief, [])
    assert proposer.usage == dict(calls=2, input_tokens=2000, output_tokens=30, cached_tokens=900, estimated=True)
    assert proposer.name == "llm:fake:fake-1"


# ---- what it is told ---------------------------------------------------------------------------------------------------------------------

def test_results_are_one_line_each_and_say_where_a_candidate_stopped_and_how_the_ones_that_ran_did():
    lines = [outcome_line(outcome(1, impl(a=1), REJECTED, "cheap", reason="residual 3e-2\nexceeds 1e-4")),
             outcome_line(outcome(2, impl(a=2), MEASURED, speedup=0.41, accuracy=1.234, reason="not faster")),
             outcome_line(outcome(3, impl(a=3), ACCEPTED, speedup=1.5)),
             outcome_line(outcome(4, impl(a=3), ACCEPTED, speedup=1.5, memo=True)),
             outcome_line(outcome(5, impl(), REJECTED, "none", reason="proposer failed"))]
    assert lines[0] == '#1 effective.solve/x {"a":1} -> rejected at the cheap tier: residual 3e-2 exceeds 1e-4'
    assert lines[1] == '#2 effective.solve/x {"a":2} -> measured, 0.41x of native, accuracy ratio 1.23 (not faster)'
    assert lines[2].endswith("accepted, 1.5x of native") and lines[3].endswith("[a repeat, answered from the memo]")
    assert "rejected before any check" in lines[4]
    assert all("\n" not in line for line in lines)


def test_the_turn_names_the_best_so_far_trims_old_results_and_shows_only_the_latest_source(brief):
    history = [outcome(i, impl(a=i), ACCEPTED if i == 3 else MEASURED, speedup={3: 2.5}.get(i, 0.5)) for i in range(1, 7)]
    history[1] = outcome(2, Proposal(STAGE, source="FIRST SOURCE"), MEASURED, speedup=0.5)
    history[4] = outcome(5, Proposal(STAGE, source="LATEST SOURCE\n" + "x" * 50), MEASURED, speedup=0.5)
    turn = turn_prompt(brief, history, history_limit=4, source_chars=20)
    assert turn.startswith("Attempt 7 of 20.") and "(2 earlier results left out)" in turn
    assert "#1 " not in turn and "#6 " in turn and "Best accepted so far: #3 at 2.5x" in turn
    assert "LATEST SOURCE" in turn and "FIRST SOURCE" not in turn and "# ... (cut)" in turn
    assert "Nothing has been accepted yet." in turn_prompt(brief, [history[0]])
    assert "No attempts yet." in turn_prompt(brief, [])


def test_nothing_about_the_machine_or_the_person_is_sent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/zebedee_q")
    monkeypatch.setenv("LOGNAME", "zebedee")
    monkeypatch.setenv("PHOST_HOME", str(tmp_path / "state"))
    task = Task(SCENE, STAGE, worlds=(1, 2), label="zebedee in /home/zebedee_q/run, state " + str(tmp_path / "state"))
    leaky = task_brief(task, resolve(task), None, None)
    model = Scripted(say('{"proposal": {"impl": "a/b"}}'))
    reason = f"/home/zebedee_q/x failed for zebedee; see {tmp_path}/state/journal.jsonl"
    LLMProposer(model).propose(leaky, [outcome(1, impl(), REJECTED, "static", reason=reason)])
    sent = model.calls[0]["system"] + model.calls[0]["messages"][0]["content"]
    assert "zebedee" not in sent and "/home/" not in sent and str(tmp_path) not in sent
    assert "<user>" in sent and "<state>" in sent
