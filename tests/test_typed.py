"""Typed forms in the arena: the card that describes one, and the preview that tries one without spending a check."""

import json

import pytest

pytest.importorskip("phost_author")

import phost
from phost import ContractError
from phost_agent.arena import Arena, ArenaTask, Contestant
from phost_agent.cards import render_card, stage_card
from phost_agent.llm import Reply
from phost_agent.typed import MAX_ROWS, preview_text, typed_note, typed_stages
from phost_lab.bootstrap import load_registry
from conftest import SCENE, STAGE
from test_arena import REPORT, Model, act, arena

TYPED = f"{STAGE}/expression"
CG_STEP = "x = x + a*(rhs - A @ x)/diag"                # a Jacobi pass: fine for a small weight, it blows up at a=1 here


@pytest.fixture(scope="module")
def typed_registry():
    return load_registry(plugins=("phost_author.expression_impls",), registry=phost.Registry())


@pytest.fixture(scope="module")
def plain_registry():
    return load_registry(registry=phost.Registry())


def stages_of(registry):
    return sorted(c.name for c in registry.contracts())


def preview(registry, **action):
    return preview_text(registry, stages_of(registry), dict(action="preview", **action))


# ---- what a contestant is told -----------------------------------------------------------------------------------------------------

def test_the_stages_that_take_a_typed_form_are_known_only_when_phost_author_is_loaded(typed_registry, plain_registry):
    assert STAGE in typed_stages(typed_registry, stages_of(typed_registry)) and "dof.passive_force" in typed_stages(typed_registry, stages_of(typed_registry))
    assert typed_stages(plain_registry, stages_of(plain_registry)) == []
    assert typed_note(plain_registry, STAGE) == ""


def test_a_solver_stage_card_says_what_a_step_may_say_and_starts_from_a_working_program(typed_registry):
    note = typed_note(typed_registry, STAGE)
    assert f"`{TYPED}`" in note and "`init`" in note and "`step`" in note and "iterations" in note and "tolerance" in note
    assert "Reductions:" in note and "dot" in note and "x = 0*rhs" in note and "Ap = A @ p" in note
    assert "defaults 1 for a, 0 for b..h" in note
    compact = render_card(stage_card(typed_registry, STAGE), compact=True, typed=note)
    assert note in compact and "## typed law" not in compact and f"`{TYPED}` (expression [approximate]) Typed solver step: the typed form above" in compact
    full = render_card(stage_card(typed_registry, STAGE), typed=note)
    assert note in full and "## typed law" not in full and "'x = 0*rhs\\n" not in full         # the program is not also repeated as a repr


def test_a_card_without_a_note_keeps_the_raw_form_in_the_full_card(typed_registry):
    full = render_card(stage_card(typed_registry, STAGE))
    assert "## typed law" in full and "typed form:" not in full


def test_a_pointwise_form_is_described_without_a_default_program(typed_registry):
    note = typed_note(typed_registry, "dof.passive_force")
    assert "`dof.passive_force/expression`" in note and "`expr`" in note and "The defaults, a working program" not in note
    assert note.count("Functions:") == 1                                        # the implementation's own text lists them already


def test_the_system_prompt_offers_preview_only_when_there_is_something_to_preview(tmp_path, typed_registry, plain_registry):
    typed = arena(tmp_path / "t", typed_registry, ("a", Model()))
    typed.run()
    assert '"action": "preview"' in typed.system and f"Typed forms: {STAGE}" in typed.system and "dof.passive_force" not in typed.system.split("Typed forms:")[1].split("\n")[0]
    plain = arena(tmp_path / "p", plain_registry, ("a", Model()))
    plain.run()
    assert '"action": "preview"' not in plain.system and "{preview}" not in plain.system


def test_a_typed_stage_starts_the_contest_with_its_form_on_the_pinned_card(tmp_path, typed_registry):
    model = Model()
    a = arena(tmp_path, typed_registry, ("a", model), turns=1)
    a.run()
    first = model.calls[0]["messages"][0]["content"]
    assert f"typed form: `{TYPED}`" in first and "Ap = A @ p" in first


# ---- the preview ------------------------------------------------------------------------------------------------------------------

def test_a_preview_answers_one_line_for_the_params_and_one_for_each_variant(typed_registry):
    text = preview(typed_registry, stage=STAGE, variants=[dict(iterations=4), dict(step=CG_STEP, iterations=30)])
    lines = text.splitlines()
    assert lines[0].startswith(f"Preview of `{TYPED}`") and "nothing was checked, timed or counted" in lines[0]
    assert len(lines) == 4
    assert lines[1].startswith("- params: residual at pass 1:") and "under the 0.01 it claims from pass" in lines[1]
    assert lines[2].startswith("- variant 1 (iterations=4): ") and "never under the 0.01 it claims" in lines[2]
    assert lines[3].startswith('- variant 2 (step="x = x + a*(rhs - A @ x)/diag", iterations=30): ') and "never under" in lines[3]


def test_a_program_that_diverges_shows_it_and_one_that_converges_shows_when(typed_registry):
    good = preview(typed_registry, stage=STAGE).splitlines()[1]
    bad = preview(typed_registry, stage=STAGE, params=dict(step=CG_STEP, iterations=30)).splitlines()[1]
    assert "under the 0.01 it claims from pass" in good and "solution error" in good
    last = float(bad.split("30: ")[1].split(" ")[0].rstrip(","))
    assert last > 1 and "never under" in bad                                      # the residual grew, which is what it should say


def test_a_program_that_stops_being_a_number_says_where(typed_registry):
    text = preview(typed_registry, stage=STAGE, params=dict(step="x = x/0", iterations=5))
    assert "not finite" in text.splitlines()[1] and "pass 1" in text.splitlines()[1]


def test_the_condition_number_is_a_list_and_each_gets_its_own_line(typed_registry):
    lines = preview(typed_registry, stage=STAGE, condition=[10, 10000]).splitlines()
    assert [line.split(": ")[0] for line in lines[1:]] == ["- params, condition 10", "- params, condition 1e+04"]
    easy, hard = (float(line.split("20: ")[1].split(" ")[0].rstrip(",")) for line in lines[1:])
    assert easy < 1e-6 and hard > easy                                            # the same program is harder when it is worse conditioned


def test_a_variant_that_is_not_accepted_does_not_cost_the_others(typed_registry):
    text = preview(typed_registry, stage=STAGE, variants=[dict(step="y = wat"), dict(iterations=5), dict(iterationz=5), dict(iterations=0)])
    lines = text.splitlines()
    assert len(lines) == 6                                                        # the head, the params, and four variants
    assert "not accepted" in lines[2] and "unknown name 'wat'" in lines[2]
    assert lines[3].startswith("- variant 2 (iterations=5): residual at pass")
    assert "not accepted" in lines[4] and "iterationz" in lines[4]
    assert "not accepted" in lines[5] and "minimum 1" in lines[5]
    only = preview(typed_registry, stage=STAGE, variants=[dict(iterations=0)]).splitlines()
    assert "not accepted" in only[2]


def test_a_bad_condition_is_told_once_not_once_per_line(typed_registry):
    text = preview(typed_registry, stage=STAGE, condition=[10, 1e12])
    assert text.count("not accepted") == 1 and "condition 1e+12" in text


def test_a_preview_asks_for_no_more_than_it_will_answer(typed_registry):
    with pytest.raises(ContractError, match=f"at most {MAX_ROWS} lines"):
        preview(typed_registry, stage=STAGE, variants=[dict(iterations=i) for i in range(1, MAX_ROWS + 1)])
    with pytest.raises(ContractError, match="at most"):
        preview(typed_registry, stage=STAGE, variants=[{}] * 7, condition=[10, 100])
    preview(typed_registry, stage=STAGE, variants=[dict(iterations=i) for i in range(1, MAX_ROWS)])      # the params and eleven are twelve


def test_a_preview_of_the_wrong_thing_says_what_can_be_previewed(typed_registry):
    with pytest.raises(ContractError, match=r"no typed form for 'no.such'.*effective\.solve"):
        preview(typed_registry, stage="no.such")
    with pytest.raises(ContractError, match='"params" is an object'):
        preview(typed_registry, stage=STAGE, params="step")
    with pytest.raises(ContractError, match='"condition" is a number or a list'):
        preview(typed_registry, stage=STAGE, condition="hard")
    with pytest.raises(ContractError, match='"size" is a number'):
        preview(typed_registry, stage=STAGE, size="big")


def test_a_pointwise_law_is_swept_and_a_channel_law_is_read(typed_registry):
    text = preview(typed_registry, stage="dof.passive_force", sweep="v", lo=-2, hi=2, params=dict(expr="-a*v", a=3.0),
                   variants=[dict(expr="1/0")])
    lines = text.splitlines()
    assert lines[1].startswith("- params: over v -2..2: force in [-6, 6]") and "damping 3..3" in lines[1]
    assert "NOT FINITE" in lines[2]
    channels = preview(typed_registry, stage="probe.state", params=dict(expr="peak = maxabs(qvel)"))
    assert channels.splitlines()[1].startswith("- params: peak = ")


def test_a_preview_is_short_beside_the_source_it_replaces(typed_registry):
    step = "Ap = A @ p\nalpha = rz / max(dot(p, Ap), 1e-30)\nx = x + alpha*p\nr = r - alpha*Ap\nz = r/diag\nrz_new = dot(r, z)\np = z + (rz_new / max(rz, 1e-30))*p\nrz = rz_new"
    asked = json.dumps(dict(action="preview", stage=STAGE, params=dict(step=step), variants=[dict(iterations=n) for n in (5, 10, 20)]))
    answered = preview(typed_registry, stage=STAGE, params=dict(step=step), variants=[dict(iterations=n) for n in (5, 10, 20)])
    assert len(asked) < 600 and len(answered) < 1200                                # four ideas, asked and answered; a kernel is several thousand characters


# ---- in the arena -----------------------------------------------------------------------------------------------------------------

def test_a_preview_in_the_arena_spends_no_check_and_no_attempt_and_is_a_read(tmp_path, typed_registry):
    model = Model(act(action="preview", stage=STAGE, variants=[dict(iterations=3)]), act(action="look", what="board"))
    a = arena(tmp_path, typed_registry, ("a", model), turns=2)
    a.run()
    assert a.calls["check"] == [] and a.calls["timed"] == [] and a.attempts() == []
    assert model.calls[1]["messages"][-1]["content"].startswith(f"Preview of `{TYPED}`")
    turns = [r for r in a.journal.records("arena") if r.get("phase") == "turn"]
    assert [t["action"] for t in turns] == ["preview", "look"]
    assert (1, f"preview {STAGE}") in a.by_name["a"].reads


def test_a_preview_and_a_check_in_one_turn_are_two_actions_of_one_call(tmp_path, typed_registry):
    graph = {"stages": {STAGE: {"impl": TYPED, "params": {"iterations": 12}}}}
    model = Model(Reply(json.dumps({"actions": [dict(action="preview", stage=STAGE, variants=[dict(iterations=12)]),
                                                dict(action="check", graph=graph, note="cg")]}), 100, 20))
    a = arena(tmp_path, typed_registry, ("a", model), turns=1)
    a.run()
    assert len(model.calls) == 1 and len(a.attempts()) == 1 and {r["tier"] for r in a.calls["check"]} == {"cheap", "real"}
    assert a.by_name["a"].pending.count("[1 of 2] preview") == 1 and "[2 of 2] check" in a.by_name["a"].pending
    attempt = a.attempts()[0]
    assert attempt["graph"]["stages"][STAGE]["impl"] == TYPED


def test_preview_is_an_unknown_action_where_nothing_is_typed(tmp_path, plain_registry):
    model = Model(act(action="preview", stage=STAGE), act(action="look", what="board"))
    a = arena(tmp_path, plain_registry, ("a", model), turns=2)
    a.run()
    assert "Unknown action 'preview'; the actions are look, check and submit." in model.calls[1]["messages"][-1]["content"]


def test_where_something_is_typed_the_unknown_action_message_names_preview(tmp_path, typed_registry):
    model = Model(act(action="frobnicate"), act(action="look", what="board"))
    a = arena(tmp_path, typed_registry, ("a", model), turns=2)
    a.run()
    assert "the actions are look, preview, check and submit." in model.calls[1]["messages"][-1]["content"]


def test_a_closed_contest_can_still_preview_since_it_shows_no_one_elses_work(tmp_path, typed_registry):
    model = Model(act(action="preview", stage=STAGE), act(action="look", what="board"))
    a = arena(tmp_path, typed_registry, ("a", model), turns=2, open_code=False)
    a.run()
    assert model.calls[1]["messages"][-1]["content"].startswith("Preview of")


def test_a_preview_that_cannot_be_read_comes_back_as_not_accepted(tmp_path, typed_registry):
    model = Model(act(action="preview", stage="no.such"), act(action="look", what="board"))
    a = arena(tmp_path, typed_registry, ("a", model), turns=2)
    a.run()
    assert "Not accepted: no typed form for 'no.such'" in model.calls[1]["messages"][-1]["content"]
