import ast
import json

import pytest

import phost
from phost import ContractError
from phost_agent import Gate, Proposal, Task, resolve, stage_card, task_brief, render_brief, render_card, write_candidate
from phost_lab.bootstrap import load_registry
from conftest import GOOD_SOLVER, SCENE, STAGE


@pytest.fixture(scope="module")
def registry():
    return load_registry()


# ---- proposals ---------------------------------------------------------------------------------------

def test_a_proposal_names_an_implementation_or_source_and_never_both_or_neither():
    Proposal("s", impl="s/x")
    Proposal("s", source="class Runtime: pass")
    for kwargs in (dict(), dict(impl="s/x", source="x = 1")):
        with pytest.raises(ContractError, match="exactly one"):
            Proposal("s", **kwargs)


def test_a_source_proposal_takes_no_params():
    with pytest.raises(ContractError, match="no params"):
        Proposal("s", source="x = 1", params={"k": 1})


def test_a_proposal_round_trips_and_refuses_unknown_keys():
    p = Proposal("s", impl="s/x", params={"k": 1}, note="why", tokens=12)
    assert Proposal.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    with pytest.raises(ContractError, match="unknown keys"):
        Proposal.from_dict(dict(stage="s", impl="s/x", priority=1))
    with pytest.raises(ContractError, match="stage is required"):
        Proposal.from_dict(dict(impl="s/x"))


def test_a_source_proposal_is_identified_by_its_text():
    a, b = Proposal("s", source="x = 1"), Proposal("s", source="x = 2")
    assert a.impl_id != b.impl_id and a.impl_id == Proposal("s", source="x = 1", note="other words").impl_id
    assert a.impl_id.startswith("s/agent-")


def test_a_candidate_is_written_as_a_plugin_the_lab_can_import(registry, tmp_path, monkeypatch):
    proposal = Proposal(STAGE, source=GOOD_SOLVER)
    candidate = write_candidate(proposal, registry, tmp_path)
    again = write_candidate(proposal, registry, tmp_path)
    assert candidate == again and len(list(tmp_path.glob("*.py"))) == 2          # equal source is one candidate
    monkeypatch.syspath_prepend(str(tmp_path))
    fresh = phost.Registry()
    from phost.stages import register_all
    register_all(fresh)
    import importlib
    importlib.import_module(candidate.module).register(fresh)
    impl = fresh.impl(candidate.impl_id)
    assert impl.stage == STAGE and impl.kind is phost.ImplKind.KERNEL and impl.factory is not None


def test_only_source_becomes_a_candidate(registry, tmp_path):
    with pytest.raises(ContractError, match="only a source proposal"):
        write_candidate(Proposal(STAGE, impl="effective.solve/native"), registry, tmp_path)


# ---- cards and briefs --------------------------------------------------------------------------------

def test_a_stage_card_says_what_the_stage_is_and_what_exists_for_it(registry):
    card = stage_card(registry, STAGE)
    assert card["name"] == STAGE and card["swap_class"] == "exact" and "native run" in card["judging"]
    assert [p["name"] for p in card["ports"]] == ["h", "rhs", "qacc"]
    assert any(i["id"] == "effective.solve/native" for i in card["impls"])
    assert card["tolerances"] and card["entry"] == "solve"
    json.dumps(card)                                                    # a card is plain data


def test_a_card_renders_as_markdown_with_every_port(registry):
    text = render_card(stage_card(registry, STAGE))
    assert text.startswith("# stage effective.solve") and "| qacc | out |" in text and "## gates" in text


def test_a_brief_carries_the_scene_the_ceiling_the_funnel_and_the_shape_of_an_answer(registry):
    task = Task(SCENE, STAGE)
    gate = Gate(True, "up to 4.00x available", SCENE, STAGE, "exact", share=0.75, ceiling=4.0, basis="memo")
    brief = task_brief(task, resolve(task, registry), gate)
    assert brief["scene"]["engine"] == "chain" and brief["scene"]["device_kind"] == "cpu"
    assert brief["ceiling"]["ceiling"] == 4.0
    assert [f["tier"][0] for f in brief["funnel"]] == list("012345")
    assert set(brief["answer"]["shapes"]) == {"impl", "source"}
    assert brief["budget"]["workers"] == 1                              # one at a time unless a budget says otherwise
    json.dumps(brief)


def test_the_source_skeleton_in_a_brief_is_valid_python_naming_the_stages_entry_point(registry):
    task = Task(SCENE, STAGE)
    skeleton = task_brief(task, resolve(task, registry), None)["answer"]["source_skeleton"]
    tree = ast.parse(skeleton)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert {n.name for n in cls.body if isinstance(n, ast.FunctionDef)} == {"__init__", "solve", "metrics"}
    assert "world" in skeleton                                          # the ports and their shapes are in it


def test_a_brief_for_a_stage_whose_reference_is_the_implementations_own_asks_for_one(registry):
    task = Task(SCENE, "dof.passive_force", goal="behaviour")
    skeleton = task_brief(task, resolve(task, registry), None)["answer"]["source_skeleton"]
    assert "def reference(inputs, params)" in skeleton


def test_a_brief_renders_with_the_card_after_it(registry):
    task = Task(SCENE, STAGE)
    text = render_brief(task_brief(task, resolve(task, registry), None))
    assert text.startswith(f"# task {task.id}: improve `{STAGE}`") and "## ports" in text and "```json" in text


def test_an_unknown_stage_is_refused_when_the_task_is_resolved(registry):
    with pytest.raises(ContractError, match="no contract"):
        resolve(Task(SCENE, "no.such.stage"), registry)
