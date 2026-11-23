"""The pieces the arena's prompt is built from, on their own: edits to an earlier source, several actions in a reply, the window,
the notebook and the board's delta."""

import pytest

from phost import ContractError
from phost_agent.arena_text import Blocks, MAX_ACTIONS, parse_actions, render_board_delta
from phost_agent.context import OLD_RESULT, SLACK, Exchange, assemble, condense, notebook_text, pinned_text, trim
from phost_agent.edits import apply_hunks, line_changes, parse_hunks

SOURCE = "import numpy as np\n\nclass Runtime:\n    def solve(self, s):\n        x = 1\n        y = 2\n        return x + y\n"


def hunk(old, new):
    return f"<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n"


# ---- edits ------------------------------------------------------------------------------------------------------------------------

def test_a_hunk_replaces_the_one_place_it_matches_and_hunks_apply_in_order():
    block = hunk("        x = 1", "        x = 10") + hunk("        return x + y", "        return x * y")
    new = apply_hunks(SOURCE, parse_hunks(block))
    assert "x = 10" in new and "return x * y" in new and new.endswith("\n") and "x = 1\n" not in new
    assert line_changes(SOURCE, new) == (2, 2)
    later = apply_hunks(SOURCE, parse_hunks(hunk("        x = 1", "        x = 5") + hunk("        x = 5", "        x = 6")))
    assert "x = 6" in later                                       # the second hunk sees the first one's result


def test_trailing_whitespace_does_not_stop_a_match_and_an_empty_replace_deletes_lines():
    assert "y = 2" not in apply_hunks("a\ny = 2   \nb\n", parse_hunks(hunk("y = 2", "").replace("\n\n=======", "\n=======")))


@pytest.mark.parametrize("block,message", [
    (hunk("not in the source", "x"), "matches nowhere"),
    ("", "no hunks"),
    ("<<<<<<< SEARCH\nx\n", "not closed"),
    ("stray text\n", "expected"),
])
def test_an_edit_that_cannot_apply_says_why(block, message):
    with pytest.raises(ContractError, match=message):
        apply_hunks(SOURCE, parse_hunks(block))


def test_a_search_that_matches_two_places_or_names_nothing_is_refused_with_where():
    twice = "def f():\n    return 1\n\ndef g():\n    return 1\n"
    with pytest.raises(ContractError, match=r"matches 2 places \(lines 2, 5\)"):
        apply_hunks(twice, parse_hunks(hunk("    return 1", "    return 2")))
    with pytest.raises(ContractError, match="empty"):
        apply_hunks(SOURCE, [("   ", "x")])
    with pytest.raises(ContractError, match=r"nearest line is 5: 'x = 1'"):
        apply_hunks(SOURCE, parse_hunks(hunk("        x = 11", "x")))


# ---- several actions in a reply ----------------------------------------------------------------------------------------------------

def test_one_action_and_a_list_of_actions_both_parse_and_blocks_are_taken_in_order_per_stage():
    one, blocks = parse_actions('Plan: {"action": "check", "graph": {}}\n```python s\nA\n```\n```python s\nB\n```\n```edit s\nE\n```')
    assert [a["action"] for a in one] == ["check"]
    many, blocks = parse_actions('{"actions": [{"action": "look", "what": "board"}, {"action": "check", "graph": {}}]}'
                                 '\n```python s\nA\n```\n```python s\nB\n```\n```edit s\nE\n```')
    assert [a["action"] for a in many] == ["look", "check"]
    assert blocks.take("python", "s") == "A\n" and blocks.take("python", "s") == "B\n" and blocks.take("python", "s") is None
    assert blocks.take("edit", "s") == "E\n" and blocks.take("edit", "s") is None


def test_an_untagged_block_is_used_only_when_the_caller_says_it_is_unambiguous():
    _, blocks = parse_actions('{"action": "check"}\n```python\nA\n```')
    assert blocks.take("python", "s") is None and blocks.take("python", "s", untagged_ok=True) == "A\n"
    assert Blocks([("", "x"), ("s", "y")]).untagged("python") == 1


@pytest.mark.parametrize("text,message", [
    ("", "empty"), ("no json here", "no JSON object"), ('{"actions": []}', "non-empty list"),
    ('{"actions": [1, 2]}', "non-empty list"), ('{"actions": [{"what": "board"}]}', "non-empty list")])
def test_a_reply_that_holds_no_action_says_what_was_wrong(text, message):
    with pytest.raises(ValueError, match=message):
        parse_actions(text)


def test_json_inside_a_code_block_is_not_mistaken_for_the_action():
    actions, _ = parse_actions('```python s\nd = {"action": "submit"}\n```\n{"action": "check"}')
    assert [a["action"] for a in actions] == ["check"] and MAX_ACTIONS == 6


# ---- the window and the pinned cards ----------------------------------------------------------------------------------------

def exchanges(n):
    return [Exchange(i, f"u{i}", f"a{i}") for i in range(1, n + 1)]


def test_the_window_is_trimmed_in_blocks_so_the_start_of_the_prompt_holds_still_between_trims():
    starts = []
    kept = []
    for n in range(1, 16):
        kept = trim(kept + exchanges(n)[-1:], 4)
        starts.append(kept[0].turn)
    assert SLACK == 3 and max(len(trim(exchanges(n), 4)) for n in range(1, 30)) == 4 + SLACK
    assert starts[:7] == [1] * 7 and starts[7] == 5 and len(set(starts[7:14])) <= 2 and starts == sorted(starts)


def test_assemble_puts_the_pinned_text_first_and_keeps_roles_alternating():
    messages = assemble("PINNED", exchanges(2), "newest")
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == "PINNED\n\nu1" and messages[-1]["content"] == "newest"
    assert assemble("", [], "only")[0]["content"] == "only"
    assert pinned_text({"a": "CARD a"}).endswith("\n\nCARD a") and pinned_text({"a": "CARD a", "b": "CARD b"}).endswith("CARD a\n\nCARD b")
    assert pinned_text({}) == ""


CODE = "```python effective.solve\n" + "\n".join(f"line {i}" for i in range(20)) + "\n```"


def test_the_newest_exchanges_are_sent_whole_and_older_ones_lose_long_results_and_code_but_keep_their_ids():
    long = "Attempt a1b2c3: it failed.\n" + "x" * 3000
    window = [Exchange(1, long, '{"action": "check"}\n' + CODE), Exchange(2, long, '{"action": "check"}\n' + CODE),
              Exchange(3, long, '{"action": "check"}\n' + CODE), Exchange(4, long, '{"action": "check"}\n' + CODE)]
    out = condense(window, whole=2, kept={1: [("attempt", "a1b2c3")], 2: [("attempt", "d4e5f6"), ("entry", "9a8b7c")]})
    assert out[2:] == window[2:]                                              # the newest two are exactly as they were
    for old in out[:2]:
        assert len(old.user) < OLD_RESULT + 150 and old.user.startswith("Attempt a1b2c3: it failed.") and "look at its id again" in old.user
        assert old.assistant.startswith('{"action": "check"}') and "line 7" not in old.assistant
    assert "[python effective.solve: 20 lines of source, kept as attempt a1b2c3; look at it or build on it with base]" in out[0].assistant
    assert "kept as attempt d4e5f6, entry 9a8b7c" in out[1].assistant
    assert "not kept" in condense(window, whole=2)[0].assistant              # a turn whose source was never run has no id to name
    assert window[0].user == long and "line 7" in window[0].assistant         # the originals are untouched: the window is built from them again


def test_short_results_short_code_and_non_source_blocks_are_not_condensed():
    short = Exchange(1, "Passed.", '```json\n{"action": "check"}\n```\n```python s\nx = 1\n```')
    assert condense([short, Exchange(2, "u", "a")], whole=1) == [short, Exchange(2, "u", "a")]
    assert condense([short], whole=0)[0] == short and condense([], whole=2) == []
    many = [Exchange(i, "r" * 1000, "a") for i in range(1, 4)]
    assert condense(many, whole=99) == many                                   # a large number sends everything whole


# ---- the notebook -----------------------------------------------------------------------------------------------------------

def attempt(i, turn, ok=True, note="", action="check"):
    reasons = [] if ok else [dict(stage="s", tier="cheap", reason="residual too big")]
    return dict(id=f"a{i:07d}", turn=turn, action=action, ok=ok, note=note, reasons=reasons,
                graph=dict(stages={"s": dict(source="x\ny\n", entry="Runtime", impl="s/agent-1")}))


def test_the_notebook_has_one_line_for_each_of_the_contestants_own_turns_before_the_window():
    runs = [dict(turn=2, entry_ids=["e1", "e2"], speedup=3.5, klass="exact", note="banded", graph=attempt(0, 2)["graph"])]
    text = notebook_text(attempts=[attempt(1, 1, note="first try"), attempt(2, 3, ok=False), attempt(3, 9)], runs=runs,
                         reads=[(1, "card s"), (8, "entry zz")], best=dict(id="e2", speedup=3.43, klass="exact"), before=5)
    assert "a0000001 turn 1 check" in text and "note: first try" in text and "FAILED at the cheap tier" in text
    assert "e1/e2 turn 2 timed" in text and "3.5x (exact); note: banded" in text
    assert "a0000003" not in text and "entry zz" not in text and "Already read: card s" in text      # turn 9 and 8 are in the window
    assert "Your best timed entry so far: e2, 3.43x" in text and "build on one with base" in text
    assert notebook_text(attempts=[attempt(1, 1)], runs=[], reads=[], best=None, before=None) == ""
    assert notebook_text(attempts=[attempt(1, 6)], runs=[], reads=[], best=None, before=5) == ""


def test_a_long_notebook_keeps_the_newest_lines_whole_and_lists_the_older_ones_by_id():
    many = [attempt(i, i) for i in range(1, 40)]
    text = notebook_text(attempts=many, runs=[], reads=[], best=None, before=100)
    assert "older, ids only: a0000001 (turn 1)" in text and "a0000015 (turn 15)" in text
    assert "- a0000039 turn 39 check" in text and "- a0000016 turn 16 check" in text and "- a0000015 turn 15 check" not in text


# ---- the board's delta -------------------------------------------------------------------------------------------------------------

def group(*rows):
    return dict(klass="exact", device="cpu", objectives=[dict(name="speedup", sense="max")], stages=[],
                rows=[dict(id=i, label=label, objectives=dict(speedup=s), rank=1, graph={}) for i, label, s in rows])


def test_the_delta_shows_only_new_rows_with_where_they_stand_and_marks_everything_seen():
    seen = {"e1"}
    text = render_board_delta([group(("e2", "beta", 5.0), ("e1", "alpha", 3.0), ("e3", "gamma", 1.5))], seen, you="alpha")
    assert "beta: speedup 5x" in text and "now 1 of 3 in exact on cpu" in text and "gamma" in text and "alpha" not in text
    assert seen == {"e1", "e2", "e3"} and render_board_delta([group(("e2", "beta", 5.0))], seen) == ""


# ---- the measured facts, tidied ----------------------------------------------------------------------------------------------

def test_tidy_shortens_a_librarys_internal_names_and_leaves_numpys_output_alone():
    from phost_agent.facts import facts_block, tidy
    warp = "  system.h: warp._src.types.array dtype=<class 'warp._src.types.float32'> shape=(2, 464, 464) strides=(861184, 1856, 4)"
    assert tidy(warp) == "  system.h: warp.array dtype=float32 shape=(2, 464, 464) strides=(861184, 1856, 4)"
    numpy = "  system.h: numpy.ndarray dtype=float32 shape=(2, 32, 32) strides=(4096, 128, 4) c_contiguous=True writeable=True"
    assert tidy(numpy) == numpy and tidy(tidy(warp)) == tidy(warp)
    assert "warp.array dtype=float32" in facts_block(warp) and "_src" not in facts_block(warp)
