import json

import pytest

from phost_agent import Journal, Memo, key_for
from phost_score import LedgerError


def test_a_journal_is_a_hash_chain_and_says_where_it_was_edited(journal):
    journal.append("task", phase="start", note="a")
    journal.append("tier", key="k", tier="static", ok=True)
    journal.append("attempt", note="c")
    assert journal.verify().ok and len(journal.records()) == 3
    lines = journal.path.read_text().splitlines()
    record = json.loads(lines[1])
    record["entry"]["ok"] = False
    lines[1] = json.dumps(record)
    journal.path.write_text("\n".join(lines) + "\n")
    problem = Journal(journal.path).verify().problems[0]
    assert "line 2" in problem and "changed" in problem
    with pytest.raises(LedgerError, match="damaged"):
        Journal(journal.path).records()


def test_only_known_kinds_are_recorded(journal):
    with pytest.raises(ValueError, match="no journal record kind"):
        journal.append("note", text="x")


def test_non_finite_numbers_are_stored_as_null_not_refused(journal):
    journal.append("tier", key="k", tier="cheap", ok=True, metrics={"residual": float("nan"), "other": 1.5})
    assert journal.records("tier")[0]["metrics"] == {"residual": None, "other": 1.5}


def test_records_are_read_again_when_the_file_changes(journal):
    assert journal.records() == []
    journal.append("attempt", n=1)
    assert len(journal.records()) == 1
    journal.append("attempt", n=2)
    assert [e["n"] for e in journal.records("attempt")] == [1, 2]


def test_a_key_is_the_content_not_the_order_or_the_name():
    a = key_for(dict(stage="s", impl="i", params={"x": 1, "y": 2}))
    assert a == key_for(dict(params={"y": 2, "x": 1}, impl="i", stage="s"))
    assert a != key_for(dict(stage="s", impl="i", params={"x": 1, "y": 3}))


def test_the_memo_returns_the_latest_result_of_each_tier(journal):
    memo = Memo(journal)
    memo.store("k1", "static", ok=True)
    memo.store("k1", "cheap", ok=False, reason="residual too big", seconds=0.5)
    memo.store("k2", "cheap", ok=True)
    memo.store("k1", "cheap", ok=True)                       # a later result replaces the earlier one
    found = memo.lookup("k1")
    assert set(found) == {"static", "cheap"} and found["cheap"]["ok"] is True
    assert memo.lookup("none") == {}


def test_the_memo_survives_a_new_process_because_it_is_the_journal(journal):
    Memo(journal).store("k", "run", ok=True, verdict="accepted", speedup=1.5)
    again = Memo(Journal(journal.path)).lookup("k")["run"]
    assert again["verdict"] == "accepted" and again["speedup"] == 1.5


def test_only_the_tiers_about_a_candidate_are_kept(journal):
    with pytest.raises(ValueError, match="no tier"):
        Memo(journal).store("k", "ceiling", ok=True)
