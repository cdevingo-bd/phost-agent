"""Two contestants through the command line on a real scene: real checks, real timed runs, a real leaderboard. Programs stand in for the models."""

import json
import sys

import pytest

from phost_agent.cli import main
from phost_score import default_ledger, Ledger
from conftest import GOOD_SOLVER, SCENE, STAGE

BASE = ["--scene", SCENE, "--worlds", "1,2", "--steps", "5", "--launches", "2", "--repeats", "3"]


@pytest.fixture
def cli(capsys, tmp_path):
    def call(*argv):
        code = main(["--journal", str(tmp_path / "journal.jsonl"), *argv])
        out = capsys.readouterr()
        return code, out.out, out.err
    return call


def contestant_program(tmp_path, name):
    """A model that enters one graph on its first turn and reads the stage card on every turn after (it cannot leave)."""
    action = json.dumps({"action": "submit", "note": f"{name}: a dense float64 solve",
                         "graph": {"stages": {STAGE: {"source": GOOD_SOLVER, "entry": "Runtime"}}}})
    path = tmp_path / f"{name}.py"
    path.write_text(f"import sys\ntext = sys.stdin.read()\nprint({json.dumps({'action': 'look', 'what': 'card', 'stage': STAGE})!r} if '[assistant]' in text else {action!r})\n")
    return f"{sys.executable} {path}"


def test_two_models_enter_and_the_board_and_the_ledger_show_both(cli, tmp_path):
    out_path = tmp_path / "report.json"
    code, out, err = cli("arena", *BASE, "--contestant", f"alpha=cmd:{contestant_program(tmp_path, 'alpha')}",
                         "--contestant", f"beta=cmd:{contestant_program(tmp_path, 'beta')}", "--turns", "4", "--submits", "2",
                         "--allow-unsandboxed", "--out", str(out_path))
    assert code == 0, err
    assert "[alpha] turn 1" in out and "[beta] turn 1" in out and "[alpha] leaves: turns" in out and "[beta] leaves: turns" in out
    assert f"Board for {SCENE} at 2 worlds" in out
    report = json.loads(out_path.read_text())
    assert {c["name"] for c in report["contestants"]} == {"alpha", "beta"} and all(c["submits"] == 1 for c in report["contestants"])
    labels = {r.entry["label"] for r in Ledger(default_ledger()).records()}
    assert labels == {"alpha", "beta"}
    assert "where the host" in report["facts"].lower() or "not measured" in report["facts"].lower()


def test_a_contestant_name_defaults_to_its_provider_and_a_bad_provider_is_refused_up_front(cli, tmp_path):
    code, _, err = cli("arena", *BASE, "--contestant", "nope:x")
    assert code == 1 and "no LLM provider 'nope'" in err
    code, _, err = cli("arena", *BASE, "--contestant", "openai:some-model")
    assert code == 1 and "OPENAI_API_KEY" in err


def test_dash_serves_the_same_state_so_the_contestants_are_on_its_board_and_its_link_is_printed(cli, tmp_path, monkeypatch):
    """`arena --dash` is one command: the link comes up before the first turn, and what the contest ranks is what the dashboard shows."""
    pytest.importorskip("phost_dash")
    import urllib.request
    from phost_agent import cli as cli_module
    seen = {}

    def hold(dash):                                        # the contest is over; ask the dashboard over HTTP before it goes down
        board = urllib.request.Request(f"http://127.0.0.1:{dash.server_address[1]}/api/board?scene={SCENE}",
                                       headers={"X-Phost-Token": dash.guard.token})
        with urllib.request.urlopen(board, timeout=10) as reply:
            seen["status"], body = reply.status, json.loads(reply.read())
        seen["rows"] = [(r["label"], r["trust"]) for g in body["groups"] for r in g["rows"]]
        seen["url"] = dash.url
    monkeypatch.setattr(cli_module, "_hold", hold)
    code, out, err = cli("arena", *BASE, "--contestant", f"alpha=cmd:{contestant_program(tmp_path, 'alpha')}",
                         "--contestant", f"beta=cmd:{contestant_program(tmp_path, 'beta')}", "--turns", "3", "--submits", "2",
                         "--allow-unsandboxed", "--dash", "--no-browser")
    assert code == 0, err
    assert out.index("phost-dash is running") < out.index("[alpha] turn 1") and seen["url"] in out
    assert seen["status"] == 200 and set(seen["rows"]) == {("alpha", "self-reported"), ("beta", "self-reported")}
    assert "The contest is over. The dashboard stays up" in out
