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
    """A model that enters one graph on its first turn and leaves on its second."""
    action = json.dumps({"action": "submit", "note": f"{name}: a dense float64 solve",
                         "graph": {"stages": {STAGE: {"source": GOOD_SOLVER, "entry": "Runtime"}}}})
    path = tmp_path / f"{name}.py"
    path.write_text(f"import sys\ntext = sys.stdin.read()\nprint({json.dumps({'action': 'done'})!r} if '[assistant]' in text else {action!r})\n")
    return f"{sys.executable} {path}"


def test_two_models_enter_and_the_board_and_the_ledger_show_both(cli, tmp_path):
    out_path = tmp_path / "report.json"
    code, out, err = cli("arena", *BASE, "--contestant", f"alpha=cmd:{contestant_program(tmp_path, 'alpha')}",
                         "--contestant", f"beta=cmd:{contestant_program(tmp_path, 'beta')}", "--turns", "4", "--submits", "2",
                         "--allow-unsandboxed", "--out", str(out_path))
    assert code == 0, err
    assert "[alpha] turn 1" in out and "[beta] turn 1" in out and "[alpha] leaves: done" in out and "[beta] leaves: done" in out
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
