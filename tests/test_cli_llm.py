"""The command line for a model-driven search, with a program standing in for the model (no service is reached)."""

import json
import sys

import pytest

from phost_agent.cli import main
from conftest import GOOD_SOLVER, SCENE, STAGE

BASE = ["--scene", SCENE, "--stage", STAGE, "--worlds", "1,2", "--steps", "5", "--launches", "2", "--repeats", "3"]


@pytest.fixture
def cli(capsys, tmp_path):
    def call(*argv):
        code = main(["--journal", str(tmp_path / "journal.jsonl"), *argv])
        out = capsys.readouterr()
        return code, out.out, out.err
    return call


def test_llms_lists_every_provider_and_what_each_needs(cli):
    code, out, _ = cli("llms")
    assert code == 0
    for name in ("anthropic", "openai", "ollama", "compat", "cmd"):
        assert name in out
    assert "ANTHROPIC_API_KEY" in out and "OPENAI_API_KEY" in out and "--proposer llm:" in out


def test_prompt_shows_what_a_model_would_be_sent_first_and_sends_nothing(cli):
    code, out, _ = cli("prompt", *BASE)
    assert code == 0 and "=== system" in out and "=== first turn ===" in out and "Attempt 1 of 20." in out
    assert "ONE candidate" in out and "## implementations" in out and "tokens in the first call" in out


def test_a_search_driven_by_a_model_goes_the_whole_way(cli, tmp_path):
    model = tmp_path / "model.py"
    stop = json.dumps({"stop": "one correct solver is enough"})
    proposal = json.dumps({"proposal": {"stage": STAGE, "entry": "Runtime", "note": "a dense float64 solve"}})
    model.write_text(f"import sys\nprompt = sys.stdin.read()\nif 'Results so far' in prompt:\n    print({stop!r})\nelse:\n"
                     f"    print({proposal!r})\n    print('```python')\n    print({GOOD_SOLVER!r})\n    print('```')\n")
    out_path = tmp_path / "report.json"
    code, out, err = cli("run", *BASE, "--proposer", f"llm:cmd:{sys.executable} {model}", "--allow-unsandboxed", "--out", str(out_path))
    assert code == 0, err
    assert "finished: the proposer has nothing more to try: one correct solver is enough" in out
    assert "funnel: proposed 1 -> static 1 -> cheap 1 -> real 1 -> timed 1" in out
    assert "llm:cmd:" in out and "2 calls" in out and "(estimated: this provider does not report usage)" in out
    report = json.loads(out_path.read_text())
    assert report["funnel"]["proposed"] == 1 and report["outcomes"][0]["proposal"]["note"] == "a dense float64 solve"
    assert report["spend"]["tokens"] > 0
    code, status, _ = cli("status")
    assert code == 0 and "journal intact" in status


def test_a_provider_that_cannot_work_is_refused_before_the_search_begins(cli, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code, _, err = cli("run", *BASE, "--proposer", "llm:openai:some-model")
    assert code == 1 and "OPENAI_API_KEY" in err
    code, _, err = cli("run", *BASE, "--proposer", "llm:nope:x")
    assert code == 1 and "no LLM provider 'nope'" in err
    code, _, err = cli("run", *BASE, "--proposer", "llm:ollama:m", "--llm-option", "bogus=1")
    assert code == 1 and "unknown option" in err
    code, _, err = cli("run", *BASE, "--proposer", "llm")
    assert code == 1 and "llm:PROVIDER" in err
