import json
import sys
import textwrap

import pytest

from phost_agent.cli import main
from phost_agent import Journal
from conftest import GOOD_SOLVER, SCENE, STAGE, WRONG_SOLVER

BASE = ["--scene", SCENE, "--stage", STAGE, "--worlds", "1,2", "--steps", "5", "--launches", "2", "--repeats", "3"]


@pytest.fixture
def cli(capsys, tmp_path):
    journal = str(tmp_path / "journal.jsonl")

    def call(*argv):
        code = main(["--journal", journal, *argv])
        out = capsys.readouterr()
        return code, out.out, out.err

    call.journal = journal
    return call


def test_a_card_is_markdown_or_json(cli):
    code, out, _ = cli("card", STAGE)
    assert code == 0 and out.startswith("# stage effective.solve")
    code, out, _ = cli("card", STAGE, "--json")
    assert code == 0 and json.loads(out)["name"] == STAGE


def test_a_brief_says_the_ceiling_only_if_one_was_measured(cli):
    _, before, _ = cli("brief", *BASE)
    assert before.startswith(f"# task ") and "Ceiling:" not in before and "## ports" in before
    cli("ceiling", *BASE)
    _, after, _ = cli("brief", *BASE)
    assert "Ceiling:" in after
    code, out, _ = cli("brief", *BASE, "--json")
    brief = json.loads(out)
    assert code == 0 and brief["ceiling"]["ok"] and brief["budget"]["workers"] == 1


def test_the_ceiling_is_measured_once_and_then_read_from_the_journal(cli):
    code, first, _ = cli("ceiling", *BASE)
    code2, second, _ = cli("ceiling", *BASE)
    assert code == code2 == 0 and "[measured" in first and "[memo]" in second
    code3, third, _ = cli("ceiling", *BASE, "--refresh")
    assert "[measured" in third


def test_a_stage_with_no_speed_to_gain_exits_2_with_the_reason(cli):
    code, out, _ = cli("ceiling", "--scene", SCENE, "--stage", "dof.passive_force")
    assert code == 2 and out.startswith("REFUSED") and "model_changing" in out


def write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return str(path)


def test_check_runs_one_proposal_through_the_cheap_tiers_and_exits_by_the_result(cli, tmp_path):
    good = write(tmp_path, "good.json", dict(stage=STAGE, source=GOOD_SOLVER))
    code, out, _ = cli("check", *BASE, good, "--allow-unsandboxed")
    assert code == 0 and "static  pass" in out and "cheap   pass" in out and "real    pass" in out
    bad = write(tmp_path, "bad.json", dict(stage=STAGE, source=WRONG_SOLVER))
    code, out, _ = cli("check", *BASE, bad, "--allow-unsandboxed")
    assert code == 2 and "cheap   REJECTED" in out and "max_relative_residual" in out and "real    not run" in out
    code, out, _ = cli("check", *BASE, bad, "--allow-unsandboxed")
    assert "(from the memo)" in out


def test_run_with_a_script_goes_the_whole_way_and_writes_the_report(cli, tmp_path):
    script = write(tmp_path, "script.json", [dict(stage=STAGE, source=WRONG_SOLVER), dict(stage=STAGE, source=GOOD_SOLVER)])
    out_path = tmp_path / "report.json"
    code, out, _ = cli("run", *BASE, "--proposer", f"script:{script}", "--allow-unsandboxed", "--out", str(out_path))
    assert code == 0 and "ceiling: pass" in out and "finished: the proposer has nothing more to try" in out
    assert "rejected at cheap" in out and "funnel: proposed 2 -> static 2 -> cheap 1 -> real 1 -> timed 1" in out
    report = json.loads(out_path.read_text())
    assert report["status"] == "finished" and report["funnel"]["proposed"] == 2 and len(report["outcomes"]) == 2
    code, status, _ = cli("status")
    assert code == 0 and "journal intact" in status and "proposed 2" in status


def test_run_with_a_command_proposer_that_gives_one_proposal_then_stops(cli, tmp_path):
    program = tmp_path / "agent.py"
    program.write_text(textwrap.dedent(f"""
        import json, sys
        request = json.load(sys.stdin)
        if request["history"]:
            print(json.dumps({{"stop": "done"}}))
        else:
            print(json.dumps({{"proposal": {{"stage": {STAGE!r}, "source": {GOOD_SOLVER!r}}}, "tokens": 777}}))
    """))
    code, out, _ = cli("run", *BASE, "--proposer", f"command:{sys.executable} {program}", "--allow-unsandboxed")
    assert code == 0 and "tokens 777" in out and "finished" in out


def test_run_with_the_coordinate_baseline_on_a_stage_with_no_implementations_says_nothing_to_try(cli):
    code, out, _ = cli("run", *BASE, "--proposer", "coordinate")
    assert code == 0 and "finished: the proposer has nothing more to try" in out


def test_status_json_and_a_damaged_journal_exit_3(cli):
    cli("ceiling", *BASE)
    code, out, _ = cli("status", "--json")
    assert code == 0 and json.loads(out)["intact"] and json.loads(out)["ceilings"] == 1
    path = cli.journal
    lines = open(path).read().splitlines()
    record = json.loads(lines[0])
    record["entry"]["seconds"] = 99
    open(path, "w").write(json.dumps(record) + "\n")
    code, out, _ = cli("status")
    assert code == 3 and "DAMAGED" in out


def test_the_lease_command_says_who_holds_a_device(cli, tmp_path):
    from phost_agent import Lease
    assert cli("lease", "--device", "cpu")[1].strip() == "cpu: free"
    with Lease("cpu").hold("tier 4 of a search"):
        code, out, _ = cli("lease", "--device", "cpu")
        assert "tier 4 of a search" in out
        assert json.loads(cli("lease", "--device", "cpu", "--json")[1])["holder"] == "tier 4 of a search"


@pytest.mark.parametrize("argv,message", [
    (["card", "no.such.stage"], "no contract"),
    (["ceiling", "--scene", "nowhere", "--stage", STAGE], "no scene"),
    (["run", *BASE, "--proposer", "bogus"], "--proposer is"),
    (["run", *BASE, "--workers", "3"], "cpu_seconds"),
])
def test_bad_input_is_a_one_line_error_and_exit_1(cli, argv, message):
    code, _, err = cli(*argv)
    assert code == 1 and message in err and err.startswith("phost-agent:")
