import json
import sys
import textwrap

import pytest

import phost
from phost import ContractError, Impl, ImplKind, ParamSpec
from phost_agent import (ACCEPTED, REJECTED, CommandProposer, CoordinateProposer, Outcome, Proposal, ProposerError,
                         ScriptedProposer)
from phost_lab.bootstrap import load_registry

SPRING = "dof.passive_force/spring-damper"
STAGE = "dof.passive_force"


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def outcome(proposal, speedup=None, verdict=REJECTED):
    return Outcome(1, proposal, "k", "run" if speedup else "static", verdict, speedup=speedup)


def drive(proposer, registry, speedups, limit=40):
    """Ask for proposals until it is done, answering each with ``speedups(proposal)``."""
    history, given = [], []
    for _ in range(limit):
        proposal = proposer.propose({}, history)
        if proposal is None:
            return given
        given.append(proposal)
        s = speedups(proposal)
        history.append(outcome(proposal, s, ACCEPTED if s else REJECTED))
    return given


def test_a_script_is_replayed_in_order_and_then_runs_out():
    script = ScriptedProposer([dict(stage="s", impl="a"), Proposal("s", impl="b")])
    assert [script.propose({}, []).impl for _ in range(2)] == ["a", "b"]
    assert script.propose({}, []) is None


@pytest.fixture(scope="module")
def tuned():
    """A registry whose implementation of the stage has one parameter of each kind (never run: only proposed)."""
    reg = load_registry(registry=phost.Registry())
    reg.add_impl(Impl(id=f"{STAGE}/tunable", stage=STAGE, stage_version=1, version="1", title="tunable", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: None, reference=lambda inputs, params: {},
                      params=(ParamSpec("gain", "float", 2.0, min=0.5, max=3.0), ParamSpec("mode", "str", "a", choices=("a", "b", "c")),
                              ParamSpec("flag", "bool", False), ParamSpec("n", "int", 4, min=1, max=8),
                              ParamSpec("unused", "float", 0.0))))
    return reg


TUNABLE = f"{STAGE}/tunable"


def test_the_coordinate_baseline_tries_every_implementations_defaults_first(tuned):
    first = CoordinateProposer(tuned, STAGE).propose({}, [])
    assert first.impl in (SPRING, TUNABLE)
    seen = {p.impl for p in drive(CoordinateProposer(tuned, STAGE), tuned, lambda p: 1.0)[:2]}
    assert seen == {SPRING, TUNABLE}                                   # both defaults before any neighbour


def test_it_then_moves_one_parameter_at_a_time_around_the_best_point(tuned):
    proposer = CoordinateProposer(tuned, STAGE, impls=[TUNABLE])
    default = proposer.propose({}, [])
    assert default.params == dict(gain=2.0, mode="a", flag=False, n=4, unused=0.0)
    best = Proposal(STAGE, impl=TUNABLE, params={**default.params, "gain": 3.0})
    history = [outcome(default, 1.0, ACCEPTED), outcome(best, 3.0, ACCEPTED)]
    proposal = proposer.propose({}, history)
    assert proposal.params == {**best.params, "gain": 1.5}              # one step from the best point (3.0), not from the default


def test_it_never_repeats_itself_and_it_ends(tuned):
    proposer = CoordinateProposer(tuned, STAGE, impls=[TUNABLE])
    given = drive(proposer, tuned, lambda p: 1.0)
    keys = [proposer._id(tuned.impl(p.impl), p.params) for p in given]
    assert len(keys) == len(set(keys))
    assert len(given) == 1 + 2 + 2 + 1 + 2          # the defaults, then gain (2), mode (2), flag (1), n (2); unused is at zero


def test_a_parameter_at_zero_is_not_scaled(tuned):
    given = drive(CoordinateProposer(tuned, STAGE, impls=[TUNABLE]), tuned, lambda p: 1.0)
    assert {p.params["unused"] for p in given} == {0.0}


def test_values_stay_inside_the_parameters_declared_range_and_type(tuned):
    specs = {p.name: p for p in tuned.impl(TUNABLE).params}
    given = drive(CoordinateProposer(tuned, STAGE, impls=[TUNABLE]), tuned, lambda p: 1.0)
    assert any(p.params["n"] != 4 for p in given) and any(p.params["gain"] != 2.0 for p in given)
    for p in given:
        assert isinstance(p.params["n"], int) and specs["gain"].min <= p.params["gain"] <= specs["gain"].max
        assert specs["n"].min <= p.params["n"] <= specs["n"].max and p.params["mode"] in ("a", "b", "c")


def test_every_choice_and_the_flag_get_tried(tuned):
    given = drive(CoordinateProposer(tuned, STAGE, impls=[TUNABLE]), tuned, lambda p: 1.0)
    assert {p.params["mode"] for p in given} == {"a", "b", "c"} and {p.params["flag"] for p in given} == {False, True}


def test_the_baseline_can_be_restricted_and_refuses_an_implementation_that_is_not_there(tuned):
    assert CoordinateProposer(tuned, STAGE, impls=[SPRING]).propose({}, []).impl == SPRING
    with pytest.raises(ContractError, match="no such implementation"):
        CoordinateProposer(tuned, STAGE, impls=["dof.passive_force/nope"])


def test_the_native_implementation_is_never_proposed(tuned):
    assert all(p.impl != f"{STAGE}/native" for p in drive(CoordinateProposer(tuned, STAGE), tuned, lambda p: 1.0))


# ---- an external command -----------------------------------------------------------------------------------

def script(tmp_path, body):
    path = tmp_path / "proposer.py"
    path.write_text(textwrap.dedent(body))
    return [sys.executable, str(path)]


def test_a_command_gets_the_brief_and_history_and_answers_with_a_proposal(tmp_path):
    argv = script(tmp_path, """
        import json, sys
        request = json.load(sys.stdin)
        n = len(request["history"])
        print(json.dumps({"proposal": {"stage": request["brief"]["stage"], "impl": "x/y", "params": {"n": n}}, "tokens": 321}))
    """)
    proposer = CommandProposer(argv)
    first = proposer.propose({"stage": "s"}, [])
    second = proposer.propose({"stage": "s"}, [outcome(first)])
    assert (first.stage, first.params, first.tokens) == ("s", {"n": 0}, 321) and second.params == {"n": 1}


def test_a_command_can_say_it_is_done(tmp_path):
    assert CommandProposer(script(tmp_path, 'print(\'{"stop": "out of ideas"}\')')).propose({}, []) is None


@pytest.mark.parametrize("body,message", [
    ("import sys; sys.exit(3)", "exited with code 3"),
    ("raise RuntimeError('boom')", "RuntimeError: boom"),
    ("print('not json')", "did not print JSON"),
    ("print('[1]')", "must print a JSON object"),
    ("print('{}')", "neither"),
    ("print('{\"proposal\": {\"stage\": \"s\"}}')", "malformed proposal"),
])
def test_a_command_that_misbehaves_is_a_proposer_error_not_a_crash(tmp_path, body, message):
    with pytest.raises(ProposerError, match=message):
        CommandProposer(script(tmp_path, body)).propose({}, [])


def test_a_command_that_does_not_exist_or_is_too_slow_is_a_proposer_error(tmp_path):
    with pytest.raises(ProposerError, match="could not start"):
        CommandProposer(["/no/such/program"]).propose({}, [])
    with pytest.raises(ProposerError, match="did not answer"):
        CommandProposer(script(tmp_path, "import time; time.sleep(30)"), timeout=0.5).propose({}, [])


def test_the_history_a_command_sees_is_json(tmp_path):
    p = Proposal("s", impl="a/b")
    argv = script(tmp_path, """
        import json, sys
        h = json.load(sys.stdin)["history"]
        print(json.dumps({"stop": h[0]["verdict"]}) if h[0]["speedup"] == 2.5 else "x")
    """)
    assert CommandProposer(argv).propose({}, [outcome(p, 2.5, ACCEPTED)]) is None
