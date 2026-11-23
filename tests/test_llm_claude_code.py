"""The Claude Code provider against a stand-in ``claude`` executable that records how it was called. No real Claude is reached."""

import json
import os
import stat
import sys

import pytest

from phost_agent.llm import LLMError, LLMFatal, make_client

FAKE = '''#!{python}
import json, os, sys
argv = sys.argv[1:]
system = argv[argv.index("--system-prompt") + 1] if "--system-prompt" in argv else None
record = dict(argv=argv, cwd=os.getcwd(), listing=os.listdir("."), stdin=sys.stdin.read(), system=system,
              has_key="ANTHROPIC_API_KEY" in os.environ, has_token="ANTHROPIC_AUTH_TOKEN" in os.environ)
open({log!r}, "w").write(json.dumps(record))
mode = open({mode!r}).read().strip() if os.path.exists({mode!r}) else "json"
if mode == "json":
    print(json.dumps(dict(type="result", is_error=False, result='{{"action": "done"}}',
                          usage=dict(input_tokens=30, output_tokens=7, cache_read_input_tokens=900, cache_creation_input_tokens=100))))
elif mode == "text":
    print("plain words")
elif mode == "notlogged":
    print(json.dumps(dict(type="result", is_error=True, result="Invalid API key - Please run /login")))
    sys.exit(1)
elif mode == "crash":
    sys.stderr.write("boom\\n")
    sys.exit(2)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    log, mode = tmp_path / "log.json", tmp_path / "mode"
    path = tmp_path / "claude"
    path.write_text(FAKE.format(python=sys.executable, log=str(log), mode=str(mode)))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)

    class Fake:
        binary = str(path)

        def called(self):
            return json.loads(log.read_text())

        def mode(self, value):
            mode.write_text(value)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return Fake()


def client(fake, spec="claude-code:opus", **options):
    return make_client(spec, {"binary": fake.binary, **options})


def test_it_runs_claude_as_a_plain_text_model_with_no_tools_no_memory_and_no_saved_session(fake, tmp_path):
    reply = client(fake).complete("THE SYSTEM", [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"},
                                                {"role": "user", "content": "third"}], max_tokens=10)
    call = fake.called()
    argv = call["argv"]
    assert argv[0] == "-p" and argv[argv.index("--output-format") + 1] == "json" and argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert "--bare" not in argv                        # bare mode would ignore the login and want a key
    assert call["system"] == "THE SYSTEM"              # our system prompt replaces the coding agent's
    assert "[user]\nfirst" in call["stdin"] and "[assistant]\nsecond" in call["stdin"] and call["stdin"].rstrip().endswith("third")
    assert call["cwd"] != os.getcwd() and call["listing"] == []                              # an empty directory: nothing to load


def test_a_system_prompt_too_big_for_an_argument_is_refused_before_anything_runs(fake):
    with pytest.raises(LLMFatal, match="too long"):
        client(fake).complete("x" * 130_000, [{"role": "user", "content": "u"}], max_tokens=1)


def test_a_key_in_the_environment_is_not_passed_on_unless_asked(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-travel")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    client(fake).complete("s", [{"role": "user", "content": "u"}], max_tokens=1)
    assert fake.called()["has_key"] is False and fake.called()["has_token"] is False
    client(fake, keep_api_key="true").complete("s", [{"role": "user", "content": "u"}], max_tokens=1)
    assert fake.called()["has_key"] is True


def test_the_answer_and_the_real_token_counts_come_from_the_json_result(fake):
    reply = client(fake).complete("s", [{"role": "user", "content": "u"}], max_tokens=1)
    assert reply.text == '{"action": "done"}' and reply.input_tokens == 1030 and reply.output_tokens == 7
    assert reply.cached_tokens == 900 and not reply.estimated and reply.stop == "end"


def test_without_a_model_it_uses_whatever_the_login_is_set_to_and_plain_text_is_taken_as_the_answer(fake):
    fake.mode("text")
    c = client(fake, "claude-code")
    reply = c.complete("s", [{"role": "user", "content": "u"}], max_tokens=1)
    assert "--model" not in fake.called()["argv"] and c.model == "default"
    assert reply.text.strip() == "plain words" and reply.estimated


def test_not_being_logged_in_ends_the_search_and_any_other_failure_costs_a_turn(fake):
    fake.mode("notlogged")
    with pytest.raises(LLMFatal, match="log in"):
        client(fake).complete("s", [{"role": "user", "content": "u"}], max_tokens=1)
    fake.mode("crash")
    with pytest.raises(LLMError, match=r"exit 2\): boom") as caught:
        client(fake).complete("s", [{"role": "user", "content": "u"}], max_tokens=1)
    assert not isinstance(caught.value, LLMFatal)
    with pytest.raises(LLMFatal, match="was not found"):
        make_client("claude-code", {"binary": "/no/such/claude"}).complete("s", [{"role": "user", "content": "u"}], max_tokens=1)


def test_the_provider_is_listed_and_takes_only_its_own_options():
    from phost_agent.llm import providers
    assert "claude-code" in {p.name for p in providers()}
    with pytest.raises(Exception, match="unknown option"):
        make_client("claude-code", {"temperature": "0"})


def test_effort_is_passed_to_claude_when_set_and_left_out_when_not(fake):
    client(fake).complete("s", [{"role": "user", "content": "u"}], max_tokens=10)
    assert "--effort" not in fake.called()["argv"]
    client(fake, effort="medium").complete("s", [{"role": "user", "content": "u"}], max_tokens=10)
    argv = fake.called()["argv"]
    assert argv[argv.index("--effort") + 1] == "medium"
    with pytest.raises(Exception, match="effort is one of low, medium, high, xhigh, max, not 'huge'"):
        make_client("claude-code", {"effort": "huge"})
