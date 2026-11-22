"""The provider layer: the registry, the OpenAI-compatible client against a real local HTTP server, the program client, and the
Claude client against a stand-in for the SDK. No test here reaches a real service."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from phost import ContractError
from phost_agent.llm import (AnthropicClient, CommandClient, LLMError, LLMFatal, OpenAICompatClient, load_plugin, make_client,
                             parse_options, providers, redact, register_provider)


class Server:
    """A local server that answers from a script and records what it was sent."""

    def __init__(self):
        self.script, self.requests = [], []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(dict(path=self.path, headers=dict(self.headers), body=body))
                status, payload, headers = owner.script.pop(0) if owner.script else (200, completion("ok"), {})
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def completion(text, finish="stop", usage=True, cached=0):
    body = dict(model="served-model", choices=[dict(message=dict(role="assistant", content=text), finish_reason=finish)])
    if usage:
        body["usage"] = dict(prompt_tokens=120, completion_tokens=30, prompt_tokens_details=dict(cached_tokens=cached))
    return body


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


# ---- the registry -----------------------------------------------------------------------------------------------------------

def test_the_shipped_providers_are_listed_and_an_unknown_one_is_refused():
    assert {p.name for p in providers()} >= {"anthropic", "openai", "ollama", "compat", "cmd"}
    with pytest.raises(ContractError, match="no LLM provider 'nope'"):
        make_client("nope:x")


def test_options_are_key_value_pairs():
    assert parse_options(["effort=low", "body.temperature=0.2"]) == {"effort": "low", "body.temperature": "0.2"}
    with pytest.raises(ContractError, match="KEY=VALUE"):
        parse_options(["effort"])


def test_a_plugin_adds_a_provider_and_a_name_is_taken_once(tmp_path, monkeypatch):
    (tmp_path / "my_llm_plugin.py").write_text(
        "from phost_agent.llm import Reply\n"
        "class Echo:\n    name = 'echo'\n    model = 'echo-1'\n"
        "    def complete(self, system, messages, *, max_tokens):\n        return Reply('{\"stop\": \"echo\"}', 1, 1)\n"
        "def register_llm(add):\n    add('echo-test', lambda model, options: Echo(), doc='an echo')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    load_plugin("my_llm_plugin")
    client = make_client("echo-test")
    assert client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5).text == '{"stop": "echo"}'
    with pytest.raises(ContractError, match="already registered"):
        register_provider("echo-test", lambda model, options: None)
    (tmp_path / "no_hook.py").write_text("x = 1\n")
    with pytest.raises(ContractError, match="no register_llm"):
        load_plugin("no_hook")


# ---- the OpenAI-compatible client ---------------------------------------------------------------------------------------------

def test_a_request_is_a_chat_completion_with_the_system_prompt_first(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", server.url)
    client = make_client("openai:some-model", {"body.temperature": "0.2"})
    server.script.append((200, completion("hello", cached=100), {}))
    reply = client.complete("SYSTEM", [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
                                       {"role": "user", "content": "three"}], max_tokens=500)
    sent = server.requests[0]
    assert sent["path"] == "/v1/chat/completions" and sent["headers"]["Authorization"] == "Bearer sk-test-key"
    assert sent["body"]["model"] == "some-model" and sent["body"]["max_completion_tokens"] == 500 and "max_tokens" not in sent["body"]
    assert sent["body"]["temperature"] == 0.2
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["body"]["messages"][0]["content"] == "SYSTEM"
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.cached_tokens, reply.stop) == ("hello", 120, 30, 100, "end")
    assert reply.model == "served-model" and not reply.estimated and reply.tokens == 150


def test_other_servers_take_max_tokens_and_no_key_unless_one_is_named(server):
    client = make_client("compat:m", {"base_url": server.url})
    client.complete("s", [{"role": "user", "content": "u"}], max_tokens=64)
    sent = server.requests[0]
    assert sent["body"]["max_tokens"] == 64 and "Authorization" not in sent["headers"]
    local = make_client("ollama:llama", {"base_url": server.url, "max_tokens_field": "max_tokens"})
    assert local.name == "ollama" and local.model == "llama"


def test_a_reply_cut_off_at_the_limit_says_so_and_a_missing_usage_is_estimated(server):
    client = OpenAICompatClient("m", base_url=server.url)
    server.script.append((200, completion("cut", finish="length", usage=False), {}))
    reply = client.complete("s" * 400, [{"role": "user", "content": "u"}], max_tokens=5)
    assert reply.stop == "length" and reply.estimated and reply.input_tokens >= 100


def test_a_refused_credential_or_model_ends_the_search_and_is_not_retried(server):
    client = OpenAICompatClient("m", base_url=server.url, sleep=lambda s: None)
    for status in (400, 401, 403, 404):
        server.script.append((status, {"error": {"message": "nope, secret-key-abc is wrong"}}, {}))
        with pytest.raises(LLMFatal, match=f"HTTP {status}"):
            client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)
    assert len(server.requests) == 4


def test_a_rate_limit_is_retried_after_the_wait_it_asks_for_and_a_dead_server_is_an_ordinary_failure(server):
    waits = []
    client = OpenAICompatClient("m", base_url=server.url, sleep=waits.append)
    server.script += [(429, {"error": {"message": "slow down"}}, {"retry-after": "3"}), (200, completion("ok"), {})]
    assert client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5).text == "ok" and waits == [3.0]
    server.script += [(500, {"error": "boom"}, {})] * 3
    with pytest.raises(LLMError, match="HTTP 500") as caught:
        client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)
    assert not isinstance(caught.value, LLMFatal)
    down = OpenAICompatClient("m", base_url="http://127.0.0.1:9/v1", retries=0, sleep=lambda s: None)
    with pytest.raises(LLMError, match="unreachable"):
        down.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)


def test_a_key_is_never_sent_in_the_clear_to_another_machine_and_must_come_from_the_environment(monkeypatch):
    with pytest.raises(LLMFatal, match="plain http"):
        OpenAICompatClient("m", base_url="http://example.com/v1", api_key="k")
    OpenAICompatClient("m", base_url="http://localhost:1/v1", api_key="k")              # this machine: fine
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMFatal, match="OPENAI_API_KEY"):
        make_client("openai:m")
    with pytest.raises(ContractError, match="needs a model"):
        monkeypatch.setenv("OPENAI_API_KEY", "k") or make_client("openai")
    with pytest.raises(ContractError, match="unknown option"):
        make_client("compat:m", {"base_url": "http://localhost:1/v1", "efort": "low"})
    with pytest.raises(ContractError, match="needs --llm-option base_url"):
        monkeypatch.delenv("LLM_BASE_URL", raising=False) or make_client("compat:m")


# ---- a program as the model ------------------------------------------------------------------------------------------------------

def test_a_program_gets_the_prompt_on_stdin_and_its_stdout_is_the_answer():
    code = "import sys; text = sys.stdin.read(); print('SEEN', '[user]' in text, 'SYS' in text)"
    client = make_client(f"cmd:{sys.executable} -c \"{code}\"")
    reply = client.complete("SYS", [{"role": "user", "content": "hello"}], max_tokens=10)
    assert reply.text.strip() == "SEEN True True" and reply.estimated and reply.input_tokens >= 1


def test_a_program_that_fails_is_a_failure_and_one_that_is_missing_ends_the_search():
    with pytest.raises(LLMError, match="exited with code 3: oops"):
        CommandClient([sys.executable, "-c", "import sys; sys.stderr.write('oops'); sys.exit(3)"]).complete("s", [], max_tokens=1)
    with pytest.raises(LLMFatal, match="could not start"):
        CommandClient(["/no/such/program"]).complete("s", [], max_tokens=1)
    with pytest.raises(LLMError, match="did not answer"):
        CommandClient([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2).complete("s", [], max_tokens=1)


# ---- Claude, through a stand-in for the SDK -----------------------------------------------------------------------------------------

class FakeMessages:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    def create(self, **request):
        self.calls.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def sdk_response(text="{}", stop="end_turn", **usage):
    block = SimpleNamespace(type="text", text=text)
    thinking = SimpleNamespace(type="thinking", thinking="")
    use = SimpleNamespace(input_tokens=usage.get("input", 50), output_tokens=usage.get("output", 20),
                          cache_read_input_tokens=usage.get("read", 0), cache_creation_input_tokens=usage.get("written", 0))
    return SimpleNamespace(content=[thinking, block], stop_reason=stop, usage=use, model="claude-opus-5-5")


def claude(outcome, **options):
    messages = FakeMessages(outcome)
    return AnthropicClient(client=SimpleNamespace(messages=messages), **options), messages


def test_claude_is_asked_with_a_cached_system_block_and_no_sampling_parameters():
    client, messages = claude(sdk_response("answer", read=900, written=100, input=40))
    reply = client.complete("SYSTEM", [{"role": "user", "content": "u"}], max_tokens=777)
    sent = messages.calls[0]
    assert sent["model"] == "claude-opus-5-5" and sent["max_tokens"] == 777
    assert sent["system"] == [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}]
    assert sent["messages"] == [{"role": "user", "content": "u"}]
    assert not ({"temperature", "top_p", "top_k", "thinking", "output_config"} & sent.keys())
    assert reply.text == "answer" and reply.input_tokens == 1040 and reply.cached_tokens == 900 and reply.output_tokens == 20
    assert reply.stop == "end" and reply.model == "claude-opus-5-5"


def test_claude_takes_another_model_an_effort_and_no_cache_on_request():
    client, messages = claude(sdk_response(), effort="low", cache=False)
    client = AnthropicClient("claude-sonnet-5-5", effort="low", cache=False, client=SimpleNamespace(messages=messages))
    client.complete("S", [{"role": "user", "content": "u"}], max_tokens=10)
    sent = messages.calls[0]
    assert sent["model"] == "claude-sonnet-5-5" and sent["output_config"] == {"effort": "low"}
    assert sent["system"] == [{"type": "text", "text": "S"}]
    with pytest.raises(ContractError, match="effort must be one of"):
        AnthropicClient(effort="huge", client=object())
    with pytest.raises(ContractError, match="unknown option"):
        make_client("anthropic", {"temp": "0"})


@pytest.mark.parametrize("stop,expected", [("max_tokens", "length"), ("refusal", "refusal"), ("tool_use", "other")])
def test_claude_stop_reasons_are_translated(stop, expected):
    client, _ = claude(sdk_response(stop=stop))
    assert client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5).stop == expected


def named(name, message="m", status=None):
    error = type(name, (Exception,), {})(message)
    error.message = message
    return error


def test_claude_failures_are_sorted_into_ones_that_repeat_and_ones_that_may_not():
    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"):
        client, _ = claude(named(name, "bad"))
        with pytest.raises(LLMFatal, match=name):
            client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)
    # the SDK's classes subclass these names; a stand-in with the right base name is classified the same way
    rate = type("RateLimitError", (type("APIStatusError", (Exception,), {}),), {})("slow")
    client, _ = claude(rate)
    with pytest.raises(LLMError, match="RateLimitError") as caught:
        client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)
    assert not isinstance(caught.value, LLMFatal)
    client, _ = claude(TypeError('Could not resolve authentication method. Expected either api_key or auth_token to be set.'))
    with pytest.raises(LLMFatal, match="ANTHROPIC_API_KEY"):
        client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)
    client, _ = claude(KeyError("a real bug"))                 # not the provider's business: it is not swallowed
    with pytest.raises(KeyError):
        client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)


def test_claude_without_the_sdk_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMFatal, match="pip install anthropic"):
        AnthropicClient()


def test_the_real_sdk_accepts_the_request_this_client_builds():
    anthropic = pytest.importorskip("anthropic")
    import inspect
    parameters = inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters
    assert {"model", "max_tokens", "system", "messages", "output_config"} <= set(parameters)
    assert issubclass(anthropic.RateLimitError, anthropic.APIStatusError)
    assert "AuthenticationError" in {c.__name__ for c in anthropic.AuthenticationError.__mro__}


def test_the_real_sdk_reports_an_unreachable_server_as_an_ordinary_failure(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")               # nothing listens here: no request leaves this machine
    client = AnthropicClient(client=anthropic.Anthropic(api_key="not-a-key", base_url="http://127.0.0.1:9", max_retries=0, timeout=3))
    with pytest.raises(LLMError, match="APIConnectionError") as caught:
        client.complete("s", [{"role": "user", "content": "u"}], max_tokens=5)
    assert not isinstance(caught.value, LLMFatal)


# ---- nothing about the machine or the person leaves it --------------------------------------------------------------------------------

def test_redact_hides_the_home_directory_the_state_directory_and_the_user_name_but_not_longer_words(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", "/home/zebedee_q")
    monkeypatch.setenv("LOGNAME", "zebedee")
    monkeypatch.setenv("PHOST_HOME", str(tmp_path / "state"))
    text = f"/home/zebedee_q/work and {tmp_path}/state/journal.jsonl by zebedee, not zebedeeish or xzebedee"
    out = redact(text)
    assert "/home/zebedee_q" not in out and str(tmp_path) not in out
    assert out.startswith("~/work and <state>/journal.jsonl by <user>,") and "zebedeeish" in out and "xzebedee" in out
