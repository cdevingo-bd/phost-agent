"""Claude through the Claude Code on this machine: your own login, no API key.

    --contestant opus=claude-code:opus         the CLI's own alias for the latest Opus (or a full model id)
    --contestant me=claude-code                whatever model your Claude Code is set to

Each call runs ``claude -p`` (print mode) as a plain text model: no tools (it cannot read or write a file or run a command), no
MCP servers, no skills, no session saved, a replacement system prompt (ours, not the coding agent's) given with --system-prompt, from an empty directory (so
no project, CLAUDE.md or memory of yours is loaded). ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN`` are removed from its
environment so it uses your login and not a key that would bill the API; ``--llm-option keep_api_key=true`` keeps them. It is the
same Claude as the API, on whatever your login covers, and it counts against that plan's usage. Not ``--bare``: that mode ignores
your login and wants a key.

Options: ``binary`` (the executable, default ``claude``), ``timeout`` (seconds a call may take), ``keep_api_key``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

from phost import ContractError

from .base import LLMError, LLMFatal, Reply, estimate_tokens

_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ClaudeCodeClient:
    name = "claude-code"

    def __init__(self, model: str | None = None, *, binary: str = "claude", timeout: float = 600.0, keep_api_key: bool = False):
        self.model, self.binary, self.timeout, self.keep_api_key = model or "default", binary, timeout, keep_api_key
        self._request_model = model
        self._cwd = tempfile.mkdtemp(prefix="phost-agent-claude-")        # empty: nothing of the project is there to load

    def __del__(self):
        shutil.rmtree(getattr(self, "_cwd", ""), ignore_errors=True)

    def _argv(self, system: str) -> list[str]:
        argv = [self.binary, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
                "--no-session-persistence", "--system-prompt", system]
        if self._request_model:
            argv += ["--model", self._request_model]
        return argv

    def complete(self, system: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Reply:
        transcript = ("The conversation so far; reply as the assistant to the last user message.\n\n"
                      + "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages) + "\n")
        env = {k: v for k, v in os.environ.items() if self.keep_api_key or k not in _KEYS}
        if len(system.encode()) > 120_000:                    # one argument may be 128 KB on Linux
            raise LLMFatal("the system prompt is too long to pass to claude on the command line")
        try:
            done = subprocess.run(self._argv(system), input=transcript, capture_output=True, text=True,
                                  timeout=self.timeout, cwd=self._cwd, env=env)
        except subprocess.TimeoutExpired:
            raise LLMError(f"claude did not answer within {self.timeout:g} s") from None
        except FileNotFoundError:
            raise LLMFatal(f"{self.binary!r} was not found: install Claude Code, or point at it with --llm-option binary=/path") from None
        except OSError as error:
            raise LLMFatal(f"claude could not start: {error}") from None
        text, usage, failed = _read(done.stdout)
        if done.returncode != 0 or failed:
            stderr_tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else ""
            detail = (text.strip() or stderr_tail)[:300]
            message = f"claude -p failed (exit {done.returncode})" + (f": {detail}" if detail else "")
            if any(word in detail.lower() for word in ("log in", "login", "/login", "api key", "unauthorized", "not authenticated")):
                raise LLMFatal(message + " (run `claude` once and log in)")
            raise LLMError(message)
        if usage is None:
            return Reply(text, estimate_tokens(system + transcript), estimate_tokens(text), estimated=True)
        read, written = usage.get("cache_read_input_tokens", 0) or 0, usage.get("cache_creation_input_tokens", 0) or 0
        return Reply(text, int(usage.get("input_tokens", 0) or 0) + read + written, int(usage.get("output_tokens", 0) or 0), int(read))


def _read(stdout: str) -> tuple[str, Mapping | None, bool]:
    """``(text, usage or None, the CLI said it failed)`` from ``--output-format json``; plain text is taken as the answer."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, None, False
    if isinstance(data, list):                              # a stream of events: the last one that is a result
        data = next((d for d in reversed(data) if isinstance(d, dict) and d.get("type") == "result"), {})
    if not isinstance(data, dict) or not isinstance(data.get("result"), str):
        return stdout, None, False
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return data["result"], usage, bool(data.get("is_error"))


def _factory(model, options):
    options = dict(options)
    binary = options.pop("binary", "claude")
    timeout = float(options.pop("timeout", 600.0))
    keep = options.pop("keep_api_key", "false").lower() in ("1", "true", "yes", "on")
    if options:
        raise ContractError(f"unknown option(s) for the claude-code provider: {sorted(options)}; known: binary, timeout, keep_api_key")
    return ClaudeCodeClient(model, binary=binary, timeout=timeout, keep_api_key=keep)


def register(add) -> None:
    add("claude-code", _factory, doc="Claude through the Claude Code on this machine: your login, no API key (model: an alias like opus "
                                     "or sonnet, or an id)", needs="Claude Code installed and logged in")
