"""Any program as the model: it gets the whole prompt as text on stdin and prints its answer on stdout.

    --proposer llm:cmd:"my-model-cli --flag"

That reaches a model this package has no adapter for (a vendor's command-line tool, a script around a local runtime), and keeps its
credentials and traffic that program's own. The prompt is the system text, then each turn labelled ``[user]`` or ``[assistant]``.
A program does not report tokens, so they are estimated (about four characters each) and marked as such.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Mapping, Sequence

from phost import ContractError

from .base import LLMError, LLMFatal, Reply, estimate_tokens


class CommandClient:
    name = "cmd"

    def __init__(self, program: str | Sequence[str], *, timeout: float = 600.0):
        self.argv = shlex.split(program) if isinstance(program, str) else list(program)
        if not self.argv:
            raise ContractError("the cmd provider needs a program: --proposer llm:cmd:<program and arguments>")
        self.model, self.timeout = self.argv[0], timeout

    def complete(self, system: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Reply:
        prompt = system + "".join(f"\n\n[{m['role']}]\n{m['content']}" for m in messages) + "\n"
        try:
            done = subprocess.run(self.argv, input=prompt, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise LLMError(f"{self.argv[0]} did not answer within {self.timeout:g} s") from None
        except OSError as error:
            raise LLMFatal(f"{self.argv[0]} could not start: {error}") from None
        if done.returncode != 0:
            tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else ""
            raise LLMError(f"{self.argv[0]} exited with code {done.returncode}" + (f": {tail}" if tail else ""))
        return Reply(done.stdout, estimate_tokens(prompt), estimate_tokens(done.stdout), estimated=True)


def _factory(program, options):
    options = dict(options)
    timeout = float(options.pop("timeout", 600.0))
    if options:
        raise ContractError(f"unknown option(s) for the cmd provider: {sorted(options)}; known: timeout")
    return CommandClient(program or "", timeout=timeout)


def register(add) -> None:
    add("cmd", _factory, doc="any program: prompt on stdin, answer on stdout; tokens are estimated", needs="the program")
