"""What every model provider is reduced to: text in, text and a token count out.

A provider is anything that can answer ``complete(system, messages, max_tokens=...)`` with a ``Reply``. The proposer knows
nothing else about it, so trying another model is a different ``--proposer llm:...`` argument and not different code. A
provider that needs more says so in its own options (``--llm-option``); the proposer never sees them.

``system`` is the part that does not change between turns of one search (the instructions and the brief), so a provider that
can cache a prefix has a prefix to cache. ``messages`` is the conversation so far, ``[{"role": "user" | "assistant",
"content": text}, ...]``, always ending on a user turn.
"""

from __future__ import annotations

import getpass
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..proposers import ProposerError, ProposerFatal


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int = 0            # everything sent, whether or not the provider served part of it from its cache
    output_tokens: int = 0
    cached_tokens: int = 0           # of input_tokens: served from the provider's cache (billed less, where it is)
    stop: str = "end"                # end | length (cut off at max_tokens) | refusal | other
    model: str = ""                  # what answered, when the provider says
    estimated: bool = False          # the counts are a guess (a provider that does not report usage)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ChatClient(Protocol):
    name: str                        # the provider: "anthropic", "openai", ...
    model: str

    def complete(self, system: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Reply:
        """One answer. Raise ``LLMError`` for a failure worth another try and ``LLMFatal`` for one that would repeat."""


class LLMError(ProposerError):
    """A call failed in a way that may not repeat (rate limit, server error, a dropped connection): the attempt is lost, the
    search goes on."""


class LLMFatal(LLMError, ProposerFatal):
    """A call failed in a way that will repeat (no key, a refused credential, an unknown model, a missing SDK): the search ends."""


def estimate_tokens(text: str) -> int:
    """About four characters a token. For a provider that does not say, and for ``phost-agent prompt``."""
    return max(1, len(text) // 4)


def redact(text: str) -> str:
    """Nothing about the machine or the person goes in a request: the home directory and the user name are replaced, and so is
    the state directory. The brief is generated from registries and holds none of them; this is for the day it does."""
    swaps: list[tuple[str, str]] = []
    state = os.environ.get("PHOST_HOME")
    if state:
        swaps.append((str(Path(state)), "<state>"))
    swaps.append((str(Path.home()), "~"))
    for text_, pattern in swaps:
        if text_ and text_ != "/":
            text = text.replace(text_, pattern)
    try:
        user = getpass.getuser()
    except Exception:                                  # no passwd entry, no environment: nothing to hide by name
        user = ""
    if len(user) >= 3:
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(user)}(?![A-Za-z0-9_])", "<user>", text)
    return text
