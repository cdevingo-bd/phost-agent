"""``LLMProposer``: a language model, behind any provider, asked for the next candidate each turn.

    proposer = LLMProposer(make_client("anthropic"))                  # or "openai:<model>", "ollama:<model>", "cmd:<program>"
    run_loop(task, proposer, budget=Budget(attempts=12, tokens=200_000))

Each call sends the system prompt (stable: instructions and brief) and one user turn (the results so far). An unusable answer is
sent back once with the reason, and the two calls count as one attempt; a second failure costs the attempt. What a proposal cost is
every token sent and received, cached or not, so a token budget bounds the traffic and not the bill (``usage`` has the split). A
failure that would repeat (no key, an unknown model) ends the search; one that may not (a rate limit, a dropped connection, a
reply cut off, a refusal) costs the attempt and the loop goes on.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from phost import ContractError

from ..outcome import Outcome
from ..proposal import Proposal
from ..proposers import ProposerError
from .base import ChatClient, LLMError, redact
from .prompt import parse_reply, system_prompt, turn_prompt


class LLMProposer:
    def __init__(self, client: ChatClient, *, max_tokens: int = 16000, history_limit: int = 40, source_chars: int = 6000,
                 repair: int = 1):
        if max_tokens < 1 or history_limit < 1 or repair < 0:
            raise ValueError("max_tokens and history_limit must be positive, repair not negative")
        self.client, self.max_tokens, self.history_limit = client, max_tokens, history_limit
        self.source_chars, self.repair = source_chars, repair
        self.name = f"llm:{client.name}:{client.model}"
        self.stop_reason = ""
        self.usage: dict[str, Any] = dict(calls=0, input_tokens=0, output_tokens=0, cached_tokens=0, estimated=False)
        self._system: tuple[int, str] | None = None

    def _system_text(self, brief: Mapping[str, Any]) -> str:
        if self._system is None or self._system[0] != id(brief):          # one brief per search: built once, byte-identical after
            self._system = (id(brief), redact(system_prompt(brief)))
        return self._system[1]

    def _count(self, reply) -> None:
        self.usage["calls"] += 1
        self.usage["input_tokens"] += reply.input_tokens
        self.usage["output_tokens"] += reply.output_tokens
        self.usage["cached_tokens"] += reply.cached_tokens
        self.usage["estimated"] = self.usage["estimated"] or reply.estimated

    def propose(self, brief: Mapping[str, Any], history: Sequence[Outcome]) -> Proposal | None:
        system = self._system_text(brief)
        messages = [{"role": "user", "content": redact(turn_prompt(brief, history, history_limit=self.history_limit,
                                                                  source_chars=self.source_chars))}]
        tokens = 0
        try:
            for round_ in range(self.repair + 1):
                reply = self.client.complete(system, messages, max_tokens=self.max_tokens)
                tokens += reply.tokens
                self._count(reply)
                if reply.stop == "refusal":
                    raise ProposerError("the model declined to answer", tokens=tokens)
                if reply.stop == "length":
                    raise ProposerError(f"the reply was cut off at {self.max_tokens} tokens; raise --llm-max-tokens", tokens=tokens)
                try:
                    parsed = parse_reply(reply.text)
                    if "stop" in parsed:
                        self.stop_reason = parsed["stop"]
                        return None
                    body = {"stage": brief["task"]["stage"], **parsed["proposal"]}
                    proposal = Proposal.from_dict(body)
                    break
                except (ValueError, ContractError) as error:
                    if round_ == self.repair:
                        raise ProposerError(f"the reply could not be used: {error}", tokens=tokens) from None
                    messages += [{"role": "assistant", "content": reply.text.strip() or "(empty)"},
                                 {"role": "user", "content": f"That reply could not be used: {error}. Reply again with the single "
                                                             "JSON object described above (and the python block if you propose source)."}]
        except LLMError as error:
            error.tokens += tokens                      # what earlier calls of this proposal cost is still spent
            raise
        return dataclasses.replace(proposal, tokens=tokens)
