"""Claude, through the official Anthropic SDK (``pip install anthropic``). One provider among several: nothing outside this file
knows about it.

    --proposer llm:anthropic                       the default model (claude-opus-5-5)
    --proposer llm:anthropic:claude-sonnet-5-5     another one, by its exact id
    --llm-option effort=low                        output_config.effort: low | medium | high | xhigh | max (omitted: the model's own)
    --llm-option cache=false                       no cache breakpoint on the system prompt (default: one, so the brief is
                                                   paid for in full once and read from the cache after)

Credentials are the SDK's own: ``ANTHROPIC_API_KEY``, or a profile from ``ant auth login``. A key is never taken from an argument,
so it is not in a shell history or a journal. The request carries no sampling parameters (current models reject them) and leaves
thinking to the model's default. The server-side refusal fallback is not enabled: a refusal comes back as a failed attempt.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from phost import ContractError

from .base import LLMError, LLMFatal, Reply

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_FATAL = {"AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError", "UnprocessableEntityError"}
_STOP = {"end_turn": "end", "stop_sequence": "end", "max_tokens": "length", "refusal": "refusal"}


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str | None = None, *, effort: str | None = None, cache: bool = True, timeout: float = 600.0,
                 max_retries: int = 2, client=None):
        if effort is not None and effort not in EFFORTS:
            raise ContractError(f"effort must be one of {', '.join(EFFORTS)}, not {effort!r}")
        self.model, self.effort, self.cache = model or DEFAULT_MODEL, effort, cache
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise LLMFatal("the anthropic provider needs the SDK: pip install anthropic") from None
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)      # the SDK retries 429 and 5xx itself
        self._client = client

    def complete(self, system: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Reply:
        block: dict = {"type": "text", "text": system}
        if self.cache:
            block["cache_control"] = {"type": "ephemeral"}
        request: dict = dict(model=self.model, max_tokens=max_tokens, system=[block],
                             messages=[dict(role=m["role"], content=m["content"]) for m in messages])
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        try:
            response = self._client.messages.create(**request)
        except TypeError as error:                                  # the SDK's own "no credentials" error is a TypeError
            if "authentication" in str(error).lower():
                raise LLMFatal("no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`") from None
            raise
        except Exception as error:
            names = {c.__name__ for c in type(error).__mro__}
            detail = str(getattr(error, "message", None) or error)[:300]
            kind = type(error).__name__
            if names & _FATAL:
                raise LLMFatal(f"{kind}: {detail}") from None
            if names & {"APIError", "APIStatusError", "APIConnectionError", "APITimeoutError", "RateLimitError"}:
                raise LLMError(f"{kind}: {detail}") from None
            raise
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        usage = response.usage
        read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        return Reply(text, input_tokens=int(usage.input_tokens or 0) + read + written, output_tokens=int(usage.output_tokens or 0),
                     cached_tokens=read, stop=_STOP.get(response.stop_reason, "other"), model=str(getattr(response, "model", "") or ""))


def _factory(model, options):
    options = dict(options)
    effort = options.pop("effort", None)
    cache = options.pop("cache", "true").lower() not in ("0", "false", "no", "off")
    timeout = float(options.pop("timeout", 600.0))
    if options:
        raise ContractError(f"unknown option(s) for the anthropic provider: {sorted(options)}; known: effort, cache, timeout")
    return AnthropicClient(model, effort=effort, cache=cache, timeout=timeout)


def register(add) -> None:
    add("anthropic", _factory, doc=f"Claude through the Anthropic SDK; model defaults to {DEFAULT_MODEL}; options effort, cache, timeout",
        needs="pip install anthropic; ANTHROPIC_API_KEY (or ant auth login)")
