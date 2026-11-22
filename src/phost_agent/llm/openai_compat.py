"""Any server that speaks the OpenAI chat-completions protocol, over the standard library (no SDK to install).

    --proposer llm:openai:<model>                  OpenAI (OPENAI_API_KEY; OPENAI_BASE_URL if you route it elsewhere)
    --proposer llm:ollama:<model>                  a local Ollama (http://localhost:11434/v1, no key)
    --proposer llm:compat:<model> --llm-option base_url=https://host/v1
                                                   vLLM, LM Studio, OpenRouter, Together, a gateway: anything that answers
                                                   POST {base_url}/chat/completions (key in LLM_API_KEY if it wants one)

Options: ``base_url``, ``api_key_env`` (the name of the variable that holds the key), ``max_tokens_field`` (``max_completion_tokens``
for OpenAI, ``max_tokens`` for most others), ``timeout``, ``retries``, and ``body.NAME=VALUE`` to put any extra field in the request
(``body.temperature=0.2``, ``body.reasoning_effort=low``). No model name is built in: they go stale, so you name one. A key comes only
from the environment, and is refused over plain http to anything that is not this machine.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

from phost import ContractError

from .base import LLMError, LLMFatal, Reply, estimate_tokens

_LOCAL = {"localhost", "127.0.0.1", "::1"}
_STOP = {"stop": "end", "length": "length", "content_filter": "refusal"}


class OpenAICompatClient:
    def __init__(self, model: str, *, base_url: str, api_key: str | None = None, name: str = "openai",
                 max_tokens_field: str = "max_tokens", timeout: float = 300.0, retries: int = 2,
                 body: Mapping[str, Any] | None = None, sleep=time.sleep):
        if not model:
            raise ContractError(f"the {name} provider needs a model: --proposer llm:{name}:<model>")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ContractError(f"base_url must be an http(s) URL, not {base_url!r}")
        if api_key and parsed.scheme != "https" and parsed.hostname not in _LOCAL:
            raise LLMFatal(f"refusing to send an API key over plain http to {parsed.hostname}; use https")
        self.name, self.model = name, model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._key, self._field, self._timeout, self._retries = api_key, max_tokens_field, timeout, retries
        self._body, self._sleep = dict(body or {}), sleep

    def complete(self, system: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Reply:
        payload = {**self._body, "model": self.model, self._field: max_tokens,
                   "messages": [{"role": "system", "content": system}, *[dict(role=m["role"], content=m["content"]) for m in messages]]}
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        data = json.dumps(payload).encode()
        last = ""
        for attempt in range(self._retries + 1):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return self._parse(json.loads(response.read()), system, messages)
            except urllib.error.HTTPError as error:
                detail = _detail(error)
                if error.code in (400, 401, 403, 404, 422):
                    raise LLMFatal(f"HTTP {error.code} from {self.name}: {detail}") from None
                last = f"HTTP {error.code} from {self.name}: {detail}"
                wait = _retry_after(error)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                last, wait = f"{self.name} unreachable: {getattr(error, 'reason', error)}", None
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                raise LLMError(f"{self.name} answered something that is not a chat completion") from None
            if attempt < self._retries:
                self._sleep(min(wait if wait is not None else 2.0 ** attempt, 30.0))
        raise LLMError(last)

    def _parse(self, data: Mapping[str, Any], system: str, messages) -> Reply:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        stop = "refusal" if message.get("refusal") and not content else _STOP.get(choice.get("finish_reason") or "stop", "other")
        usage = data.get("usage") or {}
        text = content or ""
        if usage.get("prompt_tokens") is None:
            sent = system + "".join(m["content"] for m in messages)
            return Reply(text, estimate_tokens(sent), estimate_tokens(text), stop=stop, model=str(data.get("model") or ""), estimated=True)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return Reply(text, int(usage["prompt_tokens"]), int(usage.get("completion_tokens") or 0), int(cached), stop,
                     str(data.get("model") or ""))


def _detail(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read())
        inner = body.get("error", body)
        text = inner.get("message") if isinstance(inner, dict) else str(inner)
    except Exception:
        text = error.reason if isinstance(error.reason, str) else ""
    return str(text or "")[:300]


def _retry_after(error: urllib.error.HTTPError) -> float | None:
    try:
        return float(error.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _build(name: str, model, options, *, base_url: str | None, key_env: str | None, key_required: bool, field: str):
    options = dict(options)
    url = options.pop("base_url", None) or base_url
    key_name = options.pop("api_key_env", None) or key_env
    field = options.pop("max_tokens_field", field)
    timeout = float(options.pop("timeout", 300.0))
    retries = int(options.pop("retries", 2))
    body = {k[5:]: _value(options.pop(k)) for k in [k for k in options if k.startswith("body.")]}
    if options:
        raise ContractError(f"unknown option(s) for the {name} provider: {sorted(options)}; known: base_url, api_key_env, "
                            "max_tokens_field, timeout, retries, body.NAME")
    if not url:
        raise ContractError(f"the {name} provider needs --llm-option base_url=https://host/v1 (or LLM_BASE_URL)")
    key = os.environ.get(key_name) if key_name else None
    if key_required and not key:
        raise LLMFatal(f"the {name} provider needs a key in the environment variable {key_name}")
    return OpenAICompatClient(model or "", base_url=url, api_key=key, name=name, max_tokens_field=field, timeout=timeout,
                              retries=retries, body=body)


def _openai(model, options):
    return _build("openai", model, options, base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                  key_env="OPENAI_API_KEY", key_required=True, field="max_completion_tokens")


def _ollama(model, options):
    return _build("ollama", model, options, base_url="http://localhost:11434/v1", key_env=None, key_required=False, field="max_tokens")


def _compat(model, options):
    return _build("compat", model, options, base_url=os.environ.get("LLM_BASE_URL"), key_env="LLM_API_KEY", key_required=False,
                  field="max_tokens")


def register(add) -> None:
    add("openai", _openai, doc="OpenAI chat completions (name a model); options base_url, api_key_env, max_tokens_field, body.*",
        needs="OPENAI_API_KEY")
    add("ollama", _ollama, doc="a local Ollama server over its OpenAI-compatible endpoint (name a model)", needs="ollama serving")
    add("compat", _compat, doc="any OpenAI-compatible server: vLLM, LM Studio, OpenRouter, a gateway (name a model, set base_url)",
        needs="base_url (option or LLM_BASE_URL); LLM_API_KEY if the server wants one")
