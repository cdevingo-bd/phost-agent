"""Providers by name, and ``make_client`` which turns ``anthropic``, ``anthropic:claude-opus-5-5``, ``openai:<model>`` or
``cmd:<program>`` into a client. A plugin adds a provider with ``register_provider`` (see ``load_plugin``)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Mapping

from phost import ContractError

from .base import ChatClient

Factory = Callable[[str | None, Mapping[str, str]], ChatClient]       # (model or program, options) -> client


@dataclass(frozen=True)
class Provider:
    name: str
    factory: Factory
    doc: str = ""
    needs: str = ""                  # what must be set for it to work, for ``phost-agent llms``


PROVIDERS: dict[str, Provider] = {}


def register_provider(name: str, factory: Factory, *, doc: str = "", needs: str = "") -> Provider:
    """Make ``name`` usable as ``--proposer llm:<name>[:<model>]``. A name is taken once: registering the same factory again
    is a no-op, a different one is an error."""
    existing = PROVIDERS.get(name)
    if existing is not None:
        if existing.factory is factory:
            return existing
        raise ContractError(f"LLM provider {name!r} is already registered")
    PROVIDERS[name] = Provider(name, factory, doc, needs)
    return PROVIDERS[name]


def providers() -> list[Provider]:
    _builtin()
    return sorted(PROVIDERS.values(), key=lambda p: p.name)


def make_client(spec: str, options: Mapping[str, str] | None = None) -> ChatClient:
    """``provider`` or ``provider:model`` (for ``cmd`` the part after the colon is the program and its arguments)."""
    _builtin()
    name, _, rest = spec.partition(":")
    if name not in PROVIDERS:
        raise ContractError(f"no LLM provider {name!r}; known: {', '.join(sorted(PROVIDERS)) or 'none'}")
    return PROVIDERS[name].factory(rest or None, dict(options or {}))


def load_plugin(module: str) -> None:
    """Import ``module`` and call its ``register_llm(register_provider)``, which adds providers."""
    mod = importlib.import_module(module)
    hook = getattr(mod, "register_llm", None)
    if hook is None:
        raise ContractError(f"{module} has no register_llm(register_provider)")
    hook(register_provider)


def parse_options(pairs) -> dict[str, str]:
    """``["effort=low", "body.temperature=0.2"]`` -> a dict. Values stay text; each provider reads what it knows."""
    out: dict[str, str] = {}
    for pair in pairs or ():
        key, sep, value = str(pair).partition("=")
        if not sep or not key:
            raise ContractError(f"--llm-option wants KEY=VALUE, not {pair!r}")
        out[key] = value
    return out


_done = False


def _builtin() -> None:
    global _done
    if _done:
        return
    _done = True
    from . import claude, claude_code, command, openai_compat
    claude_code.register(register_provider)
    claude.register(register_provider)
    openai_compat.register(register_provider)
    command.register(register_provider)
