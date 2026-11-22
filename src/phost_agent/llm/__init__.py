"""Language models as proposers, behind any provider.

    from phost_agent.llm import LLMProposer, make_client
    proposer = LLMProposer(make_client("claude-code:opus"))     # Claude through the Claude Code on this machine: no API key
    proposer = LLMProposer(make_client("anthropic"))            # Claude through the API (the default model)
    proposer = LLMProposer(make_client("openai:<model>"))       # OpenAI
    proposer = LLMProposer(make_client("ollama:<model>"))       # a local model
    proposer = LLMProposer(make_client("compat:<model>", {"base_url": "https://host/v1"}))
    proposer = LLMProposer(make_client("cmd:my-llm-cli"))       # any program: prompt on stdin, answer on stdout

A provider is a ``ChatClient`` (base.py); ``register_provider`` adds one. On the command line: ``--proposer llm:<provider>[:<model>]``.
"""

from .base import ChatClient, LLMError, LLMFatal, Reply, estimate_tokens, redact
from .claude import AnthropicClient
from .claude_code import ClaudeCodeClient
from .command import CommandClient
from .openai_compat import OpenAICompatClient
from .prompt import INSTRUCTIONS, outcome_line, parse_reply, system_prompt, turn_prompt
from .proposer import LLMProposer
from .registry import Provider, load_plugin, make_client, parse_options, providers, register_provider

__all__ = ["ChatClient", "Reply", "LLMError", "LLMFatal", "estimate_tokens", "redact", "AnthropicClient", "ClaudeCodeClient", "CommandClient",
           "OpenAICompatClient", "INSTRUCTIONS", "system_prompt", "turn_prompt", "outcome_line", "parse_reply", "LLMProposer",
           "Provider", "register_provider", "providers", "make_client", "parse_options", "load_plugin"]
