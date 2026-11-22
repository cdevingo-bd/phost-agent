"""Where the agent layer keeps its state: beside the score ledger, under ``$PHOST_HOME`` (see ``phost_score.paths``)."""

from __future__ import annotations

from pathlib import Path

from phost_score import state_dir


def agent_dir() -> Path:
    return state_dir() / "agent"


def default_journal() -> Path:
    return agent_dir() / "journal.jsonl"


def candidates_dir() -> Path:
    return agent_dir() / "candidates"


def lease_dir() -> Path:
    return agent_dir() / "leases"
