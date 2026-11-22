"""A memo of what each tier said about each candidate, keyed by content and kept in the journal.

A candidate is identified by what it is, not by what the agent called it: an implementation id with its resolved
parameters, or the hash of the source text. The key adds the stage, the scene, the engine, the world count and the phost
version, because a result is only a fact about that combination. A candidate an agent proposes twice costs nothing the
second time: a failure is reported again with its reason, a pass lets the funnel skip the tiers it already passed.

The memo is a view of the journal, so it survives restarts and shares the journal's hash chain: a remembered result cannot
be edited without the chain saying where.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .journal import Journal

TIERS = ("static", "cheap", "real", "run")        # the tiers a result is kept for (tier 0 and 5 are not about a candidate)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=str)


def key_for(identity: Mapping[str, Any]) -> str:
    """The content address of an identity dict (candidate, scene, engine, worlds, versions)."""
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


class Memo:
    def __init__(self, journal: Journal):
        self.journal = journal

    def lookup(self, key: str) -> dict[str, dict]:
        """The latest stored result of each tier for ``key``: ``{tier: entry}``."""
        found: dict[str, dict] = {}
        for entry in self.journal.records("tier"):
            if entry.get("key") == key:
                found[entry["tier"]] = entry
        return found

    def store(self, key: str, tier: str, *, ok: bool, reason: str = "", seconds: float = 0.0, **detail: Any) -> dict:
        if tier not in TIERS:
            raise ValueError(f"no tier {tier!r}; known {list(TIERS)}")
        record = self.journal.append("tier", key=key, tier=tier, ok=bool(ok), reason=reason, seconds=round(float(seconds), 4),
                                     **detail)
        return record.entry
