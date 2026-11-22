"""The agent's record: every ceiling measured, every tier a candidate passed or failed, every attempt, on a hash chain.

It is a ``phost_score.Ledger`` (one JSON line per record, each carrying the hash of the one before it), so editing or
dropping the middle of the history is detected the way it is for scores, with the same limits: whoever can write the file
can recompute the chain, and dropping the newest records leaves a valid one unless the head was kept elsewhere.

Kinds of record (``entry["kind"]``):

    task      a loop began (its scene, stage, budget) or ended (why, the best result, what it spent)
    ceiling   how much of the step a stage takes on a scene: the most a replacement could gain
    tier      one tier's outcome for one candidate on one scene (this is what the memo reads)
    attempt   one proposal, how far it got, and what it cost
    arena     a contest began or ended (who, on which scene, the final board), or one contestant's turn in it
    submission a graph a contestant had timed and ranked, whole (source included), so a rival may read it
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Iterator

from phost_score import Head, Ledger, LedgerError, Record
from phost_score.entry import json_safe

from .paths import default_journal

KINDS = ("task", "ceiling", "tier", "attempt", "arena", "submission")


class Journal:
    def __init__(self, path: str | Path | None = None):
        self.ledger = Ledger(path if path is not None else default_journal())
        self._cache: tuple[tuple[int, int], list[dict]] | None = None

    @property
    def path(self) -> Path:
        return self.ledger.path

    def append(self, kind: str, **fields: Any) -> Record:
        if kind not in KINDS:
            raise ValueError(f"no journal record kind {kind!r}; known {list(KINDS)}")
        when = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
        return self.ledger.append(dict(kind=kind, time=when, **json_safe(fields)))     # non-finite floats become null

    def records(self, kind: str | None = None) -> list[dict]:
        """The entries, oldest first (a damaged journal raises ``LedgerError``). Parsed once per change of the file."""
        try:
            stat = self.path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            return []
        if self._cache is None or self._cache[0] != signature:
            self._cache = (signature, [record.entry for record in self.ledger.records()])
        entries = self._cache[1]
        return [e for e in entries if kind is None or e.get("kind") == kind]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.records())

    @property
    def head(self) -> Head:
        return self.ledger.head

    def verify(self, expected: Head | None = None):
        return self.ledger.verify(expected)


__all__ = ["Journal", "KINDS", "LedgerError"]
