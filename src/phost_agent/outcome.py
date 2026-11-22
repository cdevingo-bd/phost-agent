"""What the funnel concluded about one proposal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .proposal import Proposal

REJECTED, MEASURED, ACCEPTED = "rejected", "measured", "accepted"


@dataclass
class Outcome:
    attempt: int
    proposal: Proposal
    key: str
    reached: str                              # the last tier it was put through: none, static, cheap, real, run
    verdict: str                              # rejected | measured (ran, did not win) | accepted (ran and won)
    reason: str = ""
    speedup: float | None = None              # the least over the world counts, for a run that was judged
    accuracy_ratio: float | None = None
    memo: bool = False                        # answered from the memo: nothing was run
    seconds: dict[str, float] = field(default_factory=dict)
    entries: list[str] = field(default_factory=list)        # score entry ids written
    klass: str = ""

    @property
    def score(self) -> float | None:
        """What ranks outcomes against each other: the speedup of a judged run, None for anything that was not one."""
        return self.speedup if self.verdict in (MEASURED, ACCEPTED) else None

    def to_dict(self) -> dict[str, Any]:
        return dict(attempt=self.attempt, proposal=self.proposal.to_dict(), key=self.key, reached=self.reached,
                    verdict=self.verdict, reason=self.reason, speedup=self.speedup, accuracy_ratio=self.accuracy_ratio,
                    memo=self.memo, seconds={k: round(v, 4) for k, v in self.seconds.items()}, entries=list(self.entries),
                    klass=self.klass)
