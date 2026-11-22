"""What a loop may spend, and what it has.

``attempts``      proposals tried (a proposal answered from the memo counts: an agent that repeats itself is stalling)
``gpu_seconds``   time holding the device lease for timed runs and ceiling measurements on a GPU engine
``cpu_seconds``   time in check workers and in timed runs on a CPU engine (None: not limited)
``tokens``        what the proposer says its proposals cost (None: not limited; the layer never calls a model itself)
``stall``         consecutive attempts that did not beat the best result so far
``workers``       candidates checked at once on tiers 1-3. 1 by default. More needs ``cpu_seconds`` set, so a wider search is
                  always a budgeted one, and only engines that run on the CPU are checked in parallel (timed runs never are)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Budget:
    attempts: int = 20
    gpu_seconds: float = 600.0
    cpu_seconds: float | None = None
    tokens: int | None = None
    stall: int = 5
    workers: int = 1

    def __post_init__(self):
        if self.attempts < 1 or self.stall < 1 or self.workers < 1:
            raise ValueError("attempts, stall and workers must be at least 1")
        if self.gpu_seconds < 0 or (self.cpu_seconds is not None and self.cpu_seconds < 0):
            raise ValueError("a time budget cannot be negative")
        if self.workers > 1 and self.cpu_seconds is None:
            raise ValueError("workers > 1 needs a cpu_seconds budget: a wider search is always a budgeted one")

    def to_dict(self) -> dict:
        return dict(attempts=self.attempts, gpu_seconds=self.gpu_seconds, cpu_seconds=self.cpu_seconds, tokens=self.tokens,
                    stall=self.stall, workers=self.workers)


@dataclass
class Spend:
    attempts: int = 0
    gpu_seconds: float = 0.0
    cpu_seconds: float = 0.0
    tokens: int = 0
    stalled: int = 0                      # consecutive attempts without a new best
    memo_hits: int = 0
    refused_before_run: int = 0           # candidates stopped by tiers 1-3, which cost no timed run
    timed_runs: int = 0

    def to_dict(self) -> dict:
        return dict(attempts=self.attempts, gpu_seconds=round(self.gpu_seconds, 3), cpu_seconds=round(self.cpu_seconds, 3),
                    tokens=self.tokens, stalled=self.stalled, memo_hits=self.memo_hits,
                    refused_before_run=self.refused_before_run, timed_runs=self.timed_runs)


def exhausted(budget: Budget, spend: Spend) -> str | None:
    """Why the loop must stop now, or None."""
    if spend.attempts >= budget.attempts:
        return f"attempt budget used ({budget.attempts})"
    if spend.gpu_seconds >= budget.gpu_seconds:
        return f"gpu-seconds budget used ({budget.gpu_seconds:g})"
    if budget.cpu_seconds is not None and spend.cpu_seconds >= budget.cpu_seconds:
        return f"cpu-seconds budget used ({budget.cpu_seconds:g})"
    if budget.tokens is not None and spend.tokens >= budget.tokens:
        return f"token budget used ({budget.tokens})"
    if spend.stalled >= budget.stall:
        return f"stalled: {budget.stall} attempts in a row without a new best"
    return None
