"""The loop: ask a proposer for candidates, run each through the funnel, stop when the budget says to.

    gate     tier 0 once, before anything is proposed: a stage with no room to gain ends the search having spent one measurement
    propose  the proposer sees the brief and every earlier outcome; one candidate at a time unless the budget allows ``workers``
    funnel   tiers 1 to 5 (``funnel.py``); with ``workers`` > 1 the CPU tiers of several candidates run at once and the timed
             runs still go one at a time under the lease
    stop     attempts, device seconds, tokens or a stall (attempts in a row without a new best), whichever comes first

Everything the loop decides is in the journal: the task, the gate, each tier a candidate passed or failed, each attempt.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import phost

from .budget import Budget, Spend, exhausted
from .cards import task_brief
from .ceiling import Gate, check_ceiling
from .funnel import Funnel
from .journal import Journal
from .outcome import ACCEPTED, MEASURED, REJECTED, Outcome
from .proposers import Proposer, ProposerError
from .proposal import Proposal
from .task import Task


@dataclass
class Report:
    task: Task
    status: str                         # refused | budget | stalled | finished
    reason: str
    gate: Gate | None
    best: Outcome | None                # the fastest accepted outcome
    outcomes: list[Outcome]
    spend: Spend
    budget: Budget
    head: str = ""

    def funnel(self) -> dict[str, int]:
        """How many proposals each tier let through: the shape of the search."""
        counts = dict(proposed=len(self.outcomes), static=0, cheap=0, real=0, run=0, accepted=0)
        order = ("static", "cheap", "real", "run")
        for outcome in self.outcomes:
            got = -1 if outcome.reached == "none" else order.index(outcome.reached)
            ok = outcome.verdict != REJECTED
            for i, tier in enumerate(order):
                if i < got or (i == got and ok):
                    counts[tier] += 1
            counts["accepted"] += outcome.verdict == ACCEPTED
        return counts

    def to_dict(self) -> dict[str, Any]:
        return dict(task=self.task.to_dict(), status=self.status, reason=self.reason,
                    gate=self.gate.to_dict() if self.gate else None, best=self.best.to_dict() if self.best else None,
                    outcomes=[o.to_dict() for o in self.outcomes], spend=self.spend.to_dict(), budget=self.budget.to_dict(),
                    funnel=self.funnel(), head=self.head)


def run_loop(task: Task, proposer: Proposer, *, budget: Budget = Budget(), journal: Journal | None = None,
             registry: phost.Registry | None = None, funnel: Funnel | None = None, measure=None, refresh: bool = False,
             on_event: Callable[[Mapping[str, Any]], None] | None = None, **funnel_options: Any) -> Report:
    """Run one search. ``funnel_options`` go to ``Funnel`` (``ledger``, ``lease``, ``check``, ``timed``, ``allow_unsandboxed``)."""
    emit = on_event or (lambda event: None)
    journal = journal or Journal()
    funnel = funnel or Funnel(task, journal, registry=registry, refresh=refresh, **funnel_options)
    spend, outcomes = funnel.spend, []
    gate, seconds = check_ceiling(task, journal, registry=funnel.registry, measure=measure, lease=funnel.lease, refresh=refresh)
    if funnel.world.cpu_only:
        spend.cpu_seconds += seconds
    else:
        spend.gpu_seconds += seconds
    emit(dict(event="gate", gate=gate.to_dict()))
    journal.append("task", phase="start", task=task.to_dict(), budget=budget.to_dict(), proposer=proposer.name,
                   gate=gate.to_dict())

    def finish(status: str, reason: str, best: Outcome | None) -> Report:
        journal.append("task", phase="end", task_id=task.id, status=status, reason=reason, spend=spend.to_dict(),
                       best=best.to_dict() if best else None)
        report = Report(task, status, reason, gate, best, outcomes, spend, budget, str(journal.head))
        emit(dict(event="end", status=status, reason=reason))
        return report

    if not gate.ok:
        return finish("refused", gate.reason, None)
    brief = task_brief(task, funnel.world, gate, budget)
    best: Outcome | None = None
    while True:
        why = exhausted(budget, spend)
        if why:
            return finish("stalled" if why.startswith("stalled") else "budget", why, best)
        wanted = min(budget.workers, budget.attempts - spend.attempts)
        proposals: list[Proposal | ProposerError] = []
        for _ in range(wanted):
            try:
                proposal = proposer.propose(brief, outcomes)
            except ProposerError as error:
                proposals.append(error)
                break
            if proposal is None:
                break
            proposals.append(proposal)
        if not proposals:
            return finish("finished", "the proposer has nothing more to try", best)
        for outcome in _evaluate(funnel, proposals, spend.attempts, budget, journal):
            spend.attempts += 1
            spend.tokens += outcome.proposal.tokens if outcome.proposal else 0
            spend.memo_hits += outcome.memo
            outcomes.append(outcome)
            improved = outcome.verdict == ACCEPTED and (best is None or
                                                        (outcome.speedup or 0.0) > (best.speedup or 0.0) * (1 + 1e-9))
            if improved:
                best, spend.stalled = outcome, 0
            else:
                spend.stalled += 1
            journal.append("attempt", task_id=task.id, **outcome.to_dict(), improved=improved)
            emit(dict(event="attempt", outcome=outcome.to_dict(), improved=improved))


def _evaluate(funnel: Funnel, proposals: list, first_attempt: int, budget: Budget, journal: Journal) -> list[Outcome]:
    """One round: the funnel on each proposal, in proposal order. A proposer failure becomes a rejected attempt."""
    real = [p for p in proposals if isinstance(p, Proposal)]
    prepared = {}
    if len(real) > 1 and budget.workers > 1:
        with ThreadPoolExecutor(max_workers=min(budget.workers, len(real))) as pool:
            for proposal, prep in zip(real, pool.map(lambda p: funnel.prepare(p, include_real=funnel.world.cpu_only), real)):
                prepared[id(proposal)] = prep
    outcomes = []
    for offset, item in enumerate(proposals):
        attempt = first_attempt + offset + 1
        if isinstance(item, ProposerError):
            stub = Proposal(funnel.task.stage, impl="(proposer failed)")
            outcomes.append(Outcome(attempt, stub, "", "none", REJECTED, str(item)))
        elif id(item) in prepared:
            outcomes.append(funnel.finish(prepared[id(item)], attempt))
        else:
            outcomes.append(funnel.evaluate(item, attempt))
    return outcomes
