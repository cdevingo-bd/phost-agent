"""The funnel: what a proposal goes through before it earns a timed run, cheapest first.

    1 static   the proposal is well formed: a real stage, a registered implementation with valid parameters, or source that
               parses and defines the runtime the stage calls. No process is started.
    2 cheap    the contract check on the cheapest simulator-free host that offers the stage (CPU only).
    3 real     the contract check on the scene's own host, under the device lease when that host is on a GPU.
    4 run      a timed run with the lab: exclusive on the device (the lease), judged by ``phost_score``.
    5 ledger   what was measured is appended to the score ledger and the journal.

Each tier's result is kept in the memo under a content key, so a candidate seen before costs nothing: a failure is reported
again, a pass skips the tiers it already passed. Tier 0, the ceiling gate, is about the stage and not the candidate
(``ceiling.py``); the loop asks it once before proposing.
"""

from __future__ import annotations

import ast
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import phost
from phost import ContractError, ImplKind
from phost_lab.graph import GraphSpec, Node
from phost_score import SELF_REPORTED, Ledger, default_ledger, make_entries

from .budget import Spend
from .checker import CheckResult, run_check
from .journal import Journal
from .lease import Lease
from .memo import Memo, key_for
from .outcome import ACCEPTED, MEASURED, REJECTED, Outcome
from .proposal import Candidate, Proposal, write_candidate
from .task import Task, resolve

CHECK_WORLDS = 2                 # worlds in a contract check's host: enough to see one world's write reach another's


def run_timed(graph: GraphSpec, task: Task, candidate: Candidate | None) -> dict:
    """Tier 4 as the lab runs it: a worker process, with the candidate's directory on its import path."""
    from phost_lab.isolate import run_isolated
    from phost_lab.runner import RunConfig

    config = RunConfig(worlds=task.worlds, steps=task.steps, launches=task.launches, repeats=task.repeats, driver=task.driver)
    env = {"PYTHONPATH": str(candidate.directory)} if candidate else None
    return run_isolated(graph, task.scene, config, timeout=task.run_timeout, plugins=task.plugins,
                        scene_modules=task.scene_modules, scene_dirs=task.scene_dirs, env=env)


@dataclass
class Prepared:
    """A proposal after tiers 1 to 3 (or as many as could run), before its timed run."""

    proposal: Proposal
    key: str
    candidate: Candidate | None = None
    params: dict = field(default_factory=dict)
    passed: set = field(default_factory=set)
    rejection: tuple[str, str, bool] | None = None          # (tier, reason, answered from the memo)
    seconds: dict[str, float] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)    # what the candidate printed at each tier that ran (or was remembered)


class Funnel:
    def __init__(self, task: Task, journal: Journal, *, registry: phost.Registry | None = None, ledger: Ledger | None = None,
                 lease: Lease | None = None, check: Callable[..., CheckResult] = run_check,
                 timed: Callable[[GraphSpec, Task, Candidate | None], dict] = run_timed, allow_unsandboxed: bool = False,
                 candidates_dir=None, refresh: bool = False):
        self.task, self.journal, self.memo = task, journal, Memo(journal)
        self.world = resolve(task, registry)
        self.registry = self.world.registry
        self.ledger = ledger if ledger is not None else Ledger(default_ledger())
        self.lease = lease or Lease(self.world.lease_device)
        self.check, self.timed, self.allow_unsandboxed = check, timed, allow_unsandboxed
        self.candidates_dir, self.refresh = candidates_dir, refresh
        self.spend = Spend()
        self._lock = threading.Lock()

    # ---- identity ---------------------------------------------------------------------------------------

    def identity(self, proposal: Proposal) -> tuple[dict, dict]:
        """``(identity, resolved params)``: what the proposal is, for the memo key."""
        task = self.task
        base: dict[str, Any] = dict(stage=proposal.stage, kind=proposal.kind, scene=task.scene, engine=self.world.engine_name,
                                    worlds=list(task.worlds), steps=task.steps, goal=task.goal, driver=task.driver,
                                    phost=phost.__version__)
        if proposal.source is not None:
            return dict(base, source=proposal.source_hash, entry=proposal.entry), {}
        try:
            impl = self.registry.impl(proposal.impl)
            params = impl.resolve_params(proposal.params)
            return dict(base, impl=proposal.impl, impl_version=impl.version, params=params), params
        except ContractError:
            return dict(base, impl=proposal.impl, params=dict(proposal.params), unresolved=True), dict(proposal.params)

    # ---- tier 1 -----------------------------------------------------------------------------------------

    def static(self, proposal: Proposal) -> str | None:
        """Why the proposal cannot be tried, or None."""
        task, registry = self.task, self.registry
        if proposal.stage != task.stage:
            return f"the task is for stage {task.stage!r} and the proposal is for {proposal.stage!r}"
        try:
            contract = registry.contract(proposal.stage)
        except ContractError as error:
            return str(error)
        if proposal.impl is not None:
            try:
                impl = registry.impl(proposal.impl)
            except ContractError as error:
                return str(error)
            if impl.stage != proposal.stage:
                return f"{impl.id!r} is an implementation of {impl.stage!r}, not {proposal.stage!r}"
            if impl.kind is ImplKind.NATIVE:
                return f"{impl.id!r} is the host's own version: there is nothing to swap"
            try:
                impl.resolve_params(proposal.params)
            except ContractError as error:
                return str(error)
            return None
        return _source_problem(proposal, contract)

    # ---- tiers 2 and 3 ----------------------------------------------------------------------------------

    def _request(self, p: Prepared, tier: str) -> dict:
        plugins = list(self.task.plugins) + ([p.candidate.module] if p.candidate else [])
        return dict(stage=p.proposal.stage, impl=p.proposal.impl_id, params=p.params, tier=tier, scene=self.task.scene,
                    worlds=CHECK_WORLDS, plugins=plugins, scene_modules=list(self.task.scene_modules),
                    scene_dirs=list(self.task.scene_dirs))

    def _call(self, p: Prepared, tier: str) -> CheckResult:
        """One check request for ``p``, in the sandbox when it is source; a real check on a GPU holds the lease."""
        def run() -> CheckResult:
            return self.check(self._request(p, tier), untrusted=p.proposal.source is not None,
                              allow_unsandboxed=self.allow_unsandboxed, timeout=self.task.check_timeout,
                              candidate_dir=p.candidate.directory if p.candidate else None)
        if tier == "real" and not self.world.cpu_only:
            with self.lease.hold(f"check {p.proposal.impl_id}", timeout=self.task.check_timeout):
                return run()
        return run()

    def diagnose(self, proposal: Proposal, tier: str = "real") -> CheckResult:
        """Run one candidate through one check tier outside the memo and without a verdict: what it printed is the answer. For a
        probe the arena writes itself (see ``facts.py``), not for a contestant's proposal."""
        candidate = write_candidate(proposal, self.registry, self.candidates_dir) if proposal.source is not None else None
        return self._call(Prepared(proposal, key="", candidate=candidate, params=dict(proposal.params)), tier)

    def _checked(self, p: Prepared, tier: str, memo_tier: str) -> bool:
        """Run one check tier if the memo has no answer. True when the candidate may go on."""
        stored = None if self.refresh else self.memo.lookup(p.key).get(memo_tier)
        if stored is not None:
            if stored.get("output"):
                p.outputs[memo_tier] = stored["output"]
            if not stored["ok"]:
                p.rejection = (memo_tier, stored["reason"], True)
                return False
            p.passed.add(memo_tier)
            return True
        on_gpu = tier == "real" and not self.world.cpu_only
        started = time.monotonic()
        result = self._call(p, tier)
        seconds = time.monotonic() - started
        p.seconds[memo_tier] = seconds
        with self._lock:
            if on_gpu:
                self.spend.gpu_seconds += seconds
            else:
                self.spend.cpu_seconds += seconds
        printed = dict(output=result.output) if result.output else {}
        if result.output:
            p.outputs[memo_tier] = result.output
        if result.status == "skipped":
            self.memo.store(p.key, memo_tier, ok=True, reason=f"skipped: {result.reason}", seconds=seconds, skipped=True, **printed)
            p.passed.add(memo_tier)
            return True
        if result.ok:
            self.memo.store(p.key, memo_tier, ok=True, seconds=seconds, host=result.host, metrics=result.metrics, **printed)
            p.passed.add(memo_tier)
            return True
        reason = result.message()
        if not result.transient:
            self.memo.store(p.key, memo_tier, ok=False, reason=reason, seconds=seconds, host=result.host,
                            failures=result.failures, **printed)
        p.rejection = (memo_tier, reason, False)
        return False

    def prepare(self, proposal: Proposal, *, include_real: bool = True) -> Prepared:
        """Tiers 1 to 3 for one proposal. Safe to call from several threads at once; with ``include_real`` False it leaves
        tier 3 to ``finish`` (a real check on a GPU is serialised under the lease)."""
        identity, params = self.identity(proposal)
        p = Prepared(proposal, key_for(identity), params=params)
        stored = None if self.refresh else self.memo.lookup(p.key).get("static")
        if stored is not None:
            problem = None if stored["ok"] else stored["reason"]
            memoized = True
        else:
            problem, memoized = self.static(proposal), False
            self.memo.store(p.key, "static", ok=problem is None, reason=problem or "")
        if problem:
            p.rejection = ("static", problem, memoized)
            return p
        p.passed.add("static")
        if proposal.source is not None:
            p.candidate = write_candidate(proposal, self.registry, self.candidates_dir)
        if not self._checked(p, "cheap", "cheap"):
            return p
        if include_real:
            self._checked(p, "real", "real")
        return p

    # ---- tiers 4 and 5 ----------------------------------------------------------------------------------

    def finish(self, p: Prepared, attempt: int) -> Outcome:
        """Tier 3 if it has not run, then 4 and 5."""
        proposal = p.proposal
        if p.rejection is None and "real" not in p.passed:
            self._checked(p, "real", "real")
        if p.rejection is not None:
            tier, reason, memoized = p.rejection
            with self._lock:
                self.spend.refused_before_run += 1
            return Outcome(attempt, proposal, p.key, tier, REJECTED, reason, memo=memoized, seconds=dict(p.seconds))
        stored = None if self.refresh else self.memo.lookup(p.key).get("run")
        if stored is not None and stored.get("verdict"):
            return Outcome(attempt, proposal, p.key, "run", stored["verdict"], stored["reason"], stored.get("speedup"),
                           stored.get("accuracy_ratio"), memo=True, seconds=dict(p.seconds), entries=list(stored.get("entries", ())),
                           klass=stored.get("klass", ""))
        return self._run(p, attempt)

    def _run(self, p: Prepared, attempt: int) -> Outcome:
        task, proposal = self.task, p.proposal
        graph = GraphSpec(name=f"agent-{p.key[:8]}", host=self.world.graph_host,
                          stages={task.stage: Node(proposal.impl_id, dict(p.params))},
                          plugins=(p.candidate.module,) if p.candidate else (),
                          note=f"proposed by phost-agent: {proposal.summary()}")
        with self.lease.hold(f"run {proposal.impl_id}", timeout=task.run_timeout) as held:
            result = self.timed(graph, task, p.candidate)
        with self._lock:
            self.spend.timed_runs += 1
            if self.world.cpu_only:
                self.spend.cpu_seconds += held.seconds
            else:
                self.spend.gpu_seconds += held.seconds
        p.seconds["run"] = held.seconds
        if "error" in result:
            reason = f"the timed run failed: {result['error']}"
            transient = result["error"].startswith(("timed out", "cancelled", "worker exited"))
            if not transient:
                self.memo.store(p.key, "run", ok=False, reason=reason, seconds=held.seconds, verdict=REJECTED)
            return Outcome(attempt, proposal, p.key, "run", REJECTED, reason, seconds=dict(p.seconds))
        entries = make_entries(result, label=task.label or f"agent:{task.id}", trust=SELF_REPORTED,
                               artifacts={"source": proposal.source_hash} if proposal.source else None)
        verdict, reason, speedup, ratio, klass = self._judge(entries)
        ids = []
        for entry in entries:                                     # tier 5: the score ledger, then the journal's own account
            self.ledger.append(entry.to_dict())
            ids.append(entry.id)
        # A noisy measurement says something about the run and not the candidate: timing flagged unreliable, and an accuracy
        # ratio on an engine whose native control is itself noisy (the ratio is against that noise, so it moves between runs).
        noisy_ratio = verdict == REJECTED and reason.startswith("accuracy ratio") and not result.get("meta", {}).get("repeatable")
        transient = noisy_ratio or (verdict == MEASURED and "unreliable" in reason)
        if not transient:
            self.memo.store(p.key, "run", ok=verdict != REJECTED, reason=reason, seconds=held.seconds, verdict=verdict,
                            speedup=speedup, accuracy_ratio=ratio, entries=ids, klass=klass)
        return Outcome(attempt, proposal, p.key, "run", verdict, reason, speedup, ratio, seconds=dict(p.seconds),
                       entries=ids, klass=klass)

    def _judge(self, entries) -> tuple[str, str, float | None, float | None, str]:
        """``(verdict, reason, speedup, accuracy_ratio, class)`` of a run's entries (one per world count)."""
        task = self.task
        if not entries:
            return REJECTED, "the run produced no results", None, None, ""
        klass = entries[0].klass
        bad = [reason for e in entries if not e.eligible for reason in e.reasons]
        if bad:
            return REJECTED, "; ".join(dict.fromkeys(bad)), None, None, klass
        speedups = [e.objectives.get("speedup") for e in entries]
        ratios = [e.objectives["accuracy_ratio"] for e in entries if "accuracy_ratio" in e.objectives]
        speedup = min(speedups) if all(s is not None for s in speedups) else None
        ratio = max(ratios) if ratios else None
        if task.goal == "behaviour":                      # nothing here is about speed, so how noisy the timing was does not matter
            return ACCEPTED, "passes its checks", speedup, ratio, klass
        if not all(e.reliable for e in entries):
            why = next((note for e in entries for note in e.notes if "unreliable" in note or "GPU processes" in note),
                       "timing reliability could not be established")
            return MEASURED, f"timing unreliable, so not counted as a win: {why}", speedup, ratio, klass
        if speedup is None:
            return MEASURED, f"a {klass} swap has no speedup to rank on", None, ratio, klass
        if speedup > task.min_speedup:
            return ACCEPTED, f"{speedup:.3f}x on every world count", speedup, ratio, klass
        return MEASURED, f"{speedup:.3f}x at its worst world count, not above {task.min_speedup:g}x", speedup, ratio, klass

    def evaluate(self, proposal: Proposal, attempt: int) -> Outcome:
        """Tiers 1 to 5 for one proposal, in this thread."""
        return self.finish(self.prepare(proposal), attempt)


def _source_problem(proposal: Proposal, contract) -> str | None:
    """What is wrong with a source proposal that can be seen without running it."""
    try:
        tree = ast.parse(proposal.source)
    except SyntaxError as error:
        return f"the source does not parse: {error.msg} (line {error.lineno})"
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    node = classes.get(proposal.entry)
    if node is None:
        return f"the source defines no class {proposal.entry!r} (it defines {sorted(classes) or 'none'})"
    if not node.bases:
        methods = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        wanted = [m for m in (contract.entry, "metrics") if m]
        missing = [m for m in wanted if m not in methods]
        if missing:
            return f"{proposal.entry} has no method {missing[0]!r} (the stage calls {contract.entry!r} once per execution)"
    if contract.reference == "impl" and not any(isinstance(n, ast.FunctionDef) and n.name == "reference" for n in tree.body):
        return f"stage {contract.name} judges against the implementation's own float64 reference: define reference(inputs, params)"
    return None
