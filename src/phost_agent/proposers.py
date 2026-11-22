"""Who proposes: anything that turns a brief and a history into the next candidate.

    ScriptedProposer     replays a list of proposals (a recorded session, a test, a hand-written sweep)
    CoordinateProposer   a baseline with no learning: the defaults of every implementation of the stage, then one parameter
                         at a time around the best result so far. Anything smarter should beat it
    CommandProposer      any program: it gets ``{"brief", "history"}`` as JSON on stdin and answers ``{"proposal": {...}}``
                         (or ``{"stop": "why"}``) on stdout. An agent of your own wraps itself in one of these
    LLMProposer          a language model (``phost_agent.llm``): any provider behind one small client interface (Claude, any
                         OpenAI-compatible server, a local model, or a program), told the brief and how each earlier
                         candidate fared, answering with the next one

A proposer sees only what the brief and the history show: the stage's card, the ceiling, and how each earlier proposal fared.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
from typing import Any, Iterator, Mapping, Protocol, Sequence

import phost
from phost import ContractError, ImplKind

from .memo import canonical
from .outcome import Outcome
from .proposal import Proposal


class ProposerError(RuntimeError):
    """The proposer failed to produce a proposal (the loop records the attempt as a failure and goes on). ``tokens`` is what
    the failed call cost, which still counts against the token budget."""

    def __init__(self, message: str = "", *, tokens: int = 0):
        super().__init__(message)
        self.tokens = tokens


class ProposerFatal(ProposerError):
    """The proposer cannot go on (no key, a credential refused, an unknown model): every further call would fail the same way,
    so the loop ends the search instead of spending its attempts on it."""


class Proposer(Protocol):
    name: str

    def propose(self, brief: Mapping[str, Any], history: Sequence[Outcome]) -> Proposal | None:
        """The next candidate, or None when there is nothing left to try."""


class ScriptedProposer:
    name = "scripted"

    def __init__(self, proposals: Sequence[Proposal | Mapping[str, Any]]):
        self._queue = [p if isinstance(p, Proposal) else Proposal.from_dict(p) for p in proposals]
        self._next = 0

    def propose(self, brief, history):
        if self._next >= len(self._queue):
            return None
        self._next += 1
        return self._queue[self._next - 1]


class CoordinateProposer:
    name = "coordinate"

    def __init__(self, registry: phost.Registry, stage: str, *, impls: Sequence[str] | None = None,
                 factors: Sequence[float] = (0.5, 2.0)):
        self.registry, self.stage, self.factors = registry, stage, tuple(factors)
        found = [i for i in registry.impls_for(stage) if i.kind is not ImplKind.NATIVE]
        if impls is not None:
            unknown = set(impls) - {i.id for i in found}
            if unknown:
                raise ContractError(f"no such implementation of {stage!r}: {sorted(unknown)}")
            found = [i for i in found if i.id in impls]
        self.impls = sorted(found, key=lambda i: i.id)
        self._given: set[str] = set()

    def _resolved(self, impl, params) -> dict | None:
        try:
            return impl.resolve_params(params)
        except ContractError:
            return None

    def _id(self, impl, params) -> str:
        return f"{impl.id}|{canonical(params)}"

    def _neighbours(self, impl, center: Mapping[str, Any]) -> Iterator[dict]:
        for spec in impl.params:
            value = center[spec.name]
            options: list[Any] = []
            if spec.choices:
                options = [c for c in spec.choices if c != value]
            elif spec.type == "bool":
                options = [not value]
            elif spec.type in ("int", "float") and value:       # a number at zero is not scaled: it is usually an unused constant
                for f in self.factors:
                    v = value * f
                    v = int(round(v)) if spec.type == "int" else float(v)
                    if spec.min is not None:
                        v = max(v, spec.min)
                    if spec.max is not None:
                        v = min(v, spec.max)
                    if v != value and math.isfinite(v):
                        options.append(v)
            for option in options:
                yield {**center, spec.name: option}

    def propose(self, brief, history):
        best: dict[str, tuple[float, dict]] = {}
        seen = set(self._given)
        for outcome in history:
            proposal = outcome.proposal
            if proposal.impl is None or proposal.stage != self.stage:
                continue
            try:
                impl = self.registry.impl(proposal.impl)
            except ContractError:
                continue
            resolved = self._resolved(impl, proposal.params)
            if resolved is None:
                continue
            seen.add(self._id(impl, resolved))
            if outcome.score is not None and (impl.id not in best or outcome.score > best[impl.id][0]):
                best[impl.id] = (outcome.score, resolved)
        for impl in self.impls:                                     # every implementation's defaults first
            defaults = self._resolved(impl, {})
            if defaults is not None and self._id(impl, defaults) not in seen:
                return self._give(impl, defaults)
        ranked = sorted(self.impls, key=lambda i: -(best[i.id][0] if i.id in best else -math.inf))
        for impl in ranked:                                         # then one parameter at a time around the best point
            center = best[impl.id][1] if impl.id in best else self._resolved(impl, {})
            if center is None:
                continue
            for candidate in self._neighbours(impl, center):
                resolved = self._resolved(impl, candidate)
                if resolved is not None and self._id(impl, resolved) not in seen:
                    return self._give(impl, resolved)
        return None

    def _give(self, impl, params) -> Proposal:
        self._given.add(self._id(impl, params))
        return Proposal(self.stage, impl=impl.id, params=params, note="coordinate search")


class CommandProposer:
    name = "command"

    def __init__(self, argv: str | Sequence[str], *, timeout: float = 300.0, cwd: str | None = None):
        self.argv = shlex.split(argv) if isinstance(argv, str) else list(argv)
        self.timeout, self.cwd = timeout, cwd

    def propose(self, brief, history):
        payload = json.dumps(dict(brief=brief, history=[o.to_dict() for o in history]))
        try:
            done = subprocess.run(self.argv, input=payload, capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd)
        except subprocess.TimeoutExpired:
            raise ProposerError(f"the proposer command did not answer within {self.timeout:g} s") from None
        except OSError as error:
            raise ProposerError(f"the proposer command could not start: {error}") from None
        if done.returncode != 0:
            tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else ""
            raise ProposerError(f"the proposer command exited with code {done.returncode}" + (f": {tail}" if tail else ""))
        try:
            reply = json.loads(done.stdout)
        except json.JSONDecodeError as error:
            raise ProposerError(f"the proposer command did not print JSON ({error.msg})") from None
        if not isinstance(reply, dict):
            raise ProposerError("the proposer command must print a JSON object")
        if reply.get("stop"):
            return None
        if "proposal" not in reply:
            raise ProposerError('the proposer command printed neither {"proposal": ...} nor {"stop": ...}')
        body = dict(reply["proposal"])
        if "tokens" in reply and "tokens" not in body:
            body["tokens"] = reply["tokens"]
        try:
            return Proposal.from_dict(body)
        except ContractError as error:
            raise ProposerError(f"malformed proposal: {error}") from None
