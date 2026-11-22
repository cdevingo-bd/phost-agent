"""What an agent hands the funnel: an implementation to try, as data.

    {"stage": "effective.solve", "impl": "effective.solve/direct", "params": {...}}     a registered implementation, with parameters
    {"stage": "effective.solve", "source": "<python>", "entry": "Runtime"}              a runtime class, which becomes a kernel impl

A typed expression is the first kind: ``<stage>/expression`` is a registered implementation (phost-author) whose parameters
are the text. Nothing here runs the proposal; the funnel does, and only in worker processes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import phost
from phost import ContractError

from .memo import canonical
from .paths import candidates_dir


@dataclass(frozen=True)
class Proposal:
    stage: str
    impl: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    entry: str = "Runtime"
    note: str = ""                       # the agent's reason, kept in the journal
    tokens: int = 0                      # what producing this proposal cost, as the proposer reports it

    def __post_init__(self):
        if (self.impl is None) == (self.source is None):
            raise ContractError("a proposal names exactly one of impl and source")
        if self.source is not None and self.params:
            raise ContractError("a source proposal takes no params: its runtime class is the whole candidate")
        object.__setattr__(self, "params", dict(self.params))

    KEYS = ("stage", "impl", "params", "source", "entry", "note", "tokens")

    @property
    def kind(self) -> str:
        return "impl" if self.impl is not None else "source"

    @property
    def source_hash(self) -> str | None:
        return hashlib.sha256(self.source.encode()).hexdigest() if self.source is not None else None

    @property
    def impl_id(self) -> str:
        """The id the implementation has in the registry: its own, or one made from the source's hash."""
        return self.impl if self.impl is not None else f"{self.stage}/agent-{self.source_hash[:10]}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(stage=self.stage, params=dict(self.params), note=self.note, tokens=self.tokens)
        if self.impl is not None:
            d["impl"] = self.impl
        else:
            d.update(source=self.source, entry=self.entry)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Proposal":
        unknown = set(d) - set(cls.KEYS)
        if unknown:
            raise ContractError(f"proposal: unknown keys {sorted(unknown)}; known {list(cls.KEYS)}")
        if "stage" not in d:
            raise ContractError("proposal: a stage is required")
        try:
            return cls(stage=str(d["stage"]), impl=d.get("impl"), params=d.get("params") or {}, source=d.get("source"),
                       entry=str(d.get("entry", "Runtime")), note=str(d.get("note", "")), tokens=int(d.get("tokens", 0)))
        except (TypeError, ValueError) as error:
            raise ContractError(f"proposal: {error}") from error

    def summary(self) -> str:
        if self.impl is not None:
            return f"{self.impl} {canonical(self.params)}" if self.params else self.impl
        return f"{self.impl_id} (source, {len(self.source)} chars)"


@dataclass(frozen=True)
class Candidate:
    """A source proposal written to disk as a plugin module the lab's workers can import."""

    module: str
    directory: Path
    impl_id: str


_TEMPLATE = '''"""Written by phost-agent for one proposal. The source it wraps is {runtime} beside this file."""

import importlib.util
import pathlib
import sys

from phost import Impl, ImplKind

STAGE = {stage!r}
ENTRY = {entry!r}
IMPL_ID = {impl_id!r}
VERSION = {version!r}


def register(registry):
    path = pathlib.Path(__file__).with_name({runtime!r})
    name = path.stem
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    runtime_class = getattr(module, ENTRY, None)
    if runtime_class is None:
        raise LookupError("the source defines no " + repr(ENTRY))
    registry.add_impl(Impl(id=IMPL_ID, stage=STAGE, stage_version=VERSION, version="agent", title="phost-agent proposal",
                           kind=ImplKind.KERNEL, doc=(module.__doc__ or "").strip(),
                           factory=lambda host, params: runtime_class(host, params),
                           reference=getattr(module, "reference", None)))
'''


def write_candidate(proposal: Proposal, registry: phost.Registry, directory: Path | None = None) -> Candidate:
    """Write a source proposal as ``phost_agent_cand_<hash>.py`` (plugin) and its ``..._runtime.py`` (the source itself)
    under the candidates directory. Equal source is one candidate: writing it again changes nothing."""
    if proposal.source is None:
        raise ContractError("only a source proposal becomes a candidate module")
    directory = Path(directory) if directory is not None else candidates_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"phost_agent_cand_{proposal.source_hash[:16]}"
    runtime = f"{stem}_runtime.py"
    (directory / runtime).write_text(proposal.source)
    version = registry.contract(proposal.stage).version
    (directory / f"{stem}.py").write_text(_TEMPLATE.format(runtime=runtime, stage=proposal.stage, entry=proposal.entry,
                                                           impl_id=proposal.impl_id, version=version))
    return Candidate(stem, directory, proposal.impl_id)
