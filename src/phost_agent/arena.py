"""The arena: language models compete on a leaderboard, and nothing says how.

Every contestant gets the same scene and the same facts: where the host's step spends its time, what each stage is, how an entry
is judged, and the live board. What to try is theirs: any stage, any code, several stages at once, a rival's entry as a start. The
arena keeps the rules (each replaced stage must pass its contract checks, a timed run measures speed and drift against the host's
own step with the device to itself), the board (``phost_score``, one ledger, a contestant's name as the label) and the budgets (turns,
timed runs, tokens). It does not choose what is tried, order the attempts, or coach.

Contestants take turns in a fixed rotation, one action each, so each sees what the others did before it moves, and the single device
is never shared by two timed runs. A contestant that fails for good (no key, an unknown model) leaves; the others go on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import phost
from phost import ContractError
from phost_lab.bootstrap import load_registry
from phost_lab.graph import GraphSpec, Node
from phost_score import SELF_REPORTED, Entry, Ledger, default_ledger, leaderboard, make_entries

from .arena_text import facts_text, outcome_text, parse_action, render_board, system_prompt, what_text
from .cards import render_card, source_skeleton, stage_card
from .ceiling import check_ceiling
from .checker import run_check
from .funnel import Funnel, run_timed
from .journal import Journal
from .lease import Lease
from .llm.base import ChatClient, LLMError, LLMFatal, redact
from .proposal import Proposal
from .task import Task, resolve


@dataclass(frozen=True)
class ArenaTask:
    """A scene, and how a timed run measures it. There is no stage: choosing one is the contestants' business."""

    scene: str
    worlds: tuple[int, ...] = (1, 8)
    steps: int = 20
    launches: int = 5
    repeats: int = 11
    driver: str = "graph"
    plugins: tuple[str, ...] = ()
    scene_modules: tuple[str, ...] = ()
    scene_dirs: tuple[str, ...] = ()
    run_timeout: float = 900.0
    check_timeout: float = 300.0

    def __post_init__(self):
        object.__setattr__(self, "worlds", tuple(int(w) for w in self.worlds))
        for name in ("plugins", "scene_modules", "scene_dirs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.worlds or min(self.worlds) < 1:
            raise ValueError("worlds must be positive counts")

    def for_stage(self, stage: str) -> Task:
        return Task(scene=self.scene, stage=stage, worlds=self.worlds, steps=self.steps, launches=self.launches, repeats=self.repeats,
                    driver=self.driver, min_ceiling=1.0, plugins=self.plugins, scene_modules=self.scene_modules,
                    scene_dirs=self.scene_dirs, run_timeout=self.run_timeout, check_timeout=self.check_timeout)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | dict(worlds=list(self.worlds), plugins=list(self.plugins),
                                               scene_modules=list(self.scene_modules), scene_dirs=list(self.scene_dirs))


@dataclass
class Contestant:
    name: str
    client: ChatClient
    messages: list[dict[str, str]] = field(default_factory=list)
    turns: int = 0
    submits: int = 0
    tokens: int = 0
    device_seconds: float = 0.0
    left: str = ""                           # why it is no longer playing: turns | submits | tokens | failed: ...
    pending: str = ""                        # what it is told next
    seen: int = -1                           # the journal position of the board it last saw
    best: dict[str, Any] | None = None       # its fastest ranked entry

    def summary(self) -> dict[str, Any]:
        return dict(name=self.name, model=f"{self.client.name}:{self.client.model}", turns=self.turns, submits=self.submits,
                    tokens=self.tokens, device_seconds=round(self.device_seconds, 2), left=self.left or "still playing", best=self.best)


@dataclass
class ArenaReport:
    task: ArenaTask
    contestants: list[dict[str, Any]]
    board: list[dict[str, Any]]
    head: str
    facts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(task=self.task.to_dict(), contestants=self.contestants, board=self.board, head=self.head, facts=self.facts)


class Arena:
    def __init__(self, task: ArenaTask, contestants: Sequence[Contestant], *, journal: Journal | None = None,
                 ledger: Ledger | None = None, lease: Lease | None = None, registry: phost.Registry | None = None,
                 measure=None, check: Callable = run_check, timed: Callable = run_timed, allow_unsandboxed: bool = False,
                 candidates_dir=None, turns: int = 30, submits: int = 6, tokens: int | None = None, open_code: bool = True,
                 board_rows: int = 8, feed_rows: int = 10, max_tokens: int = 16000, keep_turns: int = 40,
                 on_event: Callable[[Mapping[str, Any]], None] | None = None):
        names = [c.name for c in contestants]
        if len(contestants) < 1 or len(set(names)) != len(names):
            raise ValueError("an arena needs at least one contestant, each under its own name")
        if turns < 1 or submits < 1 or keep_turns < 1:
            raise ValueError("turns, submits and keep_turns must be at least 1")
        self.task, self.contestants = task, list(contestants)
        self.journal = journal if journal is not None else Journal()
        self.ledger = ledger if ledger is not None else Ledger(default_ledger())
        self.registry = load_registry(task.plugins, registry)
        self.stages = sorted(getattr(c, "name", c) for c in self.registry.contracts())
        self.probe = task.for_stage(self.stages[0])
        self.world = resolve(self.probe, self.registry)
        self.lease = lease or Lease(self.world.lease_device)
        self.measure, self.check, self.timed = measure, check, timed
        self.allow_unsandboxed, self.candidates_dir = allow_unsandboxed, candidates_dir
        self.turns, self.submits, self.tokens, self.open_code = turns, submits, tokens, open_code
        self.board_rows, self.max_tokens, self.keep_turns = board_rows, max_tokens, keep_turns
        self.feed_rows = feed_rows if open_code else 0              # a closed contest shows no one's work
        self.emit = on_event or (lambda event: None)
        self.worlds = max(task.worlds)
        self._funnels: dict[str, Funnel] = {}
        self.facts, self.system = "", ""

    # ---- setup ------------------------------------------------------------------------------------------

    def _funnel(self, stage: str) -> Funnel:
        if stage not in self._funnels:
            self._funnels[stage] = Funnel(self.task.for_stage(stage), self.journal, registry=self.registry, ledger=self.ledger,
                                          lease=self.lease, check=self.check, timed=self.timed,
                                          allow_unsandboxed=self.allow_unsandboxed, candidates_dir=self.candidates_dir)
        return self._funnels[stage]

    def _prepare(self) -> None:
        report = None
        try:
            gate, _ = check_ceiling(self.probe, self.journal, registry=self.registry, measure=self.measure, lease=self.lease)
            report = next((e["report"] for e in reversed(self.journal.records("ceiling")) if e.get("scene") == self.task.scene), None)
            if report is None:
                report = dict(error=gate.reason)
        except Exception as error:                       # facts are information: a contest can run without them
            report = dict(error=str(error)[:200])
        self.facts = facts_text(report, self.task.scene, self.worlds)
        listed = [s for s in self.stages if report and "error" not in report and any(
            s in c.get("stages", {}) and c["stages"][s].get("applies") for c in report.get("cells", ()))] or self.stages
        self.stage_names = listed
        self.system = redact(system_prompt(registry=self.registry, stages=listed, facts=self.facts, scene=self.task.scene,
                                           worlds=self.task.worlds, turns=self.turns, submits=self.submits, open_code=self.open_code,
                                           feed=self.feed_rows > 0))

    # ---- the feed: every check anyone ran, for the others to read ---------------------------------------

    def attempts(self) -> list[dict[str, Any]]:
        """Every check recorded for this scene, oldest first, including those of an earlier contest in the same journal."""
        return [r for r in self.journal.records("arena") if r.get("phase") == "attempt" and r.get("scene") == self.task.scene]

    _what = staticmethod(what_text)
    _outcome = staticmethod(outcome_text)

    def feed_text(self, you: str | None = None) -> str:
        """The others' latest checks, one line each. What you ran yourself is already in your own conversation."""
        if self.feed_rows < 1:
            return ""
        others = [r for r in self.attempts() if r["contestant"] != you][-self.feed_rows:]
        if not others:
            return ""
        lines = [f"- {r['id']} {r['contestant']} turn {r['turn']}: {self._what(r)}: {self._outcome(r)}" for r in others]
        return ("Recent checks by the others, newest last (look at an attempt by its id for the source and the whole reason):\n"
                + "\n".join(lines))

    def _post(self, c: Contestant, action: str, graph: Mapping[str, Proposal], prepared, ok: bool) -> str:
        """Record a check for the feed. Returns a line saying so if this exact graph was checked before (by anyone you may read)."""
        digest = self._digest(prepared)
        earlier = [r for r in self.attempts() if r.get("digest") == digest and (self.open_code or r["contestant"] == c.name)]
        reasons = [dict(stage=s, tier=p.rejection[0], reason=str(p.rejection[1])[:1200]) for s, p in prepared.items() if p.rejection]
        attempt_id = hashlib.sha256(f"{self.task.scene}|{c.name}|{c.turns}|{time.time()}".encode()).hexdigest()[:8]
        self.journal.append("arena", phase="attempt", id=attempt_id, contestant=c.name, turn=c.turns, scene=self.task.scene,
                            action=action, ok=ok, digest=digest, graph=dict(stages=self._spec(graph)), reasons=reasons)
        if not earlier:
            return ""
        before = earlier[-1]
        who = "you" if before["contestant"] == c.name else before["contestant"]
        return f"\nThis exact graph was already checked by {who} on turn {before['turn']} (attempt {before['id']}): {self._outcome(before)}."

    @staticmethod
    def _digest(prepared) -> str:
        return hashlib.sha256(repr(sorted((s, p.key) for s, p in prepared.items())).encode()).hexdigest()

    @staticmethod
    def _spec(graph: Mapping[str, Proposal]) -> dict[str, Any]:
        return {s: ({"source": p.source, "entry": p.entry, "impl": p.impl_id} if p.source else {"impl": p.impl, "params": dict(p.params)})
                for s, p in graph.items()}

    def _attempt_text(self, record: Mapping[str, Any]) -> str:
        parts = [f"Attempt {record['id']} by {record['contestant']} on turn {record['turn']} ({record['action']}): "
                 + self._outcome(record, limit=None)]
        for stage, spec in record["graph"]["stages"].items():
            if spec.get("source"):
                parts.append(f"{stage}: source\n```python\n{spec['source'].rstrip()}\n```")
            else:
                parts.append(f"{stage}: {spec.get('impl')} {spec.get('params') or ''}".rstrip())
        parts += [f"- {r['stage']}: rejected at the {r['tier']} tier: {r['reason']}" for r in record["reasons"]]
        return "\n".join(parts)

    # ---- the board --------------------------------------------------------------------------------------

    def groups(self) -> list[dict[str, Any]]:
        records = self.ledger.records()
        graphs = {r.seq: r.entry.get("graph", {}) for r in records}
        groups = leaderboard(records, scene=self.task.scene, worlds=self.worlds)
        for group in groups:
            for row in group["rows"]:
                row["graph"] = graphs.get(row["seq"], {})          # what the entry replaced, for the board to show
        return groups

    def board_text(self, you: str | None = None) -> str:
        return render_board(self.groups(), scene=self.task.scene, worlds=self.worlds, rows=self.board_rows, you=you)

    def _position(self, entry_id: str) -> tuple[str, int, int] | None:
        for group in self.groups():
            for position, row in enumerate(group["rows"], 1):
                if row["id"] == entry_id:
                    return group["klass"], position, len(group["rows"])
        return None

    # ---- the contest ------------------------------------------------------------------------------------

    def run(self) -> ArenaReport:
        self._prepare()
        self.journal.append("arena", phase="start", task=self.task.to_dict(), contestants=[c.summary() for c in self.contestants],
                            budget=dict(turns=self.turns, submits=self.submits, tokens=self.tokens, open=self.open_code))
        for c in self.contestants:
            c.pending = "The contest begins."
        while True:
            playing = [c for c in self.contestants if not c.left]
            if not playing:
                break
            for c in playing:
                self._turn(c)
        report = ArenaReport(self.task, [c.summary() for c in self.contestants], self.groups(), str(self.journal.head), self.facts)
        self.journal.append("arena", phase="end", contestants=report.contestants, head=str(self.ledger.head))
        self.emit(dict(event="end", report=report))
        return report

    def _turn(self, c: Contestant) -> None:
        if c.turns >= self.turns:
            return self._leave(c, "turns")
        if self.tokens is not None and c.tokens >= self.tokens:
            return self._leave(c, "tokens")
        c.turns += 1
        head = self.ledger.head.seq                     # the board is read now, so a rival's entry since your last move is on it
        board = self.board_text(c.name) if head != c.seen else "(the board has not changed)"
        c.seen = head
        feed = self.feed_text(c.name)
        window = (c.messages + [{"role": "user", "content": c.pending + "\n\n" + board + ("\n\n" + feed if feed else "")}])[-(2 * self.keep_turns + 1):]
        window = [dict(role=m["role"], content=redact(m["content"])) for m in window]      # everything sent, every time
        try:
            reply = c.client.complete(self.system, window, max_tokens=self.max_tokens)
        except LLMFatal as error:
            return self._leave(c, f"failed: {error}")
        except LLMError as error:
            c.pending = f"Your last reply did not go through ({error}). Try again."
            return self.emit(dict(event="turn", contestant=c.name, turn=c.turns, action="(no reply)", text=str(error)))
        c.tokens += reply.tokens
        if reply.stop == "length":
            text, action = f"Your reply was cut off at {self.max_tokens} tokens. Say less, or split the work.", "(cut off)"
        elif reply.stop == "refusal":
            text, action = "The model declined to answer.", "(declined)"
        else:
            action, text = self._act(c, reply.text)
        c.messages = window + [{"role": "assistant", "content": reply.text.strip() or "(empty)"}]
        c.pending = text
        self.journal.append("arena", phase="turn", contestant=c.name, turn=c.turns, action=action, tokens=reply.tokens)
        self.emit(dict(event="turn", contestant=c.name, turn=c.turns, action=action, text=text))

    def _leave(self, c: Contestant, why: str) -> None:
        c.left = why
        self.emit(dict(event="left", contestant=c.name, why=why))

    # ---- actions ----------------------------------------------------------------------------------------

    def _act(self, c: Contestant, text: str) -> tuple[str, str]:
        try:
            action, blocks = parse_action(text)
        except ValueError as error:
            return "(unreadable)", f"That reply could not be read: {error}. Reply with one JSON object."
        kind = action.get("action")
        try:
            if kind == "look":
                return "look", self._look(c, action)
            if kind in ("check", "submit"):
                graph = self._graph(action.get("graph"), blocks)
                return str(kind), self._check(c, graph) if kind == "check" else self._submit(c, graph, str(action.get("note", "")))
        except ContractError as error:
            return str(kind), f"Not accepted: {error}"
        return str(kind), f'Unknown action {kind!r}; the actions are look, check and submit.'

    def _look(self, c: Contestant, action: Mapping[str, Any]) -> str:
        what = action.get("what")
        if what == "card":
            stage = str(action.get("stage", ""))
            if stage not in self.stages:
                raise ContractError(f"no stage {stage!r}; the stages are {', '.join(self.stages)}")
            card = stage_card(self.registry, stage)
            return (render_card(card) + "\n\n## the source of a candidate for this stage, as a skeleton\n\n```python\n"
                    + source_skeleton(card).rstrip() + "\n```")
        if what == "entry":
            if not self.open_code:
                raise ContractError("this contest is closed: entries show what they scored, not what they submitted")
            wanted = str(action.get("id", ""))
            for record in reversed(self.journal.records("submission")):
                if wanted in record.get("entry_ids", ()):
                    parts = [f"Entry {wanted} by {record['contestant']}: {record['speedup']:.3g}x ({record['klass']}). "
                             f"Their note: {record.get('note') or '(none)'}"]
                    for stage, spec in record["graph"]["stages"].items():
                        if spec.get("source"):
                            parts.append(f"{stage}: source\n```python\n{spec['source'].rstrip()}\n```")
                        else:
                            parts.append(f"{stage}: {spec['impl']} {spec.get('params') or ''}".rstrip())
                    return "\n".join(parts)
            recorded = self._recorded_entry(wanted)
            if recorded is None:
                raise ContractError(f"no entry {wanted!r} is on this board")
            return recorded
        if what == "attempt":
            if not self.open_code:
                raise ContractError("this contest is closed: nobody sees what another contestant checked")
            wanted = str(action.get("id", ""))
            for record in reversed(self.attempts()):
                if record["id"] == wanted:
                    return self._attempt_text(record)
            raise ContractError(f"no attempt {wanted!r} is recorded for this scene")
        raise ContractError('look takes "what": "card" (with a stage), "entry" or "attempt" (with an id)')

    def _recorded_entry(self, wanted: str) -> str | None:
        """An entry on the board that did not come from this contest (a person's, or an earlier run's), described from the
        ledger: its stages and parameters, and which stages had source (the source itself is not kept with a ledger entry)."""
        for record in reversed(self.ledger.records(strict=False)):
            entry = Entry.from_dict(record.entry)
            if entry.id != wanted:
                continue
            graph = record.entry.get("graph") or {}
            speed = entry.objectives.get("speedup")
            parts = [f"Entry {wanted} by {entry.label or '(no label)'}, recorded outside this contest ({entry.trust}): "
                     + (f"{speed:.3g}x" if speed is not None else "no speedup figure") + f" ({entry.klass}). "
                     f"Their note: {graph.get('note') or '(none)'}"]
            parts += [f"{stage}: {spec.get('impl')} {spec.get('params') or ''}".rstrip() for stage, spec in (graph.get("stages") or {}).items()]
            if entry.artifacts:
                parts.append("It also submitted source for " + ", ".join(sorted(entry.artifacts)) + ", which the ledger does not keep.")
            return "\n".join(parts)
        return None

    def _graph(self, graph: Any, blocks: list[tuple[str, str]]) -> dict[str, Proposal]:
        if not isinstance(graph, Mapping) or not isinstance(graph.get("stages"), Mapping) or not graph["stages"]:
            raise ContractError('a graph is {"stages": {"<stage>": {...}, ...}} with at least one stage')
        wants_source = [s for s, spec in graph["stages"].items()
                        if isinstance(spec, Mapping) and spec.get("impl") is None and spec.get("source") is None]
        tagged = {tag: code for tag, code in blocks if tag}
        untagged = [code for tag, code in blocks if not tag]
        proposals: dict[str, Proposal] = {}
        for stage, spec in graph["stages"].items():
            if stage not in self.stages:
                raise ContractError(f"no stage {stage!r}; the stages are {', '.join(self.stages)}")
            if not isinstance(spec, Mapping):
                raise ContractError(f"{stage}: give an object with impl and params, or entry and a python block")
            body = {**spec, "stage": stage}
            if body.get("impl") is None and body.get("source") is None:
                code = tagged.get(stage) or (untagged[0] if len(wants_source) == 1 and len(untagged) == 1 else None)
                if code is None:
                    raise ContractError(f"{stage}: no source found; put it in a ```python {stage} block after the JSON")
                body["source"] = code
            body.pop("tokens", None)
            proposals[stage] = Proposal.from_dict(body)
        return proposals

    def _prepare_all(self, graph: Mapping[str, Proposal]):
        return {stage: self._funnel(stage).prepare(proposal) for stage, proposal in graph.items()}

    @staticmethod
    def _check_lines(prepared) -> tuple[list[str], bool]:
        lines, ok = [], True
        for stage, p in prepared.items():
            if p.rejection:
                tier, reason, memo = p.rejection
                ok = False
                lines.append(f"- {stage}: rejected at the {tier} tier: {reason}" + (" (answered from the memo)" if memo else ""))
            else:
                lines.append(f"- {stage}: passes " + ", ".join(t for t in ("static", "cheap", "real") if t in p.passed))
        return lines, ok

    def _check(self, c: Contestant, graph: Mapping[str, Proposal]) -> str:
        prepared = self._prepare_all(graph)
        lines, ok = self._check_lines(prepared)
        again = self._post(c, "check", graph, prepared, ok)
        return (("The contract checks pass for every stage. Nothing was timed." if ok else "The checks failed; nothing was timed.")
                + "\n" + "\n".join(lines) + again)

    def _submit(self, c: Contestant, graph: Mapping[str, Proposal], note: str) -> str:
        if c.submits >= self.submits:
            return "You have no timed runs left."
        prepared = self._prepare_all(graph)
        lines, ok = self._check_lines(prepared)
        if not ok:
            again = self._post(c, "submit", graph, prepared, ok)
            return "Not timed: the checks failed (no timed run was used).\n" + "\n".join(lines) + again
        candidates = [p.candidate for p in prepared.values() if p.candidate]
        digest = self._digest(prepared)
        spec = GraphSpec(name=f"{c.name}-{digest[:8]}", host=self.world.graph_host,
                         stages={s: Node(p.proposal.impl_id, dict(p.params)) for s, p in prepared.items()},
                         plugins=tuple(x.module for x in candidates), note=f"entered by {c.name}")
        c.submits += 1
        with self.lease.hold(f"run {c.name}", timeout=self.task.run_timeout) as held:
            result = self.timed(spec, self.probe, candidates[0] if candidates else None)
        c.device_seconds += held.seconds
        if "error" in result:
            self._record(c, graph, prepared, [], None, note)
            return f"The timed run failed: {result['error']}. It used one of your timed runs ({self.submits - c.submits} left)."
        entries = make_entries(result, label=c.name, trust=SELF_REPORTED,
                               artifacts={f"source:{s}": p.proposal.source_hash for s, p in prepared.items() if p.proposal.source})
        ids = []
        for entry in entries:
            self.ledger.append(entry.to_dict())
            ids.append(entry.id)
        speedups = [e.objectives.get("speedup") for e in entries]
        ranked = [e for e in entries if e.eligible and e.reliable]
        verdict = self._summarise(entries)
        mine = next((e for e in entries if e.worlds == self.worlds), entries[-1] if entries else None)
        where = self._position(mine.id) if mine is not None and mine.eligible and mine.reliable else None
        low = min((s for s in speedups if s is not None), default=None)
        if ranked and low is not None and (c.best is None or low > c.best["speedup"]):
            c.best = dict(speedup=low, klass=mine.klass if mine else "", id=mine.id if mine else "")
        self._record(c, graph, prepared, ids, dict(speedup=low, klass=mine.klass if mine else ""), note)
        left = self.submits - c.submits
        tail = (f"You are #{where[1]} of {where[2]} in {where[0]} at {self.worlds} worlds." if where else
                "This run is not on the board (see above).")
        text = f"Timed and ranked ({left} timed runs left).\n{verdict}\n{tail}"
        if c.submits >= self.submits:
            c.left = "submits"
            text += "\nThat was your last timed run."
        return text

    @staticmethod
    def _summarise(entries) -> str:
        lines = []
        for e in entries:
            speed = e.objectives.get("speedup")
            parts = [f"{e.worlds} worlds: " + (f"{speed:.3g}x" if speed is not None else "no speedup figure")]
            if "accuracy_ratio" in e.objectives:
                parts.append(f"accuracy ratio {e.objectives['accuracy_ratio']:.3g}")
            parts.append(e.klass)
            parts.append("ranked" if e.eligible and e.reliable else ("not eligible: " + "; ".join(e.reasons) if not e.eligible
                                                                      else "timing unreliable, not ranked: " + "; ".join(e.notes)))
            lines.append("- " + ", ".join(parts))
        return "\n".join(lines) or "- the run produced no results"

    def _record(self, c: Contestant, graph, prepared, ids, summary, note) -> None:
        self.journal.append("submission", contestant=c.name, graph=dict(stages=self._spec(graph)), entry_ids=ids, note=note[:400],
                            speedup=(summary or {}).get("speedup"), klass=(summary or {}).get("klass", ""))
