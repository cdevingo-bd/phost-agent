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

from .arena_text import (MAX_ACTIONS, facts_text, outcome_text, parse_actions, render_board, render_board_delta, system_prompt,
                         what_text)
from .cards import render_card, source_skeleton, stage_card
from .ceiling import check_ceiling
from .checker import run_check
from .context import Exchange, assemble, notebook_text, pinned_text, trim
from .edits import apply_hunks, line_changes, parse_hunks
from .facts import facts_block, facts_identity, probe_source
from .funnel import Funnel, run_timed
from .journal import Journal
from .lease import Lease
from .llm.base import ChatClient, LLMError, LLMFatal, redact
from .memo import key_for
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
    exchanges: list[Exchange] = field(default_factory=list)      # what was said, turn by turn (only a window of it is sent)
    turns: int = 0
    submits: int = 0
    tokens: int = 0                          # input and output, as the provider counts them (input includes what it served from its cache)
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    device_seconds: float = 0.0
    left: str = ""                           # why it is no longer playing: turns | submits | tokens | failed: ...
    pending: str = ""                        # what it is told next
    seen: int = -1                           # the ledger position of the board it last saw
    board_seen: set = field(default_factory=set)      # entry ids it has been shown
    fed: set = field(default_factory=set)             # attempt ids it has been shown
    pinned: dict[str, str] = field(default_factory=dict)       # stage -> its card, shown at the top for the whole contest
    reads: list[tuple[int, str]] = field(default_factory=list)  # what it looked at, with the turn
    best: dict[str, Any] | None = None       # its fastest ranked entry

    def summary(self) -> dict[str, Any]:
        return dict(name=self.name, model=f"{self.client.name}:{self.client.model}", turns=self.turns, submits=self.submits,
                    tokens=self.tokens, input_tokens=self.tokens_in, output_tokens=self.tokens_out, cached_tokens=self.tokens_cached,
                    device_seconds=round(self.device_seconds, 2), left=self.left or "still playing", best=self.best)


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
                 board_rows: int = 8, feed_rows: int = 10, max_tokens: int = 16000, keep_turns: int = 6,
                 runtime_facts: bool = True, on_event: Callable[[Mapping[str, Any]], None] | None = None):
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
        self.board_rows, self.max_tokens, self.keep_turns, self.runtime_facts = board_rows, max_tokens, keep_turns, runtime_facts
        self.feed_rows = feed_rows if open_code else 0              # a closed contest shows no one's work
        self.emit = on_event or (lambda event: None)
        self.worlds = max(task.worlds)
        self._funnels: dict[str, Funnel] = {}
        self.facts, self.system = "", ""
        self.contest = ""                                    # names this contest in a journal that may hold several
        self.preloaded: list[str] = []                       # the stages whose cards every contestant starts with
        self._cards: dict[str, str] = {}
        self._forks: dict[str, dict[str, Any]] = {}          # source hash -> what it was edited from

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
                                           feed=self.feed_rows > 0, keep=self.keep_turns))
        self.preloaded = self._ranked_stages(report)
        for stage in self.preloaded:
            self._card_text(stage)                       # measured once here, so no contestant spends a turn on it

    def _ranked_stages(self, report: Mapping[str, Any] | None, limit: int = 3) -> list[str]:
        """The stages whose swaps are ranked on speed (exact or approximate) and apply here, the biggest share of the step first."""
        if not report or "error" in report:
            return []
        cells = [c for c in report.get("cells", ()) if c.get("stages")]
        cell = next((c for c in cells if c.get("worlds") == self.worlds), cells[-1] if cells else None)
        if cell is None:
            return []
        ranked = [(info.get("share") or 0.0, stage) for stage, info in cell["stages"].items()
                  if info.get("applies") and info.get("swap_class") in ("exact", "approximate") and stage in self.stages]
        return [stage for _, stage in sorted(ranked, reverse=True)[:limit]]

    def _card_text(self, stage: str) -> str:
        """A stage's card with the skeleton of a runtime for it and what a check's inputs measured on this scene."""
        if stage not in self._cards:
            card = stage_card(self.registry, stage)
            skeleton = source_skeleton(card).rstrip()
            text = render_card(card) + "\n\n## the source of a candidate for this stage, as a skeleton\n\n```python\n" + skeleton + "\n```"
            measured = self._measured(stage, card, skeleton) if self.runtime_facts else ""
            self._cards[stage] = text + ("\n\n" + facts_block(measured) if measured else "")
        return self._cards[stage]

    def _measured(self, stage: str, card: Mapping[str, Any], skeleton: str) -> str:
        """What a probe printed at the real tier, from the journal if this scene and stage were measured before."""
        key = key_for(facts_identity(self.task.scene, stage, skeleton))
        for record in reversed(self.journal.records("facts")):
            if record.get("key") == key:
                return record["text"]
        try:
            result = self._funnel(stage).diagnose(Proposal(stage, source=probe_source(card)), "real")
        except Exception:                                # facts are information: a contest can run without them
            return ""
        if result.output:
            self.journal.append("facts", key=key, scene=self.task.scene, stage=stage, text=result.output)
        return result.output

    # ---- the feed: every check anyone ran, for the others to read ---------------------------------------

    def attempts(self) -> list[dict[str, Any]]:
        """Every check recorded for this scene, oldest first, including those of an earlier contest in the same journal."""
        return [r for r in self.journal.records("arena") if r.get("phase") == "attempt" and r.get("scene") == self.task.scene]

    _what = staticmethod(what_text)
    _outcome = staticmethod(outcome_text)

    def feed_text(self, c: Contestant) -> str:
        """The others' checks this contestant has not been shown yet, one line each (the newest ``feed_rows``; older ones by id).
        What it ran itself is already in its own conversation, and what it was shown before is not shown again."""
        if self.feed_rows < 1:
            return ""
        new = [r for r in self.attempts() if r["contestant"] != c.name and r["id"] not in c.fed]
        if not new:
            return ""
        c.fed.update(r["id"] for r in new)
        shown, older = new[-self.feed_rows:], new[:-self.feed_rows]
        lines = [f"- {r['id']} {r['contestant']} turn {r['turn']}: {self._what(r)}: {self._outcome(r)}" for r in shown]
        if older:
            lines.insert(0, f"- {len(older)} more, older: " + ", ".join(r["id"] for r in older))
        return ("New checks by the others since you last looked, newest last (look at an attempt by its id for the source and the "
                "whole reason):\n" + "\n".join(lines))

    def _post(self, c: Contestant, action: str, graph: Mapping[str, Proposal], prepared, ok: bool, note: str = "") -> str:
        """Record a check for the feed. Returns the attempt's id, and a line saying so if this exact graph was checked before (by
        anyone you may read)."""
        digest = self._digest(prepared)
        earlier = [r for r in self.attempts() if r.get("digest") == digest and (self.open_code or r["contestant"] == c.name)]
        reasons = [dict(stage=s, tier=p.rejection[0], reason=str(p.rejection[1])[:1200]) for s, p in prepared.items() if p.rejection]
        printed = [dict(stage=s, tier=tier, text=text) for s, p in prepared.items() for tier, text in p.outputs.items()]
        attempt_id = hashlib.sha256(f"{self.task.scene}|{c.name}|{c.turns}|{time.time()}".encode()).hexdigest()[:8]
        self.journal.append("arena", phase="attempt", id=attempt_id, contestant=c.name, turn=c.turns, scene=self.task.scene,
                            action=action, ok=ok, digest=digest, graph=dict(stages=self._spec(graph)), reasons=reasons,
                            contest=self.contest, **(dict(note=note[:400]) if note else {}), **(dict(output=printed) if printed else {}))
        mine = f"\nThis is attempt {attempt_id}: look at it, or build on it with base."
        if not earlier:
            return mine
        before = earlier[-1]
        who = "you" if before["contestant"] == c.name else before["contestant"]
        return (mine + f"\nThis exact graph was already checked by {who} on turn {before['turn']} (attempt {before['id']}): "
                       f"{self._outcome(before)}.")

    @staticmethod
    def _digest(prepared) -> str:
        return hashlib.sha256(repr(sorted((s, p.key) for s, p in prepared.items())).encode()).hexdigest()

    def _spec(self, graph: Mapping[str, Proposal]) -> dict[str, Any]:
        out = {}
        for s, p in graph.items():
            if not p.source:
                out[s] = {"impl": p.impl, "params": dict(p.params)}
                continue
            out[s] = {"source": p.source, "entry": p.entry, "impl": p.impl_id}
            fork = self._forks.get(p.source_hash)
            if fork:
                out[s].update(base=fork["base"], edit=fork["edit"])
        return out

    def _attempt_text(self, record: Mapping[str, Any]) -> str:
        parts = [f"Attempt {record['id']} by {record['contestant']} on turn {record['turn']} ({record['action']}): "
                 + self._outcome(record, limit=None)]
        for stage, spec in record["graph"]["stages"].items():
            if spec.get("source"):
                parts.append(f"{stage}: source\n```python\n{spec['source'].rstrip()}\n```")
            else:
                parts.append(f"{stage}: {spec.get('impl')} {spec.get('params') or ''}".rstrip())
        parts += [f"- {r['stage']}: rejected at the {r['tier']} tier: {r['reason']}" for r in record["reasons"]]
        for o in record.get("output", ()):
            parts.append(f"- printed by {o['stage']} at the {o['tier']} tier:\n" + "\n".join("    " + line for line in o["text"].splitlines()))
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
        self.contest = hashlib.sha256(f"{self.task.scene}|{time.time()}".encode()).hexdigest()[:10]
        self.journal.append("arena", phase="start", task=self.task.to_dict(), contestants=[c.summary() for c in self.contestants],
                            budget=dict(turns=self.turns, submits=self.submits, tokens=self.tokens, open=self.open_code,
                                        keep_turns=self.keep_turns),
                            contest=self.contest, preloaded=list(self.preloaded))
        for c in self.contestants:
            c.pending = "The contest begins."
            c.pinned = {stage: self._card_text(stage) for stage in self.preloaded}
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
        began = time.monotonic()
        window = trim(c.exchanges, self.keep_turns)
        shown = (set(c.board_seen), set(c.fed), c.seen)          # what it has been shown, to put back if the call does not go through
        notebook = self._notebook(c, window[0].turn if window else None)
        state = "\n\n".join(part for part in (notebook, self._board_state(c), self.feed_text(c)) if part)
        pinned = pinned_text(c.pinned)
        messages = assemble(pinned, window, c.pending + "\n\n" + state)
        messages = [dict(role=m["role"], content=redact(m["content"])) for m in messages]      # everything sent, every time
        self._working(c, f"asking {c.client.model}")
        asked = time.monotonic()
        try:
            reply = c.client.complete(self.system, messages, max_tokens=self.max_tokens)
        except LLMFatal as error:
            return self._leave(c, f"failed: {error}")
        except LLMError as error:
            c.board_seen, c.fed, c.seen = shown                  # it was not shown this, so the next turn shows it
            c.pending = f"Your last reply did not go through ({error}). Try again.\n\n{c.pending}"
            return self.emit(dict(event="turn", contestant=c.name, turn=c.turns, action="(no reply)", text=str(error)))
        model_seconds = time.monotonic() - asked
        c.tokens += reply.tokens
        c.tokens_in, c.tokens_out, c.tokens_cached = c.tokens_in + reply.input_tokens, c.tokens_out + reply.output_tokens, c.tokens_cached + reply.cached_tokens
        if reply.stop == "length":
            text, action = f"Your reply was cut off at {self.max_tokens} tokens. Say less, or split the work.", "(cut off)"
        elif reply.stop == "refusal":
            text, action = "The model declined to answer.", "(declined)"
        else:
            action, text = self._act(c, reply.text)
        c.exchanges = window + [Exchange(c.turns, c.pending, reply.text.strip() or "(empty)")]
        c.pending = text
        self.journal.append("arena", phase="turn", contestant=c.name, turn=c.turns, action=action, tokens=reply.tokens,
                            input_tokens=reply.input_tokens, output_tokens=reply.output_tokens, cached_tokens=reply.cached_tokens,
                            estimated=reply.estimated, contest=self.contest,
                            sent=dict(messages=len(messages), exchanges=len(window), chars=sum(len(m["content"]) for m in messages),
                                      system=len(self.system), pinned=len(pinned), notebook=len(notebook), state=len(state)))
        self.emit(dict(event="turn", contestant=c.name, turn=c.turns, action=action, text=text, input_tokens=reply.input_tokens,
                       output_tokens=reply.output_tokens, cached_tokens=reply.cached_tokens, seconds=time.monotonic() - began,
                       model_seconds=model_seconds))

    def _working(self, c: Contestant, what: str) -> None:
        """Say what is taking the time: a model call, a check, or a timed run that holds the device."""
        self.emit(dict(event="work", contestant=c.name, turn=c.turns, what=what))

    def _notebook(self, c: Contestant, before: int | None) -> str:
        """The contestant's own checks and timed runs older than its window, compiled from the journal."""
        if before is None:
            return ""
        mine = [r for r in self.attempts() if r["contestant"] == c.name and r.get("contest") == self.contest]
        runs = [r for r in self.journal.records("submission") if r.get("contest") == self.contest and r["contestant"] == c.name]
        return notebook_text(attempts=mine, runs=runs, reads=c.reads, best=c.best, before=before)

    def _board_state(self, c: Contestant) -> str:
        """The whole board the first time, then only what is new on it (all of it by looking at "board")."""
        head = self.ledger.head.seq
        if head == c.seen:
            return "(the board has not changed)"
        first = c.seen == -1
        c.seen = head
        if first:
            groups = self.groups()
            c.board_seen.update(row["id"] for g in groups for row in g["rows"])
            return render_board(groups, scene=self.task.scene, worlds=self.worlds, rows=self.board_rows, you=c.name)
        return render_board_delta(self.groups(), c.board_seen, you=c.name) or "(the board has not changed)"

    def _leave(self, c: Contestant, why: str) -> None:
        c.left = why
        self.emit(dict(event="left", contestant=c.name, why=why))

    # ---- actions ----------------------------------------------------------------------------------------

    def _act(self, c: Contestant, text: str) -> tuple[str, str]:
        """Run a reply: one action or several, in order. A turn is one call however many actions it holds."""
        try:
            actions, blocks = parse_actions(text)
        except ValueError as error:
            return "(unreadable)", f"That reply could not be read: {error}. Reply with one JSON object."
        ran = actions[:MAX_ACTIONS]
        wants = bases = 0
        for action in ran:
            stages = (action.get("graph") or {}).get("stages") if isinstance(action.get("graph"), Mapping) else None
            for spec in (stages or {}).values() if isinstance(stages, Mapping) else ():
                if isinstance(spec, Mapping) and spec.get("base") is not None:
                    bases += 1
                elif isinstance(spec, Mapping) and spec.get("impl") is None and spec.get("source") is None:
                    wants += 1
        untagged_ok = dict(python=wants == 1 and blocks.untagged("python") == 1, edit=bases == 1 and blocks.untagged("edit") == 1)
        done = [self._do(c, action, blocks, untagged_ok) for action in ran]
        if len(done) == 1 and len(actions) == 1:
            return done[0]
        text = "\n\n".join(f"[{i} of {len(actions)}] {kind}\n{out}" for i, (kind, out) in enumerate(done, 1))
        if len(actions) > len(ran):
            text += f"\n\nOnly the first {len(ran)} of your {len(actions)} actions were run."
        return "+".join(kind for kind, _ in done), text

    def _do(self, c: Contestant, action: Mapping[str, Any], blocks, untagged_ok: Mapping[str, bool]) -> tuple[str, str]:
        kind = action.get("action")
        try:
            if kind == "look":
                return "look", self._look(c, action)
            if kind in ("check", "submit"):
                graph = self._graph(action.get("graph"), blocks, c, untagged_ok)
                note = str(action.get("note", ""))
                return str(kind), self._check(c, graph, note) if kind == "check" else self._submit(c, graph, note)
        except ContractError as error:
            return str(kind), f"Not accepted: {error}"
        return str(kind), f'Unknown action {kind!r}; the actions are look, check and submit.'

    def _look(self, c: Contestant, action: Mapping[str, Any]) -> str:
        what = action.get("what")
        if what == "card":
            stage = str(action.get("stage", ""))
            if stage not in self.stages:
                raise ContractError(f"no stage {stage!r}; the stages are {', '.join(self.stages)}")
            if stage in c.pinned:
                return f"The card for {stage} is already pinned at the top of this conversation."
            c.pinned[stage] = self._card_text(stage)
            c.reads.append((c.turns, f"card {stage}"))
            return f"The card for {stage} is now pinned at the top of this conversation. It stays there, so it is not repeated below."
        if what == "board":
            groups = self.groups()
            c.board_seen.update(row["id"] for g in groups for row in g["rows"])
            return render_board(groups, scene=self.task.scene, worlds=self.worlds, rows=max(self.board_rows, 30), you=c.name)
        if what == "entry":
            if not self.open_code:
                raise ContractError("this contest is closed: entries show what they scored, not what they submitted")
            wanted = str(action.get("id", ""))
            c.reads.append((c.turns, f"entry {wanted}"))
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
            c.reads.append((c.turns, f"attempt {wanted}"))
            for record in reversed(self.attempts()):
                if record["id"] == wanted:
                    return self._attempt_text(record)
            raise ContractError(f"no attempt {wanted!r} is recorded for this scene")
        raise ContractError('look takes "what": "card" (with a stage), "entry" or "attempt" (with an id), or "board"')

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

    def _graph(self, graph: Any, blocks, c: Contestant | None = None, untagged_ok: Mapping[str, bool] | None = None) -> dict[str, Proposal]:
        if not isinstance(graph, Mapping) or not isinstance(graph.get("stages"), Mapping) or not graph["stages"]:
            raise ContractError('a graph is {"stages": {"<stage>": {...}, ...}} with at least one stage')
        untagged_ok = untagged_ok or {}
        proposals: dict[str, Proposal] = {}
        for stage, spec in graph["stages"].items():
            if stage not in self.stages:
                raise ContractError(f"no stage {stage!r}; the stages are {', '.join(self.stages)}")
            if not isinstance(spec, Mapping):
                raise ContractError(f"{stage}: give an object with impl and params, an entry and a python block, or a base")
            body = {**spec, "stage": stage}
            base = body.pop("base", None)
            if base is not None:
                if body.get("impl") is not None or body.get("source") is not None:
                    raise ContractError(f"{stage}: a base is the source, so give no impl or source with it")
                original, entry = self._source_of(str(base), stage, c)
                source = original
                edit = blocks.take("edit", stage, untagged_ok=bool(untagged_ok.get("edit")))
                if edit is not None:
                    try:
                        source = apply_hunks(original, parse_hunks(edit))
                    except ContractError as error:
                        raise ContractError(f"{stage}: the edit to {base} was not applied: {error}") from None
                added, removed = line_changes(original, source)
                body.update(source=source, entry=body.get("entry") or entry)
                self._forks[hashlib.sha256(source.encode()).hexdigest()] = dict(base=str(base), edit=f"+{added}/-{removed}")
            elif body.get("impl") is None and body.get("source") is None:
                code = blocks.take("python", stage, untagged_ok=bool(untagged_ok.get("python")))
                if code is None:
                    raise ContractError(f"{stage}: no source found; put it in a ```python {stage} block after the JSON")
                body["source"] = code
            body.pop("tokens", None)
            proposals[stage] = Proposal.from_dict(body)
        return proposals

    def _source_of(self, wanted: str, stage: str, c: Contestant | None) -> tuple[str, str]:
        """``(source, entry)`` of ``stage`` in an earlier attempt or entry: yours, or anyone's while the contest is open."""
        found = None
        for record in reversed(self.attempts()):
            if record["id"] == wanted:
                found = record
                break
        if found is None:
            for record in reversed(self.journal.records("submission")):
                if wanted in record.get("entry_ids", ()):
                    found = record
                    break
        if found is None:
            raise ContractError(f"no attempt or entry {wanted!r} is recorded for this scene")
        if not self.open_code and (c is None or found["contestant"] != c.name):
            raise ContractError("this contest is closed: you can build on your own attempts and entries only")
        specs = found["graph"]["stages"]
        spec = specs.get(stage)
        if not spec or not spec.get("source"):
            have = [s for s, x in specs.items() if x.get("source")]
            raise ContractError(f"{wanted} has no source for {stage}" + (f" (it has source for {', '.join(have)})" if have else
                                                                         " (it replaced no stage with source)"))
        return spec["source"], spec.get("entry") or "Runtime"

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
            for tier, text in p.outputs.items():
                lines.append(f"  printed by {stage} at the {tier} tier:")
                lines += ["    " + line for line in text.splitlines()]
        return lines, ok

    def _check(self, c: Contestant, graph: Mapping[str, Proposal], note: str = "") -> str:
        self._working(c, "running the checks")
        prepared = self._prepare_all(graph)
        lines, ok = self._check_lines(prepared)
        again = self._post(c, "check", graph, prepared, ok, note)
        return (("The contract checks pass for every stage. Nothing was timed." if ok else "The checks failed; nothing was timed.")
                + "\n" + "\n".join(lines) + again)

    def _submit(self, c: Contestant, graph: Mapping[str, Proposal], note: str) -> str:
        if c.submits >= self.submits:
            return "You have no timed runs left."
        self._working(c, "running the checks")
        prepared = self._prepare_all(graph)
        lines, ok = self._check_lines(prepared)
        if not ok:
            again = self._post(c, "submit", graph, prepared, ok, note)
            return "Not timed: the checks failed (no timed run was used).\n" + "\n".join(lines) + again
        candidates = [p.candidate for p in prepared.values() if p.candidate]
        digest = self._digest(prepared)
        spec = GraphSpec(name=f"{c.name}-{digest[:8]}", host=self.world.graph_host,
                         stages={s: Node(p.proposal.impl_id, dict(p.params)) for s, p in prepared.items()},
                         plugins=tuple(x.module for x in candidates), note=f"entered by {c.name}")
        c.submits += 1
        self._working(c, f"timed run at {', '.join(map(str, self.task.worlds))} worlds: it holds the device and can take minutes")
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
        text = f"Timed and ranked ({left} timed runs left).\n{verdict}\n{tail}\nYour entries: {', '.join(ids)} (look at one, or build on it with base)."
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
        self.journal.append("submission", contestant=c.name, turn=c.turns, scene=self.task.scene, contest=self.contest,
                            graph=dict(stages=self._spec(graph)), entry_ids=ids, note=note[:400],
                            speedup=(summary or {}).get("speedup"), klass=(summary or {}).get("klass", ""))
