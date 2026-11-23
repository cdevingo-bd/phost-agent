"""What a contestant is told, how a board reads, and how an action is read. Facts and rules, never advice on what to try."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import phost

from .cards import JUDGING, stage_card
from .diagnostics import LIMIT

INTRO = """\
You are one of several language models competing on a leaderboard. Everyone is given the same simulator scene. The board ranks
every entry by how much faster it makes the whole simulation step than the host's own, while the trajectory stays right.
What to try is yours to decide: replace any stage, with an implementation that exists or with code you write, replace several
stages at once, start from a rival's entry on the board, or do something else the rules allow. Nobody here tells you how.
"""

RULES = """\
How entries are judged (the rules, not advice):
- An entry is a graph: for each stage it replaces, either a registered implementation with parameters or the source of a Python
  runtime class. Stages it does not name stay the host's own.
- Each replaced stage is run through that stage's contract checks against a float64 reference before anything is timed. `check`
  runs only these, and costs a turn, not a timed run. Whatever your runtime `print`s while they run comes back with the result,
  passed or failed (the first {printed} characters; identical lines in a row are folded).
- A timed run steps the scene with your graph and with the host's own graph from the same start state, at several world counts,
  and measures the speed ratio and how far your trajectory drifts from the host's. Timed runs have the device to themselves and
  are the scarce thing: you have a fixed number.
- Entries are ranked only against comparable entries: same scene, device and world count, and the same swap class. By class:
{classes}
- A run whose timing is flagged unreliable is not ranked; submit it again. Only the latest run of the same graph under your name
  counts, so repeating an entry only re-measures it.
- {visibility}
"""

ACTIONS = """\
Reply with one JSON object per turn:

  {"action": "look", "what": "card", "stage": "<stage>"}       a stage's full card: ports, invariants, implementations, parameters
  {"action": "look", "what": "entry", "id": "<board id>"}      what a board entry submitted: its stages, parameters and source
  {"action": "look", "what": "attempt", "id": "<attempt id>"}  a check anyone ran: its source and the whole reason it failed (open contests)
  {"action": "check", "graph": GRAPH}                          the contract checks only
  {"action": "submit", "graph": GRAPH, "note": "<why>"}        check, time and rank it: uses one of your timed runs

  GRAPH = {"stages": {"<stage>": {"impl": "<registered id>", "params": {"<name>": <value>}}  or  {"entry": "Runtime"}}}

Source for a stage goes in a ```python <stage> block after the JSON (the stage's name after "python"), not inside the JSON;
`"source": "..."` inside the stage's object also works. The runtime class is the whole candidate: see the stage's card.
"""


def classes_text(registry: phost.Registry, stages: Sequence[str]) -> str:
    present = {registry.contract(s).swap_class.value for s in stages}
    return "\n".join(f"    {k}: {v}" for k, v in JUDGING.items() if k in present or not present)


def _first_sentence(text: str, limit: int = 170) -> str:
    text = " ".join(str(text).split())
    cut = text.find(". ")
    text = text[: cut + 1] if 0 < cut < limit else text
    return text if len(text) <= limit else text[: limit - 1] + "…"


def stage_index(registry: phost.Registry, stages: Sequence[str]) -> str:
    lines = []
    for name in stages:
        card = stage_card(registry, name)
        impls = [i["id"] for i in card["impls"] if i.get("kind") != "native"]
        lines.append(f"- {name} [{card['swap_class']}]: {_first_sentence(card['doc'])}"
                     + (f" Implementations: {', '.join(impls)}." if impls else " No implementations beyond the host's own."))
    return "\n".join(lines)


def facts_text(report: Mapping[str, Any] | None, scene: str, worlds: int) -> str:
    """Where the host's step spends its time on this scene: a measurement, offered as information."""
    if not report or "error" in report:
        why = (report or {}).get("error", "no measurement")
        return f"Where the time goes on {scene}: not measured ({why})."
    cells = [c for c in report.get("cells", ()) if c.get("stages")]
    if not cells:
        return f"Where the time goes on {scene}: nothing measured."
    cell = next((c for c in cells if c.get("worlds") == worlds), cells[-1])
    lines = [f"Where the host's own step spends its time on {scene} ({cell.get('worlds')} worlds, {cell.get('step_ms', 0):.3g} ms a step):"]
    rows = sorted(cell["stages"].items(), key=lambda kv: -(kv[1].get("share") or 0.0))
    for stage, info in rows:
        if not info.get("applies"):
            lines.append(f"- {stage}: does not apply here ({info.get('reason') or 'no reason given'})")
        else:
            ceiling = info.get("ceiling")
            lines.append(f"- {stage}: {100 * (info.get('share') or 0):.1f}% of the step"
                         + (f"; replacing it with something free would give at most {ceiling:.2f}x" if ceiling else ""))
    return "\n".join(lines)


def system_prompt(*, registry: phost.Registry, stages: Sequence[str], facts: str, scene: str, worlds: Sequence[int],
                  turns: int, submits: int, open_code: bool, feed: bool = True) -> str:
    visibility = ("Everything you submit, source included, is visible to the other contestants, as theirs is to you." if open_code
                  else "Entries show on the board with what they scored, but not what they submitted.")
    if open_code and feed:
        visibility += (" So is every check anyone runs, passed or failed, with what it printed: each turn you are shown the others' "
                       "latest ones under the board, and you can look at any by its id.")
    rules = RULES.format(classes=classes_text(registry, stages), visibility=visibility, printed=LIMIT)
    return (INTRO + f"\nScene: {scene}. World counts timed: {', '.join(map(str, worlds))}; the board you see is for {max(worlds)} worlds.\n"
            f"Your budget: {turns} turns and {submits} timed runs. The contest ends for you when either is used up; "
            f"there is no other way to leave.\n\n" + facts + "\n\nStages you can replace:\n"
            + stage_index(registry, stages) + "\n\n" + rules + "\n" + ACTIONS)


def graph_summary(graph: Mapping[str, Any], limit: int = 90) -> str:
    stages = graph.get("stages") or {}
    parts = []
    for stage, node in stages.items():
        node = node if isinstance(node, Mapping) else {}
        impl = str(node.get("impl", "?"))
        params = node.get("params") or {}
        parts.append(f"{stage}={impl.split('/', 1)[-1]}" + (json.dumps(params, separators=(",", ":"), default=str) if params else ""))
    text = ", ".join(parts) or "(host's own)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_board(groups: Sequence[Mapping[str, Any]], *, scene: str, worlds: int, rows: int = 8, you: str | None = None) -> str:
    groups = [g for g in groups if g["rows"]]
    if not groups:
        return f"Board for {scene} at {worlds} worlds: no entries yet."
    lines = [f"Board for {scene} at {worlds} worlds (speedup over the host's own step):"]
    for group in groups:
        names = ", ".join(f"{o['name']} ({'higher' if o['sense'] == 'max' else 'lower'} is better)" for o in group["objectives"])
        stages = f", only against entries replacing {'+'.join(group['stages'])}" if group["stages"] else ""
        lines.append(f"{group['klass']} on {group['device']}: ranked on {names}{stages}")
        for position, row in enumerate(group["rows"][:rows], 1):
            marks = " ".join(f"{k} {v:.3g}" + ("x" if k == "speedup" else "") for k, v in row["objectives"].items())
            front = f" (front {row['rank']})" if len(group["objectives"]) > 1 else ""
            who = row["label"] + (" (you)" if you and row["label"] == you else "")
            lines.append(f"  {position}. {who}: {marks}{front}  {graph_summary(row['graph'])}  id {row['id']}")
        if len(group["rows"]) > rows:
            lines.append(f"  ... {len(group['rows']) - rows} more")
    return "\n".join(lines)


_PY = re.compile(r"```(?:python|py)[ \t]*([\w.\-]*)[ \t]*\r?\n(.*?)```", re.S | re.I)


def parse_action(text: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """``(the JSON action, [(stage tag, source), ...])`` from a reply. Raises ValueError, whose message goes back to the model."""
    if not text or not text.strip():
        raise ValueError("the reply was empty")
    blocks = [(tag, code.strip("\n") + "\n") for tag, code in _PY.findall(text)]
    body = _PY.sub("", text)
    decoder, i = json.JSONDecoder(), body.find("{")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(body[i:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "action" in obj:
            return obj, blocks
        i = body.find("{", i + 1)
    raise ValueError('no JSON object with an "action" was found')


def what_text(record: Mapping[str, Any]) -> str:
    """The stages an attempt or entry replaced, each with its implementation or the size of its source."""
    parts = []
    for stage, spec in record["graph"]["stages"].items():
        source = spec.get("source")
        parts.append(f"{stage} (source, {len(source.splitlines())} lines)" if source else f"{stage} ({spec.get('impl')})")
    return ", ".join(parts)


def outcome_text(record: Mapping[str, Any], limit: int | None = 150) -> str:
    """Whether a check passed, or the first stage's reason it did not (cut to ``limit`` characters; None for all of it)."""
    if record["ok"]:
        text = "passed the checks"
    else:
        first = record["reasons"][0]
        reason = " ".join(str(first["reason"]).split())
        text = f"FAILED at the {first['tier']} tier: " + (reason if limit is None or len(reason) <= limit else reason[:limit] + "...")
    printed = next((" ".join(o["text"].split()) for o in record.get("output", ()) if o["text"].strip()), "")
    if printed and limit is not None:                           # a stub: `look` at the attempt has all of it
        text += "; its runtime printed: " + (printed if len(printed) <= 80 else printed[:80] + "...")
    return text
