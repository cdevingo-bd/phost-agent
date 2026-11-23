"""What a contestant's prompt is made of, and what is kept of its past: state is shown once, history is a short window.

A turn used to send the whole conversation again, and each earlier turn carried its own copy of the board and the feed, every
source the model had typed and every card it had read. That is most of what was sent: the prompt grew by thousands of tokens a
turn and the cost of a contest grew with the square of its length. The prompt is now built from parts that each have one job:

  pinned     the cards the contestant is working from, once, at the top; changes only when a card is added
  window     the last few exchanges, trimmed in blocks so that the start of the prompt stays the same for several turns in a row
             (a provider that caches the start of a prompt can serve it); the newest two are whole, and in the older ones a long
             result is cut and the code the contestant wrote is replaced by the id it is kept under (``condense``)
  notebook   one line for each earlier check and timed run that is no longer in the window: id, what, outcome, the contestant's
             own note, and the first line it printed; compiled from the arena's records, so nothing is asked of the model and
             nothing is lost: any id in it can be looked at or built on
  state      what is new on the board and in the others' checks since the contestant's last move; only in the newest message

Nothing here chooses what is tried. It decides what is repeated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .arena_text import outcome_text, what_text

SLACK = 3                  # exchanges kept beyond the window before it is trimmed back (so the prompt's start is stable for SLACK + 1 turns)
NOTEBOOK_LINES = 24        # lines shown in full; older ones are listed by id
WHOLE_TURNS = 2            # the newest exchanges sent as they were; older ones in the window are condensed
OLD_RESULT = 600           # characters of an older result that are repeated (the start: what happened and why)
CODE_LINES = 6             # a code block of at least this many lines is replaced in an older reply
_FENCE = re.compile(r"```(python|py|edit)[ \t]*([\w.\-]*)[ \t]*\r?\n(.*?)```", re.S | re.I)


@dataclass
class Exchange:
    """One turn: what the contestant was told (the result of its last turn) and what it answered."""

    turn: int
    user: str
    assistant: str


def trim(exchanges: Sequence[Exchange], keep: int) -> list[Exchange]:
    """The window to keep: all of them until there are more than ``keep + SLACK``, then the newest ``keep``."""
    return list(exchanges) if len(exchanges) <= keep + SLACK else list(exchanges[-keep:])


def condense(exchanges: Sequence[Exchange], *, whole: int = WHOLE_TURNS,
             kept: Mapping[int, Sequence[tuple[str, str]]] | None = None) -> list[Exchange]:
    """The window as it is sent: the newest ``whole`` exchanges as they were, the older ones shortened.

    A result longer than ``OLD_RESULT`` characters is cut after that many (what a look returned or why a check failed was read when it
    arrived; the id is still in its first line, so it can be looked at again). A python or edit block in an older reply is replaced by
    the ids the arena kept that turn's source under (``kept``: turn to ``(kind, id)`` pairs), which can be looked at or built on.
    Nothing is removed from the record: this is only what is repeated."""
    kept = kept or {}
    cut = max(0, len(exchanges) - max(whole, 0))
    return [ex if i >= cut else Exchange(ex.turn, _cut_result(ex.user), _cut_code(ex.assistant, kept.get(ex.turn, ())))
            for i, ex in enumerate(exchanges)]


def _cut_result(text: str) -> str:
    if len(text) <= OLD_RESULT:
        return text
    return (text[:OLD_RESULT].rstrip() + f"\n[{len(text) - OLD_RESULT} more characters of this result are not repeated here; "
                                        f"look at its id again to read all of it]")


def _cut_code(text: str, kept: Sequence[tuple[str, str]]) -> str:
    where = ("kept as " + ", ".join(f"{kind} {ident}" for kind, ident in kept) + "; look at it or build on it with base") if kept else "not kept"

    def one(match: re.Match) -> str:
        lines = match.group(3).count("\n")
        if lines < CODE_LINES:
            return match.group(0)
        tag = f" {match.group(2)}" if match.group(2) else ""
        return f"[{match.group(1).lower()}{tag}: {lines} lines of source, {where}]"

    return _FENCE.sub(one, text)


def assemble(pinned: str, exchanges: Sequence[Exchange], newest: str) -> list[dict[str, str]]:
    """The messages for one call: the window, then the newest user message; the pinned text leads the first of them."""
    messages: list[dict[str, str]] = []
    for ex in exchanges:
        messages += [dict(role="user", content=ex.user), dict(role="assistant", content=ex.assistant)]
    messages.append(dict(role="user", content=newest))
    if pinned:
        messages[0] = dict(role="user", content=pinned + "\n\n" + messages[0]["content"])
    return messages


def pinned_text(cards: Mapping[str, str]) -> str:
    if not cards:
        return ""
    return ("Pinned for the whole contest (the stages whose speed is ranked, and any card you look at; they are not repeated "
            "below):\n\n" + "\n\n".join(cards.values()))


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def notebook_text(*, attempts: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], reads: Sequence[tuple[int, str]],
                  best: Mapping[str, Any] | None, before: int | None) -> str:
    """The contestant's own earlier work, one line each, for the turns before ``before`` (the first turn still in the window).

    ``attempts`` are its checks (and refused submits) from the arena's attempt records, ``runs`` its timed submissions,
    ``reads`` what it looked at as ``(turn, label)``. Empty when there is nothing older than the window."""
    if before is None:
        return ""
    items: list[tuple[int, str, str]] = []                                  # (turn, id, line)
    for r in attempts:
        if r["turn"] < before:
            note = f"; note: {_short(r['note'], 120)}" if r.get("note") else ""
            items.append((r["turn"], r["id"], f"- {r['id']} turn {r['turn']} {r['action']} [{what_text(r)}]: "
                                              f"{outcome_text(r, limit=110)}{note}"))
    for r in runs:
        if r["turn"] < before:
            ids = r.get("entry_ids") or ["?"]
            speed = f"{r['speedup']:.3g}x" if r.get("speedup") is not None else "no speedup figure"
            note = f"; note: {_short(r['note'], 120)}" if r.get("note") else ""
            items.append((r["turn"], ids[-1], f"- {'/'.join(ids)} turn {r['turn']} timed [{what_text(r)}]: {speed} ({r.get('klass') or '?'}){note}"))
    items.sort(key=lambda item: item[0])
    seen = [f"{label}" for turn, label in reads if turn < before]
    if not items and not seen:
        return ""
    lines = ["Notebook (your own earlier work, compiled from the record; it is not in the conversation above; look at an id for all of it, "
             "or build on one with base):"]
    older, shown = items[:-NOTEBOOK_LINES], items[-NOTEBOOK_LINES:]
    if older:
        lines.append("  older, ids only: " + ", ".join(f"{i} (turn {t})" for t, i, _ in older))
    lines += [line for _, _, line in shown]
    if best:
        lines.append(f"Your best timed entry so far: {best['id']}, {best['speedup']:.3g}x at its slowest world count ({best['klass']}).")
    if seen:
        lines.append("Already read: " + ", ".join(dict.fromkeys(seen)))
    return "\n".join(lines)
