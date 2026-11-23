"""Edit an earlier source instead of retyping it.

A graph node can name a ``base`` (an attempt or an entry that has source) and send an ``edit`` block for the stage: one or more
hunks, each the text to find and the text to put there.

    <<<<<<< SEARCH
    old lines
    =======
    new lines
    >>>>>>> REPLACE

Each SEARCH must match one place in the source as it is at that point (hunks apply in order). Trailing whitespace is ignored when
matching. A failure says what was wrong and where the nearest text is, because a refused edit costs a turn.
"""

from __future__ import annotations

import difflib

from phost import ContractError

START, MIDDLE, END = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"


def parse_hunks(text: str) -> list[tuple[str, str]]:
    """``[(search, replace), ...]`` from an edit block. Raises ContractError for a block that is not well formed."""
    hunks: list[tuple[str, str]] = []
    search: list[str] | None = None
    replace: list[str] | None = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = line.strip()
        if marker == START and search is None:
            search, replace = [], None
        elif marker == MIDDLE and search is not None and replace is None:
            replace = []
        elif marker == END and replace is not None:
            hunks.append(("\n".join(search or []), "\n".join(replace)))
            search = replace = None
        elif replace is not None:
            replace.append(line)
        elif search is not None:
            search.append(line)
        elif marker:
            raise ContractError(f"edit block, line {number}: expected '{START}', found {marker[:40]!r}")
    if search is not None:
        raise ContractError(f"edit block: a hunk is not closed (it needs '{MIDDLE}' and then '{END}')")
    if not hunks:
        raise ContractError(f"edit block: no hunks (each is '{START}', the old text, '{MIDDLE}', the new text, '{END}')")
    return hunks


def _locate(lines: list[str], wanted: list[str]) -> list[int]:
    n = len(wanted)
    return [i for i in range(len(lines) - n + 1) if all(lines[i + j].rstrip() == wanted[j].rstrip() for j in range(n))]


def apply_hunks(source: str, hunks: list[tuple[str, str]]) -> str:
    """The source after every hunk. Raises ContractError naming the first hunk that does not match exactly one place."""
    lines = source.rstrip("\n").split("\n")
    for number, (search, replace) in enumerate(hunks, 1):
        wanted = search.split("\n")
        if not search.strip():
            raise ContractError(f"hunk {number}: the SEARCH text is empty; name the lines to replace")
        found = _locate(lines, wanted)
        if not found:
            close = difflib.get_close_matches(wanted[0].strip(), [line.strip() for line in lines], n=1, cutoff=0.5)
            near = next((i for i, line in enumerate(lines, 1) if close and line.strip() == close[0]), None)
            raise ContractError(f"hunk {number}: the SEARCH text matches nowhere in the source"
                                + (f"; the nearest line is {near}: {lines[near - 1].strip()[:80]!r}" if near else ""))
        if len(found) > 1:
            raise ContractError(f"hunk {number}: the SEARCH text matches {len(found)} places (lines "
                                + ", ".join(str(i + 1) for i in found[:5]) + "); add surrounding lines so it matches one")
        i = found[0]
        lines[i:i + len(wanted)] = replace.split("\n") if replace != "" else []
    return "\n".join(lines) + "\n"


def line_changes(old: str, new: str) -> tuple[int, int]:
    """``(lines added, lines removed)`` going from ``old`` to ``new``."""
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed
