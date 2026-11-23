"""What a candidate says while a check runs: its own ``print`` calls, returned with the check's result.

A check used to have one way back for information: its verdict. A candidate that wanted to know a shape, a library, or how long a
variant takes on the real inputs had to fail on purpose and put the answer in the exception text, which also meant the same run
could not say whether the candidate was right. Capturing what the candidate prints gives that information its own channel: it
comes back whether the check passes or fails, a check can measure and verify at once, and raising still works as before.

Only calls made from a candidate's own module are captured (``phost_agent_cand_*``). Everything else a library prints goes where
it always went.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

CANDIDATE_PREFIX = "phost_agent_cand_"
LIMIT = 4000                                   # characters returned per check; the rest is counted, not kept


class Diagnostics:
    """Collects printed text. Identical lines in a row are folded (a ``solve`` called thirty times prints thirty lines)."""

    def __init__(self, limit: int = LIMIT):
        self.limit, self.lines, self.kept, self.dropped, self._partial = limit, [], 0, 0, ""

    def write(self, text: str) -> None:
        text = self._partial + text
        *whole, self._partial = text.split("\n")
        for line in whole:
            self._line(line)

    def _line(self, line: str) -> None:
        if self.lines and self.lines[-1][0] == line:
            self.lines[-1][1] += 1
            return
        if self.kept + len(line) + 1 > self.limit:
            self.dropped += len(line) + 1
            return
        self.lines.append([line, 1])
        self.kept += len(line) + 1

    def text(self) -> str:
        if self._partial:                                           # a print with end="" and nothing after it
            self._line(self._partial)
            self._partial = ""
        out = [line + (f"  (x{count})" if count > 1 else "") for line, count in self.lines]
        if self.dropped:
            out.append(f"... {self.dropped} more characters not shown")
        return "\n".join(out)


def install(diagnostics: Diagnostics) -> Any:
    """Route a candidate's ``print`` to ``diagnostics``; returns the real ``print`` (put it back with ``builtins.print = real``)."""
    real = builtins.print

    def captured(*args, sep=" ", end="\n", file=None, flush=False):
        name = sys._getframe(1).f_globals.get("__name__", "")
        if name.startswith(CANDIDATE_PREFIX) and file in (None, sys.stdout, sys.stderr):
            diagnostics.write((" " if sep is None else sep).join(str(a) for a in args) + ("\n" if end is None else end))
            return None
        return real(*args, sep=sep, end=end, file=file, flush=flush)

    builtins.print = captured
    return real
