"""What a model is told, and how its answer is read. Provider-neutral: plain text in, one JSON object out.

The system prompt is the instructions plus the rendered brief, and does not change during a search, so a provider that caches a
prefix pays for it once. Each turn adds only the results so far, kept short: one line per earlier candidate, and the source of
the latest source candidate (so a model can edit its own code instead of rewriting it from memory).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ..cards import render_brief
from ..outcome import ACCEPTED, REJECTED, Outcome

INSTRUCTIONS = """\
You are the proposer in an automated search. Each turn you propose ONE candidate implementation of the stage described below.
A harness then checks it, times it against the host's native version, and tells you how it fared in the next turn. You run
nothing yourself, so reason from the stage card, the ceiling and the results you are shown.

Reply with one JSON object and nothing else (a source block excepted, where noted):

  {"proposal": {"stage": "<the stage>", "impl": "<a registered implementation id>", "params": {"<name>": <value>}, "note": "<one sentence: why>"}}
      a registered implementation, with parameters (the implementations and their parameters are listed below)
  {"proposal": {"stage": "<the stage>", "entry": "Runtime", "note": "<one sentence: why>"}}
      new code: the complete Python source of the runtime class goes in ONE ```python block after the JSON, not inside it
  {"stop": "<why no further attempt is worth making>"}

How to search well:
- An attempt that fails the checks costs an attempt and teaches little. One answered from the memo (the same implementation
  with the same parameters) costs an attempt and teaches nothing. Spend attempts on candidates that could win.
- Use the results. If a change helped, go further the same way. If it hurt, try the other way or another implementation.
  Change one thing at a time unless you have a reason not to.
- The ceiling is the most the stage can give. Do not expect more than it.
- The native implementation is what every candidate is compared against. Proposing it gains nothing.
- Stop when the ideas left are unlikely to beat the best result so far.
- Keep the note to one sentence. Do not repeat the brief back.
"""


def system_prompt(brief: Mapping[str, Any]) -> str:
    return INSTRUCTIONS + "\n" + render_brief(brief)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def outcome_line(o: Outcome, *, reason_chars: int = 200) -> str:
    head = f"#{o.attempt} {o.proposal.summary()}"
    if o.verdict == REJECTED:
        where = "before any check" if o.reached == "none" else f"at the {o.reached} tier"
        tail = f"rejected {where}: {_one_line(o.reason, reason_chars)}"
    else:
        parts = [f"{o.verdict}"]
        if o.speedup is not None:
            parts.append(f"{o.speedup:.3g}x of native")
        if o.accuracy_ratio is not None:
            parts.append(f"accuracy ratio {o.accuracy_ratio:.3g}")
        tail = ", ".join(parts) + (f" ({_one_line(o.reason, reason_chars)})" if o.reason else "")
    return f"{head} -> {tail}" + (" [a repeat, answered from the memo]" if o.memo else "")


def turn_prompt(brief: Mapping[str, Any], history: Sequence[Outcome], *, history_limit: int = 40, source_chars: int = 6000) -> str:
    total = (brief.get("budget") or {}).get("attempts")
    lines = [f"Attempt {len(history) + 1}" + (f" of {total}." if total else ".")]
    if not history:
        lines.append("No attempts yet.")
    else:
        shown = list(history)[-history_limit:]
        lines.append("Results so far:")
        if len(shown) < len(history):
            lines.append(f"({len(history) - len(shown)} earlier results left out)")
        lines += [outcome_line(o) for o in shown]
        accepted = [o for o in history if o.verdict == ACCEPTED]
        if accepted:
            best = max(accepted, key=lambda o: o.speedup or 0.0)
            lines.append(f"Best accepted so far: #{best.attempt} at {best.speedup:.3g}x of native.")
        else:
            lines.append("Nothing has been accepted yet.")
        sourced = [o for o in history if o.proposal.source is not None]
        if sourced:
            last = sourced[-1]
            code = last.proposal.source
            if len(code) > source_chars:
                code = code[:source_chars] + "\n# ... (cut)"
            lines += ["", f"The source of your latest source candidate (#{last.attempt}):", "```python", code.rstrip(), "```"]
    lines += ["", "Propose the next candidate, or stop."]
    return "\n".join(lines)


_PY = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.S | re.I)


def _first_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and ({"proposal", "stop"} & obj.keys() or "stage" in obj):
            return obj
        i = text.find("{", i + 1)
    return None


def parse_reply(text: str) -> dict[str, Any]:
    """``{"stop": why}`` or ``{"proposal": {...}}`` from a model's reply. The JSON may be bare or in a code fence; source may be
    inside it or in a ```python block after it. Anything else raises ValueError, whose message goes back to the model."""
    if not text or not text.strip():
        raise ValueError("the reply was empty")
    blocks = [b.strip("\n") for b in _PY.findall(text)]
    obj = _first_object(_PY.sub("", text))
    if obj is None:
        raise ValueError("no JSON object with a proposal or a stop was found")
    if obj.get("stop"):
        return {"stop": str(obj["stop"])}
    proposal = obj.get("proposal", obj if "stage" in obj else None)
    if not isinstance(proposal, dict):
        raise ValueError('the JSON object has neither "proposal" nor "stop"')
    proposal = {k: v for k, v in proposal.items() if k != "tokens"}          # what a proposal cost is measured, not claimed
    if proposal.get("impl") is None and proposal.get("source") is None and blocks:
        proposal["source"] = blocks[0] + "\n"
    return {"proposal": proposal}
