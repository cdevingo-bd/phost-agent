"""Typed forms in the arena: what a contestant is told about them, and a way to try one without spending a check.

A stage that declares an equation form (``phost.equation``) has a ``<stage>/expression`` implementation when phost-author is
loaded as a plugin: the candidate is a few lines of text in ``params``, not a source file. It is the cheap way to explore. An idea is
a few hundred characters, where a kernel is several thousand, and an idea that is wrong in a way the text shows (it blows up, it
never reaches the tolerance it claims) can be found out with no worker process, no sandbox and no check at all.

``preview`` runs a typed form on a sample and answers in a line for each variant of it. The sample is random and is not this scene's
system: a program that fails on it will fail its check, and one that does well on it has only earned the check. Nothing is timed
and nothing is recorded as an attempt. The arithmetic is phost-author's own (the checked expression language, NumPy, a bounded
number of passes on at most 64 unknowns), so no code a contestant wrote runs here.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import phost
from phost import ContractError, ImplKind

MAX_ROWS = 12            # lines in one answer: (the params and each variant) times each condition number
SIZE, CONDITION = 12, 100.0


def typed_impl(registry: phost.Registry, stage: str):
    """The stage's typed-form implementation, or None (the stage has no form, or phost-author is not loaded)."""
    if registry.equation(stage) is None:
        return None
    return next((i for i in registry.impls_for(stage) if i.kind == ImplKind.EXPRESSION), None)


def typed_stages(registry: phost.Registry, stages: Sequence[str]) -> list[str]:
    return [s for s in stages if typed_impl(registry, s) is not None]


def _vocabulary(kind: str, doc: str) -> str:
    """The functions and reductions the text may use, unless the implementation's own description already lists them."""
    if "Functions:" in doc:
        return ""
    try:
        from phost_author import expr
    except ImportError:
        return ""
    text = "Functions: " + ", ".join(sorted(expr.FUNCTIONS)) + "."
    if kind in ("iteration", "channels"):
        text += " Reductions: " + ", ".join(sorted(expr.REDUCTIONS)) + "."
    return text


def _range(spec) -> str:
    bounds = [f"{b:g}" for b in (spec.min, spec.max) if b is not None]
    return ", " + "..".join(bounds) if len(bounds) == 2 else ""


def _constants(params) -> str:
    """``a``..``h`` as one line: a run of the same default is said once (``1 for a, 0 for b..h``)."""
    runs: list[list] = []
    for p in params:
        if runs and runs[-1][0] == p.default:
            runs[-1][2] = p.name
        else:
            runs.append([p.default, p.name, p.name])
    given = ", ".join(f"{d:g} for " + (a if a == b else f"{a}..{b}") for d, a, b in runs)
    return f"- `{params[0].name}`..`{params[-1].name}` (float): constants the text can use; defaults {given}"


def typed_note(registry: phost.Registry, stage: str) -> str:
    """The typed form of a stage as Markdown: what the text may say, the parameters, and the default text to start from.

    Empty when the stage has no typed form here."""
    impl, form = typed_impl(registry, stage), registry.equation(stage)
    if impl is None or form is None:
        return ""
    lines = [f"## typed form: `{impl.id}`", "", " ".join(impl.doc.split()), "", "Parameters:"]
    constants = [p for p in impl.params if len(p.name) == 1]
    blocks: list[tuple[str, str]] = []
    for p in impl.params:
        if p in constants:
            continue
        if isinstance(p.default, str) and "\n" in p.default:
            lines.append(f"- `{p.name}` (text): {p.doc}")
            blocks.append((p.name, p.default))
        else:
            lines.append(f"- `{p.name}` ({p.type}{_range(p)}, default {p.default!r}): {p.doc}")
    if constants:
        lines.append(_constants(constants))
    vocabulary = _vocabulary(form.kind, impl.doc)
    if vocabulary:
        lines += ["", vocabulary]
    if blocks:
        lines += ["", "The defaults, a working program to start from:"]
        for name, text in blocks:
            lines += [f"  {name}:"] + [f"      {line}" for line in text.splitlines()]
    return "\n".join(lines)


# ---- preview ---------------------------------------------------------------------------------------------

def _short(value: Any) -> str:
    if isinstance(value, str):
        lines = value.count("\n") + 1
        return f"<{lines} lines>" if lines > 1 else (json.dumps(value) if len(value) <= 28 else json.dumps(value[:25] + "..."))
    return json.dumps(value, default=str)


def _changed(variant: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={_short(v)}" for k, v in variant.items()) or "no change"


def _marks(passes: int) -> list[int]:
    marks, k = [], 1
    while k < passes:
        marks.append(k)
        k *= 2
    return marks + [passes]


def _number(x: float) -> str:
    return f"{x + 0.0:.2g}"                 # + 0.0 so that -0 reads as 0


def _iteration_row(result: Mapping[str, Any], tolerance: float) -> str:
    history = result["residuals"]
    if not result["finite"]:
        reached = next((k for k, r in enumerate(history, 1) if r != r or r in (float("inf"), float("-inf"))), len(history))
        return f"not finite: the residual stops being a number at pass {reached}"
    shown = ", ".join(f"{k}: {_number(history[k - 1])}" for k in _marks(len(history)))
    hit = next((k for k, r in enumerate(history, 1) if r <= tolerance), None)
    reach = f"under the {_number(tolerance)} it claims from pass {hit}" if hit else f"never under the {_number(tolerance)} it claims"
    return (f"residual at pass {shown} (start {_number(result['start'])}); {reach}; "
            f"solution error {_number(result['solution_error'])}")


def _pointwise_row(result: Mapping[str, Any]) -> str:
    parts = []
    for name, values in result["outputs"].items():
        parts.append(f"{name} in [{_number(min(values))}, {_number(max(values))}]" if all(v == v for v in values) else f"{name} has NaN")
    damping = result.get("derivative")
    if damping:
        low, high = min(damping["damping"]), max(damping["damping"])
        parts.append(f"d/d{damping['of']} = {damping['text']} (damping {_number(low)}..{_number(high)})")
    return f"over {result['sweep']} {_number(result['x'][0])}..{_number(result['x'][-1])}: " + "; ".join(parts) + ("" if result["finite"] else "; NOT FINITE")


def _channels_row(result: Mapping[str, Any]) -> str:
    return "; ".join(f"{c['label']} = {_number(c['value'])}" for c in result["channels"]) + ("" if result["finite"] else "; NOT FINITE")


def _numbers(value: Any, name: str, count: int = 1) -> list[float]:
    """One number or a list of at most ``count`` of them, as floats."""
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    try:
        numbers = [float(x) for x in items]
    except (TypeError, ValueError):
        raise ContractError(f'"{name}" is a number' + (f" or a list of at most {count}" if count > 1 else "")) from None
    if not numbers or len(numbers) > count:
        raise ContractError(f'"{name}" is one number' + (f" or a list of at most {count}" if count > 1 else ""))
    return numbers


def preview_text(registry: phost.Registry, stages: Sequence[str], action: Mapping[str, Any]) -> str:
    """The answer to a ``preview`` action: ``{"stage", "params", "variants", "condition"}``; raises ``ContractError`` for the model to read."""
    typed = typed_stages(registry, stages)
    stage = action.get("stage")
    if stage not in typed:
        raise ContractError(f"no typed form for {stage!r}; the stages that take one are {', '.join(typed) or '(none here)'}")
    try:
        from phost_author.expression_impls import preview
    except ImportError:
        raise ContractError("a preview needs phost-author, which is not installed") from None
    impl, form = typed_impl(registry, stage), registry.equation(stage)
    base, variants = action.get("params") or {}, action.get("variants") or []
    if not isinstance(base, Mapping) or not isinstance(variants, list) or not all(isinstance(v, Mapping) for v in variants):
        raise ContractError('"params" is an object, and "variants" a list of objects of the params to change')
    iteration = form.kind == "iteration"
    conditions = _numbers(action.get("condition", CONDITION), "condition", MAX_ROWS) if iteration else [None]
    rows = [("params", {})] + [(f"variant {i}", v) for i, v in enumerate(variants, 1)]
    if len(rows) * len(conditions) > MAX_ROWS:
        raise ContractError(f"a preview answers at most {MAX_ROWS} lines (the params and each variant, for each condition number); "
                            f"this asks for {len(rows) * len(conditions)}")
    size, seed = int(_numbers(action.get("size", SIZE), "size")[0]), int(_numbers(action.get("seed", 0), "seed")[0])
    head = (f"Preview of `{impl.id}` in NumPy on a random symmetric positive definite system of {size} unknowns, seed {seed}; "
            "residual is |Ax - b| / |b|. It is not this scene's system, and nothing was checked, timed or counted:" if iteration else
            f"Preview of `{impl.id}`; nothing was checked, timed or counted:")
    lines = [head]
    for label, variant in rows:
        name = label if label == "params" or not variant else f"{label} ({_changed(variant)})"
        try:                                              # a line that is not accepted does not cost the others
            resolved = impl.resolve_params({**base, **variant})
        except ContractError as error:
            lines.append(f"- {name}: not accepted: {error}")
            continue
        constants = {c: float(resolved[c]) for c in "abcdefgh" if c in resolved}
        for condition in conditions:
            where = f", condition {_number(condition)}" if iteration and len(conditions) > 1 else ""
            try:
                if iteration:
                    result = preview(registry, stage, constants=constants, init=resolved["init"], step=resolved["step"],
                                     iterations=resolved["iterations"], size=size, condition=condition, seed=seed)
                    lines.append(f"- {name}{where}: " + _iteration_row(result, float(resolved["tolerance"])))
                else:
                    options = {k: action[k] for k in ("sweep", "at", "lo", "hi") if k in action} if form.kind == "pointwise" else {}
                    result = preview(registry, stage, resolved.get("expr", ""), constants, size=size, seed=seed, **options)
                    lines.append(f"- {name}: " + {"pointwise": _pointwise_row, "channels": _channels_row}[form.kind](result))
            except ContractError as error:
                lines.append(f"- {name}{where}: not accepted: {error}")
                break
    return "\n".join(lines)
