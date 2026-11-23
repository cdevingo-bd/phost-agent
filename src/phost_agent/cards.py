"""What an agent is told, generated from the registries so it is never out of date: a card per stage and a brief per task.

A card says what a stage is: its ports and invariants, how a swap is judged, the metrics and gates, what the implementations
that exist take, and the form a typed law takes. A brief is a card plus a task: the scene, the ceiling, what counts as a win,
the funnel the proposal goes through, the budget, and the shape of an answer. Both are plain dicts (so a proposer can take
them as JSON) and ``render_*`` gives the same thing as Markdown for a person or a model.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import phost
from phost import ImplKind

from .budget import Budget
from .ceiling import Gate
from .task import Task, World

JUDGING = {
    "exact": "must give the host's own answer to rounding: gated on the stage's own metrics and on accuracy against the native "
             "run, ranked on speedup",
    "approximate": "may differ from the host's answer by a claimed tolerance, which is enforced; ranked on speedup and "
                   "accuracy together (a Pareto front, no weighted sum)",
    "model_changing": "changes the physics, so there is no native answer to match; judged against its own float64 law, ranked on "
                      "step time among runs with the same stages",
    "observer": "must leave the trajectory alone; ranked on overhead",
}

FUNNEL = (
    ("0 ceiling", "the stage is worth the most a replacement could gain at all (asked once, before proposing)"),
    ("1 static", "the proposal is well formed: a real stage and implementation, valid parameters, or source that parses and "
                 "defines the runtime the stage calls"),
    ("2 cheap check", "the stage's contract check on a simulator-free host"),
    ("3 real check", "the same check on this scene's own host"),
    ("4 timed run", "exclusive on the device, against the host's native control, judged by phost-score"),
    ("5 ledger", "what was measured is appended to the score ledger and the journal"),
)

PROPOSAL = {
    "impl": {"stage": "<the task's stage>", "impl": "<registered implementation id>", "params": {"<name>": "<value>"},
             "note": "<why you think this helps>"},
    "source": {"stage": "<the task's stage>", "source": "<python text defining class Runtime>", "entry": "Runtime",
               "note": "<why>"},
}


def stage_card(registry: phost.Registry, stage: str) -> dict[str, Any]:
    contract = registry.contract(stage)
    form = registry.equation(stage)
    card = contract.to_dict()
    card["judging"] = JUDGING.get(contract.swap_class.value, "")
    card["equation"] = form.to_dict() if form is not None else None
    card["impls"] = [dict(i.describe(), applicable_note="see the brief's scene for where it applies")
                     for i in registry.impls_for(stage)]
    for impl in card["impls"]:
        del impl["applicable_note"]
    return card


def source_skeleton(card: Mapping[str, Any], *, list_ports: bool = True) -> str:
    """The shape of a runtime for the stage. ``list_ports=False`` leaves out the per-port comment, for a card whose table has it."""
    ports = "".join(f"\n        #   {p['name']:<14} {p['role']:<6} {p['dtype']:<8} shape {tuple(p['shape'])}" for p in card["ports"])
    if not list_ports:
        ports = " one for each port in the table above"
    entry = card["entry"] or "run"
    ref = ""
    if card["reference"] == "impl":
        ref = "\n\ndef reference(inputs, params):\n    \"\"\"Your own float64 version of what the runtime computes: outputs by port name.\"\"\"\n    ..."
    dims = {d.rstrip("+") for p in card["ports"] for d in p["shape"] if isinstance(d, str)}
    padded = any(isinstance(d, str) and d.endswith("+") for p in card["ports"] for d in p["shape"])
    notes = ("\n        # system.<port> is the array for that port: " + ", ".join(f"system.{p['name']}" for p in card["ports"])
             if list_ports else "\n        # system.<port> is the array for that port, named as in the table")
    if "nv" in dims:
        notes += ("\n        # nv is self.host.facts().nv" + (" (also .world_count). A trailing + (nv+) means at least that size: hosts "
                                                           "pad, so use the first nv entries, not the array's own size" if padded else ""))
    return (f"class Runtime:\n    def __init__(self, host, params):\n        self.host = host\n\n"
            f"    def {entry}(self, system):\n        # called once per execution of the stage; the system has these ports:{ports}{notes}\n"
            f"        ...\n\n    def metrics(self):\n        return {{}}{ref}\n")


def task_brief(task: Task, world: World, gate: Gate | None, budget: Budget | None = None) -> dict[str, Any]:
    card = stage_card(world.registry, task.stage)
    return dict(
        task=task.to_dict(),
        scene=dict(world.scene.public(), engine=world.engine_name, device_kind=world.device_kind),
        stage=card,
        ceiling=gate.to_dict() if gate is not None else None,
        win=dict(goal=task.goal, min_speedup=task.min_speedup, worlds=list(task.worlds),
                 rule="the speedup must beat min_speedup at every world count, with the contract checks and the accuracy gate "
                      "passing, on a run whose timing is reliable" if task.goal == "speed"
                 else "the candidate must pass its checks and run"),
        funnel=[dict(tier=t, what=w) for t, w in FUNNEL],
        answer=dict(shapes=PROPOSAL, source_skeleton=source_skeleton(card)),
        budget=(budget or Budget()).to_dict(),
        rules=["a candidate proposed twice is answered from the memo and still counts as an attempt",
               "a source candidate has no params; an implementation candidate's params are validated against its specs",
               "a candidate makes no network calls and runs without network access where it is sandboxed"],
    )


def render_card(card: Mapping[str, Any], *, compact: bool = False) -> str:
    """The card as Markdown. ``compact`` keeps what every candidate needs (ports, invariants, what is gated, which implementations
    exist) and leaves for a look at the stage what only matters when building on something: the implementations' parameters, the
    definition of each metric that is not gated, and the typed law."""
    judged = "" if compact else f": {card['judging']}"
    lines = [f"# stage {card['name']} (v{card['version']})", "", card["doc"], "", f"swap class **{card['swap_class']}**{judged}", ""]
    if card["entry"]:
        lines += [f"The host calls `{card['entry']}` on the runtime once per execution of the stage.", ""]
    lines += ["## ports", "", "| name | role | dtype | shape | meaning |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {p['name']} | {p['role']} | {p['dtype']} | {tuple(p['shape'])} | {p['doc']} |" for p in card["ports"]]
    if card["invariants"]:
        lines += ["", "## invariants", ""] + [f"- {i}" for i in card["invariants"]]
    gates = {t["metric"]: t["max"] for t in card["tolerances"]}
    if compact:
        gated = [m for m in card["metrics"] if m["name"] in gates]
        others = [m["name"] for m in card["metrics"] if m["name"] not in gates]
        if gated:
            lines += ["", "## gates (a runtime's `metrics()` may report these)", ""]
            lines += [f"- `{m['name']}` must stay at or below {gates[m['name']]}: {m['doc']}" for m in gated]
        if others:
            lines += ["", "Other metrics: " + ", ".join(f"`{n}`" for n in others) + "."]
    else:
        if card["metrics"]:
            lines += ["", "## metrics", ""] + [f"- `{m['name']}` ({m['unit']}, {m['better']} is better, {m['kind']}): {m['doc']}"
                                             for m in card["metrics"]]
        if card["tolerances"]:
            lines += ["", "## gates", ""] + [f"- `{t['metric']}` must stay at or below {t['max']}" for t in card["tolerances"]]
        if card["equation"]:
            lines += ["", "## typed law", "", f"The stage takes a typed law: `{card['equation']}`"]
    lines += ["", "## implementations", ""]
    for impl in card["impls"]:
        extra = f" [{impl['swap_class']}]" if impl["swap_class"] else ""
        if compact:
            lines.append(f"- `{impl['id']}` ({impl['kind']}{extra}) {impl['title']}")
            continue
        specs = ", ".join(f"{p['name']}:{p['type']}={p['default']!r}" for p in impl["params"]) or "no params"
        lines.append(f"- `{impl['id']}` ({impl['kind']}{extra}) {impl['title']}: {specs}" + (f". {impl['fit']}" if impl["fit"] else ""))
    if compact:
        lines += ["", "(Compact card. Looking at the stage pins the full one: the implementations' parameters, every metric's "
                      "definition and the typed law.)"]
    return "\n".join(lines) + "\n"


def render_brief(brief: Mapping[str, Any]) -> str:
    task, scene, gate, win = brief["task"], brief["scene"], brief["ceiling"], brief["win"]
    lines = [f"# task {task['id']}: improve `{task['stage']}` on `{task['scene']}`", "",
             f"Scene: {scene.get('description') or '(hidden)'} Engine `{scene['engine']}` on {scene['device_kind']}.", ""]
    if gate:
        figure = "unknown" if gate["ceiling"] is None else f"{gate['ceiling']:.2f}x"
        lines += [f"Ceiling: {figure}. {gate['reason']}", ""]
    lines += [f"A win: {win['rule']} (worlds {win['worlds']}, speedup above {win['min_speedup']:g}).", "",
              "Every proposal goes through, in order, and stops at the first failure:", ""]
    lines += [f"{i}. {f['tier']}: {f['what']}" for i, f in enumerate(brief["funnel"])]
    lines += ["", "Answer with one JSON object, either of these shapes:", "", "```json",
              *[json.dumps(v) for v in brief["answer"]["shapes"].values()], "```", "",
              "A source candidate's skeleton:", "", "```python", brief["answer"]["source_skeleton"].rstrip(), "```", "",
              "Budget: " + ", ".join(f"{k} {v}" for k, v in brief["budget"].items() if v is not None) + ".", ""]
    lines += [f"- {r}" for r in brief["rules"]]
    return "\n".join(lines) + "\n\n" + render_card(brief["stage"])
