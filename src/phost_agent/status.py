"""A summary of the journal: what the agent layer has done, what it cost, and whether the record is intact."""

from __future__ import annotations

import hashlib
from typing import Any

from .arena_text import outcome_text, what_text
from .journal import Journal
from .lease import Lease


def summarize(journal: Journal, *, device: str | None = None) -> dict[str, Any]:
    verification = journal.verify()
    entries = [r.entry for r in verification.records]
    tasks = [e for e in entries if e["kind"] == "task" and e.get("phase") == "end"]
    attempts = [e for e in entries if e["kind"] == "attempt"]
    ceilings = [e for e in entries if e["kind"] == "ceiling"]
    tiers = [e for e in entries if e["kind"] == "tier"]
    reached = {"static": 0, "cheap": 0, "real": 0, "run": 0}
    for a in attempts:
        order = list(reached)
        got = -1 if a["reached"] == "none" else order.index(a["reached"])
        for i, tier in enumerate(order):
            if i < got or (i == got and a["verdict"] != "rejected"):
                reached[tier] += 1
    gpu = sum(t.get("spend", {}).get("gpu_seconds", 0.0) for t in tasks)
    cpu = sum(t.get("spend", {}).get("cpu_seconds", 0.0) for t in tasks)
    accepted = [a for a in attempts if a["verdict"] == "accepted"]
    best = max(accepted, key=lambda a: a.get("speedup") or 0.0, default=None)
    started = {e["task"]["id"]: e["task"] for e in entries if e["kind"] == "task" and e.get("phase") == "start"}
    out: dict[str, Any] = dict(
        intact=verification.ok, problems=list(verification.problems), head=str(verification.head),
        tasks=dict(finished=len(tasks), refused=sum(1 for t in tasks if t["status"] == "refused"),
                   by_status={s: sum(1 for t in tasks if t["status"] == s) for s in sorted({t["status"] for t in tasks})}),
        attempts=len(attempts), funnel=dict(proposed=len(attempts), **reached, accepted=len(accepted)),
        memo=dict(hits=sum(1 for a in attempts if a.get("memo")), rate=None, tier_results=len(tiers)),
        spend=dict(gpu_seconds=round(gpu, 3), cpu_seconds=round(cpu, 3), tokens=sum(a["proposal"].get("tokens", 0) for a in attempts)),
        ceilings=len(ceilings), best=None)
    if attempts:
        out["memo"]["rate"] = out["memo"]["hits"] / len(attempts)
    if best is not None:
        task = started.get(best["task_id"], {})
        out["best"] = dict(scene=task.get("scene"), stage=task.get("stage"), speedup=best["speedup"],
                           impl=best["proposal"].get("impl") or "source", task_id=best["task_id"])
    if device is not None:
        holder = Lease(device).holder()
        out["lease"] = dict(device=device, holder=holder)
    return out


def render_status(summary: dict[str, Any]) -> str:
    f, m, s = summary["funnel"], summary["memo"], summary["spend"]
    lines = [f"journal {'intact' if summary['intact'] else 'DAMAGED: ' + summary['problems'][0]} (head {summary['head'][:23]}...)",
             f"tasks finished {summary['tasks']['finished']} ({', '.join(f'{k} {v}' for k, v in summary['tasks']['by_status'].items()) or 'none'}), "
             f"ceilings measured {summary['ceilings']}",
             f"funnel: proposed {f['proposed']} -> static {f['static']} -> cheap {f['cheap']} -> real {f['real']} -> timed {f['run']} "
             f"-> accepted {f['accepted']}",
             "memo: " + (f"{m['hits']} of {summary['attempts']} attempts answered from it ({m['rate']:.0%})" if m["rate"] is not None
                         else "no attempts yet"),
             f"spend: gpu {s['gpu_seconds']:.1f} s, cpu {s['cpu_seconds']:.1f} s, tokens {s['tokens']}"]
    if summary.get("best"):
        lines.append(f"best: {summary['best']['speedup']:.3f}x ({summary['best']['impl']})")
    if "lease" in summary:
        holder = summary["lease"]["holder"]
        lines.append(f"lease {summary['lease']['device']}: " + (f"held by pid {holder.get('pid')} {holder.get('holder') or ''}".strip()
                                                               if holder else "free"))
    return "\n".join(lines)


def _candidate(proposal: dict) -> str:
    """A short name for what was proposed: the implementation and a few of its parameters, or the source's id."""
    if proposal.get("impl") is None:
        return "source " + hashlib.sha256(proposal.get("source", "").encode()).hexdigest()[:10]
    shown = [f"{k}={v if not isinstance(v, str) else repr(v[:18]) + ('…' if len(v) > 18 else '')}"
             for k, v in list(proposal.get("params", {}).items())[:3]]
    return proposal["impl"] + (" " + " ".join(shown) if shown else "")


def overview(journal: Journal, *, lease_dir=None, devices: tuple[str, ...] = ("cuda:0", "cpu"), limit: int = 25) -> dict[str, Any]:
    """``summarize`` plus the latest attempts and tasks and who holds each device: what a screen needs. A damaged journal still
    shows whatever parses, with ``intact`` False and the first problems."""
    out = summarize(journal)
    entries = [r.entry for r in journal.ledger.records(strict=False)]
    started = {e["task"]["id"]: e for e in entries if e["kind"] == "task" and e.get("phase") == "start"}
    out["recent"] = []
    for a in reversed([e for e in entries if e["kind"] == "attempt"][-limit:]):
        task = started.get(a["task_id"], {}).get("task", {})
        out["recent"].append(dict(task_id=a["task_id"], scene=task.get("scene"), stage=task.get("stage"), attempt=a["attempt"],
                                  candidate=_candidate(a["proposal"]), reached=a["reached"], verdict=a["verdict"],
                                  speedup=a.get("speedup"), accuracy_ratio=a.get("accuracy_ratio"), memo=bool(a.get("memo")),
                                  improved=bool(a.get("improved")), reason=a.get("reason", ""), time=a.get("time")))
    out["task_list"] = []
    for e in reversed([e for e in entries if e["kind"] == "task" and e.get("phase") == "end"][-10:]):
        start = started.get(e["task_id"], {})
        task, spend, best = start.get("task", {}), e.get("spend", {}), e.get("best") or {}
        out["task_list"].append(dict(task_id=e["task_id"], scene=task.get("scene"), stage=task.get("stage"), goal=task.get("goal"),
                                     proposer=start.get("proposer"), status=e["status"], reason=e.get("reason", ""),
                                     attempts=spend.get("attempts", 0), gpu_seconds=spend.get("gpu_seconds", 0.0),
                                     cpu_seconds=spend.get("cpu_seconds", 0.0), best_speedup=best.get("speedup"), time=e.get("time")))
    out["leases"] = {device: Lease(device, lease_dir).holder() for device in devices}
    out["problems"] = out["problems"][:5]
    return out


def arena_overview(journal: Journal, *, checks: int = 30, entries: int = 15) -> dict[str, Any] | None:
    """The latest contest in the journal (``phost-agent arena``): who is in it, what each has done so far, every check posted and
    every entry submitted, newest first. None when no contest has been started. A contest with no ``end`` record is shown as
    running: a process that was stopped leaves the same trace, so ``last_activity`` is what says how long it has been quiet."""
    records = [r.entry for r in journal.ledger.records(strict=False)]
    starts = [i for i, e in enumerate(records) if e["kind"] == "arena" and e.get("phase") == "start"]
    if not starts:
        return None
    mine = records[starts[-1]:]
    start = mine[0]
    end = next((e for e in mine if e["kind"] == "arena" and e.get("phase") == "end"), None)
    turns = [e for e in mine if e["kind"] == "arena" and e.get("phase") == "turn"]
    posted = [e for e in mine if e["kind"] == "arena" and e.get("phase") == "attempt"]
    entered = [e for e in mine if e["kind"] == "submission"]
    left = {c["name"]: c.get("left", "") for c in (end or {}).get("contestants", ())}
    players = []
    for c in start["contestants"]:
        own_turns = [t for t in turns if t["contestant"] == c["name"]]
        own_checks = [a for a in posted if a["contestant"] == c["name"]]
        own_entries = [e for e in entered if e["contestant"] == c["name"]]
        speeds = [e["speedup"] for e in own_entries if e.get("speedup") is not None]
        players.append(dict(name=c["name"], model=c.get("model", ""), turns=len(own_turns), tokens=sum(t.get("tokens", 0) for t in own_turns),
                            last_action=own_turns[-1]["action"] if own_turns else "", last_time=own_turns[-1]["time"] if own_turns else None,
                            checks=len(own_checks), passed=sum(1 for a in own_checks if a["ok"]), entries=len(own_entries),
                            best_speedup=max(speeds, default=None),
                            left=left.get(c["name"]) if left.get(c["name"]) not in (None, "still playing") else ""))
    times = [e["time"] for e in mine if e.get("time")]
    return dict(
        scene=start["task"]["scene"], worlds=start["task"].get("worlds"), started=start.get("time"), ended=end.get("time") if end else None,
        status="finished" if end else "running", last_activity=max(times, default=None), budget=start.get("budget", {}),
        contestants=players, earlier_contests=len(starts) - 1,
        checks=[dict(id=a["id"], contestant=a["contestant"], turn=a["turn"], action=a["action"], ok=a["ok"], time=a.get("time"),
                     what=what_text(a), outcome=outcome_text(a, limit=300)) for a in reversed(posted[-checks:])],
        entries=[dict(contestant=e["contestant"], speedup=e.get("speedup"), klass=e.get("klass", ""), note=(e.get("note") or "")[:240],
                      what=what_text(e), time=e.get("time")) for e in reversed(entered[-entries:])])
