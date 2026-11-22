"""A summary of the journal: what the agent layer has done, what it cost, and whether the record is intact."""

from __future__ import annotations

import hashlib
from typing import Any

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
