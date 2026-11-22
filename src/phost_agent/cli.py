"""``phost-agent``: briefs and cards for an agent, the ceiling gate, a one-candidate check, the loop, and the record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phost import ContractError

from .budget import Budget
from .cards import render_brief, render_card, stage_card, task_brief
from .ceiling import check_ceiling, recorded_gate
from .funnel import Funnel
from .journal import Journal
from .lease import Lease
from .loop import run_loop
from .proposal import Proposal
from .proposers import CommandProposer, CoordinateProposer, ScriptedProposer
from .status import render_status, summarize
from .task import Task, resolve


def _csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def _task(args) -> Task:
    return Task(scene=args.scene, stage=args.stage, worlds=args.worlds, steps=args.steps, launches=args.launches,
                repeats=args.repeats, driver=args.driver, min_ceiling=args.min_ceiling,
                min_speedup=args.min_speedup, goal=args.goal, plugins=tuple(args.plugin), scene_modules=tuple(args.scene_module),
                scene_dirs=tuple(args.scene_dir), label=args.label or "", run_timeout=args.run_timeout)


def _add_task_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scene", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--worlds", type=_csv_ints, default=(1, 8), help="comma-separated world counts a timed run measures")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--launches", type=int, default=5)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--driver", default="graph", help="how the host is stepped while timed: graph (captured where the device can), eager")
    p.add_argument("--goal", choices=("speed", "behaviour"), default="speed")
    p.add_argument("--min-ceiling", type=float, default=1.05)
    p.add_argument("--min-speedup", type=float, default=1.0)
    p.add_argument("--plugin", action="append", default=[], help="a module with register(registry); repeatable")
    p.add_argument("--scene-module", action="append", default=[])
    p.add_argument("--scene-dir", action="append", default=[])
    p.add_argument("--label", default="")
    p.add_argument("--run-timeout", type=float, default=900.0)


def _journal(args) -> Journal:
    return Journal(args.journal) if args.journal else Journal()


def cmd_card(args) -> int:
    from phost_lab.bootstrap import load_registry
    card = stage_card(load_registry(args.plugin), args.stage)
    print(json.dumps(card, indent=2) if args.json else render_card(card), end="" if not args.json else "\n")
    return 0


def cmd_brief(args) -> int:
    task, journal = _task(args), _journal(args)
    world = resolve(task)
    gate = check_ceiling(task, journal, registry=world.registry)[0] if args.measure else recorded_gate(task, journal, registry=world.registry)
    brief = task_brief(task, world, gate, _budget(args))
    print(json.dumps(brief, indent=2) if args.json else render_brief(brief), end="" if not args.json else "\n")
    return 0


def cmd_ceiling(args) -> int:
    gate, seconds = check_ceiling(_task(args), _journal(args), refresh=args.refresh)
    if args.json:
        print(json.dumps(gate.to_dict(), indent=2))
    else:
        print(f"{'pass' if gate.ok else 'REFUSED'}: {gate.reason} [{gate.basis}" + (f", {seconds:.1f} s of device time" if seconds else "") + "]")
    return 0 if gate.ok else 2


def cmd_check(args) -> int:
    task, journal = _task(args), _journal(args)
    proposal = Proposal.from_dict(json.loads(Path(args.proposal).read_text()))
    funnel = Funnel(task, journal, allow_unsandboxed=args.allow_unsandboxed, refresh=args.refresh)
    prepared = funnel.prepare(proposal, include_real=not args.cheap_only)
    for tier in ("static", "cheap", "real"):
        status = "pass" if tier in prepared.passed else ("REJECTED" if prepared.rejection and prepared.rejection[0] == tier else "not run")
        print(f"{tier:<7} {status}")
    if prepared.rejection:
        print(f"  {prepared.rejection[1]}" + (" (from the memo)" if prepared.rejection[2] else ""))
        return 2
    return 0


def _budget(args) -> Budget:
    return Budget(attempts=args.attempts, gpu_seconds=args.gpu_seconds, cpu_seconds=args.cpu_seconds, tokens=args.tokens,
                  stall=args.stall, workers=args.workers)


def _proposer(args, funnel_registry, task: Task):
    spec = args.proposer
    if spec == "coordinate":
        return CoordinateProposer(funnel_registry, task.stage, impls=args.impl or None)
    kind, _, rest = spec.partition(":")
    if kind == "script" and rest:
        return ScriptedProposer(json.loads(Path(rest).read_text()))
    if kind == "command" and rest:
        return CommandProposer(rest, timeout=args.proposer_timeout)
    raise ContractError(f"--proposer is 'coordinate', 'script:FILE' or 'command:PROGRAM', not {spec!r}")


def _what(proposal: Proposal, registry) -> str:
    """The implementation, and the parameters that differ from its defaults."""
    if proposal.impl is None:
        return f"{proposal.impl_id} (source)"
    try:
        defaults = {p.name: p.default for p in registry.impl(proposal.impl).params}
    except ContractError:
        defaults = {}
    changed = [f"{k}={v!r}" for k, v in proposal.params.items() if defaults.get(k) != v]
    return proposal.impl + (" " + " ".join(changed) if changed else "")


def cmd_run(args) -> int:
    task, journal = _task(args), _journal(args)
    world = resolve(task)
    proposer = _proposer(args, world.registry, task)

    def show(event):
        if event["event"] == "gate":
            g = event["gate"]
            print(f"ceiling: {'pass' if g['ok'] else 'REFUSED'}: {g['reason']} [{g['basis']}]")
        elif event["event"] == "attempt":
            o = event["outcome"]
            what = _what(Proposal.from_dict(o["proposal"]), world.registry)
            tail = f" {o['speedup']:.3f}x" if o["speedup"] else ""
            print(f"#{o['attempt']:<3} {what[:52]:<52} {o['verdict']:<8} at {o['reached']}{tail}{' (memo)' if o['memo'] else ''}"
                  f"{' *' if event['improved'] else ''}  {o['reason'][:70]}")
    report = run_loop(task, proposer, budget=_budget(args), journal=journal, refresh=args.refresh, on_event=show,
                      allow_unsandboxed=args.allow_unsandboxed, registry=world.registry)
    funnel = report.funnel()
    print(f"{report.status}: {report.reason}")
    print(f"funnel: proposed {funnel['proposed']} -> static {funnel['static']} -> cheap {funnel['cheap']} -> real {funnel['real']} "
          f"-> timed {funnel['run']} -> accepted {funnel['accepted']}")
    spend = report.spend
    print(f"spent: {spend.attempts} attempts, gpu {spend.gpu_seconds:.1f} s, cpu {spend.cpu_seconds:.1f} s, tokens {spend.tokens}, "
          f"{spend.memo_hits} from the memo")
    if report.best:
        print(f"best: {report.best.proposal.summary()}  {report.best.speedup:.3f}x")
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_status(args) -> int:
    summary = summarize(_journal(args), device=args.device)
    print(json.dumps(summary, indent=2) if args.json else render_status(summary))
    return 0 if summary["intact"] else 3


def cmd_lease(args) -> int:
    holder = Lease(args.device).holder()
    print(json.dumps(holder) if args.json else (f"{args.device}: held by pid {holder.get('pid')} {holder.get('holder') or ''}".strip()
                                                if holder else f"{args.device}: free"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="phost-agent", description=__doc__)
    parser.add_argument("--journal", help="the agent's journal (default: under $PHOST_HOME, see phost-score)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("card", help="what a stage is, as Markdown or JSON")
    p.add_argument("stage")
    p.add_argument("--plugin", action="append", default=[])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_card)

    def budget_arguments(q):
        q.add_argument("--attempts", type=int, default=20)
        q.add_argument("--gpu-seconds", type=float, default=600.0)
        q.add_argument("--cpu-seconds", type=float, default=None)
        q.add_argument("--tokens", type=int, default=None)
        q.add_argument("--stall", type=int, default=5)
        q.add_argument("--workers", type=int, default=1, help="candidates checked at once on the CPU tiers (needs --cpu-seconds)")

    p = sub.add_parser("brief", help="the brief an agent is given for a task")
    _add_task_arguments(p)
    budget_arguments(p)
    p.add_argument("--measure", action="store_true", help="measure the ceiling if the journal has none (uses the device)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_brief)

    p = sub.add_parser("ceiling", help="tier 0: how much could any replacement gain (measures, then remembers)")
    _add_task_arguments(p)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ceiling)

    p = sub.add_parser("check", help="put one proposal (a JSON file) through tiers 1-3")
    _add_task_arguments(p)
    p.add_argument("proposal")
    p.add_argument("--cheap-only", action="store_true")
    p.add_argument("--allow-unsandboxed", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="run the loop: propose, funnel, record, until the budget says stop")
    _add_task_arguments(p)
    budget_arguments(p)
    p.add_argument("--proposer", default="coordinate", help="coordinate | script:FILE | command:PROGRAM")
    p.add_argument("--proposer-timeout", type=float, default=300.0)
    p.add_argument("--impl", action="append", default=[], help="restrict the coordinate proposer to these implementations")
    p.add_argument("--allow-unsandboxed", action="store_true")
    p.add_argument("--refresh", action="store_true", help="ignore the memo and the recorded ceiling")
    p.add_argument("--out", help="write the full report as JSON")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="what the journal says: funnel, memo hit rate, spend, whether it is intact")
    p.add_argument("--device", help="also say who holds this device's lease")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("lease", help="who holds a device")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_lease)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (ContractError, ValueError, OSError) as error:
        print(f"phost-agent: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
