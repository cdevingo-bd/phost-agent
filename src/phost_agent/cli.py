"""``phost-agent``: briefs and cards for an agent, the ceiling gate, a one-candidate check, the loop, and the record."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

from phost import ContractError

from .arena import Arena, ArenaTask, Contestant
from .budget import Budget
from .cards import render_brief, render_card, stage_card, task_brief
from .ceiling import check_ceiling, recorded_gate
from .funnel import Funnel
from .journal import Journal
from .lease import Lease
from .loop import run_loop
from .proposal import Proposal
from .proposers import CommandProposer, CoordinateProposer, ProposerError, ScriptedProposer
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
    if kind == "llm" and rest:
        from .llm import LLMProposer, load_plugin, make_client, parse_options
        for module in args.llm_plugin:
            load_plugin(module)
        return LLMProposer(make_client(rest, parse_options(args.llm_option)), max_tokens=args.llm_max_tokens)
    raise ContractError(f"--proposer is 'coordinate', 'script:FILE', 'command:PROGRAM' or 'llm:PROVIDER[:MODEL]', not {spec!r}")


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
    usage = getattr(proposer, "usage", None)
    if usage:
        print(f"{proposer.name}: {usage['calls']} calls, {usage['input_tokens']} tokens in ({usage['cached_tokens']} from its cache), "
              f"{usage['output_tokens']} out" + (" (estimated: this provider does not report usage)" if usage["estimated"] else ""))
    if report.best:
        print(f"best: {report.best.proposal.summary()}  {report.best.speedup:.3f}x")
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
    return 0


def _contestants(args) -> list[Contestant]:
    from .llm import LLMProposer, load_plugin, make_client, parse_options        # noqa: F401  (LLMProposer: same provider specs)
    for module in args.llm_plugin:
        load_plugin(module)
    options, out = parse_options(args.llm_option), []
    for spec in args.contestant:
        name, _, rest = spec.partition("=")
        if not rest or ":" in name or "/" in name:                # no "NAME=": the contestant is named after its provider and model
            name, rest = "", spec
        client = make_client(rest, options)
        out.append(Contestant(name or f"{client.name}:{client.model}", client))
    return out


def cmd_arena(args) -> int:
    task = ArenaTask(scene=args.scene, worlds=args.worlds, steps=args.steps, launches=args.launches, repeats=args.repeats,
                     driver=args.driver, plugins=tuple(args.plugin), scene_modules=tuple(args.scene_module),
                     scene_dirs=tuple(args.scene_dir), run_timeout=args.run_timeout)

    def show(event):
        if event["event"] == "turn":
            first = " ".join(event["text"].split())[:110]
            cost = (f"{event['input_tokens']:>6} in {event['output_tokens']:>6} out {event['seconds']:>4.0f} s "
                    if "input_tokens" in event else "")
            print(f"[{event['contestant']}] turn {event['turn']:<3} {event['action']:<14} {cost}{first}")
        elif event["event"] == "work":
            print(f"[{event['contestant']}] turn {event['turn']:<3} ...            {event['what']}")
        elif event["event"] == "left":
            print(f"[{event['contestant']}] leaves: {event['why']}")
    contestants = _contestants(args)                       # refuse a bad provider before opening anything
    dash = _start_dash(args) if args.dash else None
    arena = Arena(task, contestants, journal=_journal(args), allow_unsandboxed=args.allow_unsandboxed, turns=args.turns,
                  submits=args.submits, tokens=args.tokens, open_code=not args.closed, feed_rows=args.feed,
                  max_tokens=args.llm_max_tokens, keep_turns=args.keep_turns, runtime_facts=not args.no_facts, on_event=show)
    try:
        report = arena.run()
        print("\n" + arena.board_text())
        for c in report.contestants:
            best = f"{c['best']['speedup']:.3f}x ({c['best']['klass']})" if c["best"] else "no ranked entry"
            cached = f" ({c['cached_tokens']} from the provider's cache)" if c.get("cached_tokens") else ""
            print(f"{c['name']}: {best}; {c['turns']} turns, {c['submits']} timed runs, {c['tokens']} tokens "
                  f"(input {c['input_tokens']}{cached}, output {c['output_tokens']}), left: {c['left']}")
        if args.out:
            Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        if dash is not None:
            print(f"\nThe contest is over. The dashboard stays up at {dash.url}\nCtrl-C stops it.", flush=True)
            _hold(dash)
    finally:
        if dash is not None:
            dash.shutdown_all()
    return 0


def _start_dash(args):
    """The dashboard, on this machine's loopback, over the state the contest writes to: its Results tab is the live board, with
    the contestants' entries ranked as anyone's are. Needs phost-dash."""
    try:
        from phost_dash.app import Config
        from phost_dash.cli import serve, serve_in_thread
    except ImportError as error:
        raise ContractError(f"--dash needs phost-dash installed ({error})") from None
    from phost_score import state_dir
    dash = serve(Config(state=state_dir(), plugins=tuple(args.plugin), scene_modules=tuple(args.scene_module),
                        scene_dirs=tuple(args.scene_dir)), port=args.dash_port)
    serve_in_thread(dash)
    print(f"phost-dash is running over {state_dir()}. Open this address (it contains a one-time key; keep it to yourself):\n\n"
          f"  {dash.url}\n", flush=True)
    if not args.no_browser:
        webbrowser.open(dash.url)
    return dash


def _hold(dash) -> None:
    """Keep the dashboard up until the user stops it."""
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nstopping")


def cmd_prompt(args) -> int:
    """What a language model would be sent first for this task. Sends nothing."""
    from .llm import estimate_tokens, redact, system_prompt, turn_prompt
    task, journal = _task(args), _journal(args)
    world = resolve(task)
    brief = task_brief(task, world, recorded_gate(task, journal, registry=world.registry), _budget(args))
    system, turn = redact(system_prompt(brief)), redact(turn_prompt(brief, []))
    print("=== system (the same on every turn: providers that cache a prefix cache this) ===\n" + system)
    print("=== first turn ===\n" + turn)
    print(f"\nabout {estimate_tokens(system) + estimate_tokens(turn)} tokens in the first call "
          f"({estimate_tokens(system)} system, {estimate_tokens(turn)} turn); later turns add one line per earlier candidate.")
    return 0


def cmd_llms(args) -> int:
    from .llm import providers
    for provider in providers():
        print(f"{provider.name:<12} {provider.doc}" + (f"\n{'':<12} needs: {provider.needs}" if provider.needs else ""))
    print("\nuse one with: phost-agent run ... --proposer llm:<provider>[:<model>]   (options: --llm-option KEY=VALUE)")
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

    p = sub.add_parser("arena", help="language models compete on the leaderboard for a scene; nothing tells them how")
    p.add_argument("--scene", required=True)
    p.add_argument("--contestant", action="append", required=True, metavar="[NAME=]PROVIDER[:MODEL]",
                   help="a model to enter (see `phost-agent llms`); repeat for each rival, e.g. --contestant opus=anthropic "
                        "--contestant local=ollama:llama3")
    p.add_argument("--worlds", type=_csv_ints, default=(1, 8), help="comma-separated world counts a timed run measures")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--launches", type=int, default=5)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--driver", default="graph")
    p.add_argument("--plugin", action="append", default=[], help="a module with register(registry); repeatable")
    p.add_argument("--scene-module", action="append", default=[])
    p.add_argument("--scene-dir", action="append", default=[])
    p.add_argument("--run-timeout", type=float, default=900.0)
    p.add_argument("--turns", type=int, default=30, help="model calls each contestant may make")
    p.add_argument("--submits", type=int, default=6, help="timed runs each contestant may use (the scarce thing)")
    p.add_argument("--tokens", type=int, default=None, help="tokens each contestant may spend (sent and received)")
    p.add_argument("--closed", action="store_true", help="the board shows scores only; contestants cannot read each other's entries")
    p.add_argument("--keep-turns", type=int, default=6,
                   help="how many of its latest turns a contestant is sent word for word (the rest is the notebook; trimmed in blocks "
                        "of 3, so 60 keeps everything)")
    p.add_argument("--no-facts", action="store_true",
                   help="do not run the probe that measures what a check hands a runtime (shapes, dtypes, installed libraries)")
    p.add_argument("--feed", type=int, default=10, help="how many of the others' latest checks each turn shows (0: none; open contests only)")
    p.add_argument("--llm-option", action="append", default=[], metavar="KEY=VALUE", help="a provider option, applied to every contestant")
    p.add_argument("--llm-max-tokens", type=int, default=16000)
    p.add_argument("--llm-plugin", action="append", default=[])
    p.add_argument("--allow-unsandboxed", action="store_true")
    p.add_argument("--dash", action="store_true", help="also run phost-dash on this machine's loopback over the same state, print its "
                                                       "link, and keep it up after the contest ends (Ctrl-C stops it)")
    p.add_argument("--dash-port", type=int, default=0, help="port for --dash (default: any free port)")
    p.add_argument("--no-browser", action="store_true", help="with --dash: print the link and do not open a browser")
    p.add_argument("--out", help="write the final report as JSON")
    p.set_defaults(fn=cmd_arena)

    p = sub.add_parser("prompt", help="what a language model would be sent first for a task (sends nothing)")
    _add_task_arguments(p)
    budget_arguments(p)
    p.set_defaults(fn=cmd_prompt)

    p = sub.add_parser("llms", help="the language-model providers there are, and what each needs")
    p.set_defaults(fn=cmd_llms)

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
    p.add_argument("--proposer", default="coordinate",
                   help="coordinate | script:FILE | command:PROGRAM | llm:PROVIDER[:MODEL] (see `phost-agent llms`)")
    p.add_argument("--proposer-timeout", type=float, default=300.0)
    p.add_argument("--llm-option", action="append", default=[], metavar="KEY=VALUE",
                   help="a provider's own option (effort=low, base_url=..., body.temperature=0.2); repeatable")
    p.add_argument("--llm-max-tokens", type=int, default=16000, help="the longest reply a model may give (thinking counts against it)")
    p.add_argument("--llm-plugin", action="append", default=[], help="a module with register_llm(register_provider); repeatable")
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
    except (ContractError, ProposerError, ValueError, OSError) as error:
        print(f"phost-agent: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
