"""An agent layer for phost: an arena where language models compete on a leaderboard, and the funnel, memo, lease and budgets
around it. What an agent is told, what a proposal must pass before it earns a timed run, and what a search may spend. A proposer is anything that returns the next candidate: a script, a baseline search, a program of yours, or a
language model behind any provider (``phost_agent.llm``)."""

from .arena import Arena, ArenaReport, ArenaTask, Contestant
from .budget import Budget, Spend, exhausted
from .cards import render_brief, render_card, stage_card, task_brief
from .ceiling import Gate, check_ceiling, gate_from_report, recorded_gate
from .checker import CheckResult, run_check
from .funnel import Funnel, Prepared
from .journal import Journal
from .lease import Lease, LeaseTimeout
from .loop import Report, run_loop
from .memo import Memo, key_for
from .outcome import ACCEPTED, MEASURED, REJECTED, Outcome
from .proposal import Candidate, Proposal, write_candidate
from .proposers import CommandProposer, CoordinateProposer, Proposer, ProposerError, ProposerFatal, ScriptedProposer
from .status import arena_overview, overview, render_status, summarize
from .task import Task, World, resolve

__version__ = "0.1.0.dev0"
__all__ = ["Arena", "ArenaReport", "ArenaTask", "Contestant", "Budget", "Spend", "exhausted", "stage_card", "task_brief", "render_card", "render_brief", "Gate", "check_ceiling",
           "gate_from_report", "recorded_gate", "CheckResult", "run_check", "Funnel", "Prepared", "Journal", "Lease",
           "LeaseTimeout", "Report", "run_loop", "Memo", "key_for", "Outcome", "ACCEPTED", "MEASURED", "REJECTED", "Proposal",
           "Candidate", "write_candidate", "Proposer", "ProposerError", "ProposerFatal", "ScriptedProposer", "CoordinateProposer",
           "CommandProposer", "summarize", "overview", "arena_overview", "render_status", "Task", "World", "resolve"]
