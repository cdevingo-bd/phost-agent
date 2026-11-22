"""An agent layer for phost: what an agent is told, what a proposal must pass before it earns a timed run, and what a
search may spend. It calls no model; a proposer is anything that returns the next candidate."""

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
from .proposers import CommandProposer, CoordinateProposer, Proposer, ProposerError, ScriptedProposer
from .status import overview, render_status, summarize
from .task import Task, World, resolve

__version__ = "0.1.0.dev0"
__all__ = ["Budget", "Spend", "exhausted", "stage_card", "task_brief", "render_card", "render_brief", "Gate", "check_ceiling",
           "gate_from_report", "recorded_gate", "CheckResult", "run_check", "Funnel", "Prepared", "Journal", "Lease",
           "LeaseTimeout", "Report", "run_loop", "Memo", "key_for", "Outcome", "ACCEPTED", "MEASURED", "REJECTED", "Proposal",
           "Candidate", "write_candidate", "Proposer", "ProposerError", "ScriptedProposer", "CoordinateProposer",
           "CommandProposer", "summarize", "overview", "render_status", "Task", "World", "resolve"]
