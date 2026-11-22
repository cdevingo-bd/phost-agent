"""Run the check worker, in a sandbox when the candidate is source an agent wrote.

An implementation that is already registered is phost's own code, so its check runs in a plain worker process: the process
is for isolation of faults, not trust. A source proposal is code nobody reviewed. It runs under ``phost_author.sandbox``
(``bwrap``: read-only filesystem, no network, private home, CPU and file-size limits) when that package is installed and
``bwrap`` is present, and otherwise is refused unless ``allow_unsandboxed`` says the caller accepts running it as itself.

What the sandbox does not cover: tier 4, the timed run, is a lab worker (``phost_lab.isolate``), which is a separate process
and not a sandbox. A source candidate has passed the sandboxed checks by then; it still executes there with the caller's
permissions. Read ``README.md`` before pointing an agent that writes source at a machine you care about.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .worker import PREFIX


@dataclass
class CheckResult:
    status: str                              # pass | fail | skipped | error
    host: str = ""
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""                         # why skipped or errored
    seconds: float = 0.0
    sandbox: str = "none"
    transient: bool = False                  # a timeout or a missing result: about this run, not about the candidate (not memoized)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def message(self) -> str:
        if self.status == "fail":
            return "; ".join(self.failures) or "failed its contract check"
        return self.reason

    def to_dict(self) -> dict[str, Any]:
        return dict(status=self.status, host=self.host, failures=list(self.failures), metrics=dict(self.metrics),
                    reason=self.reason, seconds=round(self.seconds, 4), sandbox=self.sandbox, transient=self.transient)


def sandbox_available() -> bool:
    try:
        from phost_author import sandbox
    except ImportError:
        return False
    return sandbox.available()


def _parse(stdout: str) -> dict | None:
    lines = [line for line in stdout.splitlines() if line.startswith(PREFIX)]
    return json.loads(lines[-1][len(PREFIX):]) if lines else None


def run_check(request: Mapping[str, Any], *, untrusted: bool = False, allow_unsandboxed: bool = False,
              timeout: float = 300.0, candidate_dir: Path | None = None) -> CheckResult:
    """Run one check request (see ``worker.py``). Never raises for a failing candidate: the result says what happened."""
    request = dict(request, candidate_dir=str(candidate_dir) if candidate_dir else None)
    started = time.perf_counter()
    if untrusted and not sandbox_available() and not allow_unsandboxed:
        return CheckResult("error", reason="source proposals run in a sandbox (phost-author and bwrap) and none is available; "
                                           "pass allow_unsandboxed to run them as this user", seconds=0.0, transient=True)
    if untrusted and sandbox_available():
        from phost_author import sandbox
        policy = sandbox.Policy(wall_seconds=timeout, cpu_seconds=max(int(timeout), 1),
                                extra_read_only=(str(candidate_dir),) if candidate_dir else ())
        done = sandbox.run([sys.executable, "-s", "-m", "phost_agent.worker"], stdin=json.dumps(request), policy=policy,
                           allow_unsandboxed=allow_unsandboxed)
        stdout, stderr, code, timed_out, kind = done.stdout, done.stderr, done.returncode, done.timed_out, done.sandbox
        if done.quota_exceeded:
            return CheckResult("error", reason="wrote more than the sandbox scratch quota", sandbox=kind,
                               seconds=time.perf_counter() - started)
    else:
        kind = "none"
        try:
            done = subprocess.run([sys.executable, "-m", "phost_agent.worker"], input=json.dumps(request), capture_output=True,
                                  text=True, timeout=timeout)
            stdout, stderr, code, timed_out = done.stdout, done.stderr, done.returncode, False
        except subprocess.TimeoutExpired as error:
            stdout, stderr, code, timed_out = (error.stdout or ""), (error.stderr or ""), None, True
            stdout = stdout if isinstance(stdout, str) else stdout.decode(errors="replace")
            stderr = stderr if isinstance(stderr, str) else stderr.decode(errors="replace")
    seconds = time.perf_counter() - started
    if timed_out:
        return CheckResult("error", reason=f"timed out after {timeout:g} s", sandbox=kind, seconds=seconds, transient=True)
    payload = _parse(stdout)
    if payload is None:
        tail = stderr.strip().splitlines()[-1] if stderr.strip() else ""
        return CheckResult("error", reason=f"the worker exited with code {code} and no result" + (f": {tail}" if tail else ""),
                           sandbox=kind, seconds=seconds, transient=True)
    status = payload["status"]
    if status == "error":
        return CheckResult("error", reason=payload["error"], sandbox=kind, seconds=seconds)
    return CheckResult(status, host=payload.get("host", ""), failures=payload.get("failures", []),
                       metrics=payload.get("metrics", {}), reason=payload.get("reason", ""), sandbox=kind, seconds=seconds)
