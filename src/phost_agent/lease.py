"""An exclusive lease on a device, so a timed run is not a measurement of two things competing for it.

A lease is an advisory lock (``flock``) on one file per device under the agent's state directory, with a few words of who
holds it written beside. Because it is a lock the kernel holds, a holder that dies releases it: there is no stale lease to
clear by hand. It serialises every process that uses the same state directory, which is the point and also the limit: a
process outside this layer (another user's job on the same GPU) is not stopped by it; ``other_gpu_processes`` in a lab run
says when one was seen.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from pathlib import Path

from .paths import lease_dir


class LeaseTimeout(TimeoutError):
    pass


def _slug(device: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", device) or "device"


class Lease:
    """``with Lease("cuda:0").hold("run 12", timeout=60): ...`` or ``acquire``/``release``. A lease that is dropped without
    being released gives the device up when it is garbage collected (the file is closed), so keep the object while it is held."""

    def __init__(self, device: str, directory: str | Path | None = None):
        self.device = device
        self.directory = Path(directory) if directory is not None else lease_dir()
        self.path = self.directory / f"{_slug(device)}.lease"
        self._handle = None
        self._since = 0.0

    def acquire(self, holder: str = "", timeout: float | None = 60.0, poll: float = 0.1) -> "Lease":
        if self._handle is not None:
            raise RuntimeError(f"this lease on {self.device} is already held by its owner")
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    raise LeaseTimeout(f"{self.device} is held ({self.describe_holder() or 'by another process'}); "
                                       f"waited {timeout:g} s") from None
                time.sleep(poll)
        self._handle, self._since = handle, time.monotonic()
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(holder=holder, pid=os.getpid(), device=self.device,
                                     since=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
        handle.flush()
        return self

    def release(self) -> float:
        """Give the device up. Returns the seconds it was held."""
        if self._handle is None:
            return 0.0
        held = time.monotonic() - self._since
        handle, self._handle = self._handle, None
        handle.seek(0)
        handle.truncate()
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        return held

    def hold(self, holder: str = "", timeout: float | None = 60.0) -> "_Held":
        return _Held(self, holder, timeout)

    @property
    def held(self) -> bool:
        return self._handle is not None

    def holder(self) -> dict | None:
        """Who holds the device now (any process), or None when it is free. Reading takes the lock for an instant."""
        if self._handle is not None:
            return json.loads(self.path.read_text() or "null")
        if not self.path.exists():
            return None
        with open(self.path, "a+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                probe.seek(0)
                try:
                    return json.loads(probe.read() or "null") or dict(holder="", pid=None, device=self.device)
                except json.JSONDecodeError:
                    return dict(holder="", pid=None, device=self.device)
            fcntl.flock(probe, fcntl.LOCK_UN)
            return None

    def describe_holder(self) -> str:
        info = self.holder()
        if not info:
            return ""
        return f"pid {info.get('pid')} {info.get('holder') or ''}".strip()


class _Held:
    def __init__(self, lease: Lease, holder: str, timeout: float | None):
        self.lease, self.holder, self.timeout, self.seconds = lease, holder, timeout, 0.0

    def __enter__(self) -> "_Held":
        self.lease.acquire(self.holder, self.timeout)
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = self.lease.release()
