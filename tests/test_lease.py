import os
import subprocess
import sys
import textwrap
import time

import pytest

from phost_agent import Lease, LeaseTimeout


def test_a_lease_excludes_a_second_holder_until_released(tmp_path):
    first, second = Lease("cuda:0", tmp_path), Lease("cuda:0", tmp_path)
    first.acquire("run a")
    with pytest.raises(LeaseTimeout, match="cuda:0 is held"):
        second.acquire("run b", timeout=0.3, poll=0.05)
    assert first.release() >= 0.0
    second.acquire("run b", timeout=1.0)
    assert second.held
    second.release()


def test_a_lease_names_its_holder_to_anyone_who_asks(tmp_path):
    lease = Lease("cuda:0", tmp_path)
    assert lease.holder() is None
    with lease.hold("tier 4 of task 7"):
        seen = Lease("cuda:0", tmp_path).holder()
        assert seen["holder"] == "tier 4 of task 7" and seen["pid"] == os.getpid()
    assert Lease("cuda:0", tmp_path).holder() is None


def test_leases_on_different_devices_do_not_exclude_each_other(tmp_path):
    with Lease("cuda:0", tmp_path).hold("a"), Lease("cpu", tmp_path).hold("b", timeout=0.2):
        pass


def test_a_holder_that_dies_frees_the_device_with_no_cleanup(tmp_path):
    code = textwrap.dedent(f"""
        import sys, time
        from phost_agent import Lease
        lease = Lease("cuda:0", {str(tmp_path)!r}).acquire("doomed")      # keep it: dropping the object releases the lease
        print("held", flush=True)
        time.sleep(60)
    """)
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                             env=dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p)))
    try:
        assert child.stdout.readline().strip() == "held"
        mine = Lease("cuda:0", tmp_path)
        assert mine.holder()["holder"] == "doomed"
        with pytest.raises(LeaseTimeout):
            mine.acquire(timeout=0.2, poll=0.05)
    finally:
        child.kill()
        child.wait()
    deadline = time.monotonic() + 5
    while mine.holder() is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    mine.acquire("after", timeout=2.0)
    mine.release()


def test_holding_twice_through_one_lease_is_a_bug_and_says_so(tmp_path):
    lease = Lease("cpu", tmp_path)
    with lease.hold("x"):
        with pytest.raises(RuntimeError, match="already held"):
            lease.acquire("y")


def test_the_hold_reports_how_long_it_was_held(tmp_path):
    with Lease("cpu", tmp_path).hold("x") as held:
        time.sleep(0.05)
    assert held.seconds >= 0.05
