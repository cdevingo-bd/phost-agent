"""A check worker: build a host, install one candidate on it, run the stage's contract check, print the verdict.

Reads one JSON request on stdin and prints one result line (``PHOST_AGENT_RESULT <json>``). It runs in its own process so a
candidate that crashes, hangs or leaks a device context cannot take the loop with it, and, for a source proposal, inside
the sandbox ``phost_author`` provides when there is one (see ``checker.py``).

``tier`` says which host:

    cheap   the cheapest simulator-free host that offers the stage: phost's fake host, else the NumPy chain host. It needs
            no device, so many of these can run at once.
    real    the scene's own host, built by the scene's engine (MuJoCo Warp, the chain, ...): the engine says how to
            get one a contract check can drive, so nothing here knows which simulator it is.
"""

from __future__ import annotations

import contextlib
import json
import sys
import traceback

from .diagnostics import Diagnostics, install

PREFIX = "PHOST_AGENT_RESULT "


def cheap_host(stage: str):
    """``(host, name)`` for the first simulator-free host that offers ``stage``, or None."""
    from phost.hosts.chain import ChainHost, make_model
    from phost.testing.fake import FakeHost

    for name, build in (("fake", lambda: FakeHost()), ("chain", lambda: ChainHost(make_model(24, 3), worlds=2))):
        host = build()
        if stage in host.stages():
            return host, name
    return None


@contextlib.contextmanager
def real_host(scene, worlds: int):
    """The scene's own host, built by the scene's engine (MuJoCo Warp, the chain, ...), driveable by a contract check and with the
    engine's device open for the length of the block. Yields ``(host, engine name)``. Nothing here names a simulator."""
    from phost_lab.engine import check_host, engine_for

    engine = engine_for(scene)
    with engine.load(scene) as loaded:
        yield check_host(loaded, worlds), engine.name


def main() -> int:
    request = json.loads(sys.stdin.read())
    diagnostics = Diagnostics()
    install(diagnostics)                 # what the candidate prints comes back with the result, pass or fail

    def emit(**payload) -> None:
        printed = diagnostics.text()
        print(PREFIX + json.dumps(dict(payload, **(dict(output=printed) if printed else {}))), flush=True)

    try:
        if request.get("candidate_dir"):
            sys.path.insert(0, request["candidate_dir"])
        import phost
        from phost import Selection
        from phost.testing.contract import check_impl
        from phost_lab.bootstrap import load_registry, load_scenes

        registry = load_registry(request.get("plugins", ()))
        stage, impl_id = request["stage"], request["impl"]
        selection = Selection(stage, impl_id, request.get("params", {}))
        try:
            if request["tier"] == "cheap":
                built = cheap_host(stage)
                if built is None:
                    emit(status="skipped", reason="no simulator-free host offers this stage")
                    return 0
                host, host_name = built
                verdict = check_impl(host, selection, registry)
            else:
                scene = load_scenes(request.get("scene_modules", ()), request.get("scene_dirs", ())).get(request["scene"])
                with real_host(scene, int(request.get("worlds", 2))) as (host, host_name):
                    verdict = check_impl(host, selection, registry)
        except phost.ApplicabilityError as error:
            if request["tier"] == "cheap":
                emit(status="skipped", host=host_name, reason=f"does not apply on the {host_name} host: {error}")
                return 0
            raise
        emit(status="pass" if verdict.ok else "fail", host=host_name, failures=list(verdict.failures),
             metrics={k: float(v) for k, v in verdict.metrics.items()})
        return 0
    except BaseException as error:
        emit(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-3000:])
        return 1


if __name__ == "__main__":
    sys.exit(main())
