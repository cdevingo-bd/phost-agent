"""What a check's inputs are, measured once: shapes, dtypes and layout at the real tier, the host's facts, what the sandbox has.

Every contestant used to find this out for itself, by failing a check on purpose, one turn at a time and with the whole conversation
sent each time. A probe is a runtime generated from the stage's card that prints what it is given and stops; it runs through the
same check worker, in the same sandbox, as a contestant's runtime, and what it printed comes back through the diagnostics channel.
The text is kept in the journal under a key made from the scene, the stage and the card's skeleton, so a later contest on the same
scene reads it instead of measuring it again. It is information about the inputs. It says nothing about what to do with them.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

LIBRARIES = ("numpy", "scipy", "numba", "jax", "torch", "warp", "cupy")

_TEMPLATE = '''"""A probe written by phost-agent: prints what a check gives a runtime, then stops."""
import importlib.util
import os
import sys

PORTS = {ports!r}
LIBRARIES = {libraries!r}


def _describe(v):
    if v is None:
        return "missing"
    bits = [type(v).__module__ + "." + type(v).__name__]
    for attr in ("dtype", "shape", "strides"):
        if hasattr(v, attr):
            try:
                bits.append(attr + "=" + str(getattr(v, attr)))
            except Exception:
                pass
    flags = getattr(v, "flags", None)
    if flags is not None and hasattr(flags, "c_contiguous"):
        bits.append("c_contiguous=" + str(flags.c_contiguous) + " writeable=" + str(flags.writeable))
    return " ".join(bits)


def _blas():
    try:
        import numpy
        info = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return str(info.get("name"))
    except Exception:
        return "unknown"


class Runtime:
    def __init__(self, host, params):
        self.host = host
        print("host.facts(): " + repr(host.facts()))
        import numpy
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        print("python " + sys.version.split()[0] + ", numpy " + numpy.__version__ + ", blas " + _blas() + ", " + str(cpus)
              + " cpus available; installed: " + ", ".join(m + "=" + ("yes" if importlib.util.find_spec(m) else "no") for m in LIBRARIES))

    def {entry}(self, system):
        print("system is " + type(system).__module__ + "." + type(system).__name__)
        for name in PORTS:
            print("  system." + name + ": " + _describe(getattr(system, name, None)))
        raise RuntimeError("probe: stopped on purpose after printing the inputs")

    def metrics(self):
        return {{}}
{reference}'''


def probe_source(card: Mapping[str, Any]) -> str:
    """The source of a probe for the stage the card describes."""
    reference = ""
    if card["reference"] == "impl":
        reference = '\n\ndef reference(inputs, params):\n    raise RuntimeError("probe: stopped on purpose")\n'
    return _TEMPLATE.format(ports=[p["name"] for p in card["ports"]], libraries=list(LIBRARIES), entry=card["entry"] or "run",
                            reference=reference)


def facts_identity(scene: str, stage: str, skeleton: str) -> dict[str, str]:
    """What the measured text is a fact about: the scene, the stage and the shape of what a runtime is handed."""
    return dict(kind="facts", scene=scene, stage=stage, skeleton=hashlib.sha256(skeleton.encode()).hexdigest()[:16])


_CLASS = re.compile(r"<class '(?:\w+\.)*(\w+)'>")                       # dtype=<class 'warp._src.types.float32'>  ->  dtype=float32
_TYPE = re.compile(r"\b(\w+)(?:\.\w+)+\.(\w+)(?= dtype=)")                 # warp._src.types.array dtype=  ->  warp.array dtype=


def tidy(output: str) -> str:
    """The measured text with a library's internal module paths shortened: the same facts in fewer tokens, since it is sent every turn.

    What the probe printed stays in the journal as it was; text that has no such paths (numpy's) is returned unchanged."""
    return _TYPE.sub(r"\1.\2", _CLASS.sub(r"\1", output))


def facts_block(output: str) -> str:
    return ("Measured on this scene at the real tier, by a runtime that printed what a check hands it and stopped (this is what yours "
            "receives in a check; a timed run uses the world counts above):\n" + "\n".join("    " + line for line in tidy(output).splitlines()))
