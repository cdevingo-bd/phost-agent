"""What a candidate prints while a check runs comes back with the result: captured only from the candidate's own module."""

import builtins
import sys
import types

from phost_agent.diagnostics import CANDIDATE_PREFIX, LIMIT, Diagnostics, install


def test_lines_are_kept_in_order_and_identical_lines_in_a_row_are_folded():
    d = Diagnostics()
    d.write("shape (2, 256)\n")
    for _ in range(30):
        d.write("calling solve\n")
    d.write("done")                                              # no newline at the end: still a line
    assert d.text() == "shape (2, 256)\ncalling solve  (x30)\ndone"


def test_what_does_not_fit_is_counted_not_kept():
    d = Diagnostics(limit=50)
    for i in range(20):
        d.write(f"line number {i}\n")
    text = d.text()
    assert text.startswith("line number 0\nline number 1") and "more characters not shown" in text
    assert len(text.split("\n")[0]) < 50 and LIMIT == 4000


def candidate_module(name):
    module = types.ModuleType(name)
    exec("def speak(*a, **k):\n    print(*a, **k)\n", module.__dict__)
    return module


def test_only_a_candidates_prints_are_captured_and_the_real_print_comes_back(capsys):
    d = Diagnostics()
    real = install(d)
    try:
        candidate_module(CANDIDATE_PREFIX + "abc_runtime").speak("W =", 2, sep="")
        candidate_module(CANDIDATE_PREFIX + "abc_runtime").speak("to stderr", file=sys.stderr)
        candidate_module(CANDIDATE_PREFIX + "abc_runtime").speak("somewhere else", file=open("/dev/null", "w"))
        candidate_module("some_library").speak("a library's own output")
        print("the worker's own line")
    finally:
        builtins.print = real
    out = capsys.readouterr().out
    assert d.text() == "W =2\nto stderr"
    assert "a library's own output" in out and "the worker's own line" in out and "W =2" not in out
    assert builtins.print is real
