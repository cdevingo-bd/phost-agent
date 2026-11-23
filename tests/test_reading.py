from phost_agent.reading import DIFF_LIMIT, OUTLINE_LIMIT, outline, source_diff

SOURCE = '''"""A banded Cholesky solve of the effective system.

More text that the outline does not repeat."""
import numpy as np
from scipy.linalg import solve_banded

# approach: reorder, then band
BANDWIDTH = 8


def reorder(h, depth=2) -> list:
    """Reverse Cuthill-McKee ordering."""
    return h


class Runtime(object):
    """The runtime."""

    def __init__(self, host, params):
        self.host = host

    @staticmethod
    def solve(self, system):
        # factor once
        pass
'''


def test_an_outline_names_the_shape_of_a_source_in_a_fraction_of_its_size():
    text = outline(SOURCE)
    assert text.splitlines()[0] == f"{len(SOURCE.splitlines())} lines" and "module: A banded Cholesky solve of the effective system." in text
    assert "imports: numpy, scipy" in text and "def reorder(h, depth=2) -> list  l.11: Reverse Cuthill-McKee ordering." in text
    assert "class Runtime(object)  l.16: The runtime." in text and "  def __init__(self, host, params)  l.19" in text
    assert "@staticmethod def solve(self, system)" in text and "module names: BANDWIDTH" in text
    assert "comments: # approach: reorder, then band | # factor once" in text
    assert "More text" not in text
    long = SOURCE + "\n".join(f"        y{i} = {i} * 2.0 + 1.0" for i in range(150)) + "\n"          # a body of a real size
    assert len(outline(long)) < 0.15 * len(long) and "y77" not in outline(long)


def test_an_outline_of_a_very_large_source_is_capped_and_one_that_does_not_parse_still_says_where_things_are():
    big = "\n".join(f"def f{i}(x):\n    return x" for i in range(400))
    assert len(outline(big)) <= OUTLINE_LIMIT and outline(big).endswith("…")
    broken = outline("class A:\n    def f(self):\n  oops(\n")
    assert "does not parse" in broken and "class A:" in broken and "def f(self):" in broken
    assert outline("") == "1 lines" or outline("").startswith("1 lines")


def test_a_diff_shows_only_what_changed_and_is_cut_when_it_is_long():
    new = SOURCE.replace("BANDWIDTH = 8", "BANDWIDTH = 16")
    text = source_diff(SOURCE, new, old_name="a1", new_name="b2")
    assert text.startswith("--- a1\n+++ b2") and "-BANDWIDTH = 8" in text and "+BANDWIDTH = 16" in text and len(text) < 0.4 * len(SOURCE)
    assert source_diff(SOURCE, SOURCE) == "identical source"
    huge = source_diff("\n".join(f"a{i}" for i in range(2000)), "\n".join(f"b{i}" for i in range(2000)))
    assert len(huge) <= DIFF_LIMIT + 60 and "more diff lines not shown" in huge


def test_a_long_annotated_argument_list_is_shown_as_names_so_the_outline_reaches_the_end_of_the_source():
    kernel = ("import warp as wp\n\n@wp.kernel\ndef k(" + ", ".join(f"a{i}: wp.array2d(dtype=wp.float32)" for i in range(14)) + "):\n    pass\n\n"
              "class Runtime:\n    def solve(self, system):\n        pass\n")
    text = outline(kernel)
    assert "@wp.kernel def k(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)" in text and "wp.array2d" not in text
    assert "class Runtime" in text and "def solve(self, system)" in text and not text.endswith("…")
