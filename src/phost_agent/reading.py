"""Cheaper ways to read someone else's source: its outline, and what changed between two versions.

Reading a rival's whole source costs thousands of tokens, and every one of them is sent again for several turns. Most of what a
contestant wants from a read is smaller than that: what the approach is (which library, which functions, what the author said in
comments), or what a fork changed. These are views of the same record, offered beside the whole source, never in place of it.
"""

from __future__ import annotations

import ast
import difflib

OUTLINE_LIMIT = 1400
DIFF_LIMIT = 3000


def _first_line(text: str | None, limit: int = 110) -> str:
    line = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _modules(tree: ast.AST) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
    return sorted(found)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = "".join(f"@{ast.unparse(d)} " for d in node.decorator_list)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{decorators}def {node.name}({ast.unparse(node.args)}){returns}"


def outline(source: str, *, limit: int = OUTLINE_LIMIT) -> str:
    """The shape of a source: its size, what it imports, its classes and functions with the first line of each docstring, and the
    first comments (where an author usually says what the approach is). Falls back to the lines that open a def or a class when the
    source does not parse."""
    lines = source.rstrip("\n").split("\n")
    comments = [line.strip()[:110] for line in lines if line.strip().startswith("#")][:6]
    out = [f"{len(lines)} lines"]
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        out.append(f"(does not parse: line {error.lineno}: {error.msg})")
        out += [f"l.{i}: {line.strip()[:110]}" for i, line in enumerate(lines, 1) if line.lstrip().startswith(("class ", "def "))][:25]
    else:
        if ast.get_docstring(tree):
            out.append("module: " + _first_line(ast.get_docstring(tree)))
        if _modules(tree):
            out.append("imports: " + ", ".join(_modules(tree)))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
                out.append(f"class {node.name}{bases}  l.{node.lineno}" + (f": {_first_line(ast.get_docstring(node))}" if ast.get_docstring(node) else ""))
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(f"  {_signature(member)}  l.{member.lineno}" + (f": {_first_line(ast.get_docstring(member))}" if ast.get_docstring(member) else ""))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(f"{_signature(node)}  l.{node.lineno}" + (f": {_first_line(ast.get_docstring(node))}" if ast.get_docstring(node) else ""))
        names = [t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)]
        if names:
            out.append("module names: " + ", ".join(names[:12]))
    if comments:
        out.append("comments: " + " | ".join(comments))
    text = "\n".join(out)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def source_diff(old: str, new: str, *, old_name: str = "a", new_name: str = "b", limit: int = DIFF_LIMIT) -> str:
    """A unified diff of two sources with two lines of context, cut at ``limit`` characters (and saying how much was cut)."""
    changed = list(difflib.unified_diff(old.splitlines(), new.splitlines(), old_name, new_name, lineterm="", n=2))
    if not changed:
        return "identical source"
    text, shown = "", 0
    for line in changed:
        if len(text) + len(line) + 1 > limit:
            break
        text += line + "\n"
        shown += 1
    if shown < len(changed):
        text += f"... {len(changed) - shown} more diff lines not shown"
    return text.rstrip("\n")
