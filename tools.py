"""Harness-side tools for fourfold agents (fabe-integration, 2026-11-07; brief: quincy/TOOLS-BRIEF.md).

- AgentFS: one agent's filesystem with qwenFS's tool interface and response strings
  (github.com/vixhal-baraiya/qwenFS, README tool table and docs/fs-agent.md examples).
- compute(expr): a qwenputer-style exact computer over integers and Fractions (no eval/exec).

Pure Python 3.12, stdlib only, deterministic. Swappable later for the in-weights versions.
"""
from __future__ import annotations

import ast
import re
from fractions import Fraction

# ----------------------------------------------------------------------------- filesystem

READ_CAP = 1500          # characters of a read's returned text
READ_CUT = "\n[cut: {rest} more characters]"
_WORD = re.compile(r"[A-Za-z0-9_']+")


def _norm_path(p) -> str | None:
    """qwenFS paths: short, folders by '/', no leading '/', no empty or '..' parts."""
    if not isinstance(p, str):
        return None
    p = p.strip().strip("/")
    if not p or len(p) > 200 or any(c in p for c in "\n\r\t\0"):
        return None
    parts = p.split("/")
    if any(not part or part in (".", "..") for part in parts):
        return None
    return "/".join(parts)


class AgentFS:
    """one agent's filesystem, qwenFS semantics."""

    MAX_FILES = 128
    TOOLS = ("write_file", "append_file", "read_file", "list_files", "link_files", "unlink_files",
             "list_links", "list_backlinks", "search_files")

    def __init__(self):
        self.files: dict[str, str] = {}             # path -> current text (insertion order = listing order)
        self.links: list[tuple[str, str]] = []      # (name, target) pairs, each at most once

    # -- the tool surface -----------------------------------------------------------------
    def call(self, tool: str, **args) -> str:
        """returns the response text the agent will see."""
        fn = getattr(self, "_t_" + str(tool), None)
        if fn is None or tool not in self.TOOLS:
            return f"error: unknown tool {tool!r}"
        try:
            return fn(**args)
        except TypeError:
            return f"error: {tool} takes {self._params(tool)}"

    @staticmethod
    def _params(tool):
        return {"write_file": "name, text", "append_file": "name, text", "read_file": "name", "list_files": "no parameters",
                "link_files": "name, target", "unlink_files": "name, target", "list_links": "file",
                "list_backlinks": "path", "search_files": "query"}[tool]

    def _t_write_file(self, name, text):
        path = _norm_path(name)
        if path is None:
            return "error: bad path"
        if not isinstance(text, str):
            return "error: text must be a string"
        if path not in self.files and len(self.files) >= self.MAX_FILES:
            return f"error: file limit reached ({self.MAX_FILES} files); rewrite or reuse a file"
        self.files[path] = text
        return f"saved\n-{path}"

    def _t_append_file(self, name, text):
        path = _norm_path(name)
        if path is None:
            return "error: bad path"
        if not isinstance(text, str):
            return "error: text must be a string"
        if path not in self.files:
            if len(self.files) >= self.MAX_FILES:
                return f"error: file limit reached ({self.MAX_FILES} files); rewrite or reuse a file"
            self.files[path] = text
        else:
            self.files[path] = self.files[path] + "\n" + text if self.files[path] else text
        return f"saved\n-{path}"

    def _t_read_file(self, name):
        path = _norm_path(name)
        if path is None or path not in self.files:
            return "not found"
        text = self.files[path]
        if len(text) > READ_CAP:
            return text[:READ_CAP] + READ_CUT.format(rest=len(text) - READ_CAP)
        return text

    def _t_list_files(self):
        return self._listing(self.files)

    def _t_link_files(self, name, target):
        a, b = _norm_path(name), _norm_path(target)
        if a is None or b is None:
            return "error: bad path"
        if a not in self.files or b not in self.files:
            return "not found"
        if (a, b) not in self.links:
            self.links.append((a, b))
        return f"linked\n-{a}\nto\n-{b}"

    def _t_unlink_files(self, name, target):
        a, b = _norm_path(name), _norm_path(target)
        if a is None or b is None:
            return "error: bad path"
        if a not in self.files or b not in self.files:
            return "not found"
        if (a, b) in self.links:
            self.links.remove((a, b))
        return f"removed\n-{a}\nto\n-{b}"

    def _t_list_links(self, file):
        path = _norm_path(file)
        if path is None or path not in self.files:
            return "not found"
        return self._listing([b for a, b in self.links if a == path])

    def _t_list_backlinks(self, path):
        p = _norm_path(path)
        if p is None or p not in self.files:
            return "not found"
        return self._listing([a for a, b in self.links if b == p])

    def _t_search_files(self, query):
        if not isinstance(query, str):
            return "empty"
        words = [w.lower() for w in _WORD.findall(query)]
        if not words:
            return "empty"
        word = words[0]                         # one whole word, case-insensitive, against each file's CURRENT text
        hits = [p for p, text in self.files.items() if word in {w.lower() for w in _WORD.findall(text)}]
        return self._listing(hits)

    @staticmethod
    def _listing(paths) -> str:
        paths = list(paths)
        return "\n".join("-" + p for p in paths) if paths else "empty"

    # -- state ------------------------------------------------------------------------------
    def to_json(self) -> dict:
        """whole state, for ticks.jsonl / resume."""
        return {"files": [[p, t] for p, t in self.files.items()], "links": [list(l) for l in self.links]}

    @classmethod
    def from_json(cls, d: dict) -> "AgentFS":
        fs = cls()
        for p, t in d.get("files", []):
            fs.files[str(p)] = str(t)
        for a, b in d.get("links", []):
            if (a, b) not in fs.links:
                fs.links.append((str(a), str(b)))
        return fs


# ----------------------------------------------------------------------------- computer

MAX_EXPR = 300
MAX_EXPONENT = 64
MAX_DIGITS = 200
MAX_LITERAL_DIGITS = 30
_FUNCS = {"sum", "min", "max"}


class ComputeError(ValueError):
    pass


def _check_size(v: Fraction) -> Fraction:
    if len(str(abs(v.numerator))) > MAX_DIGITS or len(str(v.denominator)) > MAX_DIGITS:
        raise ComputeError(f"result exceeds {MAX_DIGITS} digits")
    return v


def _ev(node) -> Fraction:
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ComputeError("only integer literals are allowed")
        if len(str(abs(node.value))) > MAX_LITERAL_DIGITS:
            raise ComputeError(f"integer literal longer than {MAX_LITERAL_DIGITS} digits")
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left), _ev(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return _check_size(a + b)
        if isinstance(op, ast.Sub):
            return _check_size(a - b)
        if isinstance(op, ast.Mult):
            return _check_size(a * b)
        if isinstance(op, ast.Div):
            if b == 0:
                raise ComputeError("division by zero")
            return _check_size(a / b)
        if isinstance(op, ast.FloorDiv):
            if b == 0:
                raise ComputeError("division by zero")
            return _check_size(Fraction(a // b))
        if isinstance(op, ast.Mod):
            if b == 0:
                raise ComputeError("division by zero")
            return _check_size(a % b)
        if isinstance(op, ast.Pow):
            if b.denominator != 1:
                raise ComputeError("exponent must be an integer")
            e = b.numerator
            if abs(e) > MAX_EXPONENT:
                raise ComputeError(f"exponent exceeds {MAX_EXPONENT}")
            if e < 0 and a == 0:
                raise ComputeError("division by zero")
            if a != 0 and a.denominator == 1 and e > 0 and len(str(abs(a.numerator))) * e > MAX_DIGITS + 1:
                raise ComputeError(f"result exceeds {MAX_DIGITS} digits")
            return _check_size(a ** e)
        raise ComputeError("unsupported operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ComputeError("only sum(...), min(...), max(...) are allowed")
        if not node.args:
            raise ComputeError(f"{node.func.id}() needs at least one argument")
        vals = [_ev(a) for a in node.args]
        f = {"sum": lambda xs: _check_size(sum(xs, Fraction(0))), "min": min, "max": max}[node.func.id]
        return f(vals)
    raise ComputeError("unsupported syntax")


def _fmt(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def compute(expr: str) -> str:
    """exact arithmetic: '= 1234', '= 7/3', or 'error: <why>'."""
    if not isinstance(expr, str):
        return "error: expression must be a string"
    expr = expr.strip()
    if not expr:
        return "error: empty expression"
    if len(expr) > MAX_EXPR:
        return f"error: expression longer than {MAX_EXPR} characters"
    if not all(32 <= ord(c) < 127 for c in expr):
        return "error: unsupported character"
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "error: cannot parse expression"
    try:
        return "= " + _fmt(_ev(tree))
    except ComputeError as e:
        return f"error: {e}"
    except RecursionError:
        return "error: expression too deeply nested"
