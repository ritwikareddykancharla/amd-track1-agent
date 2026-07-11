"""Deterministic zero-token solvers.

If we can *prove* the answer with plain Python, we never touch the API at
all — local computation is free under the scoring rules. Currently:
arithmetic expressions and common percentage/sqrt phrasings. Each solver
returns None unless it is confident, so a miss simply falls through to the
API tier — a wrong free answer should be structurally impossible.
"""
from __future__ import annotations

import ast
import math
import operator
import re

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{round(value, 6):g}"
    return str(value)


_EXPR = re.compile(r"(-?\d+(?:\.\d+)?(?:\s*[-+*/^%]\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?)+)")

# Percent/sqrt must match the *entire* prompt (bar a question lead-in), not
# just a substring — "15% of 240" inside a longer word problem is a sub-step,
# and answering it alone would be wrong. Same guard philosophy as the 0.6
# coverage ratio on the AST path.
_LEAD = r"^\s*(?:what is|what's|calculate|compute|find|evaluate)?\s*"
_TAIL = r"\s*[?.!]*\s*$"
_PERCENT_OF = re.compile(
    _LEAD + r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)" + _TAIL, re.I
)
_SQRT = re.compile(
    _LEAD + r"(?:the\s+)?square root of\s+(\d+(?:\.\d+)?)" + _TAIL, re.I
)


def solve_math(prompt: str) -> str | None:
    """Solve pure-arithmetic prompts deterministically; None if not provable."""
    m = _PERCENT_OF.search(prompt)
    if m:
        return _format_number(float(m.group(1)) / 100.0 * float(m.group(2)))

    m = _SQRT.search(prompt)
    if m:
        return _format_number(math.sqrt(float(m.group(1))))

    m = _EXPR.search(prompt)
    if m:
        expr = m.group(1).replace("^", "**")
        # Only trust it when the expression is essentially the whole question —
        # otherwise it's a word problem and numbers alone mislead.
        stripped = re.sub(r"[\s?=.]|what is|calculate|compute|evaluate", "", prompt, flags=re.I)
        expr_stripped = re.sub(r"\s", "", m.group(1))
        if len(expr_stripped) >= 0.6 * len(stripped):
            try:
                return _format_number(_safe_eval(ast.parse(expr, mode="eval")))
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                return None
    return None
