from __future__ import annotations
import ast
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

# ---------- Trace objects ----------
@dataclass
class ToolEvent:
    name: str
    input: str
    ok: bool
    output: str

@dataclass
class ToolResult:
    ok: bool
    content: str
    event: ToolEvent

# ---------- Mini facts store ----------
class FactsStore:
    def __init__(self) -> None:
        self._facts: Dict[str, str] = {}

    def remember(self, key: str, value: str) -> ToolResult:
        k = key.strip().lower()
        v = value.strip()
        self._facts[k] = v
        evt = ToolEvent("facts.remember", f"{k}={v}", True, "Saved.")
        return ToolResult(True, "Noted.", evt)

    def recall(self, key: str) -> ToolResult:
        k = key.strip().lower()
        if k not in self._facts:
            evt = ToolEvent("facts.recall", k, False, "Not found.")
            return ToolResult(False, "I don't have that saved yet.", evt)
        v = self._facts[k]
        evt = ToolEvent("facts.recall", k, True, v)
        return ToolResult(True, v, evt)

# ---------- Safe calculator (AST) ----------
_ALLOWED = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}

def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numbers are allowed.")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _ALLOWED:
            raise ValueError("Operation not allowed.")
        return _ALLOWED[op](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _ALLOWED:
            raise ValueError("Operation not allowed.")
        return _ALLOWED[op](_eval(node.operand))
    raise ValueError("Invalid expression.")

def calculator(expr: str) -> ToolResult:
    try:
        value = _eval(ast.parse(expr, mode="eval").body)
        out = str(value)
        evt = ToolEvent("calculator", expr, True, out)
        return ToolResult(True, out, evt)
    except Exception as e:
        evt = ToolEvent("calculator", expr, False, str(e))
        return ToolResult(False, "Sorry, I couldn't compute that.", evt)

# ---------- Clock ----------
def clock(_: str = "") -> ToolResult:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evt = ToolEvent("clock", "", True, now)
    return ToolResult(True, now, evt)
