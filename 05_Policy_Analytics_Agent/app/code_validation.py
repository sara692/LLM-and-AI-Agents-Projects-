"""AST validation of LLM-generated pandas code.

The model's code is NEVER executed before it has been parsed and inspected here.
Parsing into an AST does not run anything; we then walk every node and reject
whatever could escape the sandbox, read/write files, or dump raw rows.

Layers enforced here (see README for the full defence-in-depth picture):
  1. forbidden syntax      - imports, loops, functions, comprehensions, try/with ...
  2. forbidden names       - exec, eval, open, getattr, dunder names ...
  3. forbidden attributes  - dunders, to_*/read_* I/O, .query/.eval, .format ...
  4. protected columns     - PII columns can never be referenced
  5. aggregation contract  - code must contain a reducing call (mean/sum/...)
                             and may only head()/nlargest() *aggregated* data
  6. result contract       - code must assign a variable called `result`
"""

from __future__ import annotations

import ast
from typing import Iterable

from .errors import CodeValidationError

MAX_CODE_CHARS = 4000


def _node_types(*names: str) -> tuple[type, ...]:
    """Collect AST classes by name, skipping ones this Python version lacks."""
    return tuple(getattr(ast, n) for n in names if hasattr(ast, n))


FORBIDDEN_NODES = _node_types(
    "Import", "ImportFrom",          # no importing os/requests/...
    "With", "AsyncWith",             # no `with open(...)`
    "While", "For", "AsyncFor",      # no loops -> no row-by-row dumping
    "Try", "TryStar",                # no hiding logic in try/except
    "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Delete", "Global", "Nonlocal", "Raise",
    "Await", "Yield", "YieldFrom",
    "ListComp", "SetComp", "DictComp", "GeneratorExp",  # loops in disguise
    "Match",
)

FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "input", "breakpoint", "help", "exit", "quit",
    "__import__", "os", "sys", "subprocess", "shutil", "socket", "pathlib",
    "print", "getattr", "setattr", "delattr", "hasattr", "globals", "locals", "vars",
    "dir", "type", "object", "super", "classmethod", "staticmethod", "property",
    "memoryview", "id", "callable",
}

# Attributes that read raw rows, perform I/O, evaluate strings, or reach internals.
DENIED_ATTRS = {
    "format", "format_map",              # str.format can traverse attributes
    "eval", "query", "pipe",             # string evaluation / arbitrary callables
    "style", "plot", "hist", "boxplot",
    "sample", "iloc", "iat", "at",       # raw row access
    "iterrows", "itertuples", "items", "iteritems",
    "values", "tolist", "to_list",       # (to_* is also denied by prefix below)
    "merge", "join",                     # cross joins can explode memory
    "attrs", "memory_usage", "info",
}
ALLOWED_TO_ATTRS = {
    "to_datetime", "to_numeric", "to_period", "to_timestamp", "to_timedelta", "to_frame",
}

# A call to one of these means the code is *reducing* data (aggregating it).
REDUCERS = {
    "mean", "sum", "count", "min", "max", "median", "std", "var", "nunique", "size",
    "agg", "aggregate", "describe", "value_counts", "corr", "quantile",
    "idxmax", "idxmin", "prod", "sem", "skew", "kurt", "mode",
    "pivot_table", "crosstab",
}
# Row selectors are only fine when applied to something already aggregated.
ROW_SELECTORS = {"head", "tail", "nlargest", "nsmallest"}


def _attr_denied(attr: str) -> bool:
    a = attr.lower()
    if a.startswith("_") or a.startswith("read_"):
        return True
    if a.startswith("to_") and a not in ALLOWED_TO_ATTRS:
        return True
    return a in DENIED_ATTRS


def _has_reducer(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr in REDUCERS for n in ast.walk(node))


def validate_code(code: str, protected_columns: Iterable[str] = ()) -> None:
    """Raise CodeValidationError if `code` is not safe, aggregated analysis code."""
    if not code or not code.strip():
        raise CodeValidationError("No code was produced.")
    if len(code) > MAX_CODE_CHARS:
        raise CodeValidationError(f"Code is too long ({len(code)} > {MAX_CODE_CHARS} chars).")

    try:
        tree = ast.parse(code)  # parsing never executes the code
    except SyntaxError as exc:
        raise CodeValidationError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc

    protected = {c.lower() for c in protected_columns}
    assigns_result = False
    reduced_names: set[str] = set()

    for node in ast.walk(tree):
        # 1. forbidden syntax
        if isinstance(node, FORBIDDEN_NODES):
            raise CodeValidationError(f"Forbidden operation: {type(node).__name__}")

        # 2. forbidden names
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise CodeValidationError(f"Forbidden name used: {node.id}")

        # 3 + 4. attributes
        elif isinstance(node, ast.Attribute):
            if _attr_denied(node.attr):
                raise CodeValidationError(f"Forbidden attribute: .{node.attr}")
            if node.attr.lower() in protected:
                raise CodeValidationError(
                    f"Access to protected column '{node.attr}' is not allowed."
                )

        # 4. string constants (column labels, dunder smuggling)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise CodeValidationError("Strings containing '__' are not allowed.")
            if node.value.strip().lower() in protected:
                raise CodeValidationError(
                    f"Access to protected column '{node.value}' is not allowed."
                )

        # bookkeeping: result assignment + which variables hold aggregated data
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = getattr(node, "value", None)
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id == "result":
                        assigns_result = True
                    if value is not None and _has_reducer(value):
                        reduced_names.add(target.id)

    # 6. result contract
    if not assigns_result:
        raise CodeValidationError("Code must assign the final answer to a variable named `result`.")

    # 5. aggregation contract
    if not _has_reducer(tree):
        raise CodeValidationError(
            "Only aggregated answers are allowed: use mean/sum/count/min/max/groupby-agg/"
            "value_counts/etc. Row-level output is not permitted."
        )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ROW_SELECTORS
        ):
            base = node.func.value
            on_aggregate = _has_reducer(base) or any(
                isinstance(n, ast.Name) and n.id in reduced_names for n in ast.walk(base)
            )
            if not on_aggregate:
                raise CodeValidationError(
                    f".{node.func.attr}() on raw rows is not allowed; aggregate first."
                )
