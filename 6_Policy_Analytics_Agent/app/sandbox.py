from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import queue as queue_mod
import threading
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .code_validation import validate_code
from .errors import ExecutionError, OutputPolicyError, SandboxError, SandboxTimeout

logger = logging.getLogger("agent.sandbox")

SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "round": round, "abs": abs, "min": min, "max": max, "sum": sum,
    "float": float, "int": int, "str": str, "bool": bool,
}

SAFE_PD = SimpleNamespace(
    DataFrame=pd.DataFrame, Series=pd.Series, concat=pd.concat, cut=pd.cut, qcut=pd.qcut,
    to_datetime=pd.to_datetime, to_numeric=pd.to_numeric, isna=pd.isna, notna=pd.notna,
    NA=pd.NA, Timestamp=pd.Timestamp, Timedelta=pd.Timedelta, NamedAgg=pd.NamedAgg,
    crosstab=pd.crosstab, pivot_table=pd.pivot_table, Grouper=pd.Grouper,
)


# ---------------------------------------------------------------------------
# Output policy + serialisation
# ---------------------------------------------------------------------------
def serialize_result(
    result: Any, n_source_rows: int, protected_columns: Iterable[str], max_rows: int
) -> dict:

    protected = {c.lower() for c in protected_columns}

    if result is None:
        raise OutputPolicyError("`result` is None; assign the final answer to `result`.")

    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, (bool, int, float, str)):
        if isinstance(result, float):
            result = None if (math.isnan(result) or math.isinf(result)) else round(result, 4)
        if isinstance(result, str) and len(result) > 500:
            raise OutputPolicyError("Text result is too long.")
        return {"kind": "scalar", "value": result}

    if isinstance(result, dict):
        result = pd.Series(result, name="value")
    elif isinstance(result, (list, tuple)):
        result = pd.Series(list(result), name="value")

    if isinstance(result, pd.Series):
        name = str(result.name) if result.name is not None else "value"
        frame = result.to_frame(name=name)
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
    elif isinstance(result, pd.DataFrame):
        frame = result
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
    else:
        raise OutputPolicyError(f"Unsupported result type: {type(result).__name__}")

    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["_".join(str(p) for p in col if str(p) != "") for col in frame.columns]
    frame.columns = [str(c) for c in frame.columns]
    if "index" in frame.columns and "label" not in frame.columns:
        frame = frame.rename(columns={"index": "label"})  # unnamed index of a Series/describe()

    bad = [c for c in frame.columns if c.lower() in protected]
    if bad:
        raise OutputPolicyError(f"Result contains protected column(s): {bad}.")
    if not frame.columns.is_unique:
        raise OutputPolicyError("Result has duplicate column names.")

    n = len(frame)
    if n > max_rows:
        raise OutputPolicyError(
            f"Result has {n} rows (limit {max_rows}). Only aggregated, summary-sized answers are allowed."
        )
    if n_source_rows > 5 and n >= n_source_rows:
        raise OutputPolicyError("Result looks like raw rows, not an aggregation.")

    records = json.loads(frame.round(4).to_json(orient="records", date_format="iso"))
    return {"kind": "table", "columns": list(frame.columns), "rows": records}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def _execute(code: str, df: pd.DataFrame, protected: list[str], max_rows: int) -> dict:
    namespace: dict[str, Any] = {
        "__builtins__": dict(SAFE_BUILTINS),
        "df": df.copy(),
        "pd": SAFE_PD,
    }
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 (validated + restricted)
    except SandboxError:
        raise
    except Exception as exc:  # code errors become feedback for the repair loop
        raise ExecutionError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
    if "result" not in namespace:
        raise ExecutionError("Code must assign the final answer to a variable named `result`.")
    return serialize_result(namespace["result"], len(df), protected, max_rows)


def _run_in_thread(code, df, protected, max_rows, timeout) -> dict:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["ok"] = _execute(code, df, protected, max_rows)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["err"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        # A thread cannot be killed; use sandbox_mode="process" for hard isolation.
        raise SandboxTimeout(f"Execution exceeded {timeout:.0f}s.")
    if "err" in box:
        raise box["err"]
    return box["ok"]


def _child_entry(code, df, protected, max_rows, out_queue) -> None:
    try:
        out_queue.put(("ok", _execute(code, df, protected, max_rows)))
    except BaseException as exc:  # noqa: BLE001
        out_queue.put(("err", f"{type(exc).__name__}: {str(exc)[:300]}"))


def _run_in_process(code, df, protected, max_rows, timeout) -> dict:
    ctx = mp.get_context("spawn") 
    out_queue = ctx.Queue()
    proc = ctx.Process(target=_child_entry, args=(code, df, protected, max_rows, out_queue), daemon=True)
    proc.start()
    try:
        status, payload = out_queue.get(timeout=timeout)
    except queue_mod.Empty:
        proc.terminate()
        proc.join(2)
        raise SandboxTimeout(f"Execution exceeded {timeout:.0f}s.") from None
    finally:
        proc.join(2)
        if proc.is_alive():
            proc.terminate()
    if status == "err":
        raise ExecutionError(payload)
    return payload


def run_in_sandbox(
    code: str,
    df: pd.DataFrame,
    protected_columns: Iterable[str] = (),
    *,
    mode: str = "thread",
    timeout: float = 10.0,
    max_rows: int = 50,
) -> dict:
    protected = list(protected_columns)
    validate_code(code, protected)  
    if mode == "process":
        return _run_in_process(code, df, protected, max_rows, timeout)
    return _run_in_thread(code, df, protected, max_rows, timeout)
