from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field

import pandas as pd

from .config import Settings, get_settings
from .datasets import Dataset
from .errors import LLMError
from .llm import LLMClient
from .prompts import code_messages, repair_prompt
from .sandbox import run_in_sandbox

logger = logging.getLogger("agent.analysis")

GRACEFUL_FAILURE_MESSAGE = "I could not compute this safely. Try rephrasing."

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_HARMLESS_IMPORT = re.compile(r"^\s*(?:import\s+(?:pandas\s+as\s+pd|pandas|numpy\s+as\s+np)|from\s+pandas\s+import\s+.*)\s*$")


def extract_code(text: str) -> str:
    """Pull Python out of a model reply. Prefers the first fenced block; strips comments,
    blank lines and the harmless `import pandas as pd` models love to add. Everything else
    (including real imports) is left for the AST validator to reject."""
    if not text:
        return ""
    match = _FENCE.search(text)
    if match:
        body = match.group(1)
    else:
        body = text.replace("```python", "").replace("```", "")
    lines = []
    for line in body.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if _HARMLESS_IMPORT.match(line):
            continue
        lines.append(line)
    return textwrap.dedent("\n".join(lines)).strip()


def generate_code(
    question: str, dataset: Dataset, llm: LLMClient, column_map: dict | None = None,
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    schema = dataset.llm_schema(include_values=s.expose_category_values)
    raw = llm.chat(code_messages(question, schema, column_map, s.max_result_rows), max_tokens=600)
    return extract_code(raw)


@dataclass
class ExecutionOutcome:
    success: bool
    result: dict | None
    code: str
    attempts: int = 0
    repairs: int = 0
    errors: list[str] = field(default_factory=list)
    events: list[tuple[str, bool, str]] = field(default_factory=list)  # (action, ok, detail)


def execute_with_retry(
    question: str,
    dataset: Dataset | pd.DataFrame,
    llm: LLMClient,
    *,
    column_map: dict | None = None,
    max_retries: int | None = None,
    settings: Settings | None = None,
    initial_code: str | None = None,
) -> ExecutionOutcome:
    """Generate code (unless given), run it, and auto-repair on failure.

    `max_retries` is the total number of execution attempts (default 5 from settings).
    Never raises for code problems; returns ExecutionOutcome(success=False, ...) instead.
    An LLMError while *generating the first code* is propagated so the caller can report
    that the model is unavailable.
    """
    s = settings or get_settings()
    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_dataframe("adhoc", dataset, s.protected_patterns)
    limit = max_retries if max_retries is not None else s.max_retries
    schema = dataset.llm_schema(include_values=s.expose_category_values)

    outcome = ExecutionOutcome(success=False, result=None, code=initial_code or "")
    code = initial_code
    if code is None:
        code = generate_code(question, dataset, llm, column_map, s)
        outcome.events.append(("generate_code", bool(code.strip()), ""))
    outcome.code = code

    for attempt in range(1, limit + 1):
        outcome.attempts = attempt
        logger.info("Attempt %d/%d - executing generated code", attempt, limit)
        try:
            outcome.result = run_in_sandbox(
                code,
                dataset.df,
                dataset.protected_columns,
                mode=s.sandbox_mode,
                timeout=s.sandbox_timeout_s,
                max_rows=s.max_result_rows,
            )
        except Exception as exc:  # noqa: BLE001 - every failure feeds the repair loop
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            outcome.errors.append(error)
            outcome.events.append(("execute_code", False, error))
            logger.warning("Attempt %d/%d FAILED: %s", attempt, limit, error)

            if attempt == limit:
                logger.error("Max retries reached - giving up")
                break

            logger.info("Asking the LLM to repair the code...")
            try:
                raw = llm.chat(repair_prompt(schema, question, code, error, s.max_result_rows), max_tokens=600)
            except LLMError as llm_exc:
                outcome.events.append(("repair_code", False, f"LLM unavailable: {llm_exc}"))
                break
            fixed = extract_code(raw)
            outcome.repairs += 1
            if fixed.strip():
                code = fixed
                outcome.code = code
                outcome.events.append(("repair_code", True, ""))
            else:
                outcome.events.append(("repair_code", False, "empty fix; keeping previous code"))
            continue

        outcome.success = True
        outcome.events.append(("execute_code", True, ""))
        logger.info("Attempt %d/%d SUCCEEDED (repairs used: %d)", attempt, limit, outcome.repairs)
        return outcome

    return outcome
