from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .analysis import GRACEFUL_FAILURE_MESSAGE, execute_with_retry
from .config import Settings, get_settings
from .datasets import Dataset
from .errors import LLMError
from .llm import LLMClient
from .policy import classify_request
from .prompts import answer_messages
from .schema_mapping import (
    apply_clarifications,
    build_clarification,
    infer_column_map,
    needs_clarification,
)

logger = logging.getLogger("agent")

OUTCOMES = ("answer", "reject", "clarify", "fail")


@dataclass
class TraceStep:
    action: str
    ok: bool = True
    detail: str = ""


@dataclass
class AgentResponse:
    question: str
    outcome: str = "fail"  # answer | reject | clarify | fail
    message: str = GRACEFUL_FAILURE_MESSAGE
    answer: str | None = None
    result: dict | None = None
    code: str | None = None
    column_map: dict = field(default_factory=dict)
    clarification: dict | None = None
    repairs: int = 0
    attempts: int = 0
    trace: list[TraceStep] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def steps(self) -> int:
        return len(self.trace)

    def add(self, action: str, ok: bool = True, detail: str = "") -> None:
        self.trace.append(TraceStep(action, ok, detail))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = self.steps
        return data


def format_result_text(result: dict | None, max_chars: int = 2500) -> str:
    """Compact text view of an (already aggregated) result payload."""
    if not result:
        return "(no result)"
    if result["kind"] == "scalar":
        return str(result["value"])
    lines = [" | ".join(result["columns"])]
    for row in result["rows"]:
        lines.append(" | ".join(str(row.get(c)) for c in result["columns"]))
    return "\n".join(lines)[:max_chars]


class AnalyticsAgent:
    def __init__(self, llm: LLMClient, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ public
    def run(
        self,
        question: str,
        dataset: Dataset,
        clarifications: dict[str, str] | None = None,
    ) -> AgentResponse:
        """Answer one question. Never raises: every failure becomes a safe response."""
        start = time.perf_counter()
        response = AgentResponse(question=question)
        try:
            self._run(response, dataset, clarifications or {})
        except Exception as exc:  # noqa: BLE001 - last line of defence: do NOT crash
            logger.exception("Unexpected agent error")
            response.outcome = "fail"
            response.message = GRACEFUL_FAILURE_MESSAGE
            response.add("error", False, type(exc).__name__)
        response.latency_s = round(time.perf_counter() - start, 3)
        logger.info(
            "outcome=%s steps=%d repairs=%d latency=%.2fs",
            response.outcome, response.steps, response.repairs, response.latency_s,
        )
        return response

    # ----------------------------------------------------------------- workflow
    def _run(self, resp: AgentResponse, dataset: Dataset, clarifications: dict[str, str]) -> None:
        s = self.settings

        # 1. classify_request - deterministic policy, before any LLM call
        decision = classify_request(
            resp.question,
            known_tokens=dataset.known_tokens,
            known_full_names=dataset.known_full_names,
        )
        resp.add("classify_request", True, "authorized" if decision.authorized else decision.rule or "")
        if not decision.authorized:
            resp.add("reject_request", True, decision.rule or "")
            resp.outcome = "reject"
            resp.message = decision.reason
            logger.info("Request rejected by policy rule '%s'", decision.rule)
            return

        # 2. infer_column_map - vocabulary -> real columns (no pandas execution)
        columns = dataset.visible_columns
        column_map = infer_column_map(resp.question, columns, self.llm if s.use_llm_mapping else None)
        column_map = apply_clarifications(column_map, clarifications, columns)
        resp.column_map = column_map
        resp.add("infer_column_map", True, json.dumps(column_map))

        # 3. ambiguity policy: ONE short question, never run the analysis
        if needs_clarification(resp.question, column_map):
            clarification = build_clarification(column_map)
            resp.add("ask_clarification", True, clarification["term"] if clarification else "")
            resp.outcome = "clarify"
            resp.clarification = clarification
            resp.message = clarification["question"] if clarification else "Could you clarify your question?"
            return

        # 4. run_analysis - generate, validate, execute, repair
        try:
            outcome = execute_with_retry(
                resp.question, dataset, self.llm, column_map=column_map, settings=s
            )
        except LLMError as exc:
            resp.add("generate_code", False, f"LLM unavailable: {exc}")
            resp.outcome = "fail"
            resp.message = f"The language model is unavailable right now ({exc})."
            return

        for action, ok, detail in outcome.events:
            resp.add(action, ok, detail)
        resp.code = outcome.code
        resp.repairs = outcome.repairs
        resp.attempts = outcome.attempts

        if not outcome.success:
            resp.add("fail", False, outcome.errors[-1] if outcome.errors else "")
            resp.outcome = "fail"
            resp.message = GRACEFUL_FAILURE_MESSAGE
            return

        resp.result = outcome.result

        # 5. answer_user - the LLM only ever sees the aggregated result
        resp.answer = self._explain(resp.question, resp.result)
        resp.add("answer_user", True)
        resp.outcome = "answer"
        resp.message = resp.answer

        # 6. finish
        resp.add("finish", True)

    def _explain(self, question: str, result: dict | None) -> str:
        text = format_result_text(result)
        try:
            return self.llm.chat(answer_messages(question, text), max_tokens=200)
        except Exception as exc:  # noqa: BLE001 - the numbers are still correct without prose
            logger.warning("Answer phrasing skipped: %s", type(exc).__name__)
            if result and result["kind"] == "scalar":
                return f"The result is {result['value']}."
            return "Here is the aggregated result."
