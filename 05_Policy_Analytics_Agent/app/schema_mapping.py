from __future__ import annotations

import logging
import re

from .llm import LLMClient, parse_json_object
from .prompts import mapping_messages

logger = logging.getLogger("agent.mapping")

# user word -> candidate columns. Candidates are intersected with the dataset's real
# columns: one hit = confident mapping, several hits = ambiguous -> ask the user.
SYNONYMS: dict[str, list[str]] = {
    "dept": ["department"],
    "division": ["department"],
    "team": ["department"],
    "exp": ["years_experience"],
    "experience": ["years_experience"],
    "tenure": ["years_experience"],
    "seniority": ["years_experience"],
    "income": ["salary", "revenue"],
    "wage": ["salary"],
    "pay": ["salary", "bonus"],
    "money": ["salary", "bonus", "revenue", "unit_price"],
    "earnings": ["salary", "bonus", "revenue"],
    "compensation": ["salary", "bonus"],
    "sales": ["revenue"],
    "turnover": ["revenue"],
    "qty": ["quantity"],
    "units": ["quantity"],
    "price": ["unit_price"],
    "area": ["region"],
    "territory": ["region"],
    "category": ["product_category"],
    "rating": ["performance_score"],
    "performance": ["performance_score"],
}


def _in_question(term: str, question: str) -> bool:
    return re.search(rf"\b{re.escape(term)}s?\b", question, re.I) is not None


def deterministic_map(question: str, columns: list[str]) -> dict:
    lookup = {c.lower(): c for c in columns}
    out: dict = {}
    for term, candidates in SYNONYMS.items():
        if term in lookup or not _in_question(term, question):
            continue  # exact column names need no mapping
        present = [lookup[c] for c in candidates if c in lookup]
        if len(present) == 1:
            out[term] = present[0]
        elif len(present) > 1:
            out[term] = present
    return out


def _clean_llm_map(raw: dict, question: str, columns: list[str]) -> dict:
    """Keep only entries that are real, relevant and non-trivial."""
    lookup = {c.lower(): c for c in columns}
    cleaned: dict = {}
    for term, value in raw.items():
        term = str(term).strip()
        if not term or term.lower() in lookup or not _in_question(term, question):
            continue
        values = value if isinstance(value, list) else [value]
        real = []
        for v in values:
            col = lookup.get(str(v).strip().lower())
            if col and col not in real:
                real.append(col)
        if len(real) == 1:
            cleaned[term.lower()] = real[0]
        elif len(real) > 1:
            cleaned[term.lower()] = real
    return cleaned


def infer_column_map(question: str, columns: list[str], llm: LLMClient | None = None) -> dict:
    """Return {user term: column | [candidate columns]}. No dataframe code is run.

    `columns` must be the *visible* (non-protected) columns only.
    """
    column_map = deterministic_map(question, columns)
    if llm is not None:
        try:
            raw = parse_json_object(llm.chat(mapping_messages(question, columns), max_tokens=200))
            for term, value in _clean_llm_map(raw, question, columns).items():
                column_map.setdefault(term, value)
        except Exception as exc:  # noqa: BLE001 - mapping is best-effort, never fatal
            logger.warning("LLM column mapping skipped: %s", type(exc).__name__)
    return column_map


def needs_clarification(question: str, column_map: dict) -> bool:
    """True when any user term maps to more than one column (low confidence)."""
    return any(isinstance(v, list) and len(v) > 1 for v in column_map.values())


def build_clarification(column_map: dict) -> dict | None:
    """ONE short question, exposing column names only (never rows)."""
    for term, value in column_map.items():
        if isinstance(value, list) and len(value) > 1:
            options = list(value)
            if len(options) == 2:
                listed = f"{options[0]} or {options[1]}"
            else:
                listed = ", ".join(options[:-1]) + f" or {options[-1]}"
            return {
                "term": term,
                "options": options,
                "question": f"By '{term}', do you mean {listed}?",
            }
    return None


def apply_clarifications(column_map: dict, clarifications: dict[str, str], columns: list[str]) -> dict:
    """Resolve ambiguous terms with the user's answers (validated against real columns)."""
    lookup = {c.lower(): c for c in columns}
    resolved = dict(column_map)
    for term, chosen in (clarifications or {}).items():
        col = lookup.get(str(chosen).strip().lower())
        if col:
            resolved[str(term).lower()] = col
    return resolved
