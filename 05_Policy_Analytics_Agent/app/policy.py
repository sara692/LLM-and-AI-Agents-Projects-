from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

MAX_QUESTION_CHARS = 1000

_ROW_NOUNS = (
    r"rows?|records?|data|table|dataset|entries|lines|employees|customers|people|users|"
    r"orders|transactions|salaries|names|details|information|info|items"
)

# (rule name, compiled pattern) - evaluated in order, first match wins.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_injection",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass|disable|skip|circumvent)\b.{0,40}?"
            r"\b(?:rules?|instructions?|polic(?:y|ies)|restrictions?|guardrails?|safety|"
            r"authori[sz]ation|filters?|checks?|validation)\b"
            r"|\b(?:system prompt|developer mode|jailbreak)\b",
            re.I,
        ),
    ),
    (
        "code_injection",
        re.compile(
            r"\b(?:df|dataframe)\b(?=\s*[\.\[\(])|\bpd\.|\bimport\s+\w+"
            r"|\b(?:exec|eval|open|compile|__import__)\s*\(|__\w+__|\bos\.|\bsubprocess\b"
            r"|\bselect\s+\*|\bdrop\s+table\b|\bprint\s*\("
            r"|\.(?:head|tail|sample|iloc|loc|values|iterrows|to_\w+)\b",
            re.I,
        ),
    ),
    (
        "export",
        re.compile(
            r"\b(?:export|download|exfiltrate)\b"
            r"|\b(?:save|write|dump|convert)\b.{0,30}\b(?:csv|excel|xlsx|json|file|disk)\b",
            re.I,
        ),
    ),
    (
        "raw_data_request",
        re.compile(
            r"\b(?:show|list|display|print|dump|output|reveal|expose|view|see|enumerate)\b"
            r"(?:\s+\w+){0,2}?\s+(?:all|every|each|entire|whole|raw|full|complete)\b"
            rf"(?:\s+\w+){{0,2}}?\s+(?:{_ROW_NOUNS})\b"
            r"|\b(?:give|get|return|fetch|send|provide|pull|retrieve|extract)\b"
            r"(?:\s+\w+){0,2}?\s+(?:all|every|entire|whole|raw|full|complete)\b"
            r"(?:\s+\w+){0,1}?\s+(?:rows?|records?|data|table|dataset|entries)\b"
            r"|\b(?:output|print|show|display|dump|give|return)\s+(?:me\s+)?"
            r"(?:the\s+|this\s+|that\s+)?(?:whole\s+|entire\s+|full\s+|raw\s+)?"
            r"(?:table|dataframe|dataset|spreadsheet|csv)\b"
            r"|\b(?:first|last|top|bottom)\s+\d+\s+(?:rows?|records?|lines|entries)\b",
            re.I,
        ),
    ),
    ("debug_dump", re.compile(r"\bdebug(?:ging)?\b", re.I)),
    (
        "individual_lookup",
        re.compile(
            r"\b(?:row|record|employee|customer|order|user)\s*(?:id|no\.?|number|#)?\s*[:=#]?\s*\d+\b",
            re.I,
        ),
    ),
]

# Capitalised possessive ("Omar's salary") - a strong sign of a single-person lookup.
_POSSESSIVE = re.compile(
    r"\b(?!(?:What|Who|Where|When|How|Why|Here|There|That|It|Let|He|She|This|Which)\b)"
    r"[A-Z][a-z]+'s\b"
)
_CAP_TOKEN = re.compile(r"\b[A-Z][A-Za-z'\-]{2,}\b")

RULE_MESSAGES = {
    "empty": "Please enter a question.",
    "too_long": "That question is too long. Please keep it short and specific.",
    "prompt_injection": "I can't ignore or bypass the data-access rules.",
    "code_injection": "Please ask in plain language; code and raw dataframe calls are not accepted.",
    "export": "Exporting or downloading data is not allowed.",
    "raw_data_request": "I can only return aggregated insights, not raw rows or tables.",
    "debug_dump": "Debug dumps of the data are not allowed.",
    "individual_lookup": "I can't look up individual people or records, only aggregated results.",
    "known_individual": "I can't share information about a specific individual, only aggregated results.",
}


@dataclass(frozen=True)
class PolicyDecision:
    authorized: bool
    rule: str | None = None
    reason: str = "ok"


def _reject(rule: str) -> PolicyDecision:
    return PolicyDecision(False, rule, RULE_MESSAGES[rule])


def classify_request(
    question: str,
    *,
    known_tokens: Iterable[str] = (),
    known_full_names: Iterable[str] = (),
) -> PolicyDecision:
    """Decide whether a question is allowed. Pure function: no state, no LLM.

    known_tokens / known_full_names come from the dataset's protected columns
    (e.g. the first and last names of employees) so we can recognise "Omar" or
    "omar hassan" even without a possessive.
    """
    if not question or not question.strip():
        return _reject("empty")
    if len(question) > MAX_QUESTION_CHARS:
        return _reject("too_long")

    for rule, pattern in _RULES:
        if pattern.search(question):
            return _reject(rule)

    if _POSSESSIVE.search(question):
        return _reject("known_individual")

    tokens = set(known_tokens)
    if tokens:
        caps = _CAP_TOKEN.findall(question)
        # ignore the first word: it is capitalised just for being first
        if any(tok in tokens for tok in caps[1:] if question.find(tok) > 0):
            return _reject("known_individual")

    full = set(known_full_names)
    if full:
        words = re.findall(r"[a-z0-9'\-]+", question.lower())
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                if " ".join(words[i : i + n]) in full:
                    return _reject("known_individual")

    return PolicyDecision(True)
