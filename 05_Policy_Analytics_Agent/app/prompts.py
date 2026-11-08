from __future__ import annotations

import json

from .llm import Messages


def code_messages(question: str, schema: dict, column_map: dict | None, max_rows: int) -> Messages:
    mapping_note = (
        f"\nUser term -> column mapping (use these exact column names): {json.dumps(column_map)}"
        if column_map
        else ""
    )
    system = f"""You are a careful data analyst writing pandas code.
A pandas DataFrame named `df` is already loaded, and `pd` (a limited pandas namespace) is available.

Dataset schema (only these columns exist): {json.dumps(schema)}{mapping_note}

STRICT RULES:
- Use ONLY the column names listed above, spelled exactly as written.
- Return AGGREGATED answers only: mean, sum, count, min, max, median, nunique, value_counts,
  groupby(...).agg(...), pivot_table, etc. Never return raw rows.
- Do NOT use head(), tail(), sample(), iloc, loops, functions, classes, comprehensions or lambdas that print.
- Do NOT import anything. Do NOT print. Do NOT read or write files.
- The final answer must be stored in a variable named `result` (a number, a Series or a small DataFrame
  of at most {max_rows} rows).
- Output ONLY executable Python code inside one ```python block. No explanations."""
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def repair_prompt(schema: dict, question: str, code: str, error: str, max_rows: int = 50) -> Messages:
    """Ask the model to fix broken code. (Deliverable: repair_prompt(schema, question, code, error))."""
    system = f"""You are a data analyst fixing broken pandas code.

Dataset schema (only these columns exist): {json.dumps(schema)}

STRICT RULES:
- Use ONLY the column names listed above, spelled exactly as written.
- The answer must be AGGREGATED (mean/sum/count/groupby-agg/value_counts...). Never raw rows.
- Do NOT import anything, print, loop, define functions, or touch files.
- Store the final answer in a variable named `result` (at most {max_rows} rows).
- Output ONLY the corrected Python code inside one ```python block."""
    user = f"""Original question: {question}

Broken code:
```python
{code}
```

Error received:
{error}

Fix the code so it runs correctly and respects every rule."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mapping_messages(question: str, columns: list[str]) -> Messages:
    system = f"""You are a schema mapper. Map the user's words to actual column names.
Available columns: {json.dumps(columns)}

Return ONLY a JSON object {{"<word from question>": "<column>"}}, for example
{{"dept": "department", "exp": "years_experience"}}.
- Include only words that appear in the question and are NOT already an exact column name.
- If a word could refer to several columns, map it to a list: {{"money": ["salary", "bonus"]}}.
- If nothing needs mapping return {{}}.
Output ONLY valid JSON."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]


def answer_messages(question: str, result_text: str) -> Messages:
    system = (
        "You explain analytics results to a business user in plain English. "
        "Use ONLY the numbers in the result; never invent values. "
        "Answer in at most 3 sentences. Do not output JSON or code."
    )
    user = f"Question: {question}\nAggregated result:\n{result_text}\n\nExplain this result."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
