"""Shared fixtures: a tiny deterministic dataset and a scripted fake LLM (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import Settings
from app.datasets import Dataset

GOOD_CODE = "result = df.groupby('department')['salary'].mean().reset_index()"


class ScriptedLLM:
    """Fake LLM that routes by the system prompt, so each stage is controllable.

    codes:   replies for the code-generation stage, consumed in order (last one repeats)
    repairs: replies for the repair stage, consumed in order (last one repeats)
    """

    def __init__(self, codes=None, repairs=None, mapping="{}", answer="Test answer."):
        self.codes = list(codes or [f"```python\n{GOOD_CODE}\n```"])
        self.repairs = list(repairs or [f"```python\n{GOOD_CODE}\n```"])
        self.mapping = mapping
        self.answer = answer
        self.stages: list[str] = []

    @staticmethod
    def _next(queue: list[str]) -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def chat(self, messages, max_tokens=512, temperature=0.0):
        system = messages[0]["content"]
        if "schema mapper" in system:
            self.stages.append("mapping")
            return self.mapping
        if "fixing broken" in system:
            self.stages.append("repair")
            return self._next(self.repairs)
        if "writing pandas code" in system:
            self.stages.append("code")
            return self._next(self.codes)
        self.stages.append("answer")
        return self.answer


@pytest.fixture
def settings() -> Settings:
    return Settings(sandbox_mode="thread", sandbox_timeout_s=5.0, max_retries=5)


@pytest.fixture
def employees_df() -> pd.DataFrame:
    return pd.DataFrame({
        "employee_id": range(1, 13),
        "name": ["Omar Hassan", "Sara Ali", "Ahmed Nasser", "Laila Saleh", "Youssef Adel", "Nour Samir",
                 "Hana Khalil", "Karim Fawzy", "Mona Ibrahim", "Tarek Mansour", "Salma Mahmoud", "Ali Nasser"],
        "department": ["Eng", "Eng", "Eng", "Eng", "Sales", "Sales", "Sales", "HR", "HR", "HR", "Fin", "Fin"],
        "years_experience": [1, 3, 5, 8, 2, 4, 6, 3, 7, 10, 2, 9],
        "salary": [5000, 6000, 7000, 9000, 4000, 4500, 5500, 3500, 4800, 6200, 5200, 8000],
        "bonus": [500, 600, 700, 900, 400, 450, 550, 350, 480, 620, 520, 800],
    })


@pytest.fixture
def employees(employees_df, settings) -> Dataset:
    return Dataset.from_dataframe("employees", employees_df, settings.protected_patterns)
