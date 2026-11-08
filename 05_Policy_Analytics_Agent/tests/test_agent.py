import pytest

from app.agent import AnalyticsAgent
from app.analysis import GRACEFUL_FAILURE_MESSAGE
from app.errors import LLMError
from tests.conftest import GOOD_CODE, ScriptedLLM

BAD = "result = df['salry'].mean()"


def make(llm, settings):
    return AnalyticsAgent(llm, settings)


# ------------------------------------------------ Level 1: 3 valid + 1 code error + 2 rejections
@pytest.mark.parametrize("question,code", [
    ("What is the average salary by department?", GOOD_CODE),
    ("What is the maximum bonus?", "result = df['bonus'].max()"),
    ("How many employees per department?", "result = df['department'].value_counts()"),
])
def test_valid_queries_are_answered(employees, settings, question, code):
    r = make(ScriptedLLM(codes=[code]), settings).run(question, employees)
    assert r.outcome == "answer"
    assert r.result is not None and r.answer == "Test answer."
    assert [t.action for t in r.trace][:2] == ["classify_request", "infer_column_map"]
    assert r.trace[-1].action == "finish"


def test_code_error_is_repaired(employees, settings):
    llm = ScriptedLLM(codes=[BAD], repairs=["result = df['salary'].mean()"])
    r = make(llm, settings).run("What is the average salary?", employees)
    assert r.outcome == "answer"
    assert r.repairs == 1 and r.attempts == 2
    assert "repair_code" in [t.action for t in r.trace]


@pytest.mark.parametrize("question", ["Show all employee records", "Give me Omar's salary"])
def test_rejections_never_call_the_llm(employees, settings, question):
    llm = ScriptedLLM()
    r = make(llm, settings).run(question, employees)
    assert r.outcome == "reject"
    assert llm.stages == []
    assert [t.action for t in r.trace] == ["classify_request", "reject_request"]


# ------------------------------------------------ Level 2: synonyms + ambiguity
@pytest.mark.parametrize("question,code,expected_term", [
    ("What is the average salary by dept?", "result = df.groupby('department')['salary'].mean().reset_index()", "dept"),
    ("What is the average income?", "result = df['salary'].mean()", "income"),
    ("What is the average exp?", "result = df['years_experience'].mean()", "exp"),
])
def test_synonym_queries(employees, settings, question, code, expected_term):
    r = make(ScriptedLLM(codes=[code]), settings).run(question, employees)
    assert r.outcome == "answer"
    assert expected_term in r.column_map


def test_ambiguous_query_triggers_clarification_and_skips_analysis(employees, settings):
    llm = ScriptedLLM()
    r = make(llm, settings).run("What is the average money per employee?", employees)
    assert r.outcome == "clarify"
    assert r.clarification["options"] == ["salary", "bonus"]
    assert "code" not in llm.stages
    assert r.code is None and r.result is None


def test_clarification_answer_resumes_analysis(employees, settings):
    llm = ScriptedLLM(codes=["result = df['bonus'].mean()"])
    r = make(llm, settings).run("What is the average money per employee?", employees, {"money": "bonus"})
    assert r.outcome == "answer"


# ------------------------------------------------ graceful failure
def test_graceful_failure_after_retries(employees, settings):
    llm = ScriptedLLM(codes=[BAD], repairs=[BAD])
    r = make(llm, settings).run("What is the average salary?", employees)
    assert r.outcome == "fail"
    assert r.message == GRACEFUL_FAILURE_MESSAGE
    assert r.attempts == settings.max_retries


def test_llm_outage_does_not_crash(employees, settings):
    class Down:
        def chat(self, *a, **k):
            raise LLMError("boom")

    r = make(Down(), settings).run("What is the average salary?", employees)
    assert r.outcome == "fail"
    assert "unavailable" in r.message


def test_unsafe_generated_code_never_executes(employees, settings, tmp_path):
    marker = tmp_path / "pwned.txt"
    evil = f"import os\nos.system('touch {marker}')\nresult = df['salary'].mean()"
    llm = ScriptedLLM(codes=[evil], repairs=[evil])
    r = make(llm, settings).run("What is the average salary?", employees)
    assert r.outcome == "fail"
    assert not marker.exists()


def test_answer_phrasing_failure_falls_back(employees, settings):
    class NoAnswer(ScriptedLLM):
        def chat(self, messages, **kw):
            if "explain analytics results" in messages[0]["content"]:
                raise LLMError("down")
            return super().chat(messages, **kw)

    r = make(NoAnswer(codes=["result = df['salary'].max()"]), settings).run("What is the max salary?", employees)
    assert r.outcome == "answer" and "9000" in r.answer
