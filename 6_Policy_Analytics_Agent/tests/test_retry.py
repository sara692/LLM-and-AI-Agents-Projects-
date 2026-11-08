import logging

from app.analysis import GRACEFUL_FAILURE_MESSAGE, execute_with_retry, extract_code
from tests.conftest import GOOD_CODE, ScriptedLLM

BAD = "result = df.groupby('department')['salry'].mean().reset_index()"  # typo'd column


def test_extract_code_variants():
    assert extract_code("Sure!\n```python\nresult = 1\n```\nbye") == "result = 1"
    assert extract_code("import pandas as pd\n# c\nresult = df['a'].sum()") == "result = df['a'].sum()"
    assert extract_code("") == ""


def test_success_first_try(employees, settings):
    llm = ScriptedLLM(codes=[GOOD_CODE])
    out = execute_with_retry("avg salary by dept", employees, llm, settings=settings)
    assert out.success and out.attempts == 1 and out.repairs == 0


def test_failure_then_repair_then_success(employees, settings, caplog):
    """One failure -> repair -> success, with the sequence visible in the logs."""
    llm = ScriptedLLM(codes=[BAD], repairs=[GOOD_CODE])
    with caplog.at_level(logging.INFO, logger="agent.analysis"):
        out = execute_with_retry("avg salary by dept", employees, llm, settings=settings)
    assert out.success
    assert out.attempts == 2 and out.repairs == 1
    assert "KeyError" in out.errors[0]
    log = caplog.text
    assert log.index("FAILED") < log.index("repair the code") < log.index("SUCCEEDED")
    assert [e[0] for e in out.events] == ["generate_code", "execute_code", "repair_code", "execute_code"]


def test_gives_up_after_max_retries(employees, settings):
    llm = ScriptedLLM(codes=[BAD], repairs=[BAD])
    out = execute_with_retry("avg salary by dept", employees, llm, settings=settings, max_retries=3)
    assert not out.success
    assert out.attempts == 3 and out.repairs == 2 and len(out.errors) == 3
    assert llm.stages.count("repair") == 2


def test_validation_errors_also_trigger_repair(employees, settings):
    llm = ScriptedLLM(codes=["import os\nresult = df['salary'].mean()"], repairs=["result = df['salary'].mean()"])
    out = execute_with_retry("avg salary", employees, llm, settings=settings)
    assert out.success and out.repairs == 1
    assert "Forbidden" in out.errors[0]


def test_accepts_plain_dataframe(employees_df, settings):
    out = execute_with_retry("avg", employees_df, ScriptedLLM(codes=["result = df['salary'].mean()"]), settings=settings)
    assert out.success


def test_graceful_message_constant():
    assert GRACEFUL_FAILURE_MESSAGE == "I could not compute this safely. Try rephrasing."
