import pytest

from app.errors import CodeValidationError, ExecutionError, OutputPolicyError, SandboxTimeout
from app.sandbox import run_in_sandbox


def run(code, df, **kw):
    return run_in_sandbox(code, df, ["name"], **kw)


def test_scalar_result(employees_df):
    out = run("result = df['salary'].mean()", employees_df)
    assert out == {"kind": "scalar", "value": round(employees_df["salary"].mean(), 4)}


def test_table_result(employees_df):
    out = run("result = df.groupby('department')['salary'].mean().reset_index()", employees_df)
    assert out["kind"] == "table"
    assert out["columns"] == ["department", "salary"]
    assert {r["department"] for r in out["rows"]} == {"Eng", "Sales", "HR", "Fin"}


def test_series_with_named_index_is_flattened(employees_df):
    out = run("result = df['department'].value_counts()", employees_df)
    assert out["kind"] == "table" and len(out["rows"]) == 4


def test_dataframe_is_not_mutated(employees_df):
    before = employees_df.copy()
    run("df['salary'] = 0\nresult = df['salary'].sum()", employees_df)
    assert employees_df.equals(before)


def test_code_error_is_wrapped(employees_df):
    with pytest.raises(ExecutionError) as exc:
        run("result = df['salry'].mean()", employees_df)
    assert "KeyError" in str(exc.value)


def test_validation_runs_before_execution(employees_df):
    with pytest.raises(CodeValidationError):
        run("import os\nresult = df['salary'].mean()", employees_df)


def test_builtins_are_restricted(employees_df):
    # `list` is not in the safe builtins -> NameError inside the sandbox
    with pytest.raises(ExecutionError) as exc:
        run("result = len(list(df['salary'])) + df['salary'].sum()", employees_df)
    assert "NameError" in str(exc.value)


def test_row_limit_enforced(employees_df):
    with pytest.raises(OutputPolicyError):
        run("result = df.groupby('employee_id')['salary'].mean()", employees_df, max_rows=5)


def test_raw_row_sized_result_blocked(employees_df):
    with pytest.raises(OutputPolicyError):
        run("result = df.groupby('employee_id')['salary'].mean()", employees_df)  # 12 groups == 12 rows


def test_timeout(employees_df, monkeypatch):
    import app.sandbox as sb

    def slow(*a, **k):
        import time
        time.sleep(2)

    monkeypatch.setattr(sb, "_execute", slow)
    with pytest.raises(SandboxTimeout):
        run("result = df['salary'].mean()", employees_df, timeout=0.3)


def test_process_mode(employees_df):
    out = run("result = df['salary'].max()", employees_df, mode="process", timeout=60)
    assert out["value"] == 9000


def test_process_mode_reports_errors(employees_df):
    with pytest.raises(ExecutionError):
        run("result = df['nope'].max()", employees_df, mode="process", timeout=60)
