import json

from app.agent import AnalyticsAgent
from app.evaluation import DEFAULT_TEST_CASES, compute_metrics, run_eval
from tests.conftest import ScriptedLLM


def test_red_team_set_has_at_least_12_prompts():
    assert sum(c["category"] == "red_team" for c in DEFAULT_TEST_CASES) >= 12


def test_run_eval_end_to_end(employees, settings, tmp_path):
    llm = ScriptedLLM(codes=["result = df['salary'].mean()"])
    agent = AnalyticsAgent(llm, settings)
    report = run_eval(DEFAULT_TEST_CASES, agent=agent, dataset=employees, out_dir=tmp_path, settings=settings)

    m = report["metrics"]
    assert m["rejection_recall"] == 1.0          # every attack blocked
    assert m["rejection_precision"] == 1.0       # nothing valid rejected
    assert m["success_rate"] == 1.0
    assert m["clarification_accuracy"] == 1.0
    assert m["overall_accuracy"] == 1.0
    assert (tmp_path / "eval_report.json").exists()
    assert (tmp_path / "eval_report.csv").exists()
    assert "Which attacks were blocked?" in (tmp_path / "eval_summary.md").read_text()
    saved = json.loads((tmp_path / "eval_report.json").read_text())
    assert saved["results"][0]["trace"].startswith("classify_request")


def test_repair_rate_counts_repaired_queries(employees, settings):
    llm = ScriptedLLM(codes=["result = df['salry'].mean()"], repairs=["result = df['salary'].mean()"])
    agent = AnalyticsAgent(llm, settings)
    cases = [{"question": "What is the average salary?", "expected": "answer", "category": "valid"}]
    report = run_eval(cases, agent=agent, dataset=employees, settings=settings)
    assert report["metrics"]["repair_rate"] == 1.0
    assert report["results"][0]["repairs"] == 1


def test_compute_metrics_handles_false_positives():
    results = [
        {"expected": "answer", "outcome": "reject", "passed": False, "steps": 2, "repairs": 0, "attempts": 0},
        {"expected": "reject", "outcome": "reject", "passed": True, "steps": 2, "repairs": 0, "attempts": 0},
    ]
    m = compute_metrics(results)
    assert m["rejection_precision"] == 0.5 and m["success_rate"] == 0.0
