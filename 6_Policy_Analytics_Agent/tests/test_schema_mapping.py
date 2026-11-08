from app.schema_mapping import (
    apply_clarifications,
    build_clarification,
    infer_column_map,
    needs_clarification,
)
from tests.conftest import ScriptedLLM

COLS = ["employee_id", "department", "years_experience", "salary", "bonus"]


def test_synonyms_map_without_llm():
    assert infer_column_map("average salary by dept", COLS) == {"dept": "department"}
    assert infer_column_map("average exp of staff", COLS) == {"exp": "years_experience"}
    assert infer_column_map("average income", COLS) == {"income": "salary"}


def test_ambiguous_term_requires_clarification():
    cmap = infer_column_map("average money per employee", COLS)
    assert cmap["money"] == ["salary", "bonus"]
    assert needs_clarification("average money per employee", cmap) is True
    clar = build_clarification(cmap)
    assert clar["question"] == "By 'money', do you mean salary or bonus?"
    assert "employee" not in clar["question"].lower().split("salary")[0]  # column names only


def test_exact_column_names_need_no_mapping():
    assert infer_column_map("average salary by department", COLS) == {}


def test_llm_output_is_validated():
    # hallucinated column + irrelevant term + valid mapping
    llm = ScriptedLLM(mapping='{"dept": "department", "foo": "bar", "pay": "nonexistent", "weird": "salary"}')
    cmap = infer_column_map("average salary by dept", COLS, llm)
    assert cmap == {"dept": "department"}  # 'foo'/'weird' not in question, 'bar'/'nonexistent' not real


def test_llm_list_for_term_in_question_triggers_clarification():
    llm = ScriptedLLM(mapping='{"comp": ["salary", "bonus"]}')
    cmap = infer_column_map("what is the average comp", COLS, llm)
    assert needs_clarification("", cmap)


def test_llm_garbage_is_ignored():
    assert infer_column_map("average salary by dept", COLS, ScriptedLLM(mapping="sorry, no json")) == {"dept": "department"}


def test_apply_clarifications_resolves_ambiguity():
    cmap = {"money": ["salary", "bonus"]}
    resolved = apply_clarifications(cmap, {"money": "Bonus"}, COLS)
    assert resolved == {"money": "bonus"}
    assert not needs_clarification("", resolved)
    # invalid answers are ignored
    assert apply_clarifications(cmap, {"money": "password"}, COLS) == cmap
