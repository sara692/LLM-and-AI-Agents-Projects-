import pytest

from app.code_validation import validate_code
from app.errors import CodeValidationError

PROTECTED = ["name"]

SAFE = [
    "result = df['salary'].mean()",
    "result = df.groupby('department')['salary'].mean().reset_index()",
    "result = df.groupby('department').agg(avg=('salary', 'mean'), n=('salary', 'count'))",
    "tmp = df.groupby('department')['salary'].sum()\nresult = tmp.sort_values(ascending=False).head(3)",
    "result = df['department'].value_counts()",
    "result = df.groupby('department')['salary'].mean().nlargest(2)",
    "result = df[df['salary'] > 5000]['salary'].mean()",
    "result = df['salary'].apply(lambda x: x * 2).sum()",
]

UNSAFE = [
    ("import os\nresult = df['salary'].mean()", "Forbidden operation"),
    ("from os import path\nresult = df['salary'].mean()", "Forbidden operation"),
    ("for r in df['salary']:\n    result = r", "Forbidden operation"),
    ("result = [x for x in df['salary']]", "Forbidden operation"),
    ("def f():\n    return 1\nresult = df['salary'].mean()", "Forbidden operation"),
    ("with open('x') as f:\n    result = df['salary'].mean()", "Forbidden operation"),
    ("result = eval('1+1')", "Forbidden name"),
    ("result = open('data.csv').read()", "Forbidden name"),
    ("print(df['salary'].mean())\nresult = 1", "Forbidden name"),
    ("result = df.__class__", "Forbidden"),
    ("result = df['salary'].mean().__class__.__mro__", "Forbidden"),
    ("result = getattr(df, 'salary').mean()", "Forbidden name"),
    ("result = df.to_csv('x.csv')", "Forbidden attribute"),
    ("result = df['salary'].mean() and df.to_json()", "Forbidden attribute"),
    ("result = pd.read_csv('x.csv')", "Forbidden attribute"),
    ("result = df.query('salary > 1').salary.mean()", "Forbidden attribute"),
    ("result = '{0.__class__}'.format(df)", "Forbidden"),
    ("result = df['name'].value_counts()", "protected column"),
    ("result = df.groupby('name')['salary'].mean()", "protected column"),
    ("result = df.name.nunique()", "protected column"),
    ("result = df.head(5)", "aggregated"),
    ("result = df[df['salary'] > 5000]", "aggregated"),
    ("result = df['salary'].head(3)", "aggregated"),
    ("result = df['salary'].mean() + df['salary'].head(3).sum()", "raw rows"),
    ("result = df.sort_values('salary').nlargest(3, 'salary')['salary'].sum()", "raw rows"),
    ("x = df['salary'].mean()", "result"),
    ("result = df['salary'].mean(", "SyntaxError"),
    ("", "No code"),
]


@pytest.mark.parametrize("code", SAFE)
def test_safe_code_passes(code):
    validate_code(code, PROTECTED)


@pytest.mark.parametrize("code,fragment", UNSAFE)
def test_unsafe_code_is_rejected(code, fragment):
    with pytest.raises(CodeValidationError) as exc:
        validate_code(code, PROTECTED)
    assert fragment.lower() in str(exc.value).lower()
