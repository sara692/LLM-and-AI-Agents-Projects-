# Project walkthrough

## How a question flows

```
Streamlit (UI) → FastAPI (/ask) → AnalyticsAgent.run()
   1. policy.classify_request       → reject?
   2. schema_mapping.infer_column_map → ask clarification?
   3. analysis.execute_with_retry   → LLM writes code → validate → sandbox → repair loop
   4. LLM explains the aggregated result
```

---

## `app/` (the core)

### `config.py`: settings

- **`Settings`**: one frozen dataclass holding every option (API key, model name, retry count, sandbox timeout, max result rows, protected-column patterns).
- **`Settings.from_env()`**: reads these from environment variables or `.env`.
- **`get_settings()`**: returns one cached copy for the whole app.
- **`_env_int/_env_float/_env_bool/_env_list`**: small helpers that parse env values safely, falling back to defaults.

### `errors.py`: exception types

A small hierarchy so each failure can be handled separately:

- **`LLMError`**: the model is unreachable or returned junk.
- **`CodeValidationError`**: the AST validator rejected the code.
- **`ExecutionError`**: the code crashed while running.
- **`SandboxTimeout`**: the code ran too long.
- **`OutputPolicyError`**: the result broke the "aggregated only" rule.

### `logging_config.py`

- **`setup_logging()`**: configures console logs, so you can see the "attempt failed → repair → succeeded" sequence.

### `llm.py`: talking to OpenRouter

- **`LLMClient`**: a tiny interface (`chat(messages)`). The agent depends only on this, so tests can plug in a fake.
- **`OpenRouterClient.chat()`**: sends the request to OpenRouter and retries on 429/5xx errors. It supports fallback models and raises `LLMError` on problems.
- **`OpenRouterClient._parse()`**: pulls the text out of the response and checks it's valid and non-empty.
- **`parse_json_object()`**: extracts a JSON object from messy model output (markdown fences, extra prose). It never trusts raw output.

### `policy.py`: the first gate (no LLM involved)

- **`classify_request(question, ...)`**: checks the question against regex rules and returns allow or reject. It catches:
  - prompt injection ("ignore the rules")
  - code ("df.head()")
  - exports
  - raw-data requests ("show all rows")
  - debug dumps
  - individual lookups ("Omar's salary", or any real name from the data)
- **`PolicyDecision`**: the result (`authorized`, `rule`, `reason`).
- **`RULE_MESSAGES`**: the user-friendly text for each rule.

### `code_validation.py`: inspects LLM code before it runs

- **`validate_code(code, protected_columns)`**: parses the code into an AST (without executing it) and rejects:
  - imports, loops, comprehensions, functions and `try`/`with`
  - dangerous names (`eval`, `open`, `getattr`…)
  - dangerous attributes (`__class__`, `to_csv`, `read_*`, `.query`, `.format`, `iloc`…)
  - any reference to a protected column
  - code with no aggregation (`mean`, `sum`, `groupby…`)
  - `head()`/`nlargest()` applied to raw rows
  - code that never assigns `result`
- **`_attr_denied()`, `_has_reducer()`**: helpers for the attribute check and the "is this aggregating?" check.

### `sandbox.py`: runs validated code in a restricted environment

- **`run_in_sandbox()`**: validates, then executes the code with a timeout, in either a thread or a child process.
- **`_execute()`**: runs the code with minimal builtins, a private copy of the dataframe, and a pandas namespace stripped of file functions. It reads `result` at the end.
- **`serialize_result()`**: the output guard. It turns the result into JSON and rejects it if there are too many rows (over 50), it looks like raw rows, or it contains protected columns.
- **`_run_in_thread()` / `_run_in_process()`**: the two timeout strategies. Process mode can kill runaway code.
- **`SAFE_BUILTINS`, `SAFE_PD`**: the only tools the generated code can use.

### `datasets.py`: loading data and hiding PII

- **`Dataset`**: wraps a dataframe plus its protected columns and known names.
  - **`from_dataframe()`**: detects PII columns by name pattern and records the names in them (so the policy can spot "Omar").
  - **`visible_columns`**: the columns the LLM and user prompts may see.
  - **`public_schema()`**: names, types and protected flags for the API (no rows).
  - **`llm_schema()`**: what the LLM receives: column names, types and category labels, never rows.
- **`DatasetStore`**: in-memory registry.
  - **`load_directory()`**: loads the CSVs from `data/`.
  - **`add_upload()`**: validates and loads user uploads (size, format, duplicates).
  - **`get()` / `list()` / `default_id`**: look up datasets.
- **`is_text_series()`**: detects text columns, including pandas 3's new `str` dtype (this fixed a bug found during testing).

### `prompts.py`: all prompt text in one place

- **`code_messages()`**: the instructions for writing pandas code, with the schema and strict rules.
- **`repair_prompt(schema, question, code, error)`**: asks the model to fix its broken code given the error.
- **`mapping_messages()`**: asks the model to map user words to columns.
- **`answer_messages()`**: asks the model to explain the aggregated result in 3 sentences or fewer.

### `schema_mapping.py`: Level 2, vocabulary and ambiguity

- **`SYNONYMS`**: a table (`dept→department`, `exp→years_experience`, `money→salary/bonus`…).
- **`deterministic_map()`**: maps terms using the table (no LLM). One match means a confident mapping; several means ambiguous.
- **`infer_column_map(question, columns, llm)`**: combines the table with the LLM's suggestions, and **validates every LLM answer** against real columns.
- **`needs_clarification()`**: true if any term maps to several columns.
- **`build_clarification()`**: builds one short question, such as "By 'money', do you mean salary or bonus?".
- **`apply_clarifications()`**: applies the user's answer to resolve the ambiguity.

### `analysis.py`: Level 1, self-correction

- **`extract_code()`**: pulls Python out of the model reply and removes harmless `import pandas`.
- **`generate_code()`**: asks the LLM for the first version of the code.
- **`execute_with_retry()`**: the repair loop. It runs the code; on failure it sends the error to the LLM to fix, then retries, up to 5 attempts. It never crashes, and returns an `ExecutionOutcome` with attempts, repairs and events.
- **`GRACEFUL_FAILURE_MESSAGE`**: "I could not compute this safely. Try rephrasing."

### `agent.py`: the orchestrator

- **`AnalyticsAgent.run()`**: runs the whole workflow and **never raises**; any unexpected error becomes a safe failure response.
- **`AnalyticsAgent._run()`**: the steps in order: policy → mapping → clarify → analysis → answer, recording each step in a trace.
- **`AnalyticsAgent._explain()`**: has the LLM phrase the answer, with a plain fallback if the LLM fails.
- **`AgentResponse` / `TraceStep`**: the output structure (outcome, answer, result, code, repairs, trace…).
- **`format_result_text()`**: turns the result into compact text for the explanation prompt.

### `evaluation.py`: Level 3, red-team and metrics

- **`DEFAULT_TEST_CASES`**: 20 prompts (5 valid, 2 ambiguous, 13 attacks).
- **`run_eval()`**: runs each case through the real agent and records trace, outcome and repairs. It writes JSON, CSV and a markdown summary.
- **`compute_metrics()`**: success rate, rejection precision and recall, clarification rate, average steps and repair rate.
- **`build_summary()`**: the written report: what failed, which attacks were blocked, the biggest remaining risk and the next improvement.
- **`main()`**: the command-line entry point (`python -m app.evaluation`).

### `api.py`: FastAPI service

- **`get_store()` / `get_agent()`**: create the dataset store and the agent once, and share them.
- **`GET /health`**: status, model name, and whether the API key is set.
- **`GET /datasets`, `GET /datasets/{id}/schema`**: list datasets and show their columns.
- **`POST /datasets/upload`**: upload your own CSV or Excel file.
- **`POST /classify`**: policy check only, no LLM call.
- **`POST /ask`**: runs the agent.
- **`AskRequest` / `AskResponse`**: the input and output models.

---

## `frontend/streamlit_app.py`: the UI

- **`api()`**: wrapper for HTTP calls to FastAPI, with friendly errors.
- **`ask()`**: sends a question and stores the reply in the chat.
- **Sidebar**: dataset picker, schema viewer (PII shown as hidden), file upload, and example questions.
- **`render_result()`**: shows a number or a table, plus a bar chart when it fits.
- **`render_assistant()`**: shows the status badge, the answer, the clarification buttons, the trace, and the generated code.

## `scripts/generate_sample_data.py`

- **`make_employees()`**, **`make_sales()`**: create reproducible fake datasets, saved by **`main()`** into `data/`.

## `tests/`

- **`conftest.py`**: `ScriptedLLM` is a fake model so tests need no network, plus the shared fixtures.
- **`test_policy.py`**: attacks are rejected and valid questions pass.
- **`test_code_validation.py`**: safe and unsafe code samples.
- **`test_sandbox.py`**: results, restricted builtins, row limits, timeouts and process mode.
- **`test_retry.py`**: failure → repair → success (checked in the logs), and giving up after the limit.
- **`test_schema_mapping.py`**: synonyms, ambiguity, hallucinated LLM columns.
- **`test_agent.py`**: the full workflow, including clarify, reject, graceful failure and LLM outage.
- **`test_evaluation.py`**: metrics and report files.
- **`test_api.py`**: endpoints, upload and the clarification flow.

## Other files

- **`requirements.txt`**: the Python dependencies.
- **`.env.example`**: template for your API key and options (copy to `.env`).
- **`pytest.ini`**: test configuration.
- **`.gitignore`**: keeps `.env`, caches and reports out of git.
- **`data/*.csv`**: the sample employees and sales data.
- **`README.md`**: setup, architecture, API reference and limitations.
