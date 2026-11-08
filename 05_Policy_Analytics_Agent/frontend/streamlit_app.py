"""Streamlit front-end for the analytics agent.

    streamlit run frontend/streamlit_app.py

Talks to the FastAPI service over HTTP (default http://localhost:8000).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Policy-Aware Analytics Agent", page_icon="📊", layout="wide")

DEFAULT_API = os.getenv("API_URL", "http://localhost:8000")

OUTCOME_BADGE = {
    "answer": ("✅ Answer", "success"),
    "reject": ("⛔ Rejected by policy", "error"),
    "clarify": ("❓ Needs clarification", "info"),
    "fail": ("⚠️ Could not compute", "warning"),
}

EXAMPLES = {
    "employees": [
        "What is the average salary by dept?",
        "What is the average exp of employees?",
        "How many employees are in each department?",
        "What is the average money per employee?",
        "Show all employee records",
        "Give me Omar's salary",
    ],
    "sales_dataset": [
        "What is the total revenue by region?",
        "Which product category has the highest average revenue?",
        "How many orders per region?",
        "Show me the first 10 rows",
        "Export to CSV",
    ],
}


# ----------------------------------------------------------------------------- API
def api(method: str, path: str, **kwargs):
    url = st.session_state.api_url.rstrip("/") + path
    try:
        resp = httpx.request(method, url, timeout=kwargs.pop("timeout", 120), **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the API at {url} ({type(exc).__name__}). Is it running?"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, str(detail)
    return resp.json(), None


# --------------------------------------------------------------------------- state
st.session_state.setdefault("api_url", DEFAULT_API)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)  # {"question":..., "clarifications": {...}}


def ask(question: str, dataset_id: str, clarifications: dict | None = None) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    data, err = api("POST", "/ask", json={
        "question": question, "dataset_id": dataset_id, "clarifications": clarifications or {},
    })
    if err:
        st.session_state.messages.append({"role": "assistant", "error": err})
    else:
        st.session_state.messages.append({"role": "assistant", "data": data, "question": question})


# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📊 Analytics Agent")
    st.text_input("API URL", key="api_url")
    health, health_err = api("GET", "/health", timeout=5)
    if health_err:
        st.error(health_err)
        st.stop()
    st.caption(f"Model: `{health['model']}` · sandbox: `{health['sandbox_mode']}`")
    if not health["llm_configured"]:
        st.warning("OPENROUTER_API_KEY is not set on the server. Policy rejections still work, "
                   "but answers need the model.")

    datasets, err = api("GET", "/datasets")
    if err or not datasets:
        st.error(err or "No datasets available.")
        st.stop()
    ids = [d["id"] for d in datasets]
    default_idx = next((i for i, d in enumerate(datasets) if d["is_default"]), 0)
    dataset_id = st.selectbox(
        "Dataset", ids, index=default_idx,
        format_func=lambda i: next(f"{d['name']} ({d['rows']} rows)" for d in datasets if d["id"] == i),
    )

    schema, err = api("GET", f"/datasets/{dataset_id}/schema")
    if schema:
        with st.expander("Schema (names only, never rows)", expanded=False):
            st.dataframe(
                pd.DataFrame(schema["columns"]).assign(
                    protected=lambda d: d["protected"].map({True: "🔒 hidden", False: ""})
                ),
                hide_index=True, width="stretch",
            )

    with st.expander("Upload your own CSV / Excel"):
        up = st.file_uploader("File", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if up is not None and st.button("Upload dataset"):
            data, err = api("POST", "/datasets/upload", files={"file": (up.name, up.getvalue())})
            if err:
                st.error(err)
            else:
                st.success(f"Loaded as '{data['id']}'. Protected columns: {data['protected_columns'] or 'none'}")
                st.rerun()

    st.subheader("Try an example")
    for q in EXAMPLES.get(dataset_id, []):
        if st.button(q, key=f"ex_{q}", width="stretch"):
            st.session_state.pending = {"question": q, "clarifications": {}}

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.header("Ask about your data")
st.caption("Aggregated answers only - raw rows, individuals and exports are blocked by policy.")


# ------------------------------------------------------------------------- render
def render_result(result: dict | None) -> None:
    if not result:
        return
    if result["kind"] == "scalar":
        st.metric("Result", result["value"])
        return
    frame = pd.DataFrame(result["rows"], columns=result["columns"])
    st.dataframe(frame, hide_index=True, width="stretch")
    numeric = frame.select_dtypes("number").columns
    if len(frame) > 1 and len(frame.columns) >= 2 and len(numeric) >= 1:
        label_col = next((c for c in frame.columns if c not in numeric), frame.columns[0])
        st.bar_chart(frame.set_index(label_col)[numeric[0]])


def render_assistant(msg: dict, idx: int, is_last: bool) -> None:
    if "error" in msg:
        st.error(msg["error"])
        return
    data = msg["data"]
    label, kind = OUTCOME_BADGE.get(data["outcome"], (data["outcome"], "info"))
    getattr(st, kind)(label)
    st.write(data["message"])

    if data["outcome"] == "clarify" and data.get("clarification") and is_last:
        c = data["clarification"]
        cols = st.columns(len(c["options"]))
        for col, option in zip(cols, c["options"]):
            if col.button(option, key=f"clar_{idx}_{option}"):
                st.session_state.pending = {
                    "question": msg["question"], "clarifications": {c["term"]: option},
                }
                st.rerun()

    render_result(data.get("result"))

    cap = f"{data['steps']} steps · {data['repairs']} repair(s) · {data['latency_s']}s"
    st.caption(cap)
    with st.expander("Agent trace"):
        st.dataframe(pd.DataFrame(data["trace"]), hide_index=True, width="stretch")
        if data.get("column_map"):
            st.write("Column map:", data["column_map"])
    if data.get("code"):
        with st.expander("Generated code"):
            st.code(data["code"], language="python")


last_idx = len(st.session_state.messages) - 1
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render_assistant(msg, i, is_last=(i == last_idx))

typed = st.chat_input("e.g. What is the average salary by department?")
if typed:
    st.session_state.pending = {"question": typed, "clarifications": {}}

if st.session_state.pending:
    job, st.session_state.pending = st.session_state.pending, None
    with st.spinner("Thinking..."):
        ask(job["question"], dataset_id, job["clarifications"])
    st.rerun()
