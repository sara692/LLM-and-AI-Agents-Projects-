import os
from dotenv import load_dotenv
from typing import Dict
import logging

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough       # ← add this
from langchain_core.messages import trim_messages, HumanMessage, AIMessage

from prompts import chat_prompt
from db import init_db, insert_message, load_history, load_history_paginated, delete_session, search_similar, save_summary, load_summary
from embeddings import embed,_get_embedder

from pathlib import Path

env_path = Path(__file__).parent / ".env"

load_dotenv(dotenv_path=env_path)



MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", 6))
RECENT_WINDOW_SIZE = int(os.getenv("RECENT_WINDOW_SIZE", 6))
SUMMARY_UPDATE_EVERY = int(os.getenv("SUMMARY_UPDATE_EVERY", 4))
HF_MODEL=os.getenv("HF_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")
PG_PATH= os.getenv("pg_path")
embedder_name=os.getenv("embedder_name")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trimmer = trim_messages(
    max_tokens=RECENT_WINDOW_SIZE,          # number of messages to keep
    strategy="last",       # keep the LAST N messages
    token_counter=len,     # count by number of messages, not tokens
    include_system=False,   
    
)

def _get_relevant_history(session_id: str, question: str) -> list:
    """
    Instead of last N messages, return most semantically
    similar N messages to the current question.
    """
    query_vec = embed(question)

    if query_vec is None:
        # fallback: if embedding fails, use last N messages from lc_history
        msgs = _ensure_session(session_id)["lc_history"].messages
        return msgs[-RECENT_WINDOW_SIZE:]

    # search DB for most similar messages
    rows = search_similar(PG_PATH, session_id, query_vec, k=RECENT_WINDOW_SIZE)

    # convert DB rows back to LangChain message objects
    messages = []
    for row in rows:
        if row["role"] == "human":
            messages.append(HumanMessage(content=row["content"]))
        elif row["role"] == "ai":
            messages.append(AIMessage(content=row["content"]))

    return messages

llm = ChatOpenAI(
    model="Qwen/Qwen3-Coder-Next:novita",
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.2,
)

# ✅ Replace with this:
chat_chain = (
    RunnablePassthrough.assign(
        history=lambda x: _get_relevant_history(
            x["session_id"], x["question"]   # ← pass both
        )
    )
    | chat_prompt
    | llm
)

# dict: session_id -> {"lc_history": InMemoryChatMessageHistory, "raw_history": [{"role":..., "content":...}, ...]}
_STORE: Dict[str, Dict[str, any]] = {}

def _ensure_session(session_id: str):
    if session_id not in _STORE:
        lc_hist = InMemoryChatMessageHistory()
        rows = load_history(PG_PATH, session_id)
        for row in rows:
            if row["role"] == "human":
                lc_hist.add_user_message(row["content"])
            elif row["role"] == "ai":
                lc_hist.add_ai_message(row["content"])

        saved_summary = load_summary(PG_PATH, session_id)  # ← restore summary

        _STORE[session_id] = {
            "lc_history": lc_hist,
            "summary": saved_summary,       # ← not empty anymore!
            "turns_since_summary": 0,
        }
    return _STORE[session_id]

def get_history(session_id: str):
    return _ensure_session(session_id)["lc_history"]

chat_with_memory = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ── Summarization ──────────────────────────────────────────────────────────────
def _update_summary(session_id: str) -> None:
    """Generate a rolling summary from old summary + recent raw_history."""
    sess = _ensure_session(session_id)
    rows = load_history(PG_PATH, session_id)

    # Build a readable transcript from raw_history
    transcript = "\n".join(
    f"{r['role'].upper()}: {r['content']}"
    for r in rows       
)

    old_summary = sess["summary"]
    prompt = f"""You are a summarization assistant.
Your job is to create a concise rolling summary of a conversation.

Previous summary:
{old_summary if old_summary else "None"}

New conversation so far:
{transcript}

Write an updated summary that captures all key facts (names, ages, jobs, topics discussed).
Be concise but complete. Write in English."""

    response = llm.invoke(prompt)
    sess["summary"] = response.content
    save_summary(PG_PATH, session_id, sess["summary"])

def chat(question: str, session_id: str = "default") -> str:
    question = (question or "").strip()
    if not question:
        return "Please write a message first."
 
    sess = _ensure_session(session_id)
 
    # 1. Persist user message to DB (with embedding)
    user_embedding = embed(question)
    insert_message(PG_PATH, session_id, "human", question, user_embedding)
 
    # 2. Update lc_history in RAM
    sess["lc_history"].add_user_message(question)
 
    # 3. Maybe update rolling summary
    sess["turns_since_summary"] += 1
    if sess["turns_since_summary"] >= SUMMARY_UPDATE_EVERY:
        _update_summary(session_id)
        sess["turns_since_summary"] = 0
 
    # 4. Call LLM
    msg = chat_with_memory.invoke(
    {
        "question": question,
        "summary":  sess["summary"],
        "session_id": session_id,    # ← add this so chain can use it
    },
    config={"configurable": {"session_id": session_id}},
    )
    answer = msg.content
 
    # 5. Persist AI response to DB (with embedding)
    ai_embedding = embed(answer)
    insert_message(PG_PATH, session_id, "ai", answer, ai_embedding)
 
    # 6. Update lc_history in RAM
    sess["lc_history"].add_ai_message(answer)
 
    return answer
 
 # ── Public helpers ─────────────────────────────────────────────────────────────

def get_raw_history(session_id: str = "default") -> list[dict]:
    """Return full chronological history from PostgreSQL."""
    rows = load_history(PG_PATH, session_id)
    return [{"role": r["role"], "content": r["content"]} for r in rows]
 
def clear_session(session_id: str = "default") -> None:
    """Delete all DB rows + evict from RAM cache."""
    delete_session(PG_PATH, session_id)
    _STORE.pop(session_id, None)
 
def get_summary(session_id: str = "default") -> str:
    return _ensure_session(session_id)["summary"]

model = _get_embedder()
EMBED_DIM=model.get_sentence_embedding_dimension()

def startup() -> None:
    """Call once when the process starts (from api.py lifespan or __main__)."""
    init_db(PG_PATH, EMBED_DIM)

if __name__ == "__main__":
    startup()
    session_id = input("Session ID: ").strip() or "default"
 
    while True:
        q = input("\nQuestion (or exit): ").strip()
        if q.lower() == "exit":
            _update_summary(session_id)   # ← generates fresh summary AND saves it to DB
            print("Summary saved.")
            break
        if q.lower() == "/history":
            for i, m in enumerate(get_raw_history(session_id), 1):
                print(f"{i:02d}. [{m['role']}] {m['content']}")
            continue
        if q.lower() == "/summary":
            print("\n=== SUMMARY ===")
            print(get_summary(session_id) or "No summary yet.")
            continue
        if q.lower() == "/clear":
            clear_session(session_id)
            print("Session cleared.")
            continue
        print("\n---\n" + chat(q, session_id=session_id))