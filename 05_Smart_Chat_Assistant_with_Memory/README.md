# 🧠 Smart Chat Assistant with Memory

## 📖 Overview

**Smart Chat Assistant with Memory** is a conversational AI system built as a **personal learning project and portfolio piece**, developed during the **Digital Egypt Pioneers Initiative (DEPI)** training program.

The project explores how to build a chatbot that maintains **persistent, intelligent memory** across sessions — solving one of the most common limitations of standard chatbots: forgetting everything after each restart.

Unlike standard chatbots, this system combines:

- **PostgreSQL + pgvector** for permanent message storage
- **Semantic similarity search** to retrieve the most relevant past messages
- **Rolling summarization** to compress long conversations into key facts
- **Sliding context window** for efficient LLM token usage

The result is an assistant that **never truly forgets** — even across server restarts, long conversations, or sessions months apart.

---

## 🖥️ System Interface

The system is accessible through two interfaces:

- **Gradio UI** — a visual chat interface for end users
- **FastAPI REST API** — a structured backend for developers and integrations

Both interfaces share the same backend, allowing simultaneous access from multiple clients.

---

## 🏗️ System Architecture

```
User (Gradio UI)
      ↓
FastAPI Backend (api.py)
      ↓
Chat Logic + Memory Manager (app.py)
      ├── Embedding Model (embeddings.py)       ← converts text to vectors
      ├── Semantic Search   ──────────────────┐
      ├── Rolling Summary                     │
      └── LLM (Qwen via HuggingFace Router)   │
                                              ↓
                                    PostgreSQL + pgvector (db.py)
                                    ├── chat_messages  (all messages + embeddings)
                                    └── chat_sessions  (rolling summary per session)
```

Every user message goes through:

1. **Embedding** — converted to a vector fingerprint
2. **Persistent storage** — saved to PostgreSQL with its embedding
3. **Semantic retrieval** — most relevant past messages fetched for context
4. **Summary injection** — rolling summary of key facts always included
5. **LLM call** — model answers with full context
6. **Response storage** — answer saved back to DB with embedding

---

## 🚀 Core Features

### 🗄️ Persistent Memory
All messages are stored in PostgreSQL. History survives server restarts, crashes, and long gaps between sessions.

### 🔍 Semantic Context Retrieval
Instead of blindly sending the last N messages to the LLM, the system uses **vector similarity search** to find the most relevant messages to the current question — even if they are 100 messages old.

### 📝 Rolling Summarization
Every few messages, the LLM generates an updated summary of the entire conversation. This summary is always injected into the prompt, ensuring key facts (names, ages, jobs, topics) are never forgotten.

### 💾 Summary Persistence
Summaries are saved to PostgreSQL and restored on session open — so even the summary survives restarts.

### 🔎 Semantic Search Endpoint
A dedicated API endpoint allows searching past messages by meaning, not just keywords — powered by cosine similarity over stored embeddings.

### 🖥️ Dual Interface
Both a visual Gradio UI and a REST API are provided — for end users and developers respectively.

---

## 🔄 Memory Workflow

Each user message passes through this pipeline:

**1️⃣ Session Load**
On first access, all past messages are loaded from PostgreSQL and the saved summary is restored.

**2️⃣ Message Embedding**
The user's question is converted to a 384-dimensional vector using a local sentence-transformers model.

**3️⃣ Semantic Retrieval**
The vector is compared against all stored message embeddings. The top-K most similar messages are selected as context — not just the most recent ones.

**4️⃣ Summary Check**
Every `SUMMARY_UPDATE_EVERY` turns, the LLM generates a fresh rolling summary from the full transcript and saves it to the database.

**5️⃣ LLM Call**
The model receives: rolling summary + semantically relevant messages + current question.

**6️⃣ Response Storage**
The answer is embedded and saved to PostgreSQL for future retrieval.

**7️⃣ Exit Save**
When the user exits, a final summary is generated and persisted — capturing any facts from the last few messages that hadn't been summarized yet.

---

## 🔒 Session Isolation

Each `session_id` is fully isolated:

- Messages are filtered by `session_id` in every DB query
- Semantic search only searches within the current session
- Clearing one session never affects another
- Two users with different session IDs never share context

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a question, receive an answer |
| `GET` | `/history/{session_id}` | Full chronological history (paginated) |
| `GET` | `/history/{session_id}/search?query=&k=` | Semantic similarity search over messages |
| `GET` | `/summary/{session_id}` | Current rolling summary |
| `GET` | `/session/{session_id}/stats` | Message counts per role |
| `POST` | `/clear` | Delete all messages for a session |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 📂 Project Structure

```
├── app.py                  # Core chat logic, memory management, LLM orchestration
├── api.py                  # FastAPI REST backend
├── db.py                   # All PostgreSQL operations (insert, load, search, delete)
├── embeddings.py           # Sentence-transformer embedding model (lazy-loaded)
├── prompts.py              # LangChain prompt templates
├── ui_gradio_api.py        # Gradio visual chat interface
├── .env                    # Environment variables (not committed)
├── .env.example            # Template for environment setup
├── docker-compose.yml      # PostgreSQL + pgvector container
└── TASK0_NOTES.md          # Architecture notes and design decisions
```

---

## ⚙️ Configuration

All settings are controlled via `.env`:

```dotenv
# LLM
HF_MODEL=Qwen/Qwen3-Coder-Next:novita
HF_TOKEN=hf_your_token_here

# Memory settings
RECENT_WINDOW_SIZE=6          # How many semantically relevant messages to send
SUMMARY_UPDATE_EVERY=4        # How often to regenerate the rolling summary

# PostgreSQL
pg_path=postgresql://postgres:password@localhost:5432/chatdb

# Embedding model
embedder_name=all-MiniLM-L6-v2
```

---

## 🛠️ Installation

**1️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**2️⃣ Start PostgreSQL with pgvector**
```bash
docker-compose up -d
```

**3️⃣ Configure environment**
```bash
cp .env.example .env
# Fill in your HF_TOKEN and pg_path
```

**4️⃣ Run the API**
```bash
uvicorn api:api --reload --host 127.0.0.1 --port 8000
```

**5️⃣ Run the Gradio UI** (separate terminal)
```bash
python ui_gradio_api.py
```

**6️⃣ Open in browser**
```
http://127.0.0.1:7860   ← Gradio UI
http://127.0.0.1:8000/docs  ← Swagger API docs
```

---

## 📦 Requirements

```
langchain
langchain-openai
langchain-core
fastapi
uvicorn
psycopg[binary]
pgvector
sentence-transformers
gradio
python-dotenv
numpy
```

---

## 🧠 Embedding Model

**Model:** `all-MiniLM-L6-v2` (sentence-transformers)

Chosen for:
- Runs fully **locally** — no API key or network call needed
- Produces compact **384-dimensional** vectors
- Strong semantic performance for conversational content
- Fast inference even on CPU

If the model fails to load or encode, the system gracefully falls back to the last N messages from RAM — it never crashes silently.

---

## 🎯 Project Goal

This project was built as a **personal learning journey** and **portfolio demonstration** during the DEPI training program — exploring how to go from a simple in-memory chatbot all the way to a production-style system with:

- vector similarity search over persistent embeddings
- rolling summarization for key fact retention
- semantic context retrieval instead of naive recency-based windowing
- fully persistent PostgreSQL storage

Every feature was built incrementally — starting from basic LangChain memory, then adding sliding windows, then PostgreSQL persistence, then embeddings, then semantic retrieval — making this project a complete walkthrough of modern AI memory techniques.

---

## 🤝 Acknowledgments & Credits

This project was developed under the umbrella of the **Digital Egypt Pioneers Initiative (DEPI)** scholarship.

Special thanks and appreciation to:

**DEPI (Digital Egypt Pioneers Initiative)**
For providing the scholarship, resources, and the professional environment to foster innovation in the field of AI and data engineering.

**Eng. Baraa Abu Salout**
For his exceptional mentorship, technical guidance, and for sharing his deep expertise in AI systems, LangChain, and production-grade architecture design.


