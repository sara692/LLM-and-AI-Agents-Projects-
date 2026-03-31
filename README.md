# 🤖 LLM and AI Agents Projects

![Timeline](https://img.shields.io/badge/Timeline-January%202026%20–%20Present-1E90FF?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-2ECC40?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-RAG%20%2B%20Memory-F47C3C?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat-square&logo=postgresql&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.x-F97316?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference%20API-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Persistent Memory](https://img.shields.io/badge/Memory-Persistent%20PostgreSQL-4B0082?style=flat-square)
![Semantic Search](https://img.shields.io/badge/Search-Semantic%20Vector-00B4D8?style=flat-square)
![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green?style=flat-square)

> A collection of hands-on projects exploring Large Language Models, AI Agents, and hybrid decision systems.
> Each project is self-contained with its own code, data, and documentation.
> Built during the **Digital Egypt Pioneers Initiative (DEPI)** training program.

---

## 🗂️ Projects

| # | Project | Description | Stack |
|---|---------|-------------|-------|
| 01 | [Loan Decision Engine](./01_Loan_Decision_Engine/) | Hybrid rule-based + LLM system for automated loan decisions with semantic search, AND condition logic, and natural language explanation | Phi-3.5 Mini · SentenceTransformers · BitsAndBytes |
| 02 | [Self-Correcting Analytics Agent](./02_Self_Correcting_Analytics_Agent/) | Decision-driven analytics agent with AST code validation, auto-repair loop, schema mapping, ambiguity handling, and red-team evaluation harness | Phi-3.5 Mini · Pandas · AST · BitsAndBytes |
| 03 | [Multi-Stage NLP Workflow with LangGraph & HuggingFace](./03_Multi_Stage_NLP_Workflow/) | Sequential NLP pipeline that summarizes raw text, translates it to French, and performs sentiment analysis using free HuggingFace models orchestrated via LangGraph | LangGraph · HuggingFace API · BART · Helsinki-NLP · DistilBERT |
| 04 | [Smart Travel Agent using LangGraph & Tavily](./04_Smart_Travel_Agent/) | ReAct-style agentic travel assistant where the LLM autonomously decides which tools to call — real-time web search, current date, and math — to answer travel queries | LangGraph · HuggingFace · Tavily · Qwen2.5 |
| 05 | [Smart Chat Assistant with Memory](./05_Smart_Chat_Assistant/) | Production-style chat system with persistent PostgreSQL memory, semantic vector retrieval, rolling summarization, and a full REST API + Gradio UI | LangChain · PostgreSQL · pgvector · FastAPI · Gradio · Qwen3 |

---

## 🏗️ Repository Structure

```
LLM-and-AI-Agents/
│
├── README.md
│
├── 01_Loan_Decision_Engine/
│   ├── README.md
│   ├── loan_decision_engine_v2.py
│   └── Loan_Policy_v2.csv
│
├── 02_Self_Correcting_Analytics_Agent/
│   ├── README.md
│   ├── agent.ipynb
│   ├── sales_dataset.csv
│   ├── eval_report.csv
│   └── agent_flowchart.svg
│
├── 03_Multi_Stage_NLP_Workflow/
│   ├── README.md
│   └── nlp_pipeline.ipynb
│
├── 04_Smart_Travel_Agent/
│   ├── README.md
│   └── travel_agent.ipynb
│
└── 05_Smart_Chat_Assistant/
    ├── README.md
    ├── app.py
    ├── api.py
    ├── db.py
    ├── embeddings.py
    ├── prompts.py
    ├── ui_gradio_api.py
    ├── docker-compose.yml
    └── .env.example
```

---

## 🧠 Skills Demonstrated

### 🔤 Language Models & Prompting
- Prompt engineering with system instructions, memory injection, and structured templates
- Multi-turn conversation management with LangChain `RunnableWithMessageHistory`
- Rolling summarization — compressing long conversations into key facts using LLM
- Working with open-source LLMs via HuggingFace Inference API (Qwen, Phi, BART, DistilBERT)
- 4-bit quantization with BitsAndBytes for local model efficiency

### 🤖 AI Agents & Orchestration
- LangGraph stateful graph workflows with conditional routing and node transitions
- ReAct-style agentic tool use — LLM autonomously decides which tools to call
- Multi-stage NLP pipelines (summarization → translation → sentiment)
- Self-correcting agents with automatic code repair loops
- AST-based code safety validation before execution

### 🗄️ Memory & Vector Databases
- Persistent chat memory with PostgreSQL + pgvector
- Sentence-transformer embeddings (all-MiniLM-L6-v2) for semantic search
- Cosine similarity search over stored message vectors
- Semantic context retrieval — finding relevant past messages, not just recent ones
- Session isolation and multi-user memory management

### 🔧 Backend & APIs
- FastAPI REST API design with request/response models, pagination, and error handling
- Lifespan events for database initialization at startup
- Semantic search endpoint over vector embeddings
- Session statistics and management endpoints

### 🖥️ Frontend & UI
- Gradio chat interface connected to a live REST backend
- Multi-panel UI with history viewer, summary display, semantic search, and stats
- Real-time chat with session ID support for multi-user scenarios

### 🔒 Hybrid Decision Systems
- Rule-based engines combined with LLM reasoning
- Policy enforcement with AND condition logic
- Red-team evaluation harness for adversarial prompt testing
- Evaluation metrics: success rate, rejection precision, repair rate

### 🛠️ Engineering Practices
- Modular project structure with separation of concerns
- Environment-based configuration with `.env` files
- Docker Compose for reproducible infrastructure
- Graceful error handling and fallback strategies

---

## 📈 Project Progression

The projects form a natural learning arc — each one builds on patterns from the previous:

```
01 Loan Engine          → hybrid rules + LLM, semantic search
        ↓
02 Analytics Agent      → self-correction, AST safety, evaluation
        ↓
03 NLP Workflow         → LangGraph orchestration, multi-stage pipelines
        ↓
04 Travel Agent         → ReAct agents, autonomous tool use, web search
        ↓
05 Chat with Memory     → persistent memory, vector DB, full API + UI
```

Each project introduces one or two new concepts while reinforcing the previous ones.

---

## 🛠️ Setup

Each project has its own README with full setup and run instructions.

**General requirements:**
```bash
pip install transformers sentence-transformers bitsandbytes accelerate torch pandas
```

**For LangGraph projects (03 & 04):**
```bash
pip install langgraph langchain langchain-huggingface huggingface_hub tavily-python python-dotenv
```

**For Smart Chat Assistant (05):**
```bash
pip install langchain langchain-openai fastapi uvicorn psycopg[binary] pgvector sentence-transformers gradio python-dotenv numpy
docker-compose up -d   # starts PostgreSQL + pgvector
```

---

## 🛠️ Skills & Technologies

### 🐍 Core Language
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)

---

### 🤖 LLMs & Model Serving
> Projects 02, 03, 04, 05

![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference%20API-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Qwen](https://img.shields.io/badge/LLM-Qwen3-6B4FBB?style=flat-square)
![Phi](https://img.shields.io/badge/LLM-Phi--3.5%20Mini-6B4FBB?style=flat-square)
![Prompt Engineering](https://img.shields.io/badge/Skill-Prompt%20Engineering-8B5CF6?style=flat-square)

---

### 🧠 AI Agents & Orchestration
> Projects 02, 03, 04, 05

![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Workflows-F97316?style=flat-square)
![ReAct](https://img.shields.io/badge/Agent-ReAct%20Pattern-DC2626?style=flat-square)
![Tool Use](https://img.shields.io/badge/Agent-Autonomous%20Tool%20Use-DC2626?style=flat-square)
![Self Correcting](https://img.shields.io/badge/Agent-Self--Correcting%20Loop-EF4444?style=flat-square)
![Multi Stage](https://img.shields.io/badge/Pipeline-Multi--Stage%20NLP-0EA5E9?style=flat-square)

---

### 🔍 Memory & Vector Search
> Project 05

![pgvector](https://img.shields.io/badge/VectorDB-pgvector-336791?style=flat-square&logo=postgresql&logoColor=white)
![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-16A34A?style=flat-square)
![Semantic Search](https://img.shields.io/badge/Search-Semantic%20Cosine%20Similarity-00B4D8?style=flat-square)
![SentenceTransformers](https://img.shields.io/badge/Library-SentenceTransformers-F59E0B?style=flat-square)
![Persistent Memory](https://img.shields.io/badge/Memory-Persistent%20PostgreSQL-4B0082?style=flat-square)
![Rolling Summary](https://img.shields.io/badge/Memory-Rolling%20Summarization-7C3AED?style=flat-square)

---

### 🗄️ Databases & Storage
> Project 05

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Production%20DB-336791?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

---

### 🔧 Backend & APIs
> Project 05

![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi&logoColor=white)
![REST](https://img.shields.io/badge/API-REST%20Design-009688?style=flat-square)
![Uvicorn](https://img.shields.io/badge/Server-Uvicorn-4B5563?style=flat-square)
![Pydantic](https://img.shields.io/badge/Validation-Pydantic-E92063?style=flat-square)

---

### 🖥️ Frontend & UI
> Project 05

![Gradio](https://img.shields.io/badge/Gradio-6.x%20UI-F97316?style=flat-square)
![Chat UI](https://img.shields.io/badge/UI-Chat%20Interface-6366F1?style=flat-square)

---

### 🔒 Security & Evaluation
> Project 02

![AST](https://img.shields.io/badge/Security-AST%20Code%20Validation-DC2626?style=flat-square)
![Sandbox](https://img.shields.io/badge/Security-Sandboxed%20Execution-DC2626?style=flat-square)
![RedTeam](https://img.shields.io/badge/Evaluation-Red--Team%20Testing-B91C1C?style=flat-square)
![Metrics](https://img.shields.io/badge/Evaluation-Success%20%2F%20Repair%20Rate-B91C1C?style=flat-square)

---

### 🌐 External Integrations
> Projects 04, 05

![Tavily](https://img.shields.io/badge/Search-Tavily%20Web%20Search-0F766E?style=flat-square)
![HuggingFace Router](https://img.shields.io/badge/Router-HuggingFace%20Inference%20Router-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

### 📊 Data & NLP
> Projects 02, 03

![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Summarization%20%7C%20Translation%20%7C%20Sentiment-3B82F6?style=flat-square)
![BART](https://img.shields.io/badge/Model-BART%20Summarization-8B5CF6?style=flat-square)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT%20Sentiment-8B5CF6?style=flat-square)
![Helsinki](https://img.shields.io/badge/Model-Helsinki--NLP%20Translation-8B5CF6?style=flat-square)

---

### 🏗️ Hybrid Systems
> Project 01, 05

![Hybrid](https://img.shields.io/badge/Architecture-Hybrid%20Rules%20%2B%20LLM-F47C3C?style=flat-square)
![RAG](https://img.shields.io/badge/Architecture-RAG%20%2B%20Memory-F47C3C?style=flat-square)
![Decision Engine](https://img.shields.io/badge/System-Decision%20Engine-F47C3C?style=flat-square)


---

## 🤝 Acknowledgments & Credits

These projects were developed under the umbrella of the **Digital Egypt Pioneers Initiative (DEPI)** scholarship.

Special thanks and appreciation to:

**Eng. Baraa Abu Salout**
For his exceptional mentorship, technical guidance, and for sharing his deep expertise in AI agent architectures and LangGraph workflows.

**DEPI (Digital Egypt Pioneers Initiative)**
For providing the scholarship, resources, and the professional environment to foster innovation in the field of AI and data analytics.

---

## 📬 Contact

**Sara Ibrahim**
- 📧 Email: saraomran433@gmail.com
