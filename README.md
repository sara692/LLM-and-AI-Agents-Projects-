# 🤖 LLM and AI Agents Projects

![Timeline](https://img.shields.io/badge/Timeline-DEPI-blue) ![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LangChain](https://img.shields.io/badge/LangChain-LangGraph-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791) ![Gradio](https://img.shields.io/badge/Gradio-UI-ff7c00) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow) ![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20API-6467f2)

A collection of hands-on projects exploring Large Language Models, AI Agents, and hybrid decision systems. Each project is self-contained with its own code, data, and documentation. Built during the Digital Egypt Pioneers Initiative (DEPI) training program.

## 🗂️ Projects

| # | Project | Description | Stack |
|---|---------|-------------|-------|
| 01 | [Loan Decision Engine](./01_Loan_Decision_Engine) | Hybrid rule-based + LLM system for automated loan decisions with semantic search, AND condition logic, and natural language explanation | Phi-3.5 Mini · SentenceTransformers · BitsAndBytes |
| 02 | [Multi-Stage NLP Workflow with LangGraph & HuggingFace](./02_Multi_Stage_NLP_Workflow) | Sequential NLP pipeline that summarizes raw text, translates it to French, and performs sentiment analysis using free HuggingFace models orchestrated via LangGraph | LangGraph · HuggingFace API · BART · Helsinki-NLP · DistilBERT |
| 03 | [Smart Travel Agent using LangGraph & Tavily](./03_Smart_Travel_Agent) | ReAct-style agentic travel assistant where the LLM autonomously decides which tools to call (real-time web search, current date, and math) to answer travel queries | LangGraph · HuggingFace · Tavily · Qwen2.5 |
| 04 | [Smart Chat Assistant with Memory](./04_Smart_Chat_Assistant_with_Memory) | Production-style chat system with persistent PostgreSQL memory, semantic vector retrieval, rolling summarization, and a full REST API + Gradio UI | LangChain · PostgreSQL · pgvector · FastAPI · Gradio · Qwen3 |
| 05 | [Policy-Aware Analytics Agent](./05_Policy_Analytics_Agent) | Analytics agent that answers questions about a dataset with **aggregated results only**. Policy gate for unsafe questions, AST code validation, restricted execution, self-repair loop (max 5 retries), hidden PII columns, and a FastAPI + Streamlit app | Qwen2.5-Coder (OpenRouter) · Pandas · AST · FastAPI · Streamlit |

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
├── 02_Multi_Stage_NLP_Workflow/
│   ├── README.md
│   └── nlp_pipeline.ipynb
│
├── 03_Smart_Travel_Agent/
│   ├── README.md
│   └── travel_agent.ipynb
│
├── 04_Smart_Chat_Assistant_with_Memory/
│   ├── README.md
│   ├── app.py
│   ├── api.py
│   ├── db.py
│   ├── embeddings.py
│   ├── prompts.py
│   ├── ui_gradio_api.py
│   ├── docker-compose.yml
│   └── .env.example
│
└── 05_Policy_Analytics_Agent/
    ├── README.md
    ├── llm.py              # OpenRouter call
    ├── policy.py           # regex gate: rejects unsafe questions
    ├── agent.py            # AST check + restricted run + repair loop
    ├── api.py              # FastAPI: POST /ask, GET /columns
    ├── app.py              # Streamlit chat UI
    ├── test_lite.py        # pytest (no API key needed)
    ├── requirements.txt
    ├── .env.example
    └── data/
        └── employees.csv
```

## 🧠 Skills Demonstrated

### 🔤 Language Models & Prompting
- Prompt engineering with system instructions, memory injection, and structured templates
- Multi-turn conversation management with LangChain `RunnableWithMessageHistory`
- Rolling summarization: compressing long conversations into key facts using an LLM
- Working with open-source LLMs via the HuggingFace Inference API (Qwen, Phi, BART, DistilBERT)
- Calling hosted models through OpenRouter (Qwen2.5-Coder) with a single OpenAI-compatible endpoint
- 4-bit quantization with BitsAndBytes for local model efficiency

### 🤖 AI Agents & Orchestration
- LangGraph stateful graph workflows with conditional routing and node transitions
- ReAct-style agentic tool use: the LLM autonomously decides which tools to call
- Multi-stage NLP pipelines (summarization → translation → sentiment)
- Self-correcting agents with automatic code repair loops
- AST-based code safety validation before execution

### 🗄️ Memory & Vector Databases
- Persistent chat memory with PostgreSQL + pgvector
- Sentence-transformer embeddings (all-MiniLM-L6-v2) for semantic search
- Cosine similarity search over stored message vectors
- Semantic context retrieval: finding relevant past messages, not just recent ones
- Session isolation and multi-user memory management

### 🔧 Backend & APIs
- FastAPI REST API design with request/response models, pagination, and error handling
- Lifespan events for database initialization at startup
- Semantic search endpoint over vector embeddings
- Session statistics and management endpoints
- Serving an AI agent behind a REST endpoint (`POST /ask`) with a safe, structured response

### 🖥️ Frontend & UI
- Gradio chat interface connected to a live REST backend
- Multi-panel UI with history viewer, summary display, semantic search, and stats
- Real-time chat with session ID support for multi-user scenarios
- Streamlit chat UI that shows the answer, the result table, and the generated code

### 🔒 Hybrid Decision Systems & Safety
- Rule-based engines combined with LLM reasoning
- Policy enforcement with AND condition logic
- Defence in depth for LLM-generated code: policy gate → AST check → restricted execution → output check
- Hiding personal-data columns from the model and from results
- Red-team prompts and evaluation metrics: success rate, rejection precision, repair rate

### 🛠️ Engineering Practices
- Modular project structure with separation of concerns
- Environment-based configuration with `.env` files
- Docker Compose for reproducible infrastructure
- Graceful error handling and fallback strategies
- Offline unit tests with a faked LLM (pytest)

## 📈 Project Progression

The projects form a natural learning arc, and each one builds on patterns from the previous:

```
01 Loan Engine          → hybrid rules + LLM, semantic search
        ↓
02 NLP Workflow         → LangGraph orchestration, multi-stage pipelines
        ↓
03 Travel Agent         → ReAct agents, autonomous tool use, web search
        ↓
04 Chat with Memory     → persistent memory, vector DB, full API + UI
        ↓
05 Policy Analytics     → self-correction, AST safety, policy gate, API + UI
```

Project 05 brings several earlier ideas together: the policy rules of 01, the agent loop of 03, and the FastAPI + UI structure of 04.

## 🛠️ Setup

Each project has its own README with full setup and run instructions.

**General requirements (Project 01):**
```bash
pip install transformers sentence-transformers bitsandbytes accelerate torch pandas
```

**For LangGraph projects (02 & 03):**
```bash
pip install langgraph langchain langchain-huggingface huggingface_hub tavily-python python-dotenv
```

**For Smart Chat Assistant (04):**
```bash
pip install langchain langchain-openai fastapi uvicorn psycopg[binary] pgvector sentence-transformers gradio python-dotenv numpy
docker-compose up -d   # starts PostgreSQL + pgvector
```

**For Policy-Aware Analytics Agent (05):**
```bash
cd 05_Policy_Analytics_Agent
pip install -r requirements.txt
cp .env.example .env          # add your OPENROUTER_API_KEY
uvicorn api:app --reload      # terminal 1: API
streamlit run app.py          # terminal 2: UI
pytest                        # tests (no API key needed)
```

## 🛠️ Skills & Technologies

### 🐍 Core Language
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

### 🤖 LLMs & Model Serving
*Projects 01, 02, 03, 04, 05*

HuggingFace · OpenRouter · Qwen · Phi · Prompt Engineering

### 🧠 AI Agents & Orchestration
*Projects 02, 03, 04, 05*

LangChain · LangGraph · ReAct · Tool Use · Self-Correcting · Multi-Stage

### 🔍 Memory & Vector Search
*Project 04*

pgvector · Embeddings · Semantic Search · SentenceTransformers · Persistent Memory · Rolling Summary

### 🗄️ Databases & Storage
*Project 04*

PostgreSQL · Docker

### 🔧 Backend & APIs
*Projects 04, 05*

FastAPI · REST · Uvicorn · Pydantic

### 🖥️ Frontend & UI
*Projects 04, 05*

Gradio · Streamlit · Chat UI

### 🔒 Security & Evaluation
*Project 05*

AST Validation · Restricted Execution · Policy Gate · Red-Team Prompts · pytest

### 🌐 External Integrations
*Projects 03, 04, 05*

Tavily · HuggingFace Router · OpenRouter

### 📊 Data & NLP
*Projects 02, 05*

Pandas · NLP · BART · DistilBERT · Helsinki

### 🏗️ Hybrid Systems
*Projects 01, 04*

Hybrid RAG · Decision Engine

## 📄 License
This portfolio is available for educational and demonstration purposes. Individual projects may have specific licenses based on data sources and requirements. Please check individual project directories for specific licensing information.

## 👨‍💻 Author
**Sara Ibrahim**  
*Data Scientist | Machine Learning | AI Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sara-ibrahim-omran)  
[![Email](https://img.shields.io/badge/Email-Contact%20Me-red?style=flat&logo=gmail)](mailto:Saraomran433@gmail.com)

---

### 🌟 **Let's Connect!**
I'm passionate about solving real-world problems with data, machine learning, and AI. Feel free to reach out for:
- 💼 Job opportunities and collaborations
- 📚 Technical discussions and knowledge sharing
- 🚀 Project ideas and innovative solutions

**⭐ If you find any project interesting or helpful, please give it a star!**
