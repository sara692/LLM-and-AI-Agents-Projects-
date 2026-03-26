# LLM and AI Agents Projects
A collection of hands-on projects exploring Large Language Models, AI Agents, and hybrid decision systems. Each project is self-contained with its own code, data, and documentation.

---

## Projects

| # | Project | Description | Stack |
|---|---|---|---|
| 01 | [Loan Decision Engine](./01_Loan_Decision_Engine/) | Hybrid rule-based + LLM system for automated loan decisions with semantic search, AND condition logic, and natural language explanation | Phi-3.5 Mini, SentenceTransformers, BitsAndBytes |
| 02 | [Self-Correcting Analytics Agent](./02_Self_Correcting_Analytics_Agent/) | Decision-driven analytics agent with AST code validation, auto-repair loop, schema mapping, ambiguity handling, and red-team evaluation harness | Phi-3.5 Mini, Pandas, AST, BitsAndBytes |
| 03 | [Multi-Stage NLP Workflow with LangGraph & HuggingFace](./03_Multi_Stage_NLP_Workflow/) | Sequential NLP pipeline that summarizes raw text, translates it to French, and performs sentiment analysis using free HuggingFace models orchestrated via LangGraph | LangGraph, HuggingFace Inference API, BART, Helsinki-NLP, DistilBERT |
| 04 | [Smart Travel Agent using LangGraph & Tavily](./04_Smart_Travel_Agent/) | ReAct-style agentic travel assistant where the LLM autonomously decides which tools to call — real-time web search, current date, and math — to answer travel queries | LangGraph, HuggingFace, Tavily, Qwen2.5 |

---

## Repository Structure
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
└── 04_Smart_Travel_Agent/
    ├── README.md
    └── travel_agent.ipynb
```

---

## Topics Covered

- Large Language Models (LLMs)
- Semantic search with sentence embeddings
- Rule-based decision engines
- Hybrid AI systems (rules + LLM)
- 4-bit quantization with BitsAndBytes
- AI Agents and pipeline orchestration
- AST-based code safety validation
- Self-correcting agents with auto-repair loops
- Schema mapping and ambiguity handling
- Red-team evaluation and metrics
- LangGraph stateful workflows and graph-based orchestration
- Multi-stage NLP pipelines (summarization, translation, sentiment analysis)
- ReAct-style agentic tool use with autonomous LLM decision-making
- Real-time web search integration with Tavily
- Free HuggingFace Inference API with open-source models

---

## Setup

Each project has its own README with full setup and run instructions. General requirements:
```bash
pip install transformers sentence-transformers bitsandbytes accelerate torch pandas
```

For LangGraph projects (03 & 04):
```bash
pip install langgraph langchain langchain-huggingface huggingface_hub tavily-python python-dotenv
```

---

## About

Projects are built for learning and experimentation. Each one focuses on a specific pattern or technique in the LLM and AI Agents space, with clean code, comments, and documented architecture.

---

## Contact

**Sara Ibrahim**
- Email: saraomran433@gmail.com
