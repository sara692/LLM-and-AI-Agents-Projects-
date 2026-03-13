# LLM and AI Agents Projects
A collection of hands-on projects exploring Large Language Models, AI Agents, and hybrid decision systems. Each project is self-contained with its own code, data, and documentation.

---

## Projects

| # | Project | Description | Stack |
|---|---|---|---|
| 01 | [Loan Decision Engine](./01_Loan_Decision_Engine/) | Hybrid rule-based + LLM system for automated loan decisions with semantic search, AND condition logic, and natural language explanation | Phi-3.5 Mini, SentenceTransformers, BitsAndBytes |
| 02 | [Self-Correcting Analytics Agent](./02_Self_Correcting_Analytics_Agent/) | Decision-driven analytics agent with AST code validation, auto-repair loop, schema mapping, ambiguity handling, and red-team evaluation harness | Phi-3.5 Mini, Pandas, AST, BitsAndBytes |

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
└── 02_Self_Correcting_Analytics_Agent/
    ├── README.md
    ├── agent.ipynb
    ├── sales_dataset.csv
    ├── eval_report.csv
    └── agent_flowchart.svg
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

---

## Setup

Each project has its own README with full setup and run instructions. General requirements:

```bash
pip install transformers sentence-transformers bitsandbytes accelerate torch pandas
```

---

## About

Projects are built for learning and experimentation. Each one focuses on a specific pattern or technique in the LLM and AI Agents space, with clean code, comments, and documented architecture.

---

## Contact

**Sara Ibrahim**
- Email: saraomran433@gmail.com
