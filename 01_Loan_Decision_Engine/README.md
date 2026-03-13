# 01 — Loan Decision Engine v2

A hybrid decision engine that combines **semantic search**, **rule-based evaluation**, and **LLM explanation** to automate loan decisions. The LLM never makes the decision — it only explains it.

---

## Architecture

```
User Input
  → Semantic Search          (find relevant policy rules)
  → Extract Variables        (amount, credit_score)
  → Extract Signals          (keyword hints for reranking)
  → Rerank Candidates        (boost rules matching user intent)
  → Evaluate AND Conditions  (match_and_condition per rule)
  → Priority Sort            (High > Medium > Low risk wins)
  → LLM Explanation          (explains decision in plain English)
  → Structured Output        (status, decision, variables, explanation)
```

---

## Decision Rules

| Condition | Decision | Risk |
|---|---|---|
| amount <= 5000 | APPROVED | Low |
| 5000 < amount <= 20000 | MANUAL_REVIEW | Medium |
| amount > 20000 AND credit_score < 700 | REJECTED | High |
| credit_score < 600 | REJECTED | High |

Rules are stored in `Loan_Policy_v2.csv` and can be updated without changing any code.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Phi-3.5 Mini Instruct (4-bit quantized) |
| Semantic search | SentenceTransformers — all-MiniLM-L6-v2 |
| Quantization | BitsAndBytes (NF4, float16) |
| Framework | PyTorch + HuggingFace Transformers |
| Data | Pandas |

---

## Project Structure

```
01_Loan_Decision_Engine/
│
├── README.md
├── loan_decision_engine_v2.py   ← full pipeline (13 sections)
└── Loan_Policy_v2.csv           ← policy rules
```

### Code Sections

| Section | Description |
|---|---|
| 0 | Imports |
| 1 | Model loading (Phi-3.5 Mini, 4-bit) |
| 2 | Policy rules CSV |
| 3 | DataFrame loading and full_text column |
| 4 | Semantic embedding setup |
| 5 | Signal dictionary and keyword matching |
| 6 | Numeric extraction and condition matching |
| 7 | Variable extraction (amount, credit_score) |
| 8 | AND condition matching and priority rule selection |
| 9 | Query information extraction |
| 10 | Decision pipeline |
| 11 | LLM explanation layer |
| 12 | Agent wrapper |
| 13 | Test cases and evaluation |

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/YOUR_USERNAME/LLM-and-AI-Agents.git
cd LLM-and-AI-Agents/01_Loan_Decision_Engine
```

**2. Install dependencies**

```bash
pip install transformers sentence-transformers bitsandbytes accelerate torch pandas
```

**3. Download the model**

Download `Phi-3.5-mini-instruct` from HuggingFace and save it locally. Then update this line in the script:

```python
model_path = "/your/local/path/Phi_3_5_mini_instruct"
```

**4. Run**

```bash
python loan_decision_engine_v2.py
```

---

## Test Cases

18 test cases covering all decision branches:

| Category | Cases | Expected Decision |
|---|---|---|
| Low risk (amount ≤ 5000) | 5 | APPROVED |
| Medium risk (5000 < amount ≤ 20000) | 5 | MANUAL_REVIEW |
| High risk (amount > 20000 AND credit < 700, or credit < 600) | 5 | REJECTED |
| Missing variables | 3 | NEED_MORE_INFORMATION |

---

## Example Output

```
Q: I want a loan of 25000 dollars. My credit score is 680.

Status    : ok
Decision  : REJECTED
Risk      : High
Variables : {'amount': 25000.0, 'credit_score': 680.0}
Confidence: 0.87
Explain   : Your loan request of $25,000 has been rejected based on our
            high loan policy, which requires a credit score of 700 or above
            for loans exceeding $20,000. Your current credit score of 680
            does not meet this threshold. You may consider applying for a
            lower loan amount or improving your credit score before reapplying.
```

---

## Key Design Decisions

**LLM explains, never decides.** The rule engine makes all decisions. The LLM only receives the final decision and explains it in plain English. This keeps the system auditable and predictable.

**AND condition logic without eval().** Multi-variable conditions like `amount > 20000 AND credit_score < 700` are parsed and evaluated manually using regex — no `eval()` is ever used.

**Priority sort for overlapping rules.** If multiple rules match, the highest risk level wins (`High > Medium > Low`). This ensures the most critical policy always takes precedence.

**Fallback for missing variables.** If the user does not provide enough information to evaluate any rule, the system returns `NEED_MORE_INFORMATION` instead of guessing.

---

## Contact

**Sara Ibrahim**
- Email: saraomran433@gmail.com
