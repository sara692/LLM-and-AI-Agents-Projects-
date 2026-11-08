# Multi-Stage NLP Workflow with LangGraph & HuggingFace

A sequential NLP pipeline that takes raw text through three stages — summarization, translation, and sentiment analysis — using free HuggingFace models orchestrated via LangGraph.

---

## What It Does

The pipeline processes any input text through three chained nodes:

1. **Summarize** — Condenses the input into a concise summary using BART
2. **Translate** — Translates the summary from English to French using Helsinki-NLP
3. **Sentiment Analysis** — Classifies the sentiment of the summary as Positive or Negative using DistilBERT

Each node receives the state from the previous one, and the final state contains all three outputs.

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────┐
│  Summarize  │  facebook/bart-large-cnn
└─────────────┘
    │
    ▼
┌─────────────┐
│  Translate  │  Helsinki-NLP/opus-mt-en-fr
└─────────────┘
    │
    ▼
┌──────────────────┐
│ Sentiment        │  cardiffnlp/twitter-roberta-base-sentiment-latest
│ Analysis         │
└──────────────────┘
    │
    ▼
Final State (summary + translation + sentiment)
```

---

## Models Used

| Stage | Model | Task |
|---|---|---|
| Summarization | `facebook/bart-large-cnn` | Abstractive summarization |
| Translation | `Helsinki-NLP/opus-mt-en-fr` | English → French |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Positive / Negative classification |

All models run on the **free HuggingFace Inference API** — no GPU or paid tier required.

---

## Stack

- **LangGraph** — graph-based workflow orchestration
- **HuggingFace Inference API** — free hosted model inference
- **Python** — 3.10+

---

## Setup

**1. Install dependencies**

```bash
pip install langgraph huggingface_hub python-dotenv
```

**2. Get a free HuggingFace token**

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Read role → Copy it.

**3. Create a `.env` file**

```
HUGGINGFACE_API_TOKEN=hf_your_token_here
```

---

## Usage

```python
from dotenv import load_dotenv
load_dotenv()

graph = build_graph()

result = graph.invoke({
    "input_text": "Your text here...",
    "summary": "",
    "translated_text": "",
    "analyze_sentiment": ""
})

print("Summary:   ", result["summary"])
print("Translation:", result["translated_text"])
print("Sentiment:  ", result["analyze_sentiment"])
```

**Example output:**

```
Summary:    Electric cars use battery-powered motors for zero-emission driving.
Translation: Les voitures électriques utilisent des moteurs alimentés par batterie.
Sentiment:  POSITIVE (94.32%)
```

---

## State Structure

```python
class AgentState(TypedDict):
    input_text: str        # original input
    summary: str           # output of summarize node
    translated_text: str   # output of translate node
    analyze_sentiment: str # output of sentiment node
```

---

## Key Concepts

- **LangGraph StateGraph** — defines the pipeline as a directed graph of nodes
- **State passing** — each node receives the full state and returns an updated copy
- **HuggingFace InferenceClient** — calls hosted models via REST API without local GPU
- **Sequential edges** — `summarize → translate → sentiment → END`

---

## Notes

- Translation language can be changed by swapping the Helsinki-NLP model (e.g. `opus-mt-en-ar` for Arabic, `opus-mt-en-de` for German)
- The free HF Inference API has rate limits — add `time.sleep(1)` between calls if you hit errors
- Always pass plain text to each API call, not prompt-wrapped instructions

---

## Author

**Sara Ibrahim**
- Email: saraomran433@gmail.com
