# Smart Travel Agent using LangGraph & Tavily

A ReAct-style agentic travel assistant where the LLM autonomously decides which tools to call — and when to stop — to answer travel-related queries. Built with LangGraph, a free HuggingFace model, and Tavily for real-time web search.

---

## What It Does

Unlike a hardcoded pipeline, this agent gives the LLM full control over tool selection. Given a user query, the LLM:

1. **Decides** which tool to call (or whether to answer directly)
2. **Calls the tool** via LangGraph's `ToolNode`
3. **Reads the result** and decides the next step
4. **Repeats** until it has enough information to produce a final answer

This loop — Reason → Act → Observe → Reason — is the **ReAct pattern**.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────┐
│          LLM Node            │  Qwen2.5-72B-Instruct
│  (decides: call tool or END) │
└──────────────────────────────┘
    │                    │
    │ tool_call          │ no tool_call
    ▼                    ▼
┌───────────┐          END
│ Tool Node │
│-----------│
│ • get_current_date_tool  │
│ • math_tool              │
│ • tavily_search_tool     │
└───────────┘
    │
    └──── result back to LLM Node
```

The LLM loops back after every tool call until it decides to stop.

---

## Tools

| Tool | Description |
|---|---|
| `get_current_date_tool` | Returns today's date in `YYYY-MM-DD` format |
| `math_tool` | Performs addition and subtraction from a string expression |
| `tavily_search_tool` | Searches the web for live news, travel advisories, and general information |

---

## Stack

- **LangGraph** — stateful graph orchestration with conditional edges
- **LangChain** — tool definition and LLM binding
- **langchain-huggingface** — HuggingFace LLM with ChatOpenAI-compatible interface
- **Qwen2.5-72B-Instruct** — free HuggingFace chat model with tool-calling support
- **Tavily** — real-time web search API (free tier available)

---

## Setup

**1. Install dependencies**

```bash
pip install langgraph langchain langchain-huggingface huggingface_hub tavily-python python-dotenv
```

**2. Get your API keys**

| Service | Link |
|---|---|
| HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Tavily API key | [app.tavily.com](https://app.tavily.com) |

**3. Create a `.env` file**

```
HUGGINGFACE_API_TOKEN=hf_your_token_here
TAVILY_API_KEY=tvly_your_key_here
```

---

## Usage

```python
from dotenv import load_dotenv
load_dotenv()

# Build agent with selected tools
tools = [get_current_date_tool, math_tool, tavily_search_tool]
app = build_graph_one_tool(tools)

# Run
output, history = app_call(app, "What is the current date?")
print(output)

output, history = app_call(app, "What is 47 + 83?")
print(output)

output, history = app_call(
    app,
    "I want the latest news about New York. I'm planning to visit "
    "from 2025-06-01 to 2025-06-04. Fetch travel advisories and format the output."
)
print(output)
```

---

## How `build_graph_one_tool` Works

```python
def build_graph_one_tool(tools):
    llm_with_tools = llm.bind_tools(tools)   # LLM knows about the tools
    tool_node = ToolNode(tools)               # executes whichever tool LLM picks

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_llm)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("llm")
    workflow.add_conditional_edges("llm", should_continue)  # tool or END
    workflow.add_edge("tools", "llm")                       # loop back after tool

    return workflow.compile()
```

The `should_continue` function checks if the LLM's last message contains a `tool_call`. If yes, it routes to the tools node. If no, it routes to `END`.

---

## State Structure

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
```

LangGraph's `add_messages` reducer automatically appends new messages to the history on each step.

---

## Key Concepts

- **ReAct pattern** — the LLM reasons and acts in a loop, not a fixed sequence
- **`bind_tools()`** — tells the LLM what tools are available and their signatures
- **`ToolNode`** — automatically dispatches to the correct tool based on LLM's choice
- **Conditional edges** — route to tools or END depending on LLM output
- **Message history** — full conversation passed to LLM at every step so it has context

---

## Difference from a Pipeline

| Pipeline (fixed) | This Agent (ReAct) |
|---|---|
| Steps are hardcoded | LLM decides the steps |
| Always runs all nodes | Only calls tools it needs |
| Order never changes | Order adapts to the query |
| No reasoning between steps | LLM reasons after every tool result |

---

## Notes

- `Qwen2.5-72B-Instruct` is used because it supports the `/v1/chat/completions` endpoint on the free HF router with tool calling. Mistral-7B-Instruct-v0.3 does **not** work as a chat model on this router.
- If Qwen2.5-72B is slow, try `Qwen/Qwen2.5-7B-Instruct` as a faster alternative.
- Tavily free tier allows 1000 searches/month.

---

## Author

**Sara Ibrahim**
- Email: saraomran433@gmail.com
