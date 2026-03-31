# run:  python ui_gradio_api.py

import requests
import gradio as gr

API_BASE = "http://127.0.0.1:8000"


def api_chat(message, history, session_id):
    message = (message or "").strip()
    if not message:
        return "", history
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"session_id": session_id, "question": message},
            timeout=60
        )
        r.raise_for_status()
        answer = r.json()["answer"]
    except Exception as e:
        answer = f"API error: {e}"

    history = history or []
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history


def clear_and_update(sid):
    try:
        r = requests.post(f"{API_BASE}/clear", json={"session_id": sid}, timeout=30)
        r.raise_for_status()
        return [], f"Cleared session: {sid}"
    except Exception as e:
        return gr.update(), f"API error: {e}"


def fetch_history(sid):
    try:
        r = requests.get(f"{API_BASE}/history/{sid}", timeout=30)
        r.raise_for_status()
        items = r.json().get("history", [])
        lines = []
        for i, it in enumerate(items, 1):
            lines.append(f"{i:02d}. [{it.get('role')}] {it.get('content')}")
        return "\n".join(lines) if lines else "(No history)"
    except Exception as e:
        return f"API error: {e}"


def fetch_summary(sid):
    try:
        r = requests.get(f"{API_BASE}/summary/{sid}", timeout=30)
        r.raise_for_status()
        summary = r.json().get("summary", "")
        return summary if summary else "(No summary yet — chat more messages first)"
    except Exception as e:
        return f"API error: {e}"


def semantic_search(sid, query, k):
    try:
        r = requests.get(
            f"{API_BASE}/history/{sid}/search",
            params={"query": query, "k": int(k)},
            timeout=30
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return "(No results found)"
        lines = []
        for i, res in enumerate(results, 1):
            lines.append(f"{i:02d}. [{res['role']}] (score: {res['score']}) {res['content']}")
        return "\n".join(lines)
    except Exception as e:
        return f"API error: {e}"


def fetch_stats(sid):
    try:
        r = requests.get(f"{API_BASE}/session/{sid}/stats", timeout=30)
        r.raise_for_status()
        data   = r.json()
        counts = data.get("stats", {})
        total  = data.get("total", 0)
        lines  = [f"{role}: {count} messages" for role, count in counts.items()]
        lines.append(f"Total: {total}")
        return "\n".join(lines) if lines else "(No stats)"
    except Exception as e:
        return f"API error: {e}"


with gr.Blocks(title="Chat via FastAPI") as demo:
    gr.Markdown("## Gradio UI (calls FastAPI backend)")

    session_id   = gr.Textbox(label="Session ID", value="student1")
    chatbot      = gr.Chatbot(label="Chat")
    msg          = gr.Textbox(label="Message")
    status       = gr.Markdown("")
    history_box  = gr.Textbox(label="Session History (raw)",   lines=10)
    summary_box  = gr.Textbox(label="Conversation Summary",    lines=5)
    search_query = gr.Textbox(label="Semantic Search Query")
    search_k     = gr.Slider(label="Top K results", minimum=1, maximum=20, value=5, step=1)
    search_box   = gr.Textbox(label="Search Results",          lines=8)
    stats_box    = gr.Textbox(label="Session Stats",           lines=4)

    with gr.Row():
        clear_btn   = gr.Button("Clear History")
        show_btn    = gr.Button("Show History")
        summary_btn = gr.Button("Show Summary")
        stats_btn   = gr.Button("Show Stats")

    with gr.Row():
        search_btn  = gr.Button("Search Messages")

    msg.submit(api_chat,              inputs=[msg, chatbot, session_id],          outputs=[msg, chatbot])
    clear_btn.click(clear_and_update, inputs=[session_id],                        outputs=[chatbot, status])
    show_btn.click(fetch_history,     inputs=[session_id],                        outputs=[history_box])
    summary_btn.click(fetch_summary,  inputs=[session_id],                        outputs=[summary_box])
    stats_btn.click(fetch_stats,      inputs=[session_id],                        outputs=[stats_box])
    search_btn.click(semantic_search, inputs=[session_id, search_query, search_k],outputs=[search_box])

demo.launch()