# run:  uvicorn api:api --reload --host 127.0.0.1 --port 8000

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np

from app import chat, clear_session, get_summary, get_raw_history, startup, PG_PATH
from db import load_history_paginated, search_similar, session_stats
from embeddings import embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


api = FastAPI(title="HF LangChain Chat API", version="2.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    question:   str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    answer:     str

class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        answer = chat(req.question, req.session_id)
        return ChatResponse(session_id=req.session_id, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Chat error: {e}")


@api.get("/history/{session_id}")
def history(
    session_id: str,
    limit:  int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0,   ge=0),
):
    try:
        rows  = load_history_paginated(PG_PATH, session_id, limit=limit, offset=offset)
        items = [{"role": r["role"], "content": r["content"]} for r in rows]
        return {"session_id": session_id, "history": items,
                "limit": limit, "offset": offset, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB error: {e}")


@api.get("/history/{session_id}/search")
def semantic_search(
    session_id: str,
    query: str = Query(..., min_length=1),
    k:     int = Query(default=5, ge=1, le=50),
):
    try:
        query_vec = embed(query)
        if query_vec is None:
            raise HTTPException(status_code=503, detail="Embedding model unavailable.")
        results = search_similar(PG_PATH, session_id, np.array(query_vec), k=k)
        items = [
            {"role": r["role"], "content": r["content"], "score": round(float(r["score"]), 4)}
            for r in results
        ]
        return {"session_id": session_id, "query": query, "results": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search error: {e}")


@api.get("/summary/{session_id}")
def summary(session_id: str):
    try:
        return {"session_id": session_id, "summary": get_summary(session_id)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error: {e}")


@api.post("/clear")
def clear(req: ClearRequest):
    try:
        clear_session(req.session_id)
        return {"status": "cleared", "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB error: {e}")


@api.get("/session/{session_id}/stats")
def stats(session_id: str):
    try:
        counts = session_stats(PG_PATH, session_id)
        total  = sum(counts.values())
        return {"session_id": session_id, "stats": counts, "total": total}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB error: {e}")