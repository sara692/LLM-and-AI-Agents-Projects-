from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .agent import AnalyticsAgent
from .config import Settings, get_settings
from .datasets import DatasetStore
from .llm import OpenRouterClient
from .logging_config import setup_logging
from .policy import classify_request

logger = logging.getLogger("agent.api")

_store: DatasetStore | None = None
_agent: AnalyticsAgent | None = None


def get_store() -> DatasetStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = DatasetStore(settings)
        _store.load_directory()
    return _store


def get_agent() -> AnalyticsAgent:
    global _agent
    if _agent is None:
        settings = get_settings()
        _agent = AnalyticsAgent(OpenRouterClient(settings), settings)
    return _agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    store = get_store()
    logger.info("Ready. Datasets: %s | model: %s", [d.id for d in store.list()], get_settings().openrouter_model)
    yield


app = FastAPI(
    title="Policy-Aware Self-Correcting Analytics Agent",
    version=__version__,
    description="Ask questions in plain English; get aggregated answers only.",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------- models
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    dataset_id: str | None = None
    clarifications: dict[str, str] = Field(
        default_factory=dict, description="Answers to a previous clarification: {term: column}"
    )


class ClassifyRequest(BaseModel):
    question: str = Field(max_length=2000)
    dataset_id: str | None = None


class TraceModel(BaseModel):
    action: str
    ok: bool
    detail: str = ""


class AskResponse(BaseModel):
    question: str
    outcome: str  # answer | reject | clarify | fail
    message: str
    answer: str | None = None
    result: dict[str, Any] | None = None
    code: str | None = None
    column_map: dict[str, Any] = {}
    clarification: dict[str, Any] | None = None
    repairs: int = 0
    attempts: int = 0
    steps: int = 0
    latency_s: float = 0.0
    trace: list[TraceModel] = []
    dataset_id: str


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health(settings: Settings = Depends(get_settings), store: DatasetStore = Depends(get_store)):
    return {
        "status": "ok",
        "version": __version__,
        "llm_configured": bool(settings.openrouter_api_key),
        "model": settings.openrouter_model,
        "sandbox_mode": settings.sandbox_mode,
        "datasets": len(store.list()),
    }


@app.get("/datasets")
def list_datasets(store: DatasetStore = Depends(get_store)):
    return [
        {
            "id": d.id,
            "name": d.name,
            "rows": int(len(d.df)),
            "columns": int(len(d.df.columns)),
            "protected_columns": d.protected_columns,
            "is_default": d.id == store.default_id,
        }
        for d in store.list()
    ]


@app.get("/datasets/{dataset_id}/schema")
def dataset_schema(dataset_id: str, store: DatasetStore = Depends(get_store)):
    try:
        ds = store.get(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset '{dataset_id}'.") from None
    return {"id": ds.id, "name": ds.name, "rows": int(len(ds.df)), "columns": ds.public_schema()}


@app.post("/datasets/upload", status_code=201)
async def upload_dataset(file: UploadFile = File(...), store: DatasetStore = Depends(get_store)):
    content = await file.read()
    try:
        ds = store.add_upload(file.filename or "upload.csv", content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"id": ds.id, "name": ds.name, "rows": int(len(ds.df)), "protected_columns": ds.protected_columns}


@app.post("/classify")
def classify(req: ClassifyRequest, store: DatasetStore = Depends(get_store)):
    try:
        ds = store.get(req.dataset_id)
    except KeyError:
        raise HTTPException(404, "Unknown dataset.") from None
    decision = classify_request(
        req.question, known_tokens=ds.known_tokens, known_full_names=ds.known_full_names
    )
    return {"authorized": decision.authorized, "rule": decision.rule, "reason": decision.reason}


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    store: DatasetStore = Depends(get_store),
    agent: AnalyticsAgent = Depends(get_agent),
):
    try:
        ds = store.get(req.dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset '{req.dataset_id}'.") from None
    response = agent.run(req.question, ds, req.clarifications)
    return AskResponse(**response.to_dict(), dataset_id=ds.id)
