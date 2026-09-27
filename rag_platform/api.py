"""FastAPI entry point. Run: uvicorn rag_platform.api:app --reload."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag_platform.core import DocumentStore
from rag_platform.generation import ModelConfig, generate_answer


class Document(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class Question(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = DocumentStore(os.getenv("RAG_DB_PATH", "rag.db"))
    app.state.store = store
    try:
        yield
    finally:
        store.close()


app = FastAPI(title="Grounded Retrieval Reference", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
def ingest(document: Document, request: Request) -> dict:
    try:
        count = request.app.state.store.ingest(document.source, document.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"source": document.source, "passages": count}


@app.post("/query")
def query(question: Question, request: Request) -> dict:
    return request.app.state.store.answer(question.query)


@app.post("/query/generate")
def query_generate(question: Question, request: Request) -> dict:
    try:
        config = ModelConfig.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return generate_answer(request.app.state.store, question.query, config)
