"""FastAPI application entry point for the research query backend.

The app keeps HTTP concerns thin and delegates persistence plus LLM extraction
to dedicated modules so the interview solution is easy to reason about and
extend.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from database import QueryRepository
from models import (
    APIErrorResponse,
    QueryCreateRequest,
    QueryResponse,
)
from services.llm_service import ExtractionServiceError, GroqExtractionService


load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "queries.db")
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

ALLOWED_METHODS = ["GET", "POST"]
ALLOWED_HEADERS = ["Content-Type", "Authorization"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = QueryRepository(DATABASE_PATH)
    await repository.initialize()
    app.state.repository = repository
    app.state.llm_service = GroqExtractionService.from_environment()
    yield
    await repository.close()


app = FastAPI(
    title="AI Research Query API",
    version="1.0.0",
    description="FastAPI backend for extracting structured research intent from natural-language queries.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)


INDEX_HTML = Path(__file__).with_name("index.html")


def get_repository() -> QueryRepository:
    repository = app.state.repository
    if repository is None:
        raise RuntimeError("Repository is not initialized")
    return repository


def get_llm_service() -> GroqExtractionService:
    llm_service = app.state.llm_service
    if llm_service is None:
        raise RuntimeError("LLM service is not initialized")
    return llm_service


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="index.html not found")
    return FileResponse(INDEX_HTML)


def _to_http_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@app.post(
    "/queries",
    response_model=QueryResponse,
    responses={
        400: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
        503: {"model": APIErrorResponse},
    },
    status_code=status.HTTP_201_CREATED,
)
async def create_query(request: QueryCreateRequest) -> QueryResponse:
    repository = get_repository()
    llm_service = get_llm_service()

    try:
        extracted_data = await llm_service.extract_query_intent(request.query)
    except ExtractionServiceError as exc:
        raise _to_http_error(exc) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unexpected error while processing the query",
        ) from exc

    record = await repository.create_query(request.query, extracted_data)
    return QueryResponse.from_record(record)


@app.get(
    "/queries/{query_id}",
    response_model=QueryResponse,
    responses={404: {"model": APIErrorResponse}},
)
async def get_query(query_id: str) -> QueryResponse:
    repository = get_repository()
    record = await repository.get_query(query_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"query '{query_id}' not found",
        )
    return QueryResponse.from_record(record)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
