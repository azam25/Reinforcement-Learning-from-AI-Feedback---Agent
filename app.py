"""
FastAPI service for the RLAIF RAG Agent.

Endpoints:
    POST /ingest   — Load and index a PDF document.
    POST /ask      — Answer a question against the indexed document.
    GET  /health   — Liveness check.

Run locally:
    uvicorn RLAIF_RAG_Agent.app:app --reload

Or via the settings:
    python -m RLAIF_RAG_Agent.app
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import RetrievalAgent as retrieval_agent
from .context import AgentConfig, AgentResponse, RequestContext
from .config_instance import pedantic_instance
from . import RAGAgent
from .settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "RLAIF RAG Agent service starting (model=%s, host=%s:%d)",
        settings.model_name,
        settings.service_host,
        settings.service_port,
    )
    yield
    logger.info("RLAIF RAG Agent service shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RLAIF RAG Agent",
    description=(
        "Production-grade Retrieval-Augmented Generation agent with "
        "RLAIF self-evaluation, iterative context refinement, and "
        "conversational memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the PDF file to ingest.")
    force_rebuild: bool = Field(
        False,
        description="Force re-embedding even if a cached index exists.",
    )


class IngestResponse(BaseModel):
    message: str
    file_path: str


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="The question to answer.",
    )
    rlaif: bool = Field(True, description="Enable RLAIF self-evaluation.")
    verbose: int = Field(1, ge=0, le=2, description="Verbosity level (0-2).")


class AskResponse(BaseModel):
    answer: str
    has_answer: bool
    context_iterations: int
    source_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model": settings.model_name}


@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
@limiter.limit("10/minute")
async def ingest(request: Request, body: IngestRequest):
    """
    Load a PDF, embed it, and persist the FAISS index to disk.
    Subsequent calls with the same file will load from cache unless
    force_rebuild=true.
    """
    try:
        pedantic_instance.retrieval = retrieval_agent.load_and_process_document(
            body.file_path,
            force_rebuild=body.force_rebuild,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed for %s", body.file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return IngestResponse(
        message="Document ingested successfully.",
        file_path=body.file_path,
    )


@app.post("/ask", response_model=AskResponse, tags=["qa"])
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest):
    """
    Answer a question against the currently loaded document.
    Call /ingest first to load a document.
    """
    if pedantic_instance.retrieval is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document loaded. POST to /ingest first.",
        )

    ctx = RequestContext(
        retriever=pedantic_instance.retrieval,
        config=AgentConfig(
            doc_type=pedantic_instance.DocType,
            llm_task=pedantic_instance.llmTask,
            output_format=pedantic_instance.OutputFormat,
            rlaif=body.rlaif,
            short_memory=pedantic_instance.ShortMemory,
            long_memory=pedantic_instance.LongMemory,
            verbose=body.verbose,
        ),
    )

    try:
        response: AgentResponse = RAGAgent.get_answer(body.question, ctx)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Answer generation failed for question: %s", body.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {exc}",
        )

    return AskResponse(
        answer=response.answer,
        has_answer=response.has_answer,
        context_iterations=response.context_iterations,
        source_count=len(response.source_documents),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "RLAIF_RAG_Agent.app:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
