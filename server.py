"""
DermIQ API Server
FastAPI backend — serves the RAG pipeline and the frontend UI
"""

import sys
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Make sure rag_engine is importable regardless of working directory ────────
sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import DermIQEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("dermiq-api")


# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE INSTANCE  (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────
engine = DermIQEngine()


# ─────────────────────────────────────────────────────────────────────────────
#  LIFESPAN — runs once on startup and once on shutdown
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DermIQ server starting up...")
    try:
        engine.initialize(force_rebuild=False)
        logger.info("RAG engine ready — server is accepting requests")
    except Exception as exc:
        logger.error(f"Engine initialization failed: {exc}")
        logger.warning(
            "Server is running but /api/query will return 503 until the engine is ready."
        )
    yield
    logger.info("DermIQ server shutting down")


# ─────────────────────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermIQ — Skin Disease RAG API",
    description=(
        "Production-grade Retrieval Augmented Generation API for skin disease "
        "classification. Powered by LangGraph + Cohere command-r-plus + "
        "sentence-transformers + FAISS."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the frontend (index.html lives in ../frontend/) ────────────────────
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"Frontend mounted from: {frontend_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Clinical question about a skin condition",
        json_schema_extra={
            "example": "What are the symptoms and treatment options for psoriasis?"
        },
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        return v.strip()


class QueryResponse(BaseModel):
    question:        str
    answer:          str
    sources:         list[str]
    confidence:      str
    processing_time: float
    docs_retrieved:  int
    status:          str = "success"


class RebuildRequest(BaseModel):
    confirm: bool = Field(
        ..., description="Must be true to trigger a full index rebuild"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend UI."""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({
        "message": "DermIQ API is running",
        "docs":    "/docs",
        "health":  "/health",
    })


@app.get("/health", tags=["System"])
async def health_check():
    """System health check — returns engine status and vector store stats."""
    stats = engine.get_stats()
    return {
        "status":  "healthy" if engine.is_ready else "initializing",
        "service": "DermIQ RAG API",
        "version": "1.0.0",
        "engine":  stats,
    }


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def run_query(request: QueryRequest):
    """
    Submit a skin disease question to the RAG pipeline.

    Pipeline steps:
    1. Embed the question with all-MiniLM-L6-v2
    2. MMR search in FAISS vector store
    3. Build context from retrieved chunks
    4. Generate grounded answer via Cohere command-r-plus
    """
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="RAG engine is still initializing — please retry in a moment.",
        )
    try:
        result = engine.query(request.question)
        return QueryResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected query error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )


@app.post("/api/rebuild-index", tags=["Admin"])
async def rebuild_vector_index(request: RebuildRequest):
    """
    Force a full rebuild of the FAISS vector index from source documents.
    Use this after adding new .txt files to docs/skin_diseases/.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set 'confirm': true to proceed with rebuild.",
        )
    try:
        engine.build_vectorstore(force_rebuild=True)
        return {
            "status":  "success",
            "message": "Vector index rebuilt successfully",
            "stats":   engine.get_stats(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stats", tags=["System"])
async def get_engine_stats():
    """Return detailed statistics about the vector store and engine configuration."""
    return engine.get_stats()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
