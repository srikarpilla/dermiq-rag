"""
DermIQ API Server — Render-compatible
Uvicorn binds the port IMMEDIATELY, engine loads in background thread.
This prevents Render's port-scan timeout during heavy model initialization.
"""

import sys
import os
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import DermIQEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("dermiq-api")

# ── Singleton engine ──────────────────────────────────────────────────────────
engine = DermIQEngine()


def _init_engine_background():
    """
    Run in a daemon thread so uvicorn can bind the port instantly.
    Render detects the open port, then this finishes in the background.
    Queries return 503 until ready — the frontend shows "Initializing..." status.
    """
    logger.info("Background thread: starting engine initialization...")
    try:
        engine.initialize(force_rebuild=False)
        logger.info("Background thread: engine ready ✓")
    except Exception as exc:
        logger.error(f"Background thread: engine init failed — {exc}", exc_info=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DermIQ server starting — launching engine in background thread")
    t = threading.Thread(target=_init_engine_background, daemon=True)
    t.start()
    # Yield immediately — port is open, Render is happy
    yield
    logger.info("DermIQ server shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermIQ — Skin Disease RAG API",
    description=(
        "Production-grade RAG API for skin disease classification. "
        "LangGraph + Cohere command-r-plus-08-2024 + sentence-transformers + FAISS."
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

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"Frontend mounted from: {frontend_path}")


# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Clinical question about a skin condition",
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
    confirm: bool = Field(..., description="Must be true to trigger rebuild")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({
        "message": "DermIQ API is running",
        "status":  "ready" if engine.is_ready else "initializing",
        "docs":    "/docs",
        "health":  "/health",
    })


@app.get("/health", tags=["System"])
async def health_check():
    """Health check — also used by the frontend to poll engine readiness."""
    stats = engine.get_stats()
    return {
        "status":  "healthy" if engine.is_ready else "initializing",
        "service": "DermIQ RAG API",
        "version": "1.0.0",
        "engine":  stats,
    }


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def run_query(request: QueryRequest):
    """Submit a clinical skin disease question to the RAG pipeline."""
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Engine is still initializing — please wait ~2 minutes and retry.",
        )
    try:
        result = engine.query(request.question)
        return QueryResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Query error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error. Please retry.")


@app.post("/api/rebuild-index", tags=["Admin"])
async def rebuild_vector_index(request: RebuildRequest):
    """Force a full FAISS index rebuild from source documents."""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Set 'confirm': true to proceed.")
    try:
        engine.build_vectorstore(force_rebuild=True)
        return {"status": "success", "stats": engine.get_stats()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stats", tags=["System"])
async def get_engine_stats():
    return engine.get_stats()


# ── Entry point ───────────────────────────────────────────────────────────────
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
