"""
Cosmic-Net FastAPI backend — entry point.

Startup:
  1. Load best_model.pt into memory (once)
  2. Register all routers under /api/v1
  3. Configure CORS for the React frontend

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model_loader import load_model
from app.core.logging import get_logger
from app.routers import predict, explain, virial, overview

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, nothing special on shutdown."""
    logger.info("Starting Cosmic-Net API — loading model...")
    load_model()
    logger.info("Model ready. API is live.")
    yield
    logger.info("Cosmic-Net API shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Neural surrogate for dark matter halo mass prediction. "
        "GNN inference + MC-Dropout uncertainty + PGExplainer + Virial physics check. "
        "Based on HaloGraphNet (Villanueva-Domingo et al., 2022)."
    ),
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI — used as the paper's "code available" link
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the React frontend (localhost:3000 locally, Azure Static Web App in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.azurestaticapps.net",   # update with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"

app.include_router(predict.router,  prefix=PREFIX, tags=["Prediction"])
app.include_router(explain.router,  prefix=PREFIX, tags=["Explainability"])
app.include_router(virial.router,   prefix=PREFIX, tags=["Physics"])
app.include_router(overview.router, prefix=PREFIX, tags=["Data"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": settings.APP_VERSION}
