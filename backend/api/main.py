"""
FastAPI application entry point — Umba Fraud Detection Platform.
Run with: uvicorn api.main:app --reload --port 8000 --host 127.0.0.1
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers import (
    analytics, transactions, predictions, review,
    training, explainability, pipeline,
)
from api.routers.auth import router as auth_router, require_session
from pipeline.db import init_db
from config.settings import FRONTEND_ORIGIN

app = FastAPI(
    title="Umba Fraud Detection API",
    description=(
        "Real-time fraud detection: ML pipeline, "
        "model comparison, explainability, and human-in-the-loop review."
    ),
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "https://fraudshield-ui.onrender.com",
        "http://localhost:4200",
        "http://localhost:4300",
        "http://127.0.0.1:4300",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Public ───────────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api")

# ── Protected ─────────────────────────────────────────────────────────────────

app.include_router(pipeline.router,        prefix="/api/pipeline",        tags=["Pipeline"])
app.include_router(analytics.router,       prefix="/api/analytics",       tags=["Analytics"])
app.include_router(transactions.router,    prefix="/api/transactions",    tags=["Transactions"])
app.include_router(predictions.router,     prefix="/api/predictions",     tags=["Predictions"])
app.include_router(review.router,          prefix="/api/review",          tags=["Review"])
app.include_router(training.router,        prefix="/api/training",        tags=["Training"])
app.include_router(explainability.router,  prefix="/api/explainability",  tags=["Explainability"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Umba Fraud Detection API v1.0"}


@app.on_event("startup")
def on_startup():
    init_db()
    # Auto-ingest raw CSVs if DB is empty (first deploy on a fresh instance)
    import threading
    from pipeline.db import Session as _Session, Transaction as _Transaction
    from config.settings import RAW_DATA_DIR as _RAW
    from pathlib import Path as _Path
    session = _Session()
    try:
        count = session.query(_Transaction).count()
    finally:
        session.close()
    if count == 0 and (_Path(_RAW) / "train.csv").exists():
        print("[STARTUP] Empty DB detected — auto-ingesting raw CSVs...")
        from api.routers.pipeline import _ingest_bg
        t = threading.Thread(target=_ingest_bg, args=(_RAW,), daemon=True)
        t.start()


# ── Serve Angular SPA (must come LAST) ───────────────────────────────────────
# Angular 17+ builder outputs to dist/<project>/browser/
_CANDIDATE_PATHS = [
    # Relative: works locally and when rootDir=backend on Render
    Path(__file__).resolve().parent.parent.parent / "frontend" / "dist" / "fraud-detection" / "browser",
    Path(__file__).resolve().parent.parent.parent / "frontend" / "dist" / "fraud-detection",
    # Absolute Render path
    Path("/opt/render/project/src/frontend/dist/fraud-detection/browser"),
    Path("/opt/render/project/src/frontend/dist/fraud-detection"),
]

_FRONTEND_DIST: Path | None = next((p for p in _CANDIDATE_PATHS if p.exists()), None)

if _FRONTEND_DIST:
    print(f"[SPA] Serving Angular from: {_FRONTEND_DIST}")
    # Serve static assets (js, css, images) directly
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets") if (_FRONTEND_DIST / "assets").exists() else None

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        # Skip API routes (shouldn't reach here, but safety net)
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404)
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        # All other paths → index.html (Angular client-side routing)
        return FileResponse(str(_FRONTEND_DIST / "index.html"))
else:
    print("[SPA] Angular dist not found — serving API only")

    @app.get("/", include_in_schema=False)
    def root():
        return {"status": "ok", "docs": "/api/docs", "health": "/api/health"}
