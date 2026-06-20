"""
Central configuration for the Umba Fraud Detection Platform.

Directory layout — everything lives under fraud_detection/:

  candidate/
    IVR/                            ← separate reference project
    fraud_detection/
      DATA_DICTIONARY.md
      README.md
      SETUP.md
      backend/
        config/settings.py          ← this file
        data/
          raw/                      ← train.csv, test.csv, identity.csv  (RAW_DATA_DIR)
            train.csv
            test.csv
            identity.csv
            sample_submission.csv
          fraud.db                  ← SQLite database           (DATA_DIR)
          mlflow.db                 ← MLflow tracking
          predictions.csv           ← submission output
        models/                     ← .pkl artifacts             (MODEL_DIR)
        .env
      frontend/                     ← Angular SPA

DB_MODE: sqlite (default) or postgres — set in backend/.env
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_FILE.exists():
        load_dotenv(str(_ENV_FILE))
except ImportError:
    pass

# ── Path resolution ───────────────────────────────────────────────────────────
# settings.py: candidate/fraud_detection/backend/config/settings.py
_CONFIG_DIR  = Path(__file__).resolve().parent        # .../backend/config
_BACKEND_DIR = _CONFIG_DIR.parent                      # .../backend

BASE_DIR     = str(_BACKEND_DIR.parent)                # .../fraud_detection

# Raw CSVs — inside backend/data/raw/ (all data owned by backend)
RAW_DATA_DIR = str(_BACKEND_DIR / "data" / "raw")

# Runtime data dir — DB, MLflow, predictions go here
# On Render, use the persistent disk mount; locally use backend/data/
_RENDER_DISK = "/data"
if os.path.exists(_RENDER_DISK) and os.access(_RENDER_DISK, os.W_OK):
    DATA_DIR = _RENDER_DISK
    print(f"[DATA] Using Render persistent disk: {DATA_DIR}")
else:
    DATA_DIR = str(_BACKEND_DIR / "data")

# Trained model artifacts
MODEL_DIR    = str(_BACKEND_DIR / "models")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR,     exist_ok=True)
os.makedirs(MODEL_DIR,    exist_ok=True)

# Validate CSVs are present and warn clearly
_raw = Path(RAW_DATA_DIR)
_missing = [f for f in ("train.csv", "test.csv", "identity.csv") if not (_raw / f).exists()]
if _missing:
    print(f"[WARN] Missing CSVs in {RAW_DATA_DIR}: {_missing}")
else:
    print(f"[DATA] Raw CSVs: {RAW_DATA_DIR}")
    print(f"[DATA] Runtime data: {DATA_DIR}")
    print(f"[DATA] Models: {MODEL_DIR}")

# ── Database ──────────────────────────────────────────────────────────────────
DB_MODE = os.getenv("DB_MODE", "sqlite").lower().strip()

if DB_MODE == "postgres":
    _PG_USER   = os.getenv("DB_USER",     "postgres")
    _PG_PASS   = os.getenv("DB_PASSWORD", "")
    _PG_HOST   = os.getenv("DB_HOST",     "localhost")
    _PG_PORT   = os.getenv("DB_PORT",     "5432")
    _PG_DB     = os.getenv("DB_NAME",     "fraud_detection")

    from urllib.parse import quote_plus as _qp
    _PG_PASS_ENC = _qp(_PG_PASS)
    _PG_URL = (
        f"postgresql+psycopg2://{_PG_USER}:{_PG_PASS_ENC}"
        f"@{_PG_HOST}:{_PG_PORT}/{_PG_DB}"
    )

    try:
        import psycopg2 as _pg2
        _c = _pg2.connect(host=_PG_HOST, port=int(_PG_PORT), dbname=_PG_DB,
                          user=_PG_USER, password=_PG_PASS, connect_timeout=5)
        _c.close()
        DB_URL              = _PG_URL
        DB_PATH             = None
        MLFLOW_TRACKING_URI = _PG_URL
        print(f"[DB] PostgreSQL at {_PG_HOST}:{_PG_PORT}/{_PG_DB}")
    except Exception as _e:
        print(f"[DB] PostgreSQL unavailable ({_e}) — falling back to SQLite")
        DB_MODE = "sqlite"

if DB_MODE == "sqlite":
    DB_PATH             = os.path.join(DATA_DIR, "fraud.db")
    DB_URL              = f"sqlite:///{DB_PATH}"
    MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(DATA_DIR, 'mlflow.db')}"
    print(f"[DB] SQLite: {DB_PATH}")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection")

# ── Model thresholds ──────────────────────────────────────────────────────────
FRAUD_THRESHOLD         = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
REVIEW_THRESHOLD_LOW    = float(os.getenv("REVIEW_THRESHOLD_LOW",  "0.3"))
REVIEW_THRESHOLD_HIGH   = float(os.getenv("REVIEW_THRESHOLD_HIGH", "0.7"))

# ── Auth ──────────────────────────────────────────────────────────────────────
SESSION_SECRET_KEY     = os.getenv("SESSION_SECRET_KEY",    "change-me-in-production")
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "28800"))
FRONTEND_ORIGIN        = os.getenv("FRONTEND_ORIGIN",       "http://localhost:4300")

# ── Pipeline ──────────────────────────────────────────────────────────────────
FLAML_TIME_BUDGET    = int(os.getenv("FLAML_TIME_BUDGET", "120"))   # seconds
CV_FOLDS             = int(os.getenv("CV_FOLDS", "5"))
RANDOM_STATE         = 42
