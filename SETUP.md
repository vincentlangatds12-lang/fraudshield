# FraudShield — Setup & Run Guide

Umba Fraud Detection Platform · Python 3.11+ · Angular 20 · FastAPI · SQLite · MLflow

---

## Project Structure

```
fraud_detection/
├── backend/
│   ├── api/
│   │   ├── main.py                    ← FastAPI (port 8000)
│   │   └── routers/
│   │       ├── auth.py                ← JWT session login
│   │       ├── analytics.py           ← 20+ dashboard endpoints
│   │       ├── transactions.py        ← transaction explorer
│   │       ├── predictions.py         ← real-time /score
│   │       ├── review.py              ← human-in-the-loop queue
│   │       ├── training.py            ← trigger pipeline + rechampion
│   │       ├── explainability.py      ← SHAP · LIME · feature importance
│   │       └── pipeline.py            ← ingest CSVs → DB
│   ├── config/settings.py             ← all env config
│   ├── pipeline/
│   │   ├── run_pipeline.py            ← master runner (16 stages)
│   │   ├── feature_engineering.py     ← 20+ feature groups
│   │   ├── train.py                   ← 6 FLAML models + stacking ensemble
│   │   ├── optuna_tuning.py           ← Optuna HP tuning
│   │   ├── imbalance.py               ← SMOTE · ADASYN · class_weight
│   │   ├── explain.py                 ← SHAP + LIME
│   │   ├── mlflow_tracking.py         ← MLflow experiment logging
│   │   └── db.py                      ← SQLAlchemy schema
│   ├── data/
│   │   ├── raw/                       ← train.csv · test.csv · identity.csv
│   │   ├── fraud.db                   ← SQLite (auto-created)
│   │   ├── mlflow.db                  ← MLflow tracking
│   │   └── predictions.csv            ← submission output
│   ├── models/                        ← .pkl artifacts (auto-created)
│   └── .env
│
├── frontend/                          ← Angular 20 SPA (port 4300)
│   └── src/app/pages/
│       ├── dashboard/                 ← 16 KPIs + hero metric + model table
│       ├── transactions/              ← explorer with filters
│       ├── model-comparison/          ← all 7 models × 8 metrics
│       ├── model-monitoring/          ← calibration · score dist · MLflow
│       ├── explainability/            ← SHAP global · LIME · feature importance
│       ├── review-queue/              ← human-in-the-loop labeling
│       ├── analytics-3d/              ← scatter · heatmap · polar · boxplot
│       └── training/                  ← 16-step pipeline UI + MLflow runs
│
├── DATA_DICTIONARY.md
├── README.md
└── SETUP.md                           ← this file
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ (use `genai` conda env) |
| Node.js | 18+ |
| npm | 9+ |

---

## Backend Setup

### 1. Activate the conda environment

```bash
conda activate genai
```

### 2. Install all dependencies

```bash
cd fraud_detection/backend
pip install -r requirements.txt
```

Key packages: `fastapi uvicorn pydantic sqlalchemy python-dotenv python-jose joblib numpy pandas scikit-learn lightgbm xgboost catboost flaml imbalanced-learn shap lime mlflow optuna scipy`

### 3. Start the API server

```bash
cd fraud_detection/backend
conda activate genai
uvicorn api.main:app --reload --port 8000 --host 127.0.0.1
```

**API:** http://localhost:8000  
**Swagger docs:** http://localhost:8000/docs

---

## Frontend Setup

### Start dev server

```bash
cd fraud_detection/frontend
npx ng serve --port 4300
```

**App:** http://localhost:4300

### Build for production

```bash
npx ng build --configuration development
```

Output → `frontend/dist/fraud-detection/browser/` — served automatically by FastAPI.

---

## Running the Pipeline (from Frontend)

1. Open **http://localhost:4300**
2. Login: `analyst@umba.com` / `umba2026`
3. Go to **ML Pipeline** tab (⚙️ icon)
4. Click **Ingest Data** — loads all CSVs into SQLite (160k transactions)
5. Click **Run Full Pipeline** — runs all 16 stages:

| Stage | What happens |
|-------|-------------|
| 1 | Load raw data |
| 2 | Data integrity checks (temporal split, leakage guard) |
| 3 | Feature engineering — 20+ groups (velocity, target encoding, network, PCA) |
| 4 | Imbalance analysis — SMOTE vs ADASYN vs class_weight |
| 5 | Temporal train/val split (last 20% by TransactionDT) |
| 6–11 | FLAML AutoML (300s each): RF · LR · CatBoost · LightGBM · XGBoost · ExtraTrees |
| 12 | Stacking ensemble — OOF meta-features → FLAML meta-learner |
| 13 | Log all 7 runs to MLflow |
| 14 | SHAP global explanations for champion |
| 15 | Persist to database (deduplication guard) |
| 16 | Generate `predictions.csv` |

**Total runtime: ~35–45 minutes** (6 × 300s FLAML + stacking ensemble)

### Champion selection formula
```
Composite = 0.50 × Recall + 0.20 × PR-AUC + 0.20 × AUC-ROC + 0.10 × F1
```
Recall is weighted highest — every missed fraud is a real loss.

---

## Demo Login Credentials

| Email | Password | Role |
|-------|----------|------|
| `analyst@umba.com` | `umba2026` | Fraud Analyst |
| `admin@umba.com` | `admin2026` | Admin |
| `demo@umba.com` | `demo` | Viewer |

---

## MLflow UI

```bash
cd fraud_detection/backend
mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000
```

Open: http://localhost:5000

---

## Quick Reference

```bash
# Backend (conda activate genai first)
uvicorn api.main:app --reload --port 8000

# Frontend dev server
npx ng serve --port 4300

# Re-evaluate champion with current weights (POST)
curl -X POST http://localhost:8000/api/training/rechampion

# Check health
curl http://localhost:8000/api/health

# MLflow UI
mlflow ui --backend-store-uri sqlite:///backend/data/mlflow.db --port 5000
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `from jose import jwt` error | Run `conda activate genai` before uvicorn |
| `No module named 'flaml'` | `pip install flaml` in genai env |
| Pipeline error stage 3 | Check `data/raw/` has train.csv, test.csv, identity.csv |
| No champion model | Run pipeline first from the Training tab |
| 401 on all API calls | Login at /login — cookie-based session |
| Port 4300 in use | Kill existing process or use different port |
| Frontend shows blank dashboard | Backend must be running on port 8000 |
