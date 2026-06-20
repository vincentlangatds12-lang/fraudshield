# FraudShield — Umba Fraud Detection Platform

**Live Dashboard:** https://fraudshield-ui.onrender.com
**API:** https://fraudshield-5.onrender.com/api/docs

---

## Quick Start (local)

```bash
# 1. Clone
git clone https://github.com/vincentlangatds12-lang/fraudshield.git
cd fraudshield

# 2. Backend
cd backend
pip install -r requirements.txt
cp .env.example .env          # edit DB_MODE, SESSION_SECRET_KEY as needed

# 3. Run pipeline (trains all models, generates predictions.csv)
python pipeline/run_pipeline.py

# 4. Start API
uvicorn api.main:app --reload --port 8000

# 5. Frontend (separate terminal)
cd ../frontend
npm install --legacy-peer-deps
npm run build -- --configuration production
# or for dev: npm start
```

The API will be at `http://localhost:8000/api/docs`.

---

## Repository Structure

```
fraudshield/
├── backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI app entry point
│   │   └── routers/                 # analytics, auth, predictions,
│   │                                # review, training, explainability,
│   │                                # transactions, pipeline
│   ├── config/settings.py           # central config (paths, DB, thresholds)
│   ├── pipeline/
│   │   ├── run_pipeline.py          # master pipeline runner (10 stages)
│   │   ├── train.py                 # model training + evaluation
│   │   ├── feature_engineering.py  # 20+ feature groups
│   │   ├── imbalance.py             # class_weight / ADASYN strategies
│   │   ├── explain.py               # SHAP global explanations
│   │   ├── mlflow_tracking.py       # MLflow experiment logging
│   │   └── db.py                    # SQLAlchemy schema + session
│   ├── data/
│   │   ├── raw/                     # train.csv, test.csv, identity.csv
│   │   └── predictions.csv          # model scores on test.csv
│   ├── models/                      # trained .pkl artifacts
│   └── requirements.txt
├── frontend/                        # Angular 20 SPA
│   └── src/app/
│       ├── pages/                   # dashboard, transactions, training,
│       │                            # review-queue, explainability,
│       │                            # model-comparison, model-monitoring
│       └── core/                    # auth, api service, interceptors
├── build.sh                         # Render build script
├── render.yaml                      # Render deployment config
└── README.md
```

---

## Approach

### Data & Feature Engineering

The dataset has three warts that required careful handling:

1. **Temporal leakage** — `test.csv` transactions occur strictly after `train.csv` (verified via `TransactionDT`). Validation uses a temporal hold-out (last 20% of training data by time), not random split, to avoid overly optimistic metrics.

2. **Currency scale mismatch** — KES and NGN amounts differ by ~12×. All amounts are normalised to USD equivalents (`amt_usd`) and log-transformed (`log_amt`, `log_amt_usd`) before modelling.

3. **`flagged_for_review` leakage** — this field is derived from model scores and is explicitly excluded from all features.

**Feature groups (20+):**
- Amount features: raw, USD-normalised, log-transformed, amount bins
- Time features: hour of day, day of week, weekend/night flags, cyclical sin/cos encodings
- Currency/channel/country one-hot encodings
- Identity join features: device type, browser, OS (where available)
- Interaction features: amount × channel, amount × country
- Recipient/sender account age and velocity features

### Model Selection

Trained 10 models: 5 classifiers × 2 imbalance strategies:

| Classifier | Strategy |
|---|---|
| Logistic Regression | class_weight, ADASYN |
| LightGBM (FLAML-tuned) | class_weight, ADASYN |
| XGBoost (FLAML-tuned) | class_weight, ADASYN |
| Random Forest (FLAML-tuned) | class_weight, ADASYN |
| CatBoost (FLAML-tuned) | class_weight, ADASYN |

FLAML AutoML is used for hyperparameter optimisation (300s budget per model, `average_precision` metric).

### Champion Selection

Champion is selected by composite score:

```
composite = 0.50 × recall + 0.20 × PR-AUC + 0.20 × AUC-ROC + 0.10 × F1
```

Recall is weighted highest because the operational cost of missing fraud (false negative) far exceeds the cost of a false alarm.

### Threshold Selection

For every model, thresholds are scanned from 0 → 1 at step 0.02. The threshold maximising F1 on the validation set is selected as the operating point. This gives a principled, operationally meaningful threshold rather than a default 0.5.

### Evaluation Metrics

Primary metric: **PR-AUC** (average precision) — the correct metric under heavy class imbalance (~3.5% fraud rate). Also tracked: AUC-ROC, F1 (fraud class), Precision, Recall, MCC, KS statistic.

---

## Predictions

`backend/data/predictions.csv` contains fraud probability scores for all 40,000 transactions in `test.csv`, in the format:

```csv
TransactionID,isFraud_prob
1120000,0.0019
1120001,0.0011
...
```

To regenerate:
```bash
cd backend
python pipeline/run_pipeline.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/predictions/score` | Score a single transaction |
| POST | `/api/predictions/score-batch` | Score a batch |
| GET | `/api/analytics/summary` | Dashboard KPIs |
| GET | `/api/training/status` | Pipeline status |
| POST | `/api/training/run` | Trigger pipeline |
| GET | `/api/explainability/feature-importance` | Feature importance |
| GET | `/api/explainability/shap/global` | Global SHAP values |
| GET | `/api/review/queue` | Human review queue |

Full interactive docs: `/api/docs`

---

## Dashboard Features

- **Overview** — fraud rate, alarm count, score distribution, top flagged transactions
- **Transactions** — paginated transaction list with filters
- **Advanced Analytics** — 3D risk landscape, model comparison charts, precision-recall curves, calibration plots
- **Explainability** — SHAP global summary, LIME per-transaction explanations, feature importance
- **ML Pipeline** — trigger training, monitor progress, compare model versions
- **Human-in-the-Loop** — review queue for uncertain predictions, analyst decision recording

---

## Trade-offs & What I'd Improve

**Trade-offs made:**
- Used FLAML AutoML with a 300s budget per model rather than exhaustive grid search — faster iteration, good enough for a v1
- SQLite for storage — zero-config for deployment, would switch to PostgreSQL for production
- Temporal validation only (no cross-validation) — cross-validation with time-series data requires careful gap handling; the temporal hold-out is simpler and less leaky

**With more time:**
- Calibration — apply Platt scaling or isotonic regression so probabilities are meaningful for risk scoring, not just ranking
- Drift detection — monitor input feature distributions and prediction score distributions over time using PSI/KS tests
- Active learning loop — use the review queue decisions as labelled data to retrain the model periodically
- Stacking ensemble — combine base model predictions as features for a meta-learner
- Graph features — network-based features capturing transaction velocity and shared identifiers across accounts

---

## AI Tool Usage

Claude (via Kiro) was used extensively throughout this project:

- **Scaffolding** — initial project structure, FastAPI router setup, Angular component boilerplate
- **Pipeline code** — feature engineering ideas, imbalance strategy implementation, MLflow integration
- **Deployment** — Render configuration, CORS setup, build debugging

**What I reviewed and validated:**
- All feature engineering logic was manually checked for leakage (specifically `flagged_for_review` and temporal ordering)
- The champion scoring formula was my own decision — I chose to weight recall at 50% based on the operational context (missing fraud is more costly than false alarms)
- The threshold scanning approach (F1-optimal) was chosen deliberately over default 0.5
- SHAP explanations were verified to match expected feature importance rankings

The AI accelerated implementation significantly. All modelling decisions, metric choices, and trade-offs are mine.
