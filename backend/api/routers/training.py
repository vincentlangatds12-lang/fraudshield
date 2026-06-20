"""
/api/training — trigger pipeline, get MLflow runs, model versions.
"""
from __future__ import annotations

import sys, os

# ── Fix OpenMP double-load conflict (libiomp vs libomp) ──────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import threading
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import desc

from pipeline.db import Session, ModelRun
from pipeline.mlflow_tracking import get_all_runs
from config.settings import MLFLOW_TRACKING_URI

router = APIRouter()

_pipeline_status: dict = {"running": False, "last_run_id": None, "last_error": None}
_pipeline_thread: threading.Thread | None = None


class PipelineRequest(BaseModel):
    data_dir:     str | None = None
    flaml_budget: int = 300   # default 300s per model


def _run_pipeline_bg(data_dir: str | None, flaml_budget: int):
    """Run pipeline in a real OS thread — not anyio background task.
    This prevents asyncio CancelledError when the HTTP request completes."""
    global _pipeline_status
    _pipeline_status["running"]    = True
    _pipeline_status["last_error"] = None
    try:
        from pipeline.run_pipeline import run
        result = run(data_dir=data_dir, flaml_budget=flaml_budget)
        _pipeline_status["last_run_id"] = result.get("run_id")
    except Exception as exc:
        import traceback
        _pipeline_status["last_error"] = str(exc)
        print(f"[PIPELINE] Error: {exc}")
        traceback.print_exc()
    finally:
        _pipeline_status["running"] = False


@router.post("/run")
def trigger_pipeline(body: PipelineRequest):
    """Start the pipeline in a real daemon thread (not anyio BackgroundTask)
    to prevent CancelledError when the HTTP request completes."""
    global _pipeline_thread
    if _pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    t = threading.Thread(
        target=_run_pipeline_bg,
        args=(body.data_dir, body.flaml_budget),
        daemon=True,
        name="pipeline-worker",
    )
    t.start()
    _pipeline_thread = t
    return {"status": "started", "message": "Pipeline running in background (daemon thread)"}


@router.get("/status")
def get_pipeline_status():
    return {
        "running":      _pipeline_status["running"],
        "last_run_id":  _pipeline_status["last_run_id"],
        "last_error":   _pipeline_status["last_error"],
    }


@router.get("/mlflow-runs")
def get_mlflow_runs():
    return get_all_runs()


@router.get("/mlflow-url")
def get_mlflow_url():
    return {"url": MLFLOW_TRACKING_URI}


@router.get("/versions")
def get_model_versions():
    session = Session()
    try:
        runs = session.query(ModelRun)\
                      .order_by(desc(ModelRun.trained_at)).limit(100).all()
        return [
            {
                "id":                 r.id,
                "run_id":             r.run_id,
                "classifier_name":    r.classifier_name,
                "imbalance_strategy": r.imbalance_strategy,
                "is_champion":        r.is_champion,
                "auc_roc":            round(r.auc_roc or 0, 4),
                "pr_auc":             round(r.pr_auc or 0, 4),
                "f1_fraud":           round(r.f1_fraud or 0, 4),
                "trained_at":         r.trained_at.isoformat() if r.trained_at else None,
            }
            for r in runs
        ]
    finally:
        session.close()


@router.get("/metrics/{model_run_id}")
def get_run_metrics(model_run_id: int):
    session = Session()
    try:
        run = session.get(ModelRun, model_run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Model run not found")
        return {
            "classifier_name":    run.classifier_name,
            "imbalance_strategy": run.imbalance_strategy,
            "is_champion":        run.is_champion,
            "accuracy":           round(run.accuracy or 0, 4),
            "auc_roc":            round(run.auc_roc or 0, 4),
            "pr_auc":             round(run.pr_auc or 0, 4),
            "f1_fraud":           round(run.f1_fraud or 0, 4),
            "precision_fraud":    round(run.precision_fraud or 0, 4),
            "recall_fraud":       round(run.recall_fraud or 0, 4),
            "mcc":                round(run.mcc or 0, 4),
            "ks_statistic":       round(run.ks_statistic or 0, 4),
            "training_duration_s":round(run.training_duration_s or 0, 1),
            "trained_at":         run.trained_at.isoformat() if run.trained_at else None,
        }
    finally:
        session.close()


@router.post("/rechampion")
def rechampion():
    """
    Re-evaluate all trained models with the composite scoring formula and update
    the is_champion flag. Automatically regenerates predictions.csv with the new champion.
    Composite = 0.50*recall + 0.20*pr_auc + 0.20*auc_roc + 0.10*f1
    """
    session = Session()
    try:
        runs = session.query(ModelRun).all()
        if not runs:
            return {"status": "no_models", "message": "No models in DB — run the pipeline first"}

        def _score(r) -> float:
            return (0.50 * (r.recall_fraud  or 0) +
                    0.20 * (r.pr_auc        or 0) +
                    0.20 * (r.auc_roc       or 0) +
                    0.10 * (r.f1_fraud      or 0))

        best     = max(runs, key=_score)
        prev_champ_id = next((r.id for r in runs if r.is_champion), None)

        for r in runs:
            r.is_champion = (r.id == best.id)
        session.commit()

        result = {
            "status":        "ok",
            "champion":      best.classifier_name,
            "strategy":      best.imbalance_strategy,
            "composite":     round(_score(best), 4),
            "recall":        round(best.recall_fraud  or 0, 4),
            "pr_auc":        round(best.pr_auc        or 0, 4),
            "auc_roc":       round(best.auc_roc       or 0, 4),
            "f1_fraud":      round(best.f1_fraud      or 0, 4),
            "total_models":  len(runs),
            "predictions_updated": False,
        }

        # ── Auto-regenerate predictions.csv if champion changed ───────────────
        if best.artifact_path:
            try:
                import joblib, numpy as np, pandas as pd
                from pathlib import Path
                from config.settings import RAW_DATA_DIR, DATA_DIR

                artifact = joblib.load(best.artifact_path)
                clf_name = artifact.get("classifier_name", "")

                # Load test features
                test_csv = pd.read_csv(Path(RAW_DATA_DIR) / "test.csv")
                test_sorted = test_csv.sort_values("TransactionDT").reset_index(drop=True)

                # Rebuild features (import locally to avoid circular import at module load)
                from pipeline.feature_engineering import build_features
                train_csv    = pd.read_csv(Path(RAW_DATA_DIR) / "train.csv")
                identity_csv = pd.read_csv(Path(RAW_DATA_DIR) / "identity.csv")
                _, X_test_df, _ = build_features(train_csv, test_csv, identity_csv)
                X_test = X_test_df.values.astype(np.float32)

                # Score
                if clf_name == "stacking_ensemble":
                    base_paths = artifact.get("base_artifacts", [])
                    meta_X = np.zeros((len(X_test), len(base_paths)), dtype=np.float32)
                    for bi, bp in enumerate(base_paths):
                        try:
                            ba  = joblib.load(bp)
                            scl = ba.get("scaler")
                            Xs  = scl.transform(X_test) if scl else X_test
                            meta_X[:, bi] = ba["model"].predict_proba(Xs)[:, 1]
                        except Exception:
                            meta_X[:, bi] = 0.5
                    proba = artifact["model"].predict_proba(meta_X)[:, 1]
                else:
                    scaler = artifact.get("scaler")
                    Xs     = scaler.transform(X_test) if scaler else X_test
                    proba  = artifact["model"].predict_proba(Xs)[:, 1]

                submission = pd.DataFrame({
                    "TransactionID": test_sorted["TransactionID"].values,
                    "isFraud_prob":  proba,
                })
                out = Path(DATA_DIR) / "predictions.csv"
                submission.to_csv(out, index=False)
                result["predictions_updated"] = True
                result["predictions_path"]    = str(out)
                result["predictions_rows"]    = len(submission)
                print(f"[RECHAMPION] predictions.csv updated — {len(submission)} rows — champion={best.classifier_name}")

            except Exception as e:
                result["predictions_error"] = str(e)
                print(f"[RECHAMPION] predictions.csv update failed: {e}")

        return result
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
