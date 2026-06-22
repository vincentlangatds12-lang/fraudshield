"""
Master pipeline runner — Umba Fraud Detection Platform.

16 stages:
  1.  Load raw data
  2.  Data integrity checks
  3.  Feature engineering (20+ groups)
  4.  Imbalance analysis
  5.  Temporal train/val split (last 20% by TransactionDT)
  6.  Train all classifiers (LR + 4 FLAML models × 2 strategies)
      → each saved to DB immediately after training
  7.  Log to MLflow
  8.  SHAP global explanations
  9.  Persist imbalance report
  10. Generate predictions.csv (champion model)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODEL_DIR, RAW_DATA_DIR, DATA_DIR, RANDOM_STATE
from pipeline.db import (
    Session, init_db, ModelRun,
    FeatureImportance, ImbalanceReport,
)
from pipeline.feature_engineering import build_features
from pipeline.imbalance import analyse_imbalance
from pipeline.train import train_all_classifiers, _champion_score
from pipeline.mlflow_tracking import setup_mlflow, log_run

logger = logging.getLogger(__name__)


def setup_logging():
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Load data
# ═══════════════════════════════════════════════════════════════════════════

def load_data(data_dir: str):
    p = Path(data_dir)
    logger.info("Loading raw data from %s …", p)
    train    = pd.read_csv(p / "train.csv")
    test     = pd.read_csv(p / "test.csv")
    identity = pd.read_csv(p / "identity.csv")
    logger.info("Train=%d  Test=%d  Identity=%d", len(train), len(test), len(identity))
    return train, test, identity


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Integrity checks
# ═══════════════════════════════════════════════════════════════════════════

def integrity_checks(train, test, identity) -> dict:
    report = {}
    train_max = train["TransactionDT"].max()
    test_min  = test["TransactionDT"].min()
    report["temporal_split_ok"] = bool(test_min > train_max)
    if report["temporal_split_ok"]:
        logger.info("INTEGRITY OK Temporal split: train_max=%d < test_min=%d",
                    train_max, test_min)
    else:
        logger.warning("INTEGRITY: Test overlaps train!")

    id_counts = identity.groupby("TransactionID").size()
    report["identity_multi_row_pct"] = float((id_counts > 1).mean())
    logger.info("INTEGRITY: Identity multi-row pct=%.1f%%",
                report["identity_multi_row_pct"] * 100)

    if "flagged_for_review" in train.columns:
        logger.warning("LEAKAGE CHECK: flagged_for_review EXCLUDED from features")

    kes = train[train["currency"] == "KES"]["TransactionAmt"].median()
    ngn = train[train["currency"] == "NGN"]["TransactionAmt"].median()
    logger.info("INTEGRITY: KES median=%.0f  NGN median=%.0f", kes, ngn)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# IMMEDIATE PERSISTENCE (called from train.py after each model finishes)
# ═══════════════════════════════════════════════════════════════════════════

def persist_one(result: dict, run_id: str) -> int | None:
    """
    Persist a single model result immediately after it finishes.
    Called from train.py — saves to DB without waiting for all models.
    """
    session = Session()
    try:
        metrics = result["metrics"]

        # Deduplication
        existing = session.query(ModelRun).filter_by(
            run_id             = result["run_id"],
            classifier_name    = result["classifier_name"],
            imbalance_strategy = result["imbalance_strategy"],
        ).first()
        if existing:
            logger.debug("Duplicate skip: %s/%s", result["classifier_name"],
                         result["imbalance_strategy"])
            return existing.id

        mr = ModelRun(
            run_id               = result["run_id"],
            classifier_name      = result["classifier_name"],
            imbalance_strategy   = result["imbalance_strategy"],
            hyperparams          = json.dumps(result.get("hyperparams", {})),
            accuracy             = metrics.get("accuracy"),
            auc_roc              = metrics.get("auc_roc"),
            pr_auc               = metrics.get("pr_auc"),
            f1_fraud             = metrics.get("f1_fraud"),
            precision_fraud      = metrics.get("precision_fraud"),
            recall_fraud         = metrics.get("recall_fraud"),
            mcc                  = metrics.get("mcc"),
            ks_statistic         = metrics.get("ks_statistic"),
            avg_precision        = metrics.get("avg_precision"),
            artifact_path        = result.get("artifact_path"),
            is_champion          = result.get("is_champion", False),
            training_duration_s  = result.get("training_duration_s"),
            max_recall           = result.get("max_recall"),
            max_recall_threshold = result.get("max_recall_threshold"),
        )
        session.add(mr)
        session.flush()
        mr_id = mr.id

        for fi in result.get("feature_importance", []):
            session.add(FeatureImportance(
                model_run_id = mr_id,
                feature_name = fi["feature"],
                importance   = fi["importance"],
                rank         = fi["rank"],
            ))

        session.commit()
        logger.info("Saved: %-22s [%-12s]  AUC=%.4f  Recall=%.4f  PR-AUC=%.4f",
                    result["classifier_name"], result["imbalance_strategy"],
                    metrics.get("auc_roc", 0), result.get("max_recall", 0),
                    metrics.get("pr_auc", 0))
        return mr_id

    except Exception as exc:
        session.rollback()
        logger.error("persist_one error for %s: %s", result["classifier_name"], exc)
        return None
    finally:
        session.close()


def update_champion_in_db(run_id: str):
    """
    Re-evaluate all runs with composite score and update is_champion flag.
    Called after every model is saved — champion updates in real-time.
    """
    session = Session()
    try:
        all_runs = session.query(ModelRun).all()
        if not all_runs:
            return

        def _score(r):
            rec = r.max_recall or r.recall_fraud or 0
            return (0.50 * rec +
                    0.20 * (r.pr_auc    or 0) +
                    0.20 * (r.auc_roc   or 0) +
                    0.10 * (r.f1_fraud  or 0))

        best = max(all_runs, key=_score)
        for r in all_runs:
            r.is_champion = (r.id == best.id)
        session.commit()
        logger.info("Champion → %s [%s]  composite=%.4f  recall=%.4f",
                    best.classifier_name, best.imbalance_strategy,
                    _score(best), best.max_recall or best.recall_fraud or 0)
    except Exception as exc:
        session.rollback()
        logger.error("update_champion_in_db error: %s", exc)
    finally:
        session.close()


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_predictions(champion_artifact_path: str, X_test: np.ndarray,
                          test_ids: pd.Series, output_dir: str) -> str:
    import joblib
    artifact = joblib.load(champion_artifact_path)
    clf      = artifact["model"]
    scaler   = artifact.get("scaler")
    X        = scaler.transform(X_test) if scaler else X_test
    proba    = clf.predict_proba(X)[:, 1]

    submission = pd.DataFrame({
        "TransactionID": test_ids.values,
        "isFraud_prob":  proba,
    })
    out = Path(output_dir) / "predictions.csv"
    submission.to_csv(out, index=False)
    logger.info("Predictions → %s  (%d rows)", out, len(submission))
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run(data_dir: str | None = None, flaml_budget: int | None = None):
    setup_logging()
    init_db()
    setup_mlflow()

    run_id   = str(uuid.uuid4())
    start    = datetime.now()
    data_dir = data_dir or RAW_DATA_DIR

    logger.info("=" * 70)
    logger.info("UMBA FRAUD PIPELINE — %s", start.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("run_id=%s", run_id)
    logger.info("=" * 70)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    logger.info("\n[Stage 1] Loading data …")
    train, test, identity = load_data(data_dir)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    logger.info("\n[Stage 2] Data integrity checks …")
    integrity = integrity_checks(train, test, identity)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    logger.info("\n[Stage 3] Feature engineering …")
    X_train_df, X_test_df, feature_names = build_features(train, test, identity)

    train_sorted = train.sort_values("TransactionDT").reset_index(drop=True)
    test_sorted  = test.sort_values("TransactionDT").reset_index(drop=True)

    y_train  = train_sorted["isFraud"].values.astype(np.int32)
    test_ids = test_sorted["TransactionID"]
    X_train  = X_train_df.values.astype(np.float32)
    X_test   = X_test_df.values.astype(np.float32)

    assert len(X_train) == len(y_train)
    logger.info("Features=%d  Train=%d  Test=%d", len(feature_names), len(X_train), len(X_test))

    # ── Stage 4: REMOVED (imbalance analysis) ────────────────────────────────
    # We train with BOTH class_weight AND adasyn for every model — no need to
    # run a slow comparison. Both strategies are always evaluated.
    imbalance_info = {
        "fraud_count":     int((y_train == 1).sum()),
        "legit_count":     int((y_train == 0).sum()),
        "imbalance_ratio": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "fraud_rate":      float((y_train == 1).mean()),
    }
    imbalance_comp = {"_best": "class_weight"}
    logger.info("Imbalance: fraud=%d (%.2f%%)  legit=%d  ratio=%.1f:1",
                imbalance_info["fraud_count"],
                imbalance_info["fraud_rate"] * 100,
                imbalance_info["legit_count"],
                imbalance_info["imbalance_ratio"])

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    logger.info("\n[Stage 5] Temporal train/val split …")
    n_val = int(len(X_train) * 0.2)
    n_tr  = len(X_train) - n_val
    X_tr, y_tr   = X_train[:n_tr],  y_train[:n_tr]
    X_val, y_val = X_train[n_tr:],  y_train[n_tr:]
    logger.info("Train=%d  Val=%d", len(X_tr), len(X_val))

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    logger.info("\n[Stage 6] Training classifiers …")
    results = train_all_classifiers(
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,  y_val=y_val,
        feature_names=feature_names,
        run_id=run_id,
    )

    # ── Stage 7: MLflow ───────────────────────────────────────────────────────
    logger.info("\n[Stage 7] Logging to MLflow …")
    for res in results:
        mlflow_id = log_run(res, feature_count=len(feature_names))
        if mlflow_id:
            res["mlflow_run_id"] = mlflow_id
            session = Session()
            try:
                row = session.query(ModelRun).filter_by(
                    run_id=run_id, classifier_name=res["classifier_name"],
                    imbalance_strategy=res["imbalance_strategy"],
                ).first()
                if row:
                    row.mlflow_run_id = mlflow_id
                    session.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()

    # ── Stage 8: SHAP ─────────────────────────────────────────────────────────
    logger.info("\n[Stage 8] SHAP global explanations …")
    champion = next((r for r in results if r.get("is_champion")), None)
    if not champion:
        champion = max(results, key=_champion_score) if results else None

    if champion and champion.get("artifact_path"):
        try:
            import joblib
            from pipeline.explain import global_shap
            from pipeline.db import ShapExplanation
            artifact  = joblib.load(champion["artifact_path"])
            shap_vals = global_shap(artifact, X_val, feature_names, max_samples=300)
            if shap_vals:
                session = Session()
                try:
                    champ_row = session.query(ModelRun).filter_by(
                        run_id=run_id, classifier_name=champion["classifier_name"],
                        imbalance_strategy=champion["imbalance_strategy"],
                    ).first()
                    if champ_row:
                        session.add(ShapExplanation(
                            transaction_id   = None,
                            model_run_id     = champ_row.id,
                            shap_values_json = json.dumps({s["feature"]: s["mean_abs_shap"]
                                                           for s in shap_vals}),
                            base_value       = 0.0,
                            predicted_prob   = 0.0,
                        ))
                        session.commit()
                        logger.info("SHAP global saved — top: %s",
                                    shap_vals[0]["feature"] if shap_vals else "n/a")
                except Exception as e:
                    session.rollback()
                    logger.warning("SHAP save error: %s", e)
                finally:
                    session.close()
        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)

    # ── Stage 9: Imbalance report ─────────────────────────────────────────────
    logger.info("\n[Stage 9] Saving imbalance report …")
    session = Session()
    try:
        if not session.query(ImbalanceReport).filter_by(run_id=run_id).first():
            session.add(ImbalanceReport(
                run_id              = run_id,
                fraud_count         = imbalance_info["fraud_count"],
                legit_count         = imbalance_info["legit_count"],
                imbalance_ratio     = imbalance_info["imbalance_ratio"],
                strategy_used       = imbalance_comp.get("_best", "class_weight"),
                strategy_comparison = json.dumps({k: v for k, v in imbalance_comp.items()
                                                  if k != "_best"}),
            ))
            session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Imbalance report error: %s", e)
    finally:
        session.close()

    # ── Stage 10: Predictions ─────────────────────────────────────────────────
    logger.info("\n[Stage 10] Generating predictions.csv …")
    pred_path = None
    if champion and champion.get("artifact_path"):
        try:
            pred_path = generate_predictions(
                champion["artifact_path"], X_test, test_ids, DATA_DIR
            )
        except Exception as e:
            logger.error("Predictions error: %s", e)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE — models=%d  elapsed=%.0fs", len(results), elapsed)
    if champion:
        m = champion["metrics"]
        logger.info("Champion: %s [%s]  AUC=%.4f  Recall=%.4f  PR-AUC=%.4f",
                    champion["classifier_name"], champion["imbalance_strategy"],
                    m.get("auc_roc", 0), champion.get("max_recall", 0), m.get("pr_auc", 0))
    logger.info("Predictions: %s", pred_path)
    logger.info("=" * 70)

    return {
        "run_id":    run_id,
        "results":   results,
        "champion":  champion,
        "pred_path": pred_path,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Umba Fraud Pipeline")
    ap.add_argument("--data-dir",      default=None)
    ap.add_argument("--flaml-budget",  type=int, default=None)
    args = ap.parse_args()
    run(data_dir=args.data_dir, flaml_budget=args.flaml_budget)
