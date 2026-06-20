"""
/api/predictions — score a transaction or batch in real time.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc

from pipeline.db import Session, ModelRun, Prediction, ReviewQueueItem
from config.settings import MODEL_DIR, FRAUD_THRESHOLD, REVIEW_THRESHOLD_LOW, REVIEW_THRESHOLD_HIGH

router = APIRouter()

_LOADED_MODELS: dict[str, Any] = {}


def _load_champion() -> dict:
    """Load the champion model artifact, caching in memory."""
    if "champion" in _LOADED_MODELS:
        return _LOADED_MODELS["champion"]

    session = Session()
    try:
        champion = session.query(ModelRun).filter_by(is_champion=True)\
                          .order_by(desc(ModelRun.trained_at)).first()
        if not champion or not champion.artifact_path:
            raise HTTPException(
                status_code=503,
                detail="No champion model found. Run the pipeline first.",
            )
        if not Path(champion.artifact_path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found: {champion.artifact_path}",
            )

        import joblib
        artifact = joblib.load(champion.artifact_path)
        artifact["_db_run_id"] = champion.id
        _LOADED_MODELS["champion"] = artifact
        print(f"[PREDICT] Champion loaded: {champion.classifier_name}")
        return artifact
    finally:
        session.close()


class TransactionInput(BaseModel):
    TransactionID:  int   = Field(..., description="Unique transaction ID")
    TransactionDT:  int   = Field(..., description="Seconds offset timestamp")
    TransactionAmt: float = Field(..., description="Amount in local currency")
    country:        str   = Field(..., description="KE or NG")
    currency:       str   = Field(..., description="KES or NGN")
    channel:        str   = Field(..., description="mobile_money / p2p / bank_transfer / card / airtime / bill_pay")
    card_type:      str   = Field(default="debit")
    card_bank:      str   = Field(default="unknown")
    # Optional fields
    P_emaildomain:  str | None = None
    R_emaildomain:  str | None = None
    card1:          float | None = None
    recipient_account_age_days: int | None = None
    sender_prev_txn_count:      int | None = None


class BatchInput(BaseModel):
    transactions: list[TransactionInput]


def _score_features(artifact: dict, feature_dict: dict) -> float:
    """
    Build a feature vector from a raw transaction dict and score it.
    Handles both regular models and the stacking ensemble.
    """
    feature_names = artifact["feature_names"]
    clf_name      = artifact.get("classifier_name", "")

    # ── Stacking ensemble: score each base model, then meta-learner ──────────
    if clf_name == "stacking_ensemble":
        import joblib as _jl
        base_paths = artifact.get("base_artifacts", [])
        meta_probs = []
        for bp in base_paths:
            try:
                base_art = _jl.load(bp)
                p = _score_features(base_art, feature_dict)
                meta_probs.append(p)
            except Exception:
                meta_probs.append(0.5)
        meta_X = np.array(meta_probs, dtype=np.float32).reshape(1, -1)
        meta_clf = artifact["model"]
        return float(meta_clf.predict_proba(meta_X)[0, 1])

    # ── Regular model ─────────────────────────────────────────────────────────
    clf    = artifact["model"]
    scaler = artifact.get("scaler")

    # Build a 1-row dataframe matching feature names
    row = {f: 0.0 for f in feature_names}

    # Map known fields
    amt = feature_dict.get("TransactionAmt", 0)
    _KES_TO_USD = 1 / 128.0
    _NGN_TO_USD = 1 / 1570.0
    currency = feature_dict.get("currency", "KES")
    row["amt_usd"]     = amt * (_KES_TO_USD if currency == "KES" else _NGN_TO_USD)
    row["log_amt"]     = float(np.log1p(amt))
    row["log_amt_usd"] = float(np.log1p(row["amt_usd"]))

    t = feature_dict.get("TransactionDT", 0)
    row["hour"]       = (t // 3600) % 24
    row["dow"]        = (t // 86400) % 7
    row["is_weekend"] = int(row["dow"] >= 5)
    row["is_night"]   = int(row["hour"] < 6 or row["hour"] >= 22)
    row["hour_sin"]   = float(np.sin(2 * np.pi * row["hour"] / 24))
    row["hour_cos"]   = float(np.cos(2 * np.pi * row["hour"] / 24))
    row["dow_sin"]    = float(np.sin(2 * np.pi * row["dow"] / 7))
    row["dow_cos"]    = float(np.cos(2 * np.pi * row["dow"] / 7))

    if "recipient_account_age_days" in row:
        age = feature_dict.get("recipient_account_age_days")
        row["recip_age_missing"] = 1 if age is None else 0
        row["recip_age_days"]    = age if age is not None else -1

    # Build vector
    x = np.array([row.get(f, 0.0) for f in feature_names], dtype=np.float32).reshape(1, -1)

    if scaler is not None:
        x = scaler.transform(x)

    # Handle FLAML wrapper
    clf_inner = clf
    if hasattr(clf, "model"):
        clf_inner = clf.model.estimator

    return float(clf_inner.predict_proba(x)[0, 1])


@router.post("/score")
def score_transaction(body: TransactionInput):
    """Score a single transaction and return fraud probability + alarm decision."""
    artifact = _load_champion()
    thr = artifact.get("threshold", FRAUD_THRESHOLD)

    try:
        prob     = _score_features(artifact, body.dict())
        is_alarm = prob >= thr
        in_review = (REVIEW_THRESHOLD_LOW <= prob < REVIEW_THRESHOLD_HIGH)

        # Persist prediction if transaction exists in DB
        session = Session()
        try:
            from pipeline.db import Prediction as Pred
            pred = Pred(
                transaction_id=body.TransactionID,
                model_run_id=artifact.get("_db_run_id"),
                fraud_prob=prob,
                is_alarm=is_alarm,
                threshold_used=thr,
            )
            try:
                session.add(pred)
                session.commit()
            except Exception:
                session.rollback()
        finally:
            session.close()

        return {
            "transaction_id": body.TransactionID,
            "fraud_prob":     round(prob, 4),
            "is_alarm":       is_alarm,
            "threshold":      thr,
            "in_review_zone": in_review,
            "decision":       "FRAUD_ALARM" if is_alarm else ("REVIEW" if in_review else "CLEAR"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}")


@router.post("/score-batch")
def score_batch(body: BatchInput):
    """Score a batch of transactions."""
    artifact = _load_champion()
    thr      = artifact.get("threshold", FRAUD_THRESHOLD)
    results  = []

    for txn in body.transactions:
        try:
            prob = _score_features(artifact, txn.dict())
            results.append({
                "transaction_id": txn.TransactionID,
                "fraud_prob":     round(prob, 4),
                "is_alarm":       prob >= thr,
                "decision":       "FRAUD_ALARM" if prob >= thr else (
                    "REVIEW" if REVIEW_THRESHOLD_LOW <= prob < REVIEW_THRESHOLD_HIGH else "CLEAR"
                ),
            })
        except Exception as exc:
            results.append({"transaction_id": txn.TransactionID, "error": str(exc)})

    return {"results": results, "threshold": thr, "count": len(results)}


@router.get("/model-info")
def get_model_info():
    """Return info about the currently loaded champion model."""
    session = Session()
    try:
        champion = session.query(ModelRun).filter_by(is_champion=True)\
                          .order_by(desc(ModelRun.trained_at)).first()
        if not champion:
            return {"status": "no_model", "message": "Run the pipeline first"}
        return {
            "classifier_name":    champion.classifier_name,
            "imbalance_strategy": champion.imbalance_strategy,
            "auc_roc":            round(champion.auc_roc or 0, 4),
            "pr_auc":             round(champion.pr_auc or 0, 4),
            "f1_fraud":           round(champion.f1_fraud or 0, 4),
            "trained_at":         champion.trained_at.isoformat() if champion.trained_at else None,
            "artifact_path":      champion.artifact_path,
            "threshold":          FRAUD_THRESHOLD,
        }
    finally:
        session.close()
