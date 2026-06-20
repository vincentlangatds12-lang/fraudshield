"""
/api/explainability — SHAP · LIME · Feature Importance

Always returns data when the pipeline has run:
- feature-importance → DB FeatureImportance table (50 rows guaranteed after pipeline)
- shap/global        → DB ShapExplanation → falls back to FeatureImportance
- shap/transaction   → DB ShapExplanation per transaction
- lime/transaction   → DB ShapExplanation per transaction
"""
from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc

from pipeline.db import Session, ModelRun, Transaction, ShapExplanation, FeatureImportance

router = APIRouter()


def _get_champion(session) -> ModelRun | None:
    return session.query(ModelRun).filter_by(is_champion=True)\
                  .order_by(desc(ModelRun.trained_at)).first()


@router.get("/feature-importance")
def get_feature_importance(top_n: int = Query(20, ge=5, le=50)):
    """
    Feature importance for the champion model.
    Source: DB FeatureImportance table (populated after every pipeline run).
    Falls back to any available model if champion has none.
    """
    session = Session()
    try:
        champion = _get_champion(session)

        # Try champion's FI first
        if champion:
            rows = session.query(FeatureImportance)\
                          .filter_by(model_run_id=champion.id)\
                          .order_by(FeatureImportance.rank).limit(top_n).all()
            if rows:
                total = sum(r.importance for r in rows) or 1e-9
                return [{"rank": r.rank, "feature": r.feature_name,
                         "importance": round(r.importance, 6),
                         "pct": round(r.importance / total * 100, 2)}
                        for r in rows]

        # Fallback — any model's FI (sorted by importance desc)
        any_rows = session.query(FeatureImportance)\
                          .order_by(FeatureImportance.importance.desc())\
                          .limit(top_n).all()
        if any_rows:
            total = sum(r.importance for r in any_rows) or 1e-9
            return [{"rank": i + 1, "feature": r.feature_name,
                     "importance": round(r.importance, 6),
                     "pct": round(r.importance / total * 100, 2)}
                    for i, r in enumerate(any_rows)]

        return []
    finally:
        session.close()


@router.get("/shap/global")
def get_shap_global(top_n: int = Query(20, ge=5, le=50)):
    """
    Global SHAP summary (mean |SHAP|).
    Falls back to FeatureImportance if SHAP not computed.
    """
    session = Session()
    try:
        champion = _get_champion(session)
        if not champion:
            return []

        # Check pre-computed SHAP
        stored = session.query(ShapExplanation)\
                        .filter_by(model_run_id=champion.id)\
                        .filter(ShapExplanation.transaction_id.is_(None)).first()
        if stored:
            vals = json.loads(stored.shap_values_json or "{}")
            sorted_vals = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            return [{"rank": i + 1, "feature": f, "mean_abs_shap": round(abs(v), 6)}
                    for i, (f, v) in enumerate(sorted_vals)]

        # Fallback → FeatureImportance as proxy
        rows = session.query(FeatureImportance)\
                      .filter_by(model_run_id=champion.id)\
                      .order_by(FeatureImportance.rank).limit(top_n).all()
        if rows:
            return [{"rank": r.rank, "feature": r.feature_name,
                     "mean_abs_shap": round(r.importance, 6)}
                    for r in rows]

        # Any model
        any_rows = session.query(FeatureImportance)\
                          .order_by(FeatureImportance.importance.desc())\
                          .limit(top_n).all()
        return [{"rank": i + 1, "feature": r.feature_name,
                 "mean_abs_shap": round(r.importance, 6)}
                for i, r in enumerate(any_rows)]
    finally:
        session.close()


@router.get("/shap/transaction/{transaction_id}")
def get_shap_local(transaction_id: int):
    session = Session()
    try:
        txn = session.get(Transaction, transaction_id)
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")
        stored = session.query(ShapExplanation)\
                        .filter_by(transaction_id=transaction_id)\
                        .order_by(desc(ShapExplanation.computed_at)).first()
        if stored:
            return {
                "transaction_id": transaction_id,
                "base_value":     stored.base_value,
                "predicted_prob": stored.predicted_prob,
                "shap_values":    json.loads(stored.shap_values_json or "{}"),
            }
        return {"status": "not_available",
                "message": "SHAP not computed for this transaction — run pipeline to generate explanations"}
    finally:
        session.close()


@router.get("/lime/transaction/{transaction_id}")
def get_lime_local(transaction_id: int):
    session = Session()
    try:
        stored = session.query(ShapExplanation)\
                        .filter_by(transaction_id=transaction_id)\
                        .order_by(desc(ShapExplanation.computed_at)).first()
        if stored:
            vals = json.loads(stored.shap_values_json or "{}")
            if vals:
                sorted_vals = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
                return {
                    "transaction_id": transaction_id,
                    "predicted_prob": stored.predicted_prob,
                    "explanation": [{"feature": f, "weight": round(v, 6), "positive": v > 0}
                                    for f, v in sorted_vals],
                }
        return {"status": "not_available",
                "message": "LIME not computed for this transaction"}
    finally:
        session.close()


@router.get("/summary")
def get_explain_summary():
    session = Session()
    try:
        champion   = _get_champion(session)
        shap_count = session.query(ShapExplanation).count()
        fi_count   = session.query(FeatureImportance).count()
        global_shap = session.query(ShapExplanation)\
                             .filter(ShapExplanation.transaction_id.is_(None)).count()
        return {
            "champion_classifier":    champion.classifier_name if champion else None,
            "champion_strategy":      champion.imbalance_strategy if champion else None,
            "shap_explanations_count":shap_count,
            "global_shap_available":  global_shap > 0,
            "feature_importance_rows":fi_count,
            "status":                 "ready" if champion else "no_model",
            "fallback_active":        global_shap == 0 and fi_count > 0,
        }
    finally:
        session.close()
