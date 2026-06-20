"""
/api/analytics — all dashboard metrics endpoints (15+).

Metrics exposed:
  1.  summary              — headline KPIs (total txns, fraud rate, best AUC, etc.)
  2.  fraud-by-channel     — fraud rate per channel
  3.  fraud-by-country     — fraud rate KE vs NG
  4.  amount-distribution  — fraud vs legit amount buckets
  5.  review-queue-depth   — pending / reviewed counts
  6.  imbalance-report     — class imbalance analysis
  7.  daily-fraud-trend    — fraud volume over time buckets
  8.  model-comparison     — all classifiers all metrics table
  9.  feature-importance   — top N features for champion model
  10. shap-summary         — global mean |SHAP| values
  11. score-distribution   — histogram of fraud_prob scores
  12. precision-recall-curve
  13. top-flagged          — highest-scored transactions
  14. calibration          — predicted prob vs actual fraud rate buckets
  15. review-decisions     — analyst confirmed vs overridden
  16. channel-velocity     — transaction count by channel over time
"""

from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from fastapi import APIRouter, Query
from sqlalchemy import func, desc

from pipeline.db import (
    Session, Transaction, ModelRun, Prediction, ReviewQueueItem,
    ReviewDecision, FeatureImportance, ShapExplanation, ImbalanceReport,
)

router = APIRouter()


# ── 1. Summary KPIs ──────────────────────────────────────────────────────────

@router.get("/summary")
def get_summary():
    session = Session()
    try:
        total_train = session.query(Transaction).filter_by(split="train").count()
        total_test  = session.query(Transaction).filter_by(split="test").count()
        total_fraud = session.query(Transaction).filter(
            Transaction.split == "train",
            Transaction.is_fraud == 1,
        ).count()
        fraud_rate = total_fraud / max(total_train, 1)

        # Champion model metrics
        champion = session.query(ModelRun).filter_by(is_champion=True)\
                          .order_by(desc(ModelRun.trained_at)).first()

        # Review queue
        pending_reviews = session.query(ReviewQueueItem).filter_by(status="pending").count()
        completed_reviews = session.query(ReviewQueueItem).filter_by(status="reviewed").count()

        # Total predictions
        total_preds = session.query(Prediction).count()
        alarm_count = session.query(Prediction).filter_by(is_alarm=True).count()

        return {
            "total_transactions": total_train + total_test,
            "total_train":        total_train,
            "total_test":         total_test,
            "total_fraud_train":  total_fraud,
            "fraud_rate":         round(fraud_rate, 6),
            "fraud_pct":          round(fraud_rate * 100, 3),
            "champion_classifier": champion.classifier_name if champion else None,
            "champion_auc_roc":   round(champion.auc_roc or 0, 4) if champion else None,
            "champion_pr_auc":    round(champion.pr_auc or 0, 4) if champion else None,
            "champion_f1":        round(champion.f1_fraud or 0, 4) if champion else None,
            "champion_ks":        round(champion.ks_statistic or 0, 4) if champion else None,
            # ── Key fraud-detection metric: recall (% of frauds caught) ──────
            "champion_recall":    round(champion.recall_fraud or 0, 4) if champion else None,
            "champion_precision": round(champion.precision_fraud or 0, 4) if champion else None,
            "champion_mcc":       round(champion.mcc or 0, 4) if champion else None,
            "champion_accuracy":  round(champion.accuracy or 0, 4) if champion else None,
            "pending_reviews":    pending_reviews,
            "completed_reviews":  completed_reviews,
            "total_predictions":  total_preds,
            "total_alarms":       alarm_count,
        }
    finally:
        session.close()


# ── 2. Fraud by channel ────────────────────────────────────────────────────

@router.get("/fraud-by-channel")
def get_fraud_by_channel():
    session = Session()
    try:
        rows = session.query(
            Transaction.channel,
            func.count(Transaction.transaction_id).label("total"),
            func.sum(Transaction.is_fraud).label("fraud_count"),
        ).filter(Transaction.split == "train", Transaction.channel.isnot(None))\
         .group_by(Transaction.channel).all()

        return [
            {
                "channel":    r.channel,
                "total":      r.total,
                "fraud_count":int(r.fraud_count or 0),
                "fraud_rate": round(int(r.fraud_count or 0) / max(r.total, 1), 4),
            }
            for r in rows
        ]
    finally:
        session.close()


# ── 3. Fraud by country ────────────────────────────────────────────────────

@router.get("/fraud-by-country")
def get_fraud_by_country():
    session = Session()
    try:
        rows = session.query(
            Transaction.country,
            func.count(Transaction.transaction_id).label("total"),
            func.sum(Transaction.is_fraud).label("fraud_count"),
        ).filter(Transaction.split == "train", Transaction.country.isnot(None))\
         .group_by(Transaction.country).all()

        return [
            {
                "country":    r.country,
                "total":      r.total,
                "fraud_count":int(r.fraud_count or 0),
                "fraud_rate": round(int(r.fraud_count or 0) / max(r.total, 1), 4),
            }
            for r in rows
        ]
    finally:
        session.close()


# ── 4. Amount distribution ────────────────────────────────────────────────

@router.get("/amount-distribution")
def get_amount_distribution(bins: int = Query(20, ge=5, le=50)):
    session = Session()
    try:
        rows = session.query(
            Transaction.transaction_amt, Transaction.is_fraud
        ).filter(Transaction.split == "train").all()

        if not rows:
            return {"fraud": [], "legit": []}

        import numpy as np
        fraud_amts = [r.transaction_amt for r in rows if r.is_fraud == 1 and r.transaction_amt]
        legit_amts = [r.transaction_amt for r in rows if r.is_fraud == 0 and r.transaction_amt]

        # Use log scale for better visualisation
        all_amts   = fraud_amts + legit_amts
        log_edges  = np.linspace(np.log1p(min(all_amts)), np.log1p(max(all_amts)), bins + 1)
        edges      = np.expm1(log_edges).tolist()

        fraud_hist, _ = np.histogram(fraud_amts, bins=edges)
        legit_hist, _ = np.histogram(legit_amts, bins=edges)

        labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
        return {
            "labels": labels,
            "fraud":  fraud_hist.tolist(),
            "legit":  legit_hist.tolist(),
        }
    finally:
        session.close()


# ── 5. Review queue depth ────────────────────────────────────────────────

@router.get("/review-queue-depth")
def get_review_queue_depth():
    session = Session()
    try:
        pending   = session.query(ReviewQueueItem).filter_by(status="pending").count()
        reviewed  = session.query(ReviewQueueItem).filter_by(status="reviewed").count()
        skipped   = session.query(ReviewQueueItem).filter_by(status="skipped").count()
        confirmed = session.query(ReviewDecision).filter_by(decision="confirmed_fraud").count()
        overridden= session.query(ReviewDecision).filter_by(decision="confirmed_legit").count()
        return {
            "pending": pending, "reviewed": reviewed, "skipped": skipped,
            "confirmed_fraud": confirmed, "overridden_to_legit": overridden,
        }
    finally:
        session.close()


# ── 6. Imbalance report ────────────────────────────────────────────────────

@router.get("/imbalance-report")
def get_imbalance_report():
    session = Session()
    try:
        report = session.query(ImbalanceReport)\
                        .order_by(desc(ImbalanceReport.computed_at)).first()
        if not report:
            return {"error": "No imbalance report available — run the pipeline first"}
        return {
            "fraud_count":    report.fraud_count,
            "legit_count":    report.legit_count,
            "imbalance_ratio":round(report.imbalance_ratio, 2),
            "strategy_used":  report.strategy_used,
            "strategy_comparison": json.loads(report.strategy_comparison or "{}"),
        }
    finally:
        session.close()


# ── 7. Daily fraud trend ───────────────────────────────────────────────────

@router.get("/daily-fraud-trend")
def get_daily_fraud_trend():
    """
    Returns fraud/legit counts bucketed by TransactionDT quantiles (simulated days).
    """
    session = Session()
    try:
        rows = session.query(
            Transaction.transaction_dt,
            Transaction.is_fraud,
        ).filter(Transaction.split == "train").all()

        if not rows:
            return {"buckets": [], "fraud": [], "legit": []}

        import numpy as np
        dts    = np.array([r.transaction_dt for r in rows])
        labels = np.array([r.is_fraud for r in rows])

        # Bucket into ~30 time bins
        n_bins   = 30
        edges    = np.linspace(dts.min(), dts.max(), n_bins + 1)
        bin_ids  = np.digitize(dts, edges, right=True).clip(1, n_bins)

        fraud_counts = np.zeros(n_bins, dtype=int)
        legit_counts = np.zeros(n_bins, dtype=int)
        for bid, lbl in zip(bin_ids, labels):
            if lbl == 1:
                fraud_counts[bid - 1] += 1
            else:
                legit_counts[bid - 1] += 1

        bucket_labels = [f"T{i+1}" for i in range(n_bins)]
        return {
            "buckets": bucket_labels,
            "fraud":   fraud_counts.tolist(),
            "legit":   legit_counts.tolist(),
        }
    finally:
        session.close()


# ── 8. Model comparison ───────────────────────────────────────────────────

@router.get("/model-comparison")
def get_model_comparison():
    session = Session()
    try:
        runs = session.query(ModelRun).order_by(desc(ModelRun.trained_at)).limit(50).all()
        return [
            {
                "id":                 r.id,
                "run_id":             r.run_id,
                "classifier_name":    r.classifier_name,
                "imbalance_strategy": r.imbalance_strategy,
                "is_champion":        r.is_champion,
                "accuracy":           round(r.accuracy or 0, 4),
                "auc_roc":            round(r.auc_roc or 0, 4),
                "pr_auc":             round(r.pr_auc or 0, 4),
                "f1_fraud":           round(r.f1_fraud or 0, 4),
                "precision_fraud":    round(r.precision_fraud or 0, 4),
                "recall_fraud":       round(r.recall_fraud or 0, 4),
                "mcc":                round(r.mcc or 0, 4),
                "ks_statistic":       round(r.ks_statistic or 0, 4),
                "training_duration_s":round(r.training_duration_s or 0, 1),
                "trained_at":         r.trained_at.isoformat() if r.trained_at else None,
            }
            for r in runs
        ]
    finally:
        session.close()


# ── 9. Feature importance ─────────────────────────────────────────────────

@router.get("/feature-importance")
def get_feature_importance(top_n: int = Query(20, ge=5, le=50)):
    session = Session()
    try:
        champion = session.query(ModelRun).filter_by(is_champion=True)\
                          .order_by(desc(ModelRun.trained_at)).first()
        if not champion:
            return []

        fi = session.query(FeatureImportance)\
                    .filter_by(model_run_id=champion.id)\
                    .order_by(FeatureImportance.rank).limit(top_n).all()
        return [
            {"feature": f.feature_name, "importance": round(f.importance, 6), "rank": f.rank}
            for f in fi
        ]
    finally:
        session.close()


# ── 10. SHAP summary ──────────────────────────────────────────────────────

@router.get("/shap-summary")
def get_shap_summary(top_n: int = Query(20, ge=5, le=50)):
    """Retrieve pre-computed global SHAP values from the latest pipeline run."""
    session = Session()
    try:
        champion = session.query(ModelRun).filter_by(is_champion=True)\
                          .order_by(desc(ModelRun.trained_at)).first()
        if not champion:
            return []

        # SHAP stored in ShapExplanation table with transaction_id=NULL for global
        rows = session.query(ShapExplanation)\
                      .filter_by(model_run_id=champion.id, transaction_id=None)\
                      .first()
        if not rows:
            return []

        vals = json.loads(rows.shap_values_json or "{}")
        sorted_vals = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": f, "mean_abs_shap": round(abs(v), 6), "rank": i + 1}
                for i, (f, v) in enumerate(sorted_vals[:top_n])]
    finally:
        session.close()


# ── 11. Score distribution ────────────────────────────────────────────────

@router.get("/score-distribution")
def get_score_distribution(bins: int = Query(20, ge=10, le=50)):
    """
    Score distribution histogram.
    If predictions exist: use actual model scores.
    If no predictions yet (pipeline just started): show training label distribution
    as a proxy so the chart isn't empty.
    """
    session = Session()
    try:
        rows = session.query(Prediction.fraud_prob).limit(10000).all()

        if rows:
            import numpy as np
            probs = np.array([r.fraud_prob for r in rows])
            counts, edges = np.histogram(probs, bins=bins, range=(0, 1))
            bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
            return {"bins": bin_labels, "counts": counts.tolist(), "source": "predictions"}

        # Fallback: use isFraud label distribution from transactions
        # Show fraud vs legit as a binary distribution indicator
        fraud_count = session.query(Transaction).filter(
            Transaction.is_fraud == 1, Transaction.split == "train"
        ).count()
        legit_count = session.query(Transaction).filter(
            Transaction.is_fraud == 0, Transaction.split == "train"
        ).count()

        if fraud_count + legit_count == 0:
            return {"bins": [], "counts": [], "source": "empty"}

        # Simulate a rough score distribution: fraud near 1, legit near 0
        import numpy as np
        np.random.seed(42)
        fraud_scores = np.clip(np.random.beta(2, 1, min(fraud_count, 2000)), 0, 1)
        legit_scores = np.clip(np.random.beta(1, 5, min(legit_count // 5, 2000)), 0, 1)
        all_scores   = np.concatenate([fraud_scores, legit_scores])
        counts, edges = np.histogram(all_scores, bins=bins, range=(0, 1))
        bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
        return {"bins": bin_labels, "counts": counts.tolist(), "source": "simulated_pre_pipeline"}
    finally:
        session.close()


# ── 12. Calibration ───────────────────────────────────────────────────────

@router.get("/calibration")
def get_calibration(n_bins: int = Query(10, ge=5, le=20)):
    """Compare mean predicted probability vs actual fraud rate per prob bucket."""
    session = Session()
    try:
        rows = session.query(
            Prediction.fraud_prob,
            Transaction.is_fraud,
        ).join(Transaction, Prediction.transaction_id == Transaction.transaction_id)\
         .filter(Transaction.split == "train").limit(50000).all()

        if not rows:
            return {"buckets": [], "mean_predicted": [], "actual_rate": []}

        import numpy as np
        probs  = np.array([r.fraud_prob for r in rows])
        actual = np.array([r.is_fraud   for r in rows])
        edges  = np.linspace(0, 1, n_bins + 1)

        mean_pred, actual_rate, labels = [], [], []
        for i in range(n_bins):
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
            if mask.sum() == 0:
                continue
            mean_pred.append(float(probs[mask].mean()))
            actual_rate.append(float(actual[mask].mean()))
            labels.append(f"{edges[i]:.1f}–{edges[i+1]:.1f}")

        return {"buckets": labels, "mean_predicted": mean_pred, "actual_rate": actual_rate}
    finally:
        session.close()


# ── 13. Top flagged transactions ──────────────────────────────────────────

@router.get("/top-flagged")
def get_top_flagged(limit: int = Query(20, ge=5, le=100)):
    session = Session()
    try:
        rows = session.query(
            Prediction.transaction_id,
            Prediction.fraud_prob,
            Prediction.is_alarm,
            Transaction.transaction_amt,
            Transaction.channel,
            Transaction.country,
            Transaction.currency,
            Transaction.card_type,
            Transaction.is_fraud,
        ).join(Transaction, Prediction.transaction_id == Transaction.transaction_id)\
         .order_by(desc(Prediction.fraud_prob)).limit(limit).all()

        return [
            {
                "transaction_id":  r.transaction_id,
                "fraud_prob":      round(r.fraud_prob, 4),
                "is_alarm":        r.is_alarm,
                "amount":          r.transaction_amt,
                "channel":         r.channel,
                "country":         r.country,
                "currency":        r.currency,
                "card_type":       r.card_type,
                "actual_fraud":    r.is_fraud,
            }
            for r in rows
        ]
    finally:
        session.close()


# ── 14. Review decisions summary ─────────────────────────────────────────

@router.get("/review-decisions")
def get_review_decisions():
    session = Session()
    try:
        rows = session.query(
            ReviewDecision.decision,
            func.count(ReviewDecision.id).label("count"),
        ).group_by(ReviewDecision.decision).all()
        return {r.decision: r.count for r in rows}
    finally:
        session.close()


# ── 15. Channel velocity ──────────────────────────────────────────────────

@router.get("/channel-velocity")
def get_channel_velocity():
    """Transaction counts per channel per time bucket (30 buckets)."""
    session = Session()
    try:
        rows = session.query(
            Transaction.transaction_dt,
            Transaction.channel,
        ).filter(Transaction.split == "train", Transaction.channel.isnot(None)).all()

        if not rows:
            return {"buckets": [], "series": {}}

        import numpy as np
        dts      = np.array([r.transaction_dt for r in rows])
        channels = [r.channel for r in rows]
        n_bins   = 30
        edges    = np.linspace(dts.min(), dts.max(), n_bins + 1)
        bin_ids  = np.digitize(dts, edges, right=True).clip(1, n_bins)
        buckets  = [f"T{i+1}" for i in range(n_bins)]
        unique_channels = list(set(channels))
        series = {ch: [0] * n_bins for ch in unique_channels}
        for bid, ch in zip(bin_ids, channels):
            series[ch][bid - 1] += 1

        return {"buckets": buckets, "series": series}
    finally:
        session.close()


# ── 16. Precision-recall curve data ──────────────────────────────────────

@router.get("/precision-recall-curve")
def get_pr_curve():
    session = Session()
    try:
        rows = session.query(
            Prediction.fraud_prob,
            Transaction.is_fraud,
        ).join(Transaction, Prediction.transaction_id == Transaction.transaction_id)\
         .filter(Transaction.split == "train").limit(50000).all()

        if not rows:
            return {"precision": [], "recall": [], "thresholds": []}

        import numpy as np
        from sklearn.metrics import precision_recall_curve

        probs  = np.array([r.fraud_prob for r in rows])
        actual = np.array([r.is_fraud   for r in rows])

        precision, recall, thresholds = precision_recall_curve(actual, probs)
        # Downsample to 100 points for the frontend
        step = max(1, len(precision) // 100)
        return {
            "precision":   precision[::step].tolist(),
            "recall":      recall[::step].tolist(),
            "thresholds":  thresholds[::step].tolist() if len(thresholds) > 0 else [],
        }
    finally:
        session.close()


# ── 3D KPI Endpoints ──────────────────────────────────────────────────────────

@router.get("/3d/risk-landscape")
def get_3d_risk_landscape(max_points: int = Query(800, ge=100, le=2000)):
    """
    3D Scatter — Fraud Risk Landscape.
    Returns sampled transactions as {x: log_amt, y: hour, z: fraud_prob, channel, is_fraud}
    Used to render the 3D scatter: Amount × Hour-of-Day × Fraud Probability.
    """
    session = Session()
    try:
        from sqlalchemy import desc

        # Join predictions with transactions to get scored rows
        rows = session.query(
            Transaction.transaction_amt,
            Transaction.transaction_dt,
            Transaction.channel,
            Transaction.currency,
            Transaction.is_fraud,
            Prediction.fraud_prob,
        ).join(Prediction, Prediction.transaction_id == Transaction.transaction_id)\
         .filter(Transaction.transaction_amt.isnot(None))\
         .order_by(desc(Prediction.fraud_prob))\
         .limit(max_points).all()

        if not rows:
            # Fallback: return transaction data without predictions
            rows2 = session.query(
                Transaction.transaction_amt,
                Transaction.transaction_dt,
                Transaction.channel,
                Transaction.currency,
                Transaction.is_fraud,
            ).filter(
                Transaction.split == "train",
                Transaction.transaction_amt.isnot(None),
            ).limit(max_points).all()

            import random
            random.seed(42)
            result = []
            for r in rows2:
                amt   = float(r.transaction_amt or 0)
                dt    = int(r.transaction_dt or 0)
                hour  = (dt // 3600) % 24
                # Simulate a plausible probability for display purposes
                base  = 0.05 if r.is_fraud == 0 else 0.75
                prob  = max(0.0, min(1.0, base + random.gauss(0, 0.15)))
                result.append({
                    "x": round(float(np.log1p(amt)), 4),
                    "y": hour,
                    "z": round(prob, 4),
                    "amount": round(amt, 2),
                    "channel": r.channel or "unknown",
                    "currency": r.currency or "KES",
                    "is_fraud": r.is_fraud,
                })
            return result

        result = []
        for r in rows:
            amt  = float(r.transaction_amt or 0)
            dt   = int(r.transaction_dt or 0)
            hour = (dt // 3600) % 24
            result.append({
                "x": round(float(np.log1p(amt)), 4),
                "y": hour,
                "z": round(float(r.fraud_prob), 4),
                "amount":   round(amt, 2),
                "channel":  r.channel or "unknown",
                "currency": r.currency or "KES",
                "is_fraud": r.is_fraud,
            })
        return result
    finally:
        session.close()


@router.get("/3d/model-performance-cube")
def get_3d_model_performance_cube():
    """
    3D Bar — Model Performance Cube.
    Returns {classifier, metric, value} for all trained models × 6 key metrics.
    Frontend renders as 3D bar chart (classifier × metric → height = score).
    """
    session = Session()
    try:
        from sqlalchemy import desc
        runs = session.query(ModelRun).order_by(desc(ModelRun.trained_at)).limit(20).all()
        if not runs:
            return {"classifiers": [], "metrics": [], "values": []}

        metric_keys = [
            ("AUC-ROC",   "auc_roc"),
            ("PR-AUC",    "pr_auc"),
            ("F1-Fraud",  "f1_fraud"),
            ("Precision", "precision_fraud"),
            ("Recall",    "recall_fraud"),
            ("KS-Stat",   "ks_statistic"),
        ]

        classifiers = list({r.classifier_name for r in runs})
        # Keep best run per classifier
        best: dict = {}
        for r in runs:
            if r.classifier_name not in best or (r.pr_auc or 0) > (best[r.classifier_name].pr_auc or 0):
                best[r.classifier_name] = r

        values = []
        for clf_idx, clf in enumerate(classifiers):
            run = best[clf]
            for met_idx, (met_label, met_key) in enumerate(metric_keys):
                val = float(getattr(run, met_key, 0) or 0)
                values.append({
                    "classifier_idx": clf_idx,
                    "metric_idx":     met_idx,
                    "classifier":     clf,
                    "metric":         met_label,
                    "value":          round(val, 4),
                    "is_champion":    run.is_champion,
                })

        return {
            "classifiers": classifiers,
            "metrics":     [m[0] for m in metric_keys],
            "values":      values,
        }
    finally:
        session.close()


@router.get("/3d/prt-surface")
def get_3d_prt_surface():
    """
    3D Surface — Precision-Recall-Threshold surface.
    Computes precision, recall, and F1 at 30 threshold levels from predictions on train split.
    Returns a grid for 3D surface rendering: threshold × metric.
    """
    session = Session()
    try:
        rows = session.query(
            Prediction.fraud_prob,
            Transaction.is_fraud,
        ).join(Transaction, Prediction.transaction_id == Transaction.transaction_id)\
         .filter(Transaction.split == "train").limit(50000).all()

        if not rows:
            return {"thresholds": [], "surface": []}

        import numpy as np
        probs  = np.array([r.fraud_prob for r in rows])
        actual = np.array([r.is_fraud   for r in rows], dtype=int)

        thresholds = np.linspace(0.05, 0.95, 30)
        surface = []

        for thr in thresholds:
            pred      = (probs >= thr).astype(int)
            tp        = int(((pred == 1) & (actual == 1)).sum())
            fp        = int(((pred == 1) & (actual == 0)).sum())
            fn        = int(((pred == 0) & (actual == 1)).sum())
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            alarm_rate= float(pred.mean())
            surface.append({
                "threshold":  round(float(thr), 3),
                "precision":  round(precision, 4),
                "recall":     round(recall, 4),
                "f1":         round(f1, 4),
                "alarm_rate": round(alarm_rate, 4),
                "tp": tp, "fp": fp, "fn": fn,
            })

        return {
            "thresholds": [round(float(t), 3) for t in thresholds],
            "surface":    surface,
        }
    finally:
        session.close()


# ── Heatmap, Polar, Boxplot endpoints ────────────────────────────────────────

@router.get("/heatmap/fraud-risk")
def get_fraud_risk_heatmap():
    """
    Fraud Risk Heatmap — Hour of Day × Channel.
    Returns a matrix of fraud rates: rows=channels, cols=hours(0-23).
    Cell value = fraud_count / total_count for that (channel, hour) bucket.
    Empty buckets return 0.
    """
    session = Session()
    try:
        rows = session.query(
            Transaction.channel,
            Transaction.transaction_dt,
            Transaction.is_fraud,
        ).filter(
            Transaction.split == "train",
            Transaction.channel.isnot(None),
            Transaction.is_fraud.isnot(None),
        ).all()

        if not rows:
            return {"channels": [], "hours": list(range(24)), "matrix": [], "max_rate": 0}

        from collections import defaultdict
        import numpy as np

        # Aggregate (channel, hour) → [total, fraud]
        buckets: dict = defaultdict(lambda: [0, 0])
        channels_set = set()
        for r in rows:
            ch   = r.channel
            hour = ((r.transaction_dt or 0) // 3600) % 24
            channels_set.add(ch)
            buckets[(ch, hour)][0] += 1
            if r.is_fraud == 1:
                buckets[(ch, hour)][1] += 1

        channels = sorted(channels_set)
        hours    = list(range(24))

        # Build matrix: list of [hour, channel_idx, fraud_rate, total, fraud_count]
        matrix = []
        max_rate = 0.0
        for ci, ch in enumerate(channels):
            for h in hours:
                total, fraud = buckets.get((ch, h), [0, 0])
                rate = round(fraud / max(total, 1), 4)
                max_rate = max(max_rate, rate)
                matrix.append({
                    "channel":     ch,
                    "channel_idx": ci,
                    "hour":        h,
                    "fraud_rate":  rate,
                    "total":       total,
                    "fraud_count": fraud,
                })

        return {
            "channels": channels,
            "hours":    hours,
            "matrix":   matrix,
            "max_rate": round(max_rate, 4),
        }
    finally:
        session.close()


@router.get("/polar/fraud-by-dow")
def get_fraud_by_dow():
    """
    Polar Plot — Fraud count by Day of Week × Country.
    Returns counts for Mon-Sun split by KE and NG.
    """
    session = Session()
    try:
        rows = session.query(
            Transaction.transaction_dt,
            Transaction.country,
            Transaction.is_fraud,
        ).filter(
            Transaction.split == "train",
            Transaction.is_fraud == 1,
            Transaction.country.isnot(None),
        ).all()

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ke_counts = [0] * 7
        ng_counts = [0] * 7

        for r in rows:
            dow = ((r.transaction_dt or 0) // 86400) % 7
            if r.country == "KE":
                ke_counts[dow] += 1
            else:
                ng_counts[dow] += 1

        # Also get legit counts for context
        legit_rows = session.query(
            Transaction.transaction_dt,
            Transaction.country,
        ).filter(
            Transaction.split == "train",
            Transaction.is_fraud == 0,
            Transaction.country.isnot(None),
        ).all()

        legit_counts = [0] * 7
        for r in legit_rows:
            dow = ((r.transaction_dt or 0) // 86400) % 7
            legit_counts[dow] += 1

        return {
            "days":         days,
            "ke_fraud":     ke_counts,
            "ng_fraud":     ng_counts,
            "legit_counts": legit_counts,
        }
    finally:
        session.close()


@router.get("/boxplot/amount-by-channel")
def get_amount_boxplot():
    """
    Boxplot — Transaction Amount Distribution by Channel × Fraud/Legit.
    Returns min, Q1, median, Q3, max, and outlier points per (channel, fraud) group.
    """
    session = Session()
    try:
        import numpy as np

        rows = session.query(
            Transaction.channel,
            Transaction.transaction_amt,
            Transaction.currency,
            Transaction.is_fraud,
        ).filter(
            Transaction.split == "train",
            Transaction.channel.isnot(None),
            Transaction.transaction_amt.isnot(None),
            Transaction.is_fraud.isnot(None),
        ).all()

        if not rows:
            return {"channels": [], "fraud": [], "legit": []}

        from collections import defaultdict

        # Normalise amounts to USD
        _KES = 1 / 128.0
        _NGN = 1 / 1570.0

        groups: dict = defaultdict(list)
        for r in rows:
            usd = r.transaction_amt * (_KES if r.currency == "KES" else _NGN)
            key = (r.channel, int(r.is_fraud))
            groups[key].append(usd)

        channels = sorted({r.channel for r in rows})

        def _boxplot(vals):
            if not vals:
                return None
            a = np.array(vals)
            q1, median, q3 = np.percentile(a, [25, 50, 75])
            iqr    = q3 - q1
            lo     = q1 - 1.5 * iqr
            hi     = q3 + 1.5 * iqr
            inliers = a[(a >= lo) & (a <= hi)]
            outlier_sample = sorted(a[a > hi].tolist())[:20]  # cap outliers
            return {
                "min":      round(float(inliers.min()) if len(inliers) else float(a.min()), 2),
                "q1":       round(float(q1), 2),
                "median":   round(float(median), 2),
                "q3":       round(float(q3), 2),
                "max":      round(float(inliers.max()) if len(inliers) else float(a.max()), 2),
                "outliers": [round(v, 2) for v in outlier_sample],
                "count":    len(vals),
                "mean":     round(float(a.mean()), 2),
            }

        fraud_boxes = []
        legit_boxes = []
        for ch in channels:
            fraud_boxes.append({"channel": ch, "stats": _boxplot(groups.get((ch, 1), []))})
            legit_boxes.append({"channel": ch, "stats": _boxplot(groups.get((ch, 0), []))})

        return {
            "channels":   channels,
            "fraud":      fraud_boxes,
            "legit":      legit_boxes,
        }
    finally:
        session.close()
