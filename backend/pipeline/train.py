"""
Model training — 5 models × 2 strategies = up to 10 runs.

Threshold strategy:
  For every model, scan thresholds 0→1 at step 0.02.
  Pick the threshold that maximises F1 on the val set.
  All metrics (including recall) are reported at that threshold.
  This gives a consistent, operationally meaningful operating point.

Champion composite:
  recall_at_f1_optimal(50%) + PR-AUC(20%) + AUC-ROC(20%) + F1_optimal(10%)
"""

from __future__ import annotations

import os, sys, time, uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_DIR, RANDOM_STATE, CV_FOLDS
from pipeline.imbalance import apply_strategy

_FLAML_OK = False
try:
    from flaml import AutoML; _FLAML_OK = True
except ImportError:
    print("[TRAIN] FLAML not installed")

_LGB_OK = False
try:
    import lightgbm; _LGB_OK = True
except ImportError:
    print("[TRAIN] LightGBM not installed")

_XGB_OK = False
try:
    import xgboost; _XGB_OK = True
except ImportError:
    print("[TRAIN] XGBoost not installed")

_CB_OK = False
try:
    from catboost import CatBoostClassifier; _CB_OK = True
except ImportError:
    pass

_MODEL_BUDGET = 300   # FLAML budget per model per strategy (seconds)


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute all evaluation metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    fraud  = y_prob[y_true == 1]
    legit  = y_prob[y_true == 0]
    ks     = ks_2samp(fraud, legit).statistic if len(fraud) > 0 else 0.0
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "auc_roc":         float(roc_auc_score(y_true, y_prob)),
        "pr_auc":          float(average_precision_score(y_true, y_prob)),
        "f1_fraud":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_fraud": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fraud":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc":             float(matthews_corrcoef(y_true, y_pred)),
        "ks_statistic":    float(ks),
        "avg_precision":   float(average_precision_score(y_true, y_prob)),
    }


def f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    """
    Scan thresholds 0→1 at step 0.02.
    Return (optimal_threshold, f1_at_threshold, recall_at_threshold).
    Picks the threshold that maximises F1 on the fraud class.
    """
    best_f1, best_thr, best_recall = 0.0, 0.5, 0.0
    for thr in np.arange(0.02, 1.00, 0.02):
        thr  = round(float(thr), 2)
        pred = (y_prob >= thr).astype(int)
        f1   = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
        if f1 > best_f1:
            best_f1     = f1
            best_thr    = thr
            best_recall = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    return best_thr, round(best_f1, 4), round(best_recall, 4)


def _champion_score(result: dict) -> float:
    """
    Champion composite:
      recall_at_f1_optimal(50%) + PR-AUC(20%) + AUC-ROC(20%) + F1_optimal(10%)
    """
    m     = result.get("metrics", {})
    rec   = result.get("max_recall") or m.get("recall_fraud", 0) or 0
    f1    = m.get("f1_fraud", 0) or 0
    return (
        0.50 * rec +
        0.20 * (m.get("pr_auc",  0) or 0) +
        0.20 * (m.get("auc_roc", 0) or 0) +
        0.10 * f1
    )


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save(artifact: dict, models_dir: Path, run_id: str, name: str) -> str:
    path = str(models_dir / f"{run_id}_{name}.pkl")
    joblib.dump(artifact, path)
    return path


def _fi(clf, feature_names: list[str]) -> list[dict]:
    try:
        inner = clf
        if hasattr(clf, "model") and hasattr(clf.model, "estimator"):
            inner = clf.model.estimator
        fi = (inner.feature_importances_ if hasattr(inner, "feature_importances_")
              else np.abs(inner.coef_[0]) if hasattr(inner, "coef_") else None)
        if fi is None:
            return []
        pairs = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": float(v), "rank": i + 1}
                for i, (f, v) in enumerate(pairs[:50])]
    except Exception:
        return []


def _make_result(name, run_id, strategy, metrics, path, hyperparams,
                 duration, thr, fi, max_recall, max_recall_thr) -> dict:
    return {
        "run_id":               run_id,
        "classifier_name":      name,
        "imbalance_strategy":   strategy,
        "metrics":              metrics,
        "artifact_path":        path,
        "hyperparams":          hyperparams,
        "training_duration_s":  duration,
        "threshold":            thr,
        "feature_importance":   fi,
        "is_champion":          False,
        "max_recall":           max_recall,       # recall at F1-optimal threshold
        "max_recall_threshold": max_recall_thr,   # F1-optimal threshold
    }


def _persist_one(result: dict, run_id: str) -> None:
    """Save to DB immediately + update champion flag in real-time."""
    try:
        from pipeline.run_pipeline import persist_one, update_champion_in_db
        persist_one(result, run_id)
        update_champion_in_db(run_id)
    except Exception as e:
        print(f"[TRAIN] persist_one warning: {e}")


def _post_train(clf, scaler, X_val, y_val, name, run_id, strategy,
                feature_names, models_dir, artifact_name, hyperparams, duration) -> dict:
    """
    Post-training routine shared by all models:
     1. Score on val set
     2. Find F1-optimal threshold (scan 0→1, step 0.02)
     3. Compute all metrics at that threshold (recall is thus at F1-optimal thr)
     4. Save artifact + result to DB
    """
    X_v    = scaler.transform(X_val) if scaler is not None else X_val
    y_prob = clf.predict_proba(X_v)[:, 1]

    opt_thr, opt_f1, opt_recall = f1_optimal_threshold(y_val, y_prob)
    mets = compute_metrics(y_val, y_prob, threshold=opt_thr)

    print(f"  thr={opt_thr:.2f} (F1-optimal) → "
          f"AUC={mets['auc_roc']:.4f}  Recall={opt_recall:.4f}  "
          f"F1={opt_f1:.4f}  PR-AUC={mets['pr_auc']:.4f}  ({duration:.0f}s)")

    path = _save({"model": clf, "scaler": scaler, "feature_names": feature_names,
                  "threshold": opt_thr, "trained_at": pd.Timestamp.utcnow().isoformat(),
                  "classifier_name": name, "run_id": run_id},
                 models_dir, run_id, artifact_name)

    r = _make_result(name, run_id, strategy, mets, path, hyperparams,
                     duration, opt_thr, _fi(clf, feature_names), opt_recall, opt_thr)
    _persist_one(r, run_id)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_all_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_names: list[str],
    imbalance_strategy: str  = "class_weight",  # kept for API compat
    imbalance_ratio:    float = 10.0,
    run_id:             str | None = None,
) -> list[dict]:

    run_id     = run_id or str(uuid.uuid4())
    models_dir = Path(MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    results    = []
    strategies = ["class_weight", "adasyn"]

    # ── 1. Logistic Regression (sklearn, both strategies) ─────────────────────
    for strategy in strategies:
        print(f"\n[TRAIN] ── logistic_regression [{strategy}] ──")
        t0 = time.time()
        try:
            X_s, y_s = apply_strategy(X_train, y_train, strategy, RANDOM_STATE)
            scaler   = StandardScaler()
            X_tr_sc  = scaler.fit_transform(X_s)
            lr = LogisticRegression(
                max_iter=2000, solver="saga", C=0.1,
                class_weight="balanced", random_state=RANDOM_STATE, tol=1e-3, n_jobs=-1,
            )
            lr.fit(X_tr_sc, y_s)
            duration = time.time() - t0
            r = _post_train(lr, scaler, X_val, y_val,
                            "logistic_regression", run_id, strategy,
                            feature_names, models_dir,
                            f"logistic_regression_{strategy}",
                            {"C": 0.1, "solver": "saga"}, duration)
            results.append(r)
        except Exception as e:
            print(f"  LR/{strategy} ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── 2–5. FLAML-tuned models (both strategies) ─────────────────────────────
    flaml_models: list[tuple] = []
    if _LGB_OK: flaml_models.append(("lightgbm",     ["lgbm"]))
    if _CB_OK:  flaml_models.append(("catboost",     ["catboost"]))
    flaml_models.append(           ("random_forest", ["rf"]))
    if _XGB_OK: flaml_models.append(("xgboost",      ["xgboost"]))

    for clf_name, estimators in flaml_models:
        if not _FLAML_OK:
            print(f"  [SKIP] {clf_name} — FLAML not installed"); continue
        for strategy in strategies:
            print(f"\n[TRAIN] ── {clf_name} [{strategy}] (FLAML {_MODEL_BUDGET}s, metric=ap) ──")
            t0 = time.time()
            try:
                X_s, y_s = apply_strategy(X_train, y_train, strategy, RANDOM_STATE)
                automl   = AutoML()
                automl.fit(
                    X_s, y_s,
                    time_budget    = _MODEL_BUDGET,
                    metric         = "ap",          # average_precision — built-in
                    task           = "classification",
                    seed           = RANDOM_STATE,
                    eval_method    = "cv",
                    n_splits       = 3,
                    verbose        = 0,
                    estimator_list = estimators,
                )
                duration = time.time() - t0
                print(f"  Best estimator: {automl.best_estimator}")
                r = _post_train(automl, None, X_val, y_val,
                                clf_name, run_id, strategy,
                                feature_names, models_dir,
                                f"{clf_name}_{strategy}",
                                {"best_estimator": automl.best_estimator,
                                 "best_config":    automl.best_config,
                                 "budget_s":       _MODEL_BUDGET,
                                 "metric":         "ap"},
                                duration)
                results.append(r)
            except Exception as e:
                import traceback
                print(f"  {clf_name}/{strategy} ERROR: {e}")
                traceback.print_exc()

    # ── Pick champion ──────────────────────────────────────────────────────────
    if results:
        best = max(results, key=_champion_score)
        best["is_champion"] = True
        m  = best["metrics"]
        print(f"\n[TRAIN] ═══ Champion: {best['classifier_name']} [{best['imbalance_strategy']}] ═══")
        print(f"  Composite={_champion_score(best):.4f}  "
              f"Recall@F1thr={best.get('max_recall', 0):.4f}  "
              f"Threshold={best['threshold']:.2f}  "
              f"AUC={m.get('auc_roc',0):.4f}  PR-AUC={m.get('pr_auc',0):.4f}")
        print(f"\n[TRAIN] All {len(results)} runs:")
        for r in sorted(results, key=_champion_score, reverse=True):
            m2 = r["metrics"]
            print(f"  {'⭐ ' if r.get('is_champion') else '   '}"
                  f"{r['classifier_name']:<22} [{r['imbalance_strategy']:<12}]  "
                  f"AUC={m2.get('auc_roc',0):.4f}  "
                  f"Recall={r.get('max_recall', m2.get('recall_fraud',0)):.4f}  "
                  f"F1={m2.get('f1_fraud',0):.4f}  "
                  f"Thr={r.get('threshold',0.5):.2f}  "
                  f"PR-AUC={m2.get('pr_auc',0):.4f}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION HELPER
# ═════════════════════════════════════════════════════════════════════════════

def cross_validate_champion(
    X: np.ndarray, y: np.ndarray,
    champion_artifact_path: str,
    n_splits: int = CV_FOLDS,
) -> dict:
    from sklearn.model_selection import StratifiedKFold
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_mets = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        try:
            clf = (AutoML() if _FLAML_OK else
                   LogisticRegression(max_iter=500, class_weight="balanced",
                                      random_state=RANDOM_STATE))
            if _FLAML_OK:
                clf.fit(X[tr_idx], y[tr_idx], time_budget=60, metric="ap",
                        task="classification", seed=RANDOM_STATE, verbose=0,
                        estimator_list=["lgbm", "rf"])
            else:
                clf.fit(X[tr_idx], y[tr_idx])
            prob   = clf.predict_proba(X[val_idx])[:, 1]
            opt_thr, opt_f1, opt_recall = f1_optimal_threshold(y[val_idx], prob)
            m      = compute_metrics(y[val_idx], prob, threshold=opt_thr)
            m["max_recall"]           = opt_recall
            m["max_recall_threshold"] = opt_thr
            fold_mets.append(m)
            print(f"  Fold {fold+1}: AUC={m['auc_roc']:.4f}  "
                  f"Recall={opt_recall:.4f}  F1={opt_f1:.4f}  Thr={opt_thr:.2f}")
        except Exception as e:
            print(f"  Fold {fold+1} error: {e}")
    if not fold_mets:
        return {}
    return {
        k: {"mean": float(np.mean([fm[k] for fm in fold_mets if k in fm])),
            "std":  float(np.std([fm[k]  for fm in fold_mets if k in fm]))}
        for k in fold_mets[0]
    }
