"""
Class imbalance analysis and resampling strategies.

Strategies compared:
  1. none           — raw data, no resampling (rely on class_weight='balanced')
  2. class_weight   — model-level balanced class weights (default production choice)
  3. smote          — Synthetic Minority Over-sampling Technique
  4. adasyn         — Adaptive Synthetic Sampling
  5. threshold_tune — vanilla model + optimal threshold search on val set

Returns a report dict and the resampled (X, y) for the chosen strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False
    print("[IMBALANCE] imbalanced-learn not installed — SMOTE/ADASYN disabled")


def analyse_imbalance(y: pd.Series | np.ndarray) -> dict:
    """Return counts and ratio for the target variable."""
    y = np.asarray(y)
    fraud_count = int((y == 1).sum())
    legit_count = int((y == 0).sum())
    total       = len(y)
    ratio       = legit_count / max(fraud_count, 1)
    fraud_rate  = fraud_count / total

    print(f"[IMBALANCE] Fraud={fraud_count} ({fraud_rate:.2%})  "
          f"Legit={legit_count}  Ratio={ratio:.1f}:1")

    return {
        "fraud_count":  fraud_count,
        "legit_count":  legit_count,
        "total":        total,
        "imbalance_ratio": ratio,
        "fraud_rate":   fraud_rate,
        "is_imbalanced": ratio >= 5,
    }


def apply_strategy(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "class_weight",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the named resampling strategy.
    Returns (X_resampled, y_resampled).
    For 'class_weight' and 'threshold_tune', returns original data unchanged.
    """
    strategy = strategy.lower()

    if strategy in ("none", "class_weight", "threshold_tune"):
        return X, y

    if not _IMBLEARN_AVAILABLE:
        print(f"[IMBALANCE] {strategy} requested but imbalanced-learn missing — using original data")
        return X, y

    if strategy == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=5)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    try:
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"[IMBALANCE] {strategy}: {len(y)} → {len(y_res)} samples "
              f"(fraud: {(y==1).sum()} → {(y_res==1).sum()})")
        return X_res, y_res
    except Exception as exc:
        print(f"[IMBALANCE] {strategy} failed ({exc}) — using original data")
        return X, y


def compare_strategies(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    cv_folds: int = 3,
) -> dict:
    """
    Quick cross-validated comparison of imbalance strategies using
    LogisticRegression as a fast proxy estimator.

    Returns dict: strategy_name → {auc_roc, pr_auc, f1}
    """
    strategies_to_test = ["class_weight"]
    if _IMBLEARN_AVAILABLE:
        strategies_to_test += ["smote", "adasyn"]

    results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for strat in strategies_to_test:
        aucs, prauc, f1s = [], [], []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            X_tr_r, y_tr_r = apply_strategy(X_tr, y_tr, strat, random_state)

            cw = "balanced" if strat in ("none", "class_weight") else None
            clf = LogisticRegression(
                max_iter=200, random_state=random_state,
                class_weight=cw, solver="saga", C=0.1,
            )
            try:
                clf.fit(X_tr_r, y_tr_r)
                proba = clf.predict_proba(X_val)[:, 1]
                aucs.append(roc_auc_score(y_val, proba))
                prauc.append(average_precision_score(y_val, proba))
                pred = (proba >= 0.5).astype(int)
                f1s.append(f1_score(y_val, pred, pos_label=1, zero_division=0))
            except Exception as exc:
                print(f"[IMBALANCE] Fold {fold} error for {strat}: {exc}")

        results[strat] = {
            "auc_roc": float(np.mean(aucs)) if aucs else 0.0,
            "pr_auc":  float(np.mean(prauc)) if prauc else 0.0,
            "f1":      float(np.mean(f1s)) if f1s else 0.0,
        }
        print(f"[IMBALANCE] {strat:15s}  AUC={results[strat]['auc_roc']:.4f}  "
              f"PR-AUC={results[strat]['pr_auc']:.4f}  F1={results[strat]['f1']:.4f}")

    # Pick best strategy by PR-AUC (honest metric for imbalanced data)
    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"[IMBALANCE] Best strategy: {best}")
    results["_best"] = best
    return results


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Search for the probability threshold that maximises F1 on the fraud class.
    Returns the optimal threshold float.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1    = 0.0
    best_thr   = 0.5
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1   = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr
    print(f"[THRESHOLD] Optimal threshold: {best_thr:.2f}  (F1={best_f1:.4f})")
    return float(best_thr)
