"""
5-Fold Cross-Validation — Champion Logistic Regression Model
============================================================

UPDATED VERSION:
- FIXED threshold = 0.5 (production reality)
- Accuracy added as first-class metric
- No threshold optimization leakage
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from scipy.stats import ks_2samp

from config.settings import RAW_DATA_DIR, DATA_DIR
from pipeline.feature_engineering import build_features

# ── Model config ──────────────────────────────────────────────────────────────
CHAMPION_CONFIG = {
    "C": 0.1,
    "class_weight": "balanced",
    "solver": "saga",
    "max_iter": 2000,
    "tol": 1e-3,
    "n_jobs": -1,
    "random_state": 42,
}

N_FOLDS = 5
RANDOM_STATE = 42

# 🔥 FIXED PRODUCTION THRESHOLD
FIXED_THRESHOLD = 0.5

# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 70)
print("Champion Model — 5-Fold CV (FIXED threshold = 0.5)")
print("=" * 70)

raw = Path(RAW_DATA_DIR)
train_df    = pd.read_csv(raw / "train.csv")
test_df     = pd.read_csv(raw / "test.csv")
identity_df = pd.read_csv(raw / "identity.csv")

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

# ── Feature engineering ──────────────────────────────────────────────────────
print("\n[1] Feature engineering …")

X_train_df, _, feature_names = build_features(train_df, test_df, identity_df)

train_sorted = train_df.sort_values("TransactionDT").reset_index(drop=True)
y_all = train_sorted["isFraud"].values.astype(np.int32)
X_all = X_train_df.values.astype(np.float32)

print(f"Features: {len(feature_names)} | Samples: {len(X_all):,}")

# ── Cross-validation ──────────────────────────────────────────────────────────
print(f"\n[2] Running {N_FOLDS}-Fold CV (threshold = 0.5)")
print("-" * 70)

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # Train
    clf = LogisticRegression(**CHAMPION_CONFIG)
    clf.fit(X_tr_s, y_tr)

    # Probabilities
    y_prob = clf.predict_proba(X_val_s)[:, 1]

    # 🔥 FIXED THRESHOLD
    y_pred = (y_prob >= FIXED_THRESHOLD).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc_roc = roc_auc_score(y_val, y_prob)
    pr_auc  = average_precision_score(y_val, y_prob)

    acc     = accuracy_score(y_val, y_pred)
    f1      = f1_score(y_val, y_pred, zero_division=0)
    recall  = recall_score(y_val, y_pred, zero_division=0)
    prec    = precision_score(y_val, y_pred, zero_division=0)
    mcc     = matthews_corrcoef(y_val, y_pred)

    ks_stat = ks_2samp(
        y_prob[y_val == 1],
        y_prob[y_val == 0]
    ).statistic

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results.append({
        "fold": fold,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
        "precision": prec,
        "mcc": mcc,
        "ks_stat": ks_stat,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "val_fraud": int(y_val.sum()),
    })

    print(
        f"Fold {fold}: "
        f"AUC={auc_roc:.4f} | "
        f"PR-AUC={pr_auc:.4f} | "
        f"Acc={acc:.4f} | "
        f"Recall={recall:.4f} | "
        f"Precision={prec:.4f} | "
        f"F1={f1:.4f} | "
        f"TP={tp} FP={fp} FN={fn}"
    )

# ── Summary ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SUMMARY — FIXED THRESHOLD = 0.5")
print("=" * 70)

metrics = [
    ("AUC-ROC", "auc_roc"),
    ("PR-AUC", "pr_auc"),
    ("Accuracy", "accuracy"),
    ("Recall", "recall"),
    ("Precision", "precision"),
    ("F1 Score", "f1"),
    ("MCC", "mcc"),
    ("KS-Statistic", "ks_stat"),
]

for label, col in metrics:
    vals = df[col]
    print(f"{label:<15}: {vals.mean():.4f} ± {vals.std():.4f}")

print("\nConfusion Matrix (TOTAL across folds)")
print("-" * 70)

print(f"TP: {df.tp.sum():>6}")
print(f"FP: {df.fp.sum():>6}")
print(f"TN: {df.tn.sum():>6}")
print(f"FN: {df.fn.sum():>6}")

total_fraud = df["val_fraud"].sum()
caught = df["tp"].sum()

print("\nFraud Capture Rate:")
print(f"{caught}/{total_fraud} = {caught/total_fraud*100:.2f}%")

# ── Save results ─────────────────────────────────────────────────────────────
out_path = Path(DATA_DIR) / "champion_cv_fixed_threshold_results.csv"
df.to_csv(out_path, index=False)

print(f"\nSaved results → {out_path}")

# ── Final verdict ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CHAMPION MODEL VERDICT (PRODUCTION VIEW)")
print("=" * 70)

print(f"Mean AUC-ROC : {df.auc_roc.mean():.4f}")
print(f"Mean PR-AUC  : {df.pr_auc.mean():.4f}")
print(f"Recall @0.5  : {df.recall.mean():.4f}")
print(f"Precision @0.5: {df.precision.mean():.4f}")
print(f"Accuracy @0.5 : {df.accuracy.mean():.4f}")
print("=" * 70)