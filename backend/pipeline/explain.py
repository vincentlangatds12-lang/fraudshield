"""
Explainability module — SHAP and LIME.

Provides:
  - global_shap()     → summary bar values (mean |SHAP|) for all features
  - local_shap()      → per-transaction SHAP waterfall values
  - local_lime()      → per-transaction LIME explanation
  - feature_importance_from_model() → model-native importance
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Any

_SHAP_OK = False
_LIME_OK = False

try:
    import shap
    _SHAP_OK = True
except ImportError:
    print("[EXPLAIN] shap not installed — SHAP disabled")

try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_OK = True
except ImportError:
    print("[EXPLAIN] lime not installed — LIME disabled")


def _unwrap_model(artifact: dict) -> Any:
    """Handle FLAML wrapping vs direct sklearn model."""
    clf = artifact["model"]
    if hasattr(clf, "model"):  # FLAML AutoML
        return clf.model.estimator
    return clf


def global_shap(
    artifact: dict,
    X_sample: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> list[dict]:
    """
    Compute mean |SHAP| values over a sample of the training set.
    Returns list of {feature, mean_abs_shap, rank}.
    """
    if not _SHAP_OK:
        return []

    clf = _unwrap_model(artifact)
    scaler = artifact.get("scaler")
    X = scaler.transform(X_sample) if scaler is not None else X_sample

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]

    try:
        # Tree-based models → TreeExplainer (fast)
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X)
            # Binary classification — take fraud class
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.LinearExplainer(clf, X)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        pairs = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "mean_abs_shap": float(v), "rank": i + 1}
                for i, (f, v) in enumerate(pairs[:30])]
    except Exception as exc:
        print(f"[SHAP] global_shap error: {exc}")
        return []


def local_shap(
    artifact: dict,
    x_row: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    SHAP values for a single transaction.
    Returns {feature: shap_value, ...} and base_value.
    """
    if not _SHAP_OK:
        return {"values": {}, "base_value": 0.0}

    clf = _unwrap_model(artifact)
    scaler = artifact.get("scaler")
    x = scaler.transform(x_row.reshape(1, -1)) if scaler is not None else x_row.reshape(1, -1)

    try:
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(x)
            if isinstance(shap_vals, list):
                vals      = shap_vals[1][0]
                base_val  = float(explainer.expected_value[1])
            else:
                vals     = shap_vals[0]
                base_val = float(explainer.expected_value)
        else:
            explainer = shap.LinearExplainer(clf, x)
            shap_vals = explainer.shap_values(x)
            vals     = (shap_vals[1] if isinstance(shap_vals, list) else shap_vals)[0]
            base_val = float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                             else explainer.expected_value[1])

        return {
            "values":     {f: float(v) for f, v in zip(feature_names, vals)},
            "base_value": base_val,
        }
    except Exception as exc:
        print(f"[SHAP] local_shap error: {exc}")
        return {"values": {}, "base_value": 0.0}


def local_lime(
    artifact: dict,
    x_row: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    num_features: int = 15,
) -> list[dict]:
    """
    LIME explanation for a single transaction.
    Returns list of {feature, weight, positive} sorted by |weight|.
    """
    if not _LIME_OK:
        return []

    clf = _unwrap_model(artifact)
    scaler = artifact.get("scaler")

    X_bg = scaler.transform(X_train) if scaler is not None else X_train
    x    = scaler.transform(x_row.reshape(1, -1))[0] if scaler is not None else x_row

    try:
        explainer = LimeTabularExplainer(
            training_data=X_bg,
            feature_names=feature_names,
            class_names=["legit", "fraud"],
            mode="classification",
            random_state=42,
        )

        def predict_fn(data):
            return clf.predict_proba(data)

        exp = explainer.explain_instance(x, predict_fn, num_features=num_features)
        raw = exp.as_list(label=1)

        return sorted(
            [{"feature": feat, "weight": float(w), "positive": w > 0} for feat, w in raw],
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )
    except Exception as exc:
        print(f"[LIME] local_lime error: {exc}")
        return []
