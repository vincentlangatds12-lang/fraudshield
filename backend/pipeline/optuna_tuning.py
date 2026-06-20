"""
Optuna hyperparameter tuning for LightGBM and XGBoost.

Top Kaggle scorers always tune these two — they respond well to hyperparameter
optimisation and can gain 2-5% AUC with proper tuning.

Objective metric: average_precision (PR-AUC proxy) — honest under class imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

_OPTUNA_OK = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_OK = True
except ImportError:
    print("[OPTUNA] optuna not installed — skipping hyperparameter tuning")

_LGB_OK = False
try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    pass

_XGB_OK = False
try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    pass


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_ratio: float = 10.0,
    n_trials: int = 50,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Tune LightGBM with Optuna. Returns best hyperparams dict.
    Falls back to sensible defaults if optuna/lightgbm not installed.
    """
    defaults = {
        "n_estimators":      500,
        "learning_rate":     0.05,
        "max_depth":         7,
        "num_leaves":        63,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "scale_pos_weight":  imbalance_ratio,
    }

    if not _OPTUNA_OK or not _LGB_OK:
        print("[OPTUNA] LightGBM tuning skipped — using defaults")
        return defaults

    print(f"[OPTUNA] Tuning LightGBM — {n_trials} trials, {cv_folds}-fold CV …")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":          trial.suggest_int("max_depth", 4, 10),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":   imbalance_ratio,
            "random_state":       random_state,
            "n_jobs":             -1,
            "verbose":            -1,
        }
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr   = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            prob = clf.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, prob))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"[OPTUNA] LightGBM best PR-AUC={study.best_value:.4f}  params={best}")
    return best


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_ratio: float = 10.0,
    n_trials: int = 40,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """Tune XGBoost with Optuna. Returns best hyperparams dict."""
    defaults = {
        "n_estimators":    500,
        "learning_rate":   0.05,
        "max_depth":       7,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "gamma":           0.1,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "scale_pos_weight":imbalance_ratio,
    }

    if not _OPTUNA_OK or not _XGB_OK:
        print("[OPTUNA] XGBoost tuning skipped — using defaults")
        return defaults

    print(f"[OPTUNA] Tuning XGBoost — {n_trials} trials, {cv_folds}-fold CV …")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":          trial.suggest_int("max_depth", 4, 9),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":   imbalance_ratio,
            "eval_metric":        "aucpr",
            "use_label_encoder":  False,
            "random_state":       random_state,
            "n_jobs":             -1,
            "verbosity":          0,
        }
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr   = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False)
            prob = clf.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, prob))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"[OPTUNA] XGBoost best PR-AUC={study.best_value:.4f}  params={best}")
    return best
