"""
MLflow experiment tracking for the Umba Fraud Detection Platform.

Logs:
  - all model hyperparameters
  - all 8 evaluation metrics
  - artifact path (model pkl)
  - imbalance strategy
  - feature count
  - training duration
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

_MLFLOW_OK = False
try:
    import mlflow
    _MLFLOW_OK = True
except ImportError:
    print("[MLFLOW] mlflow not installed — tracking disabled")


def setup_mlflow() -> bool:
    if not _MLFLOW_OK:
        return False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"[MLFLOW] Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"[MLFLOW] Experiment:   {MLFLOW_EXPERIMENT_NAME}")
        return True
    except Exception as exc:
        print(f"[MLFLOW] Setup failed: {exc}")
        return False


def log_run(result: dict, feature_count: int) -> str | None:
    """
    Log a single classifier result to MLflow.
    Returns the mlflow run_id or None.
    """
    if not _MLFLOW_OK:
        return None

    try:
        with mlflow.start_run(run_name=f"{result['classifier_name']}_{result['run_id'][:8]}") as run:
            # Tags
            mlflow.set_tags({
                "classifier":          result["classifier_name"],
                "imbalance_strategy":  result["imbalance_strategy"],
                "run_batch_id":        result["run_id"],
                "is_champion":         str(result.get("is_champion", False)),
            })

            # Params
            params = result.get("hyperparams", {})
            params["feature_count"]       = feature_count
            params["imbalance_strategy"]  = result["imbalance_strategy"]
            params["threshold"]           = result.get("threshold", 0.5)
            for k, v in params.items():
                try:
                    mlflow.log_param(str(k)[:250], str(v)[:500])
                except Exception:
                    pass

            # Metrics
            for metric_name, value in result["metrics"].items():
                try:
                    mlflow.log_metric(metric_name, float(value))
                except Exception:
                    pass

            mlflow.log_metric("training_duration_s", result.get("training_duration_s", 0))

            # Artifact
            artifact_path = result.get("artifact_path")
            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

            print(f"[MLFLOW] Logged run: {run.info.run_id} ({result['classifier_name']})")
            return run.info.run_id

    except Exception as exc:
        print(f"[MLFLOW] Log failed for {result['classifier_name']}: {exc}")
        return None


def get_all_runs() -> list[dict]:
    """Retrieve all runs for the experiment."""
    if not _MLFLOW_OK:
        return []
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.pr_auc DESC"],
            max_results=100,
        )
        return [
            {
                "run_id":      r.info.run_id,
                "run_name":    r.info.run_name,
                "status":      r.info.status,
                "classifier":  r.data.tags.get("classifier", ""),
                "strategy":    r.data.tags.get("imbalance_strategy", ""),
                "is_champion": r.data.tags.get("is_champion", "False") == "True",
                "metrics":     dict(r.data.metrics),
                "params":      dict(r.data.params),
                "start_time":  r.info.start_time,
            }
            for r in runs
        ]
    except Exception as exc:
        print(f"[MLFLOW] get_all_runs error: {exc}")
        return []
