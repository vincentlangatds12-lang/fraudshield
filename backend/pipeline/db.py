"""
Database schema — Umba Fraud Detection Platform.

Tables:
  transactions      — raw + enriched transaction records
  model_runs        — one row per training run (all classifiers)
  model_artifacts   — best model per run (pkl path, metrics JSON)
  predictions       — model scores on test/live transactions
  review_queue      — human-in-the-loop uncertain cases
  review_decisions  — analyst accept/override decisions
  feature_importance— top feature weights per model run
  shap_values       — per-transaction SHAP explanations (sample)
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine, Index,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_URL

Base = declarative_base()


# ── Transaction ───────────────────────────────────────────────────────────────

class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id          = Column(Integer, primary_key=True)
    split                   = Column(String, default="train")  # train | test
    transaction_dt          = Column(Integer)
    transaction_amt         = Column(Float)
    country                 = Column(String)
    currency                = Column(String)
    channel                 = Column(String)
    card_type               = Column(String)
    card_bank               = Column(String)
    is_fraud                = Column(Integer, nullable=True)   # null for test
    flagged_for_review      = Column(Float, nullable=True)
    # Enriched features (stored after feature engineering)
    features_json           = Column(Text, nullable=True)      # JSON dict
    ingested_at             = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="transaction")
    review_items = relationship("ReviewQueueItem", back_populates="transaction")


# ── Model Run ─────────────────────────────────────────────────────────────────

class ModelRun(Base):
    """One row per classifier per training run."""
    __tablename__ = "model_runs"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    run_id              = Column(String, nullable=False)        # UUID, shared across classifiers in same batch
    mlflow_run_id       = Column(String, nullable=True)
    classifier_name     = Column(String, nullable=False)        # e.g. 'lightgbm', 'flaml'
    imbalance_strategy  = Column(String, default="class_weight")
    hyperparams         = Column(Text, default="{}")            # JSON
    # Metrics
    accuracy            = Column(Float, nullable=True)
    auc_roc             = Column(Float, nullable=True)
    pr_auc              = Column(Float, nullable=True)
    f1_fraud            = Column(Float, nullable=True)
    precision_fraud     = Column(Float, nullable=True)
    recall_fraud        = Column(Float, nullable=True)
    mcc                 = Column(Float, nullable=True)
    ks_statistic        = Column(Float, nullable=True)
    avg_precision       = Column(Float, nullable=True)
    # ── Max-recall threshold scan ─────────────────────────────────────────────
    # Threshold scanned 0→1 at step 0.02 to find the value that maximises recall.
    # This is stored separately so analysts can compare models by their recall ceiling.
    max_recall           = Column(Float, nullable=True)   # highest recall achievable
    max_recall_threshold = Column(Float, nullable=True)   # threshold that gives max_recall
    # Status
    artifact_path       = Column(String, nullable=True)
    is_champion         = Column(Boolean, default=False)
    trained_at          = Column(DateTime, default=datetime.utcnow)
    training_duration_s = Column(Float, nullable=True)
    notes               = Column(Text, nullable=True)

    feature_importance = relationship("FeatureImportance", back_populates="model_run")


# ── Predictions ───────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id  = Column(Integer, ForeignKey("transactions.transaction_id"))
    model_run_id    = Column(Integer, ForeignKey("model_runs.id"))
    fraud_prob      = Column(Float, nullable=False)
    is_alarm        = Column(Boolean, default=False)   # prob >= threshold
    threshold_used  = Column(Float, nullable=True)
    predicted_at    = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="predictions")


# ── Review Queue ──────────────────────────────────────────────────────────────

class ReviewQueueItem(Base):
    """Transactions with uncertain scores queued for human review."""
    __tablename__ = "review_queue"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id  = Column(Integer, ForeignKey("transactions.transaction_id"))
    fraud_prob      = Column(Float, nullable=False)
    model_run_id    = Column(Integer, ForeignKey("model_runs.id"), nullable=True)
    status          = Column(String, default="pending")  # pending | reviewed | skipped
    created_at      = Column(DateTime, default=datetime.utcnow)
    reviewed_at     = Column(DateTime, nullable=True)
    reviewed_by     = Column(String, nullable=True)

    transaction = relationship("Transaction", back_populates="review_items")
    decision    = relationship("ReviewDecision", back_populates="queue_item", uselist=False)


class ReviewDecision(Base):
    """Analyst confirm/override decisions."""
    __tablename__ = "review_decisions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    queue_item_id   = Column(Integer, ForeignKey("review_queue.id"))
    decision        = Column(String)   # 'confirmed_fraud' | 'confirmed_legit' | 'uncertain'
    analyst_label   = Column(Integer, nullable=True)  # 0 or 1 (for retraining)
    notes           = Column(Text, nullable=True)
    decided_at      = Column(DateTime, default=datetime.utcnow)
    decided_by      = Column(String, default="analyst")

    queue_item = relationship("ReviewQueueItem", back_populates="decision")


# ── Feature Importance ────────────────────────────────────────────────────────

class FeatureImportance(Base):
    __tablename__ = "feature_importance"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"))
    feature_name = Column(String)
    importance   = Column(Float)
    rank         = Column(Integer)

    model_run = relationship("ModelRun", back_populates="feature_importance")


# ── SHAP Values ───────────────────────────────────────────────────────────────

class ShapExplanation(Base):
    """Sampled SHAP explanations — one row per (transaction, model_run)."""
    __tablename__ = "shap_explanations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id  = Column(Integer, ForeignKey("transactions.transaction_id"))
    model_run_id    = Column(Integer, ForeignKey("model_runs.id"))
    shap_values_json= Column(Text)   # JSON: {feature: shap_value, ...}
    base_value      = Column(Float)
    predicted_prob  = Column(Float)
    computed_at     = Column(DateTime, default=datetime.utcnow)


# ── Class Imbalance Report ─────────────────────────────────────────────────────

class ImbalanceReport(Base):
    __tablename__ = "imbalance_reports"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    run_id              = Column(String)
    fraud_count         = Column(Integer)
    legit_count         = Column(Integer)
    imbalance_ratio     = Column(Float)
    strategy_used       = Column(String)
    strategy_comparison = Column(Text)   # JSON: strategy → {auc, f1, pr_auc}
    computed_at         = Column(DateTime, default=datetime.utcnow)


# ── Active Learning Labels ────────────────────────────────────────────────────

class ActiveLearningLabel(Base):
    """
    Analyst decisions converted to training labels for active learning.

    Every time an analyst confirms or overrides a prediction in the review queue,
    a row is written here. The pipeline's Stage 0 reads these rows and merges
    them into the training set before feature engineering, so the next model run
    benefits from human feedback.

    used_in_run_id: populated by the pipeline when this label has been trained on,
                    to avoid double-counting across runs.
    """
    __tablename__ = "active_learning_labels"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id    = Column(Integer, ForeignKey("transactions.transaction_id"))
    analyst_label     = Column(Integer, nullable=False)   # 0=legit, 1=fraud
    decision          = Column(String)                    # confirmed_fraud | confirmed_legit | uncertain
    source            = Column(String, default="review_queue")  # review_queue | manual
    labeled_by        = Column(String, default="analyst")
    labeled_at        = Column(DateTime, default=datetime.utcnow)
    used_in_run_id    = Column(String, nullable=True)     # run_id that consumed this label
    notes             = Column(Text, nullable=True)

    transaction = relationship("Transaction")


# ── Engine + Session ──────────────────────────────────────────────────────────

if "sqlite" in DB_URL:
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
    )

Session = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine)
    print(f"[DB] Initialised: {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    init_db()
