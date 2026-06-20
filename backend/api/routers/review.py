"""
/api/review — Human-in-the-loop review queue.

Workflow:
  1. Pipeline populates review_queue with uncertain predictions (0.3–0.7 score)
  2. Analyst fetches the queue via GET /queue
  3. Analyst submits a decision via POST /decide
  4. Labeled samples can be used for retraining
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc

from pipeline.db import (
    Session, ReviewQueueItem, ReviewDecision, Transaction, Prediction,
)
from config.settings import REVIEW_THRESHOLD_LOW, REVIEW_THRESHOLD_HIGH

router = APIRouter()


class DecisionRequest(BaseModel):
    queue_item_id: int
    decision:      str   # 'confirmed_fraud' | 'confirmed_legit' | 'uncertain'
    analyst_label: Optional[int] = None   # 0 or 1
    notes:         Optional[str] = None
    decided_by:    str = "analyst"


@router.get("/queue")
def get_review_queue(
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status:    str = Query("pending"),
):
    """Fetch the review queue — paginated."""
    session = Session()
    try:
        q = session.query(ReviewQueueItem).filter_by(status=status)\
                   .order_by(desc(ReviewQueueItem.fraud_prob))

        total = q.count()
        items = q.offset((page - 1) * page_size).limit(page_size).all()

        result = []
        for item in items:
            txn = session.get(Transaction, item.transaction_id)
            result.append({
                "queue_item_id": item.id,
                "transaction_id":item.transaction_id,
                "fraud_prob":    round(item.fraud_prob, 4),
                "status":        item.status,
                "created_at":    item.created_at.isoformat() if item.created_at else None,
                "reviewed_at":   item.reviewed_at.isoformat() if item.reviewed_at else None,
                "transaction": {
                    "amount":    txn.transaction_amt if txn else None,
                    "channel":   txn.channel if txn else None,
                    "country":   txn.country if txn else None,
                    "currency":  txn.currency if txn else None,
                    "card_type": txn.card_type if txn else None,
                    "card_bank": txn.card_bank if txn else None,
                    "is_fraud":  txn.is_fraud if txn else None,
                } if txn else None,
            })

        return {"total": total, "page": page, "page_size": page_size, "items": result}
    finally:
        session.close()


@router.get("/item/{queue_item_id}")
def get_review_item(queue_item_id: int):
    session = Session()
    try:
        item = session.get(ReviewQueueItem, queue_item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found")

        txn  = session.get(Transaction, item.transaction_id)
        pred = session.query(Prediction)\
                      .filter_by(transaction_id=item.transaction_id)\
                      .order_by(desc(Prediction.predicted_at)).first()

        return {
            "queue_item_id": item.id,
            "transaction_id":item.transaction_id,
            "fraud_prob":    round(item.fraud_prob, 4),
            "status":        item.status,
            "created_at":    item.created_at.isoformat() if item.created_at else None,
            "transaction": {
                "amount":         txn.transaction_amt,
                "channel":        txn.channel,
                "country":        txn.country,
                "currency":       txn.currency,
                "card_type":      txn.card_type,
                "card_bank":      txn.card_bank,
                "transaction_dt": txn.transaction_dt,
                "is_fraud":       txn.is_fraud,
            } if txn else None,
            "prediction": {
                "fraud_prob": round(pred.fraud_prob, 4) if pred else None,
                "is_alarm":   pred.is_alarm if pred else None,
                "threshold":  pred.threshold_used if pred else None,
            } if pred else None,
            "existing_decision": {
                "decision":    item.decision.decision if item.decision else None,
                "decided_at":  item.decision.decided_at.isoformat() if item.decision else None,
                "decided_by":  item.decision.decided_by if item.decision else None,
            } if item.decision else None,
        }
    finally:
        session.close()


@router.post("/decide")
def submit_decision(body: DecisionRequest):
    """Analyst submits a confirm/override decision."""
    valid_decisions = {"confirmed_fraud", "confirmed_legit", "uncertain"}
    if body.decision not in valid_decisions:
        raise HTTPException(
            status_code=422,
            detail=f"decision must be one of: {valid_decisions}",
        )

    session = Session()
    try:
        item = session.get(ReviewQueueItem, body.queue_item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found")

        if item.decision:
            # Update existing decision
            item.decision.decision      = body.decision
            item.decision.analyst_label = body.analyst_label
            item.decision.notes         = body.notes
            item.decision.decided_at    = datetime.utcnow()
            item.decision.decided_by    = body.decided_by
        else:
            session.add(ReviewDecision(
                queue_item_id = body.queue_item_id,
                decision      = body.decision,
                analyst_label = body.analyst_label if body.analyst_label is not None
                                else (1 if body.decision == "confirmed_fraud" else 0),
                notes         = body.notes,
                decided_by    = body.decided_by,
            ))

        item.status      = "reviewed"
        item.reviewed_at = datetime.utcnow()
        item.reviewed_by = body.decided_by

        session.commit()
        return {"status": "ok", "queue_item_id": body.queue_item_id, "decision": body.decision}
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()


@router.get("/stats")
def get_review_stats():
    session = Session()
    try:
        total     = session.query(ReviewQueueItem).count()
        pending   = session.query(ReviewQueueItem).filter_by(status="pending").count()
        reviewed  = session.query(ReviewQueueItem).filter_by(status="reviewed").count()
        confirmed = session.query(ReviewDecision).filter_by(decision="confirmed_fraud").count()
        overridden= session.query(ReviewDecision).filter_by(decision="confirmed_legit").count()
        uncertain = session.query(ReviewDecision).filter_by(decision="uncertain").count()
        return {
            "total": total, "pending": pending, "reviewed": reviewed,
            "confirmed_fraud": confirmed, "overridden_to_legit": overridden,
            "uncertain": uncertain,
            "review_rate": round(reviewed / max(total, 1), 4),
        }
    finally:
        session.close()


@router.post("/populate-queue")
def populate_review_queue(model_run_id: int = Query(None)):
    """
    Populate the review queue from uncertain predictions (score between thresholds).
    Called automatically after pipeline runs; can also be triggered manually.
    """
    session = Session()
    try:
        from pipeline.db import Prediction
        q = session.query(Prediction).filter(
            Prediction.fraud_prob >= REVIEW_THRESHOLD_LOW,
            Prediction.fraud_prob <  REVIEW_THRESHOLD_HIGH,
        )
        if model_run_id:
            q = q.filter(Prediction.model_run_id == model_run_id)

        existing = {r.transaction_id for r in session.query(ReviewQueueItem).all()}
        added = 0
        for pred in q.all():
            if pred.transaction_id not in existing:
                session.add(ReviewQueueItem(
                    transaction_id=pred.transaction_id,
                    fraud_prob=pred.fraud_prob,
                    model_run_id=pred.model_run_id,
                ))
                existing.add(pred.transaction_id)
                added += 1

        session.commit()
        return {"status": "ok", "added_to_queue": added}
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
