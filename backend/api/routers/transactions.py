"""
/api/transactions — paginated transaction list + detail + predict on-demand.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc

from pipeline.db import Session, Transaction, Prediction, ReviewQueueItem

router = APIRouter()


@router.get("/")
def list_transactions(
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    split:     str = Query(None),
    channel:   str = Query(None),
    country:   str = Query(None),
    is_fraud:  int = Query(None),
    min_prob:  float = Query(None),
):
    session = Session()
    try:
        q = session.query(Transaction)
        if split:
            q = q.filter(Transaction.split == split)
        if channel:
            q = q.filter(Transaction.channel == channel)
        if country:
            q = q.filter(Transaction.country == country)
        if is_fraud is not None:
            q = q.filter(Transaction.is_fraud == is_fraud)

        if min_prob is not None:
            # Filter by prediction score
            q = q.join(Prediction, Prediction.transaction_id == Transaction.transaction_id)\
                 .filter(Prediction.fraud_prob >= min_prob)

        total = q.count()
        txns  = q.order_by(desc(Transaction.transaction_dt))\
                 .offset((page - 1) * page_size).limit(page_size).all()

        result = []
        for t in txns:
            latest_pred = session.query(Prediction)\
                                 .filter_by(transaction_id=t.transaction_id)\
                                 .order_by(desc(Prediction.predicted_at)).first()
            in_review = session.query(ReviewQueueItem)\
                               .filter_by(transaction_id=t.transaction_id).first()
            result.append({
                "transaction_id": t.transaction_id,
                "split":          t.split,
                "transaction_dt": t.transaction_dt,
                "amount":         t.transaction_amt,
                "country":        t.country,
                "currency":       t.currency,
                "channel":        t.channel,
                "card_type":      t.card_type,
                "card_bank":      t.card_bank,
                "is_fraud":       t.is_fraud,
                "fraud_prob":     round(latest_pred.fraud_prob, 4) if latest_pred else None,
                "is_alarm":       latest_pred.is_alarm if latest_pred else None,
                "in_review":      in_review is not None,
                "review_status":  in_review.status if in_review else None,
            })

        return {"total": total, "page": page, "page_size": page_size, "transactions": result}
    finally:
        session.close()


@router.get("/{transaction_id}")
def get_transaction(transaction_id: int):
    session = Session()
    try:
        txn = session.get(Transaction, transaction_id)
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")

        preds = session.query(Prediction)\
                       .filter_by(transaction_id=transaction_id)\
                       .order_by(desc(Prediction.predicted_at)).all()

        review = session.query(ReviewQueueItem)\
                        .filter_by(transaction_id=transaction_id).first()

        return {
            "transaction_id":  txn.transaction_id,
            "split":           txn.split,
            "transaction_dt":  txn.transaction_dt,
            "amount":          txn.transaction_amt,
            "country":         txn.country,
            "currency":        txn.currency,
            "channel":         txn.channel,
            "card_type":       txn.card_type,
            "card_bank":       txn.card_bank,
            "is_fraud":        txn.is_fraud,
            "ingested_at":     txn.ingested_at.isoformat() if txn.ingested_at else None,
            "predictions":     [
                {
                    "fraud_prob":   round(p.fraud_prob, 4),
                    "is_alarm":     p.is_alarm,
                    "threshold":    p.threshold_used,
                    "predicted_at": p.predicted_at.isoformat() if p.predicted_at else None,
                }
                for p in preds
            ],
            "review": {
                "status":      review.status if review else None,
                "fraud_prob":  round(review.fraud_prob, 4) if review else None,
                "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
            } if review else None,
        }
    finally:
        session.close()
