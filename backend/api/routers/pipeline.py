"""
/api/pipeline — ingest raw CSVs into the database + pipeline utilities.
"""
from __future__ import annotations

import sys, os
# Fix OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from pipeline.db import Session, Transaction, init_db
from config.settings import RAW_DATA_DIR, DATA_DIR

router = APIRouter()

_ingest_status: dict = {"running": False, "ingested": 0, "error": None}


def _ingest_bg(data_dir: str):
    global _ingest_status
    _ingest_status["running"] = True
    _ingest_status["error"]   = None
    try:
        from pathlib import Path
        p = Path(data_dir)

        train    = pd.read_csv(p / "train.csv")
        test     = pd.read_csv(p / "test.csv")

        session = Session()
        try:
            # Clear existing
            session.query(Transaction).delete()
            session.commit()

            def _ingest_df(df: pd.DataFrame, split: str):
                rows = []
                for _, row in df.iterrows():
                    rows.append(Transaction(
                        transaction_id     = int(row["TransactionID"]),
                        split              = split,
                        transaction_dt     = int(row.get("TransactionDT", 0)),
                        transaction_amt    = float(row.get("TransactionAmt", 0)),
                        country            = str(row.get("country", "")) if pd.notna(row.get("country")) else None,
                        currency           = str(row.get("currency", "")) if pd.notna(row.get("currency")) else None,
                        channel            = str(row.get("channel", "")) if pd.notna(row.get("channel")) else None,
                        card_type          = str(row.get("card_type", "")) if pd.notna(row.get("card_type")) else None,
                        card_bank          = str(row.get("card_bank", "")) if pd.notna(row.get("card_bank")) else None,
                        is_fraud           = int(row["isFraud"]) if "isFraud" in row and pd.notna(row["isFraud"]) else None,
                        flagged_for_review = float(row["flagged_for_review"]) if "flagged_for_review" in row and pd.notna(row.get("flagged_for_review")) else None,
                    ))
                    if len(rows) >= 1000:
                        session.bulk_save_objects(rows)
                        session.commit()
                        rows = []
                if rows:
                    session.bulk_save_objects(rows)
                    session.commit()

            _ingest_df(train, "train")
            _ingest_df(test,  "test")
            _ingest_status["ingested"] = len(train) + len(test)
            print(f"[INGEST] Done — {_ingest_status['ingested']} transactions")
        finally:
            session.close()
    except Exception as exc:
        _ingest_status["error"] = str(exc)
        print(f"[INGEST] Error: {exc}")
    finally:
        _ingest_status["running"] = False


class IngestRequest(BaseModel):
    data_dir: str | None = None


@router.post("/ingest")
def ingest_data(body: IngestRequest, background_tasks: BackgroundTasks):
    if _ingest_status["running"]:
        raise HTTPException(status_code=409, detail="Ingest already running")
    data_dir = body.data_dir or RAW_DATA_DIR
    background_tasks.add_task(_ingest_bg, data_dir)
    return {"status": "started", "data_dir": data_dir}


@router.get("/ingest-status")
def get_ingest_status():
    session = Session()
    try:
        count = session.query(Transaction).count()
        return {
            **_ingest_status,
            "db_transaction_count": count,
        }
    finally:
        session.close()


@router.post("/reset-db")
def reset_database():
    """Drop and recreate all tables. USE WITH CAUTION."""
    from pipeline.db import Base, engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return {"status": "ok", "message": "Database reset"}
