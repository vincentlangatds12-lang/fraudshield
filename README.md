# Umba — Take-Home Assessment: Real-Time Fraud Detection

Welcome, and thanks for taking the time. This exercise mirrors the kind of work
you'd actually do on our team: take messy, real-world financial data, build a
model that drives a live decision, stand it up behind an API, and make the
results legible to non-technical colleagues — end to end, owning the whole loop.

We care far more about **judgment, rigour, and a working v1** than about polish
or completeness. Read the whole brief before you start.

---

## The scenario

Umba processes mobile-money, card, and bank-transfer transactions across Kenya
and Nigeria. A small fraction are fraudulent. When a transaction is fraudulent,
we want to **raise an alarm** so the back-office team can hold and review it
before money moves.

You're building the **v1 of that fraud-detection system**: the model, the
service that scores transactions, and a dashboard the operations team can look
at. This is a binary classification problem:

- `isFraud = 1` → fraud → should trigger an alarm
- `isFraud = 0` → legitimate

Fraud is **rare** (a few percent), the data is **messy real-world transaction
data**, and the transactions you'll be scored on happen **later in time** than
the ones you train on — just like production.

---

## Time budget

**Aim for ~4–6 hours.** This is deliberately more than you can perfectly finish
in that window. We want to see how you **prioritise** under a realistic
constraint. A focused, correct, well-explained v1 beats a sprawling,
half-working v3. If you run out of time, write down what you'd do next.

You may use **any** language, libraries, and tools.

---

## Using AI tools — encouraged

This is an **AI-native** role. We use Claude Code, Codex, and similar tools
every day, and we expect you to. **Using them here is encouraged, not
penalised.**

What we're actually assessing is your **judgment and review**: AI will happily
generate a pipeline that runs clean, passes a smoke test, and is quietly wrong.
Your job is to catch that. **You own every line you submit and should be able to
explain and defend it.** A short note on how you used AI (what you delegated,
what you caught, what you rejected) is welcome.

---

## What's in the box

```
data/
  train.csv              labelled transactions (has isFraud)        — train on this
  test.csv               later-period transactions (no isFraud)     — score these
  identity.csv           device/session feed, joins on TransactionID
  sample_submission.csv  the exact format we expect for predictions
DATA_DICTIONARY.md       what every column means — read it carefully
requirements.txt         a suggested Python environment (optional to use)
```

There's no starter code — you build the pipeline, service, and dashboard from
scratch, structured however you see fit.

This is **anonymised transaction data** with all the warts you'd expect in
production: class imbalance, missing values, two currencies, a separate identity
feed, and fields that exist for analysis but aren't all available at decision
time. **Verifying data integrity is part of the job, not a distraction from it.**

---

## What to build

### Part A — Pipeline & model · *required, weighted most heavily*
A **reproducible** pipeline that goes from the raw CSVs to a trained classifier:
preprocessing, feature engineering, training, and **honest evaluation**. Then
score `test.csv` and produce a `predictions.csv` (format below).

We're looking for sound handling of **class imbalance**, an **evaluation setup
that reflects how the model will really be used**, and **awareness of leakage
and data-integrity pitfalls**. A short write-up of your choices, your metrics,
and what you'd improve with more time is required.

### Part B — Serving API · *required*
A small service (e.g. FastAPI/Flask) that loads your trained model and exposes a
`/predict` endpoint: given a transaction (or a batch), return a fraud
probability and an alarm decision. Keep it minimal but real — input validation,
a health check, and a sensible response shape.

### Part C — Dashboard · *required, keep it simple*
A simple frontend (Streamlit, a small React/HTML page, a notebook-as-dashboard —
your call) showing the model's behaviour on the test set: e.g. predicted fraud
rate, score distribution, the top-flagged transactions, performance at your
chosen alarm threshold, and whatever else helps an ops manager trust it.

### Part D — Dockerize & deploy · *bonus*
Containerise the API (and dashboard) and include a short deployment tutorial
(`docker compose up`, env vars, how you'd run it in the cloud). Genuinely
optional — only after A–C are solid.

> Bonus thinking we love to see (notes are fine, no need to build it):
> how you'd **monitor** this model in production, **detect drift**, and
> **retrain** as repayment/chargeback outcomes accumulate.

---

## How you'll be scored

Two parts, combined:

**1. Objective — your `predictions.csv` on the hidden test labels.**
We run an automated scorer. The **primary metric is PR-AUC** (average
precision), which is the honest metric under heavy class imbalance. We also look
at the **operational view**: if ops can only manually review the top *X%* of
riskiest transactions, what fraction of fraud do you catch? And at
**calibration** (are your probabilities meaningful?).

**2. Qualitative — your code and write-up.** Roughly:

| Area | What we look for |
|---|---|
| ML methodology & evaluation | imbalance handling, validation design, metric choice, thresholding, calibration |
| Data rigour & correctness | did you catch the leakage / integrity / join pitfalls in this data? |
| Code quality & reproducibility | clean, runnable end-to-end from a fresh checkout |
| API & dashboard | works, sensible design, useful to a non-technical user |
| Communication | clear write-up; can you explain *why*, not just *what* |
| AI-native working | effective use of AI tools **with** rigorous validation |

There is no single "right" answer. Reasoned trade-offs, clearly explained, score
better than an unexplained leaderboard number.

---

## Submission

Send us a **git repo** (or a zip) containing:

1. All your code — pipeline, API, dashboard.
2. Your trained **model artifact** (or a one-command script that reproduces it).
3. **`predictions.csv`** — your scores on `test.csv`, matching
   `sample_submission.csv` exactly:

   ```csv
   TransactionID,isFraud_prob
   1120000,0.0131
   1120001,0.8742
   ...
   ```
   - one row per `TransactionID` in `test.csv` — no more, no fewer
   - `isFraud_prob` is a probability/score in `[0, 1]`

4. A top-level **`README`** with: how to run everything, your write-up
   (approach, metrics, trade-offs, next steps), and your note on AI-tool usage.

Please don't spend more than ~6 hours. We'd rather see where you chose to stop.

Good luck — we're excited to see how you think.
