# Data Dictionary

Three files. `train.csv` and `test.csv` share the same schema except that
`test.csv` has no `isFraud` column (that's what you predict). `identity.csv` is
a separate feed that joins on `TransactionID`.

The data is anonymised. Some columns are named; others are anonymised feature
blocks (`C*`, `D*`, `M*`, `V*`) whose exact definitions we don't disclose.
**Read this carefully** — several fields have characteristics that matter for
how you should (and shouldn't) use them.

---

## `train.csv` / `test.csv`

### Transaction core

| Column | Type | Description |
|---|---|---|
| `TransactionID` | int | Unique transaction identifier. Primary key. |
| `TransactionDT` | int | Timestamp as **seconds offset from a fixed reference**. The data is time-ordered; **`test.csv` covers a strictly later period than `train.csv`** (as in production, you predict the future). |
| `TransactionAmt` | float | Transaction amount in the transaction's **local currency** — see `currency`. |
| `country` | str | `KE` (Kenya) or `NG` (Nigeria). |
| `currency` | str | `KES` or `NGN`. Note the two currencies are on very different numeric scales. |
| `channel` | str | `mobile_money`, `p2p`, `bank_transfer`, `card`, `airtime`, `bill_pay`. |
| `card_type` | str | `debit`, `credit`, `prepaid`. |
| `card_bank` | str | Anonymised issuer / wallet provider. |

### Card / account (anonymised)

| Column | Type | Description |
|---|---|---|
| `card1`–`card3`, `card5` | num | Anonymised card/account attributes (issuer, type, region, etc.). High-cardinality identifiers in part. |
| `addr1`, `addr2` | num | Anonymised address/region codes. |
| `dist1`, `dist2` | num | Anonymised distance measures (e.g. billing↔transaction). Often missing. |
| `P_emaildomain` | str | Payer email domain. Real-world messy (includes blanks and typos). |
| `R_emaildomain` | str | Recipient email domain. Same caveats. |
| `recipient_account_age_days` | int | Age of the recipient account in days at transaction time. |
| `sender_prev_txn_count` | int | Number of prior transactions by the sender. |

### Anonymised feature blocks

| Block | Type | Description |
|---|---|---|
| `C1`–`C8` | int | Counting features — counts of entities/events associated with the transaction (velocity-style signals). |
| `D1`–`D5` | float | Timedelta features (days between events). May be missing; values can be noisy. |
| `M1`–`M6` | cat | Match flags: `T` / `F` / missing. These encode whether identity attributes match — e.g. account-name vs KYC-name, email, national ID, device, address, phone. |
| `V1`–`V20` | float | Anonymised engineered features (aggregations, ratios, counts). Many are correlated and many are sparse. |

### Other

| Column | Type | Description |
|---|---|---|
| `flagged_for_review` | float | Outcome flag from the back-office **manual-review queue**. Populated by reviewers as part of the review process. **Consider at what point in the transaction lifecycle this value becomes known**, relative to when you'd need to score a transaction. |
| `isFraud` | int | **Target.** `1` = confirmed fraud (raise an alarm), `0` = legitimate. Present in `train.csv` only. |

---

## `identity.csv`

Device / session information from a **separate feed**. Not every transaction has
identity data, and the granularity is **per device session** — a single
transaction may have **zero, one, or several** session rows. **Check the row
granularity and key uniqueness before you join.**

| Column | Type | Description |
|---|---|---|
| `TransactionID` | int | Foreign key to the transactions tables. **Not guaranteed unique in this file.** |
| `DeviceType` | str | `mobile` / `desktop`. |
| `DeviceInfo` | str | Free-text device string. Messy and frequently missing. |
| `id_01`–`id_11` | float | Anonymised identity/session features. Frequently missing. |

---

## `sample_submission.csv`

The exact output format we score. Replace the placeholder probabilities with
your model's scores for every `TransactionID` in `test.csv`.

| Column | Type | Description |
|---|---|---|
| `TransactionID` | int | Every id in `test.csv`, exactly once. |
| `isFraud_prob` | float | Your predicted P(fraud) in `[0, 1]`. |
