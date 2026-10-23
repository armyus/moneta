# Moneta

Moneta is a local-first recurring payment manager for subscriptions and bills. The MVP includes a dashboard, recurring payment management, manual transaction entry, CSV import, local machine-learning insights, search, filters, and a responsive layout.

## Start

Open `index.html` in a modern browser. There is no build step, account, backend, API key, or bank connection. The demo loads sample data on first use. Your changes are saved in that browser's local storage.

## Import a CSV

The importer recognizes headers such as `Date`, `Transaction Date`, `Description`, `Merchant`, `Amount`, and `Debit`. A `Category` column is optional. Use ISO date strings (`YYYY-MM-DD`) for predictable results.

```csv
Date,Description,Amount,Category
2026-08-01,Netflix,649,Entertainment
2026-09-01,Netflix,649,Entertainment
2026-08-05,Jio Fiber,999,Utilities
```

Amounts are treated as expenses and shown as rupees. CSV rows without a category receive an on-device TF-IDF + multinomial Logistic Regression prediction. The browser uses a learned model exported from `ml/train_model.py`; no third-party AI API is called. Predictions include a confidence score, and can be corrected in the Transactions table. User-reviewed and manually labeled transaction history can influence later predictions through a local similarity adjustment.

The training examples are in `ml/category-training.csv`. To retrain after adding labeled examples, run `python ml/train_model.py` from this folder. The standard-library training script prints holdout accuracy and regenerates `ml/category-model.js`, which is loaded by the web app. Recurring charge matching is a separate scoring model based on merchant similarity, amount stability, and payment timing.

The current sample model reports 95% accuracy on a deterministic 20-example holdout. This is a small, curated demonstration dataset, not a production performance guarantee; add representative, correctly labeled transactions before relying on the score for broader data.

Import is processed in your browser. The app does not upload financial data or ask for bank credentials.

## Limitations

This MVP does not connect to banks, cancel subscriptions, or send external email or push reminders. Recurring matches and expected dates are estimates and should be reviewed. Clearing site data in the browser removes locally saved records.
