"""Train and export Moneta's browser-ready category classifier (Python stdlib only)."""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "category-training.csv"
OUTPUT = ROOT / "category-model.js"
SEED = 17


STOP_WORDS = {"payment", "paid", "debit", "credit", "monthly", "plan", "subscription",
              "membership", "purchase", "order", "online", "recurring", "transaction",
              "bill", "auto", "card", "fee"}


def tokens(text: str) -> list[str]:
    words = [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def read_data():
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 24:
        raise ValueError("Add at least 24 labeled examples to category-training.csv")
    return [(r["description"].strip(), r["category"].strip()) for r in rows
            if r.get("description", "").strip() and r.get("category", "").strip()]


def make_features(texts):
    docs = [tokens(text) for text in texts]
    vocab = sorted({term for doc in docs for term in doc})
    index = {term: i for i, term in enumerate(vocab)}
    df = Counter(term for doc in docs for term in set(doc))
    idf = [math.log((1 + len(docs)) / (1 + df[term])) + 1 for term in vocab]
    vectors = []
    for doc in docs:
        counts = Counter(doc)
        vec = {index[term]: (1 + math.log(count)) * idf[index[term]]
               for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vec.values())) or 1
        vectors.append({i: value / norm for i, value in vec.items()})
    return vocab, idf, vectors


def softmax(logits):
    high = max(logits)
    exps = [math.exp(value - high) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def predict(vector, weights, bias):
    logits = [bias[c] + sum(weights[c][i] * value for i, value in vector.items())
              for c in range(len(weights))]
    return softmax(logits)


def main():
    random.seed(SEED)
    rows = read_data()
    texts, labels_text = zip(*rows)
    labels = sorted(set(labels_text))
    label_id = {label: i for i, label in enumerate(labels)}
    y = [label_id[label] for label in labels_text]
    vocab, idf, vectors = make_features(texts)

    # Deterministic holdout: every fifth row is reserved for a simple accuracy check.
    train_ids = [i for i in range(len(y)) if i % 5 != 0]
    valid_ids = [i for i in range(len(y)) if i % 5 == 0]
    classes, feature_count = len(labels), len(vocab)
    weights = [[0.0] * feature_count for _ in labels]
    bias = [0.0] * classes
    rate, l2, epochs = 0.12, 0.0008, 180

    for epoch in range(epochs):
        random.shuffle(train_ids)
        learning_rate = rate / (1 + epoch * 0.006)
        for row_id in train_ids:
            vector = vectors[row_id]
            probs = predict(vector, weights, bias)
            for c in range(classes):
                gradient = probs[c] - (1.0 if y[row_id] == c else 0.0)
                bias[c] -= learning_rate * gradient
                for feature, value in vector.items():
                    weights[c][feature] -= learning_rate * (gradient * value + l2 * weights[c][feature])

    valid_predictions = [max(range(classes), key=lambda c: predict(vectors[i], weights, bias)[c])
                         for i in valid_ids]
    correct = sum(prediction == y[row_id] for row_id, prediction in zip(valid_ids, valid_predictions))
    accuracy = correct / len(valid_ids) if valid_ids else 0
    model = {
        "name": "Moneta Local Category Classifier",
        "algorithm": "TF-IDF + multinomial Logistic Regression",
        "version": 1,
        "labels": labels,
        "vocabulary": vocab,
        "idf": [round(value, 8) for value in idf],
        "weights": [[round(value, 8) for value in row] for row in weights],
        "bias": [round(value, 8) for value in bias],
        "trainingExamples": len(rows),
        "validationExamples": len(valid_ids),
        "validationAccuracy": round(accuracy, 4),
    }
    OUTPUT.write_text("window.MONETA_CATEGORY_MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n",
                      encoding="utf-8")
    print(f"Trained {model['algorithm']} on {len(rows)} examples across {classes} categories.")
    print(f"Holdout accuracy: {accuracy:.1%} ({correct}/{len(valid_ids)}).")
    for row_id, prediction in zip(valid_ids, valid_predictions):
        if prediction != y[row_id]:
            print(f"  Review: {texts[row_id]!r} -> expected {labels_text[row_id]}, got {labels[prediction]}")
    print(f"Browser model written to {OUTPUT} ({OUTPUT.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
