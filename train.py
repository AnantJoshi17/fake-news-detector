"""Train the TruthLens writing-style classifier.

Data (put in data/):
  Fake.csv, True.csv         Kaggle "Fake and Real News" (ISOT) dataset   [required]
  WELFake_Dataset.csv        WELFake, 72k articles from 4 sources          [optional]

Outputs (in models/):
  model.pkl, vectorizer.pkl  the classifier and its TF-IDF vocabulary
  domain.pkl                 centroid of the training data, used to warn on off-topic input
  metrics.json               test-set numbers shown in the app
"""
import json
import os
import pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from text_utils import clean_text, ensure_nltk

warnings.filterwarnings("ignore")

DATA = Path("data")
MODELS = Path("models")
MIN_TOKENS = 20          # drop near-empty articles (many Fake.csv rows have no body)
DOMAIN_PERCENTILE = 2    # inputs less similar than 98% of test articles get an off-topic warning
LEAK_TERMS = {"reuters", "getty", "featured image", "21wire", "pic twitter", "image via"}


def find_file(*names):
    for name in names:
        path = DATA / name
        if path.exists():
            return path
    return None


def load_isot():
    fake_path = find_file("Fake.csv", "fake.csv")
    true_path = find_file("True.csv", "true.csv")
    if not (fake_path and true_path):
        raise SystemExit(
            "Missing data/Fake.csv or data/True.csv. Download them from "
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )
    fake = pd.read_csv(fake_path)[["title", "text"]].assign(label=1)
    true = pd.read_csv(true_path)[["title", "text"]].assign(label=0)
    df = pd.concat([fake, true], ignore_index=True).assign(source="ISOT")
    print(f"  ISOT: {len(df)} articles")
    return df


def load_welfake():
    path = find_file("WELFake_Dataset.csv")
    if path is None:
        return None
    df = pd.read_csv(path)[["title", "text", "label"]].dropna(subset=["text"])
    # WELFake documents 0 = fake, 1 = real (opposite of ours: 1 = fake), and some
    # copies online are flipped. Check from the data: real news carries Reuters datelines.
    has_dateline = df["text"].str.contains(r"\(Reuters\)", case=False, na=False)
    rate = has_dateline.groupby(df["label"]).mean()
    real_label = rate.idxmax() if rate.max() > 2 * rate.min() + 0.01 else 1
    df["label"] = (df["label"] != real_label).astype(int)
    print(f"  WELFake: {len(df)} articles (treating original label {real_label} as REAL)")
    return df.assign(source="WELFake")


def main():
    print("=" * 50)
    print("TRUTHLENS — TRAINING")
    print("=" * 50)
    ensure_nltk()

    print("Loading datasets...")
    frames = [load_isot()]
    welfake = load_welfake()
    if welfake is not None:
        frames.append(welfake)
    df = pd.concat(frames, ignore_index=True)

    print("Cleaning text and stripping source artifacts (Reuters, Getty...) — a few minutes...")
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    before = len(df)
    df = df[df["content"].str.split().str.len() >= MIN_TOKENS]
    # The same article often appears twice (or in both datasets). Duplicates that land
    # in both train and test inflate accuracy, and copies with opposite labels are noise.
    conflicting = df.groupby("content")["label"].transform("nunique") > 1
    df = df[~conflicting].drop_duplicates(subset="content").reset_index(drop=True)
    counts = df["label"].value_counts()
    print(f"  Removed {before - len(df)} empty/duplicate articles")
    print(f"  Using {len(df)} | Real: {counts.get(0, 0)} | Fake: {counts.get(1, 0)}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    print("Vectorizing with TF-IDF (fit on training split only)...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["content"])
    X_test = vectorizer.transform(test_df["content"])
    y_train, y_test = train_df["label"], test_df["label"]
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "n_articles": int(len(df)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "datasets": sorted(df["source"].unique().tolist()),
        "source_artifacts_stripped": True,
        "trained_on": date.today().isoformat(),
    }
    print("\n--- RESULTS ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(classification_report(y_test, pred, target_names=["Real", "Fake"]))

    # Leakage check: the strongest words should describe writing, not publishers.
    names = vectorizer.get_feature_names_out()
    order = np.argsort(model.coef_[0])
    top_real, top_fake = names[order[:20]], names[order[-20:]][::-1]
    print("Top REAL words:", ", ".join(top_real))
    print("Top FAKE words:", ", ".join(top_fake))
    leaks = [w for w in list(top_real) + list(top_fake) if any(t in w for t in LEAK_TERMS)]
    if leaks:
        print(f"WARNING: source artifacts still among top features: {leaks}")

    # Domain reference: how similar a typical article is to the training data.
    centroid = np.asarray(X_train.mean(axis=0)).ravel()
    centroid /= np.linalg.norm(centroid)
    test_sims = X_test @ centroid
    domain = {
        "centroid": centroid.astype(np.float32),
        "threshold": float(np.percentile(test_sims, DOMAIN_PERCENTILE)),
    }

    os.makedirs(MODELS, exist_ok=True)
    with open(MODELS / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS / "domain.pkl", "wb") as f:
        pickle.dump(domain, f)
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model, vectorizer, domain.pkl and metrics.json to {MODELS}/")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()
