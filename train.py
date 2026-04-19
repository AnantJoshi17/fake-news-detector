import re
import pickle
import warnings
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

warnings.filterwarnings("ignore")


def download_nltk():
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def load_data():
    print("Loading datasets...")
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/true.csv")
    fake["label"] = 1
    true["label"] = 0
    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    counts = df["label"].value_counts()
    print(f"  Total: {len(df)} | Real: {counts[0]} | Fake: {counts[1]}")
    return df


def main():
    print("=" * 50)
    print("FAKE NEWS DETECTOR — TRAINING")
    print("=" * 50)

    download_nltk()
    df = load_data()

    print("Cleaning text (this takes ~1-2 min)...")
    df["content"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    ).apply(clean_text)
    df = df[df["content"].str.strip() != ""]

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["content"])
    y = df["label"]
    print(f"  Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n--- RESULTS ---")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=["Real", "Fake"]))

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("\nModel saved to models/")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()