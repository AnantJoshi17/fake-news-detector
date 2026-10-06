"""Model loading, input guardrails and explanations for TruthLens.

The model classifies WRITING STYLE learned from 2016–17 news articles. It cannot
check whether a statement is true, so this module refuses inputs it has no basis
to judge (short claims, non-English text) instead of guessing.
"""
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from text_utils import clean_text

MODELS = Path(__file__).parent / "models"

MIN_WORDS = 40            # below this it is a headline or claim, not an article
MIN_KNOWN_FEATURES = 20   # TF-IDF features the model actually recognises
MIN_LATIN_RATIO = 0.8     # share of letters that are a–z (catches Hindi, etc.)
FAKE_THRESHOLD = 0.65     # prob_fake at or above → "Likely fake"
REAL_THRESHOLD = 0.35     # prob_fake at or below → "Likely real"; between → "Uncertain"
TOP_DRIVERS = 5

# Shown in the app header when models/metrics.json is missing (original model).
DEFAULT_METRICS = {"accuracy": 0.992, "n_articles": 44898, "n_features": 50000}


@dataclass
class Analysis:
    status: str                        # "ok" | "too_short" | "unsupported"
    message: str = ""
    verdict: str = ""                  # "real" | "fake" | "uncertain"
    prob_fake: float = 0.0
    prob_real: float = 0.0
    fake_drivers: list = field(default_factory=list)   # [(word, weight)]
    real_drivers: list = field(default_factory=list)
    off_topic: bool = False


class Predictor:
    def __init__(self, models_dir=MODELS):
        models_dir = Path(models_dir)
        with open(models_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(models_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.coef = self.model.coef_[0]
        fake_col = list(self.model.classes_).index(1)
        self.fake_col, self.real_col = fake_col, 1 - fake_col

        self.domain = None
        if (models_dir / "domain.pkl").exists():
            with open(models_dir / "domain.pkl", "rb") as f:
                self.domain = pickle.load(f)

        self.metrics = dict(DEFAULT_METRICS)
        if (models_dir / "metrics.json").exists():
            self.metrics.update(json.loads((models_dir / "metrics.json").read_text()))

    def analyse(self, text):
        words = text.split()
        letters = [c for c in text if c.isalpha()]
        latin = sum(1 for c in letters if c.isascii())
        if letters and latin / len(letters) < MIN_LATIN_RATIO:
            return Analysis("unsupported", "TruthLens currently supports English text only.")

        if len(words) < MIN_WORDS:
            return Analysis(
                "too_short",
                "This looks like a headline or a single claim. TruthLens judges the writing "
                "style of full news articles, so it can't tell whether a short statement is "
                f"true. Paste the complete article (at least {MIN_WORDS} words).",
            )

        features = self.vectorizer.transform([clean_text(text)])
        if features.nnz < MIN_KNOWN_FEATURES:
            return Analysis(
                "unsupported",
                "Too few words in this text match the news vocabulary the model was trained "
                "on, so any verdict would be a guess.",
            )

        probs = self.model.predict_proba(features)[0]
        prob_fake, prob_real = float(probs[self.fake_col]), float(probs[self.real_col])
        if prob_fake >= FAKE_THRESHOLD:
            verdict = "fake"
        elif prob_fake <= REAL_THRESHOLD:
            verdict = "real"
        else:
            verdict = "uncertain"

        fake_drivers, real_drivers = self._drivers(features)
        off_topic = False
        if self.domain is not None:
            similarity = float((features @ self.domain["centroid"])[0])
            off_topic = similarity < self.domain["threshold"]

        return Analysis("ok", verdict=verdict, prob_fake=prob_fake, prob_real=prob_real,
                        fake_drivers=fake_drivers, real_drivers=real_drivers,
                        off_topic=off_topic)

    def _drivers(self, features):
        """Words that pushed the score toward fake or real (TF-IDF value × weight)."""
        idx = features.indices
        contrib = features.data * self.coef[idx]
        if self.fake_col == 0:   # coef_ points toward classes_[1]
            contrib = -contrib
        order = np.argsort(contrib)
        fake = [(self.feature_names[idx[i]], float(contrib[i]))
                for i in order[::-1][:TOP_DRIVERS] if contrib[i] > 0]
        real = [(self.feature_names[idx[i]], float(-contrib[i]))
                for i in order[:TOP_DRIVERS] if contrib[i] < 0]
        return fake, real
