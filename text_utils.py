"""Text preprocessing shared by training (train.py) and inference (predictor.py).

Keeping one copy of the cleaning code guarantees the app transforms text exactly
the way the model saw it during training.
"""
import re

import nltk

NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

# Publisher boilerplate that reveals WHERE an article came from rather than
# WHAT it says. In the Kaggle (ISOT) dataset every real article is a Reuters
# wire story and many fake ones carry "Featured image via Getty Images", so a
# model trained on raw text learns "contains 'reuters' = real" instead of
# learning anything about the writing. These patterns remove those shortcuts.
_LEAKAGE_PATTERNS = [
    # Embedded tweet links and handles first, before credit lines eat the "pic." part
    re.compile(r"(?:pic\.)?twitter\.com/\S+", re.IGNORECASE),
    re.compile(r"@\w+"),
    # Wire datelines at the start: "WASHINGTON (Reuters) - ", "SEOUL/BEIJING (AP) — "
    re.compile(r"^.{0,80}?\((?:reuters|ap|afp)\)\s*[-–—]\s*", re.IGNORECASE),
    re.compile(r"\((?:reuters|ap|afp)\)", re.IGNORECASE),
    re.compile(r"\breuters\b", re.IGNORECASE),
    re.compile(r"\bgetty\s*images?\b", re.IGNORECASE),
    # "Featured image via ..." / "Photo by ..." credit lines (to end of sentence)
    re.compile(r"\bfeatured\s+image\b[^.\n]*", re.IGNORECASE),
    re.compile(r"\b(?:image|photo|screenshot|screen\s*capture)\s+(?:via|by|credit|courtesy)\b[^.\n]*",
               re.IGNORECASE),
    re.compile(r"\b21st\s+century\s+wire\b|\b21wire\b", re.IGNORECASE),
    # Template intro used on every Reuters "Trump tweets" article
    re.compile(r"the following statements were posted to the verified twitter accounts[^.]*\.",
               re.IGNORECASE),
]

_lemmatizer = None
_stop_words = None


def ensure_nltk():
    """Download NLTK data only if it is missing (avoids a network call on every start)."""
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.data.find(path + ".zip")   # wordnet/omw are kept zipped
            except LookupError:
                nltk.download(name, quiet=True)


def _resources():
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words("english"))
    return _lemmatizer, _stop_words


def strip_source_artifacts(text):
    """Remove publisher names, datelines and image credits (see _LEAKAGE_PATTERNS)."""
    text = str(text)
    for pattern in _LEAKAGE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def clean_text(text):
    lemmatizer, stop_words = _resources()
    text = strip_source_artifacts(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)
