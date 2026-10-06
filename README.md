# 📰 TruthLens — Hybrid Fake News Detection (GenAI + ML)

Type a claim, paste an article, or drop in a news link. TruthLens checks it in two ways:

- **GenAI fact check:** searches the live web and has an LLM judge the claim against what it finds, with sources.
- **ML style check:** a model trained on 44,000+ articles judges whether the writing *reads* like real or fake news.

The interface is styled as a newspaper front page: a blackletter masthead, Newsreader serif type, and each verdict stamped onto the page like an inked rubber stamp.

LIVE DEMO : https://fake-news-detector-truthlens.streamlit.app/

---

## 🧠 How It Works

```
User input
   │
   ├── Short claim (< 40 words) ─► Web search (Tavily) ─► LLM (Groq) judges using ONLY the results
   │                               → TRUE / FALSE / UNVERIFIED / OPINION + reason + sources
   │
   ├── News link (https://…) ────► Tavily Extract downloads the article text ─► checked as a full article
   │
   └── Full article (40+ words) ─┬► LLM extracts up to 3 key claims ─► each checked as above
                                 └► ML model (TF-IDF + Logistic Regression) scores writing style
                                    → combined final verdict
```

**Final verdict for an article**

| Fact check (GenAI) | Result |
|---|---|
| Any key claim is FALSE | **Likely Fake** |
| All key claims are TRUE | **Likely Real** (with a note if the writing looks sensational) |
| Claims can't be verified / API unavailable | The **ML writing-style** score decides |

**Why hybrid?**

- An **LLM alone** has a knowledge cutoff and can make things up. Here it never answers from memory. It only judges the search results it is given, and it cites them by number, so it cannot invent links.
- **ML alone** only knows writing style. It cannot check facts and fails on short claims.
- Together, GenAI checks the **facts** and ML checks the **style**. ML also keeps the app working when the APIs are down or out of free quota.

---

## ⚠️ Limitations

- **Breaking or very local news** often comes back UNVERIFIED, because reliable sources haven't covered it yet.
- **Opinions** ("Cricket is the best sport in the world") are labelled OPINION, not true or false.
- **Links** behind a paywall or login can't be read; paste the article text instead.
- The **ML model** was trained on 2016–17 US political news, so its style score is weaker on other topics. The app shows a warning when that happens.
- **Always open the sources** for important news.

---

## 🛠️ Technology Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.x |
| Frontend | Streamlit |
| LLM | Groq API (`openai/gpt-oss-120b` by default) |
| Web search | Tavily API |
| ML | Scikit-learn (TF-IDF + Logistic Regression) |
| NLP | NLTK (stopwords, lemmatization) |
| Data | Kaggle Fake and Real News (ISOT); optional WELFake |

---

## ✨ Features

- **Claim check** for short statements: verdict, one-line reason and source links
- **Article check:** extracts key claims and fact-checks each, plus writing-style analysis
- **Link check:** paste a news URL and the article text is downloaded and checked (paywalled pages ask you to paste the text instead)
- **One-click examples:** a true claim, a famous myth and an opinion, to see every kind of verdict
- **Explainable ML:** shows the words that pushed the score toward real or fake
- **Leakage-free training:** strips Reuters datelines, Getty credits and tweet links; removes duplicates before the split
- **Graceful fallback:** without API keys, or when the APIs fail, articles still get the ML result
- **Caching:** the same claim within an hour reuses the earlier answer, which saves free API quota

---

## ⚙️ Setup

### 1. Clone and install
```bash
git clone https://github.com/AnantJoshi17/fake-news-detector.git
cd fake-news-detector
git lfs pull          # downloads the trained model files
pip install -r requirements.txt
```

### 2. Add API keys (both have free tiers)
- Groq: https://console.groq.com/keys
- Tavily: https://app.tavily.com

Put them in `.streamlit/secrets.toml` (this file is in `.gitignore` and never pushed):
```toml
GROQ_API_KEY = "your-groq-key"
TAVILY_API_KEY = "your-tavily-key"
```

### 3. Run
```bash
streamlit run app.py
```

### Deploying on Streamlit Cloud
Open your app → **Settings → Secrets** and paste the same two lines.

### Retraining the ML model (optional)
1. Download `Fake.csv` and `True.csv` from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset (optionally `WELFake_Dataset.csv` from https://zenodo.org/record/4561253).
2. Put them in `data/`.
3. Run `python train.py`. It prints the metrics and the top words per class, then saves everything to `models/`.

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py              # Streamlit UI: routes claims vs articles, shows results
├── fact_checker.py     # GenAI: Tavily search + Groq LLM verdicts, final verdict rules
├── predictor.py        # ML: loads the model, input checks, style verdict + word drivers
├── text_utils.py       # Shared text cleaning and source-leakage stripping
├── train.py            # ML training script
├── requirements.txt
├── .streamlit/
│   ├── config.toml     # Newspaper colour theme
│   └── secrets.toml    # Your API keys (not committed)
└── models/
    ├── model.pkl         # Logistic Regression
    ├── vectorizer.pkl    # TF-IDF vocabulary
    ├── domain.pkl        # Off-topic check (created by train.py)
    └── metrics.json      # Test metrics shown in the app (created by train.py)
```

---

## SCREENSHOTS

<img width="1312" height="983" alt="Screenshot 2026-05-28 at 12 35 51 AM" src="https://github.com/user-attachments/assets/7a841e50-6aee-4c1b-8f6a-fd8bffa13876" />
<img width="3420" height="2050" alt="image" src="https://github.com/user-attachments/assets/19a8fe5f-ea84-4584-87ac-999daa419657" />

---

## 📄 License

This project is for academic purposes only.
