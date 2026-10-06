# 📰 TruthLens — Hybrid Fake News Detection (GenAI + ML)

**Live demo:** https://fake-news-detector-truthlens.streamlit.app/

Type a claim, paste an article, or drop in a news link. TruthLens checks it in two ways:

- **GenAI fact check:** searches the live web and has an LLM judge the claim using only what it finds, with cited sources.
- **ML style check:** a model trained on 44,000+ news articles scores whether the writing *reads* like real or fake news.

---

## 🧠 Architecture

```
                         ┌──────────────────────┐
                         │     User's browser   │
                         │ claim · article · URL│
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ app.py · input router│
                         │ < 40 words → claim   │
                         │ URL / 40+ → article  │
                         └─────┬──────────┬─────┘
                 every input   │          │   articles & links only
                               ▼          ▼
 ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
 │ fact_checker.py · GenAI         │  │ predictor.py · ML style model   │
 │                                 │  │                                 │
 │  Tavily Search  → top 4 results │  │  text_utils.py → clean text,    │
 │  Groq LLM       → picks claims, │  │                  strip leakage  │
 │                   judges them   │  │  TF-IDF        → 50,000 features│
 │                   (JSON output) │  │  Logistic Reg. → fake-style %   │
 │  Tavily Extract → URL → text    │  │                                 │
 └────────────────┬────────────────┘  └────────────────┬────────────────┘
                  │ claim verdicts + sources           │ style score + key words
                  └──────────────────┬─────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │ final_verdict() · one stamp  │
                      │ any claim FALSE → Likely fake│
                      │ all claims TRUE → Likely real│
                      │ otherwise → ML style decides │
                      └──────────────┬───────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │ Result: verdict stamp, reason│
                      │ sources, style words         │
                      └──────────────────────────────┘
```

### Verdict rules

| Input | What runs | Possible verdicts |
|---|---|---|
| Short claim (< 40 words) | Web search + LLM | TRUE · FALSE · UNVERIFIED · OPINION |
| Full article (40+ words) | LLM checks up to 3 key claims + ML style score | Likely real · Likely fake · Uncertain |
| News link | Tavily Extract downloads the text, then checked as an article | Likely real · Likely fake · Uncertain |

For articles, a FALSE key claim always means **Likely fake**. If the claims can't be verified (or the APIs are down), the ML style score decides: ≥ 65% fake-style → Likely fake, ≤ 35% → Likely real, otherwise Uncertain.

### Why hybrid?

- An **LLM alone** has a knowledge cutoff and can make things up. Here it never answers from memory: it judges only the search results it is given and cites them by number, so it cannot invent sources.
- **ML alone** only learns writing style. It cannot check facts and fails on short claims.
- Together, GenAI checks the **facts** and ML checks the **style**. ML also keeps the app working when the APIs are unavailable.

### ML training pipeline

```
Kaggle ISOT dataset (44,898 articles)
  → strip source leakage (Reuters datelines, Getty credits, tweet links)
  → clean text (lowercase, stopwords, lemmatization)
  → remove duplicates
  → 80/20 stratified split
  → TF-IDF (50,000 unigram + bigram features, fit on train only)
  → Logistic Regression (class-balanced)
  → model.pkl + vectorizer.pkl
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Frontend | Streamlit (custom newspaper-style UI) |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Web search & extraction | Tavily Search + Tavily Extract |
| Machine learning | scikit-learn (TF-IDF, Logistic Regression) |
| NLP | NLTK (stopwords, lemmatization) |
| Dataset | Kaggle Fake and Real News (ISOT) |
| Deployment | Streamlit Community Cloud |

---

## ✨ Features

- **Claim check** with a verdict, a one-line reason and source links
- **Article check:** fact-checks the key claims and analyses the writing style
- **Link check:** paste a news URL and the article is fetched and checked
- **One-click examples:** a true claim, a famous myth and an opinion
- **Explainable ML:** shows the words that pushed the score toward real or fake
- **Graceful fallback:** if the APIs fail, articles still get the ML result
- **Caching:** repeated checks within an hour reuse the earlier answer

---

## ⚠️ Limitations

- Breaking or very local news often comes back **UNVERIFIED** until reliable sources cover it.
- Opinions are labelled **OPINION**, not true or false.
- Paywalled links can't be read; paste the article text instead.
- The ML model was trained on 2016–17 US political news, so its style score is weaker on other topics.

---

## 📸 Screenshots

<img width="1369" alt="TruthLens home" src="https://github.com/user-attachments/assets/a6a8d467-5afc-4954-8e97-cdcfc4676aba" />
<img width="1369" alt="TruthLens claim check" src="https://github.com/user-attachments/assets/9eccf66d-3abc-42c8-a988-40b171fd0140" />
<img width="1369" alt="TruthLens article check" src="https://github.com/user-attachments/assets/ad594664-9a9d-422d-a639-5f265aa3fe3a" />

---

## 📄 License

© 2026 Anant Joshi. All rights reserved. This project is shared for viewing and portfolio purposes only; copying, redistribution or reuse of the code requires permission.

