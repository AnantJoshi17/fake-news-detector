import re
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="TruthLens · Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #080c14 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2rem 1rem 4rem !important; max-width: 720px !important; }

[data-testid="stMain"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,210,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,210,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.hero { text-align: center; padding: 3.5rem 0 2.5rem; position: relative; }
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #63d2ff;
    background: rgba(99,210,255,0.08);
    border: 1px solid rgba(99,210,255,0.2);
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.8rem, 6vw, 4.2rem);
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0f4ff;
    margin-bottom: 0.5rem;
}
.hero h1 span {
    background: linear-gradient(135deg, #63d2ff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: rgba(180,195,230,0.7);
    margin-top: 1rem;
    letter-spacing: 0.01em;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 2rem;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: #f0f4ff;
}
.stat-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(180,195,230,0.45);
    margin-top: 2px;
}
.stat-divider { width: 1px; background: rgba(99,210,255,0.15); align-self: stretch; }

.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,210,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
}
.input-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(180,195,230,0.5);
    margin-bottom: 0.75rem;
    font-weight: 500;
}

textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(99,210,255,0.15) !important;
    border-radius: 10px !important;
    color: #d0daf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    caret-color: #63d2ff !important;
    resize: none !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease !important;
}
textarea:focus {
    border-color: rgba(99,210,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,210,255,0.06) !important;
    outline: none !important;
}
textarea::placeholder { color: rgba(120,140,180,0.4) !important; }
[data-testid="stTextArea"] label { display: none !important; }
[data-testid="stTextArea"] { margin: 0 !important; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #63d2ff 0%, #a78bfa 100%) !important;
    color: #080c14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    margin-top: 1rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.result-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeSlideUp 0.4s ease;
}
.result-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeSlideUp 0.4s ease;
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: -0.02em;
}
.result-real .result-label { color: #34d399; }
.result-fake .result-label { color: #f87171; }
.result-conf {
    font-size: 0.85rem;
    font-weight: 300;
    color: rgba(180,195,230,0.55);
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

.prob-section { margin: 1.25rem 0; animation: fadeSlideUp 0.5s ease 0.1s both; }
.prob-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.prob-tag {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(180,195,230,0.6);
}
.prob-pct { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.9rem; color: #f0f4ff; }
.bar-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 100px; overflow: hidden; margin-bottom: 1rem; }
.bar-fill-real { height: 100%; background: linear-gradient(90deg, #34d399, #6ee7b7); border-radius: 100px; }
.bar-fill-fake { height: 100%; background: linear-gradient(90deg, #f87171, #fca5a5); border-radius: 100px; }

.custom-divider { border: none; border-top: 1px solid rgba(99,210,255,0.08); margin: 1.5rem 0; }

.how-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(99,210,255,0.08);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    animation: fadeSlideUp 0.5s ease 0.2s both;
}
.how-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(99,210,255,0.7);
    margin-bottom: 1rem;
}
.how-step { display: flex; gap: 0.85rem; align-items: flex-start; margin-bottom: 0.75rem; }
.how-num {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #63d2ff;
    background: rgba(99,210,255,0.1);
    border: 1px solid rgba(99,210,255,0.2);
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}
.how-text { font-size: 0.85rem; font-weight: 300; color: rgba(180,195,230,0.65); line-height: 1.6; }

.footer {
    text-align: center;
    padding: 2rem 0 0;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(120,140,180,0.3);
}
.warn-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: rgba(251,191,36,0.8);
    font-weight: 300;
    margin: 1rem 0;
}

footer { display: none !important; }
#MainMenu { display: none; }

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def setup():
    for r in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(r, quiet=True)
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words("english"))
    return model, vectorizer, lemmatizer, sw

model, vectorizer, lemmatizer, stop_words = setup()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · NLP · 99.2% Accuracy</div>
    <h1>Truth<span>Lens</span></h1>
    <p class="hero-sub">Paste any news article. Get an instant authenticity verdict.</p>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-num">44,898</div>
            <div class="stat-label">Articles trained</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-num">99.2%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-num">50K</div>
            <div class="stat-label">TF-IDF features</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-card"><div class="input-label">Article Text</div>', unsafe_allow_html=True)
news_input = st.text_area("article", placeholder="Paste your news article or headline here...", height=200, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

THRESHOLD = 0.6
analyse = st.button("Analyse Article →")

if analyse:
    if not news_input.strip():
        st.markdown('<div class="warn-box">⚠ &nbsp; Please paste some text before analysing.</div>', unsafe_allow_html=True)
    elif len(news_input.strip().split()) < 5:
        st.markdown('<div class="warn-box">⚠ &nbsp; Please enter at least a few words for a reliable result.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analysing..."):
            cleaned = clean_text(news_input)
            features = vectorizer.transform([cleaned])
            probs = model.predict_proba(features)[0]

        prob_fake = probs[1]
        prob_real = probs[0]
        prediction = 1 if prob_fake > THRESHOLD else 0
        confidence = (prob_fake if prediction == 1 else prob_real) * 100

        if prediction == 0:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-icon">✓</div>
                <div class="result-label">Real News</div>
                <div class="result-conf">{confidence:.1f}% confidence · Below fake threshold</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-icon">⚠</div>
                <div class="result-label">Fake News</div>
                <div class="result-conf">{confidence:.1f}% confidence · Exceeds fake threshold</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-section">
            <div class="prob-row">
                <span class="prob-tag">Real probability</span>
                <span class="prob-pct">{prob_real*100:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill-real" style="width:{prob_real*100:.1f}%"></div></div>
            <div class="prob-row">
                <span class="prob-tag">Fake probability</span>
                <span class="prob-pct">{prob_fake*100:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill-fake" style="width:{prob_fake*100:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="how-section">
            <div class="how-title">How this result was produced</div>
            <div class="how-step">
                <div class="how-num">1</div>
                <div class="how-text">Text is lowercased, URLs removed, punctuation stripped, and words lemmatized</div>
            </div>
            <div class="how-step">
                <div class="how-num">2</div>
                <div class="how-text">Converted into a 50,000-feature TF-IDF vector with unigrams + bigrams</div>
            </div>
            <div class="how-step">
                <div class="how-num">3</div>
                <div class="how-text">Logistic Regression outputs fake vs real probabilities — fake threshold is 60%</div>
            </div>
            <div class="how-step">
                <div class="how-num">4</div>
                <div class="how-text">Trained on 2016–2018 political news. Results on other domains may vary.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr class="custom-divider">
<div class="footer">TruthLens · Streamlit · scikit-learn · NLTK</div>
""", unsafe_allow_html=True)