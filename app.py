from html import escape

import streamlit as st

from fact_checker import DEFAULT_MODEL, FactChecker, final_verdict
from predictor import FAKE_THRESHOLD, MIN_WORDS, REAL_THRESHOLD, Predictor

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

.result-uncertain {
    background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,191,36,0.03));
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeSlideUp 0.4s ease;
}
.result-uncertain .result-label { color: #fbbf24; }
.result-note {
    font-size: 0.85rem;
    font-weight: 300;
    color: rgba(180,195,230,0.7);
    margin-top: 0.75rem;
    line-height: 1.6;
}

.info-box, .scope-box {
    background: rgba(99,210,255,0.05);
    border: 1px solid rgba(99,210,255,0.18);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: rgba(200,220,250,0.8);
    font-weight: 300;
    line-height: 1.6;
    margin: 1rem 0;
}
.scope-box { background: rgba(255,255,255,0.02); border-color: rgba(99,210,255,0.1); margin-top: 0; }
.scope-box b, .info-box b { font-weight: 500; color: #d0daf0; }

.drivers {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1rem 0;
    animation: fadeSlideUp 0.5s ease 0.15s both;
}
.driver-title {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(180,195,230,0.55);
    margin-bottom: 0.5rem;
}
.chip {
    display: inline-block;
    font-size: 0.8rem;
    padding: 3px 10px;
    border-radius: 100px;
    margin: 0 6px 6px 0;
}
.chip-fake { color: #fca5a5; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); }
.chip-real { color: #6ee7b7; background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); }
.chip-none { font-size: 0.8rem; color: rgba(180,195,230,0.4); }

.result-opinion {
    background: linear-gradient(135deg, rgba(167,139,250,0.08), rgba(167,139,250,0.03));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeSlideUp 0.4s ease;
}
.result-opinion .result-label { color: #c4b5fd; }
.mode-tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(180,195,230,0.55);
    margin-bottom: 0.75rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(99,210,255,0.7);
    margin: 2rem 0 0.75rem;
}
.sources { margin-top: 1rem; text-align: left; }
.sources a, .claim-row a {
    display: block;
    font-size: 0.82rem;
    color: #63d2ff !important;
    text-decoration: none;
    margin: 0.3rem 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.sources a:hover, .claim-row a:hover { text-decoration: underline; }
.claim-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(99,210,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    animation: fadeSlideUp 0.4s ease;
}
.claim-text { font-size: 0.9rem; color: #d0daf0; line-height: 1.5; margin: 0.4rem 0; }
.claim-reason { font-size: 0.82rem; font-weight: 300; color: rgba(180,195,230,0.65); line-height: 1.5; }
.pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 2px 10px;
    border-radius: 100px;
}
.pill-TRUE { color: #34d399; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); }
.pill-FALSE { color: #f87171; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); }
.pill-UNVERIFIED { color: #fbbf24; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); }
.pill-OPINION { color: #c4b5fd; background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.3); }
@media (max-width: 560px) {
    .drivers { grid-template-columns: 1fr; }
    .hero-stats { gap: 1.25rem; }
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
def load_predictor():
    return Predictor()


def secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:  # no secrets.toml at all
        return ""


@st.cache_resource
def load_fact_checker():
    groq_key, tavily_key = secret("GROQ_API_KEY"), secret("TAVILY_API_KEY")
    if not (groq_key and tavily_key):
        return None
    return FactChecker(groq_key, tavily_key, model=secret("GROQ_MODEL") or DEFAULT_MODEL)


predictor = load_predictor()
fact_checker = load_fact_checker()
metrics = predictor.metrics


# Cache answers for an hour so repeated checks don't use up free API limits.
@st.cache_data(ttl=3600, show_spinner=False)
def check_claim(claim):
    return fact_checker.check_claim(claim)


@st.cache_data(ttl=3600, show_spinner=False)
def check_article(article):
    return fact_checker.check_article(article)


st.markdown(f"""
<div class="hero">
    <div class="hero-badge">GenAI + ML · Fact Check</div>
    <h1>Truth<span>Lens</span></h1>
    <p class="hero-sub">Type a claim or paste a full article. TruthLens checks the facts against live web sources and analyses the writing style.</p>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-num">Live</div>
            <div class="stat-label">Web fact check</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-num">{metrics['n_articles']:,}</div>
            <div class="stat-label">Articles trained</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-num">{metrics['accuracy'] * 100:.1f}%</div>
            <div class="stat-label">ML test accuracy</div>
        </div>
    </div>
</div>
<div class="scope-box">
    <b>How it works.</b> Short claims are fact-checked by an LLM using live web search results.
    Full articles get both: their key claims are fact-checked, and an ML model analyses the
    writing style. Every fact check lists its sources, so open them for important news.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-card"><div class="input-label">Claim or Article</div>', unsafe_allow_html=True)
news_input = st.text_area(
    "article",
    placeholder='Type a claim (e.g. "Modi is PM of India") or paste a full news article...',
    height=200,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

analyse = st.button("Check →")

VERDICT_CARDS = {   # combined / ML verdicts
    "real": ("result-real", "✓", "Likely Real"),
    "fake": ("result-fake", "⚠", "Likely Fake"),
    "uncertain": ("result-uncertain", "?", "Uncertain"),
}
CLAIM_CARDS = {     # GenAI claim verdicts
    "TRUE": ("result-real", "✓", "True"),
    "FALSE": ("result-fake", "✗", "False"),
    "UNVERIFIED": ("result-uncertain", "?", "Unverified"),
    "OPINION": ("result-opinion", "💬", "Opinion"),
}
NO_KEYS = ("Fact checking needs the Groq and Tavily API keys in <b>.streamlit/secrets.toml</b> "
           "(see the README).")


def box(kind, html):
    st.markdown(f'<div class="{kind}">{html}</div>', unsafe_allow_html=True)


def card(css, icon, label, note, mode, extra=""):
    st.markdown(
        f'<div class="{css}"><div class="mode-tag">{mode}</div><div class="result-icon">{icon}</div>'
        f'<div class="result-label">{label}</div><div class="result-note">{escape(note)}</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def source_links(sources):
    return "".join(
        f'<a href="{escape(s["url"], quote=True)}" target="_blank" rel="noopener">↗ {escape(s["title"])}</a>'
        for s in sources
    )


def chips(drivers, kind):
    if not drivers:
        return '<span class="chip-none">None</span>'
    return "".join(f'<span class="chip chip-{kind}">{escape(word)}</span>' for word, _ in drivers)


def show_claims(claims):
    st.markdown('<div class="section-title">Fact check of key claims · GenAI</div>', unsafe_allow_html=True)
    for c in claims:
        st.markdown(
            f'<div class="claim-row"><span class="pill pill-{c["verdict"]}">{c["verdict"]}</span>'
            f'<div class="claim-text">{escape(c["claim"])}</div>'
            f'<div class="claim-reason">{escape(c["reason"])}</div>{source_links(c["sources"])}</div>',
            unsafe_allow_html=True,
        )


def show_style(result):
    st.markdown('<div class="section-title">Writing style · ML model</div>', unsafe_allow_html=True)
    if result.off_topic:
        box("warn-box", "⚠ &nbsp; This article's topic looks different from the ML training data "
                        "(2016–17 US politics), so treat the style score with extra caution.")
    st.markdown(
        f'<div class="prob-section">'
        f'<div class="prob-row"><span class="prob-tag">Real-style probability</span>'
        f'<span class="prob-pct">{result.prob_real*100:.1f}%</span></div>'
        f'<div class="bar-track"><div class="bar-fill-real" style="width:{result.prob_real*100:.1f}%"></div></div>'
        f'<div class="prob-row"><span class="prob-tag">Fake-style probability</span>'
        f'<span class="prob-pct">{result.prob_fake*100:.1f}%</span></div>'
        f'<div class="bar-track"><div class="bar-fill-fake" style="width:{result.prob_fake*100:.1f}%"></div></div>'
        f'</div>'
        f'<div class="drivers">'
        f'<div><div class="driver-title">Words pushing toward fake</div>{chips(result.fake_drivers, "fake")}</div>'
        f'<div><div class="driver-title">Words pushing toward real</div>{chips(result.real_drivers, "real")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def show_how_it_works():
    steps = [
        f"Short claims (under {MIN_WORDS} words) are searched on the web with Tavily. An LLM on Groq "
        "judges the claim using only those results and cites them by number, so it can't invent sources.",
        "For full articles, the LLM first picks up to 3 key factual claims, and each is checked the same way.",
        "An ML model (TF-IDF + Logistic Regression) scores the writing style: "
        f"≥{FAKE_THRESHOLD:.0%} fake-style → fake, ≤{REAL_THRESHOLD:.0%} → real, otherwise uncertain.",
        "Final verdict: a false key claim → Likely Fake; all key claims supported → Likely Real; "
        "otherwise the writing-style score decides.",
    ]
    rows = "".join(
        f'<div class="how-step"><div class="how-num">{i}</div><div class="how-text">{escape(s)}</div></div>'
        for i, s in enumerate(steps, 1)
    )
    st.markdown(f'<div class="how-section"><div class="how-title">How this result was produced</div>{rows}</div>',
                unsafe_allow_html=True)


def run_claim_check(claim):
    if fact_checker is None:
        box("info-box", f"ℹ &nbsp; This looks like a short claim. {NO_KEYS} The ML model only "
                        f"analyses full articles of {MIN_WORDS}+ words.")
        return
    try:
        with st.spinner("Searching the web and checking the claim..."):
            result = check_claim(claim)
    except Exception:
        box("warn-box", "⚠ &nbsp; The fact-check service is busy or unreachable right now. "
                        "Please try again in a minute.")
        return
    css, icon, label = CLAIM_CARDS[result["verdict"]]
    reason = result["reason"] or "The model did not give a reason."
    extra = f'<div class="sources">{source_links(result["sources"])}</div>' if result["sources"] else ""
    card(css, icon, label, reason, "Fact check · GenAI + web search", extra)
    show_how_it_works()


def run_article_check(article):
    fact, fact_failed = None, False
    with st.spinner("Checking key claims and analysing the writing style..."):
        style = predictor.analyse(article)
        if fact_checker is not None:
            try:
                fact = check_article(article)
            except Exception:
                fact_failed = True

    style_verdict = style.verdict if style.status == "ok" else None
    fact_verdict = fact["verdict"] if fact else None
    if fact_verdict is None and style_verdict is None:
        message = escape(style.message)
        if fact_checker is None:
            message += " " + NO_KEYS
        box("info-box", f"ℹ &nbsp; {message}")
        return

    verdict, why = final_verdict(fact_verdict, style_verdict)
    css, icon, label = VERDICT_CARDS[verdict]
    mode = "Overall verdict · GenAI + ML" if fact else "Writing style · ML only"
    card(css, icon, label, why, mode)

    if fact_failed:
        box("warn-box", "⚠ &nbsp; The fact-check service is busy or unreachable right now, "
                        "so this result uses the writing style only.")
    elif fact_checker is None:
        box("info-box", f"ℹ &nbsp; {NO_KEYS} With keys added, the article's claims are fact-checked too.")

    if fact and fact["claims"]:
        show_claims(fact["claims"])
    elif fact:
        box("info-box", "ℹ &nbsp; No checkable factual claims were found in this article.")

    if style.status == "ok":
        show_style(style)
    else:
        box("info-box", f"ℹ &nbsp; Writing-style analysis skipped: {escape(style.message)}")
    show_how_it_works()


if analyse:
    text = news_input.strip()
    if not text:
        box("warn-box", "⚠ &nbsp; Please type a claim or paste an article first.")
    elif len(text.split()) < MIN_WORDS:
        run_claim_check(text)
    else:
        run_article_check(text)

st.markdown("""
<hr class="custom-divider">
<div class="footer">TruthLens · Streamlit · Groq · Tavily · scikit-learn · NLTK</div>
""", unsafe_allow_html=True)
