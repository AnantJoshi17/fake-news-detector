import re
from datetime import date
from html import escape
from urllib.parse import urlparse

import streamlit as st

from fact_checker import DEFAULT_MODEL, FactChecker, final_verdict
from predictor import FAKE_THRESHOLD, MIN_WORDS, REAL_THRESHOLD, Predictor

st.set_page_config(page_title="The TruthLens · Fact Checker", page_icon="🗞️", layout="centered")

EXAMPLES = [
    "India won the 2011 Cricket World Cup",
    "The Great Wall of China is visible from space",
    "Cricket is the best sport in the world",
]

# ---------------------------------------------------------------- styles
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,800;1,6..72,400;1,6..72,500&family=UnifrakturMaguntia&display=swap');

:root {
    --paper: #EEEAE0;
    --paper-light: #F7F4EC;
    --paper-deep: #E3DED1;
    --ink: #1C1A17;
    --ink-soft: #5A544C;
    --rule-soft: #C9C2B3;
    --red: #B42318;
    --green: #1E6B3F;
    --ochre: #8A6A10;
    --blue: #2E4A8B;
}

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--paper) !important;
    color: var(--ink);
    font-family: 'Newsreader', Georgia, serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 700px !important;
    padding: 1.25rem 1rem 4rem !important;
}
p, li { font-family: 'Newsreader', Georgia, serif; }

/* ---------- masthead ---------- */
.masthead { margin-bottom: 1.5rem; }
.dateline {
    display: flex; justify-content: space-between; gap: 1rem;
    font-style: italic; font-size: 0.9rem; color: var(--ink-soft);
    border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink);
    padding: 0.3rem 0;
}
.nameplate {
    font-family: 'UnifrakturMaguntia', 'Old English Text MT', serif !important;
    font-weight: 400 !important;
    font-size: clamp(3rem, 12vw, 5.4rem) !important;
    line-height: 1.05 !important;
    text-align: center; color: var(--ink) !important;
    margin: 0.6rem 0 0.2rem !important; padding: 0 !important;
    letter-spacing: 0;
}
.motto {
    text-align: center; font-style: italic; font-size: 1rem; color: var(--ink-soft);
    border-top: 1px solid var(--ink); border-bottom: 4px double var(--ink);
    padding: 0.35rem 0; margin: 0;
}
.deck {
    font-size: 1.28rem !important; line-height: 1.55 !important; color: var(--ink);
    margin: 1.4rem 0 1rem !important; max-width: 32em;
}

/* ---------- input ---------- */
[data-testid="stTextAreaRootElement"],
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextArea"] [data-baseweb="base-input"] {
    background: var(--paper-light) !important;
    border-color: var(--ink) !important;
    border-radius: 2px !important;
}
[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within {
    box-shadow: 0 0 0 1px var(--ink) !important;
}
[data-testid="stTextArea"] textarea {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.12rem !important; line-height: 1.6 !important;
    color: var(--ink) !important; caret-color: var(--red);
    background: transparent !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #8B847A !important; font-style: italic; }
[data-testid="InputInstructions"] { display: none !important; }

.st-key-check button {
    background: var(--ink) !important; color: var(--paper) !important;
    border: 0 !important; border-radius: 2px !important;
    padding: 0.6rem 1.7rem !important; min-height: 0 !important;
}
.st-key-check button p { font: 600 1.08rem/1.2 'Newsreader', Georgia, serif !important; color: var(--paper) !important; }
.st-key-check button:hover { background: #000 !important; }

.try { font-style: italic; color: var(--ink-soft); margin: 1.1rem 0 0.1rem; font-size: 0.98rem; }
[class*="st-key-ex_"] { margin-bottom: -0.9rem; }
[class*="st-key-ex_"] button {
    width: 100%; justify-content: flex-start !important;
    background: transparent !important; border: 0 !important;
    border-bottom: 1px solid var(--rule-soft) !important; border-radius: 0 !important;
    padding: 0.5rem 0 !important; min-height: 0 !important;
}
[class*="st-key-ex_"] button div { justify-content: flex-start !important; }
[class*="st-key-ex_"] button p {
    font: italic 500 1.08rem/1.4 'Newsreader', Georgia, serif !important;
    color: var(--ink) !important; text-align: left;
}
[class*="st-key-ex_"] button:hover p { color: var(--red) !important; }
[class*="st-key-ex_"] button:hover { border-bottom-color: var(--red) !important; }

button:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--red) !important; outline-offset: 3px !important;
}

/* ---------- report ---------- */
.report { margin-top: 2.6rem; border-top: 4px double var(--ink); padding-top: 1.2rem; }
.report-head {
    display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
}
.headline {
    font-family: 'Newsreader', Georgia, serif !important;
    font-style: italic; font-weight: 500 !important;
    font-size: clamp(1.6rem, 5vw, 2.2rem) !important; line-height: 1.2 !important;
    color: var(--ink) !important; margin: 0 !important; padding: 0 !important;
    flex: 1;
}
.byline { font-size: 0.92rem; color: var(--ink-soft); margin: 0.35rem 0 0; font-style: normal; }
.byline a { color: var(--ink-soft) !important; }
.reason { font-size: 1.22rem !important; line-height: 1.6 !important; margin: 1.2rem 0 0 !important; max-width: 32em; }

/* The rubber stamp: the one loud thing on the page */
.stamp {
    --stamp: var(--ink);
    flex-shrink: 0;
    display: inline-block;
    color: var(--stamp);
    border: 5px double var(--stamp);
    border-radius: 6px;
    padding: 0.35rem 1rem 0.3rem;
    font: 800 clamp(1.5rem, 5vw, 2.1rem)/1 'Newsreader', Georgia, serif;
    text-transform: uppercase; letter-spacing: 0.1em; white-space: nowrap;
    transform: rotate(-8deg);
    opacity: 0.95;
    margin-right: 0.6rem;
    -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='120'><filter id='f'><feTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='2' seed='4'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.3 1.45'/></filter><rect width='100%' height='100%' filter='url(%23f)'/></svg>");
    mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='120'><filter id='f'><feTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='2' seed='4'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.3 1.45'/></filter><rect width='100%' height='100%' filter='url(%23f)'/></svg>");
    animation: stamp-down 0.32s cubic-bezier(0.3, 0.7, 0.4, 1.3) both;
}
@keyframes stamp-down {
    from { transform: rotate(-8deg) scale(1.9); opacity: 0; }
    to   { transform: rotate(-8deg) scale(1);   opacity: 0.95; }
}
@media (prefers-reduced-motion: reduce) { .stamp { animation: none; } }
.stamp-green { --stamp: var(--green); }
.stamp-red   { --stamp: var(--red); }
.stamp-ochre { --stamp: var(--ochre); }
.stamp-blue  { --stamp: var(--blue); }

.mark {
    display: inline-block; font: 800 0.72rem/1 'Newsreader', Georgia, serif;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--stamp); border: 2px solid var(--stamp); border-radius: 3px;
    padding: 0.2rem 0.4rem; transform: rotate(-3deg); margin-right: 0.55rem;
    vertical-align: 0.15em;
}

.section-title {
    font: 600 1.15rem/1.3 'Newsreader', Georgia, serif !important;
    border-top: 1px solid var(--ink); padding-top: 0.6rem !important; margin: 2rem 0 0.4rem !important;
    color: var(--ink) !important;
}
.sources { margin: 0.5rem 0 0; padding-left: 1.3rem; }
.sources li { margin: 0.3rem 0; font-size: 1rem; line-height: 1.45; }
.sources a { color: var(--ink) !important; text-decoration: underline; text-decoration-color: var(--rule-soft); text-underline-offset: 3px; }
.sources a:hover { text-decoration-color: var(--red); }
.domain { color: var(--ink-soft); font-size: 0.88rem; font-style: italic; }

.claims { list-style: none; padding: 0 !important; margin: 0.4rem 0 0 !important; }
.claims > li { padding: 0.8rem 0; border-bottom: 1px solid var(--rule-soft); }
.claim-text { font-size: 1.1rem; font-weight: 500; line-height: 1.45; }
.claim-reason { font-size: 1rem; line-height: 1.55; color: var(--ink-soft); margin: 0.3rem 0 0; }
.claims .sources { margin-top: 0.2rem; }

.meter { height: 8px; background: var(--paper-deep); border: 1px solid var(--ink); margin: 0.6rem 0 0.8rem; max-width: 26rem; }
.meter span { display: block; height: 100%; background: var(--red); }
.style-line { font-size: 1.08rem; margin: 0.2rem 0; }
.words { font-size: 1rem; color: var(--ink-soft); margin: 0.25rem 0; line-height: 1.5; }
.words b { color: var(--ink); font-weight: 600; }

.note {
    border-left: 3px solid var(--ink); padding: 0.2rem 0 0.2rem 0.9rem;
    font-size: 1.05rem; line-height: 1.55; margin: 1.4rem 0; font-style: italic;
}
.note-warn { border-left-color: var(--red); }
.note code { font-style: normal; font-size: 0.9em; background: var(--paper-deep); padding: 0 0.25em; }

.method {
    background: var(--paper-deep); padding: 1rem 1.2rem; margin-top: 2.2rem;
    font-size: 0.98rem; line-height: 1.55;
}
.method p { margin: 0 0 0.4rem; font-weight: 600; }
.method ol { margin: 0; padding-left: 1.2rem; }
.method li { margin: 0.25rem 0; }

.colophon {
    margin-top: 4rem; border-top: 1px solid var(--ink); padding-top: 0.6rem;
    font-size: 0.88rem; line-height: 1.5; color: var(--ink-soft); font-style: italic;
}

@media (max-width: 560px) {
    .report-head { flex-direction: column; align-items: flex-start; gap: 1rem; }
    .stamp { margin-left: 0.3rem; }
    .deck { font-size: 1.1rem; }
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------- models
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_article(url):
    return fact_checker.fetch_article(url)


# ---------------------------------------------------------------- display helpers
CLAIM_STAMPS = {
    "TRUE": ("stamp-green", "True"),
    "FALSE": ("stamp-red", "False"),
    "UNVERIFIED": ("stamp-ochre", "Unverified"),
    "OPINION": ("stamp-blue", "Opinion"),
}
ARTICLE_STAMPS = {
    "real": ("stamp-green", "Likely real"),
    "fake": ("stamp-red", "Likely fake"),
    "uncertain": ("stamp-ochre", "Uncertain"),
}
NO_KEYS = ("Fact checking needs the Groq and Tavily keys in <code>.streamlit/secrets.toml</code> "
           "(see the README).")


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def note(text, warn=False):
    html(f'<div class="note{" note-warn" if warn else ""}">{text}</div>')


def domain(url):
    return urlparse(url).netloc.removeprefix("www.")


def source_list(sources):
    if not sources:
        return ""
    items = "".join(
        f'<li><a href="{escape(s["url"], quote=True)}" target="_blank" rel="noopener">{escape(s["title"])}</a> '
        f'<span class="domain">{escape(domain(s["url"]))}</span></li>'
        for s in sources
    )
    return f'<ol class="sources">{items}</ol>'


def report_head(headline, stamp_class, stamp_text, byline=""):
    return (f'<div class="report-head"><div><h2 class="headline">{headline}</h2>{byline}</div>'
            f'<div class="stamp {stamp_class}" role="img" aria-label="Verdict: {stamp_text}">{stamp_text}</div></div>')


def show_method():
    steps = [
        f"Short claims (under {MIN_WORDS} words) are searched on the web with Tavily. An LLM on Groq "
        "judges the claim using only those search results and cites them, so it can't invent sources.",
        "For articles and links, the LLM first picks out up to three key factual claims, and each one is "
        "checked the same way.",
        "A machine-learning model (TF-IDF + Logistic Regression) scores the writing style: "
        f"{FAKE_THRESHOLD:.0%} or more fake-like reads as fake, {REAL_THRESHOLD:.0%} or less as real.",
        "Verdict: one false key claim makes an article likely fake; all key claims supported makes it "
        "likely real; otherwise the writing style decides.",
    ]
    items = "".join(f"<li>{escape(s)}</li>" for s in steps)
    html(f'<div class="method"><p>How TruthLens checked this</p><ol>{items}</ol></div>')


def show_style(result):
    fake_pct = result.prob_fake * 100
    fake_words = ", ".join(escape(w) for w, _ in result.fake_drivers) or "none"
    real_words = ", ".join(escape(w) for w, _ in result.real_drivers) or "none"
    html(
        '<h3 class="section-title">Writing style</h3>'
        f'<p class="style-line">The writing is <b>{fake_pct:.0f}%</b> similar to fake news.</p>'
        f'<div class="meter" role="img" aria-label="{fake_pct:.0f}% similar to fake news">'
        f'<span style="width:{fake_pct:.1f}%"></span></div>'
        f'<p class="words"><b>Words that read like fake news:</b> {fake_words}</p>'
        f'<p class="words"><b>Words that read like real news:</b> {real_words}</p>'
    )
    if result.off_topic:
        note("This article's topic is far from the model's training data (2016–17 US politics), "
             "so treat the writing-style score with extra caution.", warn=True)


# ---------------------------------------------------------------- checks
def run_claim_check(claim):
    if fact_checker is None:
        note(f"That looks like a short claim. {NO_KEYS} Without them, only full articles of "
             f"{MIN_WORDS}+ words can be analysed.")
        return
    try:
        with st.spinner("Searching the web for evidence…"):
            result = check_claim(claim)
    except Exception:
        note("The fact-check service didn't respond. Wait a minute and press Check it again.", warn=True)
        return
    stamp_class, stamp_text = CLAIM_STAMPS[result["verdict"]]
    reason = escape(result["reason"] or "The model did not give a reason.")
    sources = source_list(result["sources"])
    html(
        '<section class="report">'
        + report_head(f"“{escape(claim)}”", stamp_class, stamp_text)
        + f'<p class="reason">{reason}</p>'
        + (f'<h3 class="section-title">Sources</h3>{sources}' if sources else "")
        + "</section>"
    )
    show_method()


def run_article_check(article, url=None):
    fact, fact_failed = None, False
    with st.spinner("Checking the key claims and reading the writing style…"):
        style = predictor.analyse(article)
        if fact_checker is not None:
            try:
                fact = check_article(article)
            except Exception:
                fact_failed = True

    style_verdict = style.verdict if style.status == "ok" else None
    fact_verdict = fact["verdict"] if fact else None
    if fact_verdict is None and style_verdict is None:
        note(escape(style.message) + (" " + NO_KEYS if fact_checker is None else ""))
        return

    verdict, why = final_verdict(fact_verdict, style_verdict)
    stamp_class, stamp_text = ARTICLE_STAMPS[verdict]
    if url:
        safe_url = escape(url, quote=True)
        headline = f"The article on {escape(domain(url))}"
        byline = f'<p class="byline"><a href="{safe_url}" target="_blank" rel="noopener">Open the original</a></p>'
    else:
        headline, byline = "The article you pasted", ""
    html('<section class="report">' + report_head(headline, stamp_class, stamp_text, byline)
         + f'<p class="reason">{escape(why)}</p></section>')

    if fact_failed:
        note("The fact-check service didn't respond, so this verdict uses the writing style only. "
             "Press Check it again in a minute for the full check.", warn=True)
    elif fact_checker is None:
        note(f"{NO_KEYS} With them, the article's key claims are fact-checked too.")

    if fact and fact["claims"]:
        rows = ""
        for c in fact["claims"]:
            cls, label = CLAIM_STAMPS[c["verdict"]]
            rows += (f'<li><span class="mark {cls}">{label}</span><span class="claim-text">{escape(c["claim"])}</span>'
                     f'<p class="claim-reason">{escape(c["reason"])}</p>{source_list(c["sources"])}</li>')
        html(f'<h3 class="section-title">Key claims checked</h3><ul class="claims">{rows}</ul>')
    elif fact:
        note("No checkable factual claims were found in this article.")

    if style.status == "ok":
        show_style(style)
    else:
        note(f"Writing-style analysis skipped: {escape(style.message)}")
    show_method()


def run_link_check(url):
    if fact_checker is None:
        note(f"Checking a link needs the Tavily key to download the article. {NO_KEYS}")
        return
    try:
        with st.spinner(f"Reading the article on {domain(url)}…"):
            text = fetch_article(url)
    except Exception:
        note("Couldn't open that link. Check the address, or paste the article text instead.", warn=True)
        return
    if len(text.split()) < MIN_WORDS:
        note("Couldn't read enough text from that page (it may be behind a paywall or login). "
             "Paste the article text instead.", warn=True)
        return
    run_article_check(text[:20000], url=url)


# ---------------------------------------------------------------- page
today = date.today()
html(f"""
<header class="masthead">
  <div class="dateline"><span>The fact-checking desk</span><span>{today:%A}, {today.day} {today:%B %Y}</span></div>
  <h1 class="nameplate">The TruthLens</h1>
  <p class="motto">All the claims fit to check</p>
</header>
<p class="deck">Type a claim, paste a news article, or drop in a link. TruthLens searches the web
for evidence, gives a verdict, and shows you its sources.</p>
""")

if "query" not in st.session_state:
    st.session_state.query = ""


def use_example(text):
    st.session_state.query = text
    st.session_state.run_now = True


st.text_area(
    "Claim, article or link",
    key="query",
    placeholder="e.g. The Great Wall of China is visible from space",
    height=150,
    label_visibility="collapsed",
)
check = st.button("Check it", key="check")

html('<p class="try">Or try one of these:</p>')
for i, example in enumerate(EXAMPLES):
    st.button(f"“{example}”", key=f"ex_{i}", on_click=use_example, args=(example,))

if check or st.session_state.pop("run_now", False):
    text = st.session_state.query.strip()
    if text.startswith("www."):
        text = "https://" + text
    if not text:
        note("Type a claim, paste an article or add a link first.")
    elif re.fullmatch(r"https?://\S+", text):
        run_link_check(text)
    elif len(text.split()) < MIN_WORDS:
        run_claim_check(text)
    else:
        run_article_check(text)

html(f"""
<p class="colophon">Fact checks use live web search (Tavily) and an LLM (Groq). The writing-style model is
TF-IDF with Logistic Regression, trained on {metrics['n_articles']:,} news articles
({metrics['accuracy'] * 100:.1f}% test accuracy). Always open the sources for news that matters.</p>
""")
