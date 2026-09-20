"""Small, dependency-free visual primitives for the CrisisLens interface."""
from html import escape
import math
import re
import streamlit as st

ACCENTS = {"CRITICAL": "#ff787e", "HIGH": "#ffb369", "MEDIUM": "#e9cf73", "LOW": "#72d5b0"}
ICONS = {"RESCUE_BOAT": "◈", "RESCUE_TEAM": "◈", "MEDICAL_TEAM": "✚", "AMBULANCE": "✚",
         "DRINKING_WATER": "◉", "FOOD": "◉", "SHELTER": "⌂", "POWER_RESTORATION": "ϟ"}


def display_label(value):
    """Humanize canonical labels without changing storage or free-text evidence."""
    return str(value).replace("_", " ").title() if value is not None and str(value).strip() else "Not identified"


def human_text(value):
    """Only humanize enum-like tokens within a stored explanation."""
    return re.sub(r"\b[A-Z]+(?:_[A-Z]+)+\b", lambda match: display_label(match[0]), str(value or ""))


def concise(value, limit=155):
    text = human_text(value)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def chips(values, resources=False):
    return " ".join(f'<span class="cl-chip">{ICONS.get(v, "◇") + " " if resources else ""}{escape(display_label(v))}</span>'
                    for v in values or []) or '<span class="cl-muted">Not identified</span>'


def priority_badge(level):
    color = ACCENTS.get(level, "#a5b6ca")
    return f'<span class="cl-badge" style="--accent:{color}">{escape(display_label(level))}</span>'


def card(label, value, note="", accent=None):
    st.markdown(f'<div class="cl-card" style="--accent:{accent or "#76d8e7"}">'
                f'<div class="cl-eyebrow">{escape(label)}</div><div class="cl-value">{escape(str(value))}</div>'
                f'<div class="cl-muted">{escape(note)}</div></div>', unsafe_allow_html=True)


def meter(label, value, maximum, color="#76d8e7"):
    """Scale stored numeric values for display; never infer missing values."""
    valid = finite(value) and finite(maximum) and maximum > 0
    amount = min(100, max(0, value / maximum * 100)) if valid else 0
    reading = f"{value:g} / {maximum:g}" if valid else "Not identified"
    st.markdown(f'<div class="cl-meter"><div class="cl-meter-label"><span>{escape(label)}</span>'
                f'<strong>{reading}</strong></div><div class="cl-track" role="img" aria-label="{escape(label)}: {reading}">'
                f'<div style="width:{amount}%;background:{color}"></div></div></div>', unsafe_allow_html=True)


def apply_theme():
    st.markdown("""<style>
    .stApp {background:#090f1b;color:#e7edf6;color-scheme:dark;}
    [data-testid="stHeader"] {background:#090f1b;}
    [data-testid="stSidebar"] {background:#101a29;border-right:1px solid #26374b;}
    .block-container {padding-top:2rem;padding-bottom:2rem;max-width:1500px;}
    h1,h2,h3,h4 {color:#f2f6fc!important;letter-spacing:-.025em;}
    h1 {font-size:2.3rem!important;} h2 {font-size:1.4rem!important;}
    h3 {font-size:1.15rem!important;}
    [data-testid="stCaptionContainer"] {color:#adbbce;}
    [data-testid="stMetric"] {background:#121e30;border:1px solid #2a3b52;border-radius:12px;padding:16px;}
    [data-testid="stMetricValue"] {color:#f0f6ff;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-color:#2a3b52!important;}
    [data-baseweb="tab-list"] {gap:16px;border-bottom:1px solid #293a50;}
    [data-baseweb="tab"] {color:#b9c8db;background:transparent;font-weight:600;}
    [aria-selected="true"][data-baseweb="tab"] {color:#8be7ef;}
    [data-baseweb="tab-highlight"] {background:#76d8e7;}
    .stButton button {background:#183247;color:#d8faff;border:1px solid #3c6979;border-radius:8px;}
    .stButton button:hover {border-color:#92e7ef;color:#fff;background:#204256;}
    .stButton button:focus-visible {outline:2px solid #92e7ef;outline-offset:3px;}
    [data-testid="stTextArea"] textarea,[data-baseweb="select"]>div {
      background:#142238!important;color:#edf3fb!important;border-color:#40546e!important;}
    [data-baseweb="popover"],[role="listbox"],[role="option"] {background:#142238!important;color:#edf3fb!important;}
    [data-testid="stExpander"] {background:#101b2c;border-radius:10px;}
    [data-testid="stFileUploaderDropzone"] {background:#152339;color:#dce6f4;}
    .cl-card {background:linear-gradient(140deg,#152338,#101b2b);border:1px solid #2b4057;
      border-top:2px solid var(--accent);border-radius:12px;padding:17px 19px;min-height:115px;overflow-wrap:anywhere;}
    .cl-eyebrow {font-size:.76rem;letter-spacing:.10em;text-transform:uppercase;color:#aebfd4;margin-bottom:8px;font-weight:650;}
    .cl-value {font-size:1.55rem;font-weight:650;color:#f1f6fc;line-height:1.25;margin-bottom:7px;}
    .cl-muted {color:#aebfd4;font-size:.88rem;line-height:1.5;}
    .cl-badge {display:inline-block;color:var(--accent);border:1px solid var(--accent);border-radius:5px;
      padding:4px 10px;font-size:.83rem;font-weight:700;letter-spacing:.035em;}
    .cl-chip {display:inline-block;border:1px solid #38546a;background:#192e40;border-radius:6px;
      padding:5px 9px;margin:3px 5px 3px 0;color:#d2f1f3;font-size:.88rem;}
    .cl-hero {background:linear-gradient(110deg,#162539,#101b2b);border:1px solid #30485e;
      border-left:4px solid var(--accent);border-radius:14px;padding:24px;box-shadow:0 6px 24px #0002;}
    .cl-hero-top {display:flex;justify-content:space-between;align-items:center;gap:18px;}
    .cl-hero h2 {font-size:2rem!important;margin:12px 0 0!important;padding:0!important;}
    .cl-location {font-size:1.14rem;color:#c9d8eb;margin:5px 0 16px;}
    .cl-people {display:flex;flex-wrap:wrap;gap:12px 25px;margin-bottom:17px;color:#e2ebf6;}
    .cl-score {font-size:2rem;font-weight:700;color:#f1f6fc;white-space:nowrap;}
    .cl-score small {font-size:.8rem;color:#adbbce;}
    .cl-meter {margin:10px 0 17px;}
    .cl-meter-label {display:flex;justify-content:space-between;gap:15px;font-size:.92rem;margin-bottom:8px;}
    .cl-track {height:10px;background:#25374c;border-radius:5px;overflow:hidden;}
    .cl-track>div {height:100%;border-radius:5px;}
    .cl-flow {display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:8px 0 16px;}
    .cl-node {flex:1;min-width:110px;background:#122235;border:1px solid #2b465a;border-radius:8px;padding:12px;}
    .cl-arrow {align-self:center;color:#79cfdc;}
    .cl-empty {text-align:center;border:1px dashed #3b576e;border-radius:14px;padding:36px 20px;background:#101d2e;}
    @media(max-width:750px) {.block-container{padding:1rem;} .cl-hero{padding:17px;}
      .cl-hero h2{font-size:1.65rem!important;} .cl-value{font-size:1.25rem;}
      .cl-hero-top{flex-wrap:wrap;} .cl-node{min-width:140px;} }
    </style>""", unsafe_allow_html=True)
