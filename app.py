from __future__ import annotations

import html
import textwrap
import warnings
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "flight_delay_best_model.pkl"
DATA_PATH = BASE_DIR / "flights_sample_3m.csv"

st.set_page_config(
    page_title="udaanIQ | Flight Delay Estimator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root {
    --ink: #18242f;
    --ink-soft: #2b3a47;
    --muted: #66717b;
    --paper: #f4f1e9;
    --card: #fffdf8;
    --field: #fbf8f1;
    --line: #d8d1c3;
    --line-soft: #e8e2d6;
    --accent: #b45309;
    --accent-soft: #f6e7da;
    --green: #2f6b4f;
    --amber: #9a6700;
    --orange: #b45309;
    --red: #a33a2b;
    --night: #16232f;
    --night-soft: #1c2c39;
    --night-text: #f4efe4;
    --night-muted: #a3b0ba;
    --night-line: #31414f;
    --display: "Cambria", "Georgia", "Palatino Linotype", serif;
    --body: "Candara", "Corbel", "Segoe UI", sans-serif;
    --mono: "Cascadia Code", "Cascadia Mono", "Consolas", monospace;
}

html, body, [class*="css"] {
    font-family: var(--body);
}

.stApp {
    background-color: var(--paper);
    color: var(--ink);
    font-family: var(--body);
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1160px;
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
}

.app-stripe {
    height: 5px;
    background: linear-gradient(90deg, var(--night) 0 58%, var(--accent) 58% 72%, var(--line) 72% 100%);
    margin-bottom: 1.25rem;
}

.masthead {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.5rem;
    margin-bottom: 1.6rem;
}

.eyebrow {
    display: block;
    font-family: var(--mono);
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    line-height: 1.4;
    text-transform: uppercase;
}

.masthead .eyebrow {
    color: var(--accent);
}

.masthead h1 {
    font-family: var(--display);
    font-size: clamp(2.6rem, 4.7vw, 3.9rem);
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1;
    margin: 0.55rem 0 0.7rem;
    color: var(--ink);
}

.lede {
    margin: 0;
    max-width: 58ch;
    color: var(--muted);
    font-family: var(--body);
    font-size: 1.06rem;
    line-height: 1.55;
}

.model-stamp {
    flex: 0 0 auto;
    min-width: 176px;
    border-left: 1px solid var(--line);
    padding: 0.15rem 0 0.1rem 1.25rem;
    text-align: right;
}

.model-stamp .stamp-label {
    display: block;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
}

.model-stamp .stamp-value {
    display: block;
    margin-top: 0.35rem;
    font-family: var(--mono);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--ink);
}

.model-stamp .stamp-meta {
    display: block;
    margin-top: 0.3rem;
    font-family: var(--mono);
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    color: #8b8477;
}

.section-head {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 0.1rem 0 0.65rem;
}

.section-num {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--accent);
    border: 1px solid #dcc4ae;
    background: #fbf2e8;
    padding: 0.16rem 0.36rem;
    border-radius: 2px;
}

.section-title {
    font-family: var(--mono);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #5f5a51;
    white-space: nowrap;
}

.section-rule {
    flex: 1;
    height: 1px;
    background: var(--line);
}

.st-key-input_card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.15rem 1.2rem 1.05rem;
    box-shadow: 0 1px 2px rgba(24, 36, 47, 0.03);
}

.st-key-input_card [data-testid="stVerticalBlock"] {
    gap: 0.85rem;
}

[data-testid="stWidgetLabel"] {
    margin-bottom: 0.12rem;
}

[data-testid="stWidgetLabel"] p {
    font-family: var(--mono);
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6d675c;
}

[data-baseweb="input"],
[data-baseweb="select"] > div {
    background-color: var(--field) !important;
    border-color: var(--line) !important;
    border-radius: 2px !important;
}

[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

[data-baseweb="input"] input {
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
}

[data-baseweb="select"] input {
    color: var(--ink) !important;
    font-family: var(--body) !important;
    font-size: 0.97rem !important;
}

.stSelectbox div[data-baseweb="select"] {
    font-family: var(--body) !important;
}

[data-baseweb="popover"] {
    font-family: var(--body) !important;
}

[data-testid="stNumberInput"] button {
    background: transparent !important;
    border: none !important;
    color: #7a7164 !important;
}

[data-testid="stNumberInput"] button:hover {
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
}

.stButton > button {
    width: 100%;
    min-height: 2.5rem;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: transparent;
    color: #5f5a51;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: color 140ms ease, border-color 140ms ease, background 140ms ease;
}

.stButton > button:hover {
    border-color: var(--accent);
    background: #fffaf2;
    color: var(--accent);
}

.stButton > button:focus:not(:active) {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(180, 83, 9, 0.14);
    color: var(--accent);
}

.route-note {
    display: flex;
    gap: 0.55rem;
    align-items: flex-start;
    margin-top: 0.25rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--line-soft);
    color: #7b7468;
    font-family: var(--body);
    font-size: 0.85rem;
    line-height: 1.45;
}

.route-note .marker {
    flex: 0 0 auto;
    width: 6px;
    height: 6px;
    margin-top: 0.36rem;
    background: var(--accent);
}

.route-note strong {
    color: var(--ink-soft);
    font-family: var(--mono);
    font-size: 0.8rem;
    font-weight: 600;
}

.signal-note {
    margin-top: 0.15rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--line-soft);
    color: #7b7468;
    font-family: var(--body);
    font-size: 0.84rem;
    line-height: 1.5;
}

.signal-note span {
    display: block;
    margin-bottom: 0.2rem;
    color: var(--ink-soft);
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.field-warning {
    margin-top: 0.65rem;
    padding: 0.65rem 0.75rem;
    border-left: 2px solid var(--red);
    background: #f9ece8;
    color: #7f3126;
    font-family: var(--body);
    font-size: 0.88rem;
    line-height: 1.45;
}

.result-card {
    position: relative;
    overflow: visible;
    background: var(--night);
    border: 1px solid #0f1a23;
    border-radius: 4px;
    padding: 1.45rem 1.5rem 1.25rem;
    box-shadow: 0 14px 34px -24px rgba(22, 35, 47, 0.75);
    color: var(--night-text);
    font-family: var(--body);
}

.result-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent);
}

.result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.result-top .eyebrow {
    color: var(--night-muted);
}

.status-chip {
    display: inline-block;
    white-space: nowrap;
    padding: 0.2rem 0.45rem;
    border: 1px solid currentColor;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.result-main {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    margin: 1.2rem 0 0.1rem;
}

.result-number {
    font-family: var(--display);
    font-size: clamp(4.4rem, 8.6vw, 6.05rem);
    font-weight: 700;
    letter-spacing: -0.045em;
    line-height: 0.86;
    font-variant-numeric: tabular-nums;
}

.result-unit {
    font-family: var(--mono);
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--night-muted);
}

.result-headline {
    margin: 0.75rem 0 0;
    font-family: var(--display);
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--night-text);
}

.result-copy {
    margin: 0.4rem 0 0;
    max-width: 40ch;
    color: var(--night-muted);
    font-family: var(--body);
    font-size: 0.9rem;
    line-height: 1.55;
}

.scale-wrap {
    margin-top: 1.3rem;
}

.scale-track {
    position: relative;
    display: flex;
    height: 7px;
    border-radius: 1px;
    overflow: visible;
}

.scale-zone {
    height: 7px;
}

.scale-zone.green { width: 21.4%; background: #5d9474; }
.scale-zone.amber { width: 21.5%; background: #bd8b3e; }
.scale-zone.orange { width: 28.5%; background: #c06a35; }
.scale-zone.red { width: 28.6%; background: #a84b3a; }

.scale-marker {
    position: absolute;
    top: -5px;
    width: 2px;
    height: 17px;
    background: #fffdf8;
    box-shadow: 0 0 0 1px rgba(22, 35, 47, 0.7);
}

.scale-labels {
    display: grid;
    grid-template-columns: 21.4fr 21.5fr 28.5fr 28.6fr;
    margin-top: 0.42rem;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #82919c;
}

.scale-labels span {
    text-align: center;
}

.ticket-rule {
    position: relative;
    border-top: 1px dashed #3a4b59;
    margin: 1.3rem -1.5rem 1.1rem;
}

.ticket-rule::before,
.ticket-rule::after {
    content: "";
    position: absolute;
    top: -7px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--paper);
}

.ticket-rule::before { left: -7px; }
.ticket-rule::after { right: -7px; }

.route-line {
    display: flex;
    align-items: center;
}

.route-code {
    font-family: var(--display);
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    line-height: 1;
    color: var(--night-text);
}

.route-stem {
    position: relative;
    flex: 1;
    height: 1px;
    margin: 0 0.8rem;
    background: #445563;
}

.route-stem::before {
    content: "";
    position: absolute;
    left: 0;
    top: -2px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #445563;
}

.route-stem::after {
    content: "";
    position: absolute;
    right: 0;
    top: -3px;
    border-left: 6px solid #445563;
    border-top: 3px solid transparent;
    border-bottom: 3px solid transparent;
}

.route-meta {
    margin-top: 0.45rem;
    font-family: var(--mono);
    font-size: 0.69rem;
    letter-spacing: 0.07em;
    color: var(--night-muted);
}

.facts {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    margin-top: 1.15rem;
    border-top: 1px solid var(--night-line);
}

.fact {
    padding: 0.7rem 0.65rem 0 0;
    border-right: 1px solid var(--night-line);
}

.fact + .fact {
    padding-left: 0.75rem;
}

.fact:last-child {
    border-right: none;
}

.fact-label {
    display: block;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #84949f;
}

.fact-value {
    display: block;
    margin-top: 0.28rem;
    font-family: var(--mono);
    font-size: 0.87rem;
    font-weight: 600;
    color: var(--night-text);
}

.result-note {
    margin-top: 1.1rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--night-line);
    color: #84949f;
    font-family: var(--body);
    font-size: 0.77rem;
    line-height: 1.55;
}

.result-empty {
    min-height: 470px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.empty-mark {
    margin-top: 1.4rem;
    font-family: var(--display);
    font-size: 5.8rem;
    line-height: 0.9;
    color: #536676;
}

.empty-title {
    margin-top: 0.6rem;
    font-family: var(--display);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--night-text);
}

[data-testid="stExpander"] {
    margin-top: 1.65rem;
}

[data-testid="stExpander"] details {
    border: 1px solid var(--line);
    border-radius: 3px;
    background: rgba(255, 253, 248, 0.58);
}

[data-testid="stExpander"] details summary {
    padding: 0.78rem 1rem;
    font-family: var(--mono);
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5f5a51;
}

[data-testid="stExpander"] details summary:hover {
    color: var(--accent);
}

[data-testid="stExpander"] details > div {
    padding: 0 1rem 1rem;
}

.method-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1.15rem 1.8rem;
    padding-top: 0.35rem;
}

.method-block span {
    display: block;
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.32rem;
}

.method-block p {
    margin: 0;
    color: #5f6a73;
    font-family: var(--body);
    font-size: 0.9rem;
    line-height: 1.6;
}

.footer-line {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 1.7rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-family: var(--mono);
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b8477;
}

.footer-line .brand-mark {
    text-transform: none;
    letter-spacing: 0.04em;
    color: #6f685c;
}

@media (max-width: 760px) {
    .masthead {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }

    .model-stamp {
        border-left: none;
        border-top: 1px solid var(--line);
        padding: 0.75rem 0 0;
        text-align: left;
    }

    .method-grid {
        grid-template-columns: 1fr;
    }

    .footer-line {
        flex-direction: column;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_route_stats(path_text: str, nrows: int = 400_000) -> dict[tuple[str, str], tuple[float, float]]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(
            path,
            usecols=["ORIGIN", "DEST", "DISTANCE", "AIR_TIME"],
            nrows=nrows,
        )
    except Exception:
        return {}
    frame = frame.dropna(subset=["ORIGIN", "DEST", "DISTANCE", "AIR_TIME"])
    grouped = frame.groupby(["ORIGIN", "DEST"]).agg(
        distance=("DISTANCE", "median"),
        air_time=("AIR_TIME", "median"),
        flights=("DISTANCE", "size"),
    )
    grouped = grouped[grouped["flights"] >= 5]
    routes = {}
    for row in grouped.reset_index().itertuples(index=False):
        routes[(str(row.ORIGIN), str(row.DEST))] = (
            float(row.distance),
            float(row.air_time),
        )
    return routes


def extract_model_categories(model) -> dict[str, list[str]]:
    preprocess = model.named_steps["preprocess"]
    categories = {}
    for name, transformer, columns in preprocess.transformers_:
        if name != "cat" or not hasattr(transformer, "named_steps"):
            continue
        encoder = transformer.named_steps["onehot"]
        for column, values in zip(columns, encoder.categories_):
            categories[column] = [str(value) for value in values]
    return categories


def format_delay(minutes: float) -> str:
    rounded = int(round(minutes))
    if rounded > 0:
        return f"+{rounded} min"
    if rounded < 0:
        return f"−{abs(rounded)} min"
    return "0 min"


def format_duration(minutes: float) -> str:
    total = int(round(minutes))
    hours, remainder = divmod(total, 60)
    if hours and remainder:
        return f"{hours}h {remainder:02d}m"
    if hours:
        return f"{hours}h"
    return f"{remainder}m"


def format_distance(miles: float) -> str:
    return f"{int(round(miles)):,} mi"


def status_for(prediction: float) -> dict[str, str]:
    if prediction < -5:
        return {
            "chip": "early",
            "color": "#93c8a6",
            "headline": "Ahead of schedule",
            "copy": "The central estimate puts this flight ahead of its scheduled arrival.",
        }
    if prediction <= 15:
        return {
            "chip": "on time",
            "color": "#93c8a6",
            "headline": "Inside the on-time window",
            "copy": "The central estimate sits within the 15-minute window used for on-time arrivals.",
        }
    if prediction <= 45:
        return {
            "chip": "minor delay",
            "color": "#e0b46a",
            "headline": "A short delay",
            "copy": "The central estimate points to a brief arrival delay rather than a major disruption.",
        }
    if prediction <= 120:
        return {
            "chip": "delayed",
            "color": "#e6905e",
            "headline": "A meaningful delay",
            "copy": "The central estimate is well beyond the on-time window for this flight.",
        }
    return {
        "chip": "severe",
        "color": "#e58b78",
        "headline": "A long arrival delay",
        "copy": "The central estimate is more than two hours late and should be treated as a disruption.",
    }


if not MODEL_PATH.exists():
    st.error("flight_delay_best_model.pkl was not found next to app.py.")
    st.stop()

try:
    model = load_model()
except Exception:
    st.error(
        "The saved model could not be loaded. Its pipeline was pickled with "
        "scikit-learn 1.6.1; install the pinned requirements and relaunch the app."
    )
    st.stop()

features = list(model.feature_names_in_)
model_categories = extract_model_categories(model)
airlines = sorted(model_categories.get("AIRLINE", []))
airports = sorted(set(model_categories.get("ORIGIN", [])) | set(model_categories.get("DEST", [])))
route_stats = load_route_stats(str(DATA_PATH))

if not airlines or not airports:
    st.error("The saved model is missing its airline or airport categories.")
    st.stop()

default_airline = "Delta Air Lines Inc." if "Delta Air Lines Inc." in airlines else airlines[0]
default_origin = "ATL" if "ATL" in airports else airports[0]
default_dest = "JFK" if "JFK" in airports and "JFK" != default_origin else next(
    airport for airport in airports if airport != default_origin
)
default_stats = route_stats.get((default_origin, default_dest), (760.0, 102.0))

defaults = {
    "airline": default_airline,
    "origin": default_origin,
    "dest": default_dest,
    "flight_date": date.today(),
    "dep_delay": 24,
    "air_time": int(round(default_stats[1])),
    "distance": int(round(default_stats[0])),
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def sync_route_defaults() -> None:
    stats = route_stats.get((st.session_state.origin, st.session_state.dest))
    if stats:
        st.session_state.air_time = int(round(stats[1]))
        st.session_state.distance = int(round(stats[0]))


def swap_route() -> None:
    origin = st.session_state.origin
    st.session_state.origin = st.session_state.dest
    st.session_state.dest = origin
    sync_route_defaults()


st.markdown('<div class="app-stripe"></div>', unsafe_allow_html=True)

header_html = """
<div class="masthead">
    <div>
        <span class="eyebrow">Flight delay estimator / decision support</span>
        <h1>udaanIQ</h1>
        <p class="lede">Estimate arrival delay from departure timing, route, airline and calendar inputs. Adjust any field and the estimate follows.</p>
    </div>
    <div class="model-stamp">
        <span class="stamp-label">Saved model</span>
        <span class="stamp-value">Ridge / pipeline 01</span>
        <span class="stamp-meta">scikit-learn 1.6 &middot; no retraining</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

left, right = st.columns([1.24, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-head"><span class="section-num">01</span>'
        '<span class="section-title">Flight details</span><span class="section-rule"></span></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="input_card"):
        airline = st.selectbox("Airline", airlines, key="airline")

        origin_col, swap_col, dest_col = st.columns(
            [1, 0.52, 1], gap="small", vertical_alignment="bottom"
        )
        with origin_col:
            origin = st.selectbox(
                "Origin",
                airports,
                key="origin",
                on_change=sync_route_defaults,
            )
        with swap_col:
            st.button("Swap", on_click=swap_route, use_container_width=True)
        with dest_col:
            dest = st.selectbox(
                "Destination",
                airports,
                key="dest",
                on_change=sync_route_defaults,
            )

        date_col, delay_col = st.columns([1, 1], gap="small")
        with date_col:
            flight_date = st.date_input(
                "Flight date",
                key="flight_date",
                help="The model uses the month and weekday, not the year.",
            )
        with delay_col:
            dep_delay = st.number_input(
                "Departure delay (min)",
                min_value=-60,
                max_value=720,
                step=1,
                key="dep_delay",
                help="Negative values mean the flight left early.",
            )

        air_col, distance_col = st.columns([1, 1], gap="small")
        with air_col:
            air_time = st.number_input(
                "Air time (min)",
                min_value=15,
                max_value=720,
                step=1,
                key="air_time",
                help="Typical values for the selected route are filled from the dataset.",
            )
        with distance_col:
            distance = st.number_input(
                "Distance (mi)",
                min_value=20,
                max_value=6000,
                step=1,
                key="distance",
                help="Typical values for the selected route are filled from the dataset.",
            )

        route_median = route_stats.get((origin, dest))
        if route_median:
            note = (
                f"<strong>{html.escape(origin)} → {html.escape(dest)}</strong> "
                f"route median: {format_distance(route_median[0])} · "
                f"{format_duration(route_median[1])}. Values update when the route changes."
            )
        elif DATA_PATH.exists():
            note = (
                "No historical median for this route in the lookup sample. "
                "Enter distance and air time manually."
            )
        else:
            note = "Route lookup data is unavailable. Enter distance and air time manually."
        st.markdown(
            f'<div class="route-note"><span class="marker"></span><span>{note}</span></div>'
            '<div class="signal-note"><span>Model note</span>'
            "Departure delay carries most of the signal. Airline, route and date "
            "adjust the estimate at the margin.</div>",
            unsafe_allow_html=True,
        )

    route_is_valid = origin != dest
    if not route_is_valid:
        st.markdown(
            '<div class="field-warning">Origin and destination must be different '
            "for the model to produce an estimate.</div>",
            unsafe_allow_html=True,
        )

prediction = None
if route_is_valid:
    input_row = {
        "DEP_DELAY": float(dep_delay),
        "AIR_TIME": float(air_time),
        "DISTANCE": float(distance),
        "MONTH": int(flight_date.month),
        "DAY_OF_WEEK": int(flight_date.isoweekday()),
        "AIRLINE": airline,
        "ORIGIN": origin,
        "DEST": dest,
    }
    try:
        input_frame = pd.DataFrame([input_row], columns=features)
        prediction = float(model.predict(input_frame)[0])
    except Exception:
        prediction = None

with right:
    st.markdown(
        '<div class="section-head"><span class="section-num">02</span>'
        '<span class="section-title">Estimate</span><span class="section-rule"></span></div>',
        unsafe_allow_html=True,
    )

    if prediction is None:
        if not route_is_valid:
            empty_html = """
            <div class="result-card result-empty">
                <span class="eyebrow">Arrival delay / estimate</span>
                <div class="empty-mark">&mdash;</div>
                <div class="empty-title">Two airports needed</div>
                <p class="result-copy">Choose a different origin and destination to see an arrival-delay estimate.</p>
            </div>
            """
        else:
            empty_html = """
            <div class="result-card result-empty">
                <span class="eyebrow">Arrival delay / estimate</span>
                <div class="empty-mark">&mdash;</div>
                <div class="empty-title">Estimate unavailable</div>
                <p class="result-copy">The saved model could not score this combination of inputs.</p>
            </div>
            """
        st.markdown(textwrap.dedent(empty_html).strip(), unsafe_allow_html=True)
    else:
        status = status_for(prediction)
        rounded = int(round(prediction))
        if rounded > 0:
            number_text = f"+{rounded}"
        elif rounded < 0:
            number_text = f"&minus;{abs(rounded)}"
        else:
            number_text = "0"
        marker_position = max(0.0, min(100.0, ((prediction + 30) / 210) * 100))
        flight_day = flight_date.strftime("%a, %d %b")
        safe_airline = html.escape(airline)
        safe_origin = html.escape(origin)
        safe_dest = html.escape(dest)

        result_html = f"""
        <div class="result-card">
            <div class="result-top">
                <span class="eyebrow">Arrival delay / estimate</span>
                <span class="status-chip" style="color:{status['color']};">{status['chip']}</span>
            </div>
            <div class="result-main">
                <span class="result-number" style="color:{status['color']};">{number_text}</span>
                <span class="result-unit">minutes</span>
            </div>
            <p class="result-headline">{status['headline']}</p>
            <p class="result-copy">{status['copy']}</p>
            <div class="scale-wrap">
                <div class="scale-track">
                    <span class="scale-zone green"></span>
                    <span class="scale-zone amber"></span>
                    <span class="scale-zone orange"></span>
                    <span class="scale-zone red"></span>
                    <span class="scale-marker" style="left:{marker_position:.2f}%;"></span>
                </div>
                <div class="scale-labels">
                    <span>early</span><span>on time</span><span>late</span><span>severe</span>
                </div>
            </div>
            <div class="ticket-rule"></div>
            <div class="route-line">
                <span class="route-code">{safe_origin}</span>
                <span class="route-stem"></span>
                <span class="route-code">{safe_dest}</span>
            </div>
            <div class="route-meta">{safe_airline} &middot; {flight_day}</div>
            <div class="facts">
                <div class="fact">
                    <span class="fact-label">Departed</span>
                    <span class="fact-value">{format_delay(dep_delay)}</span>
                </div>
                <div class="fact">
                    <span class="fact-label">Air time</span>
                    <span class="fact-value">{format_duration(air_time)}</span>
                </div>
                <div class="fact">
                    <span class="fact-label">Distance</span>
                    <span class="fact-value">{format_distance(distance)}</span>
                </div>
            </div>
            <div class="result-note">Point estimate from the saved Ridge pipeline. It uses departure delay and air time, so it is best read after departure rather than as a pre-flight forecast.</div>
        </div>
        """
        st.markdown(textwrap.dedent(result_html).strip(), unsafe_allow_html=True)

method_html = """
<div class="method-grid">
    <div class="method-block">
        <span>Pipeline</span>
        <p>Saved scikit-learn pipeline with median imputation and standardisation for numeric fields, plus one-hot encoding for airline and airport fields. Estimator: Ridge regression.</p>
    </div>
    <div class="method-block">
        <span>Model inputs</span>
        <p>Departure delay, air time, distance, month, weekday, airline, origin and destination. The interface loads the saved model directly; nothing is retrained.</p>
    </div>
    <div class="method-block">
        <span>Route defaults</span>
        <p>Air time and distance are pre-filled with medians from a 400,000-row sample of the flight dataset. Editing either field changes the estimate.</p>
    </div>
    <div class="method-block">
        <span>Scope</span>
        <p>Departure delay and air time are only known after departure, so this is an in-flight estimate rather than a pre-departure forecast. Outputs are point estimates, not probabilities.</p>
    </div>
</div>
"""
with st.expander("How the estimate is made"):
    st.markdown(method_html, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-line"><span class="brand-mark">udaanIQ / flight delay estimator</span>'
    "<span>Model: Ridge regression · Data: flights_sample_3m.csv</span></div>",
    unsafe_allow_html=True,
)
