# PRCSI Streamlit frontend, reads from data/results committed by GitHub Actions pipeline.

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Page config.
st.set_page_config(
    page_title="Oil Fear & Greed Index",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Paths.
RESULTS_DIR = Path("data/results")
META_PATH   = RESULTS_DIR / "pipeline_metadata.json"
PARQUET     = RESULTS_DIR / "prcsi_final.parquet"

# Styling.
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background: #0a0f1e; }
  /* Full-width, remove Streamlit default side padding */
  .block-container {
    padding-top: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100% !important;
  }

  .regime-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    text-align: center;
    margin-top: 0.25rem;
    opacity: 0.75;
  }

  .signal-bearish {
    background: linear-gradient(135deg, #3f0d0d 0%, #1a0505 100%);
    border: 1px solid #dc262680;
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }
  .signal-bullish {
    background: linear-gradient(135deg, #0d2e18 0%, #051a0b 100%);
    border: 1px solid #16a34a80;
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }
  .signal-none {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }
  .signal-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.48;
    margin-bottom: 0.35rem;
  }
  .signal-direction {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
  }
  .signal-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    opacity: 0.58;
    line-height: 1.9;
  }

  .stat-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 0.85rem 1rem;
  }
  .stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.38;
    margin-bottom: 0.25rem;
  }
  .stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    font-weight: 500;
  }
  .stat-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    opacity: 0.42;
    margin-top: 0.12rem;
  }

  .section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    opacity: 0.32;
    margin: 1.4rem 0 0.55rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #1f2937;
  }

  .disclaimer {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    opacity: 0.26;
    line-height: 1.9;
    border-top: 1px solid #1f2937;
    padding-top: 1.1rem;
    margin-top: 1.6rem;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    border-bottom: 1px solid #1f2937;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.45rem 0.9rem;
    color: #6b7280;
  }
  .stTabs [aria-selected="true"] {
    color: #f9fafb;
    background: #111827;
    border-radius: 6px 6px 0 0;
  }

  /* Expander headers */
  .streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    color: #6b7280 !important;
    text-transform: uppercase !important;
  }

  .js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# Regime helpers.
REGIME_COLORS = {
    "EXTREME_FEAR": "#1d4ed8",
    "FEAR":         "#60a5fa",
    "NEUTRAL":      "#9ca3af",
    "GREED":        "#f97316",
    "EXTREME_GREED":"#dc2626",
}

def score_color(v):
    if v <= 25:   return REGIME_COLORS["EXTREME_FEAR"]
    elif v <= 45: return REGIME_COLORS["FEAR"]
    elif v <= 55: return REGIME_COLORS["NEUTRAL"]
    elif v <= 75: return REGIME_COLORS["GREED"]
    else:         return REGIME_COLORS["EXTREME_GREED"]

def regime_label(v):
    if v <= 25:   return "Extreme Fear"
    elif v <= 45: return "Fear"
    elif v <= 55: return "Neutral"
    elif v <= 75: return "Greed"
    else:         return "Extreme Greed"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#9ca3af", size=11),
    margin=dict(l=52, r=52, t=28, b=0),
)
AXIS_STYLE = dict(gridcolor="#1f2937", zeroline=False)

# Data loading.
@st.cache_data(ttl=60)
def load_metadata():
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_index():
    if not PARQUET.exists():
        return None
    d = pd.read_parquet(PARQUET)
    d.index = pd.to_datetime(d.index)
    return d

@st.cache_data(ttl=60)
def load_psi():
    path = RESULTS_DIR / "psi_final.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d.index = pd.to_datetime(d.index)
    return d

meta   = load_metadata()
df     = load_index()
psi_df = load_psi()

# Current state.
score        = meta.get("prcsi_latest", 50.0)
severity     = meta.get("prcsi_severity", 0.0)
sig_active   = meta.get("signal_active", False)
sig_dir      = meta.get("signal_direction", "NONE")
sig_tier     = meta.get("signal_tier", "none")
sig_acc_oos  = meta.get("tier_accuracy_oos")
sig_acc_full = meta.get("tier_accuracy_full")
nlp_fresh    = meta.get("nlp_is_fresh", False)
sig_30d      = meta.get("signals_last_30d", 0)
run_ts       = meta.get("full_run_timestamp", "")
prcsi_date   = meta.get("prcsi_date", "")

color  = score_color(score)
regime = regime_label(score)

psi_score      = meta.get("psi_latest", 50.0)
psi_rsi        = meta.get("psi_rsi_7")
divergence_pct = meta.get("divergence_pct_pts", 0.0) or 0.0
div_direction  = meta.get("divergence_direction", "ALIGNED")
psi_color      = score_color(psi_score)
psi_label      = regime_label(psi_score)

# Header.
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown("### WTI Oil Fear &amp; Greed Index", unsafe_allow_html=True)
with col_meta:
    if prcsi_date:
        st.markdown(
            f"<div style='text-align:right; font-family:DM Mono; font-size:0.65rem;"
            f" color:#6b7280; margin-top:0.75rem;'>"
            f"{'NLP fresh' if nlp_fresh else 'NLP carried'}"
            f" &nbsp;·&nbsp; {prcsi_date}</div>",
            unsafe_allow_html=True,
        )
st.markdown("---")

if df is None or len(df) == 0:
    st.warning("Pipeline has not produced output yet. "
               "Run the GitHub Actions econometric pipeline first.")
    st.stop()

# Tabs.
tab_live, tab_history, tab_signals = st.tabs(
    ["Live Index", "Historical Chart", "Signal Log"]
)


# TAB 1, LIVE INDEX
with tab_live:

    with st.expander("What is this index?", expanded=False):
        ca, cb = st.columns(2)
        with ca:
            st.markdown("""
**Two indices, one divergence signal.**

**PRCSI Petroleum Risk & Conviction Sentiment Index**
Measures whether institutional sentiment is extreme relative to recent conditions.
Built from NLP scoring of OPEC Monthly Oil Market Reports, EIA Short-Term Energy
Outlooks, and Saudi Aramco press coverage combined with EIA inventory data, CFTC
futures positioning, and FRED macro controls. Normalised as a rolling 252-day
percentile rank: a reading of 70 means institutional sentiment is more extreme than
70% of the past year's readings not that price is high.

**PSI Price Sentiment Index**
Measures whether price is historically expensive or cheap, and how extreme current
momentum is relative to the biggest moves since 2007. Uses an expanding window
anchored to 2007 so $105 today is judged against every WTI price ever recorded,
not just the past year.

**Why they look different and why that is correct.**
PRCSI oscillates around 50 by design. Institutional sentiment is rarely at
extremes which is exactly why the top 10% threshold fires infrequently and
carries 86.8% OOS directional accuracy. A signal that fires constantly would
be useless. PSI tracks price more closely because it is anchored to the full
price history.

**The divergence between them is the core signal.** When price momentum (PSI)
runs far ahead of institutional narrative (PRCSI), the market is moving faster
than fundamentals justify a classic information asymmetry setup
(Grossman-Stiglitz 1980). The current Hormuz spike is a live example: PSI
surged with price, PRCSI remained moderate, and the ~21 point divergence
flags exactly this asymmetry.
""")
        with cb:
            st.markdown("""
**Fear and Greed in oil — different from stocks.**

In equity markets, fear and greed indices are momentum indicators:
high greed = overbought = caution. Oil works differently.

The PRCSI is **contrarian on institutional sentiment**: when OPEC, EIA, and Aramco
communicate extreme confidence (greed), oil prices have historically tended to fall
over the following 21–42 trading days. When they communicate extreme caution (fear),
prices have tended to rise.

Why? Institutional overconfidence often precedes supply-side miscalculation.
OPEC production decisions, refinery scheduling, and inventory management are all
influenced by prevailing sentiment. Greedy institutions tend to over-supply;
fearful ones under-supply.

**Do not read PRCSI like a price chart.**
PRCSI does not track price that is PSI's job. PRCSI asks: is sentiment extreme
*relative to recent history*? PSI asks: is price extreme *relative to all history
since 2007*? They answer different questions. The gap between them — the divergence
— is where the predictive content lives.

**Validated OOS accuracy (2020–2026):** 86.8% directional accuracy at the
top 10% severity threshold over a 21-day forward horizon.
""")

    # Gauges, PRCSI, divergence, PSI.
    col_prcsi, col_div, col_psi = st.columns([1, 0.6, 1])

    def make_gauge(value, clr, height=210):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 46, "color": clr, "family": "DM Mono, monospace"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickfont": {"size": 8, "color": "#6b7280"},
                    "tickvals": [0, 25, 55, 75, 100],
                },
                "bar": {"color": clr, "thickness": 0.24},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  25],  "color": "#0d1f5c"},
                    {"range": [25, 45],  "color": "#1e3f8a"},
                    {"range": [45, 55],  "color": "#1f2937"},
                    {"range": [55, 75],  "color": "#7c2d12"},
                    {"range": [75, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#f9fafb", "width": 2},
                    "thickness": 0.82, "value": value,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="#9ca3af"),
            margin=dict(l=6, r=6, t=4, b=0),
            height=height,
        )
        return fig

    with col_prcsi:
        st.markdown(
            "<div style='text-align:center; font-family:DM Mono; font-size:0.62rem;"
            " letter-spacing:0.14em; opacity:0.36; text-transform:uppercase;"
            " margin-bottom:0.15rem;'>PRCSI — Institutional</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_gauge(score, color),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='regime-label' style='color:{color};'>{regime}</div>",
                    unsafe_allow_html=True)

    with col_div:
        div_color = ("#dc2626" if div_direction == "PSI_LEADS"
                     else "#3b82f6" if div_direction == "PRCSI_LEADS"
                     else "#6b7280")
        div_sign  = "+" if divergence_pct > 0 else ""
        div_label = {
            "PSI_LEADS":   "Price Ahead",
            "PRCSI_LEADS": "Narrative Ahead",
            "ALIGNED":     "Aligned",
        }.get(div_direction, "Aligned")
        div_note = {
            "PSI_LEADS":   "Price momentum ahead of institutional narrative",
            "PRCSI_LEADS": "Institutional narrative ahead of price",
            "ALIGNED":     "Indices aligned",
        }.get(div_direction, "")

        st.markdown(
            "<div style='text-align:center; font-family:DM Mono; font-size:0.62rem;"
            " letter-spacing:0.14em; opacity:0.36; text-transform:uppercase;"
            " margin-bottom:0.15rem;'>Divergence</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div style='text-align:center; padding:1.8rem 0.2rem 0.8rem;'>
          <div style='font-family:DM Mono; font-size:2.4rem; font-weight:500;
                      color:{div_color}; line-height:1;'>
            {div_sign}{abs(divergence_pct):.1f}
          </div>
          <div style='font-family:DM Sans; font-size:0.65rem; opacity:0.45;
                      margin-top:0.25rem;'>points</div>
          <div style='font-family:DM Mono; font-size:0.62rem; color:{div_color};
                      letter-spacing:0.1em; text-transform:uppercase;
                      margin-top:0.5rem;'>{div_label}</div>
          <div style='font-family:DM Mono; font-size:0.56rem; opacity:0.32;
                      margin-top:0.65rem; line-height:1.8;'>
            PRCSI &nbsp;{score:.1f}<br>PSI &nbsp;&nbsp;&nbsp;&nbsp;{psi_score:.1f}
          </div>
          <div style='font-family:DM Sans; font-size:0.6rem; opacity:0.28;
                      margin-top:0.55rem; line-height:1.5; padding:0 0.2rem;'>
            {div_note}
          </div>
        </div>""", unsafe_allow_html=True)

    with col_psi:
        st.markdown(
            "<div style='text-align:center; font-family:DM Mono; font-size:0.62rem;"
            " letter-spacing:0.14em; opacity:0.36; text-transform:uppercase;"
            " margin-bottom:0.15rem;'>PSI:  Price Action</div>",
            unsafe_allow_html=True,
        )
        psi_ok = psi_df is not None and len(psi_df) > 0
        st.plotly_chart(
            make_gauge(psi_score if psi_ok else 50.0, psi_color if psi_ok else "#6b7280"),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown(f"<div class='regime-label' style='color:{psi_color};'>{psi_label}</div>",
                    unsafe_allow_html=True)
        if psi_rsi is not None:
            st.markdown(
                f"<div style='text-align:center; font-family:DM Mono; font-size:0.6rem;"
                f" opacity:0.38; margin-top:0.2rem;'>RSI(7): {psi_rsi:.0f}</div>",
                unsafe_allow_html=True,
            )

    # Signal and stats.
    col_sig, col_stats = st.columns([1, 1])

    with col_sig:
        st.markdown("<div class='section-header'>Active Signal</div>",
                    unsafe_allow_html=True)
        if sig_active:
            box_cls   = "signal-bearish" if sig_dir == "BEARISH" else "signal-bullish"
            dir_color = "#f87171" if sig_dir == "BEARISH" else "#4ade80"
            tier_map  = {"top_2": "Top 2%", "top_5": "Top 5%", "top_10": "Top 10%"}
            tier_str  = tier_map.get(sig_tier, sig_tier)
            acc_oos   = f"{sig_acc_oos:.1%}" if sig_acc_oos else "n/a"
            acc_full  = f"{sig_acc_full:.1%}" if sig_acc_full else "n/a"
            st.markdown(f"""
            <div class='{box_cls}'>
              <div class='signal-title'>Signal active</div>
              <div class='signal-direction' style='color:{dir_color};'>
                {'BEARISH' if sig_dir == 'BEARISH' else 'BULLISH'}
              </div>
              <div class='signal-sub'>
                Tier &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{tier_str}<br>
                Severity &nbsp;&nbsp;&nbsp;{severity:.4f}<br>
                Horizon &nbsp;&nbsp;&nbsp;&nbsp;21–42 trading days<br>
                OOS accuracy &nbsp;<b style='color:{dir_color};'>{acc_oos}</b><br>
                Full-sample &nbsp;&nbsp;{acc_full}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='signal-none'>
              <div class='signal-title'>No active signal</div>
              <div style='color:#6b7280; font-size:0.9rem; font-weight:500;
                          margin:0.3rem 0;'>Watching</div>
              <div class='signal-sub'>
                Severity {severity:.4f} is below the action<br>
                threshold of 0.2637 (train-frozen top 10%).<br><br>
                Signals in last 30 days: {sig_30d}
              </div>
            </div>""", unsafe_allow_html=True)

    with col_stats:
        st.markdown("<div class='section-header'>Index State</div>",
                    unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-label'>Score</div>
              <div class='stat-value' style='color:{color};'>{score:.1f}</div>
              <div class='stat-sub'>0 = max fear · 100 = max greed</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-label'>Severity</div>
              <div class='stat-value'>{severity:.4f}</div>
              <div class='stat-sub'>threshold 0.2637 (top 10%)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        s3, s4 = st.columns(2)

        if df is not None and "prcsi" in df.columns and len(df) > 5:
            week_ago = float(df["prcsi"].dropna().iloc[:-5].iloc[-1])
            delta_7d = score - week_ago
            sign_str = "+" if delta_7d > 0 else ""
            d_color  = REGIME_COLORS["GREED"] if delta_7d > 0 else REGIME_COLORS["FEAR"]
        else:
            delta_7d, sign_str, d_color = 0.0, "", "#9ca3af"

        with s3:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-label'>7-day change</div>
              <div class='stat-value' style='color:{d_color};'>{sign_str}{delta_7d:.1f}</div>
              <div class='stat-sub'>points</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-label'>Signals / 30d</div>
              <div class='stat-value'>{sig_30d}</div>
              <div class='stat-sub'>top 10% firings</div>
            </div>""", unsafe_allow_html=True)

    # Index history, dual y-axis chart.
    st.markdown("<div class='section-header'>Index History</div>",
                unsafe_allow_html=True)

    if df is not None:
        _, rc = st.columns([5, 1])
        with rc:
            time_range = st.selectbox(
                "Range", ["3M", "1Y", "3Y", "5Y", "All"],
                index=4, label_visibility="collapsed",
            )
        days_map = {"3M": 90, "1Y": 365, "3Y": 1095, "5Y": 1825, "All": 99999}
        cutoff = df.index.max() - pd.Timedelta(days=days_map[time_range])
        recent = df[df.index >= cutoff]

        # Row 1, dual-axis PRCSI, PSI, price. Row 2, severity.
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.76, 0.24],
            vertical_spacing=0.03,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        # Fear/greed background bands on left axis.
        for y0, y1, clr in [(0, 25, "#0d1f5c"), (25, 45, "#1e3f8a"),
                             (45, 55, "#1f2937"), (55, 75, "#7c2d12"), (75, 100, "#450a0a")]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, opacity=0.28,
                          line_width=0, row=1, col=1)

        # WTI price, right axis, amber.
        has_price = "oil_price" in recent.columns
        if has_price:
            price = recent["oil_price"].ffill()
            fig.add_trace(go.Scatter(
                x=recent.index, y=price,
                mode="lines", name="WTI price",
                line=dict(color="#fbbf24", width=1.8),
                opacity=0.80,
                hovertemplate="%{x|%Y-%m-%d}<br>WTI: $%{y:.2f}<extra></extra>",
            ), row=1, col=1, secondary_y=True)

        # PRCSI, left axis, white.
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent["prcsi"],
            mode="lines", name="PRCSI",
            line=dict(color="#f9fafb", width=2.0),
            hovertemplate="%{x|%Y-%m-%d}<br>PRCSI: %{y:.1f}<extra></extra>",
        ), row=1, col=1, secondary_y=False)

        # PSI, left axis, orange dotted.
        if psi_df is not None:
            psi_r = psi_df[psi_df.index >= recent.index.min()]
            if len(psi_r):
                fig.add_trace(go.Scatter(
                    x=psi_r.index, y=psi_r["psi"],
                    mode="lines", name="PSI",
                    line=dict(color="#f97316", width=1.4, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>PSI: %{y:.1f}<extra></extra>",
                ), row=1, col=1, secondary_y=False)

        # Signal markers on price line.
        for direction, sym, clr, label in [
            ("BEARISH", "triangle-down", "#f87171", "Bearish"),
            ("BULLISH", "triangle-up",   "#4ade80", "Bullish"),
        ]:
            mask = recent["signal_direction"] == direction
            if not mask.any():
                continue
            if has_price:
                y_vals = price.reindex(recent.index[mask]).ffill()
                ct     = recent["prcsi"][mask].values
                tmpl   = (f"%{{x|%Y-%m-%d}}<br>{label} signal<br>"
                          f"WTI: $%{{y:.2f}}<br>PRCSI: %{{customdata:.1f}}<extra></extra>")
                fig.add_trace(go.Scatter(
                    x=recent.index[mask], y=y_vals,
                    mode="markers", name=f"{label} signal",
                    marker=dict(symbol=sym, size=10, color=clr,
                                line=dict(color="#0a0f1e", width=0.7)),
                    customdata=ct,
                    hovertemplate=tmpl,
                ), row=1, col=1, secondary_y=True)
            else:
                fig.add_trace(go.Scatter(
                    x=recent.index[mask], y=recent["prcsi"][mask],
                    mode="markers", name=f"{label} signal",
                    marker=dict(symbol=sym, size=10, color=clr),
                    hovertemplate=(f"%{{x|%Y-%m-%d}}<br>{label} "
                                   f"%{{y:.1f}}<extra></extra>"),
                ), row=1, col=1, secondary_y=False)

        fig.add_hline(y=50, line_width=0.7, line_dash="dot",
                      line_color="#6b7280", row=1, col=1)

        # Severity bars.
        sev_clrs = [
            "#dc2626" if v >= 0.2637 else "#374151"
            for v in recent["severity"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=recent.index, y=recent["severity"],
            name="Severity", showlegend=False,
            marker_color=sev_clrs,
            hovertemplate="%{x|%Y-%m-%d}<br>Severity: %{y:.4f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0.2637, line_width=1, line_dash="dash",
                      line_color="#dc2626", opacity=0.5, row=2, col=1)

        fig.update_layout(
            **PLOT_LAYOUT,
            height=500,
            showlegend=True,
            legend=dict(
                font=dict(size=9, color="#9ca3af"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                x=0, y=1.04, xanchor="left",
            ),
            hovermode="x unified",
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(
            row=1, col=1, secondary_y=False,
            range=[0, 100], tickvals=[0, 25, 45, 55, 75, 100],
            title_text="Index", title_font=dict(size=9),
            **AXIS_STYLE, tickfont=dict(size=9),
        )
        fig.update_yaxes(
            row=1, col=1, secondary_y=True,
            title_text="WTI USD", title_font=dict(size=9, color="#fbbf24"),
            tickprefix="$", showgrid=False,
            tickfont=dict(size=9, color="#fbbf24"),
            zeroline=False,
        )
        fig.update_yaxes(
            row=2, col=1,
            range=[0, 0.55], title_text="Severity",
            title_font=dict(size=9), **AXIS_STYLE, tickfont=dict(size=9),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
<div style='display:grid; grid-template-columns:repeat(3,1fr); gap:0.25rem 1.2rem;
            font-family:DM Mono,monospace; font-size:0.6rem; opacity:0.44;
            margin-top:0.35rem; line-height:1.7;'>
  <div><span style='color:#f9fafb;'>—</span> &nbsp;PRCSI (left, 0–100)</div>
  <div><span style='color:#f97316;'>···</span> &nbsp;PSI (left, 0–100)</div>
  <div><span style='color:#fbbf24;'>—</span> &nbsp;WTI price (right, USD)</div>
  <div><span style='color:#f87171;'>▼</span> &nbsp;Bearish signal on price</div>
  <div><span style='color:#4ade80;'>▲</span> &nbsp;Bullish signal on price</div>
  <div><span style='color:#dc2626;'>- -</span> &nbsp;Signal threshold 0.2637</div>
</div>
<div style='font-family:DM Mono,monospace; font-size:0.57rem; opacity:0.26;
            margin-top:0.35rem; line-height:1.6;'>
  Signal markers sit on the WTI price line. Read the price at the signal date,
  then trace forward to see the 21–42 day outcome. Index bands show institutional
  regime at the same moment. Gap between PRCSI and PSI = divergence signal.
</div>
""", unsafe_allow_html=True)

    # PRCSI vs price expander.
    with st.expander("Why does PRCSI not track price?", expanded=False):
        st.markdown("""
**This is intentional. PRCSI and PSI measure fundamentally different things.**

| | PRCSI | PSI |
|---|---|---|
| **Question** | Is institutional sentiment extreme relative to recent conditions? | Is price historically expensive, and how extreme is current momentum? |
| **Window** | Rolling 252 days (1 year) | Expanding from 2007 to present |
| **Anchor** | Last year's sentiment distribution | Full price history since 2007 |
| **Result** | Oscillates around 50 by design | Tracks price level closely |

**Why the rolling window causes PRCSI to centre around 50**

PRCSI normalises each feature as a percentile rank within the trailing 252 trading days.
By construction, roughly half the values in any rolling window will be above the median
and half below so the index gravitates toward 50. This is not a bug. It means a reading
of 65 tells you: *institutional sentiment is more bullish than 65% of the past year's
readings* regardless of whether the absolute price is $60 or $130.

**Why this is the correct design for a contrarian signal**

The 86.8% OOS directional accuracy was validated on exactly this normalisation.
If PRCSI tracked price, it would converge with PSI and the divergence between them
would carry no information. The entire thesis — that institutions have an information
advantage that creates predictable asymmetry relative to price momentum
(Grossman-Stiglitz 1980) depends on PRCSI and PSI measuring different things.

**The divergence is the signal, not the individual levels**

When PSI runs well above PRCSI, price is moving faster than institutional narrative
justifies. This gap has historically preceded reversals over a 21–42 trading day horizon.
The May 2026 Hormuz case is a live example: PSI ~74, PRCSI ~50, divergence ~21 points —
price spiked from ~$56 to $111+, but institutional communications remained measured.
That asymmetry is the information the index is designed to surface.
""")

    # Data sources expander.
    with st.expander("What data is this built on?", expanded=False):
        st.markdown("""
| Source | What it contributes | Frequency |
|---|---|---|
| **OPEC Monthly Oil Market Report** | Cartel production sentiment, supply outlook | Monthly |
| **EIA Short-Term Energy Outlook** | US government demand/supply forecasts | Monthly |
| **Saudi Aramco press coverage** | World's largest producer signals | Daily |
| **EIA Weekly Petroleum Status** | Inventory surprise — #1 short-term price driver | Weekly |
| **CFTC Commitments of Traders** | Managed money positioning (speculators) | Weekly |
| **FRED** | Fed funds rate, USD broad index, breakeven inflation | Daily |
| **WTI Futures (yfinance)** | Price, returns, volatility — PSI inputs only | Daily |

OPEC, EIA, and Aramco texts are scored by a locally-hosted LLM (`gpt-oss-20b`) using a locked
system prompt validated against historic backfill. Scores cover 6 dimensions: oil impact, supply
disruption, demand outlook, geopolitical risk, surface-vs-implied divergence, and institutional
confidence. NLP data is genuine on approximately 5.5% of trading days (publication days); other
days carry the most recent publication forward with a 21-day maximum fill.
""")

    # Disclaimer.
    last_update = run_ts[:10] if run_ts else "unknown"
    st.markdown(f"""
<div class='disclaimer'>
  PRCSI v1.0 &nbsp;·&nbsp; Last pipeline run: {last_update} &nbsp;·&nbsp;
  For research and informational purposes only. Does not constitute financial advice
  or a recommendation to buy or sell any security or commodity. Past accuracy at a
  given signal tier does not guarantee future results. Results are strongest in
  high-volatility regimes and may be weaker in stable conditions.
  Always conduct independent analysis before making investment decisions.
</div>
""", unsafe_allow_html=True)


# TAB 2, HISTORICAL CHART
with tab_history:
    st.markdown("<div class='section-header'>Full History — 2007 to Present</div>",
                unsafe_allow_html=True)

    with st.expander("How to read this chart", expanded=False):
        st.markdown("""
**Top panel — WTI Oil Price (USD)**
Raw daily closing price with moving averages MA14/MA30/MA60. A price sustained above
MA60 indicates the move has outrun the structural trend historically a mean-reversion setup.

**Middle panel PRCSI and PSI**
- **White line (PRCSI):** Institutional narrative sentiment how extreme is current
  sentiment relative to the past 252 trading days. Oscillates around 50 by design:
  this is correct behaviour, not a visual artifact. Institutional sentiment is rarely
  at extremes, which is why signals fire infrequently and carry high accuracy when
  they do.
- **Orange dotted (PSI):** Price action sentiment how extreme is current price and
  momentum relative to all WTI history since 2007. Tracks price more closely because
  it uses an expanding window anchored to 2007, not a rolling one-year window.
- **Coloured bands:** Fear (blue, 0–45), Neutral (grey, 45–55), Greed (red, 55–100).
- **Triangles:** Active signals — bearish (red, PRCSI > 50) and bullish (green,
  PRCSI < 50). Both are contrarian: institutional greed signals a fall, institutional
  fear signals a rise.

**PRCSI and PSI measure different things the gap between them is the signal.**
When PSI runs well above PRCSI, price is moving faster than institutional narrative
justifies. This information asymmetry (Grossman-Stiglitz 1980) has historically
preceded reversals. When they converge, the asymmetry has closed.
The May 2026 Hormuz spike PSI ~74, PRCSI ~50, divergence ~21 points is a
live case study of this dynamic.

**Bottom panel Signal Severity**
`Severity = |PRCSI − 50| × 2`. Bars turn red above 0.2637, the train-frozen top 10% threshold.
Only red bars correspond to active signals with validated directional accuracy.
""")
    st.markdown("")

    if df is not None:
        fig2 = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.22, 0.52, 0.26],
            vertical_spacing=0.025,
            subplot_titles=["WTI Oil Price (USD)", "PRCSI / PSI", "Signal Severity"],
        )

        if "oil_price" in df.columns:
            price = df["oil_price"].ffill()
            fig2.add_trace(go.Scatter(
                x=df.index, y=price, mode="lines", name="WTI",
                line=dict(color="#fbbf24", width=1.4),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            ), row=1, col=1)
            for window, ma_name, clr, dash in [
                (14, "MA14", "#fb923c", "dot"),
                (30, "MA30", "#f87171", "dash"),
                (60, "MA60", "#e11d48", "solid"),
            ]:
                ma = price.rolling(window, min_periods=int(window * 0.5)).mean()
                fig2.add_trace(go.Scatter(
                    x=df.index, y=ma, mode="lines", name=ma_name,
                    line=dict(color=clr, width=0.9, dash=dash), opacity=0.6,
                    hovertemplate=(f"%{{x|%Y-%m-%d}}<br>{ma_name}:"
                                   f" $%{{y:.2f}}<extra></extra>"),
                ), row=1, col=1)

        for y0, y1, clr in [(0,25,"#0d1f5c"),(25,45,"#1e3f8a"),(45,55,"#1f2937"),
                             (55,75,"#7c2d12"),(75,100,"#450a0a")]:
            fig2.add_hrect(y0=y0, y1=y1, fillcolor=clr, opacity=0.26,
                           line_width=0, row=2, col=1)

        fig2.add_trace(go.Scatter(
            x=df.index, y=df["prcsi"], mode="lines", name="PRCSI",
            line=dict(color="#f9fafb", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
        ), row=2, col=1)

        if psi_df is not None and len(psi_df):
            fig2.add_trace(go.Scatter(
                x=psi_df.index, y=psi_df["psi"], mode="lines", name="PSI",
                line=dict(color="#f97316", width=1.2, dash="dot"), opacity=0.8,
                hovertemplate="%{x|%Y-%m-%d}<br>PSI: %{y:.1f}<extra></extra>",
            ), row=2, col=1)

        for direction, sym, clr in [("BEARISH","triangle-down","#f87171"),
                                     ("BULLISH","triangle-up","#4ade80")]:
            mask = df["signal_direction"] == direction
            if mask.any():
                fig2.add_trace(go.Scatter(
                    x=df.index[mask], y=df["prcsi"][mask],
                    mode="markers", name=direction,
                    marker=dict(symbol=sym, size=7, color=clr),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{direction} %{{y:.1f}}<extra></extra>",
                ), row=2, col=1)

        fig2.add_hline(y=50, line_width=0.7, line_dash="dot",
                       line_color="#6b7280", row=2, col=1)

        fig2.add_trace(go.Scatter(
            x=df.index, y=df["severity"],
            mode="lines", name="Severity",
            fill="tozeroy", line=dict(color="#3b82f6", width=1),
            fillcolor="rgba(59,130,246,0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        ), row=3, col=1)
        fig2.add_hline(y=0.2637, line_width=1, line_dash="dash",
                       line_color="#dc2626", opacity=0.6,
                       annotation_text="10% threshold",
                       annotation_font_size=9,
                       row=3, col=1)

        fig2.update_layout(
            **PLOT_LAYOUT,
            height=660,
            showlegend=True,
            legend=dict(font=dict(size=9, color="#9ca3af"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        fig2.update_xaxes(**AXIS_STYLE)
        fig2.update_yaxes(row=1, col=1, **AXIS_STYLE,
                          tickprefix="$", tickfont=dict(size=9))
        fig2.update_yaxes(row=2, col=1, range=[0, 100],
                          **AXIS_STYLE, tickfont=dict(size=9))
        fig2.update_yaxes(row=3, col=1, range=[0, 0.55],
                          **AXIS_STYLE, tickfont=dict(size=9))
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": True, "scrollZoom": True})

        st.markdown("<div class='section-header'>Distribution of PRCSI Readings</div>",
                    unsafe_allow_html=True)
        hist_clrs = [
            "#1d4ed8" if v < 25 else
            "#3b82f6" if v < 45 else
            "#6b7280" if v < 55 else
            "#f97316" if v < 75 else "#dc2626"
            for v in df["prcsi"].dropna()
        ]
        fig3 = go.Figure(go.Histogram(
            x=df["prcsi"].dropna(), nbinsx=50, marker_color=hist_clrs,
        ))
        fig3.add_vline(x=score, line_color="#f9fafb", line_width=1.4,
                       annotation_text=f"Today {score:.0f}",
                       annotation_font_size=9)
        fig3.update_layout(**PLOT_LAYOUT, height=190, bargap=0.02, showlegend=False)
        fig3.update_xaxes(title_text="PRCSI Score", range=[0, 100], **AXIS_STYLE)
        fig3.update_yaxes(title_text="Days", **AXIS_STYLE)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# TAB 3, SIGNAL LOG
with tab_signals:
    st.markdown("<div class='section-header'>Signal History — All Active Signal Days</div>",
                unsafe_allow_html=True)

    with st.expander("How signals work", expanded=False):
        st.markdown("""
**What triggers a signal?**
A signal fires when `severity = |PRCSI − 50| × 2` exceeds **0.2637** — the 90th percentile
of all severity readings in the 2007–2019 training period. This threshold is frozen and does
not update as new data arrives, preventing threshold leakage.

**Signal tiers**
| Tier | Severity | OOS accuracy | Note |
|---|---|---|---|
| Top 10% | >= 0.2637 | **86.8%** | 5 independent blocks — primary validated tier |
| Top 5% | >= 0.2879 | 96.5% | 2 blocks — exploratory |
| Top 2% | >= 0.3146 | 97.1% | Rare — treat as confirmation only |

**Bearish vs Bullish — the contrarian logic**
- **BEARISH** = PRCSI > 50 (institutional greed) price predicted to fall.
  Institutions overconfident, supply-side miscalculation likely, correction follows.
- **BULLISH** = PRCSI < 50 (institutional fear) price predicted to rise.
  Institutions too cautious, undersupply developing, recovery follows.

This is the opposite of equity fear and greed: here it is institutional narrative greed
that is the sell signal, not price momentum greed.

**Horizon:** 21–42 trading days. Not a day-trading signal.

**Regime sensitivity:** OOS accuracy was 98.7% in volatile regimes (2020–2022)
and 64.1% in stable regimes (2023–2026). Signals during geopolitical shocks or
major supply disruptions carry higher historical reliability.
""")
    st.markdown("")

    if df is not None and "signal_active" in df.columns:
        sig_df = df[df["signal_active"]].copy().sort_index(ascending=False)

        if len(sig_df) == 0:
            st.info("No signals have fired yet in the current dataset.")
        else:
            total     = len(sig_df)
            bearish_n = (sig_df["signal_direction"] == "BEARISH").sum()
            bullish_n = (sig_df["signal_direction"] == "BULLISH").sum()
            top5_n    = sig_df["signal_tier"].isin(["top_5", "top_2"]).sum()

            c1, c2, c3, c4 = st.columns(4)
            for col, lbl, val in [
                (c1, "Total signal days", total),
                (c2, "Bearish days",  bearish_n),
                (c3, "Bullish days",  bullish_n),
                (c4, "Top 5% days",   top5_n),
            ]:
                with col:
                    st.markdown(f"""<div class='stat-card'>
                      <div class='stat-label'>{lbl}</div>
                      <div class='stat-value'>{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            sig_df["year"] = sig_df.index.year
            by_year = sig_df.groupby(["year", "signal_direction"]).size().unstack(fill_value=0)
            fig4 = go.Figure()
            if "BEARISH" in by_year.columns:
                fig4.add_trace(go.Bar(name="Bearish", x=by_year.index,
                                      y=by_year["BEARISH"], marker_color="#f87171"))
            if "BULLISH" in by_year.columns:
                fig4.add_trace(go.Bar(name="Bullish", x=by_year.index,
                                      y=by_year["BULLISH"], marker_color="#4ade80"))
            fig4.update_layout(
                **PLOT_LAYOUT, barmode="stack", height=200,
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            )
            fig4.update_xaxes(title_text="Year", **AXIS_STYLE)
            fig4.update_yaxes(title_text="Signal days", **AXIS_STYLE)
            st.plotly_chart(fig4, use_container_width=True,
                            config={"displayModeBar": False})

            st.markdown("<div class='section-header'>Signal Days — most recent first</div>",
                        unsafe_allow_html=True)
            disp = sig_df.copy()
            disp["Date"]      = disp.index.strftime("%Y-%m-%d")
            disp["Direction"] = disp["signal_direction"]
            disp["Tier"]      = disp["signal_tier"]
            disp["Score"]     = disp["prcsi"].map("{:.1f}".format)
            disp["Severity"]  = disp["severity"].map("{:.4f}".format)

            st.dataframe(
                disp[["Date", "Direction", "Tier", "Score", "Severity"]].head(200),
                use_container_width=True,
                height=400,
                column_config={
                    "Direction": st.column_config.TextColumn(width="small"),
                    "Tier":      st.column_config.TextColumn(width="small"),
                    "Score":     st.column_config.TextColumn(width="small"),
                    "Severity":  st.column_config.TextColumn(width="small"),
                },
            )
