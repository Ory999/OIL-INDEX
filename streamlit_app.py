"""
PRCSI — Oil Fear & Greed Index
Streamlit frontend for the live index.
Reads from data/results/ committed by the GitHub Actions pipeline.

Deploy to Streamlit Community Cloud:
  1. Push this file and the updated requirements.txt to GitHub root
  2. go.streamlit.io → New app → connect repo → main file = streamlit_app.py
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Fear & Greed Index",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("data/results")
META_PATH   = RESULTS_DIR / "pipeline_metadata.json"
PARQUET     = RESULTS_DIR / "prcsi_final.parquet"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .main { background: #0a0f1e; }
  .block-container { padding-top: 1.5rem; max-width: 1200px; }

  /* Hero score */
  .score-hero {
    font-family: 'DM Mono', monospace;
    font-size: 5.5rem;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -0.02em;
    text-align: center;
  }
  .regime-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    text-align: center;
    margin-top: 0.3rem;
    opacity: 0.75;
  }

  /* Signal box */
  .signal-bearish {
    background: linear-gradient(135deg, #3f0d0d 0%, #1a0505 100%);
    border: 1px solid #dc262680;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
  }
  .signal-bullish {
    background: linear-gradient(135deg, #0d2e18 0%, #051a0b 100%);
    border: 1px solid #16a34a80;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
  }
  .signal-none {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
  }
  .signal-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.55;
    margin-bottom: 0.5rem;
  }
  .signal-direction {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }
  .signal-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    opacity: 0.65;
    line-height: 1.7;
  }

  /* Stat cards */
  .stat-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
  }
  .stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 0.35rem;
  }
  .stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
  }
  .stat-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    opacity: 0.5;
    margin-top: 0.2rem;
  }

  /* Section headers */
  .section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    opacity: 0.4;
    margin: 2rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1f2937;
  }

  /* NLP badge */
  .nlp-fresh   { color: #34d399; font-size: 0.75rem; }
  .nlp-stale   { color: #9ca3af; font-size: 0.75rem; }

  /* Disclaimer */
  .disclaimer {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    opacity: 0.3;
    line-height: 1.8;
    border-top: 1px solid #1f2937;
    padding-top: 1.5rem;
    margin-top: 2rem;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #1f2937;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 1.2rem;
    color: #9ca3af;
  }
  .stTabs [aria-selected="true"] {
    color: #f9fafb;
    background: #111827;
    border-radius: 6px 6px 0 0;
  }

  /* Plotly chart background */
  .js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Helper: regime colours ─────────────────────────────────────────────────────
REGIME_COLORS = {
    "EXTREME_FEAR": "#1d4ed8",
    "FEAR":         "#60a5fa",
    "NEUTRAL":      "#9ca3af",
    "GREED":        "#f97316",
    "EXTREME_GREED":"#dc2626",
}

def score_color(score: float) -> str:
    if score <= 25:   return REGIME_COLORS["EXTREME_FEAR"]
    elif score <= 45: return REGIME_COLORS["FEAR"]
    elif score <= 55: return REGIME_COLORS["NEUTRAL"]
    elif score <= 75: return REGIME_COLORS["GREED"]
    else:             return REGIME_COLORS["EXTREME_GREED"]

def regime_label(score: float) -> str:
    if score <= 25:   return "Extreme Fear"
    elif score <= 45: return "Fear"
    elif score <= 55: return "Neutral"
    elif score <= 75: return "Greed"
    else:             return "Extreme Greed"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#9ca3af", size=11),
    margin=dict(l=0, r=0, t=30, b=0),
)
AXIS_STYLE = dict(gridcolor="#1f2937", zeroline=False)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)  # refresh cache every 30 minutes
def load_metadata() -> dict:
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data(ttl=1800)
def load_index() -> pd.DataFrame | None:
    if not PARQUET.exists():
        return None
    df = pd.read_parquet(PARQUET)
    df.index = pd.to_datetime(df.index)
    return df

meta = load_metadata()
df   = load_index()

@st.cache_data(ttl=1800)
def load_psi() -> pd.DataFrame | None:
    path = RESULTS_DIR / "psi_final.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d.index = pd.to_datetime(d.index)
    return d

psi_df = load_psi()

# ── Derive current state ──────────────────────────────────────────────────────
score      = meta.get("prcsi_latest", 50.0)
severity   = meta.get("prcsi_severity", 0.0)
sig_active = meta.get("signal_active", False)
sig_dir    = meta.get("signal_direction", "NONE")
sig_tier   = meta.get("signal_tier", "none")
sig_acc_oos = meta.get("tier_accuracy_oos")
sig_acc_full = meta.get("tier_accuracy_full")
nlp_fresh  = meta.get("nlp_is_fresh", False)
sig_30d    = meta.get("signals_last_30d", 0)
run_ts     = meta.get("full_run_timestamp", "")
prcsi_date = meta.get("prcsi_date", "")

color      = score_color(score)
regime     = regime_label(score)

# PSI state
psi_score  = meta.get("psi_latest", 50.0)
psi_regime = meta.get("psi_regime", "NEUTRAL")
psi_rsi    = meta.get("psi_rsi_7")
divergence     = meta.get("divergence", 0.0) or 0.0
divergence_pct = meta.get("divergence_pct_pts", 0.0) or 0.0
div_direction  = meta.get("divergence_direction", "ALIGNED")
psi_color  = score_color(psi_score)
psi_label  = regime_label(psi_score)

# ── HEADER ────────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown("### 🛢️ &nbsp; WTI Oil Fear &amp; Greed Index")
with col_meta:
    if prcsi_date:
        st.markdown(
            f"<div style='text-align:right; font-family:DM Mono; "
            f"font-size:0.72rem; color:#6b7280; margin-top:0.6rem;'>"
            f"{'🟢 NLP fresh' if nlp_fresh else '🟡 NLP carried'} &nbsp;·&nbsp; "
            f"{prcsi_date}</div>",
            unsafe_allow_html=True
        )

st.markdown("---")

if df is None or len(df) == 0:
    st.warning("Pipeline has not produced output yet. "
               "Run the GitHub Actions econometric pipeline first.")
    st.stop()

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
tab_live, tab_history, tab_signals, tab_method = st.tabs([
    "Live Index", "Historical Chart", "Signal Log", "Methodology"
])

# ════════════════════════════════════════════════
# TAB 1 — LIVE INDEX
# ════════════════════════════════════════════════
with tab_live:

    # ── Three-column layout: PRCSI | Divergence | PSI ──────────────────
    col_prcsi, col_div, col_psi = st.columns([1, 0.7, 1])

    def make_gauge(value, color, height=240):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 56, "color": color, "family": "DM Mono, monospace"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickfont": {"size": 9, "color": "#6b7280"},
                    "tickvals": [0, 25, 55, 75, 100],
                },
                "bar": {"color": color, "thickness": 0.28},
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
                    "thickness": 0.85, "value": value,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="#9ca3af"),
            margin=dict(l=10, r=10, t=5, b=0),
            height=height,
        )
        return fig

    with col_prcsi:
        st.markdown("<div style='text-align:center; font-family:DM Mono; font-size:0.7rem; "
                    "letter-spacing:0.15em; opacity:0.4; text-transform:uppercase; "
                    "margin-bottom:0.3rem;'>PRCSI — Institutional</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_gauge(score, color),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='regime-label' style='color:{color};'>{regime}</div>",
                    unsafe_allow_html=True)

    with col_div:
        st.markdown("<div style='text-align:center; font-family:DM Mono; font-size:0.7rem; "
                    "letter-spacing:0.15em; opacity:0.4; text-transform:uppercase; "
                    "margin-bottom:0.3rem;'>Divergence</div>",
                    unsafe_allow_html=True)
        # Divergence visual
        div_color = ("#dc2626" if div_direction == "PSI_LEADS"
                     else "#3b82f6" if div_direction == "PRCSI_LEADS"
                     else "#6b7280")
        div_sign  = "▲" if divergence_pct > 0 else "▼"
        div_label = {
            "PSI_LEADS":   "Price Ahead",
            "PRCSI_LEADS": "Narrative Ahead",
            "ALIGNED":     "Aligned",
        }.get(div_direction, "Aligned")

        st.markdown(f"""
        <div style='text-align:center; padding:1.5rem 0.5rem;'>
          <div style='font-family:DM Mono; font-size:2.4rem; font-weight:500;
                      color:{div_color}; line-height:1;'>
            {div_sign} {abs(divergence_pct):.1f}
          </div>
          <div style='font-family:DM Sans; font-size:0.75rem; opacity:0.6;
                      margin-top:0.4rem;'>points</div>
          <div style='font-family:DM Mono; font-size:0.7rem; color:{div_color};
                      letter-spacing:0.1em; text-transform:uppercase;
                      margin-top:0.6rem;'>{div_label}</div>
          <div style='font-family:DM Mono; font-size:0.62rem; opacity:0.4;
                      margin-top:0.8rem; line-height:1.6;'>
            PRCSI&nbsp;{score:.1f}<br>
            PSI&nbsp;&nbsp;&nbsp;&nbsp;{psi_score:.1f}
          </div>
        </div>
        <div style='font-family:DM Mono; font-size:0.62rem; opacity:0.35;
                    text-align:center; padding:0 0.5rem; line-height:1.6;'>
          {"⚡ Price running ahead of institutional narrative" if div_direction=="PSI_LEADS"
           else "📣 Narrative ahead of price" if div_direction=="PRCSI_LEADS"
           else "Institutional narrative and price aligned"}
        </div>""", unsafe_allow_html=True)

    with col_psi:
        st.markdown("<div style='text-align:center; font-family:DM Mono; font-size:0.7rem; "
                    "letter-spacing:0.15em; opacity:0.4; text-transform:uppercase; "
                    "margin-bottom:0.3rem;'>PSI — Price Action</div>",
                    unsafe_allow_html=True)
        psi_available = psi_df is not None and len(psi_df) > 0
        st.plotly_chart(make_gauge(psi_score if psi_available else 50.0,
                                   psi_color if psi_available else "#6b7280"),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='regime-label' style='color:{psi_color};'>{psi_label}</div>",
                    unsafe_allow_html=True)
        if psi_rsi is not None:
            st.markdown(f"<div style='text-align:center; font-family:DM Mono; "
                        f"font-size:0.68rem; opacity:0.45; margin-top:0.3rem;'>"
                        f"RSI(7): {psi_rsi:.0f}</div>",
                        unsafe_allow_html=True)

    with col_prcsi:
        pass  # signal box goes below in new layout

    # ── Signal + Stats below gauges ────────────────────────────────────
    col_right_sig, col_right_stats = st.columns([1, 1])
    with col_right_sig:
        # ── Signal status ─────────────────────────────────────────────
        st.markdown("<div class='section-header'>Active Signal</div>",
                    unsafe_allow_html=True)

        if sig_active:
            box_class = "signal-bearish" if sig_dir == "BEARISH" else "signal-bullish"
            dir_color = "#f87171" if sig_dir == "BEARISH" else "#4ade80"
            tier_map  = {"top_2": "Top 2% ★★★", "top_5": "Top 5% ★★", "top_10": "Top 10% ★"}
            tier_str  = tier_map.get(sig_tier, sig_tier)
            acc_oos   = f"{sig_acc_oos:.1%}" if sig_acc_oos else "n/a"
            acc_full  = f"{sig_acc_full:.1%}" if sig_acc_full else "n/a"

            st.markdown(f"""
            <div class='{box_class}'>
              <div class='signal-title'>Signal active</div>
              <div class='signal-direction' style='color:{dir_color};'>
                {'▼ BEARISH' if sig_dir == "BEARISH" else '▲ BULLISH'}
              </div>
              <div class='signal-sub'>
                Tier:&nbsp; {tier_str}<br>
                Severity:&nbsp; {severity:.4f}<br>
                Horizon:&nbsp; 21–42 trading days<br>
                OOS accuracy:&nbsp; <b style='color:{dir_color};'>{acc_oos}</b><br>
                Full-sample:&nbsp; {acc_full}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='signal-none'>
              <div class='signal-title'>No active signal</div>
              <div style='color:#6b7280; font-size:1rem; font-weight:500; margin:0.4rem 0;'>
                Watching
              </div>
              <div class='signal-sub'>
                Severity {severity:.4f} is below the action<br>
                threshold of 0.2637 (train-frozen top 10%).<br><br>
                Signals in last 30 days:&nbsp; {sig_30d}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Quick stats ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>Index State</div>",
                    unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>Score</div>
              <div class='stat-value' style='color:{color};'>{score:.1f}</div>
              <div class='stat-sub'>0 = max fear · 100 = max greed</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>Severity</div>
              <div class='stat-value'>{severity:.4f}</div>
              <div class='stat-sub'>threshold 0.264 (top 10%)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        s3, s4 = st.columns(2)
        # 7-day trend
        if df is not None and "prcsi" in df.columns:
            week_ago  = df["prcsi"].dropna().iloc[:-5].iloc[-1] if len(df) > 5 else score
            delta_7d  = score - float(week_ago)
            arrow     = "↑" if delta_7d > 0 else "↓"
            d_color   = REGIME_COLORS["GREED"] if delta_7d > 0 else REGIME_COLORS["FEAR"]
        else:
            delta_7d, arrow, d_color = 0, "→", "#9ca3af"

        with s3:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>7-day change</div>
              <div class='stat-value' style='color:{d_color};'>{arrow} {abs(delta_7d):.1f}</div>
              <div class='stat-sub'>points</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>Signals / 30d</div>
              <div class='stat-value'>{sig_30d}</div>
              <div class='stat-sub'>top 10% firings</div>
            </div>""", unsafe_allow_html=True)

    # ── Index history chart with range selector ──────────────────────
    st.markdown("<div class='section-header'>Index History</div>", unsafe_allow_html=True)

    if df is not None:
        # Range selector — default to full history to show context back to 2008
        range_col1, range_col2 = st.columns([3, 1])
        with range_col2:
            time_range = st.selectbox(
                "Range", ["3M", "1Y", "3Y", "5Y", "All (2007–)"],
                index=4,   # default: full history
                label_visibility="collapsed",
            )
        range_days = {
            "3M":          90,
            "1Y":          365,
            "3Y":          365 * 3,
            "5Y":          365 * 5,
            "All (2007–)": 99999,
        }[time_range]
        cutoff = df.index.max() - pd.Timedelta(days=range_days)
        recent = df[df.index >= cutoff]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )

        # PRCSI bands
        for y0, y1, clr in [(0,25,"#0d1f5c"),(25,45,"#1e3f8a"),(45,55,"#1f2937"),
                             (55,75,"#7c2d12"),(75,100,"#450a0a")]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, opacity=0.35,
                          line_width=0, row=1, col=1)

        fig.add_trace(go.Scatter(
            x=recent.index, y=recent["prcsi"],
            mode="lines", name="PRCSI",
            line=dict(color="#f9fafb", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
        ), row=1, col=1)
        # PSI overlay
        if psi_df is not None:
            psi_recent = psi_df[psi_df.index >= recent.index.min()]
            if len(psi_recent):
                fig.add_trace(go.Scatter(
                    x=psi_recent.index, y=psi_recent["psi"],
                    mode="lines", name="PSI",
                    line=dict(color="#f97316", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>PSI: %{y:.1f}<extra></extra>",
                ), row=1, col=1)

        # Signal markers
        bears = recent[recent["signal_direction"] == "BEARISH"]
        bulls = recent[recent["signal_direction"] == "BULLISH"]
        if len(bears):
            fig.add_trace(go.Scatter(
                x=bears.index, y=bears["prcsi"],
                mode="markers", name="Bearish",
                marker=dict(symbol="triangle-down", size=10, color="#f87171"),
                hovertemplate="%{x|%Y-%m-%d}<br>BEARISH %{y:.1f}<extra></extra>",
            ), row=1, col=1)
        if len(bulls):
            fig.add_trace(go.Scatter(
                x=bulls.index, y=bulls["prcsi"],
                mode="markers", name="Bullish",
                marker=dict(symbol="triangle-up", size=10, color="#4ade80"),
                hovertemplate="%{x|%Y-%m-%d}<br>BULLISH %{y:.1f}<extra></extra>",
            ), row=1, col=1)

        # Threshold lines
        fig.add_hline(y=50, line_width=1, line_dash="dot",
                      line_color="#6b7280", row=1, col=1)

        # Severity panel
        fig.add_trace(go.Bar(
            x=recent.index, y=recent["severity"],
            name="Severity",
            marker_color=[
                "#dc2626" if v >= 0.2637 else "#374151"
                for v in recent["severity"].fillna(0)
            ],
            hovertemplate="%{x|%Y-%m-%d}<br>Severity: %{y:.4f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0.2637, line_width=1, line_dash="dash",
                      line_color="#dc2626", opacity=0.6, row=2, col=1)

        fig.update_layout(
            **PLOT_LAYOUT,
            height=380,
            showlegend=True,
            legend=dict(
                font=dict(size=10, color="#9ca3af"),
                bgcolor="rgba(0,0,0,0)",
                x=1, y=1,
                xanchor="right",
            ),
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(row=1, col=1, range=[0, 100], **AXIS_STYLE, tickfont=dict(size=10))
        fig.update_yaxes(row=2, col=1, range=[0, 0.55], **AXIS_STYLE, tickfont=dict(size=10), title_text="Severity")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════
# TAB 2 — HISTORICAL CHART
# ════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='section-header'>Full History — PRCSI vs WTI Price</div>",
                unsafe_allow_html=True)

    if df is not None:
        fig2 = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.15, 0.55, 0.3],
            vertical_spacing=0.03,
            subplot_titles=["WTI Oil Price (USD)", "PRCSI Fear & Greed", "Signal Severity"],
        )

        # WTI price + moving averages
        if "oil_price" in df.columns:
            price = df["oil_price"].ffill()
            fig2.add_trace(go.Scatter(
                x=df.index, y=price,
                mode="lines", name="WTI",
                line=dict(color="#f97316", width=1.4),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            ), row=1, col=1)
            # Moving averages — shorter = lighter, longer = more opaque
            ma_configs = [
                (7,  "MA7",  "#fbbf24", 0.5, "dot"),
                (14, "MA14", "#fb923c", 0.65, "dash"),
                (30, "MA30", "#f87171", 0.8, "dashdot"),
                (60, "MA60", "#e11d48", 1.0, "solid"),
            ]
            for window, name, color, opacity, dash in ma_configs:
                ma = price.rolling(window, min_periods=int(window*0.5)).mean()
                fig2.add_trace(go.Scatter(
                    x=df.index, y=ma,
                    mode="lines", name=name,
                    line=dict(color=color, width=1.0, dash=dash),
                    opacity=opacity,
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: $%{{y:.2f}}<extra></extra>",
                ), row=1, col=1)

        # PRCSI bands
        for y0, y1, clr in [(0,25,"#0d1f5c"),(25,45,"#1e3f8a"),(45,55,"#1f2937"),
                             (55,75,"#7c2d12"),(75,100,"#450a0a")]:
            fig2.add_hrect(y0=y0, y1=y1, fillcolor=clr, opacity=0.3,
                           line_width=0, row=2, col=1)

        fig2.add_trace(go.Scatter(
            x=df.index, y=df["prcsi"],
            mode="lines", name="PRCSI",
            line=dict(color="#f9fafb", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
        ), row=2, col=1)
        if psi_df is not None and len(psi_df):
            fig2.add_trace(go.Scatter(
                x=psi_df.index, y=psi_df["psi"],
                mode="lines", name="PSI",
                line=dict(color="#f97316", width=1.2, dash="dot"),
                opacity=0.8,
                hovertemplate="%{x|%Y-%m-%d}<br>PSI: %{y:.1f}<extra></extra>",
            ), row=2, col=1)
        if psi_df is not None and len(psi_df):
            fig2.add_trace(go.Scatter(
                x=psi_df.index, y=psi_df["psi"],
                mode="lines", name="PSI",
                line=dict(color="#f97316", width=1.2, dash="dot"),
                opacity=0.8,
                hovertemplate="%{x|%Y-%m-%d}<br>PSI: %{y:.1f}<extra></extra>",
            ), row=2, col=1)

        # Signal markers on main chart
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

        fig2.add_hline(y=50, line_width=0.8, line_dash="dot",
                       line_color="#6b7280", row=2, col=1)

        # Severity
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["severity"],
            mode="lines", name="Severity",
            fill="tozeroy",
            line=dict(color="#3b82f6", width=1),
            fillcolor="rgba(59,130,246,0.15)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        ), row=3, col=1)
        fig2.add_hline(y=0.2637, line_width=1, line_dash="dash",
                       line_color="#dc2626", opacity=0.7,
                       annotation_text="10% threshold",
                       annotation_font_size=9,
                       row=3, col=1)

        fig2.update_layout(
            **PLOT_LAYOUT,
            height=640,
            showlegend=True,
            legend=dict(font=dict(size=10, color="#9ca3af"),
                        bgcolor="rgba(0,0,0,0)"),
        )
        fig2.update_xaxes(**AXIS_STYLE)
        fig2.update_yaxes(row=1, col=1, **AXIS_STYLE)
        fig2.update_yaxes(row=2, col=1, range=[0, 100], **AXIS_STYLE)
        fig2.update_yaxes(row=3, col=1, range=[0, 0.55], **AXIS_STYLE)
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": True, "scrollZoom": True})

        # Distribution of readings
        st.markdown("<div class='section-header'>Distribution of Readings</div>",
                    unsafe_allow_html=True)
        fig3 = go.Figure(go.Histogram(
            x=df["prcsi"].dropna(),
            nbinsx=50,
            marker_color=[
                "#1d4ed8" if v < 25 else
                "#3b82f6" if v < 45 else
                "#6b7280" if v < 55 else
                "#f97316" if v < 75 else "#dc2626"
                for v in df["prcsi"].dropna()
            ],
        ))
        fig3.add_vline(x=score, line_color="#f9fafb", line_width=2,
                       annotation_text=f"Today: {score:.0f}",
                       annotation_font_size=10)
        fig3.update_layout(
            **PLOT_LAYOUT,
            height=220,
            bargap=0.02,
            showlegend=False,
        )
        fig3.update_xaxes(title_text="PRCSI Score", range=[0,100], **AXIS_STYLE)
        fig3.update_yaxes(title_text="Days", **AXIS_STYLE)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════
# TAB 3 — SIGNAL LOG
# ════════════════════════════════════════════════
with tab_signals:
    st.markdown("<div class='section-header'>Signal History — All Active Signals</div>",
                unsafe_allow_html=True)

    if df is not None and "signal_active" in df.columns:
        sig_df = df[df["signal_active"]].copy()
        sig_df = sig_df.sort_index(ascending=False)

        if len(sig_df) == 0:
            st.info("No signals have fired yet in the current dataset.")
        else:
            # Group into signal episodes (consecutive active days)
            sig_df["date_str"] = sig_df.index.strftime("%Y-%m-%d")
            sig_df["score_str"] = sig_df["prcsi"].map("{:.1f}".format)
            sig_df["sev_str"]   = sig_df["severity"].map("{:.4f}".format)

            display_df = sig_df[[
                "date_str", "signal_direction", "signal_tier",
                "score_str", "sev_str"
            ]].rename(columns={
                "date_str":        "Date",
                "signal_direction":"Direction",
                "signal_tier":     "Tier",
                "score_str":       "Score",
                "sev_str":         "Severity",
            })

            # Summary stats
            c1, c2, c3, c4 = st.columns(4)
            total = len(sig_df)
            bearish_n = (sig_df["signal_direction"] == "BEARISH").sum()
            bullish_n = (sig_df["signal_direction"] == "BULLISH").sum()
            top5_n    = (sig_df["signal_tier"].isin(["top_5", "top_2"])).sum()

            for col, label, val in [
                (c1, "Total signal days", str(total)),
                (c2, "Bearish days",  str(bearish_n)),
                (c3, "Bullish days",  str(bullish_n)),
                (c4, "Top 5% days",   str(top5_n)),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='stat-card'>
                      <div class='stat-label'>{label}</div>
                      <div class='stat-value'>{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Bar chart of signal frequency by year
            sig_df["year"] = sig_df.index.year
            by_year = sig_df.groupby(["year","signal_direction"]).size().unstack(fill_value=0)
            fig4 = go.Figure()
            if "BEARISH" in by_year.columns:
                fig4.add_trace(go.Bar(name="Bearish", x=by_year.index,
                                      y=by_year["BEARISH"], marker_color="#f87171"))
            if "BULLISH" in by_year.columns:
                fig4.add_trace(go.Bar(name="Bullish", x=by_year.index,
                                      y=by_year["BULLISH"], marker_color="#4ade80"))
            fig4.update_layout(
                **PLOT_LAYOUT, barmode="stack", height=220,
                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            )
            fig4.update_xaxes(title_text="Year", **AXIS_STYLE)
            fig4.update_yaxes(title_text="Signal days", **AXIS_STYLE)
            st.plotly_chart(fig4, use_container_width=True,
                            config={"displayModeBar": False})

            # Data table
            st.markdown("<div class='section-header'>Signal Days (most recent first)</div>",
                        unsafe_allow_html=True)
            st.dataframe(
                display_df.head(200),
                use_container_width=True,
                height=420,
                column_config={
                    "Direction": st.column_config.TextColumn(width="small"),
                    "Tier":      st.column_config.TextColumn(width="small"),
                    "Score":     st.column_config.TextColumn(width="small"),
                    "Severity":  st.column_config.TextColumn(width="small"),
                },
            )


# ════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ════════════════════════════════════════════════
with tab_method:
    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("""
#### What is this index?

The **PRCSI (Petroleum Risk & Conviction Sentiment Index)** is a contrarian
institutional sentiment indicator for WTI crude oil. It aggregates signals from
EIA inventory data, COT positioning, macro controls, and NLP-scored
institutional publications (OPEC MOMR, EIA STEO, Saudi Aramco).

#### How is it constructed?

**9 stable features** are used — those significant in ≥20% of rolling Granger
causality windows on the 2007–2019 training period:

| Feature | Group | Direction |
|---|---|---|
| crude_stocks_change | EIA Fundamentals | Bearish when positive |
| eia_surprise_norm | EIA Fundamentals | Bearish when positive |
| refinery_util_pct | EIA Fundamentals | Bullish when high |
| usd_logret | Macro Controls | Bearish when positive |
| cot_net_long | COT Positioning | Bullish when high |
| sent_ema_cross | NLP Momentum | Bullish when positive |
| divergence_ema | NLP Momentum | Greed signal |
| oil_impact_score | Raw LLM | Bullish when positive |
| institutional_confidence | Raw LLM | Bullish when high |

Each feature is direction-corrected then **252-day rolling percentile ranked**
and combined using Granger-derived weights × group weights (EIA 2.5×,
Macro 2.5×, COT 1×, NLP 1×, Raw LLM 0.5×). Smoothed with EMA(span=63).
""")

    with col_m2:
        st.markdown("""
#### Prediction layer

The index is **contrarian**: high readings (greed) predict price falls.

| Severity | Tier | Signal |
|---|---|---|
| ≥ 0.2637 | Top 10% ★ | Active |
| ≥ 0.2879 | Top 5% ★★ | Active |
| ≥ 0.3146 | Top 2% ★★★ | Active |
| < 0.2637 | — | No signal |

Severity = \|index − 0.5\| × 2

**Validated out-of-sample performance (2020–2026, 21-day horizon):**

| Tier | Full-sample | OOS | Blocks |
|---|---|---|---|
| Top 10% | 67.7% | **86.8%** | 5 ✅ |
| Top 5% | 80.3% | 96.5% | 2 ⚠️ |

Block bootstrap p < 0.001. Price contrarian baseline: 49.9%.

#### Limitations

- Horizon: 21–42 trading days. **Not a day-trading signal.**
- OOS accuracy is regime-sensitive: 98.7% in volatile periods (2020–2022),
  64.1% in stable periods (2023–2026).
- NLP data is fresh on ~5.5% of days (publication days only).
  Other days carry the most recent publication forward.
- The macro orthogonalisation OLS result (p=0.0001) is inflated by
  overlapping returns. HAC-corrected p = 0.2313 (not formally significant).
- Single OOS era: results may not generalise to unseen regimes.
""")

    st.markdown("""
#### Data sources

| Source | Frequency | Coverage |
|---|---|---|
| EIA Weekly Petroleum Report | Weekly | 2007–present |
| CFTC Commitments of Traders | Weekly | 2007–present |
| FRED Macro (Fed Funds, USD) | Daily | 2007–present |
| OPEC Monthly Oil Market Report | Monthly | 2007–present |
| EIA Short-Term Energy Outlook | Monthly | 2007–present |
| Saudi Aramco press coverage | Daily | 2020–present |

Pipeline runs Mon–Fri at ~07:00 UTC via GitHub Actions.
""")

# ── FOOTER ────────────────────────────────────────────────────────────────────
last_update = run_ts[:10] if run_ts else "unknown"
st.markdown(f"""
<div class='disclaimer'>
  PRCSI v1.0 &nbsp;·&nbsp; Last pipeline run: {last_update} &nbsp;·&nbsp;
  This index is for research and informational purposes only. &nbsp;
  It does not constitute financial advice or a recommendation to buy or sell
  any security or commodity. &nbsp; Past accuracy at a given signal tier does
  not guarantee future results. &nbsp; Results are strongest in high-volatility
  regimes and may be weaker in stable market conditions. &nbsp;
  Always conduct independent analysis before making investment decisions. &nbsp;
  Source: <a href='https://github.com/Ory999/OIL-INDEX' style='color:#374151;'>
  github.com/Ory999/OIL-INDEX</a>
</div>
""", unsafe_allow_html=True)
