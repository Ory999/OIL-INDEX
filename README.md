# 🛢️ PRCSI — Oil Fear & Greed Index

**Petroleum Risk & Conviction Sentiment Index** — A contrarian institutional sentiment indicator for WTI crude oil, combining NLP-scored institutional communications with quantitative fundamentals.

[![Pipeline](https://github.com/Ory999/OIL-INDEX/actions/workflows/econometric%20pipeline.yml/badge.svg)](https://github.com/Ory999/OIL-INDEX/actions)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://oil-fear-greed.streamlit.app)

---

## What It Does

The PRCSI reads official institutional publications — OPEC Monthly Oil Market Reports, EIA Short-Term Energy Outlooks, and Saudi Aramco press coverage — and scores them for directional oil market sentiment using a locally-hosted LLM. These NLP signals are combined with EIA inventory data, CFTC COT positioning, and FRED macro controls into a single 0–100 Fear & Greed gauge.

**The index is contrarian:** high readings (greed) predict price falls. When severity exceeds the train-frozen 90th percentile threshold (0.2637), a directional signal is issued.

**Validated out-of-sample performance (2020–2026, 21-day horizon):**

| Signal tier | Full-sample accuracy | OOS accuracy | Independent blocks |
|---|---|---|---|
| Top 10% ★ | 67.7% | **86.8%** | 5 ✅ |
| Top 5% ★★ | 80.3% | 96.5% | 2 ⚠️ exploratory |

Block bootstrap p < 0.001. Price-based contrarian baseline: 49.9%.

> ⚠️ **Not financial advice.** The index is for research purposes only. Horizon is 21–42 trading days — not a day-trading signal. OOS accuracy is regime-sensitive (98.7% in volatile periods, 64.1% in stable).

---

## Live Index

The Streamlit dashboard updates Mon–Fri after each pipeline run:

**[oil-fear-greed.streamlit.app](https://oil-fear-greed.streamlit.app)**

---

## Architecture

```
Quantitative Pipeline (06:00 UTC)
  ├── 01 fetch prices.py         — WTI, VIX, USD (yfinance)
  ├── 02 fetch eia.py            — EIA inventory data (API)
  ├── 03 fetch fred.py           — Fed funds, USD broad (FRED API)
  ├── 04 fetch cot.py            — CFTC managed money positioning
  ├── 06 assemble master.py      — Merge into master_quant.parquet
  └── 07 build quant index.py    — Partial fundamentals-only index

Qualitative NLP Pipeline (06:30 UTC)
  ├── 13 collect opec.py         — OPEC MOMR PDFs
  ├── 15 collect aramco.py       — Aramco news via Google RSS
  ├── 16 collect eia steo.py     — EIA STEO PDFs
  ├── 18 build corpus.py         — Merge into combined corpus
  ├── 20 llm scoring.py          — LLM multidimensional scoring
  ├── 21 bertopic clustering.py  — Topic discovery
  ├── 22 sentiment momentum.py   — EMA/RSI/acceleration features
  └── 23 merge nlp master.py     — Join NLP onto quant master

Econometric Pipeline (triggered after qualitative)
  ├── 08 stationarity.py         — ADF tests
  ├── 09 granger causality.py    — Feature significance (full-sample, descriptive)
  ├── 10 var irf.py              — VAR model + impulse response
  ├── 11 shap analysis.py        — Feature importance
  └── 12 build full index.py     — Full PRCSI index + prediction layer

Streamlit Frontend
  └── streamlit_app.py           — Live dashboard (reads data/results/)
```

---

## Index Construction

**9 stable features** (significant in ≥20% of rolling Granger windows, 2007–2019 training period):

| Feature | Group (weight) | Direction |
|---|---|---|
| `crude_stocks_change` | EIA Fundamentals (2.5×) | Bearish ↑ |
| `eia_surprise_norm` | EIA Fundamentals (2.5×) | Bearish ↑ |
| `refinery_util_pct` | EIA Fundamentals (2.5×) | Bullish ↑ |
| `usd_logret` | Macro Controls (2.5×) | Bearish ↑ |
| `cot_net_long` | COT Positioning (1.0×) | Bullish ↑ |
| `sent_ema_cross` | NLP Momentum (1.0×) | Bullish ↑ |
| `divergence_ema` | NLP Momentum (1.0×) | Greed signal |
| `oil_impact_score` | Raw LLM (0.5×) | Bullish ↑ |
| `institutional_confidence` | Raw LLM (0.5×) | Bullish ↑ |

Each feature is direction-corrected, 252-day rolling percentile ranked, combined with Granger-derived weights, then smoothed with EMA(span=63).

**Prediction:** `severity = |index - 0.5| × 2`. Active signal when `severity ≥ 0.2637` (train-frozen). Direction: `index > 0.5 → BEARISH` (contrarian).

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### GitHub Secrets required

| Secret | Purpose |
|---|---|
| `EIA_API_KEY` | EIA Open Data API key (free at eia.gov) |
| `FRED_API_KEY` | FRED API key (free at fred.stlouisfed.org) |
| `LLM_BASE_URL` | Base URL of locally-hosted LLM via ngrok |

### LLM Setup

The qualitative pipeline requires a locally-hosted LLM (tested with `gpt-oss-20b` via LM Studio + ngrok). The system prompt is locked to match the historic backfill — do not modify `SYSTEM_PROMPT` in `scripts/qual/20 llm scoring.py` without re-scoring the full corpus.

### Running locally

```bash
# Quantitative pipeline
python3 "scripts/quant/01 fetch prices.py"
python3 "scripts/quant/02 fetch eia.py"
python3 "scripts/quant/06 assemble master.py"
python3 "scripts/quant/12 build full index.py"

# Streamlit frontend
streamlit run streamlit_app.py
```

---

## Data Sources

| Source | Frequency | Coverage |
|---|---|---|
| EIA Weekly Petroleum Status Report | Weekly | 2007–present |
| CFTC Commitments of Traders | Weekly | 2007–present |
| FRED (Fed Funds, USD Broad) | Daily | 2007–present |
| OPEC Monthly Oil Market Report | Monthly | 2007–present |
| EIA Short-Term Energy Outlook | Monthly | 2007–present |
| Saudi Aramco news (Google RSS) | Daily | 2020–present |

---

## Limitations

- **Single OOS era** — results tested on 2020–2026 only; performance in other regimes is unknown
- **Regime sensitivity** — accuracy was 98.7% in volatile periods (2020–2022) and 64.1% in stable periods (2023–2026)
- **NLP freshness** — only ~5.5% of daily values are genuine publication-day data; the rest are forward-filled
- **Macro orthogonalisation** — OLS p-values are inflated by overlapping returns; HAC-corrected p=0.2313 (not formally significant)
- **Direction only** — the index predicts direction, not magnitude or timing within the horizon

---

## Citation

If you use this index in academic work, please cite:

```bibtex
@misc{prcsi2026,
  author  = {Ory999},
  title   = {PRCSI: Petroleum Risk \& Conviction Sentiment Index},
  year    = {2026},
  url     = {https://github.com/Ory999/OIL-INDEX},
  note    = {WTI oil Fear \& Greed index combining NLP institutional scoring with quantitative fundamentals}
}
```

---

## License

MIT — see [LICENSE](LICENSE)
