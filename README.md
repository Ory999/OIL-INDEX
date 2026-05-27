# PRCSI — Oil Fear & Greed Index

**Petroleum Risk & Conviction Sentiment Index.** A contrarian institutional sentiment indicator for WTI crude oil, combining NLP-scored institutional communications with quantitative fundamentals.

[![Pipeline](https://github.com/Ory999/OIL-INDEX/actions/workflows/econometric%20pipeline.yml/badge.svg)](https://github.com/Ory999/OIL-INDEX/actions)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://oil-fear-greed.streamlit.app)

---

## What it does

The PRCSI reads official institutional publications (OPEC Monthly Oil Market Reports, EIA Short-Term Energy Outlooks, Saudi Aramco press coverage) and scores them for directional oil market sentiment using a locally hosted LLM. The NLP signals are combined with EIA inventory data, CFTC COT positioning, and FRED macro controls into a single 0 to 100 Fear & Greed gauge.

The index is contrarian. High readings (greed) predict price falls. When severity exceeds the train-frozen 90th percentile threshold (0.2637), a directional signal is issued.

Validated out-of-sample performance, 2020 to 2026, 21-day horizon:

| Signal tier | Full-sample accuracy | OOS accuracy | Independent blocks |
|---|---|---|---|
| Top 10% | 67.7% | **86.8%** | 5 |
| Top 5% | 80.3% | 96.5% | 2, exploratory |

Block bootstrap p < 0.001. Price-based contrarian baseline 49.9%.

> Not financial advice. The index is for research purposes only. Horizon is 21 to 42 trading days, not a day-trading signal. OOS accuracy is regime-sensitive (98.7% in volatile periods, 64.1% in stable).

---

## Live index

The Streamlit dashboard updates Mon to Fri after each pipeline run:

**[oil-fear-greed.streamlit.app](https://oil-fear-greed.streamlit.app)**

---

## Architecture

```
Quantitative Pipeline (06:00 UTC)
  ├── Quant 1 fetch prices.py         WTI, VIX, USD (yfinance)
  ├── Quant 2 fetch eia.py            EIA inventory data (API)
  ├── Quant 3 fetch fred.py           Fed funds, USD broad (FRED API)
  ├── Quant 4 fetch cot.py            CFTC managed money positioning
  ├── Quant 5 assemble master.py      Merge into master_quant.parquet
  ├── Quant 6 build quant index.py    Partial fundamentals-only index
  └── Quant 12 build psi index.py     Price Sentiment Index (PSI)

Qualitative NLP Pipeline (06:30 UTC)
  ├── Qual 1 collect opec.py          OPEC MOMR PDFs
  ├── Qual 5 collect aramco.py        Aramco news via Google RSS
  ├── Qual 3 collect eia steo.py      EIA STEO PDFs
  ├── Qual 4 build corpus.py          Merge into combined corpus
  ├── Qual 7 llm scoring.py           LLM multidimensional scoring
  ├── Qual 6 bertopic clustering.py   Topic discovery
  ├── Qual 9 sentiment momentum.py    EMA/RSI/acceleration features
  └── Qual 8 merge nlp master.py      Join NLP onto quant master

Econometric Pipeline (triggered after qualitative)
  ├── Quant 7 stationarity.py         ADF tests
  ├── Quant 8 granger causality.py    Feature significance (full-sample, descriptive)
  ├── Quant 9 var irf.py              VAR model and impulse response
  ├── Quant 10 shap analysis.py       Feature importance
  └── Quant 11 build full index.py    Full PRCSI index and prediction layer

Streamlit Frontend
  └── streamlit_app.py                Live dashboard (reads data/results/)
```

---

## Index construction

9 stable features (significant in 20% or more of rolling Granger windows, 2007 to 2019 training period):

| Feature | Group (weight) | Direction |
|---|---|---|
| `crude_stocks_change` | EIA Fundamentals (2.5×) | Bearish |
| `eia_surprise_norm` | EIA Fundamentals (2.5×) | Bearish |
| `refinery_util_pct` | EIA Fundamentals (2.5×) | Bullish |
| `usd_logret` | Macro Controls (2.5×) | Bearish |
| `cot_net_long` | COT Positioning (1.0×) | Bullish |
| `sent_ema_cross` | NLP Momentum (1.0×) | Bullish |
| `divergence_ema` | NLP Momentum (1.0×) | Greed signal |
| `oil_impact_score` | Raw LLM (0.5×) | Bullish |
| `institutional_confidence` | Raw LLM (0.5×) | Bullish |

Each feature is direction-corrected, 252-day rolling percentile ranked, combined with Granger-derived weights, then smoothed with EMA(span=63).

Prediction: `severity = |index - 0.5| × 2`. Active signal when `severity ≥ 0.2637` (train-frozen). Direction: `index > 0.5 → BEARISH` (contrarian).

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
| `LLM_BASE_URL` | Base URL of locally hosted LLM via ngrok |

### LLM setup

The qualitative pipeline requires a locally hosted LLM (tested with `gpt-oss-20b` via LM Studio and ngrok). The system prompt is locked to match the historic backfill. Do not modify `SYSTEM_PROMPT` in `scripts/qual/Qual 7 llm scoring.py` without re-scoring the full corpus.

### Running locally

```bash
# Quantitative pipeline
python3 "scripts/quant/Quant 1 fetch prices.py"
python3 "scripts/quant/Quant 2 fetch eia.py"
python3 "scripts/quant/Quant 5 assemble master.py"
python3 "scripts/quant/Quant 11 build full index.py"

# Streamlit frontend
streamlit run streamlit_app.py
```

---

## Data sources

| Source | Frequency | Coverage |
|---|---|---|
| EIA Weekly Petroleum Status Report | Weekly | 2007 to present |
| CFTC Commitments of Traders | Weekly | 2007 to present |
| FRED (Fed Funds, USD Broad) | Daily | 2007 to present |
| OPEC Monthly Oil Market Report | Monthly | 2007 to present |
| EIA Short-Term Energy Outlook | Monthly | 2007 to present |
| Saudi Aramco news (Google RSS) | Daily | 2020 to present |

---

## Limitations

- Single OOS era. Results tested on 2020 to 2026 only, performance in other regimes is unknown.
- Regime sensitivity. Accuracy was 98.7% in volatile periods (2020 to 2022) and 64.1% in stable periods (2023 to 2026).
- NLP freshness. Only around 5.5% of daily values are genuine publication-day data, the rest are forward-filled.
- Macro orthogonalisation. OLS p-values are inflated by overlapping returns, HAC-corrected p=0.2313 (not formally significant).
- Direction only. The index predicts direction, not magnitude or timing within the horizon.

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

MIT, see [LICENSE](LICENSE)
