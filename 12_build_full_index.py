"""
Script 12 — Build Full PRCSI Index (0–100)
Combines ALL five components: quantitative (25%) + NLP sentiment (75%).
Runs only after qualitative pipeline has produced NLP scores.

Component weights (from project scope v4.0):
  1. Supply Sentiment Momentum    30%  — OPEC/IEA EMA crossover
  2. Institutional Rhetoric       25%  — BERTopic per-topic signals
  3. Surface vs. Implied Divergence 20% — LLM divergence score
  4. Fundamentals Signal          15%  — EIA + COT composite
  5. Congressional Trade Signal   10%  — STOCK Act net buy/sell

Output: data/results/prcsi_final.parquet
        data/results/prcsi_final.csv
        data/results/prcsi_dashboard.png
"""
import os, json, logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def classify_regime(score: float) -> str:
    if score <= 25:   return "EXTREME_FEAR"
    elif score <= 45: return "FEAR"
    elif score <= 55: return "NEUTRAL"
    elif score <= 75: return "GREED"
    else:             return "EXTREME_GREED"


def norm(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalise to 0-1 using rolling percentile rank."""
    ranked = series.rank(pct=True).clip(0, 1)
    return (1 - ranked) if invert else ranked


def build_full_index():
    master_path = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        log.warning("master_with_nlp.parquet not found — falling back to quantitative only")
        master_path = FEATURES_DIR / "master_quant.parquet"

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)
    scaler = MinMaxScaler(feature_range=(0, 1))
    has_nlp = any("sent_" in c or "oil_impact" in c for c in master.columns)

    components = {}

    # ── Component 1: Supply Sentiment Momentum (30%) ──────────────────────
    c1_candidates = ["sent_ema_cross", "supply_disruption_ema",
                     "sent_roc_3d", "oil_impact_score"]
    c1_avail = [c for c in c1_candidates if c in master.columns]
    if c1_avail:
        c1_raw = master[c1_avail].fillna(0).mean(axis=1)
        components["supply_momentum"] = norm(c1_raw, invert=True)  # invert: negative = fear
        log.info(f"  C1 Supply Momentum: {len(c1_avail)} sub-signals")
    else:
        components["supply_momentum"] = pd.Series(0.5, index=master.index)

    # ── Component 2: Institutional Rhetoric Signal (25%) ──────────────────
    topic_cols = [c for c in master.columns
                  if c.startswith("topic_") and "ema_cross" not in c]
    if topic_cols:
        c2_raw = master[topic_cols].fillna(0).mean(axis=1)
        components["rhetoric"] = norm(c2_raw, invert=True)
        log.info(f"  C2 Rhetoric: {len(topic_cols)} topic streams")
    elif "finbert_score" in master.columns:
        components["rhetoric"] = norm(master["finbert_score"].fillna(0), invert=True)
        log.info("  C2 Rhetoric: using FinBERT fallback")
    else:
        components["rhetoric"] = pd.Series(0.5, index=master.index)

    # ── Component 3: Surface vs. Implied Divergence (20%) ─────────────────
    if "surface_vs_implied_divergence" in master.columns:
        components["divergence"] = norm(
            master["surface_vs_implied_divergence"].fillna(0)
        )
        log.info("  C3 Divergence: LLM divergence score")
    elif "sent_divergence" in master.columns:
        components["divergence"] = norm(master["sent_divergence"].fillna(0))
    else:
        components["divergence"] = pd.Series(0.5, index=master.index)
        log.warning("  C3 Divergence: no score available — neutral 0.5")

    # ── Component 4: Fundamentals Signal (15%) ────────────────────────────
    fund_parts = []
    if "eia_surprise_norm" in master.columns:
        fund_parts.append(norm(master["eia_surprise_norm"].fillna(0), invert=True))
    if "cot_net_long" in master.columns:
        fund_parts.append(norm(master["cot_net_long"].fillna(0)))
    if "cot_change_1w" in master.columns:
        fund_parts.append(norm(master["cot_change_1w"].fillna(0)))

    if fund_parts:
        components["fundamentals"] = pd.concat(fund_parts, axis=1).mean(axis=1)
        log.info(f"  C4 Fundamentals: {len(fund_parts)} sub-signals")
    else:
        components["fundamentals"] = pd.Series(0.5, index=master.index)

    # ── Component 5: Congressional Trade Signal (10%) ─────────────────────
    if "congress_net_signal" in master.columns:
        components["congress"] = norm(
            master["congress_net_signal"].fillna(0), invert=True
        )
        log.info("  C5 Congress: STOCK Act net signal")
    else:
        components["congress"] = pd.Series(0.5, index=master.index)

    # ── Weighted composite ────────────────────────────────────────────────
    # If NLP scores missing, redistribute NLP weights to fundamentals
    if has_nlp:
        weights = {
            "supply_momentum": 0.30,
            "rhetoric":        0.25,
            "divergence":      0.20,
            "fundamentals":    0.15,
            "congress":        0.10,
        }
    else:
        log.warning("  NLP scores absent — redistributing weights to fundamentals")
        weights = {
            "supply_momentum": 0.00,
            "rhetoric":        0.00,
            "divergence":      0.00,
            "fundamentals":    0.70,
            "congress":        0.30,
        }

    comp_df   = pd.DataFrame(components).fillna(0.5)
    raw_score = sum(comp_df[k] * w for k, w in weights.items())

    prcsi = pd.Series(
        scaler.fit_transform(raw_score.values.reshape(-1, 1)).flatten() * 100,
        index=raw_score.index,
        name="prcsi",
    )
    prcsi_smooth = prcsi.rolling(5, min_periods=1).mean()

    # ── Result DataFrame ──────────────────────────────────────────────────
    result = pd.DataFrame({
        "prcsi_raw":             prcsi,
        "prcsi":                 prcsi_smooth,
        "regime":                prcsi_smooth.apply(classify_regime),
        "comp_supply_momentum":  comp_df["supply_momentum"],
        "comp_rhetoric":         comp_df["rhetoric"],
        "comp_divergence":       comp_df["divergence"],
        "comp_fundamentals":     comp_df["fundamentals"],
        "comp_congress":         comp_df["congress"],
        "oil_price":             master["oil"]        if "oil"        in master.columns else np.nan,
        "oil_logret":            master["oil_logret"] if "oil_logret" in master.columns else np.nan,
    })

    result.to_parquet(RESULTS_DIR / "prcsi_final.parquet")
    result.to_csv(RESULTS_DIR    / "prcsi_final.csv")

    # ── Dashboard chart ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.fill_between(result.index, 0, result["prcsi"], alpha=0.12, color="#2563EB")
    ax1.plot(result.index, result["prcsi"], color="#1B2A4A", linewidth=1.2)
    for y, label, col in [
        (12, "Extreme Fear", "#7F1D1D"), (35, "Fear", "#DC2626"),
        (50, "Neutral", "#9CA3AF"),      (65, "Greed", "#15803D"),
        (87, "Extreme Greed", "#14532D"),
    ]:
        ax1.axhline(y={"Fear": 25, "Neutral": 45, "Greed": 55,
                       "Extreme Greed": 75}.get(label, y),
                    color=col, linestyle="--", alpha=0.35, linewidth=0.8)
        ax1.text(result.index[-1], y, f" {label}", va="center",
                 fontsize=8, color=col)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("PRCSI Score")
    ax1.set_title(
        f"Oil Fear & Greed Index (PRCSI) — "
        f"{'Full NLP+Quant' if has_nlp else 'Quantitative Only'}\n"
        f"Latest: {result['prcsi'].iloc[-1]:.1f} ({result['regime'].iloc[-1]})",
        fontsize=13, fontweight="bold"
    )
    ax1.grid(alpha=0.2)

    if "oil_price" in result.columns:
        ax2.plot(result.index, result["oil_price"],
                 color="#E74C3C", linewidth=0.9)
        ax2.set_ylabel("WTI Crude Oil (USD)")
        ax2.set_title("WTI Crude Oil Price")
        ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prcsi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Update metadata ───────────────────────────────────────────────────
    latest = float(result["prcsi"].iloc[-1])
    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        "prcsi_latest":            round(latest, 2),
        "prcsi_regime":            result["regime"].iloc[-1],
        "full_index_complete":     True,
        "nlp_scores_used":         has_nlp,
        "full_run_timestamp":      datetime.now().isoformat(),
    })
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"\n✓ Full PRCSI index built")
    log.info(f"  Latest score: {latest:.1f} ({result['regime'].iloc[-1]})")
    log.info(f"  NLP scores:   {'✅ included' if has_nlp else '⚠️  not included'}")
    return result


if __name__ == "__main__":
    build_full_index()
