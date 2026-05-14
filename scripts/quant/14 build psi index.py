"""
Script 14 — Build Price Sentiment Index (PSI)
Pure price-action fear/greed index for WTI crude oil.
Companion to PRCSI for divergence analysis.

Theoretical basis:
  Grossman-Stiglitz (1980): If institutional insiders know something markets
  don't, the gap between what they communicate (PRCSI) and what price action
  implies (PSI) is the information asymmetry signal.
  De Long et al. (1990): Noise traders push price beyond fundamentals in
  high-volatility regimes. Signed volatility ratio detects this.

Components (5, equal-weighted):
  1. RSI(7)              — short-term overbought/oversold
  2. Price vs MA14       — 2-week deviation (PRCSI's primary horizon)
  3. Price vs MA30       — monthly deviation
  4. Price vs MA60       — quarterly regime
  5. Signed vol ratio    — vol_5d / vol_30d × sign(return_5d)
                           positive = volatile upward spike (greed)
                           negative = volatile crash (fear)

All components: direction-corrected, 252-day rolling percentile normalised.
Smoothed with EMA(span=63) to match PRCSI construction.
Output scale: 0–100 (50 = neutral).

Outputs:
  data/results/psi_final.parquet
  data/results/psi_final.csv
"""
import os, json, logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Match PRCSI construction parameters exactly
NORM_WINDOW = 252
EMA_SMOOTH  = 63

COMPONENT_WEIGHTS = {
    "rsi_7":           1.0,   # short-term overbought/oversold
    "dev_ma14":        1.0,   # 2-week deviation
    "dev_ma30":        1.0,   # monthly deviation
    "dev_ma60":        1.0,   # quarterly deviation
    "signed_vol":      1.0,   # spike regime direction
    "price_long_rank": 1.5,   # long-term price level in historical context
                               # higher weight: most structurally important component
                               # At $111 (~2x the 2007-2026 average of ~$50), this
                               # should push PSI significantly higher than MA-only approaches
}


def classify_regime(score: float) -> str:
    if np.isnan(score): return "NEUTRAL"
    if score <= 25:     return "EXTREME_FEAR"
    elif score <= 45:   return "FEAR"
    elif score <= 55:   return "NEUTRAL"
    elif score <= 75:   return "GREED"
    else:               return "EXTREME_GREED"


def rolling_percentile(series: pd.Series, window: int = NORM_WINDOW) -> pd.Series:
    """252-day rolling percentile rank — matches PRCSI normalisation."""
    return series.rolling(
        window, min_periods=int(window * 0.5)
    ).apply(
        lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True
    )


def compute_rsi(price: pd.Series, window: int = 7) -> pd.Series:
    """Standard RSI on price series. Returns 0–100."""
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def build_psi():
    # ── Load price data ───────────────────────────────────────────────────
    prices_path = RAW_DIR / "prices.parquet"
    if not prices_path.exists():
        # Fallback: try master_quant
        prices_path = FEATURES_DIR / "master_quant.parquet"
    if not prices_path.exists():
        log.warning("No price data found — skipping PSI build")
        return None

    raw = pd.read_parquet(prices_path)
    raw.index = pd.to_datetime(raw.index)
    if hasattr(raw.index, "tz") and raw.index.tz:
        raw.index = raw.index.tz_localize(None)

    if "oil" not in raw.columns:
        log.warning("'oil' column not found — skipping PSI build")
        return None

    price  = raw["oil"].ffill()
    logret = raw["oil_logret"].ffill() if "oil_logret" in raw.columns \
             else np.log(price / price.shift(1))

    log.info(f"PSI building on {len(price)} trading days "
             f"({price.index.min().date()} → {price.index.max().date()})")

    components = pd.DataFrame(index=price.index)

    # ── Component 1: RSI(7) ───────────────────────────────────────────────
    # High RSI (overbought) = greed → direction +1
    rsi_raw = compute_rsi(price, window=7)
    components["rsi_7"] = rolling_percentile(rsi_raw)

    # ── Components 2-4: Price deviation from MAs ──────────────────────────
    # Positive deviation = price above MA = bullish = greed → direction +1
    for days, label in [(14, "dev_ma14"), (30, "dev_ma30"), (60, "dev_ma60")]:
        ma  = price.rolling(days, min_periods=int(days * 0.5)).mean()
        dev = (price - ma) / ma.replace(0, np.nan)
        components[label] = rolling_percentile(dev)

    # ── Component 5: Signed volatility ratio ─────────────────────────────
    # vol_5d / vol_30d * sign(5d return)
    # High positive = volatile price spike upward = extreme greed
    # High negative = volatile crash = extreme fear
    vol_5d  = logret.rolling(5,  min_periods=3).std() * np.sqrt(252)
    vol_30d = logret.rolling(30, min_periods=15).std() * np.sqrt(252)
    ret_5d  = price.pct_change(5)

    vol_ratio   = vol_5d / vol_30d.replace(0, np.nan).fillna(1.0)
    signed_vol  = vol_ratio * np.sign(ret_5d)
    components["signed_vol"] = rolling_percentile(signed_vol)

    # ── Component 6: Long-term price level rank ───────────────────────────
    # Rolling percentile over 2500 trading days (~10 years).
    # Answers: "Is current price historically expensive?"
    # At $111 (near 2008 highs, ~2x long-run average): rank ≈ 0.85-0.90
    # At $20 (COVID 2020): rank ≈ 0.02
    # Justification: Grossman-Stiglitz (1980) — if price deviates far from
    # long-run fundamental value, information asymmetry with institutions
    # who have multi-decade price context is most likely.
    LONG_WINDOW = min(2500, int(len(price) * 0.8))  # adaptive: up to 10 years
    components["price_long_rank"] = rolling_percentile(price, window=LONG_WINDOW)
    log.info(f"  Long-term price rank window: {LONG_WINDOW} days "
             f"({LONG_WINDOW/252:.1f} years)")

    # ── Weighted composite ────────────────────────────────────────────────
    raw_score = pd.Series(0.0, index=price.index)
    total_w   = 0.0
    for comp, w in COMPONENT_WEIGHTS.items():
        if comp in components.columns:
            raw_score += components[comp].fillna(0.5) * w
            total_w   += w

    if total_w > 0:
        raw_score /= total_w

    # FIX #5: EMA smooth matching PRCSI (span=63, no lookahead)
    psi_01  = raw_score.ewm(span=EMA_SMOOTH, min_periods=10).mean()
    psi_100 = psi_01 * 100

    log.info(f"  PSI range: {psi_01.min():.3f} → {psi_01.max():.3f}")

    # ── Build result DataFrame ────────────────────────────────────────────
    result = pd.DataFrame({
        "psi_01":       psi_01,
        "psi":          psi_100,
        "regime":       psi_100.apply(classify_regime),
        "comp_rsi_7":   components.get("rsi_7",    pd.Series(np.nan, index=price.index)),
        "comp_dev_ma14":components.get("dev_ma14", pd.Series(np.nan, index=price.index)),
        "comp_dev_ma30":components.get("dev_ma30", pd.Series(np.nan, index=price.index)),
        "comp_dev_ma60":components.get("dev_ma60", pd.Series(np.nan, index=price.index)),
        "comp_signed_vol":    components.get("signed_vol",       pd.Series(np.nan, index=price.index)),
        "comp_price_long":    components.get("price_long_rank", pd.Series(np.nan, index=price.index)),
        "oil_price":    price,
        "rsi_raw":      rsi_raw,
        "vol_ratio":    vol_ratio,
    })

    result.to_parquet(RESULTS_DIR / "psi_final.parquet")
    result.to_csv(RESULTS_DIR    / "psi_final.csv")

    # ── Dashboard chart ───────────────────────────────────────────────────
    _build_psi_chart(result)

    # ── Update metadata ───────────────────────────────────────────────────
    latest = result.dropna(subset=["psi"]).iloc[-1]
    latest_score  = round(float(latest["psi"]), 2)
    latest_regime = str(latest["regime"])
    latest_rsi    = round(float(latest["rsi_raw"]), 1) if not np.isnan(latest["rsi_raw"]) else None
    latest_vol    = round(float(latest["vol_ratio"]), 3) if not np.isnan(latest["vol_ratio"]) else None

    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        "psi_latest":         latest_score,
        "psi_regime":         latest_regime,
        "psi_date":           str(latest.name.date()),
        "psi_rsi_7":          latest_rsi,
        "psi_vol_ratio":      latest_vol,
        "psi_complete":       True,
        "psi_run_timestamp":  datetime.now().isoformat(),
    })

    # ── Compute PRCSI-PSI divergence if PRCSI already built ──────────────
    prcsi_path = RESULTS_DIR / "prcsi_final.parquet"
    if prcsi_path.exists():
        try:
            prcsi = pd.read_parquet(prcsi_path)
            prcsi.index = pd.to_datetime(prcsi.index)
            common = psi_01.index.intersection(prcsi.index)
            if len(common) > 0:
                div = prcsi.loc[common, "prcsi_01"] - psi_01[common]
                latest_div = float(div.iloc[-1]) if not div.empty else 0.0
                div_direction = (
                    "PSI_LEADS"   if latest_div < -0.10 else
                    "PRCSI_LEADS" if latest_div > 0.10 else
                    "ALIGNED"
                )
                metadata.update({
                    "divergence":           round(latest_div, 4),
                    "divergence_abs":       round(abs(latest_div), 4),
                    "divergence_direction": div_direction,
                    "divergence_pct_pts":   round(latest_div * 100, 1),
                })
                log.info(f"  Divergence PRCSI−PSI: {latest_div:+.4f} ({div_direction})")
        except Exception as e:
            log.warning(f"  Could not compute divergence: {e}")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"\n✓ PSI built: {latest_score:.1f} / 100  ({latest_regime})")
    log.info(f"  RSI(7): {latest_rsi}  |  Vol ratio: {latest_vol}")
    return result


def _build_psi_chart(result: pd.DataFrame):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10),
                                          sharex=True,
                                          gridspec_kw={"height_ratios": [1.5, 2, 1]})
    fig.suptitle("Price Sentiment Index (PSI) — WTI Crude Oil",
                 fontsize=13, fontweight="bold")

    # Panel 1: WTI price
    ax1.plot(result.index, result["oil_price"], color="#f97316", linewidth=1.0)
    for days, col in [(14,"#fbbf24"),(30,"#fb923c"),(60,"#e11d48")]:
        ma = result["oil_price"].rolling(days, min_periods=int(days*0.5)).mean()
        ax1.plot(result.index, ma, linewidth=0.8, alpha=0.7, label=f"MA{days}")
    ax1.set_ylabel("WTI Price (USD)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.2)

    # Panel 2: PSI
    ax2.fill_between(result.index, 50, result["psi"],
                     where=result["psi"] >= 50, alpha=0.15, color="#dc2626")
    ax2.fill_between(result.index, result["psi"], 50,
                     where=result["psi"] < 50, alpha=0.15, color="#2563EB")
    ax2.plot(result.index, result["psi"], color="#f9fafb", linewidth=1.2, label="PSI")
    ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("PSI (0=fear, 100=greed)", fontsize=9)
    ax2.grid(alpha=0.2)

    # Panel 3: RSI
    ax3.plot(result.index, result["rsi_raw"], color="#60a5fa", linewidth=0.9)
    ax3.axhline(70, color="#dc2626", linewidth=0.8, linestyle="--", alpha=0.6,
                label="Overbought (70)")
    ax3.axhline(30, color="#3b82f6", linewidth=0.8, linestyle="--", alpha=0.6,
                label="Oversold (30)")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI(7)", fontsize=9)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  PSI chart saved → {RESULTS_DIR / 'psi_dashboard.png'}")


if __name__ == "__main__":
    build_psi()
