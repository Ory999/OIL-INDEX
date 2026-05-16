"""
Script 14 — Build Price Sentiment Index (PSI)
Pure price-action fear/greed index for WTI crude oil.
Companion to PRCSI for divergence analysis (Grossman-Stiglitz 1980).

METHODOLOGY — MOMENTUM EXTREMITY APPROACH
==========================================
Rather than comparing price LEVELS to historical levels, the PSI measures
how extreme the current price MOVEMENT is relative to the most extreme
movements ever recorded since 2007.

Core insight: Fear and greed are momentum phenomena, not level phenomena.
Greed = rushing to buy (fast upward moves). Fear = rushing to sell (fast falls).
The absolute price level is secondary to the speed and magnitude of change.

Three windows capture different market participant timescales:
  3-month (63d) — Cyclical funds, refiners, commodity managers
  1-week   (5d) — Momentum traders, CTAs, macro funds
  1-day    (1d) — News-reactive traders, retail, short-term speculators

Reference: EXPANDING MAXIMUM rise/fall from 2007 → self-calibrating.
If a new record spike or crash occurs, it immediately becomes the benchmark.

FORMULA (per window n):
  ret_n      = price % change over n trading days
  max_rise_n = expanding max(ret_n) since 2007  [most greedy move ever]
  max_fall_n = expanding min(ret_n) since 2007  [most fearful move ever]

  score = 0.5 + 0.5 × (ret_n / reference)
  where reference = max_rise_n   if ret_n >= 0  (greed direction)
                    |max_fall_n| if ret_n <  0  (fear direction)

  → score 1.0: matches all-time record rise  (extreme greed)
  → score 0.5: flat / no change              (neutral)
  → score 0.0: matches all-time record fall  (extreme fear)

Naturally bounded [0, 1]. No additional normalisation needed.

WEIGHTS:
  3-month: 2.0  (structural momentum — highest weight)
  1-week:  1.5  (medium-term)
  1-day:   1.0  (tactical)

Final: weighted average → EMA(span=63) to match PRCSI smoothing.

ACADEMIC ALIGNMENT:
  De Bondt & Thaler (1985): Markets overreact to extreme price movements.
  Scoring current movement as a fraction of the historical extreme directly
  quantifies the degree of potential overreaction.

  Grossman-Stiglitz (1980): PRCSI-PSI divergence detects information
  asymmetry — institutions with multi-decade context perceive extreme
  momentum moves differently from price-reactive participants.

Outputs:
  data/results/psi_final.parquet
  data/results/psi_final.csv
  data/results/psi_dashboard.png
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

EMA_SMOOTH  = 63   # matches PRCSI exactly
MIN_HISTORY = 252  # 1 year before computing expanding max/min

WINDOWS = {"3m": 63, "1w": 5, "1d": 1}

COMPONENT_WEIGHTS = {
    "fg_3m": 2.0,
    "fg_1w": 1.5,
    "fg_1d": 1.0,
}


def classify_regime(score: float) -> str:
    if np.isnan(score): return "NEUTRAL"
    if score <= 25:     return "EXTREME_FEAR"
    elif score <= 45:   return "FEAR"
    elif score <= 55:   return "NEUTRAL"
    elif score <= 75:   return "GREED"
    else:               return "EXTREME_GREED"


def momentum_fear_greed(price: pd.Series, n_days: int, label: str) -> pd.Series:
    """
    Score fear/greed for a given return window relative to
    the maximum historical move in the same direction.
    Returns Series bounded [0, 1].
    """
    ret = price.pct_change(n_days)

    max_rise = ret.expanding(min_periods=MIN_HISTORY).max()
    max_fall = ret.expanding(min_periods=MIN_HISTORY).min()

    score = pd.Series(np.nan, index=price.index)

    pos = ret >= 0
    neg = ret < 0

    denom_pos = max_rise.where(max_rise > 1e-6, np.nan)
    denom_neg = max_fall.abs().where(max_fall.abs() > 1e-6, np.nan)

    score[pos] = (0.5 + 0.5 * (ret[pos] / denom_pos[pos])).clip(0, 1)
    score[neg] = (0.5 + 0.5 * (ret[neg] / denom_neg[neg])).clip(0, 1)

    # Diagnostics
    if ret.notna().any():
        lr = ret.dropna().iloc[-1]
        mr = max_rise.dropna().iloc[-1] if max_rise.notna().any() else np.nan
        mf = max_fall.dropna().iloc[-1] if max_fall.notna().any() else np.nan
        sc = score.dropna().iloc[-1] if score.notna().any() else np.nan
        log.info(f"  {label} ({n_days}d): ret={lr:+.1%}  "
                 f"max_rise={mr:+.1%}  max_fall={mf:+.1%}  → score={sc:.3f}")

    return score


def build_psi():
    prices_path = RAW_DIR / "prices.parquet"
    if not prices_path.exists():
        prices_path = FEATURES_DIR / "master_quant.parquet"
    if not prices_path.exists():
        log.warning("No price data — skipping PSI build")
        return None

    raw = pd.read_parquet(prices_path)
    raw.index = pd.to_datetime(raw.index)
    if hasattr(raw.index, "tz") and raw.index.tz:
        raw.index = raw.index.tz_localize(None)

    if "oil" not in raw.columns:
        log.warning("'oil' column missing — skipping PSI build")
        return None

    price = raw["oil"].ffill()

    log.info(f"PSI: {len(price)} trading days  "
             f"({price.index.min().date()} → {price.index.max().date()})")
    log.info(f"  Dataset price range: ${price.min():.2f} – ${price.max():.2f}")
    log.info(f"  Current price: ${price.dropna().iloc[-1]:.2f}")
    log.info(f"  Method: momentum extremity (score vs max historical move)")

    # ── Three momentum components ─────────────────────────────────────────
    components = {
        "fg_3m": momentum_fear_greed(price, WINDOWS["3m"], "3-month"),
        "fg_1w": momentum_fear_greed(price, WINDOWS["1w"], "1-week"),
        "fg_1d": momentum_fear_greed(price, WINDOWS["1d"], "1-day"),
    }

    # ── Weighted composite + EMA smooth ──────────────────────────────────
    total_w   = sum(COMPONENT_WEIGHTS.values())
    raw_score = sum(
        components[k].fillna(0.5) * w
        for k, w in COMPONENT_WEIGHTS.items()
    ) / total_w

    psi_01  = raw_score.ewm(span=EMA_SMOOTH, min_periods=10).mean()
    psi_100 = psi_01 * 100

    log.info(f"  PSI composite: {psi_01.min():.3f} → {psi_01.max():.3f}")

    # ── RSI(7) for display ────────────────────────────────────────────────
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(7, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(7, min_periods=1).mean()
    rsi_7 = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    # ── Return and reference series ───────────────────────────────────────
    ret_3m = price.pct_change(WINDOWS["3m"])
    ret_1w = price.pct_change(WINDOWS["1w"])
    ret_1d = price.pct_change(WINDOWS["1d"])

    result = pd.DataFrame({
        "psi_01":        psi_01,
        "psi":           psi_100,
        "regime":        psi_100.apply(classify_regime),
        "comp_fg_3m":    components["fg_3m"],
        "comp_fg_1w":    components["fg_1w"],
        "comp_fg_1d":    components["fg_1d"],
        "ret_3m":        ret_3m,
        "ret_1w":        ret_1w,
        "ret_1d":        ret_1d,
        "max_rise_3m":   ret_3m.expanding(min_periods=MIN_HISTORY).max(),
        "max_fall_3m":   ret_3m.expanding(min_periods=MIN_HISTORY).min(),
        "oil_price":     price,
        "rsi_raw":       rsi_7,
    })

    result.to_parquet(RESULTS_DIR / "psi_final.parquet")
    result.to_csv(RESULTS_DIR    / "psi_final.csv")

    _build_psi_chart(result)

    # ── Metadata ──────────────────────────────────────────────────────────
    latest       = result.dropna(subset=["psi"]).iloc[-1]
    latest_score = round(float(latest["psi"]), 2)
    latest_rsi   = round(float(latest["rsi_raw"]), 1) if not np.isnan(latest["rsi_raw"]) else None
    r3m = round(float(latest["ret_3m"]) * 100, 1) if not np.isnan(latest["ret_3m"]) else None
    r1w = round(float(latest["ret_1w"]) * 100, 1) if not np.isnan(latest["ret_1w"]) else None
    r1d = round(float(latest["ret_1d"]) * 100, 1) if not np.isnan(latest["ret_1d"]) else None
    mr  = round(float(latest["max_rise_3m"]) * 100, 1) if not np.isnan(latest["max_rise_3m"]) else None
    mf  = round(float(latest["max_fall_3m"]) * 100, 1) if not np.isnan(latest["max_fall_3m"]) else None

    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        "psi_latest":          latest_score,
        "psi_regime":          str(latest["regime"]),
        "psi_date":            str(latest.name.date()),
        "psi_rsi_7":           latest_rsi,
        "psi_ret_3m_pct":      r3m,
        "psi_ret_1w_pct":      r1w,
        "psi_ret_1d_pct":      r1d,
        "psi_max_rise_3m_pct": mr,
        "psi_max_fall_3m_pct": mf,
        "psi_method":          "momentum_extremity_3windows",
        "psi_complete":        True,
        "psi_run_timestamp":   datetime.now().isoformat(),
    })

    # ── Divergence from PRCSI ─────────────────────────────────────────────
    prcsi_path = RESULTS_DIR / "prcsi_final.parquet"
    if prcsi_path.exists():
        try:
            prcsi = pd.read_parquet(prcsi_path)
            prcsi.index = pd.to_datetime(prcsi.index)
            common = psi_01.index.intersection(prcsi.index)
            if len(common):
                div = prcsi.loc[common, "prcsi_01"] - psi_01[common]
                ld  = float(div.iloc[-1])
                dd  = ("PSI_LEADS"   if ld < -0.10 else
                       "PRCSI_LEADS" if ld >  0.10 else "ALIGNED")
                metadata.update({
                    "divergence":           round(ld, 4),
                    "divergence_abs":       round(abs(ld), 4),
                    "divergence_direction": dd,
                    "divergence_pct_pts":   round(ld * 100, 1),
                })
                log.info(f"  Divergence PRCSI−PSI: {ld:+.4f} ({dd})")
        except Exception as e:
            log.warning(f"  Could not compute divergence: {e}")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"\n✓ PSI: {latest_score:.1f} / 100  ({latest['regime']})")
    log.info(f"  3M: {r3m:+.1f}% vs max_rise {mr:+.1f}% / max_fall {mf:+.1f}%")
    log.info(f"  1W: {r1w:+.1f}%   1D: {r1d:+.1f}%   RSI(7): {latest_rsi}")
    return result


def _build_psi_chart(result: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.8, 1.0]})
    fig.suptitle(
        "Price Sentiment Index (PSI) — WTI Crude Oil\n"
        "Score = current move / max historical move  "
        "(1.0 = matches all-time record rise, 0.0 = matches all-time record fall)",
        fontsize=11, fontweight="bold"
    )

    ax1 = axes[0]
    ax1.plot(result.index, result["oil_price"], color="#f97316", linewidth=1.0)
    for days, col, lbl in [(14,"#fbbf24","MA14"),(30,"#fb923c","MA30"),(60,"#e11d48","MA60")]:
        ma = result["oil_price"].rolling(days, min_periods=int(days*0.5)).mean()
        ax1.plot(result.index, ma, linewidth=0.7, alpha=0.7, label=lbl)
    ax1.set_ylabel("WTI (USD)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.2)

    ax2 = axes[1]
    for y0, y1, clr in [(0,25,"#0d1f5c"),(25,45,"#1e3f8a"),(45,55,"#1f2937"),
                         (55,75,"#7c2d12"),(75,100,"#450a0a")]:
        ax2.axhspan(y0, y1, alpha=0.25, color=clr, linewidth=0)
    ax2.plot(result.index, result["psi"], color="#f9fafb", linewidth=1.4, label="PSI")
    ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("PSI", fontsize=9)
    ax2.grid(alpha=0.2)

    ax3 = axes[2]
    ax3.plot(result.index, result["comp_fg_3m"] * 100,
             color="#60a5fa", linewidth=1.0, label="3M (2×)")
    ax3.plot(result.index, result["comp_fg_1w"] * 100,
             color="#a78bfa", linewidth=0.8, alpha=0.8, label="1W (1.5×)")
    ax3.plot(result.index, result["comp_fg_1d"] * 100,
             color="#f9a8d4", linewidth=0.6, alpha=0.6, label="1D (1×)")
    ax3.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Components", fontsize=9)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Chart → {RESULTS_DIR / 'psi_dashboard.png'}")


if __name__ == "__main__":
    build_psi()
