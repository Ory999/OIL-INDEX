"""
Script 07 — Build Partial Quantitative PRCSI Index
Produces a fundamentals-only index before NLP scores arrive.
Runs before the qualitative pipeline; script 12 builds the full index.

FIX APPLIED (#5):
  MinMaxScaler was fitted on the full history, meaning the score for any
  given day depended on data not yet observed at that point (lookahead
  bias). The index values would also shift on every re-run as new data
  extended the scale. Replaced with a 252-day rolling percentile rank
  (same method as the validated notebook) so each day's score reflects
  only the preceding year's distribution.
"""
import os, logging, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NORM_WINDOW  = 252   # rolling percentile window — matches notebook
EMA_SMOOTH   = 63    # EMA smoothing span — matches notebook


def classify_regime(score: float) -> str:
    if score <= 25:   return "EXTREME_FEAR"
    elif score <= 45: return "FEAR"
    elif score <= 55: return "NEUTRAL"
    elif score <= 75: return "GREED"
    else:             return "EXTREME_GREED"


def rolling_percentile(series: pd.Series, window: int = NORM_WINDOW) -> pd.Series:
    """
    FIX #5: Rolling percentile rank — replaces MinMaxScaler(full history).
    Each day's score = fraction of preceding {window} days that were lower.
    Result is naturally bounded [0, 1] with no lookahead.
    """
    return series.rolling(
        window, min_periods=int(window * 0.5)
    ).apply(
        lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True
    )


def build_quant_index():
    master_path = FEATURES_DIR / "master_quant.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} not found — run 06 assemble master.py first")

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)

    components = {}

    # ── Fundamentals component ────────────────────────────────────────────
    # Direction corrections match notebook FEATURE_DIRECTION dict:
    #   eia_surprise_norm: -1 (surprise inventory build = bearish)
    #   crude_stocks_change: -1 (stock build = bearish)
    #   cot_net_long: +1 (net long = bullish)
    fund_parts = []
    if "eia_surprise_norm" in master.columns:
        # Invert: high surprise (build) → low percentile → FEAR
        fund_parts.append(rolling_percentile(-master["eia_surprise_norm"].fillna(0)))
    if "crude_stocks_change" in master.columns:
        fund_parts.append(rolling_percentile(-master["crude_stocks_change"].fillna(0)))
    if "cot_net_long" in master.columns:
        fund_parts.append(rolling_percentile(master["cot_net_long"].fillna(0)))
    if "cot_change_1w" in master.columns:
        fund_parts.append(rolling_percentile(master["cot_change_1w"].fillna(0)))

    if fund_parts:
        components["fundamentals"] = pd.concat(fund_parts, axis=1).mean(axis=1)
        log.info(f"  ✓ Fundamentals component built ({len(fund_parts)} sub-signals)")
    else:
        components["fundamentals"] = pd.Series(0.5, index=master.index)
        log.warning("  No fundamental columns found — using neutral 0.5")

    # ── Composite index — fundamentals only at this stage ─────────────────
    comp_df   = pd.DataFrame(components).fillna(0.5)
    raw_score = comp_df["fundamentals"]

    # FIX #5: EMA smooth (span=63, matches notebook), scale to 0-100 via rolling pct
    raw_smooth    = raw_score.ewm(span=EMA_SMOOTH, min_periods=10).mean()
    quant_prcsi   = raw_smooth * 100   # already 0-1 from rolling percentile → 0-100

    if len(quant_prcsi.dropna()) == 0:
        log.warning("  PRCSI score empty — insufficient data")
        return pd.DataFrame()

    result = pd.DataFrame({
        "prcsi_quant_raw":   quant_prcsi,
        "prcsi_quant":       quant_prcsi,   # no additional smoothing needed
        "regime_quant":      quant_prcsi.apply(
                                 lambda x: classify_regime(x) if not np.isnan(x) else "NEUTRAL"),
        "comp_fundamentals": comp_df["fundamentals"],
        "oil_price":         master["oil"]        if "oil"        in master.columns else np.nan,
        "oil_logret":        master["oil_logret"] if "oil_logret" in master.columns else np.nan,
    })

    result.to_parquet(RESULTS_DIR / "prcsi_quant_partial.parquet")
    result.to_csv(RESULTS_DIR    / "prcsi_quant_partial.csv")

    latest_score  = round(float(quant_prcsi.dropna().iloc[-1]), 2)
    latest_regime = classify_regime(latest_score)

    log.info(f"\n✓ Partial quantitative PRCSI built (rolling percentile method)")
    log.info(f"  Latest score: {latest_score:.1f} / 100")
    log.info(f"  Regime:       {latest_regime}")

    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        "prcsi_quant_latest":      latest_score,
        "prcsi_quant_regime":      latest_regime,
        "quant_pipeline_complete": True,
        "quant_run_timestamp":     datetime.now().isoformat(),
        "nlp_scores_merged":       False,
    })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return result


if __name__ == "__main__":
    build_quant_index()
