"""
Script 12 — Build Full PRCSI Index (0–100)
Combines quantitative + NLP sentiment components using the methodology
validated in the research notebook (PRCSI_Analysis_End_to_End).

FIXES APPLIED:
  FIX #3: Complete methodology rewrite — live index now matches the
          validated notebook construction (86.8% OOS accuracy at top 10%).
  FIX #4: Correct FEATURE_DIRECTION applied before normalisation.
          norm(..., invert=True) on bullish features was flipping greed/fear.
  FIX #5: MinMaxScaler on full history replaced with 252-day rolling
          percentile rank — eliminates lookahead bias in display layer.
          EMA smoothing changed from rolling mean(5) to EMA(span=63)
          to match the validated notebook construction.
  FIX #6: Features restricted to STABLE_FEATURES whitelist (9 features
          significant in ≥20% of rolling Granger windows). Episodic and
          rare features are excluded.
  FIX #7: Prediction layer added. Severity = |index - 0.5| × 2. Train-
          frozen threshold (0.2637) gates the contrarian binary signal.
          Signal direction: index > 0.5 → BEARISH (contrarian), < 0.5 → BULLISH.
          Signal tier and historical accuracy included in metadata output.

INDEX CONSTRUCTION (matches notebook Cell 19):
  1. Load 9 stable features from master_with_nlp.parquet
  2. Apply FEATURE_DIRECTION (direction-correct before normalisation)
  3. 252-day rolling percentile rank → normalised 0–1
  4. Weighted sum: group_weight × granger_weight per feature
  5. EMA smooth (span=63)
  6. Scale to 0–100 for display

WEIGHTS:
  Group weights (BEST_WEIGHTS): EIA=2.5, Macro=2.5, COT=1.0, NLP=1.0, Raw LLM=0.5
  Feature weights: loaded from granger_all_results.csv as (1-p)^2.
  Note: granger_all_results.csv is full-sample (descriptive). Ideal source
  is train-period rolling windows (data/Historic/granger_csv). If the
  Historic CSV is available it is preferred automatically.

PREDICTION LAYER:
  severity = |index_01 - 0.5| × 2    (0 = neutral, 1 = extreme)
  threshold_10 = 0.2637               (train-frozen 90th percentile)
  threshold_5  = 0.2879               (train-frozen 95th percentile)
  threshold_2  = 0.3146               (train-frozen 98th percentile)
  signal = sign(0.5 - index_01)       (contrarian: >0.5 → BEARISH)
  active when severity ≥ threshold_10
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

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Frozen parameters — match notebook Cell 02 ───────────────────────────────
TRAIN_END   = pd.Timestamp("2020-01-01")
EMA_SMOOTH  = 63
NORM_WINDOW = 252
GRANGER_POW = 2

# Train-frozen severity thresholds (computed on 2007–2019 training period).
# These values come from the validated notebook and must not be recomputed
# on live data — doing so would introduce threshold leakage.
THRESH_10 = 0.2637   # 90th percentile severity — primary signal tier
THRESH_5  = 0.2879   # 95th percentile severity
THRESH_2  = 0.3146   # 98th percentile severity

# ── Feature groups — frozen spec ──────────────────────────────────────────────
FEATURE_GROUPS = {
    "EIA Fundamentals": ["crude_stocks_change", "refinery_util_pct", "eia_surprise_norm"],
    "Macro Controls":   ["usd_logret", "fed_funds_rate_diff"],
    "COT Positioning":  ["cot_net_long"],
    "NLP Momentum":     ["sent_accel", "sent_ema_cross", "sent_roc_1d", "divergence_ema"],
    "Raw LLM Signals":  ["oil_impact_score", "supply_disruption_signal",
                          "geopolitical_risk_signal", "surface_vs_implied_divergence",
                          "institutional_confidence"],
}

# Group weights — frozen from training-period predictive ranking
BEST_WEIGHTS = {
    "EIA Fundamentals": 2.5,
    "Macro Controls":   2.5,
    "COT Positioning":  1.0,
    "NLP Momentum":     1.0,
    "Raw LLM Signals":  0.5,
}

# FIX #4: Feature direction corrections — applied before normalisation.
# +1 = higher raw value → more greed (bullish institutional communication)
# -1 = higher raw value → more fear (bearish / tightening signal)
# The index is CONTRARIAN: high index (greed) → predict price FALL.
FEATURE_DIRECTION = {
    "crude_stocks_change":           -1,   # inventory build = bearish
    "refinery_util_pct":             +1,   # high utilisation = bullish demand
    "eia_surprise_norm":             -1,   # surprise build = bearish
    "usd_logret":                    -1,   # stronger USD = cheaper oil
    "fed_funds_rate_diff":           -1,   # rising rates = bearish risk
    "cot_net_long":                  +1,   # net long = bullish positioning
    "sent_accel":                    +1,   # accelerating positive sentiment
    "sent_ema_cross":                +1,   # fast > slow = positive momentum
    "sent_roc_1d":                   +1,   # positive day-on-day change
    "divergence_ema":                +1,   # EMA of divergence — greed signal
    "oil_impact_score":              +1,   # positive LLM assessment
    "supply_disruption_signal":      +1,   # supply disruption = bullish price
    "geopolitical_risk_signal":      -1,   # geopolitical risk = uncertainty
    "surface_vs_implied_divergence": -1,   # raw divergence = information gap
    "institutional_confidence":      +1,   # confident language = bullish
}

# FIX #6: Only the 9 features stable in ≥20% of rolling Granger windows
# are used. Episodic (usd_logret, supply_disruption_signal, cot_net_long)
# and rare (sent_accel, sent_roc_1d) features are excluded from the index
# even if present in the data.
STABLE_FEATURES = [
    "crude_stocks_change",
    "eia_surprise_norm",
    "divergence_ema",
    "oil_impact_score",
    "surface_vs_implied_divergence",
    "sent_ema_cross",
    "institutional_confidence",
    "refinery_util_pct",
    "geopolitical_risk_signal",
]

# Historical accuracy by signal tier (from validated OOS test 2020-2026)
TIER_ACCURACY = {
    "top_10": {"full_sample": 0.677, "oos": 0.868, "volatile": 0.987, "stable": 0.641},
    "top_5":  {"full_sample": 0.803, "oos": 0.965},
    "top_2":  {"full_sample": 0.844, "oos": 0.971},
}


def classify_regime(score: float) -> str:
    if np.isnan(score): return "NEUTRAL"
    if score <= 25:     return "EXTREME_FEAR"
    elif score <= 45:   return "FEAR"
    elif score <= 55:   return "NEUTRAL"
    elif score <= 75:   return "GREED"
    else:               return "EXTREME_GREED"


def rolling_percentile(series: pd.Series, window: int = NORM_WINDOW) -> pd.Series:
    """
    FIX #5: 252-day rolling percentile rank.
    Returns fraction of preceding {window} days the current value exceeded.
    Naturally bounded [0,1] with no lookahead — matches notebook Cell 19.
    """
    return series.rolling(
        window, min_periods=int(window * 0.5)
    ).apply(
        lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True
    )


def load_granger_weights(stable_features: list) -> dict:
    """
    Load Granger-based feature weights.
    Prefers train-period rolling windows from data/Historic/ (truly train-frozen).
    Falls back to full-sample static results from granger_all_results.csv
    with a logged warning.
    Default weight 0.5 applied for any feature not found.
    """
    # Preferred: train-period rolling window Granger CSV (data/Historic/)
    historic_csv = HISTORIC_DIR / "granger_causality_results.csv"
    if historic_csv.exists():
        try:
            df = pd.read_csv(historic_csv)
            if "window_end" in df.columns and "p_value" in df.columns:
                df["window_end"] = pd.to_datetime(df["window_end"])
                train_df = df[df["window_end"] < TRAIN_END]
                if len(train_df) > 0:
                    weights = (
                        train_df.groupby("feature")["p_value"]
                        .mean()
                        .apply(lambda p: (1 - p) ** GRANGER_POW)
                    )
                    result = {f: float(weights.get(f, 0.5)) for f in stable_features}
                    log.info(f"✓ Granger weights from train-period rolling windows "
                             f"(pre-{TRAIN_END.date()})")
                    return result
        except Exception as e:
            log.warning(f"  Could not load historic Granger CSV: {e}")

    # Fallback: full-sample static Granger results
    static_csv = RESULTS_DIR / "granger_all_results.csv"
    if static_csv.exists():
        try:
            df = pd.read_csv(static_csv)
            if "p_value" in df.columns and "feature" in df.columns:
                weights = {
                    row["feature"]: (1 - row["p_value"]) ** GRANGER_POW
                    for _, row in df.iterrows()
                    if row["feature"] in stable_features
                }
                result = {f: float(weights.get(f, 0.5)) for f in stable_features}
                log.warning("⚠  Using full-sample Granger weights (descriptive proxy).")
                log.warning("   For train-frozen weights, provide data/Historic/granger_causality_results.csv")
                return result
        except Exception as e:
            log.warning(f"  Could not load granger_all_results.csv: {e}")

    log.warning("  No Granger results available — using equal weights (0.5)")
    return {f: 0.5 for f in stable_features}


def build_full_index():
    master_path = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        log.warning("master_with_nlp.parquet not found — falling back to quantitative only")
        master_path = FEATURES_DIR / "master_quant.parquet"

    if not master_path.exists():
        log.warning("No master dataset found — skipping full index build")
        return

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)
    has_nlp = any("sent_" in c or "oil_impact" in c for c in master.columns)

    available = [f for f in STABLE_FEATURES if f in master.columns]
    missing   = [f for f in STABLE_FEATURES if f not in master.columns]

    log.info(f"Building PRCSI (validated notebook methodology)")
    log.info(f"  Available stable features: {len(available)} / {len(STABLE_FEATURES)}")
    if missing:
        log.info(f"  Missing (will be skipped): {missing}")
    log.info(f"  NLP scores present: {has_nlp}")

    # ── Load feature weights ───────────────────────────────────────────────
    feat_weights = load_granger_weights(available)

    # ── FIX #4+#5: Direction-correct then rolling percentile normalise ────
    log.info("  Normalising features (rolling percentile, direction-corrected)...")
    normalised = pd.DataFrame(index=master.index)
    for feat in available:
        s = master[feat].copy()
        direction = FEATURE_DIRECTION.get(feat, 1)
        if direction == -1:
            s = -s          # invert so high value always = more greed
        normalised[feat] = rolling_percentile(s)

    # ── FIX #3+#6: Build weighted index using only stable features ────────
    log.info("  Building weighted index...")
    idx_vals = pd.Series(np.nan, index=master.index)

    for date in master.index:
        ws, tw = 0.0, 0.0
        for group, features in FEATURE_GROUPS.items():
            gw = BEST_WEIGHTS.get(group, 1.0)
            for feat in features:
                if feat not in STABLE_FEATURES or feat not in available:
                    continue
                v = normalised.at[date, feat] if date in normalised.index else np.nan
                if pd.isna(v):
                    continue
                w = gw * feat_weights.get(feat, 0.5)
                ws += w * v
                tw += w
        if tw > 0:
            idx_vals[date] = ws / tw

    # FIX #5: EMA smooth span=63 (matches notebook)
    idx_01     = idx_vals.ewm(span=EMA_SMOOTH, min_periods=10).mean()  # 0–1 scale
    idx_100    = idx_01 * 100                                           # 0–100 display

    log.info(f"  Index built: {idx_01.notna().sum()} non-null values | "
             f"range {idx_01.min():.3f} → {idx_01.max():.3f}")

    # ── FIX #7: Prediction layer ──────────────────────────────────────────
    severity = (idx_01 - 0.5).abs() * 2   # 0 = neutral, 1 = extreme

    def get_signal_tier(sev: float) -> str:
        if np.isnan(sev):   return "none"
        if sev >= THRESH_2: return "top_2"
        if sev >= THRESH_5: return "top_5"
        if sev >= THRESH_10: return "top_10"
        return "none"

    def get_signal_direction(idx_val: float, tier: str) -> str:
        if tier == "none" or np.isnan(idx_val):
            return "NONE"
        return "BEARISH" if idx_val > 0.5 else "BULLISH"

    signal_tier      = severity.apply(get_signal_tier)
    signal_direction = pd.Series([
        get_signal_direction(v, t)
        for v, t in zip(idx_01, signal_tier)
    ], index=master.index)
    signal_active = signal_tier != "none"

    # Identify publication days for NLP freshness flag
    pub_days = set()
    if has_nlp:
        # A publication day is one where NLP raw scores have genuine data
        # (not just forward-filled). Approximate by checking for value changes
        # in oil_impact_score.
        if "oil_impact_score" in master.columns:
            ois = master["oil_impact_score"]
            pub_days = set(ois.index[ois.diff().abs() > 0.001])

    nlp_is_fresh = pd.Series(
        [d in pub_days for d in master.index],
        index=master.index
    )

    # ── Build result DataFrame ────────────────────────────────────────────
    result = pd.DataFrame({
        "prcsi_01":         idx_01,
        "prcsi":            idx_100,
        "regime":           idx_100.apply(classify_regime),
        "severity":         severity,
        "signal_active":    signal_active,
        "signal_direction": signal_direction,
        "signal_tier":      signal_tier,
        "nlp_is_fresh":     nlp_is_fresh,
        "oil_price":        master["oil"]        if "oil"        in master.columns else np.nan,
        "oil_logret":       master["oil_logret"] if "oil_logret" in master.columns else np.nan,
    })

    result.to_parquet(RESULTS_DIR / "prcsi_final.parquet")
    result.to_csv(RESULTS_DIR    / "prcsi_final.csv")

    # ── Dashboard chart ───────────────────────────────────────────────────
    _build_dashboard(result, has_nlp)

    # ── Metadata update ───────────────────────────────────────────────────
    latest_row = result.dropna(subset=["prcsi"]).iloc[-1]
    latest_date      = str(latest_row.name.date())
    latest_score     = round(float(latest_row["prcsi"]), 2)
    latest_regime    = str(latest_row["regime"])
    latest_severity  = round(float(latest_row["severity"]), 4)
    latest_tier      = str(latest_row["signal_tier"])
    latest_direction = str(latest_row["signal_direction"])
    latest_active    = bool(latest_row["signal_active"])
    latest_fresh     = bool(latest_row["nlp_is_fresh"])

    # Count active signals in last 30 days for cadence reporting
    recent = result.last("30D")
    signals_30d     = int(recent["signal_active"].sum())
    signals_30d_correct = None   # accuracy requires outcome data — not available here

    tier_acc = TIER_ACCURACY.get(latest_tier, {})

    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        # Current index state
        "prcsi_latest":           latest_score,
        "prcsi_regime":           latest_regime,
        "prcsi_date":             latest_date,
        "prcsi_severity":         latest_severity,
        # Prediction layer
        "signal_active":          latest_active,
        "signal_direction":       latest_direction,
        "signal_tier":            latest_tier,
        "signal_threshold_used":  THRESH_10,
        "signal_horizon_days":    [21, 42],
        "tier_accuracy_full":     tier_acc.get("full_sample"),
        "tier_accuracy_oos":      tier_acc.get("oos"),
        # NLP freshness
        "nlp_is_fresh":           latest_fresh,
        # Signal cadence
        "signals_last_30d":       signals_30d,
        # Pipeline metadata
        "full_index_complete":    True,
        "nlp_scores_used":        has_nlp,
        "n_stable_features_used": len(available),
        "full_run_timestamp":     datetime.now().isoformat(),
        # Methodology note
        "index_method":           "rolling_percentile_ema63_stable9_contrarian",
        "validated_oos_accuracy": "86.8% top-10% OOS (21d horizon, 5 independent blocks)",
    })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"\n✓ Full PRCSI index built (validated methodology)")
    log.info(f"  Latest score:      {latest_score:.1f} / 100  ({latest_regime})")
    log.info(f"  Severity:          {latest_severity:.4f}")
    log.info(f"  Signal active:     {latest_active}")
    if latest_active:
        log.info(f"  Signal direction:  {latest_direction}  (tier: {latest_tier})")
        log.info(f"  Historical OOS accuracy at this tier: {tier_acc.get('oos', 'n/a')}")
    log.info(f"  NLP data fresh:    {latest_fresh}")
    log.info(f"  NLP scores:        {'✅ included' if has_nlp else '⚠️  not included'}")
    log.info(f"  Features used:     {len(available)} / {len(STABLE_FEATURES)} stable")

    return result


def _build_dashboard(result: pd.DataFrame, has_nlp: bool):
    """Build 3-panel dashboard: price / index+regime / severity+signal."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 2.5, 1]})
    fig.suptitle(
        f"Oil Fear & Greed Index (PRCSI) — "
        f"{'Full NLP+Quant' if has_nlp else 'Quantitative Only'}\n"
        f"Latest: {result['prcsi'].dropna().iloc[-1]:.1f} "
        f"({result['regime'].dropna().iloc[-1]})",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: WTI price
    if "oil_price" in result.columns:
        axes[0].plot(result.index, result["oil_price"], color="#E74C3C", linewidth=0.9)
        axes[0].set_ylabel("WTI (USD)", fontsize=9)
        axes[0].grid(alpha=0.2)

    # Panel 2: PRCSI index
    ax = axes[1]
    ax.fill_between(result.index, 50, result["prcsi"],
                    where=result["prcsi"] >= 50,
                    alpha=0.12, color="#E74C3C")
    ax.fill_between(result.index, result["prcsi"], 50,
                    where=result["prcsi"] < 50,
                    alpha=0.12, color="#2563EB")
    ax.plot(result.index, result["prcsi"], color="#1B2A4A", linewidth=1.2)
    for y, label, col in [
        (25, "Extreme Fear", "#1d4ed8"),
        (45, "Fear",         "#3b82f6"),
        (55, "Greed",        "#dc2626"),
        (75, "Extreme Greed","#7f1d1d"),
    ]:
        ax.axhline(y=y, color=col, linestyle="--", alpha=0.3, linewidth=0.8)
        ax.text(result.index[-1], y + 1, f" {label}", va="bottom",
                fontsize=7.5, color=col)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 100)
    ax.set_ylabel("PRCSI (0=fear, 1=greed)", fontsize=9)
    ax.grid(alpha=0.2)

    # Mark active signal days
    active_dates = result[result["signal_active"]].index
    if len(active_dates):
        bearish = result[result["signal_active"] & (result["signal_direction"] == "BEARISH")]
        bullish = result[result["signal_active"] & (result["signal_direction"] == "BULLISH")]
        if len(bearish):
            ax.scatter(bearish.index, bearish["prcsi"],
                       c="#dc2626", s=12, zorder=4, alpha=0.7,
                       label=f"Bearish signal (n={len(bearish)})")
        if len(bullish):
            ax.scatter(bullish.index, bullish["prcsi"],
                       c="#15803d", s=12, zorder=4, alpha=0.7,
                       label=f"Bullish signal (n={len(bullish)})")
        ax.legend(fontsize=7.5, loc="upper left")

    # Panel 3: Severity
    ax3 = axes[2]
    ax3.fill_between(result.index, 0, result["severity"], alpha=0.5, color="#1565C0")
    ax3.axhline(THRESH_10, color="#B71C1C", linewidth=1.0, linestyle="--",
                label=f"Top 10% threshold ({THRESH_10:.4f})")
    ax3.axhline(THRESH_5,  color="#7B1FA2", linewidth=0.8, linestyle=":",
                label=f"Top 5% threshold ({THRESH_5:.4f})")
    ax3.set_ylabel("Severity", fontsize=9)
    ax3.set_ylim(0, 0.55)
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.grid(alpha=0.2)

    for ax in axes:
        ax.set_xlim(result.index.min(), result.index.max())
    fig.autofmt_xdate(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prcsi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Dashboard saved → {RESULTS_DIR / 'prcsi_dashboard.png'}")


if __name__ == "__main__":
    build_full_index()
