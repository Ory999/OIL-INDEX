"""
Script 09 — Granger Causality Tests
Runs on master_with_nlp.parquet (full combined quant + NLP signal).

This is the core test of the research question:
"Does the COMPLETE PRCSI signal (fundamentals + rhetoric momentum)
 Granger-cause WTI crude oil price movements, beyond what past
 prices and fundamentals alone explain?"

Running Granger AFTER the qualitative pipeline means we test the
full combined product — which is the actual research contribution.
"""
import os, logging
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib
matplotlib.use("Agg")  # non-interactive for CI
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR  = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR   = Path(os.getenv("RESULTS_DIR",  "data/results"))
MAX_LAG       = int(os.getenv("GRANGER_MAX_LAG", "10"))
SIGNIFICANCE  = 0.05
TARGET        = "oil_logret"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_granger_battery(df: pd.DataFrame, target: str, features: list) -> pd.DataFrame:
    results = []
    df_clean = df[[target] + features].dropna()

    for feat in features:
        pair = df_clean[[target, feat]].dropna()
        if len(pair) < MAX_LAG * 5:
            continue
        try:
            gc = grangercausalitytests(pair, maxlag=MAX_LAG, verbose=False)
            best_p, best_lag, best_f = 1.0, 1, 0.0
            for lag, res in gc.items():
                p = float(res[0]["ssr_ftest"][1])
                f = float(res[0]["ssr_ftest"][0])
                if p < best_p:
                    best_p, best_lag, best_f = p, lag, f

            corr = pair[feat].corr(pair[target].shift(-best_lag))
            results.append({
                "feature":     feat,
                "best_lag":    best_lag,
                "f_stat":      round(best_f, 4),
                "p_value":     round(best_p, 6),
                "significant": best_p < SIGNIFICANCE,
                "direction":   "+" if corr > 0 else "-",
                "n_obs":       len(pair),
            })
        except Exception as e:
            log.debug(f"  Granger failed for {feat}: {e}")

    return pd.DataFrame(results).sort_values("p_value")


def run_granger_causality():
    # Always use full NLP-merged dataset if available
    master_path = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        log.warning("master_with_nlp.parquet not found — using quantitative master")
        log.warning("NLP sentiment scores are NOT included in this Granger test")
        master_path = FEATURES_DIR / "master_quant.parquet"

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)
    master = master.dropna(subset=[TARGET])

    has_nlp = any("sent_" in c or "nlp_" in c or "oil_impact" in c
                  for c in master.columns)
    log.info(f"Testing: {'FULL combined signal (quant + NLP)' if has_nlp else 'quantitative only'}")

    # ── Define feature groups for structured reporting ─────────────────────
    feature_groups = {
        "NLP Sentiment Momentum": [c for c in master.columns if any(
            x in c for x in ["sent_roc","sent_ema","sent_rsi","sent_accel",
                              "sent_velocity","sent_momentum"])],

        "NLP Raw Scores": [c for c in master.columns if any(
            x in c for x in ["oil_impact","supply_disruption","geopolitical",
                              "finbert","demand_outlook"])],

        "LLM Divergence": [c for c in master.columns if "divergence" in c],

        "BERTopic Streams": [c for c in master.columns
                              if c.startswith("topic_") and "ema" not in c],

        "EIA Fundamentals": [c for c in master.columns if any(
            x in c for x in ["eia_surprise","crude_stocks","refinery",
                              "crude_production"])],

        "COT Positioning": [c for c in master.columns
                             if "cot_" in c],

        "Congressional Trades": [c for c in master.columns
                                  if "congress_" in c],

        "Macro Controls": [c for c in master.columns if any(
            x in c for x in ["fed_funds","tips_","breakeven","umich",
                              "usd_logret","vix_logret"])],
    }

    all_results = {}
    for group_name, features in feature_groups.items():
        available = [f for f in features if f in master.columns
                     and f != TARGET]
        if not available:
            continue
        log.info(f"\nTesting: {group_name} ({len(available)} features)")
        res = run_granger_battery(master, TARGET, available)
        all_results[group_name] = res
        sig = res[res["significant"]]
        log.info(f"  Significant: {len(sig)} / {len(res)}")
        if len(sig):
            log.info(f"  Best: {sig.iloc[0]['feature']} (p={sig.iloc[0]['p_value']:.4f}, lag={sig.iloc[0]['best_lag']}d)")

    # ── Consolidate ────────────────────────────────────────────────────────
    combined = pd.concat(all_results.values(), ignore_index=True)
    for group, res in all_results.items():
        combined.loc[combined["feature"].isin(res["feature"]), "group"] = group

    combined.to_csv(RESULTS_DIR / "granger_all_results.csv", index=False)

    sig_all = combined[combined["significant"]].sort_values("p_value")
    sig_all.to_csv(RESULTS_DIR / "granger_significant.csv", index=False)

    log.info(f"\n✓ Granger complete: {len(sig_all)} significant features (p<{SIGNIFICANCE})")
    if len(sig_all):
        log.info(f"  Top 5 features:")
        for _, row in sig_all.head(5).iterrows():
            log.info(f"    {row['feature']:45s} p={row['p_value']:.4f}  lag={row['best_lag']}d  ({row['group']})")

    # ── Visualise ──────────────────────────────────────────────────────────
    if len(combined):
        top20 = combined.nsmallest(20, "p_value")
        colors = ["#15803D" if s else "#D1D5DB" for s in top20["significant"]]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(top20["feature"].str[:45], top20["f_stat"], color=colors)
        ax.axvline(x=3.84, color="#B91C1C", linestyle="--", linewidth=1.5,
                   label="p≈0.05 threshold")
        ax.set_xlabel("F-statistic")
        ax.set_title(
            f"Granger Causality — Top 20 Features → WTI Oil Log Returns\n"
            f"({'Full NLP+Quant signal' if has_nlp else 'Quantitative only'})",
            fontsize=12
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "granger_results.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Chart saved → {RESULTS_DIR / 'granger_results.png'}")

    return combined


if __name__ == "__main__":
    run_granger_causality()
