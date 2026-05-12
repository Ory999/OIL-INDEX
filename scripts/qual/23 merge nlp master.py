"""
Script 23 — Merge NLP Scores into Master Dataset
Joins sentiment momentum features with quantitative master dataset.
Outputs: data/features/master_with_nlp.parquet

This is the final combined dataset on which Granger causality is run.
The merge triggers the econometric pipeline automatically via workflow_run.

Merge strategy:
  Left join on trading date — all quantitative days are preserved.
  NLP features forward-filled with differentiated limits:
    - Raw LLM signals (oil_impact_score etc.): 21-day fill
        OPEC and EIA publish monthly — raw scores must persist for a full
        publication cycle to be usable as direct Granger causality inputs.
        Script 22 already applies this fill; the 21-day limit here ensures
        any residual gaps at the join boundary are also closed.
    - Momentum/EMA features: 5-day fill
        Short-term signals should decay — stale momentum beyond 5 days
        is not meaningful and should remain NaN.
  Rows where oil_logret is missing are dropped before saving.
"""
import os, logging, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Raw LLM signals — forward-filled 21 days (one publication cycle)
RAW_LLM_COLS = [
    "oil_impact_score",
    "supply_disruption_signal",
    "demand_outlook_signal",
    "geopolitical_risk_signal",
    "surface_vs_implied_divergence",
    "institutional_confidence",
]


def merge_nlp_master():
    quant_path = FEATURES_DIR / "master_quant.parquet"
    sent_path  = RAW_DIR      / "sentiment_features.parquet"
    out        = FEATURES_DIR / "master_with_nlp.parquet"

    # ── Load quantitative master ───────────────────────────────────────────
    if not quant_path.exists():
        log.warning("master_quant.parquet not found — quantitative pipeline may not have run yet")
        return pd.DataFrame()

    master = pd.read_parquet(quant_path)
    master.index = pd.to_datetime(master.index)
    if hasattr(master.index, "tz") and master.index.tz:
        master.index = master.index.tz_localize(None)
    log.info(f"Quantitative master: {master.shape}")

    # ── Load sentiment features ────────────────────────────────────────────
    if not sent_path.exists():
        log.warning("sentiment_features.parquet not found — saving quant-only master")
        master.to_parquet(out)
        return master

    sent = pd.read_parquet(sent_path)
    sent.index = pd.to_datetime(sent.index)
    if hasattr(sent.index, "tz") and sent.index.tz:
        sent.index = sent.index.tz_localize(None)
    log.info(f"Sentiment features:  {sent.shape}")

    # ── Avoid column name collisions ───────────────────────────────────────
    overlap = set(master.columns) & set(sent.columns)
    if overlap:
        log.info(f"  Dropping overlapping columns from NLP: {overlap}")
        sent = sent.drop(columns=list(overlap))

    # ── Left join — keep all quantitative trading days ─────────────────────
    merged = master.join(sent, how="left")
    nlp_cols = [c for c in sent.columns if c in merged.columns]

    # ── Forward fill NLP features — differentiated by signal type ─────────
    # Raw LLM signals: 21-day fill (one full OPEC/EIA publication cycle).
    # Economically justified — market holds the most recent institutional
    # view until the next publication replaces it.
    raw_llm_in_merged = [c for c in RAW_LLM_COLS if c in merged.columns]
    if raw_llm_in_merged:
        merged[raw_llm_in_merged] = merged[raw_llm_in_merged].ffill(limit=21)
        log.info(f"  Raw LLM signals ({len(raw_llm_in_merged)} cols): 21-day fill applied")

    # Momentum/EMA features: 5-day fill — short-term signals decay,
    # stale momentum beyond 5 trading days should remain NaN.
    momentum_cols = [c for c in nlp_cols if c not in raw_llm_in_merged]
    if momentum_cols:
        merged[momentum_cols] = merged[momentum_cols].ffill(limit=5)
        log.info(f"  Momentum features  ({len(momentum_cols)} cols): 5-day fill applied")

    # ── Drop rows where target variable is missing ─────────────────────────
    if "oil_logret" in merged.columns:
        before = len(merged)
        merged = merged.dropna(subset=["oil_logret"])
        dropped = before - len(merged)
        if dropped:
            log.info(f"  Dropped {dropped} rows with missing oil_logret")

    # ── Coverage stats ─────────────────────────────────────────────────────
    nlp_coverage     = merged[nlp_cols].notna().mean().mean() * 100
    nlp_col_coverage = merged[nlp_cols].notna().mean().sort_values(ascending=False)

    log.info(f"\n✓ Merged master dataset:")
    log.info(f"  Shape:        {merged.shape}")
    log.info(f"  Date range:   {merged.index.min().date()} → {merged.index.max().date()}")
    log.info(f"  NLP coverage: {nlp_coverage:.1f}%")

    # Raw LLM signal coverage breakdown
    log.info(f"\n  Raw LLM signal coverage:")
    for col in raw_llm_in_merged:
        pct = merged[col].notna().mean() * 100
        flag = " ✅" if pct >= 90 else " ⚠️"
        log.info(f"    {col:40s}: {pct:.1f}%{flag}")

    # Warn on any NLP columns still below 50% after fills
    low_coverage = nlp_col_coverage[nlp_col_coverage < 0.5]
    if len(low_coverage):
        log.warning(f"\n  Low coverage NLP columns (<50%) — these will have limited "
                    f"Granger test power:")
        for col, cov in low_coverage.items():
            log.warning(f"    {col:40s}: {cov*100:.1f}%")

    if "oil_impact_score" in merged.columns:
        log.info(f"\n  oil_impact_score (mean): {merged['oil_impact_score'].mean():+.4f}")
    if "sent_ema_cross" in merged.columns:
        log.info(f"  sent_ema_cross (mean):   {merged['sent_ema_cross'].mean():+.4f}")
    if "sent_accel" in merged.columns:
        log.info(f"  sent_accel (mean):       {merged['sent_accel'].mean():+.4f}")

    merged.to_parquet(out)
    log.info(f"\n  Saved → {out}")

    # ── Update pipeline metadata ───────────────────────────────────────────
    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    metadata.update({
        "nlp_scores_merged":    True,
        "nlp_merge_timestamp":  datetime.now().isoformat(),
        "nlp_coverage_pct":     round(nlp_coverage, 2),
        "n_nlp_features":       len(nlp_cols),
        "n_raw_llm_features":   len(raw_llm_in_merged),
        "n_momentum_features":  len(momentum_cols),
        "n_combined_features":  merged.shape[1],
        "n_combined_rows":      len(merged),
        "combined_date_start":  str(merged.index.min().date()),
        "combined_date_end":    str(merged.index.max().date()),
    })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata updated → {meta_path}")

    return merged


if __name__ == "__main__":
    merge_nlp_master()
