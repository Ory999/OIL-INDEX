"""
Script 06 — Assemble Master Dataset
Merges all quantitative streams onto a single business day index.
Outputs: data/features/master_quant.parquet

This is the spine all downstream scripts read from.
The qualitative pipeline will later merge NLP scores into
data/features/master_with_nlp.parquet — the full combined dataset
on which Granger causality is run.
"""
import os, logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",      "data/raw"))
FEATURES_DIR = Path(os.getenv("FEATURES_DIR",  "data/features"))
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def load_parquet(path: Path, name: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        # Strip timezone info for consistent merging
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        log.info(f"  ✓ Loaded {name}: {df.shape}")
        return df
    except Exception as e:
        log.warning(f"  ⚠ Could not load {name}: {e}")
        return None


def assemble_master():
    out = FEATURES_DIR / "master_quant.parquet"

    # ── Load all raw streams ──────────────────────────────────────────────
    prices  = load_parquet(RAW_DIR / "prices.parquet",           "prices")
    eia     = load_parquet(RAW_DIR / "eia_fundamentals.parquet", "EIA")
    fred    = load_parquet(RAW_DIR / "fred_macro.parquet",       "FRED")
    cot     = load_parquet(RAW_DIR / "cot_crude.parquet",        "COT")

    if prices is None or len(prices) == 0:
        raise RuntimeError("prices.parquet is empty or missing — check 01 fetch prices.py")

    if "oil_logret" not in prices.columns:
        raise RuntimeError("oil_logret column missing from prices — check 01 fetch prices.py")

    # ── Build master on business day spine (prices always available) ──────
    master = prices.copy()

    for df, name in [
        (eia,  "EIA"),
        (fred, "FRED"),
        (cot,  "COT"),
    ]:
        if df is None:
            log.warning(f"  Skipping {name} — not available")
            continue
        # Avoid duplicate column names
        overlap = set(master.columns) & set(df.columns)
        if overlap:
            df = df.drop(columns=list(overlap))
        master = master.join(df, how="left")

    # ── Forward fill weekly/monthly series to daily ───────────────────────
    fill_cols = [c for c in master.columns if any(
        x in c for x in [
            "crude_stocks", "refinery", "crude_production", "crude_imports",
            "eia_surprise", "cot_net", "cot_long",
            "tips_", "breakeven", "umich", "fed_funds", "sp500",
        ]
    )]
    master[fill_cols] = master[fill_cols].ffill(limit=5)

    # ── Derived features ──────────────────────────────────────────────────
    # EIA surprise: actual vs. expected (4-week rolling mean)
    if "crude_stocks_change" in master.columns:
        master["eia_surprise_norm"] = (
            master["crude_stocks_change"]
            - master["crude_stocks_change"].rolling(4, min_periods=1).mean()
        ) / (master["crude_stocks_change"].rolling(4, min_periods=1).std() + 1e-8)

    # COT positioning rate of change
    if "cot_net_long" in master.columns:
        master["cot_change_1w"] = master["cot_net_long"].diff(5)
        master["cot_change_4w"] = master["cot_net_long"].diff(20)

    # Dollar momentum (oil is inversely correlated with USD)
    if "usd_logret" in master.columns:
        master["usd_ema_cross"] = (
            master["usd_logret"].ewm(span=5).mean()
            - master["usd_logret"].ewm(span=20).mean()
        )

    # ── Remove rows where target variable is missing ──────────────────────
    master = master.dropna(subset=["oil_logret"])

    # ── Summary statistics ────────────────────────────────────────────────
    n_features   = master.shape[1]
    n_days       = len(master)
    pct_missing  = master.isnull().mean().mean() * 100
    date_range   = f"{master.index.min().date()} → {master.index.max().date()}"

    log.info(f"\n✓ Master quantitative dataset assembled")
    log.info(f"  Shape:         {master.shape}")
    log.info(f"  Date range:    {date_range}")
    log.info(f"  Missing data:  {pct_missing:.1f}%")
    log.info(f"  Features:      {n_features}")

    master.to_parquet(out)
    log.info(f"  Saved → {out}")

    # ── Write a summary JSON for GitHub Actions logging ───────────────────
    import json
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "n_trading_days": n_days,
        "n_features": n_features,
        "date_range_start": str(master.index.min().date()),
        "date_range_end":   str(master.index.max().date()),
        "pct_missing": round(pct_missing, 2),
        "columns": list(master.columns),
        "latest_oil_price":  round(float(master["oil"].iloc[-1]),  2) if "oil"  in master.columns else None,
        "latest_oil_logret": round(float(master["oil_logret"].dropna().iloc[-1]), 5) if len(master["oil_logret"].dropna()) > 0 else None,
    }
    summary_path = Path(os.getenv("RESULTS_DIR", "data/results"))
    summary_path.mkdir(parents=True, exist_ok=True)
    with open(summary_path / "pipeline_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Metadata saved → {summary_path / 'pipeline_metadata.json'}")

    return master


if __name__ == "__main__":
    assemble_master()
