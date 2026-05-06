"""
Script 03 — Fetch FRED Macro Controls
Source: Federal Reserve FRED API (fred.stlouisfed.org) — free with registration
Outputs: data/raw/fred_macro.parquet
Macro variables that affect oil demand and must be controlled for
so Granger tests isolate the institutional rhetoric signal.
"""
import os, logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_DATE   = os.getenv("START_DATE", "2007-01-01")
RAW_DIR      = Path(os.getenv("DATA_DIR", "data/raw"))
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRED_SERIES = {
    "fed_funds_rate":   "DFF",
    "tips_10yr":        "DFII10",
    "breakeven_10yr":   "T10YIE",
    "usd_broad":        "DTWEXBGS",
    "umich_sentiment":  "UMCSENT",
    "sp500":            "SP500",
}

def fetch_fred_data():
    out = RAW_DIR / "fred_macro.parquet"

    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — generating synthetic macro data")
        dates = pd.bdate_range(START_DATE, datetime.today())
        df = pd.DataFrame({
            "fed_funds_rate":  np.random.uniform(0, 5.5,  len(dates)),
            "tips_10yr":       np.random.normal(0.5, 1.5, len(dates)),
            "breakeven_10yr":  np.random.normal(2.2, 0.5, len(dates)),
            "usd_broad":       np.random.normal(110, 10,  len(dates)),
            "umich_sentiment": np.random.normal(75,  10,  len(dates)),
            "sp500":           np.random.normal(4000, 800, len(dates)),
        }, index=dates)
        df.to_parquet(out)
        log.warning(f"  Synthetic FRED data saved → {out}")
        return df

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    dfs = []
    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=datetime.today().strftime("%Y-%m-%d"),
            )
            s.name = name
            dfs.append(s)
            log.info(f"  ✓ {name} ({series_id}): {len(s)} observations")
        except Exception as e:
            log.warning(f"  ⚠ FRED {name} ({series_id}) failed: {e}")

    if not dfs:
        raise RuntimeError("All FRED series failed — check API key")

    df = (
        pd.concat(dfs, axis=1)
        .resample("B")
        .ffill(limit=5)
        .dropna(how="all")
    )

    if len(df) == 0:
        log.warning("  FRED data empty after resampling")
        pd.DataFrame().to_parquet(out)
        return pd.DataFrame()

    df.to_parquet(out)
    log.info(f"✓ FRED macro saved → {out}  ({len(df)} rows)")

    if len(df) > 0 and "fed_funds_rate" in df.columns:
        log.info(f"  Latest fed funds rate: {df['fed_funds_rate'].iloc[-1]:.2f}%")

    return df

if __name__ == "__main__":
    fetch_fred_data()
