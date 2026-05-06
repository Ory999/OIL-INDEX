"""
Script 02 — Fetch EIA Fundamentals
Source: EIA Open Data API (eia.gov/opendata) — free with registration
Outputs: data/raw/eia_fundamentals.parquet

EIA weekly inventory release is the #1 short-term oil price driver.
Must be controlled before attributing residual price movement to sentiment.
"""
import os, logging, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_DATE  = os.getenv("START_DATE", "2007-01-01")
RAW_DIR     = Path(os.getenv("DATA_DIR", "data/raw"))
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
RAW_DIR.mkdir(parents=True, exist_ok=True)

EIA_BASE = "https://api.eia.gov/v2"

# Series definitions — each is a critical fundamental oil driver
EIA_SERIES = {
    # US crude oil stocks (million barrels) — inventory level
    "crude_stocks_mbbl":
        "petroleum/sum/sndw/data/?facets[series][]=WCRSTUS1",
    # Refinery utilisation (%) — demand signal
    "refinery_util_pct":
        "petroleum/pnp/wiup/data/?facets[series][]=WCRRIUS2",
    # US crude production (million barrels/day)
    "crude_production_mbpd":
        "petroleum/sum/sndw/data/?facets[series][]=WCRFPUS2",
    # US crude imports (million barrels/day) — supply diversity signal
    "crude_imports_mbpd":
        "petroleum/sum/sndw/data/?facets[series][]=WCRIMUS2",
}

def fetch_eia_series(name: str, endpoint: str, api_key: str) -> pd.Series:
    url = (
        f"{EIA_BASE}/{endpoint}"
        f"&api_key={api_key}"
        f"&frequency=weekly"
        f"&data[0]=value"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&length=2000"
        f"&start={START_DATE}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    data = r.json().get("response", {}).get("data", [])
    if not data:
        raise ValueError(f"No data returned for {name}")

    s = pd.Series(
        {d["period"]: float(d["value"])
         for d in data
         if d.get("value") not in [None, ""]},
        name=name,
    )
    s.index = pd.to_datetime(s.index)
    log.info(f"  ✓ {name}: {len(s)} weekly observations")
    return s


def fetch_eia_fundamentals():
    out = RAW_DIR / "eia_fundamentals.parquet"

    if not EIA_API_KEY:
        log.warning("EIA_API_KEY not set — generating synthetic fundamentals")
        dates = pd.bdate_range(START_DATE, datetime.today())
        df = pd.DataFrame({
            "crude_stocks_mbbl":      np.random.normal(430, 20, len(dates)),
            "crude_stocks_change":    np.random.normal(0, 5,  len(dates)),
            "refinery_util_pct":      np.random.normal(90, 3,  len(dates)),
            "crude_production_mbpd":  np.random.normal(12, 1,  len(dates)),
            "crude_imports_mbpd":     np.random.normal(6,  1,  len(dates)),
            "eia_surprise":           np.random.normal(0,  3,  len(dates)),
        }, index=dates)
        df.to_parquet(out)
        log.warning(f"  Synthetic EIA data saved → {out}")
        return df

    series_list = []
    for name, endpoint in EIA_SERIES.items():
        try:
            s = fetch_eia_series(name, endpoint, EIA_API_KEY)
            series_list.append(s)
        except Exception as e:
            log.warning(f"  ⚠ EIA series {name} failed: {e}")

    if not series_list:
        raise RuntimeError("All EIA series failed — check API key")

    df = pd.concat(series_list, axis=1).sort_index()

    # Week-on-week inventory change — the market-moving surprise metric
    df["crude_stocks_change"] = df["crude_stocks_mbbl"].diff()

    # EIA surprise = actual change minus 4-week rolling average
    # Positive surprise (unexpected build) = bearish for oil
    # Negative surprise (unexpected draw)  = bullish for oil
    df["eia_surprise"] = (
        df["crude_stocks_change"]
        - df["crude_stocks_change"].rolling(4, min_periods=1).mean()
    )

    # Resample to business day frequency, forward fill (weekly → daily)
    df = df.resample("B").ffill(limit=5)

    if len(df) == 0:
        log.warning("  EIA data empty after resampling — check API response")
        pd.DataFrame().to_parquet(out)
        return pd.DataFrame()

    df.to_parquet(out)
    log.info(f"✓ EIA fundamentals saved → {out}  ({len(df)} rows)")
    log.info(f"  Latest crude stocks: {df['crude_stocks_mbbl'].iloc[-1]:.1f} Mbbl")
    log.info(f"  Latest EIA surprise: {df['eia_surprise'].iloc[-1]:.2f} Mbbl")
    return df


if __name__ == "__main__":
    fetch_eia_fundamentals()
