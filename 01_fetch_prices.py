"""
Script 01 — Fetch Market Prices
Sources: yfinance
Outputs: data/raw/prices.parquet
Gold removed from scope. WTI oil is primary target variable.
"""
import os, logging
from pathlib import Path
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_DATE = os.getenv("START_DATE", "2007-01-01")
END_DATE   = datetime.today().strftime("%Y-%m-%d")
RAW_DIR    = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "oil": "CL=F",      # WTI Crude Futures — PRIMARY TARGET
    "vix": "^VIX",      # Fear index — control variable
    "usd": "DX-Y.NYB",  # Dollar index — control variable
}

def fetch_prices():
    log.info(f"Fetching prices: {START_DATE} → {END_DATE}")

    raw = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,
    )

    # Handle both single and multi-ticker download formats
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        prices.columns = list(TICKERS.keys())
    else:
        prices = raw[["Close"]].copy()
        prices.columns = list(TICKERS.keys())

    prices.index = pd.to_datetime(prices.index)
    prices       = prices.resample("B").last().ffill()

    # Percentage returns
    returns = prices.pct_change()
    returns.columns = [f"{c}_return" for c in prices.columns]

    # Log returns — preferred for Granger causality (more stationary)
    log_ret = np.log(prices / prices.shift(1))
    log_ret.columns = [f"{c}_logret" for c in prices.columns]

    df = pd.concat([prices, returns, log_ret], axis=1).dropna(how="all")

    out = RAW_DIR / "prices.parquet"
    df.to_parquet(out)
    log.info(f"✓ Prices saved → {out}  ({len(df)} rows)")
    log.info(f"  Columns: {list(df.columns)}")
    log.info(f"  Latest oil price: ${df['oil'].iloc[-1]:.2f}")
    return df

if __name__ == "__main__":
    fetch_prices()
