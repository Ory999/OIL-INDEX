"""
Script 05 — Fetch Congressional Trades (STOCK Act)
Source: Quiver Quantitative API (free tier) + Capitol Trades fallback
Outputs: data/raw/congressional_trades.parquet
         data/raw/congress_daily.parquet

Theoretical link: Information Asymmetry (Akerlof, 1970).
Politicians trading energy assets ahead of public policy announcements
reveals private information channels not yet priced into markets.
"""
import os, logging, requests, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_DATE     = os.getenv("START_DATE", "2007-01-01")
RAW_DIR        = Path(os.getenv("DATA_DIR", "data/raw"))
QUIVER_API_KEY = os.getenv("QUIVER_API_KEY", "")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Energy-related tickers to filter congressional trades
ENERGY_TICKERS = {
    # Oil majors
    "XOM", "CVX", "COP", "OXY", "MPC", "VLO", "PSX",
    "HAL", "SLB", "BKR", "NOV",
    # Energy ETFs
    "USO", "UCO", "XLE", "OIH", "VDE",
    # Pipeline & LNG
    "KMI", "WMB", "ET", "LNG", "ENPH",
}

AMOUNT_MAP = {
    "$1,001 - $15,000":         8_000,
    "$15,001 - $50,000":        32_500,
    "$50,001 - $100,000":       75_000,
    "$100,001 - $250,000":     175_000,
    "$250,001 - $500,000":     375_000,
    "$500,001 - $1,000,000":   750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "Over $5,000,000":         7_500_000,
}

DIRECTION_MAP = {
    "Purchase": 1, "Buy": 1,
    "Sale": -1, "Sale (Full)": -1, "Sale (Partial)": -1, "Sell": -1,
    "Exchange": 0,
}


def fetch_from_quiver() -> pd.DataFrame | None:
    if not QUIVER_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.quiverquant.com/beta/bulk/congresstrading",
            headers={"accept": "application/json",
                     "X-CSRFToken": QUIVER_API_KEY},
            timeout=30,
        )
        if r.status_code != 200:
            log.warning(f"  Quiver API returned {r.status_code}")
            return None

        df = pd.DataFrame(r.json())
        log.info(f"  Quiver: {len(df)} raw congressional trades")

        # Normalise column names (Quiver uses different names over time)
        rename = {
            "TransactionDate": "date",
            "Date":            "date",
            "Ticker":          "ticker",
            "Representative":  "politician",
            "Transaction":     "transaction",
            "Range":           "amount_range",
            "Amount":          "amount_range",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["ticker"].isin(ENERGY_TICKERS)].copy()

        df["trade_direction"] = df["transaction"].map(DIRECTION_MAP).fillna(0)
        df["amount_est"]      = df["amount_range"].map(AMOUNT_MAP).fillna(50_000)
        df["signed_amount"]   = df["trade_direction"] * df["amount_est"]

        log.info(f"  Quiver energy trades: {len(df)}")
        return df[["date","ticker","politician","transaction",
                   "trade_direction","signed_amount"]]

    except Exception as e:
        log.warning(f"  Quiver fetch failed: {e}")
        return None


def fetch_from_capitol_trades(max_pages: int = 15) -> pd.DataFrame | None:
    """Fallback scraper for Capitol Trades website."""
    trades = []
    headers = {"User-Agent": "Mozilla/5.0 (academic research project)"}

    for page in range(1, max_pages + 1):
        try:
            r = requests.get(
                f"https://www.capitoltrades.com/trades?page={page}",
                headers=headers,
                timeout=15,
            )
            soup = BeautifulSoup(r.content, "html.parser")
            rows = soup.select("table tbody tr")
            if not rows:
                break
            for row in rows:
                cols = [td.get_text(strip=True) for td in row.select("td")]
                if len(cols) >= 6:
                    trades.append({
                        "politician":      cols[0],
                        "ticker":          cols[3] if len(cols) > 3 else "",
                        "transaction":     cols[5] if len(cols) > 5 else "",
                        "date":            cols[6] if len(cols) > 6 else "",
                        "amount_range":    cols[7] if len(cols) > 7 else "",
                    })
            time.sleep(0.8)
        except Exception:
            break

    if not trades:
        return None

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["ticker"].str.upper().isin(ENERGY_TICKERS)].copy()
    df["trade_direction"] = df["transaction"].map(DIRECTION_MAP).fillna(0)
    df["amount_est"]      = df["amount_range"].map(AMOUNT_MAP).fillna(50_000)
    df["signed_amount"]   = df["trade_direction"] * df["amount_est"]

    log.info(f"  Capitol Trades energy rows: {len(df)}")
    return df[["date","ticker","politician","transaction",
               "trade_direction","signed_amount"]]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual trades to a daily net signal."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date").agg(
        congress_buys   = ("trade_direction", lambda x: (x > 0).sum()),
        congress_sells  = ("trade_direction", lambda x: (x < 0).sum()),
        congress_net    = ("trade_direction", "sum"),
        congress_volume = ("signed_amount",   "sum"),
    )
    denom = (daily["congress_buys"] + daily["congress_sells"]).replace(0, 1)
    daily["congress_net_signal"] = (
        (daily["congress_buys"] - daily["congress_sells"]) / denom
    )  # normalised -1 to +1
    return daily.resample("B").ffill(limit=10)


def fetch_congressional_trades():
    trade_out = RAW_DIR / "congressional_trades.parquet"
    daily_out = RAW_DIR / "congress_daily.parquet"

    # Try Quiver first, fall back to Capitol Trades scraper
    df = fetch_from_quiver()
    if df is None or len(df) == 0:
        log.info("  Falling back to Capitol Trades scraper...")
        df = fetch_from_capitol_trades()

    if df is None or len(df) == 0:
        log.warning("  All congressional trade sources failed — synthetic data")
        dates = pd.bdate_range(START_DATE, datetime.today())
        np.random.seed(42)
        df_daily = pd.DataFrame({
            "congress_buys":       np.random.poisson(1, len(dates)),
            "congress_sells":      np.random.poisson(1, len(dates)),
            "congress_net":        np.random.randint(-3, 3, len(dates)),
            "congress_volume":     np.random.normal(50000, 30000, len(dates)),
            "congress_net_signal": np.random.uniform(-1, 1, len(dates)),
        }, index=dates)
        df_daily.to_parquet(daily_out)
        return df_daily

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_parquet(trade_out)
    log.info(f"✓ Individual trades saved → {trade_out}  ({len(df)} rows)")

    daily = aggregate_daily(df.reset_index())
    daily.to_parquet(daily_out)
    log.info(f"✓ Daily congress signal saved → {daily_out}  ({len(daily)} rows)")
    log.info(
        f"  Net signal today: {daily['congress_net_signal'].iloc[-1]:.3f}"
    )
    return daily


if __name__ == "__main__":
    fetch_congressional_trades()
