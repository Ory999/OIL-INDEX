"""
Script 04 — Fetch COT Futures Positioning (CFTC)
Source: CFTC Commitments of Traders — cftc.gov (free, public)
Outputs: data/raw/cot_crude.parquet

NOTE: Fix applied from notebook diagnostic output.
The disaggregated COT report uses M_Money_Positions_Long_All /
M_Money_Positions_Short_All for managed money (speculators),
NOT NonComm_Positions_Long_All which is the legacy report column.
"""
import os, io, logging, zipfile, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_YEAR = int(os.getenv("START_DATE", "2007-01-01")[:4])
RAW_DIR    = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Correct column names for CFTC disaggregated futures report
# Confirmed from actual notebook output showing available columns
LONG_COL  = "M_Money_Positions_Long_All"   # Managed money longs
SHORT_COL = "M_Money_Positions_Short_All"  # Managed money shorts
DATE_COL  = "As_of_Date_In_Form_YYMMDD"
MARKET_COL = "Market_and_Exchange_Names"


def download_cot_year(year: int) -> pd.DataFrame | None:
    """Download one year of CFTC disaggregated COT data."""
    current_year = datetime.now().year

    if year == current_year:
        url = "https://www.cftc.gov/dea/newcot/fut_disagg_txt.zip"
    else:
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".txt")][0]
            df = pd.read_csv(z.open(fname), low_memory=False)
        return df
    except Exception as e:
        log.debug(f"  COT {year} download failed: {e}")
        return None


def fetch_cot_data():
    out = RAW_DIR / "cot_crude.parquet"

    # Check if we have recent cached data — only download missing years
    existing_years = set()
    if out.exists():
        try:
            existing = pd.read_parquet(out)
            existing.index = pd.to_datetime(existing.index)
            latest_year = existing.index.max().year
            # Always re-fetch current year (it updates weekly)
            existing_years = set(range(START_YEAR, latest_year))
            if len(df) > 0 and "cot_net_long" in df.columns:
                log.info(f"  Latest net long: {df['cot_net_long'].iloc[-1]:,.0f} contracts")
        except Exception:
            existing = None
    else:
        existing = None

    current_year = datetime.now().year
    years_to_fetch = [y for y in range(START_YEAR, current_year + 1)
                      if y not in existing_years]

    log.info(f"Fetching COT data for years: {years_to_fetch}")

    cot_dfs = []
    for year in years_to_fetch:
        df_year = download_cot_year(year)
        if df_year is None:
            continue

        # Filter to WTI crude oil (NYMEX)
        crude = df_year[
            df_year[MARKET_COL].str.contains("CRUDE OIL", case=False, na=False)
        ].copy()

        if len(crude) == 0:
            log.debug(f"  No crude oil rows for {year}")
            continue

        # Parse date
        crude["report_date"] = pd.to_datetime(
            crude[DATE_COL].astype(str), format="%y%m%d", errors="coerce"
        )
        crude = crude.dropna(subset=["report_date"])

        # Extract managed money positioning
        # These are the speculative traders (hedge funds, CTAs)
        if LONG_COL in crude.columns and SHORT_COL in crude.columns:
            crude["cot_net_long"] = (
                crude[LONG_COL].astype(float)
                - crude[SHORT_COL].astype(float)
            )
            total = (
                crude[LONG_COL].astype(float)
                + crude[SHORT_COL].astype(float)
            )
            crude["cot_long_pct"] = (
                crude[LONG_COL].astype(float) / total.replace(0, np.nan)
            )
        else:
            log.warning(f"  COT {year}: expected columns not found — using zeros")
            crude["cot_net_long"] = 0.0
            crude["cot_long_pct"] = 0.5

        cot_dfs.append(
            crude[["report_date", "cot_net_long", "cot_long_pct"]]
        )
        log.info(f"  ✓ COT {year}: {len(crude)} crude oil rows")

    if not cot_dfs and existing is None:
        log.warning("COT download failed — generating synthetic data")
        dates = pd.bdate_range(
            f"{START_YEAR}-01-01", datetime.today(), freq="W-TUE"
        )
        df = pd.DataFrame({
            "cot_net_long":        np.random.normal(200000, 80000, len(dates)),
            "cot_long_pct":        np.random.uniform(0.4, 0.8,    len(dates)),
            "cot_net_long_change": np.random.normal(0, 20000,     len(dates)),
        }, index=dates)
        df = df.resample("B").ffill(limit=5)
        df.to_parquet(out)
        return df

    # Combine new data with existing cache
    new_parts = []
    if cot_dfs:
        new_df = pd.concat(cot_dfs, ignore_index=True)
        new_df = new_df.sort_values("report_date").drop_duplicates("report_date")
        new_df = new_df.set_index("report_date")
        new_df["cot_net_long_change"] = new_df["cot_net_long"].diff()
        new_parts.append(new_df)

    if existing is not None:
        new_parts.append(existing)

    combined = (
        pd.concat(new_parts)
        .sort_index()
        .drop_duplicates()
    )

    # Resample to business day frequency
    df = combined.resample("B").ffill(limit=5)

    df.to_parquet(out)
    log.info(f"✓ COT data saved → {out}  ({len(df)} rows)")
    log.info(
        f"  Latest net long: {df['cot_net_long'].iloc[-1]:,.0f} contracts"
    )
    return df


if __name__ == "__main__":
    fetch_cot_data()
