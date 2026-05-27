# Collect EIA Short-Term Energy Outlook narratives from eia.gov.
# Monthly forward-looking report, captures EIA market interpretation and forecasts.
# URL pattern eia.gov/outlooks/steo/archives/{Mon}{YY}.pdf, current month uses live PDF.
# Historic 2007 to 2026 corpus loaded from data/Historic, only new reports downloaded.

import os, logging, time, re, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_YEAR   = int(os.getenv("START_DATE", "2007-01-01")[:4])
RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
STEO_DIR     = RAW_DIR / "eia_steo"
STEO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS     = {"User-Agent": "Mozilla/5.0 (academic research project; oil market analysis)"}
MONTH_CAPS  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def load_historic_corpus() -> pd.DataFrame | None:
    historic_path = HISTORIC_DIR / "eia_steo_corpus.parquet"
    if not historic_path.exists():
        log.info("  No historic EIA STEO corpus found — downloading all reports")
        return None
    try:
        df = pd.read_parquet(historic_path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        log.info(f"✓ Loaded {len(df)} historic EIA STEO reports from {historic_path}")
        return df
    except Exception as e:
        log.warning(f"  Could not load historic EIA STEO corpus: {e}")
        return None


def download_steo_pdf(year: int, month: int) -> Path | None:
    # Current month uses live PDF, past months use archive.
    mon   = MONTH_CAPS[month - 1]
    yr2   = str(year)[2:]
    fname = STEO_DIR / f"eia_steo_{year}_{month:02d}.pdf"

    if fname.exists():
        return fname

    now = datetime.now()
    if year == now.year and month == now.month:
        url = "https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf"
    else:
        url = f"https://www.eia.gov/outlooks/steo/archives/{mon}{yr2}.pdf"

    try:
        r = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        if r.status_code != 200:
            return None
        content_len = int(r.headers.get("content-length", 0))
        if content_len < 10000:
            return None
        with open(fname, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        log.info(f"  ✓ Downloaded: {mon}{yr2}.pdf ({content_len/1024:.0f}KB)")
        return fname
    except Exception as e:
        log.debug(f"  EIA STEO {year}-{month:02d} failed: {e}")
        return None


def extract_steo_text(pdf_path: Path, max_pages: int = 15) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        log.warning(f"  PDF extraction failed {pdf_path.name}: {e}")
    return re.sub(r'\s+', ' ', text).strip()


def collect_eia_steo_corpus():
    out = RAW_DIR / "eia_steo_corpus.parquet"

    # Load historic corpus 2007 to 2026.
    historic_df   = load_historic_corpus()
    historic_keys = set()
    if historic_df is not None and len(historic_df) > 0:
        historic_keys = set(
            zip(historic_df["date"].dt.year.astype(int),
                historic_df["date"].dt.month.astype(int))
        )
        log.info(f"  Historic coverage: {len(historic_keys)} (year, month) pairs")

    # Download only reports not in historic.
    now         = datetime.now()
    new_records = []

    for year in range(START_YEAR, now.year + 1):
        for month_idx, mon in enumerate(MONTH_CAPS, 1):
            if year == now.year and month_idx > now.month:
                break
            if (year, month_idx) in historic_keys:
                continue   # already have this report

            pdf_path = download_steo_pdf(year, month_idx)
            if pdf_path is None:
                continue

            text = extract_steo_text(pdf_path)
            if len(text) < 300:
                continue

            new_records.append({
                "date":       pd.Timestamp(year=year, month=month_idx, day=1),
                "source":     "EIA_STEO",
                "text":       text[:8000],
                "char_count": len(text),
            })
            time.sleep(0.2)

    # Merge historic and new records.
    frames = []
    if historic_df is not None and len(historic_df) > 0:
        frames.append(historic_df)
    if new_records:
        log.info(f"  Downloaded {len(new_records)} new EIA STEO reports")
        frames.append(pd.DataFrame(new_records))
    else:
        log.info("  No new EIA STEO reports to download — all covered by historic data")

    if not frames:
        log.warning("No EIA STEO reports available — creating placeholder")
        pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"), "source": "EIA_STEO",
            "text": "EIA STEO placeholder", "char_count": 20,
        }]).to_parquet(out)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").drop_duplicates(
        subset=["date", "source"]
    ).reset_index(drop=True)
    df.to_parquet(out)

    log.info(f"✓ EIA STEO corpus: {len(df)} reports → {out}")
    log.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  From historic: {len(historic_keys)}  |  New downloads: {len(new_records)}")
    return df


if __name__ == "__main__":
    collect_eia_steo_corpus()
