"""
Script 16 — Collect EIA Short-Term Energy Outlook (STEO) Narratives
Source: eia.gov/outlooks/steo/archives/ — free, official US government
Outputs: data/raw/eia_steo_corpus.parquet

Monthly narrative report with forward-looking language on supply/demand
balance. Distinct from the raw inventory data — captures EIA's explicit
market interpretation and forecasts that move trader expectations.
"""
import os, logging, time, re, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_YEAR = int(os.getenv("START_DATE", "2007-01-01")[:4])
RAW_DIR    = Path(os.getenv("DATA_DIR", "data/raw"))
STEO_DIR   = RAW_DIR / "eia_steo"
STEO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project; oil market analysis)"}
EIA_STEO_BASE = "https://www.eia.gov/outlooks/steo/archives"


def fetch_steo_pdf_links() -> list:
    """Scrape EIA STEO archives page for all monthly PDF links."""
    links = []
    try:
        r    = requests.get(EIA_STEO_BASE, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.content, "html.parser")

        for a in soup.find_all("a", href=re.compile(r"\.pdf", re.IGNORECASE)):
            href = a.get("href", "")
            if not href.startswith("http"):
                href = "https://www.eia.gov" + href
            text = a.get_text(strip=True)

            # Parse year from link
            year_match = re.search(r'(200[7-9]|20[12]\d)', href)
            if year_match:
                year = int(year_match.group(1))
                if year >= START_YEAR:
                    links.append({"url": href, "title": text, "year": year})

        log.info(f"  Found {len(links)} EIA STEO PDF links")
    except Exception as e:
        log.warning(f"  EIA STEO archive scrape failed: {e}")

    return links


def download_and_extract_steo(url: str, year: int) -> str:
    """Download STEO PDF and extract narrative text."""
    fname = STEO_DIR / f"eia_steo_{Path(url).stem}.pdf"

    if not fname.exists():
        try:
            r = requests.get(url, headers=HEADERS, timeout=25, stream=True)
            if r.status_code == 200:
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            else:
                return ""
        except Exception:
            return ""

    text = ""
    try:
        with pdfplumber.open(fname) as pdf:
            # STEO: first 12 pages contain narrative summary and oil market section
            for page in pdf.pages[:12]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception:
        return ""

    return re.sub(r'\s+', ' ', text).strip()


def collect_eia_steo_corpus():
    out   = RAW_DIR / "eia_steo_corpus.parquet"
    links = fetch_steo_pdf_links()

    records = []
    for link in links:
        text = download_and_extract_steo(link["url"], link["year"])
        if len(text) < 200:
            continue

        # Try to parse month from URL
        month_match = re.search(
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
            link["url"].lower()
        )
        month_map = {
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
        }
        month = month_map.get(month_match.group(1), 1) if month_match else 1

        records.append({
            "date":       pd.Timestamp(year=link["year"], month=month, day=1),
            "source":     "EIA_STEO",
            "title":      link["title"],
            "text":       text[:6000],
            "char_count": len(text),
            "url":        link["url"],
        })
        time.sleep(0.3)

    if not records:
        log.warning("No EIA STEO reports collected — creating placeholder")
        pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"), "source": "EIA_STEO",
            "title": "placeholder", "text": "EIA STEO placeholder",
            "char_count": 20, "url": ""
        }]).to_parquet(out)
        return pd.DataFrame()

    df = (pd.DataFrame(records)
            .drop_duplicates("url")
            .sort_values("date")
            .reset_index(drop=True))
    df.to_parquet(out)
    log.info(f"✓ EIA STEO corpus: {len(df)} reports → {out}")
    return df


if __name__ == "__main__":
    collect_eia_steo_corpus()
