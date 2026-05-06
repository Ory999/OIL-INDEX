"""
Script 13 — Collect OPEC Monthly Oil Market Reports
Source: opec.org — free, monthly PDFs since 2007
Outputs: data/raw/opec_corpus.parquet

Primary NLP input source. ScienceDirect (2024) found OPEC sentiment
dominates IEA sentiment with 2.40% certainty equivalent return gain.
"""
import os, logging, time, re, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

START_YEAR = int(os.getenv("START_DATE", "2007-01-01")[:4])
RAW_DIR    = Path(os.getenv("DATA_DIR", "data/raw"))
PDF_DIR    = RAW_DIR / "opec_pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = {
    1:"january", 2:"february", 3:"march", 4:"april",
    5:"may", 6:"june", 7:"july", 8:"august",
    9:"september", 10:"october", 11:"november", 12:"december"
}

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project; oil market analysis)"}


def download_opec_pdf(year: int, month: int) -> Path | None:
    fname = PDF_DIR / f"opec_momr_{year}_{month:02d}.pdf"
    if fname.exists():
        return fname

    month_name = MONTHS[month]
    urls_to_try = [
        f"https://www.opec.org/assets/assetdb/momr-{month_name}-{year}.pdf",
        f"https://www.opec.org/opec_web/static_files_project/media/downloads/publications/MOMR_{month_name.capitalize()}_{year}.pdf",
        f"https://www.opec.org/opec_web/static_files_project/media/downloads/publications/MOMR{year}{month:02d}.pdf",
    ]

    for url in urls_to_try:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25, stream=True)
            if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                log.info(f"  ✓ Downloaded: {fname.name}")
                return fname
        except Exception:
            continue

    return None


def extract_pdf_text(pdf_path: Path, max_pages: int = 10) -> str:
    """
    Extract text from first N pages — executive summary and outlook sections.
    Falls back gracefully if PDF is scanned/image-based.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = min(max_pages, len(pdf.pages))
            for page in pdf.pages[:pages]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        log.warning(f"  PDF extraction failed {pdf_path.name}: {e}")

    # Clean
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def collect_opec_corpus():
    out = RAW_DIR / "opec_corpus.parquet"
    records = []
    current_year  = datetime.now().year
    current_month = datetime.now().month

    for year in range(START_YEAR, current_year + 1):
        for month in range(1, 13):
            if year == current_year and month > current_month:
                break

            pdf_path = download_opec_pdf(year, month)
            if pdf_path is None:
                log.debug(f"  OPEC {year}-{month:02d}: not available")
                continue

            text = extract_pdf_text(pdf_path)
            if len(text) < 200:
                log.debug(f"  OPEC {year}-{month:02d}: text too short (possibly scanned)")
                continue

            records.append({
                "date":       pd.Timestamp(year=year, month=month, day=1),
                "source":     "OPEC_MOMR",
                "text":       text[:6000],
                "char_count": len(text),
                "year":       year,
                "month":      month,
            })
            time.sleep(0.4)

    if not records:
        log.warning("No OPEC reports collected — creating empty corpus")
        pd.DataFrame().to_parquet(out)
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_parquet(out)
    log.info(f"✓ OPEC corpus: {len(df)} reports → {out}")
    log.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    collect_opec_corpus()
