"""
Script 14 — Collect IEA Oil Market Report Press Releases
Source: iea.org — free, publicly available summaries
Outputs: data/raw/iea_corpus.parquet

Demand-side counterpart to OPEC. Adds incremental predictive power
above OPEC alone per ScienceDirect (2024).
"""
import os, logging, time, re, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project; oil market analysis)"}
IEA_BASE = "https://www.iea.org"


def fetch_iea_report_links(max_pages: int = 8) -> list:
    """Scrape IEA OMR report listing pages for individual report URLs."""
    links = []
    for page in range(1, max_pages + 1):
        try:
            url = f"{IEA_BASE}/reports?type=Oil+Market+Report&page={page}"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.content, "html.parser")
            found = soup.find_all("a", href=re.compile(r"/reports/oil-market-report"))
            if not found:
                break
            for link in found:
                href  = IEA_BASE + link.get("href", "")
                title = link.get_text(strip=True)
                if href not in [l["url"] for l in links]:
                    links.append({"url": href, "title": title})
            time.sleep(0.5)
        except Exception as e:
            log.debug(f"  IEA page {page} failed: {e}")
            break

    log.info(f"  Found {len(links)} IEA report links")
    return links


def fetch_iea_report_text(url: str, title: str) -> dict | None:
    """Fetch text content from a single IEA report page."""
    try:
        r    = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Parse publication date from URL or page
        pub_date = pd.Timestamp.now().floor("D")
        date_match = re.search(r'(\d{4})-(\d{2})', url)
        if date_match:
            pub_date = pd.Timestamp(int(date_match.group(1)), int(date_match.group(2)), 1)
        else:
            # Try to find date in page
            for el in soup.find_all(["time", "span"], attrs={"datetime": True}):
                try:
                    pub_date = pd.to_datetime(el["datetime"])
                    break
                except Exception:
                    pass

        # Extract article body
        content_el = (soup.find("div", class_=re.compile("article|content|body|prose")) or
                      soup.find("article") or
                      soup.find("main"))
        if not content_el:
            return None

        text = re.sub(r'\s+', ' ', content_el.get_text(separator=" ", strip=True))
        if len(text) < 100:
            return None

        return {
            "date":       pub_date,
            "source":     "IEA_OMR",
            "title":      title,
            "text":       text[:6000],
            "char_count": len(text),
            "url":        url,
        }
    except Exception as e:
        log.debug(f"  IEA report fetch failed {url}: {e}")
        return None


def collect_iea_corpus():
    out = RAW_DIR / "iea_corpus.parquet"

    links   = fetch_iea_report_links()
    records = []

    for link in links:
        record = fetch_iea_report_text(link["url"], link["title"])
        if record:
            records.append(record)
        time.sleep(0.3)

    if not records:
        log.warning("No IEA reports collected — creating placeholder")
        pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"), "source": "IEA_OMR",
            "title": "placeholder", "text": "IEA data placeholder",
            "char_count": 20, "url": ""
        }]).to_parquet(out)
        return pd.DataFrame()

    df = (pd.DataFrame(records)
            .drop_duplicates("url")
            .sort_values("date")
            .reset_index(drop=True))
    df.to_parquet(out)
    log.info(f"✓ IEA corpus: {len(df)} reports → {out}")
    return df


if __name__ == "__main__":
    collect_iea_corpus()
