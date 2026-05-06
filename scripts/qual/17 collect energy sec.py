"""
Script 17 — Collect US Energy Secretary Speeches
Source: energy.gov/newsroom/speeches — free, public domain
Outputs: data/raw/energy_sec_corpus.parquet

Direct policy signals on drilling permits, pipeline approvals, SPR
releases, and sanctions. Energy Secretary statements are among the most
explicit government signals of forthcoming US energy policy changes.
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

OIL_KEYWORDS = [
    "oil", "crude", "petroleum", "energy", "pipeline", "drilling",
    "production", "opec", "gasoline", "refinery", "strategic petroleum",
    "spr", "sanctions", "lng", "natural gas", "fossil", "supply"
]


def fetch_energy_gov_speeches(max_pages: int = 15) -> list:
    """Scrape energy.gov newsroom for energy secretary speeches."""
    links = []
    base  = "https://www.energy.gov"

    for page in range(0, max_pages):
        try:
            url = f"{base}/newsroom/speeches?page={page}"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.content, "html.parser")

            articles = soup.find_all("a", href=re.compile(r"/articles/"))
            if not articles:
                break

            for article in articles:
                href  = article.get("href", "")
                if not href.startswith("http"):
                    href = base + href
                title = article.get_text(strip=True)

                # Filter to energy-relevant speeches
                if any(kw in title.lower() for kw in OIL_KEYWORDS):
                    links.append({"url": href, "title": title})

            time.sleep(0.4)
        except Exception as e:
            log.debug(f"  Energy.gov page {page} failed: {e}")
            break

    log.info(f"  Found {len(links)} relevant Energy Secretary speeches")
    return links


def fetch_speech_text(url: str, title: str) -> dict | None:
    """Fetch full text from a single speech page."""
    try:
        r    = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")

        # Parse date
        pub_date = pd.Timestamp.now().floor("D")
        for selector in ["time", "span.date", "div.date", "p.date"]:
            el = soup.select_one(selector)
            if el:
                try:
                    pub_date = pd.to_datetime(
                        el.get("datetime") or el.get_text(strip=True)
                    )
                    break
                except Exception:
                    pass

        # Extract content
        content = (soup.find("div", class_=re.compile("field-items|article|body|content")) or
                   soup.find("main"))
        if not content:
            return None

        text = re.sub(r'\s+', ' ', content.get_text(separator=" ", strip=True))
        if len(text) < 150:
            return None

        return {
            "date":       pub_date,
            "source":     "ENERGY_SECRETARY",
            "title":      title,
            "text":       text[:5000],
            "char_count": len(text),
            "url":        url,
        }
    except Exception as e:
        log.debug(f"  Speech fetch failed {url}: {e}")
        return None


def collect_energy_sec_corpus():
    out     = RAW_DIR / "energy_sec_corpus.parquet"
    links   = fetch_energy_gov_speeches()
    records = []

    for link in links:
        record = fetch_speech_text(link["url"], link["title"])
        if record:
            records.append(record)
        time.sleep(0.3)

    if not records:
        log.warning("No Energy Secretary speeches collected — creating placeholder")
        pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"), "source": "ENERGY_SECRETARY",
            "title": "placeholder", "text": "Energy Secretary data placeholder",
            "char_count": 20, "url": ""
        }]).to_parquet(out)
        return pd.DataFrame()

    df = (pd.DataFrame(records)
            .drop_duplicates("url")
            .sort_values("date")
            .reset_index(drop=True))
    df.to_parquet(out)
    log.info(f"✓ Energy Secretary corpus: {len(df)} speeches → {out}")
    return df


if __name__ == "__main__":
    collect_energy_sec_corpus()
