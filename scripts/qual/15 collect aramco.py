"""
Script 15 — Collect Saudi Aramco Press Releases
Source: aramco.com/en/news-media/news — free, public
Outputs: data/raw/aramco_corpus.parquet

Saudi Aramco is the world's largest oil producer. Their statements
on production, pricing, and market outlook often signal OPEC decisions
before official announcement — highest single-company market impact.
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
    "oil", "crude", "production", "barrel", "opec", "output", "supply",
    "demand", "price", "market", "energy", "upstream", "downstream",
    "refinery", "export", "capacity", "reserves"
]


def fetch_aramco_news(max_pages: int = 10) -> list:
    records = []
    base = "https://www.aramco.com"

    for page in range(1, max_pages + 1):
        try:
            url = f"{base}/en/news-media/news?page={page}"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.content, "html.parser")

            # Find article links
            articles = soup.find_all("a", href=re.compile(r"/en/news-media/news/"))
            if not articles:
                break

            for article in articles:
                href  = base + article.get("href", "")
                title = article.get_text(strip=True)

                # Only fetch oil-relevant articles
                if not any(kw in title.lower() for kw in OIL_KEYWORDS):
                    continue

                try:
                    ar = requests.get(href, headers=HEADERS, timeout=12)
                    asoup = BeautifulSoup(ar.content, "html.parser")

                    # Parse date
                    pub_date = pd.Timestamp.now().floor("D")
                    for el in asoup.find_all(["time", "span"],
                                              attrs={"datetime": True}):
                        try:
                            pub_date = pd.to_datetime(el["datetime"])
                            break
                        except Exception:
                            pass

                    # Extract text
                    content = (asoup.find("div", class_=re.compile("article|content|body")) or
                               asoup.find("main"))
                    if content:
                        text = re.sub(r'\s+', ' ',
                                      content.get_text(separator=" ", strip=True))
                        if len(text) > 150:
                            records.append({
                                "date":       pub_date,
                                "source":     "ARAMCO",
                                "title":      title,
                                "text":       text[:5000],
                                "char_count": len(text),
                                "url":        href,
                            })

                    time.sleep(0.3)

                except Exception as e:
                    log.debug(f"  Aramco article failed: {e}")

            time.sleep(0.5)

        except Exception as e:
            log.debug(f"  Aramco page {page} failed: {e}")
            break

    return records


def collect_aramco_corpus():
    out     = RAW_DIR / "aramco_corpus.parquet"
    records = fetch_aramco_news()

    if not records:
        log.warning("No Aramco press releases collected — creating placeholder")
        pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"), "source": "ARAMCO",
            "title": "placeholder", "text": "Aramco data placeholder",
            "char_count": 20, "url": ""
        }]).to_parquet(out)
        return pd.DataFrame()

    df = (pd.DataFrame(records)
            .drop_duplicates("url")
            .sort_values("date")
            .reset_index(drop=True))
    df.to_parquet(out)
    log.info(f"✓ Aramco corpus: {len(df)} articles → {out}")
    return df


if __name__ == "__main__":
    collect_aramco_corpus()
