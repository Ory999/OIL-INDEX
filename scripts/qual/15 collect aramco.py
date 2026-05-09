"""
Script 15 — Collect Saudi Aramco News via Google News RSS
Source: Google News RSS — free, public, no authentication required
Outputs: data/raw/aramco_corpus.parquet

Saudi Aramco is the world's largest oil producer. Their statements
on production, pricing, and market outlook often signal OPEC decisions
before official announcement — highest single-company market impact.

Collection method: Google News RSS (not aramco.com direct scraping)
aramco.com blocks automated requests. Google News RSS provides
oil-relevant press coverage without authentication or rate limiting.

Historic backfill strategy:
  2020–2026 articles pre-collected locally and stored in data/Historic/
  This script loads historic corpus first, then fetches only new articles
  not already present — avoiding redundant RSS requests on every run.
"""
import os, logging, time, re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import feedparser

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

RSS_URLS = [
    "https://news.google.com/rss/search?q=Saudi+Aramco+oil+production+OPEC&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Aramco+crude+oil+market&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Saudi+Arabia+oil+production+barrel&hl=en&gl=US&ceid=US:en",
]


def load_historic_corpus() -> pd.DataFrame | None:
    """
    Load pre-collected Aramco articles from data/Historic/aramco_corpus.parquet.
    Covers 2020–2026 (240 articles) collected during local backfill via Google News RSS.
    """
    historic_path = HISTORIC_DIR / "aramco_corpus.parquet"
    if not historic_path.exists():
        log.info("  No historic Aramco corpus found — fetching all from RSS")
        return None
    try:
        df = pd.read_parquet(historic_path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        log.info(f"✓ Loaded {len(df)} historic Aramco articles from {historic_path}")
        return df
    except Exception as e:
        log.warning(f"  Could not load historic Aramco corpus: {e}")
        return None


def fetch_rss_articles(historic_texts: set) -> list:
    """
    Fetch new Aramco articles from Google News RSS.
    Skips articles whose text is already in the historic corpus.
    """
    new_records = []

    for rss_url in RSS_URLS:
        try:
            feed = feedparser.parse(rss_url)
            log.info(f"  RSS feed: {len(feed.entries)} entries from {rss_url[:60]}...")

            for entry in feed.entries:
                title   = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")

                # Parse date
                try:
                    pub_date = pd.to_datetime(
                        entry.get("published", "")
                    ).tz_localize(None)
                except Exception:
                    pub_date = pd.Timestamp.now().floor("D")

                # Clean text
                text = re.sub(r'<[^>]+>', '', title + " " + summary)
                text = re.sub(r'\s+', ' ', text).strip()

                if len(text) < 100:
                    continue

                # Skip if already in historic corpus (text-based dedup)
                text_key = text[:200]
                if text_key in historic_texts:
                    continue

                new_records.append({
                    "date":       pub_date,
                    "source":     "ARAMCO",
                    "text":       text[:5000],
                    "char_count": len(text),
                })

            time.sleep(1.0)

        except Exception as e:
            log.warning(f"  RSS feed failed: {e}")

    return new_records


def collect_aramco_corpus():
    out = RAW_DIR / "aramco_corpus.parquet"

    # ── Load historic corpus ───────────────────────────────────────────────
    historic_df   = load_historic_corpus()
    historic_texts = set()
    if historic_df is not None and len(historic_df) > 0:
        historic_texts = set(
            historic_df["text"].str[:200].tolist()
        )
        log.info(f"  Historic text keys: {len(historic_texts)}")

    # ── Fetch new articles from RSS ────────────────────────────────────────
    new_records = fetch_rss_articles(historic_texts)

    if new_records:
        log.info(f"  Fetched {len(new_records)} new Aramco articles from RSS")
    else:
        log.info("  No new Aramco articles found — all covered by historic data")

    # ── Merge historic + new ───────────────────────────────────────────────
    frames = []
    if historic_df is not None and len(historic_df) > 0:
        frames.append(historic_df)
    if new_records:
        frames.append(pd.DataFrame(new_records))

    if not frames:
        log.warning("No Aramco articles available — creating placeholder")
        pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"), "source": "ARAMCO",
            "text": "Aramco placeholder", "char_count": 18,
        }]).to_parquet(out)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).sort_values("date").reset_index(drop=True)
    df.to_parquet(out)

    log.info(f"✓ Aramco corpus: {len(df)} articles → {out}")
    log.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  From historic: {len(historic_texts)}  |  New from RSS: {len(new_records)}")
    return df


if __name__ == "__main__":
    collect_aramco_corpus()
