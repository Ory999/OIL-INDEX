# Merge OPEC, EIA STEO, Aramco source corpora into one dated corpus.
# Excluded sources, IEA OMR paywalled, Energy Secretary no historical corpus.

import os, logging, re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

# IEA OMR paywalled, Energy Secretary not in backfill.
# Fed FOMC scored all-zero for oil impact, captured quantitatively via fred_macro.
SOURCES = [
    ("opec_corpus.parquet",     "OPEC_MOMR"),
    ("aramco_corpus.parquet",   "ARAMCO"),
    ("eia_steo_corpus.parquet", "EIA_STEO"),
]


def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_historic_corpus() -> pd.DataFrame | None:
    # Pre-built combined corpus of 695 documents 2007 to 2026.
    historic_path = HISTORIC_DIR / "combined_corpus.parquet"
    if not historic_path.exists():
        return None
    try:
        df = pd.read_parquet(historic_path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        log.info(f"✓ Loaded historic combined corpus: {len(df)} documents")
        return df
    except Exception as e:
        log.warning(f"  Could not load historic combined corpus: {e}")
        return None


def build_combined_corpus():
    out = RAW_DIR / "combined_corpus.parquet"

    # Load individual source corpora.
    dfs = []
    for filename, source_label in SOURCES:
        path = RAW_DIR / filename
        if not path.exists():
            log.warning(f"  Skipping {source_label} — {filename} not found")
            continue
        df = pd.read_parquet(path)
        if len(df) <= 1:   # skip empty and single-row placeholders
            log.warning(f"  Skipping {source_label} — placeholder only ({len(df)} rows)")
            continue
        if "text" not in df.columns:
            log.warning(f"  Skipping {source_label} — no text column")
            continue
        df = df[["date", "source", "text"]].copy()
        df["source"] = source_label
        df["date"]   = pd.to_datetime(df["date"]).dt.tz_localize(None)
        dfs.append(df)
        log.info(f"  ✓ {source_label}: {len(df)} documents")

    if not dfs:
        log.warning("No corpus files found — saving empty combined corpus")
        pd.DataFrame().to_parquet(out)
        return pd.DataFrame()

    corpus = pd.concat(dfs, ignore_index=True)
    corpus = corpus.dropna(subset=["date", "text"])
    corpus = corpus[corpus["text"].str.len() > 100]
    corpus["text_clean"] = corpus["text"].apply(clean_text)
    corpus["word_count"] = corpus["text_clean"].str.split().str.len()
    corpus["date"]       = pd.to_datetime(corpus["date"]).dt.tz_localize(None)
    corpus = corpus.sort_values("date").reset_index(drop=True)
    corpus.to_parquet(out)

    log.info(f"\n✓ Combined corpus: {len(corpus)} documents → {out}")
    log.info(f"  Sources: {corpus['source'].value_counts().to_dict()}")
    log.info(f"  Date range: {corpus['date'].min().date()} → {corpus['date'].max().date()}")
    log.info(f"  Avg word count: {corpus['word_count'].mean():.0f}")
    return corpus


if __name__ == "__main__":
    build_combined_corpus()
