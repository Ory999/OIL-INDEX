"""
Script 18 — Build Combined Corpus
Merges all institutional text sources into one dated corpus.
Outputs: data/raw/combined_corpus.parquet
"""
import os, logging, re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("opec_corpus.parquet",       "OPEC_MOMR"),
    ("iea_corpus.parquet",        "IEA_OMR"),
    ("aramco_corpus.parquet",     "ARAMCO"),
    ("eia_steo_corpus.parquet",   "EIA_STEO"),
    ("energy_sec_corpus.parquet", "ENERGY_SECRETARY"),
]


def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_combined_corpus():
    out  = RAW_DIR / "combined_corpus.parquet"
    dfs  = []

    for filename, source_label in SOURCES:
        path = RAW_DIR / filename
        if not path.exists():
            log.warning(f"  Skipping {source_label} — {filename} not found")
            continue

        df = pd.read_parquet(path)
        if len(df) == 0:
            log.warning(f"  Skipping {source_label} — empty corpus")
            continue

        # Ensure required columns
        if "text" not in df.columns:
            log.warning(f"  Skipping {source_label} — no text column")
            continue

        df = df[["date", "source", "text"]].copy()
        df["source"] = source_label
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
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
    # Strip timezone info from all dates for consistent sorting
    corpus["date"] = pd.to_datetime(corpus["date"]).dt.tz_localize(None)
    corpus = corpus.sort_values("date").reset_index(drop=True)

    corpus.to_parquet(out)
    log.info(f"\n✓ Combined corpus: {len(corpus)} documents → {out}")
    log.info(f"  Sources: {corpus['source'].value_counts().to_dict()}")
    log.info(f"  Date range: {corpus['date'].min().date()} → {corpus['date'].max().date()}")
    log.info(f"  Avg word count: {corpus['word_count'].mean():.0f}")
    return corpus


if __name__ == "__main__":
    build_combined_corpus()
