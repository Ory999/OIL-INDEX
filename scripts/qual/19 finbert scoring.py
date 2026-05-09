"""
Script 19 — FinBERT Baseline Sentiment Scoring
Model: ProsusAI/finbert — BERT pre-trained on financial text
Outputs: data/raw/finbert_scores.parquet

Produces positive/negative/neutral probabilities per document.
Serves as:
  1. Baseline for EMH test — does raw sentiment alone predict prices?
  2. Cross-validation baseline against LLM scores (r=0.13 correlation
     confirms LLM captures domain-specific oil signals beyond generic
     financial sentiment)

Historic backfill strategy:
  695 documents pre-scored locally and stored in data/Historic/
  This script loads historic scores first, then scores only new documents
  not already present — avoiding full GPU rerun on every daily pipeline run.
"""
import os, logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 16
MAX_TEXT_LEN  = 512


def load_historic_scores() -> pd.DataFrame | None:
    """
    Load pre-scored FinBERT results from data/Historic/finbert_scores.parquet.
    Covers 695 documents (2007–2026) scored during local backfill.
    Matched by (date, source) to avoid index alignment issues.
    """
    historic_path = HISTORIC_DIR / "finbert_scores.parquet"
    if not historic_path.exists():
        log.info("  No historic FinBERT scores found — scoring all documents")
        return None
    try:
        df = pd.read_parquet(historic_path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        log.info(f"✓ Loaded {len(df)} historic FinBERT scores from {historic_path}")
        return df
    except Exception as e:
        log.warning(f"  Could not load historic FinBERT scores: {e}")
        return None


def load_finbert():
    device = 0 if torch.cuda.is_available() else -1
    log.info(f"Loading FinBERT on {'GPU' if device == 0 else 'CPU'}...")
    model = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        return_all_scores=True,
        truncation=True,
        max_length=MAX_TEXT_LEN,
        device=device,
    )
    log.info("✓ FinBERT loaded")
    return model


def score_texts(model, texts: list) -> list:
    """Score a list of texts. Returns list of score dicts."""
    results = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT"):
        batch = [str(t)[:MAX_TEXT_LEN] for t in texts[i:i + BATCH_SIZE]]
        try:
            batch_results = model(batch)
            for res in batch_results:
                scores = {s["label"].lower(): s["score"] for s in res}
                pos = scores.get("positive", 0.0)
                neg = scores.get("negative", 0.0)
                neu = scores.get("neutral",  0.0)
                results.append({
                    "finbert_pos":        round(pos, 6),
                    "finbert_neg":        round(neg, 6),
                    "finbert_neu":        round(neu, 6),
                    "finbert_score":      round(pos - neg, 6),
                    "finbert_confidence": round(max(pos, neg, neu), 6),
                })
        except Exception as e:
            log.warning(f"  FinBERT batch {i} failed: {e}")
            results.extend([{
                "finbert_pos": 0.0, "finbert_neg": 0.0, "finbert_neu": 1.0,
                "finbert_score": 0.0, "finbert_confidence": 0.0,
            }] * len(batch))
    return results


def run_finbert_scoring():
    corpus_path = RAW_DIR / "combined_corpus.parquet"
    out         = RAW_DIR / "finbert_scores.parquet"

    if not corpus_path.exists():
        log.warning("combined_corpus.parquet not found — skipping FinBERT scoring")
        return pd.DataFrame()

    corpus = pd.read_parquet(corpus_path)
    if len(corpus) == 0:
        log.warning("Corpus is empty — skipping FinBERT scoring")
        return pd.DataFrame()

    corpus["date"] = pd.to_datetime(corpus["date"]).dt.tz_localize(None).dt.normalize()

    # ── Load historic FinBERT scores ───────────────────────────────────────
    historic_df = load_historic_scores()
    hist_lookup = {}
    if historic_df is not None:
        for _, r in historic_df.iterrows():
            key = (str(r["date"].date()), str(r.get("source", "")))
            hist_lookup[key] = r

    # ── Identify new documents not in historic scores ──────────────────────
    new_docs = [
        (idx, row) for idx, row in corpus.iterrows()
        if (str(row["date"].date()), str(row.get("source", ""))) not in hist_lookup
    ]

    log.info(f"\n  Corpus total:       {len(corpus)}")
    log.info(f"  From historic data: {len(corpus) - len(new_docs)}")
    log.info(f"  Need FinBERT:       {len(new_docs)}")

    # ── Score new documents ────────────────────────────────────────────────
    new_scores = {}
    if new_docs:
        model = load_finbert()
        texts = [str(row.get("text_clean", "")) for _, row in new_docs]
        scored = score_texts(model, texts)
        for (idx, _), score in zip(new_docs, scored):
            new_scores[idx] = score
        log.info(f"  ✓ Scored {len(new_docs)} new documents with FinBERT")
    else:
        log.info("  No new documents to score — all covered by historic data")

    # ── Build full output: historic + new ──────────────────────────────────
    rows = []
    for idx, row in corpus.iterrows():
        key = (str(row["date"].date()), str(row.get("source", "")))
        if idx in new_scores:
            score_row = new_scores[idx]
        elif key in hist_lookup:
            r = hist_lookup[key]
            score_row = {
                "finbert_pos":        r.get("finbert_pos",        0.0),
                "finbert_neg":        r.get("finbert_neg",        0.0),
                "finbert_neu":        r.get("finbert_neu",        1.0),
                "finbert_score":      r.get("finbert_score",      0.0),
                "finbert_confidence": r.get("finbert_confidence", 0.0),
            }
        else:
            score_row = {
                "finbert_pos": 0.0, "finbert_neg": 0.0, "finbert_neu": 1.0,
                "finbert_score": 0.0, "finbert_confidence": 0.0,
            }
        rows.append({**row.to_dict(), **score_row})

    result = pd.DataFrame(rows).reset_index(drop=True)
    result.to_parquet(out)

    log.info(f"\n✓ FinBERT scores saved → {out}")
    log.info(f"  Total documents: {len(result)}")
    log.info(f"  Score range: {result['finbert_score'].min():.3f} "
             f"to {result['finbert_score'].max():.3f}")
    log.info(f"  Mean score by source:")
    for src, grp in result.groupby("source"):
        log.info(f"    {src:25s}: {grp['finbert_score'].mean():+.4f}")

    return result


if __name__ == "__main__":
    run_finbert_scoring()
