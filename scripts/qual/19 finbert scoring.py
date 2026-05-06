"""
Script 19 — FinBERT Baseline Scoring
Model: ProsusAI/finbert — BERT pre-trained on financial text
Outputs: data/raw/finbert_scores.parquet

Produces positive/negative/neutral probabilities per document.
Serves as:
  1. Baseline for EMH test — does raw sentiment alone predict prices?
  2. Confidence-gated fallback when LLM confidence < threshold
  3. Cross-validation baseline against LLM scores
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

RAW_DIR     = Path(os.getenv("DATA_DIR",  "data/raw"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "data/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINBERT_MODEL  = "ProsusAI/finbert"
BATCH_SIZE     = 16
MAX_TEXT_LEN   = 512


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


def score_batch(model, texts: list) -> list:
    """Score a batch of texts. Returns list of score dicts."""
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
                    "finbert_score":      round(pos - neg, 6),   # -1 to +1
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

    log.info(f"Scoring {len(corpus)} documents with FinBERT...")
    model  = load_finbert()
    scores = score_batch(model, corpus["text_clean"].fillna("").tolist())
    scores_df = pd.DataFrame(scores, index=corpus.index)

    result = pd.concat([corpus[["date", "source", "text_clean"]], scores_df], axis=1)
    result.to_parquet(out)

    log.info(f"✓ FinBERT scores saved → {out}")
    log.info(f"  Score range: {result['finbert_score'].min():.3f} to {result['finbert_score'].max():.3f}")
    log.info(f"  Mean score by source:")
    for src, grp in result.groupby("source"):
        log.info(f"    {src:25s}: {grp['finbert_score'].mean():+.4f}")

    return result


if __name__ == "__main__":
    run_finbert_scoring()
