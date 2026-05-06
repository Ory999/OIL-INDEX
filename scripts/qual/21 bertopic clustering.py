"""
Script 21 — BERTopic Topic Clustering
Discovers semantic topics in the institutional corpus.
Outputs: data/raw/corpus_with_topics.parquet

Theoretical link (Behavioral Finance):
Different rhetoric types cause different magnitudes of market overreaction.
Per-topic Granger tests reveal WHICH types of institutional speech have
the strongest predictive power — this is a core thesis contribution.

Expected topics:
  - OPEC_SUPPLY:       production quotas, spare capacity, output
  - SANCTIONS:         Russia, Iran, Venezuela — supply restriction
  - DEMAND_OUTLOOK:    consumption, GDP growth, emerging markets
  - PRICE_FORECAST:    price projections, market balance
  - INVENTORY:         crude stocks, storage, build/draw
  - GEOPOLITICAL:      conflict, pipeline, trade routes
  - GREEN_TRANSITION:  EV, renewables, energy transition

IMPORTANT — Model fitting strategy:
  First run (no saved model): fits on full corpus, saves model
  Subsequent runs:            loads saved model, predicts only
  This ensures topic structure is stable across daily runs.
  Refit only when corpus grows significantly (e.g. after backfill).
"""
import os, logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR     = Path(os.getenv("DATA_DIR",     "data/raw"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR",  "data/results"))
RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "bertopic_model"

# Manual topic labels — update after first run based on actual keywords
# logged below under "Discovered Topics"
TOPIC_LABELS = {
    -1: "OUTLIER",
    0:  "OPEC_SUPPLY",
    1:  "SANCTIONS_GEOPOLITICAL",
    2:  "DEMAND_OUTLOOK",
    3:  "PRICE_FORECAST",
    4:  "INVENTORY_BALANCE",
    5:  "PRODUCTION_CAPACITY",
    6:  "GREEN_TRANSITION",
    7:  "MARKET_VOLATILITY",
}

# Minimum corpus size to fit a meaningful model
# Below this threshold topic quality will be poor
MIN_DOCS_FOR_FIT = 100


def run_bertopic_clustering():
    llm_path = RAW_DIR / "llm_scores.parquet"
    out      = RAW_DIR / "corpus_with_topics.parquet"

    if not llm_path.exists():
        log.warning("llm_scores.parquet not found — run 20 llm scoring.py first")
        return pd.DataFrame()

    corpus = pd.read_parquet(llm_path)
    if len(corpus) == 0:
        log.warning("LLM scores empty — skipping BERTopic")
        return pd.DataFrame()

    texts = corpus["text_clean"].fillna("").tolist()

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # ── Load existing model or fit new one ────────────────────────────
        if MODEL_PATH.exists():
            log.info(f"Loading existing BERTopic model from {MODEL_PATH}...")
            try:
                topic_model = BERTopic.load(
                    str(MODEL_PATH),
                    embedding_model=embedding_model
                )
                topics, probs = topic_model.transform(texts)
                log.info(f"✓ Topics predicted using existing model ({len(texts)} documents)")

            except Exception as e:
                log.warning(f"  Could not load saved model ({e}) — refitting")
                MODEL_PATH_STR = str(MODEL_PATH)
                import shutil
                shutil.rmtree(MODEL_PATH_STR, ignore_errors=True)
                topic_model = None

        else:
            topic_model = None

        # ── Fit new model if needed ────────────────────────────────────────
        if topic_model is None:
            if len(texts) < MIN_DOCS_FOR_FIT:
                log.warning(
                    f"  Only {len(texts)} documents — below minimum {MIN_DOCS_FOR_FIT} "
                    f"for reliable topic fitting. Run historical backfill first."
                )
                log.warning("  Assigning placeholder topics until backfill is complete.")
                corpus["topic_id"]    = 0
                corpus["topic_prob"]  = 0.5
                corpus["topic_label"] = "PENDING_BACKFILL"
                corpus.to_parquet(out)
                return corpus

            log.info(f"Fitting BERTopic on {len(texts)} documents (first run)...")
            topic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=10,
                min_topic_size=5,
                verbose=False,
            )
            topics, probs = topic_model.fit_transform(texts)
            log.info(f"✓ BERTopic fitted on {len(texts)} documents")

            # Log discovered topics for manual TOPIC_LABELS review
            topic_info = topic_model.get_topic_info()
            log.info("\n  ── Discovered Topics ──────────────────────────────────")
            log.info("  Review keywords below and update TOPIC_LABELS in this file:")
            for _, row in topic_info.iterrows():
                if row["Topic"] != -1:
                    keywords = topic_model.get_topic(row["Topic"])
                    kw_str   = ", ".join([w for w, _ in keywords[:8]])
                    log.info(f"    Topic {row['Topic']:2d} ({row['Count']:3d} docs): {kw_str}")
            log.info("  ───────────────────────────────────────────────────────")

            # Save model for all future daily runs
            MODEL_PATH.mkdir(parents=True, exist_ok=True)
            try:
                topic_model.save(
                    str(MODEL_PATH),
                    serialization="safetensors",
                    save_ctfidf=True
                )
                log.info(f"  Model saved → {MODEL_PATH}")
            except Exception as e:
                log.warning(f"  Could not save model: {e}")

        # ── Assign topic labels ────────────────────────────────────────────
        corpus["topic_id"]   = topics
        corpus["topic_prob"] = [
            p.max() if hasattr(p, "max") else float(p) for p in probs
        ]
        corpus["topic_label"] = (
            corpus["topic_id"].map(TOPIC_LABELS).fillna("OTHER")
        )

    except ImportError:
        log.warning("  BERTopic not installed — assigning placeholder topics")
        corpus["topic_id"]    = 0
        corpus["topic_prob"]  = 0.5
        corpus["topic_label"] = "UNKNOWN"

    corpus.to_parquet(out)
    log.info(f"\n✓ Corpus with topics saved → {out}")
    log.info(f"  Topic distribution:")
    for label, count in corpus["topic_label"].value_counts().items():
        log.info(f"    {label:30s}: {count}")

    return corpus


if __name__ == "__main__":
    run_bertopic_clustering()
