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
"""
import os, logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Manual topic labels — update after first run based on actual keywords
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
    log.info(f"Fitting BERTopic on {len(texts)} documents...")

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=10,
            min_topic_size=5,
            verbose=False,
        )

        topics, probs = topic_model.fit_transform(texts)
        corpus["topic_id"]   = topics
        corpus["topic_prob"] = [p.max() if hasattr(p, "max") else float(p)
                                 for p in probs]
        corpus["topic_label"] = corpus["topic_id"].map(TOPIC_LABELS).fillna("OTHER")

        # Print discovered topics for manual label review
        topic_info = topic_model.get_topic_info()
        log.info("\n  Discovered Topics (review and update TOPIC_LABELS):")
        for _, row in topic_info.iterrows():
            if row["Topic"] != -1:
                keywords = topic_model.get_topic(row["Topic"])
                kw_str   = ", ".join([w for w, _ in keywords[:6]])
                log.info(f"    Topic {row['Topic']:2d}: {kw_str}")

        # Save model for drift monitoring
        model_path = Path(os.getenv("RESULTS_DIR", "data/results")) / "bertopic_model"
        model_path.mkdir(exist_ok=True)
        try:
            topic_model.save(str(model_path), serialization="safetensors",
                             save_ctfidf=True)
            log.info(f"  BERTopic model saved → {model_path}")
        except Exception as e:
            log.warning(f"  Could not save BERTopic model: {e}")

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
