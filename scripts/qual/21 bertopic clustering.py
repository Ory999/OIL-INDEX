"""
Script 21 — BERTopic Topic Clustering
Discovers semantic topics in the institutional corpus.
Outputs: data/raw/corpus_with_topics.parquet

Theoretical link (Behavioral Finance):
Different rhetoric types cause different magnitudes of market overreaction.
Per-topic Granger tests reveal WHICH types of institutional speech have
the strongest predictive power — this is a core thesis contribution.

Historic backfill strategy:
  Topic assignments for 2007–2026 are loaded from data/Historic/
  The pre-fitted model (data/Historic/Bertopic/) is used for new documents.
  This avoids refitting on every daily run and ensures stable topic structure.

Topic labels (verified against data/Historic/Bertopic/topics.json):
  0  OPEC_SUPPLY       — production, million, oil
  1  SAUDI_OIL_NEWS    — saudi, arabia, reuters, iran, output
  2  DEMAND_OUTLOOK    — mb, opec, market, demand
  3  NOISE_EMAIL       — cid, email, org  [FILTERED before momentum]
  4  IEA_NARRATIVE     — mb, demand, growth, market
  5  GEOPOLITICAL      — uae, exit, opec, cartel
  6  ARAMCO_FINANCIAL  — aramco, profit, dividends
  7  HEATING_DEMAND    — winter, heating, percent
  8  NOISE_SOCIAL      — capitalist, venezuela, visual  [FILTERED before momentum]
"""
import os, logging, shutil
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORIC_MODEL_PATH  = HISTORIC_DIR / "Bertopic"
HISTORIC_TOPICS_PATH = HISTORIC_DIR / "corpus_with_topics.parquet"
MODEL_PATH           = RESULTS_DIR / "bertopic_model"

TOPIC_LABELS = {
    -1: "OUTLIER",
    0:  "OPEC_SUPPLY",       # production, million, oil
    1:  "SAUDI_OIL_NEWS",    # saudi, arabia, reuters, iran, output
    2:  "DEMAND_OUTLOOK",    # mb, opec, market, demand
    3:  "NOISE_EMAIL",       # cid, email, org — filtered before momentum
    4:  "IEA_NARRATIVE",     # mb, demand, growth, market
    5:  "GEOPOLITICAL",      # uae, exit, opec, cartel
    6:  "ARAMCO_FINANCIAL",  # aramco, profit, dividends
    7:  "HEATING_DEMAND",    # winter, heating, percent
    8:  "NOISE_SOCIAL",      # capitalist, venezuela, visual — filtered before momentum
}

NOISE_TOPICS     = {"NOISE_EMAIL", "NOISE_SOCIAL"}
MIN_DOCS_FOR_FIT = 100


def load_historic_topics() -> pd.DataFrame | None:
    if not HISTORIC_TOPICS_PATH.exists():
        log.info("  No historic topic assignments found in data/Historic/")
        return None
    try:
        df = pd.read_parquet(HISTORIC_TOPICS_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        log.info(f"✓ Loaded {len(df)} historic topic assignments from {HISTORIC_TOPICS_PATH}")
        return df
    except Exception as e:
        log.warning(f"  Could not load historic topics: {e}")
        return None


def ensure_model_available() -> bool:
    if MODEL_PATH.exists():
        return True
    if HISTORIC_MODEL_PATH.exists():
        try:
            # FIX 1: dirs_exist_ok=True prevents crash if destination already exists
            shutil.copytree(
                str(HISTORIC_MODEL_PATH),
                str(MODEL_PATH),
                dirs_exist_ok=True
            )
            log.info(f"✓ Copied historic BERTopic model → {MODEL_PATH}")
            return True
        except Exception as e:
            log.warning(f"  Could not copy historic model: {e}")
            return False
    log.info("  No BERTopic model available — will fit on current corpus")
    return False


def log_topic_info(topic_model) -> None:
    """Log discovered topic keywords with assigned labels."""
    topic_info = topic_model.get_topic_info()
    log.info("\n  ── Discovered Topics (update TOPIC_LABELS if needed) ──")
    for _, trow in topic_info.iterrows():
        if trow["Topic"] != -1:
            kws   = ", ".join([w for w, _ in topic_model.get_topic(trow["Topic"])[:8]])
            label = TOPIC_LABELS.get(trow["Topic"], "UNLABELLED")
            log.info(f"    Topic {trow['Topic']:2d} ({trow['Count']:3d} docs) → {label}: {kws}")


def save_model(topic_model) -> None:
    """Save BERTopic model to results directory."""
    shutil.rmtree(str(MODEL_PATH), ignore_errors=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    try:
        topic_model.save(str(MODEL_PATH), serialization="safetensors", save_ctfidf=True)
        log.info(f"  Model saved → {MODEL_PATH}")
    except Exception as e:
        log.warning(f"  Could not save model: {e}")


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

    # FIX 2: guard against missing text_clean column early
    if "text_clean" not in corpus.columns:
        log.warning("text_clean column missing from llm_scores.parquet — check script 20 output")
        return pd.DataFrame()

    corpus["date"] = pd.to_datetime(corpus["date"]).dt.tz_localize(None).dt.normalize()

    # ── Load historic topic assignments ────────────────────────────────────
    historic_df = load_historic_topics()
    hist_lookup = {}
    if historic_df is not None:
        for _, r in historic_df.iterrows():
            key = (str(r["date"].date()), str(r.get("source", "")))
            hist_lookup[key] = {
                "topic_id":    r.get("topic_id",    -1),
                "topic_prob":  r.get("topic_prob",  0.0),
                "topic_label": r.get("topic_label", "OUTLIER"),
            }
        log.info(f"  Historic topic keys: {len(hist_lookup)} (date+source pairs)")

    # ── Identify new documents not in historic data ────────────────────────
    new_docs_idx = [
        idx for idx, row in corpus.iterrows()
        if (str(row["date"].date()), str(row.get("source", ""))) not in hist_lookup
    ]

    log.info(f"\n  Corpus total:       {len(corpus)}")
    log.info(f"  From historic data: {len(corpus) - len(new_docs_idx)}")
    log.info(f"  New documents:      {len(new_docs_idx)}")

    # ── Assign topics to new documents ────────────────────────────────────
    new_topic_assignments = {}
    if new_docs_idx:
        model_available = ensure_model_available()

        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            if model_available and MODEL_PATH.exists():
                log.info(f"Loading BERTopic model for {len(new_docs_idx)} new documents...")
                try:
                    topic_model = BERTopic.load(
                        str(MODEL_PATH),
                        embedding_model=embedding_model
                    )
                    # FIX 3: column access not .get() on Series
                    new_texts = [
                        str(corpus.loc[idx]["text_clean"])
                        for idx in new_docs_idx
                    ]
                    new_topics, new_probs = topic_model.transform(new_texts)
                    for i, idx in enumerate(new_docs_idx):
                        tid = new_topics[i]
                        new_topic_assignments[idx] = {
                            "topic_id":    tid,
                            "topic_prob":  float(new_probs[i].max()
                                                 if hasattr(new_probs[i], "max")
                                                 else new_probs[i]),
                            "topic_label": TOPIC_LABELS.get(tid, "OTHER"),
                        }
                    log.info(f"✓ Topics predicted for {len(new_docs_idx)} new documents")

                except Exception as e:
                    log.warning(f"  Model load failed ({e}) — refitting on full corpus")

                    if len(corpus) >= MIN_DOCS_FOR_FIT:
                        topic_model = BERTopic(
                            embedding_model=embedding_model,
                            nr_topics=10,
                            min_topic_size=5,
                            verbose=False,
                            random_state=42,
                        )
                        all_texts  = corpus["text_clean"].fillna("").tolist()
                        all_topics, all_probs = topic_model.fit_transform(all_texts)
                        for i, (idx, _) in enumerate(corpus.iterrows()):
                            if idx in new_docs_idx:
                                tid = all_topics[i]
                                new_topic_assignments[idx] = {
                                    "topic_id":    tid,
                                    "topic_prob":  float(all_probs[i].max()
                                                         if hasattr(all_probs[i], "max")
                                                         else all_probs[i]),
                                    "topic_label": TOPIC_LABELS.get(tid, "OTHER"),
                                }
                        log_topic_info(topic_model)
                        save_model(topic_model)

            else:
                # No model and no historic — fit from scratch
                if len(corpus) >= MIN_DOCS_FOR_FIT:
                    log.info(f"Fitting BERTopic on {len(corpus)} documents (first run)...")
                    topic_model = BERTopic(
                        embedding_model=embedding_model,
                        nr_topics=10,
                        min_topic_size=5,
                        verbose=False,
                        random_state=42,
                    )
                    all_texts  = corpus["text_clean"].fillna("").tolist()
                    all_topics, all_probs = topic_model.fit_transform(all_texts)
                    for i, (idx, _) in enumerate(corpus.iterrows()):
                        tid = all_topics[i]
                        new_topic_assignments[idx] = {
                            "topic_id":    tid,
                            "topic_prob":  float(all_probs[i].max()
                                                 if hasattr(all_probs[i], "max")
                                                 else all_probs[i]),
                            "topic_label": TOPIC_LABELS.get(tid, "OTHER"),
                        }
                    log_topic_info(topic_model)
                    save_model(topic_model)
                else:
                    log.warning(
                        f"  Only {len(corpus)} documents — below minimum {MIN_DOCS_FOR_FIT}. "
                        f"Assigning PENDING_BACKFILL."
                    )
                    for idx in new_docs_idx:
                        new_topic_assignments[idx] = {
                            "topic_id":    0,
                            "topic_prob":  0.5,
                            "topic_label": "PENDING_BACKFILL",
                        }

        except ImportError:
            log.warning("  BERTopic not installed — assigning UNKNOWN to new documents")
            for idx in new_docs_idx:
                new_topic_assignments[idx] = {
                    "topic_id":    0,
                    "topic_prob":  0.5,
                    "topic_label": "UNKNOWN",
                }

    # ── Build final corpus with topic assignments ──────────────────────────
    topic_ids    = []
    topic_probs  = []
    topic_labels = []

    for idx, row in corpus.iterrows():
        key = (str(row["date"].date()), str(row.get("source", "")))
        if key in hist_lookup:
            t = hist_lookup[key]
        elif idx in new_topic_assignments:
            t = new_topic_assignments[idx]
        else:
            t = {"topic_id": -1, "topic_prob": 0.0, "topic_label": "OUTLIER"}

        topic_ids.append(t["topic_id"])
        topic_probs.append(t["topic_prob"])
        topic_labels.append(t["topic_label"])

    corpus["topic_id"]    = topic_ids
    corpus["topic_prob"]  = topic_probs
    corpus["topic_label"] = topic_labels

    corpus.to_parquet(out)

    log.info(f"\n✓ Corpus with topics saved → {out}")
    log.info(f"  Topic distribution:")
    for label, count in corpus["topic_label"].value_counts().items():
        noise_flag = " [filtered]" if label in NOISE_TOPICS else ""
        log.info(f"    {label:30s}: {count}{noise_flag}")

    noise_count = corpus["topic_label"].isin(NOISE_TOPICS).sum()
    log.info(f"\n  Total documents:        {len(corpus)}")
    log.info(f"  Noise (filtered later): {noise_count}")
    log.info(f"  Real content:           {len(corpus) - noise_count}")

    return corpus


if __name__ == "__main__":
    run_bertopic_clustering()
