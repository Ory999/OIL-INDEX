"""
Script 22 — Sentiment Momentum Feature Engineering
Converts daily NLP scores into momentum features.
Outputs: data/raw/sentiment_features.parquet

Theoretical link (Behavioral Finance — De Bondt & Thaler 1985):
Markets overreact to NEW information — the acceleration of sentiment
shift captures the onset of overreaction better than raw sentiment level.

Features engineered:
  - Rate of Change (ROC):     How fast is sentiment shifting?
  - EMA Crossover:            Short vs long-term trend divergence
  - RSI:                      Overbought/oversold sentiment regime
  - Acceleration:             2nd derivative — captures inflection points
  - Per-topic streams:        Separate momentum per BERTopic cluster
  - Divergence momentum:      Rate of change of surface/implied gap

Noise filtering:
  NOISE_EMAIL and NOISE_SOCIAL topics are excluded before momentum
  calculation — these contain email headers and social media noise,
  not oil market content.

Forward-fill strategy:
  Raw LLM signals (oil_impact_score etc.) are forward-filled 21 trading
  days — one full publication cycle (OPEC and EIA publish monthly).
  Without this, raw scores are NaN for ~17 of every 21 days, making them
  unusable as direct Granger causality inputs.
  Economically justified: the market holds the most recent institutional
  view until the next publication replaces it.
  Momentum/EMA features use 5-day fill — short-term signal decay.
  FinBERT removed from pipeline — historic scores preserved in
  data/Historic/finbert_scores.parquet for the paper's r=0.13 result.
"""
import os, logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "14"))
RSI_WIN  = 7

# Topics excluded from momentum calculation
NOISE_TOPICS = {"NOISE_EMAIL", "NOISE_SOCIAL"}
# Topics excluded from per-topic feature streams
SKIP_TOPICS  = {"NOISE_EMAIL", "NOISE_SOCIAL", "OUTLIER", "UNKNOWN", "PENDING_BACKFILL"}

# Raw LLM signal columns — forward-filled 21 days (one publication cycle)
RAW_LLM_COLS = [
    "oil_impact_score",
    "supply_disruption_signal",
    "demand_outlook_signal",
    "geopolitical_risk_signal",
    "surface_vs_implied_divergence",
    "institutional_confidence",
]


def compute_rsi(series: pd.Series, window: int = 7) -> pd.Series:
    """RSI adapted for sentiment series. Returns 0-100."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def build_daily_sentiment(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate document-level scores to daily time series.
    Multiple documents on same day are averaged.
    Noise topics already filtered before this function is called.

    Forward-fill strategy:
      - All columns: 5-day fill after resample (covers weekends)
      - Raw LLM signals: additional 21-day fill (covers monthly gaps)
        Applied after the 5-day fill so weekday gaps are already closed.
    """
    corpus_df = corpus_df.copy()
    corpus_df["date"] = pd.to_datetime(corpus_df["date"]).dt.normalize()

    # FinBERT removed from pipeline — not included in score_cols
    score_cols    = RAW_LLM_COLS.copy()
    available_cols = [c for c in score_cols if c in corpus_df.columns]

    # Daily mean across all documents
    daily = corpus_df.groupby("date")[available_cols].mean()

    # Per-source daily series
    for source in corpus_df["source"].unique():
        sub = (corpus_df[corpus_df["source"] == source]
               .groupby("date")["oil_impact_score"].mean())
        daily[f"sent_{source.lower()}"] = sub

    # Per-topic daily series — skip noise and placeholder topics
    if "topic_label" in corpus_df.columns:
        for topic in corpus_df["topic_label"].unique():
            if topic in SKIP_TOPICS:
                continue
            sub = (corpus_df[corpus_df["topic_label"] == topic]
                   .groupby("date")["oil_impact_score"].mean())
            daily[f"topic_{topic.lower()}"] = sub

    # Document count per day
    daily["doc_count"] = corpus_df.groupby("date").size()

    # Resample to business days — 5-day fill covers weekends and short gaps
    daily = daily.resample("B").mean().ffill(limit=5)

    # Raw LLM signals: extend to 21-day fill (one full publication cycle).
    # OPEC and EIA publish monthly — without this, raw scores are NaN for
    # ~17 of every 21 trading days, making them unusable as Granger inputs.
    available_raw = [c for c in RAW_LLM_COLS if c in daily.columns]
    daily[available_raw] = daily[available_raw].ffill(limit=21)

    filled_pct = daily[available_raw].notna().mean().mean() * 100
    log.info(f"  Raw LLM signal coverage after 21-day fill: {filled_pct:.1f}%")

    return daily


def engineer_momentum(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build all momentum features from daily sentiment scores."""
    df   = daily_df.copy()
    base = "oil_impact_score"

    if base not in df.columns:
        log.warning(f"  '{base}' not in daily sentiment — check corpus")
        if df.columns.empty:
            return df
        base = df.columns[0]
        log.warning(f"  Falling back to: {base}")

    # ── Rate of Change ────────────────────────────────────────────────────
    for lag in [1, 3, 7, 14]:
        df[f"sent_roc_{lag}d"] = df[base].diff(lag)

    # ── EMA Crossover (Jegadeesh & Titman 1993 momentum) ─────────────────
    df["sent_ema_fast"]  = df[base].ewm(span=EMA_FAST, adjust=False).mean()
    df["sent_ema_slow"]  = df[base].ewm(span=EMA_SLOW, adjust=False).mean()
    df["sent_ema_cross"] = df["sent_ema_fast"] - df["sent_ema_slow"]

    # ── RSI on sentiment ──────────────────────────────────────────────────
    df["sent_rsi"] = compute_rsi(df[base], window=RSI_WIN)

    # ── Velocity and Acceleration (1st and 2nd derivatives) ──────────────
    df["sent_velocity"] = df[base].diff(1)
    df["sent_accel"]    = df["sent_velocity"].diff(1)

    # ── Supply disruption momentum ────────────────────────────────────────
    if "supply_disruption_signal" in df.columns:
        df["supply_disruption_ema"] = df["supply_disruption_signal"].ewm(span=7).mean()
        df["supply_disruption_roc"] = df["supply_disruption_signal"].diff(3)

    # ── Geopolitical momentum ─────────────────────────────────────────────
    if "geopolitical_risk_signal" in df.columns:
        df["geo_risk_ema"] = df["geopolitical_risk_signal"].ewm(span=7).mean()
        df["geo_risk_roc"] = df["geopolitical_risk_signal"].diff(3)

    # ── Divergence momentum (Information Asymmetry — Akerlof 1970) ────────
    if "surface_vs_implied_divergence" in df.columns:
        df["divergence_ema"] = df["surface_vs_implied_divergence"].ewm(span=5).mean()
        df["divergence_roc"] = df["surface_vs_implied_divergence"].diff(3)

    # ── Per-topic EMA crossovers ──────────────────────────────────────────
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    for col in topic_cols:
        df[f"{col}_ema_cross"] = (
            df[col].ewm(span=EMA_FAST).mean() - df[col].ewm(span=EMA_SLOW).mean()
        )

    log.info(f"✓ Momentum features engineered: {len(df.columns)} total columns")
    return df


def run_sentiment_momentum():
    corpus_path = RAW_DIR / "corpus_with_topics.parquet"
    out         = RAW_DIR / "sentiment_features.parquet"

    if not corpus_path.exists():
        log.warning("corpus_with_topics.parquet not found — run 21 bertopic clustering.py first")
        return pd.DataFrame()

    corpus = pd.read_parquet(corpus_path)
    if len(corpus) == 0:
        log.warning("Corpus empty — skipping momentum engineering")
        return pd.DataFrame()

    # Filter noise topics before momentum calculation
    if "topic_label" in corpus.columns:
        noise_count = corpus["topic_label"].isin(NOISE_TOPICS).sum()
        corpus = corpus[~corpus["topic_label"].isin(NOISE_TOPICS)].copy()
        log.info(f"  Filtered {noise_count} noise documents — "
                 f"{len(corpus)} remaining for momentum")

    log.info("Building daily sentiment aggregation...")
    daily = build_daily_sentiment(corpus)

    log.info("Engineering momentum features...")
    features = engineer_momentum(daily)

    features.to_parquet(out)
    log.info(f"✓ Sentiment features saved → {out}  ({len(features)} trading days)")

    # Coverage report for raw LLM signals
    log.info("\n  Raw LLM signal coverage in output:")
    for col in RAW_LLM_COLS:
        if col in features.columns:
            pct = features[col].notna().mean() * 100
            log.info(f"    {col:40s}: {pct:.1f}%")

    if "sent_ema_cross" in features.columns:
        log.info(f"\n  sent_ema_cross range: "
                 f"{features['sent_ema_cross'].min():+.4f} to "
                 f"{features['sent_ema_cross'].max():+.4f}")
    if "sent_accel" in features.columns:
        log.info(f"  sent_accel range:     "
                 f"{features['sent_accel'].min():+.4f} to "
                 f"{features['sent_accel'].max():+.4f}")

    return features


if __name__ == "__main__":
    run_sentiment_momentum()
