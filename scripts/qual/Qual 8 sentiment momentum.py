# Convert daily NLP scores into momentum features.
# Builds ROC, EMA crossover, RSI, acceleration, per-topic streams, divergence momentum.
# Noise topics filtered before momentum calculation.
# Raw LLM signals filled 21 days, one publication cycle. Momentum features filled 5 days.
# FinBERT removed from pipeline, historic scores preserved for r=0.13 paper result.

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

# Excluded from momentum.
NOISE_TOPICS = {"NOISE_EMAIL", "NOISE_SOCIAL"}
# Excluded from per-topic streams.
SKIP_TOPICS  = {"NOISE_EMAIL", "NOISE_SOCIAL", "OUTLIER", "UNKNOWN", "PENDING_BACKFILL"}

# Forward-filled 21 days, one publication cycle.
RAW_LLM_COLS = [
    "oil_impact_score",
    "supply_disruption_signal",
    "demand_outlook_signal",
    "geopolitical_risk_signal",
    "surface_vs_implied_divergence",
    "institutional_confidence",
]


def compute_rsi(series: pd.Series, window: int = 7) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def build_daily_sentiment(corpus_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate document scores to daily series, fill 5d weekends then 21d publication cycle.
    corpus_df = corpus_df.copy()
    corpus_df["date"] = pd.to_datetime(corpus_df["date"]).dt.normalize()

    # FinBERT removed, not in score_cols.
    score_cols    = RAW_LLM_COLS.copy()
    available_cols = [c for c in score_cols if c in corpus_df.columns]

    # Daily mean across documents.
    daily = corpus_df.groupby("date")[available_cols].mean()

    # Per-source daily series.
    for source in corpus_df["source"].unique():
        sub = (corpus_df[corpus_df["source"] == source]
               .groupby("date")["oil_impact_score"].mean())
        daily[f"sent_{source.lower()}"] = sub

    # Per-topic daily series, skip noise and placeholders.
    if "topic_label" in corpus_df.columns:
        for topic in corpus_df["topic_label"].unique():
            if topic in SKIP_TOPICS:
                continue
            sub = (corpus_df[corpus_df["topic_label"] == topic]
                   .groupby("date")["oil_impact_score"].mean())
            daily[f"topic_{topic.lower()}"] = sub

    # Documents per day.
    daily["doc_count"] = corpus_df.groupby("date").size()

    # Business day resample, 5-day fill for weekends and short gaps.
    daily = daily.resample("B").mean().ffill(limit=5)

    # Raw LLM extended to 21 days for one full OPEC/EIA cycle.
    available_raw = [c for c in RAW_LLM_COLS if c in daily.columns]
    daily[available_raw] = daily[available_raw].ffill(limit=21)

    filled_pct = daily[available_raw].notna().mean().mean() * 100
    log.info(f"  Raw LLM signal coverage after 21-day fill: {filled_pct:.1f}%")

    return daily


def engineer_momentum(daily_df: pd.DataFrame) -> pd.DataFrame:
    df   = daily_df.copy()
    base = "oil_impact_score"

    if base not in df.columns:
        log.warning(f"  '{base}' not in daily sentiment — check corpus")
        if df.columns.empty:
            return df
        base = df.columns[0]
        log.warning(f"  Falling back to: {base}")

    # Rate of change.
    for lag in [1, 3, 7, 14]:
        df[f"sent_roc_{lag}d"] = df[base].diff(lag)

    # EMA crossover.
    df["sent_ema_fast"]  = df[base].ewm(span=EMA_FAST, adjust=False).mean()
    df["sent_ema_slow"]  = df[base].ewm(span=EMA_SLOW, adjust=False).mean()
    df["sent_ema_cross"] = df["sent_ema_fast"] - df["sent_ema_slow"]

    # RSI on sentiment.
    df["sent_rsi"] = compute_rsi(df[base], window=RSI_WIN)

    # First and second derivatives.
    df["sent_velocity"] = df[base].diff(1)
    df["sent_accel"]    = df["sent_velocity"].diff(1)

    # Supply disruption momentum.
    if "supply_disruption_signal" in df.columns:
        df["supply_disruption_ema"] = df["supply_disruption_signal"].ewm(span=7).mean()
        df["supply_disruption_roc"] = df["supply_disruption_signal"].diff(3)

    # Geopolitical momentum.
    if "geopolitical_risk_signal" in df.columns:
        df["geo_risk_ema"] = df["geopolitical_risk_signal"].ewm(span=7).mean()
        df["geo_risk_roc"] = df["geopolitical_risk_signal"].diff(3)

    # Divergence momentum.
    if "surface_vs_implied_divergence" in df.columns:
        df["divergence_ema"] = df["surface_vs_implied_divergence"].ewm(span=5).mean()
        df["divergence_roc"] = df["surface_vs_implied_divergence"].diff(3)

    # Per-topic EMA crossovers.
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

    # Filter noise topics before momentum.
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

    # Coverage report.
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
