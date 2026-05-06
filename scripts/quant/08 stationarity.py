"""
Script 08 — Stationarity Tests (ADF)
Runs after qualitative pipeline has merged NLP scores.
Tests all variables for stationarity before Granger causality.
"""
import os, logging
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_adf(series: pd.Series, significance: float = 0.05) -> dict:
    series = series.dropna()
    if len(series) < 30:
        return {"p_value": 1.0, "stationary": False, "adf_stat": None}
    adf_stat, p_val, _, _, critical, _ = adfuller(series, autolag="AIC")
    return {
        "adf_stat":   round(float(adf_stat), 4),
        "p_value":    round(float(p_val), 6),
        "stationary": p_val < significance,
        "crit_1pct":  round(float(critical["1%"]), 3),
        "crit_5pct":  round(float(critical["5%"]), 3),
    }


def run_stationarity_tests():
    # Use full dataset if NLP scores available, else quantitative only
    master_path = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        master_path = FEATURES_DIR / "master_quant.parquet"
        log.warning("NLP master not found — using quantitative master")

    if not master_path.exists():
        log.warning("No master dataset found — skipping stationarity tests")
        pd.DataFrame().to_csv(RESULTS_DIR / "adf_results.csv", index=False)
        return pd.DataFrame()

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)

    # Test all numeric columns
    results = []
    for col in master.select_dtypes(include=[np.number]).columns:
        if master[col].notna().sum() < 50:
            continue
        r = run_adf(master[col])
        r["variable"] = col
        results.append(r)

    if not results:
        log.warning("No columns had sufficient data for ADF tests")
        pd.DataFrame().to_csv(RESULTS_DIR / "adf_results.csv", index=False)
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("p_value")
    df.to_csv(RESULTS_DIR / "adf_results.csv", index=False)

    non_stationary = df[~df["stationary"]]["variable"].tolist()
    stationary     = df[df["stationary"]]["variable"].tolist()

    log.info(f"✓ ADF tests complete: {len(stationary)} stationary, {len(non_stationary)} non-stationary")
    if non_stationary:
        log.info(f"  Non-stationary (will be differenced): {non_stationary[:5]}...")
        # Difference non-stationary series and re-save master
        for col in non_stationary:
            if col in master.columns:
                master[f"{col}_diff"] = master[col].diff()
        master.to_parquet(master_path)
        log.info(f"  Differenced series added to {master_path}")

    return df


if __name__ == "__main__":
    run_stationarity_tests()
