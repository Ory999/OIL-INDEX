"""
Script 11 — SHAP Feature Importance
Trains gradient boosting on full combined feature set.
Shows which components (NLP vs fundamentals)
contribute most to predicting WTI oil returns.
"""
import os, logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
TARGET       = "oil_logret"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_shap_analysis():
    master_path = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        master_path = FEATURES_DIR / "master_quant.parquet"
        log.warning("NLP master not found — SHAP on quantitative features only")

    if not master_path.exists():
        log.warning("No master dataset found — skipping SHAP analysis")
        return

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)

    # Feature selection: exclude raw prices and returns (data leakage)
    exclude = ["_return", "_logret", "oil", "gold", "regime", "prcsi"]
    feature_cols = [
        c for c in master.select_dtypes(include=[np.number]).columns
        if c != TARGET
        and not any(x in c for x in exclude)
        and master[c].notna().sum() > 100
    ]

    df = master[[TARGET] + feature_cols].dropna()
    X, y = df[feature_cols], df[TARGET]

    if len(X) < 200:
        log.warning(f"Insufficient data for SHAP ({len(X)} rows) — skipping")
        return

    # Time-series cross-validation (no look-ahead)
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=42
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores.append({
            "mae": mean_absolute_error(y.iloc[test_idx], pred),
            "r2":  r2_score(y.iloc[test_idx], pred),
        })

    avg_r2  = np.mean([s["r2"]  for s in scores])
    avg_mae = np.mean([s["mae"] for s in scores])
    log.info(f"CV performance — R²: {avg_r2:.4f}  MAE: {avg_mae:.5f}")

    # Final model on full data for SHAP
    final_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42
    )
    final_model.fit(X, y)

    explainer  = shap.TreeExplainer(final_model)
    shap_vals  = explainer.shap_values(X)

    # Importance DataFrame
    importance = pd.DataFrame({
        "feature":        X.columns,
        "mean_abs_shap":  np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

    log.info(f"\nTop 10 SHAP features:")
    for _, row in importance.head(10).iterrows():
        log.info(f"  {row['feature']:45s}  {row['mean_abs_shap']:.6f}")

    # SHAP bar chart
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP Feature Importance — WTI Return Prediction\nCV R²={avg_r2:.4f}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, max_display=15, show=False)
    plt.title("SHAP Beeswarm — Feature Direction & Magnitude")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"✓ SHAP analysis complete  (R²={avg_r2:.4f})")
    return importance


if __name__ == "__main__":
    run_shap_analysis()
