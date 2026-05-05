"""
Script 10 — VAR Model + Impulse Response Functions
Uses top Granger features from script 09.
"""
import os, logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
TARGET       = "oil_logret"
MAX_VAR_VARS = 5   # Keep VAR tractable


def run_var_irf():
    # Load Granger results to pick top features
    granger_path = RESULTS_DIR / "granger_significant.csv"
    master_path  = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        master_path = FEATURES_DIR / "master_quant.parquet"

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)

    # Select top features
    if granger_path.exists():
        granger = pd.read_csv(granger_path).sort_values("p_value")
        top_feats = granger.head(MAX_VAR_VARS - 1)["feature"].tolist()
    else:
        # Fallback: use EIA surprise and COT if no Granger results yet
        top_feats = [c for c in ["eia_surprise_norm", "cot_net_long_change",
                                  "sent_ema_cross", "congress_net_signal"]
                     if c in master.columns][:MAX_VAR_VARS - 1]

    cols = [TARGET] + [f for f in top_feats if f in master.columns]
    data = master[cols].dropna()

    if len(data) < 100:
        log.warning(f"Insufficient data for VAR ({len(data)} rows) — skipping")
        return

    # Lag selection
    lag_select = VAR(data).select_order(maxlags=10)
    n_lags = max(1, lag_select.aic)
    log.info(f"VAR optimal lag (AIC): {n_lags}")

    # Fit
    var_results = VAR(data).fit(maxlags=n_lags)
    log.info(f"✓ VAR fitted: {len(cols)} variables, {n_lags} lags")

    # IRF
    irf = var_results.irf(periods=20)
    impulse_vars = [c for c in cols if c != TARGET]

    fig, axes = plt.subplots(1, len(impulse_vars),
                              figsize=(6 * len(impulse_vars), 5))
    if len(impulse_vars) == 1:
        axes = [axes]

    for ax, impulse_var in zip(axes, impulse_vars):
        try:
            imp_idx  = list(data.columns).index(impulse_var)
            res_idx  = list(data.columns).index(TARGET)
            irf_vals = irf.irfs[:, res_idx, imp_idx]
            std      = np.std(irf_vals)
            x        = range(len(irf_vals))
            ax.plot(irf_vals, color="#1B2A4A", linewidth=2)
            ax.fill_between(x, irf_vals - std, irf_vals + std,
                            alpha=0.2, color="#2563EB")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"Shock: {impulse_var[:30]}\n→ WTI Log Return", fontsize=9)
            ax.set_xlabel("Days after shock")
            ax.grid(alpha=0.3)
        except Exception as e:
            ax.set_title(f"IRF error: {str(e)[:30]}")

    plt.suptitle("Impulse Response Functions — VAR Model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "irf_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"✓ IRF plots saved → {RESULTS_DIR / 'irf_plots.png'}")

    # Save VAR summary
    with open(RESULTS_DIR / "var_summary.txt", "w") as f:
        f.write(str(var_results.summary()))
    log.info(f"✓ VAR summary saved")


if __name__ == "__main__":
    run_var_irf()
