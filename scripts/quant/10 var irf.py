"""
Script 10 — VAR Model + Impulse Response Functions
Uses top Granger-significant features from script 09.
Fits a low-dimensional VAR to avoid multicollinearity.
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "oil_logret"
MAX_VAR_VARS = 5   # keep VAR small to avoid multicollinearity


def drop_correlated(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Drop columns that are highly correlated with each other.
    Keeps the first column (TARGET) unconditionally.
    Prevents singular covariance matrix in VAR lag selection.
    """
    corr   = df.corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns
                if col != TARGET and any(upper[col] > threshold)]
    if to_drop:
        log.info(f"  Dropping {len(to_drop)} highly correlated columns: {to_drop}")
    return df.drop(columns=to_drop)


def run_var_irf():
    granger_path = RESULTS_DIR / "granger_significant.csv"
    master_path  = FEATURES_DIR / "master_with_nlp.parquet"
    if not master_path.exists():
        master_path = FEATURES_DIR / "master_quant.parquet"
    if not master_path.exists():
        log.warning("No master dataset found — skipping VAR")
        return

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)

    # ── Select top Granger-significant features ────────────────────────────
    if granger_path.exists():
        granger   = pd.read_csv(granger_path).sort_values("p_value")
        top_feats = granger.head(MAX_VAR_VARS - 1)["feature"].tolist()
        log.info(f"  Using top {len(top_feats)} Granger-significant features")
    else:
        top_feats = [c for c in ["eia_surprise_norm", "cot_net_long_change",
                                  "sent_ema_cross"]
                     if c in master.columns][:MAX_VAR_VARS - 1]
        log.info(f"  No Granger results found — using default features")

    cols = [TARGET] + [f for f in top_feats if f in master.columns]
    data = master[cols].dropna()

    if len(data) < 100:
        log.warning(f"Insufficient data for VAR ({len(data)} rows) — skipping")
        return

    impulse_vars = [c for c in cols if c != TARGET]
    if not impulse_vars:
        log.warning("No impulse variables available — skipping IRF plots")
        return

    # ── Drop highly correlated columns to prevent singular covariance ──────
    data = drop_correlated(data, threshold=0.90)
    impulse_vars = [c for c in data.columns if c != TARGET]
    if not impulse_vars:
        log.warning("All impulse variables dropped as correlated — skipping VAR")
        return

    log.info(f"VAR system: {list(data.columns)}")

    # ── Lag selection with fallback ────────────────────────────────────────
    try:
        lag_select = VAR(data).select_order(maxlags=10)
        # BUG FIX: use selected_orders["aic"] not lag_select.aic
        n_lags = lag_select.selected_orders.get("aic", 2)
        n_lags = max(1, min(int(n_lags), 10))
        log.info(f"VAR optimal lag (AIC): {n_lags}")
    except Exception as e:
        log.warning(f"  Lag selection failed ({e}) — defaulting to lag=2")
        n_lags = 2

    # ── Fit VAR ────────────────────────────────────────────────────────────
    try:
        var_results = VAR(data).fit(maxlags=n_lags, ic="aic")
        log.info(f"✓ VAR fitted: {len(data.columns)} variables, {var_results.k_ar} lags")
    except Exception as e:
        log.warning(f"  VAR fit failed ({e}) — skipping VAR/IRF")
        return

    # ── Impulse Response Functions ─────────────────────────────────────────
    try:
        irf = var_results.irf(periods=20)
    except Exception as e:
        log.warning(f"  IRF computation failed ({e}) — skipping IRF plots")
        irf = None

    if irf is not None:
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

        plt.suptitle("Impulse Response Functions — VAR Model",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "irf_plots.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"✓ IRF plots saved → {RESULTS_DIR / 'irf_plots.png'}")

    # ── VAR Summary ────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "var_summary.txt", "w") as f:
        f.write(str(var_results.summary()))
    log.info(f"✓ VAR summary saved → {RESULTS_DIR / 'var_summary.txt'}")


if __name__ == "__main__":
    run_var_irf()
