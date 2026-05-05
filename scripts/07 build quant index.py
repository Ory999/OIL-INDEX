import os, logging, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  "data/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def classify_regime(score: float) -> str:
    if score <= 25:   return "EXTREME_FEAR"
    elif score <= 45: return "FEAR"
    elif score <= 55: return "NEUTRAL"
    elif score <= 75: return "GREED"
    else:             return "EXTREME_GREED"


def normalise_to_unit(series: pd.Series) -> pd.Series:
    return series.rank(pct=True).clip(0, 1)


def build_quant_index():
    master_path = FEATURES_DIR / "master_quant.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} not found — run 06 assemble master.py first")

    master = pd.read_parquet(master_path)
    master.index = pd.to_datetime(master.index)
    scaler = MinMaxScaler(feature_range=(0, 1))

    components = {}

    fund_cols = []
    if "eia_surprise_norm" in master.columns:
        fund_cols.append(-master["eia_surprise_norm"].fillna(0))
    if "crude_stocks_change" in master.columns:
        fund_cols.append(-normalise_to_unit(master["crude_stocks_change"].fillna(0)))
    if "cot_net_long" in master.columns:
        fund_cols.append(normalise_to_unit(master["cot_net_long"].fillna(0)))
    if "cot_change_1w" in master.columns:
        fund_cols.append(normalise_to_unit(master["cot_change_1w"].fillna(0)))

    if fund_cols:
        fund_raw = pd.concat(fund_cols, axis=1).mean(axis=1)
        components["fundamentals"] = pd.Series(
            scaler.fit_transform(fund_raw.values.reshape(-1, 1)).flatten(),
            index=fund_raw.index,
        )
        log.info(f"  ✓ Fundamentals component built ({len(fund_cols)} sub-signals)")
    else:
        components["fundamentals"] = pd.Series(0.5, index=master.index)
        log.warning("  No fundamental columns found — using neutral 0.5")

    weights = {"fundamentals": 1.0}

    comp_df   = pd.DataFrame(components).fillna(0.5)
    raw_score = sum(comp_df[k] * w for k, w in weights.items() if k in comp_df)
    quant_prcsi = pd.Series(
        scaler.fit_transform(raw_score.values.reshape(-1, 1)).flatten() * 100,
        index=raw_score.index,
        name="prcsi_quant",
    )
    quant_prcsi_smooth = quant_prcsi.rolling(5, min_periods=1).mean()

    result = pd.DataFrame({
        "prcsi_quant_raw":   quant_prcsi,
        "prcsi_quant":       quant_prcsi_smooth,
        "regime_quant":      quant_prcsi_smooth.apply(classify_regime),
        "comp_fundamentals": comp_df["fundamentals"],
        "oil_price":         master["oil"]        if "oil"        in master.columns else np.nan,
        "oil_logret":        master["oil_logret"] if "oil_logret" in master.columns else np.nan,
    })

    result.to_parquet(RESULTS_DIR / "prcsi_quant_partial.parquet")
    result.to_csv(RESULTS_DIR    / "prcsi_quant_partial.csv")

    latest_score  = round(float(quant_prcsi_smooth.iloc[-1]), 2)
    latest_regime = classify_regime(latest_score)

    log.info(f"\n✓ Partial quantitative PRCSI built")
    log.info(f"  Latest score: {latest_score:.1f} / 100")
    log.info(f"  Regime:       {latest_regime}")

    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        "prcsi_quant_latest":      latest_score,
        "prcsi_quant_regime":      latest_regime,
        "quant_pipeline_complete": True,
        "quant_run_timestamp":     datetime.now().isoformat(),
        "nlp_scores_merged":       False,
    })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return result


if __name__ == "__main__":
    build_quant_index()
