"""
Member 2 - Shared machine-learning utilities

1. Purpose: load Member 1's production features, check their structure, and
   prepare the same numeric inputs for training, testing, and prediction.
2. Input: outputs/model_features.parquet produced by src/feature_engineering.py.
   A .csv.gz input is also supported for local debugging or data exchange.
3. Output: cleaned feature matrices for the model; this file does not fit a model.
4. Safety: never include future failure labels, well ID, or future maintenance
   outcomes as model input. Fix obvious invalid sensor placeholders only.
5. Scope: risk is estimated for producing wells, using data available through
   the end of each observation day. Do not treat it as live sensor monitoring.
6. Privacy: well-level data must remain local, not committed to public GitHub.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "model_features.parquet"

# Only features that Member 1 already produces from current/prior-day records.
# These names must remain identical for training and later predictions.
FEATURES = [
    "oil_produced_m3", "oil_previous_m3", "oil_baseline_7d",
    "oil_change_pct", "oil_decline_pct", "water_cut_pct",
    "pcp_pump_rpm", "pcp_pump_rpm_change",
    "pump_efficiency_pct", "pump_efficiency_pct_change",
    "tubing_pressure_kpa", "casing_pressure_kpa",
    "pcp_pump_electric_current_a", "pcp_pump_rod_torque_ftlb",
    "hours_down", "hours_on_prod", "has_downtime",
    "possible_invisible_loss",
]
EXTRA = [
    "uwi", "production_date", "well_daily_status", "flow_direction",
    "failure_next_7d", "failure_label_observable",
    "potential_oil_loss_m3",
]


def load_features(path=DEFAULT_INPUT):
    """Load feature dataset; fail clearly if Member 1's output is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python src/data_cleaning.py' and then "
            "'python src/feature_engineering.py' from the repository root."
        )
    if path.name.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        # Select only columns needed by ML, not all 41 original data fields.
        df = pd.read_parquet(path, columns=FEATURES + EXTRA)
    needed = ["uwi", "production_date", "failure_next_7d"]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce")
    df["uwi"] = df["uwi"].astype("string").str.strip()
    df = df.dropna(subset=["uwi", "production_date"])
    return df


def producing_wells(df, labels_required=True):
    """Select production well-days; use only fully observed outcome windows."""
    result = df
    if "well_daily_status" in result:
        result = result.loc[result["well_daily_status"].eq("Producing")]
    if "flow_direction" in result:
        result = result.loc[result["flow_direction"].eq("production")]
    if labels_required:
        if "failure_label_observable" not in result:
            raise ValueError("Missing failure_label_observable: regenerate model_features.")
        visible = result["failure_label_observable"]
        if visible.dtype == object:
            visible = visible.astype("string").str.lower().eq("true")
        result = result.loc[visible.fillna(False)]
    # A well can occasionally have two records for the same date.
    return result.drop_duplicates(["uwi", "production_date"], keep="last").copy()


def make_matrix(df, feature_names=FEATURES):
    """Return finite, float32 model inputs, keeping missing values as NaN."""
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col not in df.columns:
            raise ValueError(f"Required ML feature missing: {col}")
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            x = series.astype("float32")
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            x = series.astype("string").str.lower().map(
                {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
            ) if col in ("has_downtime", "possible_invisible_loss") else pd.to_numeric(series, errors="coerce")
        else:
            x = pd.to_numeric(series, errors="coerce")
        X[col] = x.astype("float32")

    # Documented placeholder (999999) is not an actual pressure measurement.
    for col in ("tubing_pressure_kpa", "casing_pressure_kpa"):
        X.loc[(X[col] >= 999999) | (X[col] < 0), col] = np.nan
    for col in ("hours_down", "hours_on_prod"):
        X.loc[(X[col] < 0) | (X[col] > 24), col] = np.nan
    for col in ("oil_produced_m3", "oil_previous_m3", "oil_baseline_7d"):
        X.loc[X[col] < 0, col] = np.nan
    X.loc[X["pcp_pump_rpm"] < 0, "pcp_pump_rpm"] = np.nan
    X.loc[X["pump_efficiency_pct"] < 0, "pump_efficiency_pct"] = np.nan
    # Limit the effect of huge percentage jumps from nearly-zero oil volumes.
    for col in ("oil_change_pct", "oil_decline_pct"):
        X[col] = X[col].clip(-500, 500)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def data_splits(df):
    """Use periods with meaningful recorded failure coverage.

    Only two failures are logged before 2025, so those years cannot support
    honest supervised training. Seven-day gaps stop outcome-window overlap.
    """
    day = df["production_date"]
    train = (day >= "2025-06-01") & (day <= "2025-12-24")
    valid = (day >= "2026-01-01") & (day <= "2026-03-24")
    test = day >= "2026-04-01"
    return train, valid, test


def labels(df):
    """Future failure within 1-7 days (strictly after observation date)."""
    y = pd.to_numeric(df["failure_next_7d"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("failure_next_7d must contain only 0 and 1")
    return y.astype("int8")
