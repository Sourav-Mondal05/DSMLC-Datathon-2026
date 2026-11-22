"""Prepare small, PRIVATE CSV files for the Streamlit dashboard.

Purpose
-------
1. Read the cleaned daily production file created by Member 1 (or the original
   production dataset when the cleaned file does not exist).
2. Keep producing well records and calculate field-wide daily totals.
3. Save only the most recent 90 days of well-level data, so Streamlit loads
   quickly without reading millions of rows every time the page refreshes.
4. Prepare a small failure-event CSV for the per-well history page.

Inputs (stored locally, not on public GitHub)
-------------------------------------------
- outputs/cleaned_production.parquet OR data/daily_production.parquet
- data/production_failures.parquet (optional)

Outputs (all in the PRIVATE outputs/ directory)
------------------------------------------------
- dashboard_daily.csv: field-level daily oil, water, downtime and well counts
- dashboard_recent_well_daily.csv: last 90 calendar days of well measurements
- dashboard_failures.csv: historical equipment failure events, if available

Important
---------
- Run from the repository root: python dashboard/prepare_dashboard_data.py
- This script does NOT train an ML model or generate risk predictions.
- The CSVs contain restricted well-level information: do NOT upload them.
- Values are historical daily readings, NOT a live data feed.
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
DATA = ROOT / "data"

PRODUCTION_REQUIRED = ["uwi", "production_date", "oil_produced_m3"]
PRODUCTION_OPTIONAL = [
    "water_produced_m3", "hours_down", "pcp_pump_rpm",
    "pump_efficiency_pct", "tubing_pressure_kpa", "casing_pressure_kpa",
    "well_daily_status", "flow_direction",
]
FAILURE_OPTIONAL = ["uwi", "failure_start_date", "failure_type", "cause", "failure_detail"]


def read_columns(path, required, optional):
    """Read existing Parquet columns, with an informative missing-column error."""
    available = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return pd.read_parquet(path, columns=required + [c for c in optional if c in available])


def build(production, failures=None, out_dir=OUTPUT):
    """Transform supplied dataframes; also usable for small synthetic tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = production.copy()
    df["uwi"] = df["uwi"].astype("string").str.strip()
    df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["uwi", "production_date"])
    df = df[df["uwi"].ne("")]
    # Only producing well-days: this dashboard is about produced oil, not injection.
    if "well_daily_status" in df.columns:
        df = df[df["well_daily_status"].astype("string").str.lower().eq("producing")]
    elif "flow_direction" in df.columns:
        df = df[df["flow_direction"].astype("string").str.lower().eq("production")]
    if df.empty:
        raise ValueError("No producing well-days found in the production data.")

    measurement_cols = [
        "oil_produced_m3", "water_produced_m3", "hours_down",
        "pcp_pump_rpm", "pump_efficiency_pct", "tubing_pressure_kpa", "casing_pressure_kpa",
    ]
    for col in measurement_cols:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["oil_produced_m3", "water_produced_m3"]:
        df.loc[df[col] < 0, col] = float("nan")
    df.loc[~df["hours_down"].between(0, 24), "hours_down"] = float("nan")
    for col in ["tubing_pressure_kpa", "casing_pressure_kpa"]:
        df.loc[(df[col] < 0) | (df[col] >= 999999), col] = float("nan")

    daily = df.groupby("production_date", as_index=False).agg(
        total_oil_m3=("oil_produced_m3", "sum"),
        total_water_m3=("water_produced_m3", "sum"),
        total_downtime_hours=("hours_down", "sum"),
        reporting_wells=("uwi", "nunique"),
    ).sort_values("production_date")
    daily.to_csv(out_dir / "dashboard_daily.csv", index=False)

    latest_date = df["production_date"].max()
    recent = df[df["production_date"] >= latest_date - pd.Timedelta(days=89)].copy()
    recent = recent[["uwi", "production_date"] + measurement_cols]
    recent = recent.sort_values(["uwi", "production_date"])
    recent.to_csv(out_dir / "dashboard_recent_well_daily.csv", index=False)

    if failures is not None and not failures.empty:
        f = failures.copy()
        f["uwi"] = f["uwi"].astype("string").str.strip()
        f["failure_start_date"] = pd.to_datetime(f["failure_start_date"], errors="coerce")
        for col in FAILURE_OPTIONAL:
            if col not in f.columns:
                f[col] = ""
        f[FAILURE_OPTIONAL].dropna(subset=["uwi", "failure_start_date"]).to_csv(
            out_dir / "dashboard_failures.csv", index=False
        )
    print(f"Saved {len(daily):,} daily summary rows")
    print(f"Saved {len(recent):,} recent well-day rows (ending {latest_date.date()})")
    print(f"Output folder: {out_dir}")
    return daily, recent


def main():
    path = OUTPUT / "cleaned_production.parquet"
    if not path.is_file():
        path = DATA / "daily_production.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing production Parquet. Put the original files in data/ or run "
            "python src/data_cleaning.py first."
        )
    print("Loading", path)
    production = read_columns(path, PRODUCTION_REQUIRED, PRODUCTION_OPTIONAL)
    failure_path = DATA / "production_failures.parquet"
    failures = (read_columns(failure_path, ["uwi", "failure_start_date"],
                             [c for c in FAILURE_OPTIONAL if c not in ["uwi", "failure_start_date"]])
                if failure_path.is_file() else None)
    build(production, failures)


if __name__ == "__main__":
    main()
