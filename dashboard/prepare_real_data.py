"""
REAL DATA EXPORT FOR THE INVISIBLE OIL LOSS DETECTOR

What this file does
-------------------
1. Reads Member 1's actual cleaned production Parquet file in manageable
   batches. It NEVER generates synthetic wells or sample measurements.
2. Creates daily field/area production totals, summaries for each real well,
   and recent well sensor histories for Streamlit charts.
3. Reads the original failure and maintenance logs to provide real event
   counts, engineering causes, and historical job summaries.
4. Writes private CSV/JSON files to outputs/; the dashboard reads those files
   quickly without reopening the five-million-row Parquet on every rerun.
5. Does not write any files to GitHub or the internet. Raw and derived
   well-level data are restricted by the competition data-use terms.
6. Sanitizes invalid numeric readings (e.g. 999999 pressure placeholders),
   drops rows with missing well IDs or dates, and reports the data range.
7. Does not infer equipment failures, hidden losses, or financial savings.
   Failure-risk scores come from Member 2's separate ML model.

Required input: outputs/cleaned_production.parquet
Optional input: data/production_failures.parquet, data/well_jobs.parquet

Run from repository root:
    python src/data_cleaning.py
    python dashboard/prepare_real_data.py

Memory note: Parquet is processed in batches of 150,000 records. The final
recent-well CSV is restricted to the most recent 120 calendar days.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
CLEANED = OUT / "cleaned_production.parquet"

RECENT_DAYS = 120
BATCH_SIZE = 150_000

MEASURES = [
    "oil_produced_m3", "water_produced_m3", "hours_down",
    "hours_on_prod", "pcp_pump_rpm", "pump_efficiency_pct",
    "tubing_pressure_kpa", "casing_pressure_kpa",
]
BASE = ["uwi", "production_date", "field_operating_area", "province", "well_daily_status"]


def normalize(batch: pd.DataFrame) -> pd.DataFrame:
    """Standardize one true production batch without manufacturing values."""
    batch = batch.copy()
    batch["uwi"] = batch["uwi"].astype("string").str.strip()
    batch["production_date"] = pd.to_datetime(batch["production_date"], errors="coerce").dt.normalize()
    batch = batch.dropna(subset=["uwi", "production_date"])
    batch = batch.loc[batch["uwi"].ne("")].copy()
    batch = batch.drop_duplicates()
    for key in MEASURES:
        if key not in batch:
            batch[key] = np.nan
        batch[key] = pd.to_numeric(batch[key], errors="coerce")
    for key in ["oil_produced_m3", "water_produced_m3", "pcp_pump_rpm"]:
        batch.loc[batch[key] < 0, key] = np.nan
    for key in ["hours_down", "hours_on_prod"]:
        batch.loc[~batch[key].between(0, 24), key] = np.nan
    for key in ["tubing_pressure_kpa", "casing_pressure_kpa"]:
        batch.loc[(batch[key] < 0) | (batch[key] >= 999999), key] = np.nan
    for key in ["field_operating_area", "province", "well_daily_status"]:
        if key not in batch:
            batch[key] = "Unknown"
        else:
            batch[key] = batch[key].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    return batch


def parse_parquet(path: Path, columns: list[str]):
    """Yield real records in small groups instead of loading the whole file."""
    source = pq.ParquetFile(path)
    available = set(source.schema.names)
    required = {"uwi", "production_date"}
    missing = required - available
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
    selected = [c for c in columns if c in available]
    for record_batch in source.iter_batches(batch_size=BATCH_SIZE, columns=selected):
        yield record_batch.to_pandas()


def build_daily(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    sums = ["oil_m3", "water_m3", "downtime_hours", "producing_well_rows", "reporting_well_rows"]
    return (combined.groupby(keys, as_index=False, dropna=False)[sums]
            .sum(min_count=1).sort_values(keys))


def main():
    if not CLEANED.is_file():
        raise FileNotFoundError(
            f"{CLEANED} is missing. Run: python src/data_cleaning.py\n"
            "Do NOT upload your original Parquet files to public GitHub."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    print("Scanning real production dates...")
    smallest, largest = None, None
    total_input = 0
    for part in parse_parquet(CLEANED, ["uwi", "production_date"]):
        days = pd.to_datetime(part["production_date"], errors="coerce")
        if days.notna().any():
            lo, hi = days.min(), days.max()
            smallest = min(smallest, lo) if smallest is not None else lo
            largest = max(largest, hi) if largest is not None else hi
        total_input += len(part)
    if largest is None:
        raise ValueError("Production file contains no valid dates.")
    recent_start = largest.normalize() - pd.Timedelta(days=RECENT_DAYS - 1)
    print(f"Records: {total_input:,}; latest: {largest.date()}; recent from {recent_start.date()}")

    daily_parts, area_parts, well_parts, recent_parts = [], [], [], []
    use_cols = BASE + MEASURES
    for n, raw in enumerate(parse_parquet(CLEANED, use_cols), start=1):
        b = normalize(raw)
        if b.empty:
            continue
        b["producing_flag"] = b["well_daily_status"].eq("Producing").astype(int)
        b["reporting_flag"] = 1
        sum_fields = {
            "oil_produced_m3": "oil_m3", "water_produced_m3": "water_m3",
            "hours_down": "downtime_hours", "producing_flag": "producing_well_rows",
            "reporting_flag": "reporting_well_rows"
        }
        renamed = b.rename(columns=sum_fields)
        agg_fields = list(sum_fields.values())
        daily_parts.append(renamed.groupby("production_date", as_index=False)[agg_fields].sum(min_count=1))
        area_parts.append(renamed.groupby(["production_date", "field_operating_area"], as_index=False)[agg_fields].sum(min_count=1))
        grouped = b.groupby("uwi", as_index=False).agg(
            first_date=("production_date", "min"), last_date=("production_date", "max"),
            oil_m3=("oil_produced_m3", "sum"), water_m3=("water_produced_m3", "sum"),
            downtime_hours=("hours_down", "sum"), observations=("production_date", "size")
        )
        well_parts.append(grouped)
        recent = b[b["production_date"] >= recent_start]
        if not recent.empty:
            recent_parts.append(recent[["uwi", "production_date", "field_operating_area", "well_daily_status"] + MEASURES].copy())
        if n % 10 == 0:
            print(f"  Processed ~{n * BATCH_SIZE:,} production rows")

    daily = build_daily(daily_parts, ["production_date"])
    area_daily = build_daily(area_parts, ["production_date", "field_operating_area"])
    wells = pd.concat(well_parts, ignore_index=True).groupby("uwi", as_index=False).agg(
        first_date=("first_date", "min"), last_date=("last_date", "max"),
        oil_m3=("oil_m3", "sum"), water_m3=("water_m3", "sum"),
        downtime_hours=("downtime_hours", "sum"), observations=("observations", "sum")
    )
    recent = pd.concat(recent_parts, ignore_index=True) if recent_parts else pd.DataFrame()
    if not recent.empty:
        recent = (recent.sort_values(["uwi", "production_date"])
                  .drop_duplicates(["uwi", "production_date"], keep="last"))
        latest_meta = (recent.sort_values("production_date")
                       .drop_duplicates("uwi", keep="last")
                       [["uwi", "field_operating_area", "well_daily_status"]])
        wells = wells.merge(latest_meta, on="uwi", how="left")
    else:
        wells["field_operating_area"] = "Unknown"
        wells["well_daily_status"] = "Unknown"

    daily.to_csv(OUT / "real_daily.csv", index=False)
    area_daily.to_csv(OUT / "real_area_daily.csv", index=False)
    wells.to_csv(OUT / "real_wells.csv", index=False)
    recent.to_csv(OUT / "real_well_history.csv", index=False)

    failure_count = None
    path_failures = DATA / "production_failures.parquet"
    if path_failures.is_file():
        fails = pd.read_parquet(path_failures)
        wanted = [c for c in ["uwi", "failure_start_date", "failure_type", "cause", "field_operating_area"] if c in fails]
        fails = fails[wanted].copy()
        fails["failure_start_date"] = pd.to_datetime(fails["failure_start_date"], errors="coerce").dt.normalize()
        fails["uwi"] = fails["uwi"].astype("string").str.strip()
        fails = fails.dropna(subset=["uwi", "failure_start_date"])
        failure_count = int(len(fails))
        fails.to_csv(OUT / "real_failures.csv", index=False)
    else:
        print("Optional failure log absent: failure chart will show a missing-file message.")

    path_jobs = DATA / "well_jobs.parquet"
    if path_jobs.is_file():
        jobs = pd.read_parquet(path_jobs, columns=["uwi", "start_date", "field_est_total", "job_category"])
        jobs["start_date"] = pd.to_datetime(jobs["start_date"], errors="coerce")
        jobs["field_est_total"] = pd.to_numeric(jobs["field_est_total"], errors="coerce")
        jobs["uwi"] = jobs["uwi"].astype("string").str.strip()
        summary = jobs.groupby("uwi", as_index=False).agg(
            jobs_logged=("start_date", "size"), last_job_date=("start_date", "max"),
            median_estimated_job_cost=("field_est_total", "median"))
        summary.to_csv(OUT / "real_jobs.csv", index=False)
    else:
        print("Optional job log absent: maintenance history will be omitted.")

    meta = {
        "source": "competition historical data — NOT synthetic or live",
        "first_date": str(smallest.date()), "last_date": str(largest.date()),
        "source_production_rows": total_input, "wells_in_production_file": int(wells["uwi"].nunique()),
        "recorded_failures": failure_count, "recent_history_start": str(recent_start.date()),
        "limitations": [
            "These are historical records, not live sensor readings.",
            "Summed daily counts describe reporting rows, not necessarily distinct producing assets.",
            "Outputs contain restricted well identifiers and must not be published without permission."
        ]
    }
    (OUT / "real_data_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE: {len(wells):,} real well IDs; {len(daily):,} dates; {failure_count} recorded failures")
    print("Private outputs written to:", OUT)


if __name__ == "__main__":
    main()
