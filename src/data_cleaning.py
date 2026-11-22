
# Data Cleaning - Invisible Oil Loss Detector
# Cleans daily production records, preserves original production values,
# flags invalid measurements, and exports a model-ready dataset.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "daily_production.parquet"
OUTPUT = ROOT / "outputs" / "cleaned_production.parquet"

TOLERANCE = 0.001


def main():
    print("Loading production data...")
    df = pd.read_parquet(INPUT)
    print(f"Original rows: {len(df):,}")

    df["uwi"] = df["uwi"].astype("string").str.strip()
    df["production_date"] = pd.to_datetime(
        df["production_date"], errors="coerce"
    )

    numeric_columns = [
        "oil_produced_m3",
        "water_produced_m3",
        "lost_oil_m3",
        "hours_down",
        "hours_on_prod",
        "pcp_pump_rpm",
        "pump_efficiency_pct",
        "tubing_pressure_kpa",
        "casing_pressure_kpa",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    duplicates = df.duplicated().sum()
    df = df.drop_duplicates().copy()

    invalid_dates = df["production_date"].isna().sum()
    invalid_uwi = (
        df["uwi"].isna() | df["uwi"].eq("")
    ).sum()

    df = df.loc[
        df["production_date"].notna()
        & df["uwi"].notna()
        & df["uwi"].ne("")
    ].copy()

    print(f"Exact duplicates removed: {duplicates:,}")
    print(f"Invalid dates removed: {invalid_dates:,}")
    print(f"Invalid well identifiers removed: {invalid_uwi:,}")

    for column in ["oil_produced_m3", "water_produced_m3"]:
        if column in df.columns:
            original = df[column].copy()

            tiny_negative = (original < 0) & (original >= -TOLERANCE)
            large_negative = original < -TOLERANCE

            df[f"{column}_original"] = original
            df[f"{column}_tiny_negative_flag"] = tiny_negative
            df[f"{column}_invalid_flag"] = large_negative

            df.loc[tiny_negative, column] = 0.0
            df.loc[large_negative, column] = float("nan")

            print(f"\n{column}")
            print(f"Tiny negatives corrected: {tiny_negative.sum():,}")
            print(f"Larger negatives flagged: {large_negative.sum():,}")

    for column in ["hours_down", "hours_on_prod"]:
        if column in df.columns:
            invalid_hours = (
                (df[column] < 0) | (df[column] > 24)
            )

            df[f"{column}_invalid_flag"] = invalid_hours
            df.loc[invalid_hours, column] = float("nan")

            print(f"Invalid {column}: {invalid_hours.sum():,}")

    df = df.sort_values(
        ["uwi", "production_date"]
    ).reset_index(drop=True)

    duplicate_well_dates = df.duplicated(
        ["uwi", "production_date"]
    ).sum()

    print(f"Duplicate well-date records: {duplicate_well_dates:,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"\nCleaned rows: {len(df):,}")
    print(f"Saved: {OUTPUT}")
    print("Production data cleaning completed.")


if __name__ == "__main__":
    main()
