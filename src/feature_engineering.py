
# Member 1 - Data Engineering
# Invisible Oil Loss Detector
#
# Creates production features, maintenance indicators,
# and future failure labels for the ML team.

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

INPUT = OUTPUTS / "cleaned_production.parquet"
OUTPUT = OUTPUTS / "model_features.parquet"

CUTOFF = pd.Timestamp("2026-10-26")
FAILURE_WINDOW = 7


def clean_dates(series):
    """Convert dates to a consistent daily resolution."""
    return pd.to_datetime(
        series, errors="coerce"
    ).dt.normalize().astype("datetime64[ns]")


def clean_uwi(series):
    """Standardize well identifiers."""
    return series.astype("string").str.strip()


def main():

    # ---------------------------------------
    # 1. Load datasets
    # ---------------------------------------

    print("Loading datasets...")

    df = pd.read_parquet(INPUT)
    jobs = pd.read_parquet(DATA / "well_jobs.parquet")
    failures = pd.read_parquet(
        DATA / "production_failures.parquet"
    )

    for dataset in (df, jobs, failures):
        dataset["uwi"] = clean_uwi(dataset["uwi"])

    df["production_date"] = clean_dates(
        df["production_date"]
    )

    jobs["start_date"] = clean_dates(jobs["start_date"])
    jobs["end_date"] = clean_dates(jobs["end_date"])

    failures["failure_start_date"] = clean_dates(
        failures["failure_start_date"]
    )

    df = df[
        df["production_date"].notna()
        & (df["production_date"] < CUTOFF)
    ].copy()

    df = df.sort_values(
        ["uwi", "production_date"]
    ).reset_index(drop=True)

    print(f"Production records: {len(df):,}")

    # ---------------------------------------
    # 2. Historical production features
    # ---------------------------------------

    print("Creating production features...")

    previous_oil = (
        df.groupby("uwi")["oil_produced_m3"].shift(1)
    )

    df["oil_previous_m3"] = previous_oil

    df["oil_baseline_7d"] = (
        previous_oil.groupby(df["uwi"])
        .transform(
            lambda x: x.rolling(
                window=7, min_periods=5
            ).mean()
        )
    )

    df["oil_change_pct"] = np.where(
        previous_oil > 0,
        100 * (
            df["oil_produced_m3"] - previous_oil
        ) / previous_oil,
        np.nan,
    )

    df["oil_decline_pct"] = np.where(
        df["oil_baseline_7d"] > 0,
        100 * (
            df["oil_baseline_7d"]
            - df["oil_produced_m3"]
        ) / df["oil_baseline_7d"],
        np.nan,
    )

    df["potential_oil_loss_m3"] = (
        df["oil_baseline_7d"]
        - df["oil_produced_m3"]
    ).clip(lower=0)

    # ---------------------------------------
    # 3. Equipment features
    # ---------------------------------------

    total_liquid = (
        df["oil_produced_m3"]
        + df["water_produced_m3"]
    )

    df["water_cut_pct"] = np.where(
        total_liquid > 0,
        100 * df["water_produced_m3"] / total_liquid,
        np.nan,
    )

    for column in [
        "pcp_pump_rpm",
        "pump_efficiency_pct",
    ]:
        if column in df.columns:
            df[f"{column}_change"] = (
                df.groupby("uwi")[column].diff()
            )

    df["has_downtime"] = (
        df["hours_down"].fillna(0) > 0
    )

    # Candidate anomaly, not confirmed oil loss.
    df["possible_invisible_loss"] = (
        (df["oil_baseline_7d"] >= 1)
        & (df["oil_decline_pct"] >= 30)
        & (df["hours_down"] == 0)
        & (df["hours_on_prod"] > 0)
    )

    # ---------------------------------------
    # 4. Connect maintenance records
    # ---------------------------------------

    print("Connecting maintenance records...")

    jobs = jobs.dropna(
        subset=["uwi", "start_date"]
    ).copy()

    jobs["end_date"] = jobs["end_date"].fillna(
        jobs["start_date"]
    )

    # Keep valid maintenance intervals.
    jobs = jobs[
        (jobs["end_date"] >= jobs["start_date"])
        & (
            (jobs["end_date"] - jobs["start_date"]).dt.days
            <= 365
        )
    ].copy()

    # Convert maintenance intervals to daily records.
    maintenance = jobs[
        ["uwi", "start_date", "end_date"]
    ].copy()

    maintenance["production_date"] = maintenance.apply(
        lambda row: pd.date_range(
            row["start_date"], row["end_date"]
        ),
        axis=1,
    )

    maintenance = maintenance[
        ["uwi", "production_date"]
    ].explode("production_date")

    maintenance["production_date"] = clean_dates(
        maintenance["production_date"]
    )

    maintenance = maintenance.drop_duplicates()
    maintenance["maintenance_on_day"] = True

    df = df.merge(
        maintenance,
        on=["uwi", "production_date"],
        how="left",
        validate="many_to_one",
    )

    df["maintenance_on_day"] = (
        df["maintenance_on_day"]
        .fillna(False)
        .astype(bool)
    )

    # ---------------------------------------
    # 5. Future failure labels
    # ---------------------------------------

    print("Creating future failure labels...")

    failures = failures.dropna(
        subset=["uwi", "failure_start_date"]
    ).copy()

    failure_events = (
        failures[
            ["uwi", "failure_start_date"]
        ]
        .drop_duplicates()
        .sort_values(
            ["uwi", "failure_start_date"]
        )
    )

    # Assign labels one well at a time.
    # Avoids pandas merge_asof dtype errors.
    df["failure_next_7d"] = np.int8(0)

    failure_groups = {
        uwi: group["failure_start_date"].to_numpy(
            dtype="datetime64[ns]"
        )
        for uwi, group in failure_events.groupby("uwi")
    }

    for uwi, indices in df.groupby("uwi").groups.items():

        event_dates = failure_groups.get(uwi)

        if event_dates is None or len(event_dates) == 0:
            continue

        observation_dates = df.loc[
            indices, "production_date"
        ].to_numpy(dtype="datetime64[ns]")

        # Find first failure strictly after each date.
        positions = np.searchsorted(
            event_dates,
            observation_dates,
            side="right",
        )

        valid = positions < len(event_dates)

        labels = np.zeros(
            len(indices), dtype=np.int8
        )

        if valid.any():
            next_dates = event_dates[positions[valid]]

            days_until = (
                next_dates - observation_dates[valid]
            ) / np.timedelta64(1, "D")

            labels[valid] = (
                (days_until >= 1)
                & (days_until <= FAILURE_WINDOW)
            ).astype(np.int8)

        df.loc[indices, "failure_next_7d"] = labels

    # The final seven days cannot be fully labeled
    # within the selected observation period.
    observation_end = CUTOFF - pd.Timedelta(days=1)

    df["failure_label_observable"] = (
        df["production_date"]
        + pd.Timedelta(days=FAILURE_WINDOW)
        <= observation_end
    )

    # ---------------------------------------
    # 6. Export model-ready data
    # ---------------------------------------

    print("Saving final dataset...")

    df = df.sort_values(
        ["uwi", "production_date"]
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_parquet(OUTPUT, index=False)

    print()
    print("DATA ENGINEERING SUMMARY")
    print("------------------------")
    print(f"Rows exported: {len(df):,}")
    print(
        "Maintenance-day records:",
        f"{df['maintenance_on_day'].sum():,}"
    )
    print(
        "Future failure labels:",
        f"{df['failure_next_7d'].sum():,}"
    )
    print(
        "Invisible-loss candidates:",
        f"{df['possible_invisible_loss'].sum():,}"
    )
    print(f"Saved: {OUTPUT}")
    print("Data engineering completed successfully.")


if __name__ == "__main__":
    main()
