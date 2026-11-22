
# Data Analysis - Invisible Oil Loss Detector
# Examines cleaned oil production records, checks production quality,
# summarizes well activity, and creates a daily production trend chart.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "cleaned_production.parquet"
OUTPUT = ROOT / "outputs"


def main():
    print("Loading cleaned production data...")

    columns = [
        "uwi",
        "production_date",
        "oil_produced_m3",
        "hours_down",
        "hours_on_prod",
    ]

    df = pd.read_parquet(INPUT, columns=columns)

    print("\nDATASET OVERVIEW")
    print(f"Total records: {len(df):,}")
    print(f"Unique wells: {df['uwi'].nunique():,}")
    print(f"First date: {df['production_date'].min()}")
    print(f"Last date: {df['production_date'].max()}")

    oil = df["oil_produced_m3"]

    print("\nOIL PRODUCTION SUMMARY")
    print(f"Missing oil values: {oil.isna().sum():,}")
    print(f"Zero oil values: {(oil == 0).sum():,}")
    print(f"Negative oil values: {(oil < 0).sum():,}")
    print(f"Average daily record: {oil.mean():.2f} m3")
    print(f"Median daily record: {oil.median():.2f} m3")

    print("\nDOWNTIME SUMMARY")
    print(f"Records with downtime: {(df['hours_down'] > 0).sum():,}")
    print(f"Missing downtime values: {df['hours_down'].isna().sum():,}")
    print(f"Records with zero operating hours: {(df['hours_on_prod'] == 0).sum():,}")

    valid = df[
        df["production_date"].notna()
        & df["oil_produced_m3"].ge(0)
    ].copy()

    daily = (
        valid.groupby("production_date")
        .agg(
            total_oil_m3=("oil_produced_m3", "sum"),
            reporting_wells=("uwi", "nunique"),
        )
        .reset_index()
        .sort_values("production_date")
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    daily.to_csv(
        OUTPUT / "daily_production_summary.csv",
        index=False,
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        daily["production_date"],
        daily["total_oil_m3"],
    )
    plt.title("Total Daily Oil Production")
    plt.xlabel("Production Date")
    plt.ylabel("Oil Produced (m3)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "daily_production_trend.png",
        dpi=150,
    )
    plt.close()

    print("\nANALYSIS COMPLETED")
    print("Saved: outputs/daily_production_summary.csv")
    print("Saved: outputs/daily_production_trend.png")


if __name__ == "__main__":
    main()
