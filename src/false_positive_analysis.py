
# Invisible Oil Loss Detector
# Step 16: Investigate possible false-positive alerts.
# A quick recovery is a warning sign, not proof of a false positive.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "production_features.parquet"
OUTPUT = ROOT / "outputs" / "false_positive_summary.csv"


def main():
    print("Loading production features...")

    columns = [
        "uwi",
        "production_date",
        "valid_oil_m3",
        "oil_baseline_7d",
        "possible_invisible_loss",
    ]

    df = pd.read_parquet(INPUT, columns=columns)
    df["production_date"] = pd.to_datetime(df["production_date"])
    df = df.sort_values(["uwi", "production_date"]).copy()

    # Compare each alert against the next actual calendar day.
    # Do not accidentally treat a date gap as a one-day recovery.
    grouped = df.groupby("uwi", sort=False)

    df["next_date"] = grouped["production_date"].shift(-1)
    df["next_oil"] = grouped["valid_oil_m3"].shift(-1)

    candidates = df.loc[
        df["possible_invisible_loss"].fillna(False)
    ].copy()

    candidates["has_next_day"] = (
        candidates["next_date"]
        == candidates["production_date"] + pd.Timedelta(days=1)
    )

    # Recovery means next-day oil production reaches at least
    # 90% of the pre-alert baseline.
    candidates["quick_recovery"] = (
        candidates["has_next_day"]
        & candidates["next_oil"].notna()
        & (candidates["oil_baseline_7d"] > 0)
        & (
            candidates["next_oil"]
            >= 0.90 * candidates["oil_baseline_7d"]
        )
    )

    total = len(candidates)
    next_day = int(candidates["has_next_day"].sum())
    recovered = int(candidates["quick_recovery"].sum())

    print("\nPOSSIBLE FALSE-POSITIVE ANALYSIS")
    print(f"Total candidate alerts: {total:,}")
    print(f"Alerts with next-day records: {next_day:,}")
    print(f"Quick recovery alerts: {recovered:,}")

    if next_day:
        print(
            "Quick recovery among alerts with next-day records: "
            f"{recovered / next_day:.2%}"
        )

    print("\nIMPORTANT")
    print("Quick recovery is only a screening indicator.")
    print("It does not establish a measured false-positive rate.")

    summary = pd.DataFrame(
        [
            {"metric": "total_candidates", "value": total},
            {"metric": "next_day_records", "value": next_day},
            {"metric": "quick_recovery_alerts", "value": recovered},
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print("False-positive screening completed.")


if __name__ == "__main__":
    main()
