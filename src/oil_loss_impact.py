
# Invisible Oil Loss Detector - Oil Loss Impact Analysis
# Estimates production shortfall for flagged anomaly records.
# These are estimated shortfalls, not confirmed recoverable losses.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "production_features.parquet"
OUTPUT = ROOT / "outputs" / "oil_loss_impact_summary.csv"

BARRELS_PER_M3 = 6.28981


def main():
    print("Loading production features...")

    columns = [
        "uwi",
        "production_date",
        "possible_invisible_loss",
        "potential_oil_loss_m3",
    ]

    df = pd.read_parquet(INPUT, columns=columns)

    # Keep only the records flagged by feature engineering.
    flagged = df[
        df["possible_invisible_loss"].fillna(False)
    ].copy()

    print(f"Potential anomaly records: {len(flagged):,}")
    print(f"Affected wells: {flagged['uwi'].nunique():,}")

    # Remove invalid or negative shortfall estimates.
    flagged["potential_oil_loss_m3"] = pd.to_numeric(
        flagged["potential_oil_loss_m3"],
        errors="coerce",
    )

    valid = flagged[
        flagged["potential_oil_loss_m3"].notna()
        & (flagged["potential_oil_loss_m3"] > 0)
    ].copy()

    print(f"Records with positive shortfall: {len(valid):,}")

    # Calculate estimated shortfall volume.
    total_m3 = valid["potential_oil_loss_m3"].sum()
    total_barrels = total_m3 * BARRELS_PER_M3

    print("\nESTIMATED PRODUCTION SHORTFALL")
    print(f"Total cubic metres: {total_m3:,.2f} m3")
    print(f"Total barrel equivalent: {total_barrels:,.2f} bbl")

    # Summarize by well, without treating each day
    # as an independent incident.
    well_summary = (
        valid.groupby("uwi", dropna=False)
        .agg(
            anomaly_days=("production_date", "nunique"),
            estimated_shortfall_m3=(
                "potential_oil_loss_m3", "sum"
            ),
        )
        .reset_index()
    )

    well_summary["estimated_shortfall_bbl"] = (
        well_summary["estimated_shortfall_m3"]
        * BARRELS_PER_M3
    )

    well_summary = well_summary.sort_values(
        "estimated_shortfall_m3",
        ascending=False,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    well_summary.to_csv(OUTPUT, index=False)

    print("\nTOP 10 WELLS BY ESTIMATED SHORTFALL")
    print(well_summary.head(10).to_string(index=False))

    print(f"\nSaved: {OUTPUT}")
    print("Oil loss impact analysis completed.")
    print(
        "Important: Estimated shortfall is not proof "
        "of recoverable oil or financial loss."
    )


if __name__ == "__main__":
    main()
