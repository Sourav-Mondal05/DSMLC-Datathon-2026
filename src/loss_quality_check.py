
# Invisible Oil Loss Detector
# Step 13: Check the quality of estimated production losses.
# This script does not modify the original datasets.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "production_features.parquet"
OUTPUT = ROOT / "outputs" / "loss_quality_summary.csv"


def main():
    print("Loading engineered production features...")

    columns = [
        "uwi",
        "production_date",
        "valid_oil_m3",
        "oil_baseline_7d",
        "oil_decline_pct",
        "potential_oil_loss_m3",
        "has_downtime",
        "possible_invisible_loss",
    ]

    df = pd.read_parquet(INPUT, columns=columns)
    candidates = df.loc[df["possible_invisible_loss"].fillna(False)].copy()

    print(f"Total candidate records: {len(candidates):,}")
    print(f"Affected wells: {candidates['uwi'].nunique():,}")

    # Check whether candidates have a usable production baseline.
    checks = {
        "missing_baseline": candidates["oil_baseline_7d"].isna(),
        "nonpositive_baseline": candidates["oil_baseline_7d"].le(0),
        "missing_actual_oil": candidates["valid_oil_m3"].isna(),
        "downtime_flagged": candidates["has_downtime"].fillna(False),
        "missing_shortfall": candidates["potential_oil_loss_m3"].isna(),
        "nonpositive_shortfall": candidates["potential_oil_loss_m3"].le(0),
    }

    print("\nQUALITY CHECKS")
    results = []

    for name, condition in checks.items():
        count = int(condition.sum())
        print(f"{name}: {count:,}")
        results.append({"check": name, "records": count})

    # Examine the distribution of estimated daily shortfalls.
    losses = candidates["potential_oil_loss_m3"].dropna()

    print("\nESTIMATED DAILY SHORTFALL DISTRIBUTION")

    if not losses.empty:
        for percentile in [0.50, 0.90, 0.95, 0.99]:
            value = losses.quantile(percentile)
            print(f"{percentile:.0%} percentile: {value:,.3f} m3")

        print(f"Maximum: {losses.max():,.3f} m3")
        print(f"Total estimated shortfall: {losses.sum():,.2f} m3")

    # Check concentration among the highest-shortfall wells.
    well_totals = (
        candidates.groupby("uwi")["potential_oil_loss_m3"]
        .sum()
        .sort_values(ascending=False)
    )

    total = well_totals.sum()
    top_10 = well_totals.head(10).sum()

    print("\nCONCENTRATION CHECK")
    if total > 0:
        print(f"Top 10 wells' share: {top_10 / total:.2%}")

    # Save a compact report.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print("Loss quality check completed.")
    print("Note: Candidate losses are not confirmed recoverable oil.")


if __name__ == "__main__":
    main()
