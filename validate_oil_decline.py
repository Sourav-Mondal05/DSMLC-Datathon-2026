
import pandas as pd
import numpy as np

# ==========================================
# DSMLC Datathon 2026
# Validate oil production decline
# ==========================================

print("Loading datasets...")

production = pd.read_parquet("data/daily_production.parquet")
failures = pd.read_parquet("data/production_failures.parquet")

production["production_date"] = pd.to_datetime(
    production["production_date"], errors="coerce"
).dt.normalize()

failures["failure_start_date"] = pd.to_datetime(
    failures["failure_start_date"], errors="coerce"
).dt.normalize()

# Focus on rod failures
rod_failures = failures[
    failures["failure_type"].astype(str).str.strip().str.lower() == "rods"
].copy()

print("\nTotal rod failure events:", len(rod_failures))

# Daily production per well
daily = (
    production.groupby(["uwi", "production_date"], as_index=False)
    ["oil_produced_m3"].sum(min_count=1)
)

# ==========================================
# Compare baseline against pre-failure days
# ==========================================

results = []

for event in rod_failures.itertuples(index=False):
    well = event.uwi
    failure_date = event.failure_start_date

    if pd.isna(failure_date):
        continue

    history = daily[
        (daily["uwi"] == well)
        & (daily["production_date"] < failure_date)
        & (
            daily["production_date"]
            >= failure_date - pd.Timedelta(days=30)
        )
    ].copy()

    history["days_before"] = (
        failure_date - history["production_date"]
    ).dt.days

    baseline = history[
        history["days_before"].between(15, 30)
    ]["oil_produced_m3"].dropna()

    recent = history[
        history["days_before"].between(1, 3)
    ]["oil_produced_m3"].dropna()

    # Require sufficient observations and positive baseline
    if len(baseline) < 10 or len(recent) < 2:
        continue

    baseline_mean = baseline.mean()
    recent_mean = recent.mean()

    if baseline_mean <= 0:
        continue

    decline_pct = (
        (baseline_mean - recent_mean) / baseline_mean
    ) * 100

    results.append({
        "uwi": well,
        "failure_date": failure_date,
        "baseline_oil": baseline_mean,
        "recent_oil": recent_mean,
        "decline_pct": decline_pct
    })

results = pd.DataFrame(results)

print("\nRod failures with sufficient production history:", len(results))

if not results.empty:
    print("\nOil production decline statistics:")
    print(results["decline_pct"].describe().round(2))

    for threshold in [10, 20, 30, 50]:
        count = (results["decline_pct"] >= threshold).sum()
        percentage = count / len(results) * 100

        print(
            f"Events with >= {threshold}% decline: "
            f"{count} ({percentage:.2f}%)"
        )

    print(
        "\nUnique wells analyzed:",
        results["uwi"].nunique()
    )

    results.to_csv(
        "outputs/oil_decline_validation.csv",
        index=False
    )

    print("\nSaved: outputs/oil_decline_validation.csv")

else:
    print("\nNo events met the validation requirements.")

print("\nValidation completed!")
