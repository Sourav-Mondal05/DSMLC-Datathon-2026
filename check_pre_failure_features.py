
import pandas as pd

features = [
    "hours_down",
    "oil_produced_m3",
    "casing_pressure_kpa",
    "tubing_pressure_kpa",
    "pcp_pump_rpm",
    "pump_efficiency_pct"
]

production = pd.read_parquet(
    "data/daily_production.parquet",
    columns=["uwi", "production_date"] + features
)

failures = pd.read_parquet(
    "data/production_failures.parquet",
    columns=["uwi", "failure_type", "failure_start_date"]
)

production["production_date"] = pd.to_datetime(
    production["production_date"], errors="coerce"
)

failures["failure_start_date"] = pd.to_datetime(
    failures["failure_start_date"], errors="coerce"
)

rod_failures = failures[
    failures["failure_type"] == "Rods"
].dropna(subset=["uwi", "failure_start_date"]).copy()

rod_failures = rod_failures.reset_index(drop=True)
rod_failures["event_id"] = rod_failures.index

# Keep only production records for wells with rod failures
production = production[
    production["uwi"].isin(rod_failures["uwi"])
].copy()

# Match each production date with subsequent failure events
merged = production.merge(
    rod_failures[["event_id", "uwi", "failure_start_date"]],
    on="uwi",
    how="inner"
)

# Use only the 30 days BEFORE each failure
days_before = (
    merged["failure_start_date"] -
    merged["production_date"]
).dt.days

pre_failure = merged[
    days_before.between(1, 30)
].copy()

print("Total rod failure events:", len(rod_failures))

print(
    "Events with 30-day production history:",
    pre_failure["event_id"].nunique()
)

print("\nFeature availability before rod failures:")
print(
    (pre_failure[features].notna().mean() * 100)
    .round(2)
    .to_string()
)

print("\nProduction observations per failure event:")
print(
    pre_failure.groupby("event_id")
    .size()
    .describe()
)

print("\nAnalysis completed!")
