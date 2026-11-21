
import pandas as pd

# Load only the columns needed for this check
production = pd.read_parquet(
    "data/daily_production.parquet",
    columns=["uwi", "production_date"]
)

failures = pd.read_parquet(
    "data/production_failures.parquet",
    columns=["uwi", "failure_type", "failure_start_date"]
)

# Convert date columns
production["production_date"] = pd.to_datetime(
    production["production_date"]
)

failures["failure_start_date"] = pd.to_datetime(
    failures["failure_start_date"]
)

# Select recorded rod failures
rod_failures = failures[
    failures["failure_type"] == "Rods"
].copy()

# Sort production dates and failure dates
production = production.sort_values(
    ["production_date", "uwi"]
)

rod_failures = rod_failures.sort_values(
    ["failure_start_date", "uwi"]
)

# Match each failure with the most recent production
# record from the same well, strictly before the failure
matched = pd.merge_asof(
    rod_failures,
    production,
    left_on="failure_start_date",
    right_on="production_date",
    by="uwi",
    direction="backward",
    allow_exact_matches=False
)

# Calculate days between last production and failure
matched["days_before_failure"] = (
    matched["failure_start_date"] -
    matched["production_date"]
).dt.days

print("Total rod failure events:", len(matched))

print(
    "Failures with earlier production records:",
    matched["production_date"].notna().sum()
)

print(
    "Failures with production within 30 days:",
    matched["days_before_failure"].between(0, 30).sum()
)

print(
    "Failures with production within 7 days:",
    matched["days_before_failure"].between(0, 7).sum()
)

print("\nDays since last production:")
print(matched["days_before_failure"].describe())

print("\nAnalysis completed!")
