
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load relevant production measurements
production = pd.read_parquet(
    "data/daily_production.parquet",
    columns=[
        "uwi",
        "production_date",
        "pcp_pump_rpm",
        "pump_efficiency_pct",
        "oil_produced_m3"
    ]
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

# Select rod failures
rod_failures = failures[
    failures["failure_type"] == "Rods"
].dropna(subset=["uwi", "failure_start_date"]).copy()

# Keep production for affected wells
production = production[
    production["uwi"].isin(rod_failures["uwi"])
]

# Join records by well
merged = production.merge(
    rod_failures[["uwi", "failure_start_date"]],
    on="uwi",
    how="inner"
)

# Calculate number of days before each failure
merged["days_before"] = (
    merged["failure_start_date"] -
    merged["production_date"]
).dt.days

# Keep only the preceding 30 days
before_failure = merged[
    merged["days_before"].between(1, 30)
]

# Calculate daily averages across failure events
averages = before_failure.groupby("days_before")[
    ["pcp_pump_rpm", "pump_efficiency_pct", "oil_produced_m3"]
].mean().sort_index()

# Create output directory
Path("outputs").mkdir(exist_ok=True)

# Create separate plots
for column, title in [
    ("pcp_pump_rpm", "Pump RPM"),
    ("pump_efficiency_pct", "Pump Efficiency"),
    ("oil_produced_m3", "Oil Production")
]:
    plt.figure(figsize=(9, 5))
    plt.plot(
        averages.index,
        averages[column],
        marker="o"
    )
    plt.xlabel("Days Before Rod Failure")
    plt.ylabel(title)
    plt.title(f"{title} Before Rod Failures")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"outputs/{column}_before_failure.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print("Saved:", filename)

print("\nAll visualizations completed!")
