
# DSMLC Datathon 2026
# Initial Dataset Inspection

import pandas as pd
from pathlib import Path

# Locate the folder containing the datasets
data_folder = Path(__file__).resolve().parent / "data"

# Official competition datasets
files = [
    "daily_production.parquet",
    "production_failures.parquet",
    "well_jobs.parquet"
]

# Inspect each dataset
for filename in files:

    print("\n" + "=" * 60)
    print("DATASET:", filename)
    print("=" * 60)

    file_path = data_folder / filename

    # Check whether the file exists
    if not file_path.exists():
        print("ERROR: File not found:", file_path)
        continue

    # Read the Parquet file
    df = pd.read_parquet(file_path)

    # Dataset dimensions
    print("\nTotal rows:", df.shape[0])
    print("Total columns:", df.shape[1])

    # Display all column names
    print("\nCOLUMN NAMES:")
    for column in df.columns:
        print("-", column)

    # Display the first 5 records
    print("\nFIRST 5 ROWS:")
    print(df.head().to_string(index=False))

    # Display column data types
    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    # Count missing values
    print("\nTOP 10 COLUMNS WITH MISSING VALUES:")
    missing = df.isnull().sum()
    print(missing.sort_values(ascending=False).head(10))

    # Count unique wells
    if "uwi" in df.columns:
        print("\nTOTAL UNIQUE WELLS:")
        print(df["uwi"].nunique())

    print("\nFinished inspecting:", filename)

print("\n" + "=" * 60)
print("ALL DATASET INSPECTIONS COMPLETED!")
print("=" * 60)
