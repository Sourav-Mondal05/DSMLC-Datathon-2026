# Invisible Oil Loss Detector — Real-Data Streamlit Dashboard

**This version does not generate demo data.** It displays real historical production, failure, and maintenance information from the competition datasets and real ML scores from Member 2.

## What is shown

1. **Overview:** Actual number of well IDs, historical oil production and downtime totals, production/downtime time series, and recorded failure types.
2. **Failure Risk:** Member 2's actual seven-day relative risk scores, ranked across wells.
3. **Maintenance:** Actual computed inspection-priority ranking with an explicitly hypothetical loss-exposure scenario.
4. **Well Explorer:** Real sensor readings and production histories for an individual well, plus recorded failure and job history.
5. **Model Results:** The logged test ROC-AUC/PR-AUC, baseline comparison, and feature importance.

## Required folder locations

```text
DSMLC-Datathon-2026/
├── dashboard/
│   ├── app.py
│   ├── prepare_real_data.py
│   ├── requirements.txt
│   └── .streamlit/config.toml
├── src/                      # existing Member 1 and Member 2 scripts
├── data/                     # LOCAL ONLY; original Parquet files
│   ├── daily_production.parquet
│   ├── production_failures.parquet
│   └── well_jobs.parquet
├── outputs/                  # LOCAL ONLY; generated data
│   ├── cleaned_production.parquet
│   ├── real_daily.csv
│   ├── real_area_daily.csv
│   ├── real_wells.csv
│   ├── real_well_history.csv
│   ├── real_failures.csv
│   ├── real_jobs.csv
│   ├── real_data_summary.json
│   ├── well_risk_predictions.csv
│   └── maintenance_priorities.csv
└── results/
    └── model_metrics.json
```

The two original ML CSVs are produced by Member 2's scripts. The `real_*.csv` files are made by `prepare_real_data.py`.

## Run in Windows PowerShell (from repository root)

```powershell
python -m pip install -r dashboard/requirements.txt
python src/data_cleaning.py
python dashboard/prepare_real_data.py
python src/predict.py
python src/optimization.py
python -m streamlit run dashboard/app.py
```

If you have not trained your model on this PC yet, run `python src/feature_engineering.py` and `python src/train_model.py` **before** `predict.py`.

Open **http://localhost:8501**. The app gives a clear error if the real production exports are missing; it never silently falls back to synthetic observations.

## Publishing safely

The competition briefing prohibits redistribution of the provided datasets. Keep original Parquet, well-level CSV exports, and potentially sensitive model artifacts out of public GitHub. Even a public app backed by a private database can reveal restricted records to its viewers. To make a publicly visible real-data demo, first get the provider's permission or build an approved non-sensitive aggregate-only version. This local version is designed to work without publishing restricted records.

## Interpretation

- A `risk_score` is **relative and not calibrated**; do not call it an 80% chance of failure.
- The maintenance priority formula is a **heuristic**, not a proven optimal dispatch plan.
- Estimated exposed oil is a **hypothetical scenario**, not recoverable oil or savings.
- Production readings are **daily historical summaries**, not live sensor data.
