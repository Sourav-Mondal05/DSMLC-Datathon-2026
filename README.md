# Data Engineering Preparation

**Project:** Invisible Oil Loss Detector  
**Role:** Member 1 — Data Engineering

## 1. Overview

Prepared historical oil-well production data for machine learning, anomaly detection, and dashboard development. The pipeline cleans production records, analyzes production trends, engineers historical features, and connects maintenance and failure information.

## 2. Files Created and Their Functions

| No. | File | Purpose and Technical Work |
|---|---|---|
| 1 | `src/data_cleaning.py` | Loads daily production Parquet data using Pandas. Removes exact duplicates and invalid identifiers/dates, converts numeric columns, and flags invalid production and operating-hour values. |
| 2 | `src/data_analysis.py` | Explores production records, missing values, and historical production trends. Generates summary statistics and visualizations. |
| 3 | `src/feature_engineering.py` | Calculates historical production features, connects maintenance information, and creates future failure labels using time-aware operations to reduce data leakage. |
| 4 | `src/anomaly_validation.py` | Examines potential production anomalies to support validation of the detection approach. |
| 5 | `src/oil_loss_impact.py` | Estimates potential oil-production shortfalls and summarizes their operational impact. |
| 6 | `src/loss_quality_check.py` | Examines the distribution and concentration of estimated oil losses and checks for extreme values. |
| 7 | `src/false_positive_analysis.py` | Screens anomaly alerts for quick production recovery, which may indicate temporary fluctuations rather than persistent losses. |
| 8 | `src/persistent_anomaly_analysis.py` | Groups consecutive anomaly records into episodes and measures how long potential production losses persist. |
| 9 | `src/maintenance_overlap_analysis.py` | Examines whether detected production anomalies overlap with maintenance activities to help distinguish explainable downtime from unexplained losses. |

## 3. Technical Methods Used

- **Python and Pandas:** Data loading, cleaning, grouping, and aggregation.
- **Parquet:** Efficient storage of large production datasets.
- **Time-series analysis:** Historical rolling averages, production changes, and anomaly indicators.
- **Data integration:** Matching production, maintenance, and failure records using well identifiers and dates.
- **Data leakage prevention:** Using historical observations when constructing predictive features.
- **Quality checks:** Identifying invalid measurements, temporary anomalies, and potentially misleading oil-loss estimates.

## 4. Main Output Files

| File | Description |
|---|---|
| `outputs/cleaned_production.parquet` | Cleaned production records for downstream processing. |
| `outputs/model_features.parquet` | Historical production features, maintenance indicators, and future failure labels for model development. |
| `outputs/production_features.parquet` | Production baselines and preliminary oil-loss indicators. |

Additional CSV summaries and PNG visualizations are generated locally inside `outputs/`.

**Important:** Generated datasets are excluded from GitHub and must be shared separately with teammates.

## 5. Pipeline Execution

Run the main scripts from the project root:

```bash
python src/data_cleaning.py
python src/data_analysis.py
python src/feature_engineering.py
```

## 6. Handoff Status

Data engineering scripts have been committed and merged into the `main` branch.

The cleaned production and model-feature datasets have been generated locally. Teammates can use these files for machine learning, Streamlit dashboard development, and further oil-loss analysis once the datasets are shared.

**Note:** Detected anomalies and estimated oil losses are preliminary indicators, not confirmed recoverable oil production.