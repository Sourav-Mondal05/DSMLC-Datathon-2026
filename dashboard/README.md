# WellGuard Streamlit Dashboard

## What this is
A simple five-tab website showing **actual outputs** from the datathon project:
- Overview: field production trend, number of wells scored, high-priority wells.
- Failure Risk: most recent model scores and relative risk rankings.
- Maintenance: inspection priorities and hypothetical production exposure.
- Well Explorer: recent per-well production, pump readings and recorded failures.
- Model Results: ROC-AUC, PR-AUC, baseline comparison and limitations.

The site **does not invent demo data**. If files are missing, it explains what to create. It does not require Canary or a live API.

## Files to put on GitHub

```text
DSMLC-Datathon-2026/
  dashboard/
    app.py
    prepare_dashboard_data.py
    requirements.txt
    README.md
    .streamlit/
      config.toml
```

You can delete the OLD `dashboard/pages/` and `dashboard/utils/` folders. The new `app.py` does not use them.

## Files to keep PRIVATE (never upload to the public GitHub)

```text
DSMLC-Datathon-2026/
  data/
    daily_production.parquet
    well_jobs.parquet
    production_failures.parquet
  outputs/
    cleaned_production.parquet
    model_features.parquet
    well_risk_predictions.csv
    maintenance_priorities.csv
    dashboard_daily.csv
    dashboard_recent_well_daily.csv
    dashboard_failures.csv
  models/
    failure_model.joblib
  results/
    model_metrics.json
```

The `results/model_metrics.json` file is aggregate metrics and may remain public after verification. `data/`, `outputs/`, and `models/` must remain excluded by `.gitignore`.

## One-time setup (PowerShell in VS Code)

From the root `DSMLC-Datathon-2026` folder:

```powershell
python -m pip install -r dashboard/requirements.txt
```

If your teammate already generated `outputs/well_risk_predictions.csv` and `outputs/maintenance_priorities.csv`, **do not retrain just for Streamlit**. Use those outputs.

Otherwise, after placing the ORIGINAL competition data in the private `data/` folder, run:

```powershell
python src/data_cleaning.py
python src/feature_engineering.py
python src/train_model.py
python src/predict.py
python src/optimization.py
```

Prepare smaller production/history CSVs for the dashboard:

```powershell
python dashboard/prepare_dashboard_data.py
```

Start the dashboard:

```powershell
python -m streamlit run dashboard/app.py
```

Open the local website shown in your terminal (normally `http://localhost:8501`).

## Before submitting
1. Check that the sidebar shows all expected inputs as loaded.
2. Check that failure risk scores are real model outputs and the latest run date is reasonable.
3. Check that maintenance rankings and well history display real records.
4. Review the metrics from the latest complete training run.
5. Do not present model scores as calibrated failure probabilities, or hypothetical exposure as proven savings.
6. Do not upload any generated CSVs or raw Parquet files to public GitHub.
7. Add a link and a screenshot of the dashboard to the main repo README, if useful.

## AI use
AI-assisted coding and documentation were used for the dashboard. The team is responsible for running it, verifying results, and disclosing AI assistance under the hackathon rules.
