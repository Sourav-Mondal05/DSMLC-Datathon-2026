# Member 2 — Machine Learning & Optimization

**Project:** Invisible Oil Loss Detector (DSMLC × Chirality × Canary Labs Datathon 2026)  
**Role:** Predictive maintenance, out-of-time ML validation, and inspection prioritization.

## 1. What this module does

1. Reads Member 1's `outputs/model_features.parquet` with historical production, pump, downtime, and seven-day future-failure labels.
2. Trains a baseline, Logistic Regression, and Histogram Gradient Boosting model.
3. Selects the model with better *validation average precision* (PR-AUC), and evaluates on a later, untouched test period.
4. Produces recent well-level risk scores for Streamlit and a possible Canary integration.
5. Combines model risk rank, recent oil output, and optional historical field job estimates to produce transparent **inspection priorities**, not automated maintenance instructions.

**Target:** Recorded equipment failure **1–7 days after** a producing well's observation date.

## 2. Important finding from the supplied dataset

Failure events are unusually concentrated in 2025–2026: there are only **2 recorded failures in 2021–2024**, versus **1,230 in 2025** and **1,641 in 2026**. Earlier dates are therefore **not used to train the supervised failure classifier**. Treating earlier dates as reliable negative cases could mislead the model because the logging coverage is unclear.

| Split | Date range | Why |
|---|---|---|
| Training | 2025-06-01 to 2025-12-24 | Learn historical patterns |
| Validation | 2026-01-01 to 2026-03-24 | Choose between models |
| Test | 2026-04-01 through the last fully label-observable day | Measure generalization without choosing on test |

The gaps after training and validation prevent **seven-day future labels** from crossing a boundary. The last few days without a complete seven-day outcome window are omitted from evaluation.

The same well can appear in multiple periods, which matches an operational deployment that predicts future behavior on existing wells. This is **not a new-well generalization experiment**.

## 3. File-by-file explanation

| File | Function |
|---|---|
| `src/ml_utils.py` | Shared feature definitions, loading, pressure-placeholder cleaning, chronological splitting |
| `src/train_model.py` | Baseline comparison, ML training, validation, test metrics, saving a model and plots |
| `src/evaluate_model.py` | Recompute selected model's held-out test metrics without refitting |
| `src/predict.py` | Score the most recent available producing records for individual wells |
| `src/optimization.py` | Rank wells for inspection, show optional historical job cost context, and calculate hypothetical production exposure |
| `tests/test_ml_pipeline.py` | Run synthetic unit tests; no restricted source records in tests |
| `requirements-ml.txt` | Python dependencies |
| `results/model_metrics.json` | Aggregate performance and validation assumptions |
| `results/model_comparison.png` | Comparison chart of test PR-AUC scores |
| `results/feature_importance.png` | Permutation-based feature-importance chart |
| `models/failure_model.joblib` | Serialized trained model — keep private unless explicitly permitted |
| `outputs/well_risk_predictions.csv` | Private output used by Streamlit and the Canary teammate |
| `outputs/maintenance_priorities.csv` | Private inspection priority list; scenario figures are not savings |

## 4. Copy files into the existing GitHub project

Your teammate's Python files already belong inside `src/`. Add **our Member 2 files into that same `src/` directory**, and place the test file in `tests/`. Add `requirements-ml.txt`, `.gitignore`, and this README at the repository root.

The machine-learning scripts expect the original competition files under `data/`. In the uploaded ZIP they were at the top level, so relocate them **locally** to `data/`, with *none* of the raw Parquet files committed to GitHub.

## 5. Run instructions (Windows / PowerShell, from project root)

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt
```

Move the three original Parquet files into the local `data/` folder **if they are still at the root**:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Move-Item .\daily_production.parquet .\data\ -ErrorAction SilentlyContinue
Move-Item .\well_jobs.parquet .\data\ -ErrorAction SilentlyContinue
Move-Item .\production_failures.parquet .\data\ -ErrorAction SilentlyContinue
```

Generate Member 1's required files (these outputs were absent from the uploaded ZIP):

```powershell
python src/data_cleaning.py
python src/feature_engineering.py
```

Train, evaluate, and generate decision outputs:

```powershell
python src/train_model.py
python src/evaluate_model.py
python src/predict.py
python src/optimization.py --assumed-downtime-days 3
python -m unittest discover -s tests -v
```

Check that the console prints a nonzero number of positive labels in every split. If the machine has limited RAM, close other applications before training. If using an older version of the Member 1 feature file, regenerate it first.

### Optional hypothetical oil value scenario

Only if **you explicitly choose** an assumed oil price, use:

```powershell
python src/optimization.py --assumed-downtime-days 3 --oil-price-cad-per-m3 70
```

If `data/well_jobs.parquet` exists, the optimizer also attaches per-well historical job counts and median `field_est_total` estimates. These are **not known future repair costs**.

The price of **70 CAD/m³ is merely an example input**, not an observed price or a statement about market conditions. This calculation estimates *hypothetical production exposed to downtime* and **does not establish savings**.

## 6. Initial evaluation results

A local validation run reconstructed the relevant production features from the original records and trained the supplied scripts. A **separate final run against Member 1's exported** `model_features.parquet` is still required before placing final figures on slides.

| Metric, held-out test period | Result from local reconstruction |
|---|---:|
| Producing well-day observations | 266,017 |
| Positive well-day labels | 7,031 |
| Positive label rate | 2.64% |
| Selected model | Histogram Gradient Boosting |
| ROC-AUC | 0.823 |
| PR-AUC / average precision | 0.190 |
| Simple oil-decline baseline PR-AUC | 0.115 |
| Precision among highest-scoring 100 **rows** | 49% |

**What this does NOT mean:** The model does not predict 82.3% of failures, achieve 82.3% accuracy, or prevent 49 failures per 100 alerts. Seven-day labels overlap across daily records, so multiple true-positive rows may correspond to the **same** recorded failure. The model's scores have not been probability-calibrated and should not be shown as literal likelihood percentages.

## 7. Data provided to other teammates

**Member 3 — Streamlit:**

- Read local `outputs/well_risk_predictions.csv` to show `uwi`, `as_of_date`, `risk_score`, `risk_percentile`, and `oil_baseline_7d`.
- Read local `outputs/maintenance_priorities.csv` for `priority`, `priority_score`, `recommended_action`, the optional historical `historical_median_job_estimate`, and hypothetical exposure fields.
- Read `results/model_metrics.json` for verified figures and limitations.
- Label `risk_score` as a *relative risk score*, not a calibrated probability.

**Member 4 — Canary Labs:**

- Use the same private prediction CSV as an intermediate staging file to import or write risk tags using the Canary version's supported format/API.
- Suggested tag concept: per-well `FailureRiskScore` and `MaintenancePriority`, indexed by `as_of_date`.
- The CSV is **not** claimed to be a ready-to-import Canary CSV Collector schema; Member 4 must map it appropriately.
- The historical competition data is **daily**, not live. Any playback or alerts must be identified as a simulation.

## 8. Risks, limitations, and audit checklist

- Some sensor columns are frequently missing, and tubing pressure can contain placeholder `999999` values; the model masks them.
- Failure logging appears incomplete before mid-2025; do not generalize performance to those years.
- The model is retrospective and observational; correlation does not establish why equipment fails.
- We used **time-based** validation, but have not independently established performance on unseen wells or field sites.
- No price or intervention-effectiveness data was provided. Economic outputs are scenarios, not financial returns.
- `possible_invisible_loss` and oil-loss indicators are not confirmed recoverable barrels.
- Before judging, rerun Member 1's exact feature pipeline, retrain the model, and use the new metrics in the slides.

## 9. What to publish and what to keep private

**Suitable for the public competition repository:** reviewed Python scripts, this README, requirements, synthetic tests, and aggregate model metrics/figures after team review.

**Keep local:** `data/`, `outputs/`, `models/`, raw/derived `.parquet` files, `.joblib` model, well-level CSV files, credentials. The competition restricts redistribution of the supplied datasets. The replacement `.gitignore` supplied here reduces accidental upload risk. Check `git status` before committing.

## 10. AI assistance disclosure

> AI-assisted development was used to draft and debug the machine-learning scripts, testing logic, model comparison, and documentation. The team inspected the data, executed the code, reviewed the evaluation metrics, and remains responsible for correctness, reproducibility, and the final claims.

This disclosure should also be included in the main project README, as required by the competition rules.
