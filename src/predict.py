"""
Member 2 - Generate well-level risk predictions for Streamlit and Canary

1. Purpose: load the trained model and score the most recent available
   producing-day record for each well. This is a DAILY historical snapshot.
2. Inputs: models/failure_model.joblib and outputs/model_features.parquet.
3. Model inputs: exactly the historical numeric features used during training.
   Never use future failure labels as prediction features.
4. Outputs: outputs/well_risk_predictions.csv (well, date, model risk score,
   historical oil baseline, and preliminary shortfall indicators).
5. Integration: Streamlit can read this CSV. Canary teammate can map a derived
   timestamp/tag/value export from this file using Canary's supported collector.
6. Interpretation: scores rank relative risk; they are NOT calibrated
   probabilities. Do not show 0.2 as a 20% failure chance.
7. Security: output contains well identifiers; NEVER commit it to public GitHub.

Usage: python src/predict.py
Usage with CSV: python src/predict.py --input outputs/model_features.csv.gz
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ml_utils import ROOT, DEFAULT_INPUT, load_features, producing_wells, make_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model", default=str(ROOT / "models" / "failure_model.joblib"))
    args = parser.parse_args()
    artifact = joblib.load(args.model)  # load only a model that you trust
    df = producing_wells(load_features(args.input), labels_required=False)
    if df.empty:
        raise ValueError("No producing wells found in the feature data.")
    latest_available_day = df["production_date"].max()
    # Exclude wells whose newest producing record is stale by over one month.
    df = df.loc[df["production_date"] >= latest_available_day - pd.Timedelta(days=30)]
    latest = (df.sort_values(["uwi", "production_date"])
              .drop_duplicates("uwi", keep="last").copy())
    X = make_matrix(latest, artifact["features"])
    scores = artifact["model"].predict_proba(X)[:, 1]
    out = latest[["uwi", "production_date", "oil_baseline_7d",
                  "potential_oil_loss_m3"]].copy()
    out = out.rename(columns={"production_date": "as_of_date",
                              "potential_oil_loss_m3": "historical_shortfall_indicator_m3"})
    out["risk_score"] = np.asarray(scores, dtype=float)
    out["risk_percentile"] = out["risk_score"].rank(pct=True, method="average")
    out["risk_score_meaning"] = "relative_score_not_calibrated_probability"
    out = out.sort_values("risk_score", ascending=False)
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True, parents=True)
    out.to_csv(folder / "well_risk_predictions.csv", index=False)
    print(f"Predictions: {len(out):,} wells; most recent day: {latest_available_day.date()}")
    print("Saved: outputs/well_risk_predictions.csv")
    print("Do not upload the well-level output to public GitHub.")


if __name__ == "__main__":
    main()
