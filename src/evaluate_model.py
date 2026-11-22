"""
Member 2 - Independently reproduce the selected model's held-out test metrics

1. Purpose: re-open the trained artifact and compute honest performance on
   the Apr-Oct 2026 test period. This helps verify the figures used in a presentation.
2. Inputs: the saved model and Member 1's model_features dataset.
3. Test rows: producing wells, fully observed seven-day failure windows,
   dates starting 2026-04-01. The model was NOT selected on this period.
4. Metrics: PR-AUC, ROC-AUC, precision@100 rows, baseline positive rate.
5. Leakage check: there is no fitting or feature selection inside this file.
6. Output: prints metrics; training produces results/model_metrics.json.
7. Important: daily-row labels overlap across days; do not describe the
   precision@100-rows metric as 'number of distinct failures caught'.

Usage: python src/evaluate_model.py
"""
import argparse
import joblib
from ml_utils import ROOT, DEFAULT_INPUT, load_features, producing_wells, make_matrix, labels, data_splits
from train_model import evaluate_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model", default=str(ROOT / "models" / "failure_model.joblib"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)  # only load trusted models
    df = producing_wells(load_features(args.input))
    _, _, test = data_splits(df)
    X = make_matrix(df.loc[test], bundle["features"])
    y = labels(df.loc[test])
    scores = bundle["model"].predict_proba(X)[:, 1]
    print("Selected model:", bundle["model_name"])
    for key, value in evaluate_scores(y, scores).items():
        print(f"{key}: {value}")
    print("Risk scores are not calibrated probabilities.")


if __name__ == "__main__":
    main()
