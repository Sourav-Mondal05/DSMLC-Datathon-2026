"""
Member 2 - Train a 7-day oil-well equipment-failure risk model

1. Purpose: learn which historical production/sensor patterns are associated
   with a recorded equipment failure in the NEXT seven days.
2. Input: Member 1's outputs/model_features.parquet, containing the
   failure_next_7d label and failure_label_observable flag.
3. Methods: compare a simple oil-decline risk baseline, Logistic Regression,
   and HistGradientBoostingClassifier. Models use the same historical inputs.
4. Validation: Jun-Dec 2025 training, Jan-Mar 2026 model selection,
   and Apr-Oct 2026 untouched test. Seven-day split gaps avoid overlap.
5. Imbalance: keep all positive training rows; randomly sample negative rows
   and weight them to preserve the natural prevalence during fitting.
6. Metrics: average precision (PR-AUC), ROC-AUC, Brier score, precision@100.
   Accuracy alone is misleading because recorded failures are relatively rare.
7. Outputs: models/failure_model.joblib, results/model_metrics.json,
   results/feature_importance.png, and results/model_comparison.png.
8. Interpretation: prediction scores are NOT guaranteed calibrated failure
   probabilities; the models identify relative maintenance risk.
9. Reproducibility: fixed random seed and recorded split dates/row counts.
10. Privacy: treat trained artifacts as private unless approved for publishing.

Usage from repository root: python src/train_model.py
Optional CSV input: python src/train_model.py --input outputs/model_features.csv.gz
"""
import argparse
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_utils import (ROOT, DEFAULT_INPUT, FEATURES, load_features,
                      producing_wells, make_matrix, data_splits, labels)

SEED = 42
MAX_NEGATIVE_TRAIN = 300_000


def evaluate_scores(y, scores, top_n=100):
    """Compute metrics on an unbiased, chronologically held-out dataset."""
    yy = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    result = {"rows": int(len(yy)), "positive_rows": int(yy.sum()),
              "prevalence": round(float(yy.mean()), 8)}
    result["average_precision"] = (float(average_precision_score(yy, s))
                                    if yy.sum() else None)
    result["roc_auc"] = (float(roc_auc_score(yy, s))
                          if len(np.unique(yy)) == 2 else None)
    result["brier_score"] = float(brier_score_loss(yy, np.clip(s, 0, 1)))
    n = min(top_n, len(yy))
    result["precision_at_100_rows"] = float(yy[np.argsort(s)[-n:]].mean())
    return result


def downsample_training(X, y):
    """Keep positives; sample negative rows with inverse sampling weights."""
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    rng = np.random.default_rng(SEED)
    take = min(MAX_NEGATIVE_TRAIN, len(neg))
    neg_sample = rng.choice(neg, take, replace=False)
    ix = np.sort(np.concatenate([pos, neg_sample]))
    weights = np.where(y.to_numpy()[ix] == 0, len(neg) / take, 1.0)
    return X.iloc[ix], y.iloc[ix], weights


def models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, random_state=SEED)),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.07, max_leaf_nodes=15,
            min_samples_leaf=70, l2_regularization=2.0,
            early_stopping=False, random_state=SEED,
        ),
    }


def fit_model(name, estimator, X, y, weights):
    if name == "logistic_regression":
        estimator.fit(X, y, model__sample_weight=weights)
    else:
        estimator.fit(X, y, sample_weight=weights)
    return estimator


def plot_importance(model, X, y, path):
    """Drop-column permutation scoring on a sample of held-out early-2026 rows.

    Note: the displayed values are permutation changes in average precision,
    not causal contributions to actual failures.
    """
    rng = np.random.default_rng(SEED)
    sample_n = min(20_000, len(X))
    ix = np.sort(rng.choice(len(X), sample_n, replace=False))
    sample = X.iloc[ix].copy()
    target = y.iloc[ix].to_numpy()
    if target.sum() < 2:
        return []
    base = average_precision_score(target, model.predict_proba(sample)[:, 1])
    importance = []
    for col in FEATURES:
        modified = sample.copy()
        modified[col] = rng.permutation(modified[col].to_numpy())
        new_ap = average_precision_score(target, model.predict_proba(modified)[:, 1])
        importance.append((col, float(base - new_ap)))
    importance.sort(key=lambda x: x[1], reverse=True)
    top = importance[:12][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([x[0] for x in top], [x[1] for x in top])
    ax.set_xlabel("Decrease in validation average precision when shuffled")
    ax.set_title("Feature importance (permutation, descriptive)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [{"feature": k, "ap_decrease": v} for k, v in importance]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()
    print("Loading model features...", flush=True)
    df = producing_wells(load_features(args.input))
    X = make_matrix(df)
    y = labels(df)
    train, valid, test = data_splits(df)
    counts = {}
    for name, mask in (("train", train), ("validation", valid), ("test", test)):
        counts[name] = {"rows": int(mask.sum()), "positive_rows": int(y.loc[mask].sum())}
        if counts[name]["positive_rows"] < 2 or counts[name]["rows"] < 100:
            raise ValueError(f"Insufficient observations in {name}: {counts[name]}")
    print("Split counts:", counts, flush=True)
    Xtr, ytr, wt = downsample_training(X.loc[train], y.loc[train])
    Xval, yval = X.loc[valid], y.loc[valid]
    Xtest, ytest = X.loc[test], y.loc[test]
    baseline = Xval["oil_decline_pct"].fillna(0).clip(lower=0, upper=100) / 100
    results = {"oil_decline_baseline": {
        "validation": evaluate_scores(yval, baseline),
        "test": evaluate_scores(ytest, Xtest["oil_decline_pct"].fillna(0).clip(0, 100) / 100),
    }}
    fitted = {}
    for name, estimator in models().items():
        print("Training", name, flush=True)
        fit_model(name, estimator, Xtr, ytr, wt)
        fitted[name] = estimator
        results[name] = {"validation": evaluate_scores(yval, estimator.predict_proba(Xval)[:, 1])}
        print(name, "validation PR-AUC", round(results[name]["validation"]["average_precision"], 6), flush=True)
    # Select ONLY on the Jan-Mar 2026 validation set, not by Apr-Oct 2026 test results.
    selected = max(fitted, key=lambda n: results[n]["validation"]["average_precision"])
    model = fitted[selected]
    for name, estimator in fitted.items():
        results[name]["test"] = evaluate_scores(ytest, estimator.predict_proba(Xtest)[:, 1])

    models_dir, results_dir = ROOT / "models", ROOT / "results"
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)
    artifact = {
        "model": model, "model_name": selected, "features": list(FEATURES),
        "training_start_date": "2025-06-01", "training_last_date": "2025-12-24",
        "validation_start_date": "2026-01-01", "validation_last_date": "2026-03-24",
        "test_start_date": "2026-04-01", "prediction_window_days": 7,
        "risk_score_is_calibrated": False,
        "explanation": "Relative model risk score, not a calibrated failure probability",
    }
    joblib.dump(artifact, models_dir / "failure_model.joblib", compress=3)
    importance = plot_importance(model, Xval, yval, results_dir / "feature_importance.png")
    summary = {
        "goal": "Recorded equipment failure within the next seven days",
        "data_scope": "Producing well-days only; label-observable dates only",
        "split_counts": counts, "selected_model": selected,
        "metrics": results, "feature_importance": importance,
        "notes": [
            "Outcomes are recorded failures, not all possible physical failures.",
            "Daily rows from one failure can have overlapping seven-day labels.",
            "Scores are not calibrated probabilities or guaranteed operational savings.",
            "Historical failure log contains only two events before 2025; those years are excluded from supervised training.",
            "Model evaluation uses later dates to reduce temporal leakage.",
            "Saved model files and well-level predictions are restricted to local use.",
        ],
    }
    (results_dir / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, [results[n]["test"]["average_precision"] for n in names])
    ax.set_ylabel("Apr-Oct 2026 test average precision (PR-AUC)")
    ax.set_title("Out-of-time model comparison (Apr-Oct 2026)")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=140)
    plt.close(fig)
    print("Selected model:", selected)
    print("2026 test metrics:", results[selected]["test"])
    print("Saved model + metrics in models/ and results/", flush=True)


if __name__ == "__main__":
    main()
