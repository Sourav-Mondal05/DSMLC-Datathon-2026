"""
Member 2 - Synthetic tests for feature preparation and maintenance ranking

1. Purpose: catch elementary integration errors without exposing private well
   data, and verify that the optimization rules work on known examples.
2. Inputs: internally constructed fictional records (NOT the competition data).
3. Expected results: missing pressure placeholders become NaN; true/false
   features become numbers; high-risk, high-production wells rank higher.
4. Run from repository root: python -m unittest discover -s tests -v
5. Scope: these are unit tests only. They do not prove model accuracy, which
   is evaluated separately against the held-out 2026 competition records.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_utils import FEATURES, make_matrix
from optimization import prioritize, attach_historical_job_costs


class TestPipeline(unittest.TestCase):
    def test_pressure_placeholder(self):
        data = {col: [1.0, 2.0] for col in FEATURES}
        data["tubing_pressure_kpa"] = [999999, 120]
        data["has_downtime"] = [True, False]
        data["possible_invisible_loss"] = [False, True]
        X = make_matrix(pd.DataFrame(data))
        self.assertTrue(np.isnan(X.loc[0, "tubing_pressure_kpa"]))
        self.assertEqual(float(X.loc[1, "tubing_pressure_kpa"]), 120)
        self.assertEqual(float(X.loc[0, "has_downtime"]), 1)

    def test_historical_job_cost_join(self):
        priorities = pd.DataFrame({"uwi": ["TEST-A", "TEST-B"]})
        jobs = pd.DataFrame({"uwi": ["TEST-A", "TEST-A"], "field_est_total": [100, 300]})
        merged = attach_historical_job_costs(priorities, jobs)
        self.assertEqual(merged.loc[0, "historical_job_count"], 2)
        self.assertEqual(merged.loc[0, "historical_median_job_estimate"], 200)
        self.assertTrue(pd.isna(merged.loc[1, "historical_job_count"]))

    def test_priority_ranking(self):
        df = pd.DataFrame({
            "uwi": ["TEST-A", "TEST-B", "TEST-C"],
            "as_of_date": ["2026-10-10"] * 3,
            "risk_score": [0.8, 0.2, 0.6],
            "oil_baseline_7d": [40, 10, 20],
        })
        out = prioritize(df)
        self.assertEqual(out.iloc[0]["uwi"], "TEST-A")
        self.assertAlmostEqual(out.iloc[0]["hypothetical_exposed_oil_m3"], 120)
        self.assertEqual(out.iloc[0]["priority"], "High")


if __name__ == "__main__":
    unittest.main()
