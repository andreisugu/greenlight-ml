import unittest

import pandas as pd

from src.metrics import calculate_gross_to_budget_surplus, evaluate_model


class GrossToBudgetSurplusTests(unittest.TestCase):
    def test_reports_weighted_gross_surplus_not_net_studio_roi(self):
        selected = pd.DataFrame(
            {"budget": [100, 300], "revenue": [400, 600]}
        )

        result = calculate_gross_to_budget_surplus(selected)

        self.assertAlmostEqual(result, (1_000 - 400) / 400)

    def test_empty_or_nonpositive_budget_selection_returns_zero(self):
        self.assertEqual(
            calculate_gross_to_budget_surplus(pd.DataFrame(columns=["budget", "revenue"])),
            0.0,
        )

    def test_evaluation_reports_macro_f1_and_labels_financial_metric_as_proxy(self):
        result = evaluate_model(
            "baseline",
            "pre-release",
            pd.Series([0, 0, 1, 1]),
            [1, 1, 1, 1],
            [0.6, 0.7, 0.8, 0.9],
            pd.DataFrame(
                {"budget": [100, 100, 100, 100], "revenue": [50, 50, 200, 300]}
            ),
        )

        self.assertEqual(result["Macro F1"], 0.3333)
        self.assertIn("Gross-to-Budget Surplus Proxy", result)
        self.assertNotIn("Net Portfolio ROI", result)
        self.assertEqual(
            calculate_gross_to_budget_surplus(
                pd.DataFrame({"budget": [0], "revenue": [100]})
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
