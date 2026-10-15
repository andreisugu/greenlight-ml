import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import (
    filter_minimum_scale,
    filter_reported_financials,
    load_and_merge_data,
)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "title": ["Reported hit", "Reported flop", "Unknown budget", "Microbudget hit"],
                "budget": [20_000, 40_000, 0, 7_000],
                "revenue": [80_000, 10_000, 30_000, 400_000],
            }
        )

    def test_reported_financial_population_keeps_valid_microbudget_and_derives_both_targets(self):
        source = self.movies.copy(deep=True)

        result = filter_reported_financials(self.movies)

        self.assertEqual(result["id"].tolist(), [1, 2, 4])
        self.assertEqual(result["target_nominal"].tolist(), [1, 0, 1])
        self.assertEqual(result["target_hollywood"].tolist(), [1, 0, 1])
        pd.testing.assert_frame_equal(self.movies, source)

    def test_reported_financial_filter_coerces_numeric_strings_and_excludes_infinity(self):
        data = pd.DataFrame(
            {
                "budget": ["100", "inf", "not reported"],
                "revenue": ["250", "500", "100"],
            }
        )

        result = filter_reported_financials(data)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["budget"], 100)
        self.assertEqual(result.iloc[0]["target_nominal"], 1)

    def test_minimum_scale_is_a_named_sensitivity_filter_not_the_primary_population(self):
        reported = filter_reported_financials(self.movies)

        result = filter_minimum_scale(reported, usd_floor=10_000)

        self.assertEqual(result["id"].tolist(), [1])
        self.assertEqual(len(reported), 3)
        self.assertEqual(len(result), 1)

    def test_minimum_scale_uses_strict_floor_for_both_financial_columns(self):
        boundary = pd.DataFrame(
            {
                "budget": [10_000, 10_001, 10_001],
                "revenue": [20_000, 10_000, 10_001],
            }
        )

        result = filter_minimum_scale(boundary, usd_floor=10_000)

        self.assertEqual(result.index.tolist(), [2])

    def test_merge_rejects_duplicate_join_keys_in_either_file(self):
        movies = self.movies.iloc[:2].copy()
        credits = pd.DataFrame(
            {
                "movie_id": [1, 1],
                "title": ["Reported hit", "Reported hit"],
                "cast": ["[]", "[]"],
                "crew": ["[]", "[]"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            movies_path = Path(temp_dir) / "movies.csv"
            credits_path = Path(temp_dir) / "credits.csv"
            movies.to_csv(movies_path, index=False)
            credits.to_csv(credits_path, index=False)

            with self.assertRaises(pd.errors.MergeError):
                load_and_merge_data(movies_path, credits_path)

    def test_merge_rejects_title_disagreement_for_matching_ids(self):
        movies = self.movies.iloc[:1].copy()
        credits = pd.DataFrame(
            {"movie_id": [1], "title": ["Different title"], "cast": ["[]"], "crew": ["[]"]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            movies_path = Path(temp_dir) / "movies.csv"
            credits_path = Path(temp_dir) / "credits.csv"
            movies.to_csv(movies_path, index=False)
            credits.to_csv(credits_path, index=False)

            with self.assertRaisesRegex(ValueError, "title mismatch"):
                load_and_merge_data(movies_path, credits_path)

    def test_merge_rejects_unmatched_ids_instead_of_silently_dropping_rows(self):
        movies = self.movies.iloc[:1].copy()
        credits = pd.DataFrame(
            {"movie_id": [99], "title": ["Orphan"], "cast": ["[]"], "crew": ["[]"]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            movies_path = Path(temp_dir) / "movies.csv"
            credits_path = Path(temp_dir) / "credits.csv"
            movies.to_csv(movies_path, index=False)
            credits.to_csv(credits_path, index=False)

            with self.assertRaisesRegex(ValueError, "key mismatch"):
                load_and_merge_data(movies_path, credits_path)


if __name__ == "__main__":
    unittest.main()
