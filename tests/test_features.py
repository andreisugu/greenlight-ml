import unittest

import numpy as np
import pandas as pd

from src.features import FeatureEncoder


def _row(genres, director, language, overview, release_date, budget, runtime,
         popularity=0.0, vote_average=0.0, vote_count=0.0):
    return {
        "genres": '[{"id": 1, "name": "%s"}]' % genres,
        "crew": '[{"job": "Director", "name": "%s"}]' % director,
        "original_language": language,
        "overview": overview,
        "release_date": release_date,
        "budget": budget,
        "runtime": runtime,
        "popularity": popularity,
        "vote_average": vote_average,
        "vote_count": vote_count,
    }


class FeatureEncoderLeakageTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([
            _row("Action", "Alice", "en", "alpha beta", "2000-01-01", 1_000, 100.0),
            _row("Action", "Alice", "en", "alpha gamma", "2000-06-01", 2_000, 120.0),
        ])
        self.test = pd.DataFrame([
            _row("Drama", "Zed", "fr", "delta epsilon", None, 500, np.nan),
        ])

    def test_unfitted_encoder_rejects_transform(self):
        with self.assertRaises(RuntimeError):
            FeatureEncoder().transform(self.train)

    def test_test_only_categories_never_create_columns(self):
        encoder = FeatureEncoder().fit(self.train)

        X_test = encoder.transform(self.test)

        columns = list(X_test.columns)
        self.assertIn("genre_Action", columns)
        self.assertNotIn("genre_Drama", columns)
        self.assertNotIn("dir_Zed", columns)
        self.assertNotIn("lang_fr", columns)

    def test_test_only_rows_map_to_reference_categories_and_zero_hot(self):
        encoder = FeatureEncoder().fit(self.train)

        X_test = encoder.transform(self.test)

        self.assertEqual(X_test["genre_Action"].iloc[0], 0)
        self.assertEqual(X_test["dir_Other"].iloc[0], 1)
        self.assertEqual(X_test["lang_Other"].iloc[0], 1)

    def test_release_date_and_runtime_impute_from_training_only(self):
        encoder = FeatureEncoder().fit(self.train)

        X_test = encoder.transform(self.test)

        self.assertEqual(X_test["runtime"].iloc[0], 110.0)   # train median of 100, 120
        self.assertEqual(X_test["release_year"].iloc[0], 2000.0)
        self.assertEqual(X_test["release_month"].iloc[0], 6.0)

    def test_tfidf_vocabulary_is_frozen_from_training(self):
        encoder = FeatureEncoder().fit(self.train)

        X_test = encoder.transform(self.test)

        self.assertFalse(any("epsilon" in column for column in X_test.columns))

    def test_train_and_test_share_identical_columns_without_nan(self):
        encoder = FeatureEncoder().fit(self.train)

        X_train = encoder.transform(self.train)
        X_test = encoder.transform(self.test)

        self.assertEqual(list(X_train.columns), list(X_test.columns))
        self.assertFalse(X_test.isna().any().any())

    def test_leakage_columns_are_opt_in(self):
        encoder = FeatureEncoder().fit(self.train)

        clean = encoder.transform(self.test)
        leaked = encoder.transform(self.test, include_leakage=True)

        self.assertNotIn("popularity", clean.columns)
        self.assertTrue({"popularity", "vote_average", "vote_count"}.issubset(leaked.columns))

    def test_transform_is_deterministic(self):
        encoder = FeatureEncoder().fit(self.train)

        first = encoder.transform(self.test)
        second = encoder.transform(self.test)

        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
