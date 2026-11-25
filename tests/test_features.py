import unittest

import numpy as np
import pandas as pd

from src.features import FeatureEncoder


def _row(genres, director, language, overview, release_date, budget, runtime,
         popularity=0.0, vote_average=0.0, vote_count=0.0, cast=("Ann", "Bob", "Cid")):
    return {
        "genres": '[{"id": 1, "name": "%s"}]' % genres,
        "crew": '[{"job": "Director", "name": "%s"}]' % director,
        "cast": '[%s]' % ", ".join('{"name": "%s"}' % actor for actor in cast),
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
        self.assertEqual(X_test["release_month"].iloc[0], 3.5)   # train median of 1, 6

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


class FeatureEncoderV1Tests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(8):
            rows.append(_row(
                "Action" if i % 2 else "Drama",
                "Alice" if i < 4 else "Bob",
                "en" if i % 2 else "fr",
                "alpha beta" if i < 4 else "gamma delta",
                f"200{i}-01-01",
                1_000 + i,
                100 + i,
            ))
        self.frame = pd.DataFrame(rows)
        self.y = [0, 1, 0, 1, 0, 1, 0, 1]

    def test_basic_feature_set_excludes_text_and_director(self):
        encoder = FeatureEncoder(include_text=False, include_director=False).fit(self.frame)

        X = encoder.transform(self.frame)

        self.assertFalse(any(c.startswith("tfidf_") for c in X.columns))
        self.assertFalse(any(c.startswith("dir_") for c in X.columns))
        self.assertIn("genre_Action", X.columns)
        self.assertIn("lang_Other", X.columns)

    def test_fit_transform_accepts_pipeline_y_argument(self):
        encoder = FeatureEncoder(include_text=False, include_director=False)

        X = encoder.fit_transform(self.frame, self.y)

        self.assertEqual(len(X), len(self.frame))

    def test_pipeline_refits_encoder_inside_cv(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ("features", FeatureEncoder(include_text=False, include_director=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ])
        grid = GridSearchCV(
            pipe,
            {"clf__C": [0.1, 1.0]},
            cv=StratifiedKFold(n_splits=4),
            n_jobs=1,
        )

        grid.fit(self.frame, self.y)

        fitted_encoder = grid.best_estimator_.named_steps["features"]
        self.assertTrue(hasattr(grid.best_estimator_, "predict_proba"))
        self.assertFalse(any(c.startswith("tfidf_") for c in fitted_encoder.feature_columns_))
        # GridSearchCV clones the pipeline, so the winning encoder is not the object
        # handed in: the encoder is fit inside the search, never once up front.
        self.assertIsNot(fitted_encoder, pipe.named_steps["features"])


class FeatureEncoderV2Tests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([
            _row("Action", "Alice", "en", "Alpha beta, 1999!", "2000-01-01", 1_000, 100.0,
                 cast=("Ann", "Bob", "Cid")),
            _row("Action", "Alice", "en", "alpha gamma", "2000-06-01", 2_000, 120.0,
                 cast=("Ann", "Bob", "Dee")),
            _row("Drama", "Zed", "fr", "delta epsilon", "2001-01-01", 3_000, 90.0,
                 cast=("Ann", "Eve", "Fay")),
        ])
        self.test = pd.DataFrame([
            _row("Drama", "Zed", "fr", "delta epsilon", "2001-01-01", 500, 90.0,
                 cast=("Ann", "Nobody", "Stranger")),
        ])

    def test_text_is_cleaned_before_vectorising(self):
        from src.features import clean_text

        self.assertEqual(clean_text("A Man's 2nd Best: 1945 (Restored!)"),
                         "a man s nd best restored")

    def test_tfidf_keeps_bigrams(self):
        encoder = FeatureEncoder(max_tfidf_features=50, include_text=True,
                                 include_cast=False).fit(self.train)

        terms = list(encoder.tfidf_.get_feature_names_out())

        self.assertIn("alpha beta", terms)

    def test_actor_frequency_matches_training_counts(self):
        encoder = FeatureEncoder(include_text=False, include_cast=True).fit(self.train)

        self.assertAlmostEqual(encoder.actor_share_["Ann"], 3 / 3)
        self.assertAlmostEqual(encoder.actor_share_["Bob"], 2 / 3)
        self.assertAlmostEqual(encoder.actor_share_["Fay"], 1 / 3)

    def test_unseen_actor_scores_zero(self):
        encoder = FeatureEncoder(include_text=False, include_cast=True).fit(self.train)

        X = encoder.transform(self.test)

        self.assertAlmostEqual(X["cast1_freq"].iloc[0], 1.0)   # Ann appears in all 3
        self.assertAlmostEqual(X["cast2_freq"].iloc[0], 0.0)   # Nobody unseen
        self.assertAlmostEqual(X["cast3_freq"].iloc[0], 0.0)   # Stranger unseen

    def test_missing_cast_column_is_rejected(self):
        frame = self.train.drop(columns=["cast"])

        with self.assertRaisesRegex(ValueError, "cast"):
            FeatureEncoder(include_text=False, include_cast=True).fit(frame)

    def test_director_frequency_matches_training_counts(self):
        encoder = FeatureEncoder(include_text=False, include_director=True,
                                 include_cast=False).fit(self.train)

        X = encoder.transform(self.train)

        self.assertAlmostEqual(X["director_freq"].iloc[0], 2 / 3)
        self.assertAlmostEqual(X["director_freq"].iloc[2], 1 / 3)


if __name__ == "__main__":
    unittest.main()
