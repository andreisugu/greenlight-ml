"""Feature engineering for multimodal film metadata.

Design rule: every data-dependent decision (category vocabularies, imputation
statistics, TF-IDF vocabulary) is learned in ``FeatureEncoder.fit`` and reused
unchanged in ``FeatureEncoder.transform``. Fitting must therefore happen on the
training split only; transform never looks at other rows to decide columns.

``FeatureEncoder`` is a scikit-learn transformer so it can sit inside a
``Pipeline`` and be refitted inside every cross-validation fold.
"""
import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def parse_json_list(json_str: str, key: str = "name") -> list:
    """Safely parse a JSON attribute list, returning [] on malformed input."""
    try:
        data = json.loads(json_str)
        return [item[key] for item in data if key in item]
    except (TypeError, ValueError):
        return []


def extract_director(crew_str: str) -> str:
    """Return the primary director name, or 'Unknown' when absent."""
    try:
        crew = json.loads(crew_str)
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "Unknown")
    except (TypeError, ValueError):
        pass
    return "Unknown"


def extract_top_cast(cast_str: str, n: int = 3) -> list:
    """Return the top ``n`` billed actor names."""
    try:
        cast = json.loads(cast_str)
        return [member.get("name", "") for member in cast[:n] if "name" in member]
    except (TypeError, ValueError):
        return []


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Train-fit feature transformer that cannot learn from held-out rows.

    Parameters
    ----------
    max_tfidf_features:
        Maximum number of TF-IDF terms retained for the synopsis text.
    top_directors, top_languages:
        Cardinality caps for the categorical one-hot encodings. Categories
        outside the training top-N collapse into an explicit ``Other`` bucket.
    include_text:
        If False, skip the synopsis TF-IDF block. V1 uses False so the
        baseline contains only pre-release structured features.
    include_director:
        If False, skip the director one-hot block. V1 uses False; V2 adds it.
    """

    def __init__(self, max_tfidf_features: int = 100, top_directors: int = 30,
                 top_languages: int = 5, include_text: bool = True,
                 include_director: bool = True):
        self.max_tfidf_features = max_tfidf_features
        self.top_directors = top_directors
        self.top_languages = top_languages
        self.include_text = include_text
        self.include_director = include_director
        self._fitted = False

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _genre_lists(df: pd.DataFrame):
        return df["genres"].apply(lambda blob: parse_json_list(blob, key="name"))

    @staticmethod
    def _directors(df: pd.DataFrame):
        return df["crew"].apply(extract_director)

    def _required_columns(self):
        columns = ["genres", "original_language", "release_date", "budget", "runtime"]
        if self.include_director:
            columns.append("crew")
        if self.include_text:
            columns.append("overview")
        return columns

    def _require(self, df: pd.DataFrame, columns=None) -> None:
        columns = self._required_columns() if columns is None else columns
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

    # -- fit / transform -------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None) -> "FeatureEncoder":
        """Learn vocabularies and imputation statistics from training rows only."""
        self._require(df)

        self.mlb_ = MultiLabelBinarizer()
        self.mlb_.fit(self._genre_lists(df))

        if self.include_director:
            director_counts = self._directors(df).value_counts()
            self.kept_directors_: List[str] = [
                name for name in director_counts.head(self.top_directors).index
                if name != "Other"
            ]
        else:
            self.kept_directors_ = []

        language_counts = df["original_language"].astype(str).value_counts()
        self.kept_languages_: List[str] = list(language_counts.head(self.top_languages).index)

        release = pd.to_datetime(df["release_date"], errors="coerce")
        self.year_median_ = float(release.dt.year.median())
        self.month_median_ = float(release.dt.month.median())
        self.runtime_median_ = float(pd.to_numeric(df["runtime"], errors="coerce").median())

        if self.include_text:
            self.tfidf_ = TfidfVectorizer(max_features=self.max_tfidf_features,
                                          stop_words="english")
            self.tfidf_.fit(df["overview"].fillna("").astype(str))
        else:
            self.tfidf_ = None

        self.feature_columns_ = list(self._transform_core(df).columns)
        self._fitted = True
        return self

    def _transform_core(self, df: pd.DataFrame) -> pd.DataFrame:
        index = df.index
        release = pd.to_datetime(df["release_date"], errors="coerce")
        release_year = release.dt.year.fillna(self.year_median_)
        release_month = release.dt.month.fillna(self.month_median_)

        # Restrict to the training vocabulary *before* encoding: test-only genres
        # map to all-zero, exactly as MultiLabelBinarizer would, but without the
        # "unknown class ignored" warning that would otherwise mask real warnings.
        known_genres = set(self.mlb_.classes_)
        genre_lists = self._genre_lists(df).apply(
            lambda labels: [g for g in labels if g in known_genres]
        )
        genre_matrix = self.mlb_.transform(genre_lists)
        genres = pd.DataFrame(
            genre_matrix,
            columns=[f"genre_{g}" for g in self.mlb_.classes_],
            index=index,
            dtype=float,
        )

        parts = [
            pd.DataFrame({
                "log_budget": np.log1p(pd.to_numeric(df["budget"], errors="coerce")),
                "runtime": pd.to_numeric(df["runtime"], errors="coerce").fillna(self.runtime_median_),
                "release_year": release_year,
                "release_month": release_month,
            }, index=index),
            genres,
        ]

        if self.include_director:
            kept_directors = set(self.kept_directors_)
            director_clean = self._directors(df).apply(
                lambda name: name if name in kept_directors else "Other"
            )
            director_columns = [f"dir_{name}" for name in self.kept_directors_] + ["dir_Other"]
            directors = pd.get_dummies(director_clean, prefix="dir", dtype=float).reindex(
                columns=director_columns, fill_value=0.0
            )
            directors.index = index
            parts.append(directors)

        kept_languages = set(self.kept_languages_)
        language_clean = df["original_language"].astype(str).apply(
            lambda code: code if code in kept_languages else "Other"
        )
        language_columns = [f"lang_{code}" for code in self.kept_languages_] + ["lang_Other"]
        languages = pd.get_dummies(language_clean, prefix="lang", dtype=float).reindex(
            columns=language_columns, fill_value=0.0
        )
        languages.index = index
        parts.append(languages)

        if self.include_text:
            tfidf_matrix = self.tfidf_.transform(df["overview"].fillna("").astype(str))
            tfidf = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"tfidf_{term}" for term in self.tfidf_.get_feature_names_out()],
                index=index,
                dtype=float,
            )
            parts.append(tfidf)

        return pd.concat(parts, axis=1)

    def transform(self, df: pd.DataFrame, include_leakage: bool = False) -> pd.DataFrame:
        """Apply the fitted transform. Never adds categories seen only in ``df``."""
        if not self._fitted:
            raise RuntimeError("FeatureEncoder must be fit before transform")

        features = self._transform_core(df).reindex(columns=self.feature_columns_, fill_value=0.0)

        if include_leakage:
            self._require(df, ["popularity", "vote_average", "vote_count"])
            features = features.copy()
            features["popularity"] = df["popularity"].fillna(0)
            features["vote_average"] = df["vote_average"].fillna(0)
            features["vote_count"] = np.log1p(df["vote_count"].fillna(0))
        return features

    def fit_transform(self, df: pd.DataFrame, y=None, include_leakage: bool = False) -> pd.DataFrame:
        """Fit on ``df`` and transform it; ``y`` is accepted for Pipeline compatibility."""
        return self.fit(df, y).transform(df, include_leakage=include_leakage)
