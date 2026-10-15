"""TMDB file validation, relational merge, and explicit population definitions."""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def get_default_data_paths() -> Tuple[str, str]:
    """Resolves default file paths for movies and credits CSVs."""
    base_dir = os.path.dirname(__file__)
    local_movies = os.path.join(base_dir, "..", "data", "tmdb_5000_movies.csv")
    local_credits = os.path.join(base_dir, "..", "data", "tmdb_5000_credits.csv")

    if os.path.exists(local_movies) and os.path.exists(local_credits):
        return local_movies, local_credits

    # Parent workspace fallback
    parent_movies = os.path.join(base_dir, "..", "..", "data", "tmdb_5000_movies.csv")
    parent_credits = os.path.join(base_dir, "..", "..", "data", "tmdb_5000_credits.csv")
    if os.path.exists(parent_movies) and os.path.exists(parent_credits):
        return parent_movies, parent_credits

    return local_movies, local_credits

def load_and_merge_data(movies_path: Optional[str] = None, credits_path: Optional[str] = None) -> pd.DataFrame:
    """Load the one-to-one TMDB relation, rejecting key/title inconsistencies."""
    default_movies, default_credits = get_default_data_paths()
    movies_path = movies_path or default_movies
    credits_path = credits_path or default_credits

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(
            f"Dataset files missing.\nLooked for:\n - {movies_path}\n - {credits_path}\n"
            "Please run 'python scripts/download_data.py' to download TMDB 5000."
        )

    df_movies = pd.read_csv(movies_path)
    df_credits = pd.read_csv(credits_path)

    if "id" not in df_movies or "title" not in df_movies:
        raise ValueError("movies file must include 'id' and 'title' columns")
    if not {"movie_id", "title"}.issubset(df_credits.columns):
        raise ValueError("credits file must include 'movie_id' and 'title' columns")

    merged = pd.merge(
        df_movies,
        df_credits,
        left_on="id",
        right_on="movie_id",
        suffixes=("", "_credits"),
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        raise ValueError(f"movie/credits key mismatch: {int(unmatched.sum())} unmatched rows")

    same_title = merged["title"].eq(merged["title_credits"])
    both_titles_missing = merged["title"].isna() & merged["title_credits"].isna()
    title_mismatch = ~(same_title | both_titles_missing)
    if title_mismatch.any():
        count = int(title_mismatch.sum())
        raise ValueError(f"title mismatch for {count} matching movie ID(s)")

    return merged.drop(columns=["movie_id", "title_credits", "_merge"])

def filter_reported_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with positive, finite budget and revenue records.

    This defines the *reported-financials analysis population*. It does not
    assert every retained movie had a theatrical release, nor that excluded
    zero values mean commercial failure. The two revenue-multiple targets are
    proxies, not verified studio profitability.
    """
    required = {"budget", "revenue"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"financial columns missing: {sorted(missing)}")

    budget = pd.to_numeric(df["budget"], errors="coerce")
    revenue = pd.to_numeric(df["revenue"], errors="coerce")
    valid = (
        budget.gt(0)
        & revenue.gt(0)
        & np.isfinite(budget)
        & np.isfinite(revenue)
    )
    reported = df.loc[valid].copy()
    reported["budget"] = budget.loc[valid]
    reported["revenue"] = revenue.loc[valid]
    reported["target_nominal"] = (reported["revenue"] > reported["budget"]).astype("int8")
    reported["target_hollywood"] = (reported["revenue"] >= 2.0 * reported["budget"]).astype("int8")
    return reported

def filter_minimum_scale(df: pd.DataFrame, usd_floor: float = 10_000) -> pd.DataFrame:
    """Optional, explicitly thresholded sensitivity subset; not an anomaly detector."""
    if usd_floor < 0 or not np.isfinite(usd_floor):
        raise ValueError("usd_floor must be a finite, non-negative amount")
    valid = df["budget"].gt(usd_floor) & df["revenue"].gt(usd_floor)
    return df.loc[valid].copy()
