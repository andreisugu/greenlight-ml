"""Data ingestion, deduplication, and sanity cleaning module for TMDB 5000."""
import os
import pandas as pd
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
    """Loads movies and credits, deduplicating keys and identical title columns."""
    if movies_path is None or credits_path is None:
        movies_path, credits_path = get_default_data_paths()

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(
            f"Dataset files missing.\nLooked for:\n - {movies_path}\n - {credits_path}\n"
            "Please run 'python scripts/download_data.py' to download TMDB 5000."
        )

    df_movies = pd.read_csv(movies_path)
    df_credits = pd.read_csv(credits_path)

    # Note: df_credits contains duplicate 'title' and foreign key 'movie_id'
    df_merged = pd.merge(df_movies, df_credits, left_on='id', right_on='movie_id', suffixes=('', '_dup'))
    df_merged.drop(columns=['movie_id', 'title_dup'], inplace=True, errors='ignore')
    return df_merged

def filter_clean_theatrical(df: pd.DataFrame, min_budget: float = 0.0, min_revenue: float = 0.0) -> pd.DataFrame:
    """Filters out unreleased/zero financial entries and optional micro-scale anomalies."""
    clean = df[(df['budget'] > min_budget) & (df['revenue'] > min_revenue)].copy()
    clean['target'] = (clean['revenue'] > clean['budget']).astype(int)
    clean['target_hollywood'] = (clean['revenue'] >= 2.0 * clean['budget']).astype(int)
    return clean
