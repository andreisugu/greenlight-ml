"""Feature engineering pipeline for multimodal metadata and NLP synopsis embeddings."""
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_json_list(json_str: str, key: str = 'name') -> list:
    """Safely parses JSON-formatted attribute strings."""
    try:
        data = json.loads(json_str)
        return [item[key] for item in data if key in item]
    except Exception:
        return []

def extract_director(crew_str: str) -> str:
    """Extracts primary director from credits crew array."""
    try:
        crew = json.loads(crew_str)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name', 'Unknown')
    except Exception:
        pass
    return "Unknown"

def extract_top_cast(cast_str: str, n: int = 3) -> list:
    """Extracts top N billed actors from credits cast array."""
    try:
        cast = json.loads(cast_str)
        return [member.get('name', '') for member in cast[:n] if 'name' in member]
    except Exception:
        return []

def build_features(df: pd.DataFrame, max_tfidf_features: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Constructs both clean pre-release feature store (X_pre) and post-release
    contaminated leakage store (X_leak) alongside binary target (y).
    """
    df = df.copy()

    # 1. Temporal decomposition
    df['release_date_dt'] = pd.to_datetime(df['release_date'], errors='coerce')
    release_year = df['release_date_dt'].dt.year.fillna(df['release_date_dt'].dt.year.median())
    release_month = df['release_date_dt'].dt.month.fillna(6)

    # 2. Multi-Hot Genres
    df['genre_names'] = df['genres'].apply(lambda x: parse_json_list(x, key='name'))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre_names'])
    df_genres = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_], index=df.index)

    # 3. Director & Languages
    df['director'] = df['crew'].apply(extract_director)
    top_directors = set(df['director'].value_counts().head(30).index)
    df['dir_clean'] = df['director'].apply(lambda d: d if d in top_directors else 'Other')
    df_dir = pd.get_dummies(df['dir_clean'], prefix='dir', drop_first=True)

    top_languages = set(df['original_language'].value_counts().head(5).index)
    df['lang_clean'] = df['original_language'].apply(lambda l: l if l in top_languages else 'Other')
    df_lang = pd.get_dummies(df['lang_clean'], prefix='lang', drop_first=True)

    # 4. NLP TF-IDF on Synopsis
    tfidf = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(max_tfidf_features)], index=df.index)

    # Base Pre-Release DataFrame
    X_pre = pd.concat([
        pd.DataFrame({
            'log_budget': np.log1p(df['budget']),
            'runtime': df['runtime'].fillna(df['runtime'].median()),
            'release_year': release_year,
            'release_month': release_month
        }, index=df.index),
        df_genres,
        df_dir,
        df_lang,
        df_tfidf
    ], axis=1)

    # Leaked DataFrame (Quarantined post-release features)
    X_leak = X_pre.copy()
    X_leak['popularity'] = df['popularity'].fillna(0)
    X_leak['vote_average'] = df['vote_average'].fillna(0)
    X_leak['vote_count'] = np.log1p(df['vote_count'].fillna(0))

    y = df['target']
    return X_pre, X_leak, y
