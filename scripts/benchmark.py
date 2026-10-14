"""Reproduces experimental results table for GreenlightML."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

def run_benchmark():
    base_dir = os.path.dirname(__file__)
    movies_path = os.path.join(base_dir, "..", "data", "tmdb_5000_movies.csv")
    credits_path = os.path.join(base_dir, "..", "data", "tmdb_5000_credits.csv")

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        parent_movies = os.path.join(base_dir, "..", "..", "data", "tmdb_5000_movies.csv")
        parent_credits = os.path.join(base_dir, "..", "..", "data", "tmdb_5000_credits.csv")
        if os.path.exists(parent_movies) and os.path.exists(parent_credits):
            movies_path, credits_path = parent_movies, parent_credits
        else:
            print("Data files missing in data/. Please run 'python scripts/download_data.py' first.")
            return

    df_m = pd.read_csv(movies_path)
    df_c = pd.read_csv(credits_path)
    df = pd.merge(df_m, df_c, left_on='id', right_on='movie_id')

    # Filter out missing financial records
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df['target'] = (df['revenue'] > df['budget']).astype(int)

    # Date feature extraction
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(df['release_date'].dt.year.median())
    df['release_month'] = df['release_date'].dt.month.fillna(6)

    # Multi-Hot Genres
    def parse_names(json_str):
        try:
            return [x['name'] for x in json.loads(json_str)]
        except Exception:
            return []

    df['genre_list'] = df['genres'].apply(parse_names)
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=[f"genre_{c}" for c in mlb.classes_], index=df.index)

    # Parse Director & Top Cast
    def get_director(crew_str):
        try:
            for x in json.loads(crew_str):
                if x.get('job') == 'Director':
                    return x['name']
        except Exception:
            pass
        return "Unknown"

    df['director'] = df['crew'].apply(get_director)
    top_dirs = set(df['director'].value_counts().head(30).index)
    df['dir_clean'] = df['director'].apply(lambda x: x if x in top_dirs else 'Other')
    dir_dummies = pd.get_dummies(df['dir_clean'], prefix='dir', drop_first=True)

    top_langs = set(df['original_language'].value_counts().head(5).index)
    df['lang_clean'] = df['original_language'].apply(lambda x: x if x in top_langs else 'Other')
    lang_dummies = pd.get_dummies(df['lang_clean'], prefix='lang', drop_first=True)

    # NLP TF-IDF on Overview
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_mat = tfidf.fit_transform(df['overview'].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=[f"tfidf_{i}" for i in range(100)], index=df.index)

    # Base Pre-Release Features
    X_pre = pd.concat([
        pd.DataFrame({
            'log_budget': np.log1p(df['budget']),
            'runtime': df['runtime'].fillna(df['runtime'].median()),
            'release_year': df['release_year'],
            'release_month': df['release_month']
        }, index=df.index),
        genre_dummies,
        dir_dummies,
        lang_dummies,
        tfidf_df
    ], axis=1)

    y = df['target']

    # Post-Release Leaked Features
    X_leak = X_pre.copy()
    X_leak['popularity'] = df['popularity']
    X_leak['vote_average'] = df['vote_average']
    X_leak['vote_count'] = np.log1p(df['vote_count'])

    X_train_pre, X_test_pre, y_train, y_test, idx_train, idx_test = train_test_split(
        X_pre, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    X_train_leak = X_leak.loc[idx_train]
    X_test_leak = X_leak.loc[idx_test]

    scaler_pre = StandardScaler()
    X_train_pre_sc = scaler_pre.fit_transform(X_train_pre)
    X_test_pre_sc = scaler_pre.transform(X_test_pre)

    scaler_leak = StandardScaler()
    X_train_leak_sc = scaler_leak.fit_transform(X_train_leak)
    X_test_leak_sc = scaler_leak.transform(X_test_leak)

    def eval_metric(name, scope, y_true, y_pred, y_prob, test_df):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5

        sel = test_df[y_pred == 1]
        b = sel['budget'].sum()
        r = sel['revenue'].sum()
        roi = (r - b) / b if b > 0 else 0
        return {
            'Model': name, 'Data Scope': scope, 'Accuracy': f"{acc:.4f}",
            'Precision': f"{prec:.4f}", 'Recall': f"{rec:.4f}",
            'F1': f"{f1:.4f}", 'ROC-AUC': f"{auc:.4f}", 'Net ROI': f"+{roi*100:.2f}%"
        }

    results = []
    test_slice = df.loc[idx_test]

    # Baseline: Market Passive Index
    y_pred_dummy = np.ones_like(y_test)
    results.append(eval_metric("Market Passive Index", "Passive", y_test, y_pred_dummy, y_pred_dummy.astype(float), test_slice))

    # Pre-release Logistic Regression
    lr_pre = LogisticRegression(max_iter=1000, random_state=42)
    lr_pre.fit(X_train_pre_sc, y_train)
    results.append(eval_metric("Logistic Regression", "Pre-release", y_test, lr_pre.predict(X_test_pre_sc), lr_pre.predict_proba(X_test_pre_sc)[:, 1], test_slice))

    # Pre-release Random Forest
    rf_pre = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_pre.fit(X_train_pre, y_train)
    results.append(eval_metric("Random Forest", "Pre-release", y_test, rf_pre.predict(X_test_pre), rf_pre.predict_proba(X_test_pre)[:, 1], test_slice))

    # Leaked Logistic Regression
    lr_leak = LogisticRegression(max_iter=1000, random_state=42)
    lr_leak.fit(X_train_leak_sc, y_train)
    results.append(eval_metric("Logistic Regression (LEAKED)", "Post-release", y_test, lr_leak.predict(X_test_leak_sc), lr_leak.predict_proba(X_test_leak_sc)[:, 1], test_slice))

    # Leaked Random Forest
    rf_leak = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_leak.fit(X_train_leak, y_train)
    results.append(eval_metric("Random Forest (LEAKED)", "Post-release", y_test, rf_leak.predict(X_test_leak), rf_leak.predict_proba(X_test_leak)[:, 1], test_slice))

    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
