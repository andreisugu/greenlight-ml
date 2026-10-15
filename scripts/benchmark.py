"""Experimental preview of the modeling pipeline.

Preprocessing is now leakage-safe: the raw rows are split *before* any
data-dependent step, and ``FeatureEncoder`` is fit on the training split only.
Standardization is likewise fit on train and applied to test.

Status: this script is still a preview, not published evidence. The target
proxy, population, and decision contract are not frozen yet, so treat scores
as provisional and do not cite them as validated generalization performance.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_and_merge_data, filter_reported_financials
from src.features import FeatureEncoder
from src.models import get_logistic_regression, get_random_forest
from src.metrics import evaluate_model

RANDOM_STATE = 42


def run_benchmark():
    try:
        df_raw = load_and_merge_data()
    except FileNotFoundError as e:
        print(e)
        return

    df_reported = filter_reported_financials(df_raw)

    # Split the raw rows FIRST so no preprocessing can observe the holdout.
    train_df, test_df = train_test_split(
        df_reported,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_reported["target_nominal"],
    )
    y_tr = train_df["target_nominal"]
    y_te = test_df["target_nominal"]

    encoder = FeatureEncoder(max_tfidf_features=100).fit(train_df)
    X_tr_pre = encoder.transform(train_df)
    X_te_pre = encoder.transform(test_df)
    X_tr_leak = encoder.transform(train_df, include_leakage=True)
    X_te_leak = encoder.transform(test_df, include_leakage=True)

    scaler_pre = StandardScaler()
    X_tr_pre_sc = scaler_pre.fit_transform(X_tr_pre)
    X_te_pre_sc = scaler_pre.transform(X_te_pre)

    scaler_leak = StandardScaler()
    X_tr_leak_sc = scaler_leak.fit_transform(X_tr_leak)
    X_te_leak_sc = scaler_leak.transform(X_te_leak)

    print("PROVISIONAL PREVIEW: leakage-safe preprocessing, but the target/decision contract is not frozen.")
    print("Do not cite these scores as validated generalization performance.\n")
    results = []

    # 1. Passive Market Index
    y_dummy = np.ones_like(y_te)
    results.append(evaluate_model("Market Passive Index", "Passive", y_te, y_dummy, None, test_df))

    # 2. Logistic Regression (Pre-release)
    lr_pre = get_logistic_regression()
    lr_pre.fit(X_tr_pre_sc, y_tr)
    results.append(evaluate_model("Logistic Regression", "Pre-release", y_te, lr_pre.predict(X_te_pre_sc), lr_pre.predict_proba(X_te_pre_sc)[:, 1], test_df))

    # 3. Random Forest (Pre-release)
    rf_pre = get_random_forest()
    rf_pre.fit(X_tr_pre, y_tr)
    results.append(evaluate_model("Random Forest", "Pre-release", y_te, rf_pre.predict(X_te_pre), rf_pre.predict_proba(X_te_pre)[:, 1], test_df))

    # 4. Logistic Regression (Leaked)
    lr_leak = get_logistic_regression()
    lr_leak.fit(X_tr_leak_sc, y_tr)
    results.append(evaluate_model("Logistic Regression (LEAKED)", "Post-release", y_te, lr_leak.predict(X_te_leak_sc), lr_leak.predict_proba(X_te_leak_sc)[:, 1], test_df))

    # 5. Random Forest (Leaked)
    rf_leak = get_random_forest()
    rf_leak.fit(X_tr_leak, y_tr)
    results.append(evaluate_model("Random Forest (LEAKED)", "Post-release", y_te, rf_leak.predict(X_te_leak), rf_leak.predict_proba(X_te_leak)[:, 1], test_df))

    df_results = pd.DataFrame(results)
    print("=" * 110)
    print("                          GREENLIGHT-ML: PREVIEW METRIC (NOT VALIDATED EVIDENCE)")
    print("=" * 110)
    print(df_results.to_string(index=False))
    print("=" * 110)
    print(f"Holdout rows: {len(test_df):,} (20% of {len(df_reported):,} reported-financials rows)\n")


if __name__ == "__main__":
    run_benchmark()
