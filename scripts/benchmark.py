"""Reproduces experimental results table for GreenlightML using src modules."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_and_merge_data, filter_clean_theatrical, filter_scale_anomalies
from src.features import build_features
from src.models import get_logistic_regression, get_random_forest
from src.metrics import evaluate_model

def run_benchmark():
    try:
        df_raw = load_and_merge_data()
    except FileNotFoundError as e:
        print(e)
        return

    df_clean = filter_clean_theatrical(df_raw)
    df_model = filter_scale_anomalies(df_clean)
    X_pre, X_leak, y = build_features(df_model)

    X_tr_pre, X_te_pre, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_pre, y, df_model.index, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_leak = X_leak.loc[idx_tr]
    X_te_leak = X_leak.loc[idx_te]
    test_slice = df_model.loc[idx_te]

    scaler_pre = StandardScaler()
    X_tr_pre_sc = scaler_pre.fit_transform(X_tr_pre)
    X_te_pre_sc = scaler_pre.transform(X_te_pre)

    scaler_leak = StandardScaler()
    X_tr_leak_sc = scaler_leak.fit_transform(X_tr_leak)
    X_te_leak_sc = scaler_leak.transform(X_te_leak)

    results = []

    # 1. Passive Market Index
    y_dummy = np.ones_like(y_te)
    results.append(evaluate_model("Market Passive Index", "Passive", y_te, y_dummy, None, test_slice))

    # 2. Logistic Regression (Pre-release)
    lr_pre = get_logistic_regression()
    lr_pre.fit(X_tr_pre_sc, y_tr)
    results.append(evaluate_model("Logistic Regression", "Pre-release", y_te, lr_pre.predict(X_te_pre_sc), lr_pre.predict_proba(X_te_pre_sc)[:, 1], test_slice))

    # 3. Random Forest (Pre-release)
    rf_pre = get_random_forest()
    rf_pre.fit(X_tr_pre, y_tr)
    results.append(evaluate_model("Random Forest", "Pre-release", y_te, rf_pre.predict(X_te_pre), rf_pre.predict_proba(X_te_pre)[:, 1], test_slice))

    # 4. Logistic Regression (Leaked)
    lr_leak = get_logistic_regression()
    lr_leak.fit(X_tr_leak_sc, y_tr)
    results.append(evaluate_model("Logistic Regression (LEAKED)", "Post-release", y_te, lr_leak.predict(X_te_leak_sc), lr_leak.predict_proba(X_te_leak_sc)[:, 1], test_slice))

    # 5. Random Forest (Leaked)
    rf_leak = get_random_forest()
    rf_leak.fit(X_tr_leak, y_tr)
    results.append(evaluate_model("Random Forest (LEAKED)", "Post-release", y_te, rf_leak.predict(X_te_leak), rf_leak.predict_proba(X_te_leak)[:, 1], test_slice))

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 105)
    print("                              GREENLIGHT-ML: BENCHMARK RECEIPTS")
    print("=" * 105)
    print(df_results.to_string(index=False))
    print("=" * 105 + "\n")

if __name__ == "__main__":
    run_benchmark()
