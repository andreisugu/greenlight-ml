"""Evaluation metrics for statistical classification and simulated portfolio ROI."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

def calculate_portfolio_roi(df_selected: pd.DataFrame) -> float:
    """
    Computes Net Portfolio ROI across selected greenlit titles:
        ROI = (Total Revenue - Total Budget) / Total Budget
    Returns 0.0 if no titles are selected.
    """
    if len(df_selected) == 0:
        return 0.0
    total_budget = df_selected['budget'].sum()
    total_revenue = df_selected['revenue'].sum()
    if total_budget <= 0:
        return 0.0
    return float((total_revenue - total_budget) / total_budget)

def evaluate_model(
    name: str,
    scope: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Generates a complete metrics receipt combining classification and financial returns."""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5

    selected_slice = test_df[y_pred == 1]
    net_roi = calculate_portfolio_roi(selected_slice)

    return {
        "Model": name,
        "Information Scope": scope,
        "Accuracy": round(acc, 4),
        "Balanced Acc": round(bacc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Net Portfolio ROI": f"+{net_roi * 100:.2f}%",
        "Titles Picked": f"{len(selected_slice)} / {len(test_df)}"
    }
