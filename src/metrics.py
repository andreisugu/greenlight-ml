"""Classification metrics and an explicitly limited gross-to-budget proxy."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

def calculate_gross_to_budget_surplus(df_selected: pd.DataFrame) -> float:
    """
    Computes a gross-to-production-budget surplus proxy:
        (sum(reported gross) - sum(production budget)) / sum(production budget)

    This is not studio net ROI: it excludes exhibitor shares, marketing,
    distribution, financing, and other recoupable costs.
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
    """Return classification measures and a limited gross-to-budget proxy."""
    y_true_values = np.asarray(y_true)
    y_pred_values = np.asarray(y_pred)
    if len(y_true_values) != len(y_pred_values) or len(test_df) != len(y_true_values):
        raise ValueError("y_true, y_pred, and test_df must have the same number of rows")
    y_prob_values = np.asarray(y_prob) if y_prob is not None else None
    if y_prob_values is not None and len(y_prob_values) != len(y_true_values):
        raise ValueError("y_prob must have the same number of rows as y_true")

    acc = accuracy_score(y_true_values, y_pred_values)
    bacc = balanced_accuracy_score(y_true_values, y_pred_values)
    prec = precision_score(y_true_values, y_pred_values, zero_division=0)
    rec = recall_score(y_true_values, y_pred_values, zero_division=0)
    f1_macro = f1_score(y_true_values, y_pred_values, average="macro", zero_division=0)
    auc = roc_auc_score(y_true_values, y_prob_values) if y_prob_values is not None else 0.5

    selected_slice = test_df.iloc[y_pred_values == 1]
    gross_surplus = calculate_gross_to_budget_surplus(selected_slice)

    return {
        "Model": name,
        "Information Scope": scope,
        "Accuracy": round(acc, 4),
        "Balanced Acc": round(bacc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Macro F1": round(f1_macro, 4),
        "ROC-AUC": round(auc, 4),
        "Gross-to-Budget Surplus Proxy": f"{gross_surplus * 100:+.2f}%",
        "Titles Picked": f"{len(selected_slice)} / {len(test_df)}"
    }
