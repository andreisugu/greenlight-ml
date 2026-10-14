"""Model factory and hyperparameter search spaces for GreenlightML."""
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression(
    C: float = 1.0,
    penalty: Optional[str] = None,
    solver: str = 'lbfgs',
    class_weight: Any = None
) -> LogisticRegression:
    """Instantiates regularized linear baseline without deprecated argument warnings."""
    kwargs = {
        "C": C,
        "solver": solver,
        "class_weight": class_weight,
        "max_iter": 1000,
        "random_state": 42
    }
    if penalty is not None:
        kwargs["penalty"] = penalty
    return LogisticRegression(**kwargs)

def get_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2
) -> RandomForestClassifier:
    """Instantiates nonlinear ensemble baseline."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """Returns systematic hyperparameter exploration grids for cross-validation."""
    return {
        "logistic_regression": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "class_weight": [None, "balanced"],
            "solver": ["lbfgs", "saga"]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"]
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0]
        }
    }
