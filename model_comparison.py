"""
Train and compare individual classifiers and ensemble models.

Models follow the educational ensemble-learning paper as closely as
scikit-learn allows:

Individual classifiers
  - Decision Tree  (J48-style tree; sklearn CART, not WEKA J48)
  - Naive Bayes
  - KNN
  - SVM
  - Multilayer Perceptron

Ensemble models
  - Bagging
  - AdaBoost
  - Random Forest
  - Stacking

Evaluation uses 10-fold stratified cross-validation and the metrics
requested from the paper: Accuracy, Precision, Recall, MAE, ROC-AUC.

Results are computed from THIS implementation. Paper numbers are stored
separately and must never be copied into these tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def build_models() -> dict:
    """Return the classifiers discussed in the paper, with sklearn names."""
    decision_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=8,
        random_state=RANDOM_STATE,
    )
    naive_bayes = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel="rbf", random_state=RANDOM_STATE)
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        max_iter=600,
        random_state=RANDOM_STATE,
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        n_estimators=25,
        random_state=RANDOM_STATE,
    )
    adaboost = AdaBoostClassifier(
        n_estimators=40,
        random_state=RANDOM_STATE,
    )
    random_forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=RANDOM_STATE,
    )
    stacking = StackingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ("nb", GaussianNB()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ],
        final_estimator=LogisticRegression(max_iter=400),
        cv=5,
    )

    return {
        "Decision Tree": {"model": decision_tree, "type": "Individual"},
        "Naive Bayes": {"model": naive_bayes, "type": "Individual"},
        "KNN": {"model": knn, "type": "Individual"},
        "SVM": {"model": svm, "type": "Individual"},
        "MLP": {"model": mlp, "type": "Individual"},
        "Bagging": {"model": bagging, "type": "Ensemble"},
        "AdaBoost": {"model": adaboost, "type": "Ensemble"},
        "Random Forest": {"model": random_forest, "type": "Ensemble"},
        "Stacking": {"model": stacking, "type": "Ensemble"},
    }


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    if y_proba is not None and y_proba.shape[1] >= 2:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def evaluate_models(x: pd.DataFrame, y: np.ndarray, n_splits: int = 10) -> pd.DataFrame:
    """
    10-fold stratified cross-validation for every model.

    Returns a table of mean metrics. These are our experiment results,
    not the numbers printed in the reference paper.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for name, spec in build_models().items():
        fold_rows = []
        for train_idx, test_idx in folds.split(x_values, y_values):
            fitted = clone(spec["model"])
            fitted.fit(x_values[train_idx], y_values[train_idx])
            preds = fitted.predict(x_values[test_idx])
            proba = None
            if hasattr(fitted, "predict_proba"):
                proba = fitted.predict_proba(x_values[test_idx])
            elif hasattr(fitted, "decision_function"):
                scores = fitted.decision_function(x_values[test_idx])
                # Convert OvR decision scores to a softmax-like probability for ROC-AUC.
                if scores.ndim == 1:
                    scores = np.column_stack([-scores, scores])
                e = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = e / e.sum(axis=1, keepdims=True)
            fold_rows.append(_fold_metrics(y_values[test_idx], preds, proba))

        summary = pd.DataFrame(fold_rows).mean()
        records.append(
            {
                "model": name,
                "type": spec["type"],
                "accuracy": round(float(summary["accuracy"]), 4),
                "precision": round(float(summary["precision"]), 4),
                "recall": round(float(summary["recall"]), 4),
                "mae": round(float(summary["mae"]), 4),
                "roc_auc": round(float(summary["roc_auc"]), 4)
                if pd.notna(summary["roc_auc"])
                else None,
            }
        )

    results = pd.DataFrame(records).sort_values("accuracy", ascending=False)
    return results.reset_index(drop=True)


def best_model_name(results: pd.DataFrame) -> str:
    """Return the actual best model from our CV table (not assumed to be RF)."""
    return str(results.iloc[0]["model"])
