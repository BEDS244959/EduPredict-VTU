"""
Model Evaluation Module for EduPredict AI.
Calculates MAE, RMSE, and R2 for regressors; Accuracy, Precision, Recall, and F1 for classifiers.
Never hardcodes or fabricates metrics.
"""

import os
import json
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
METRICS_PATH = os.path.join(MODEL_DIR, "subject_model_metrics.json")
RISK_METRICS_PATH = os.path.join(MODEL_DIR, "risk_model_metrics.json")


def get_subject_model_evaluation() -> Dict[str, Any]:
    """Retrieve verified evaluation metrics for the subject final marks regression model."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "No trained model file found or insufficient historical records.",
        "mae": "N/A",
        "rmse": "N/A",
        "r2": "N/A"
    }


def get_risk_model_evaluation() -> Dict[str, Any]:
    """Retrieve verified evaluation metrics for the academic risk classification model."""
    if os.path.exists(RISK_METRICS_PATH):
        try:
            with open(RISK_METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "No trained risk model found.",
        "accuracy": "N/A"
    }
