"""
Prediction helpers for the Streamlit form.

The saved Random Forest bundle includes the encoder maps and the
selected feature list so that form values are transformed exactly
the same way as the training data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import CATEGORICAL_COLUMNS, CLASS_ORDER, encode_frame

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"

RISK_BY_CLASS = {
    "Low": "High Risk",
    "Medium": "Moderate Risk",
    "High": "Low Risk",
}


def load_model_bundle(model_path: Path | None = None) -> dict:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` first."
        )
    return joblib.load(path)


def encode_student_row(student: dict, bundle: dict) -> pd.DataFrame:
    """Turn the form dictionary into the numeric vector the forest expects."""
    row = pd.DataFrame([student])
    if "engagement_score" in bundle["feature_columns"] and "engagement_score" not in row:
        row["engagement_score"] = row[
            ["raised_hands", "visited_resources", "announcements_view", "discussion"]
        ].mean(axis=1)
    encoded = encode_frame(row, bundle["category_maps"])
    return encoded[bundle["selected_features"]]


def _factor_messages(student: dict, importances: dict) -> list[str]:
    """Human-readable factors using the actual input values and RF ranks."""
    factors = []
    absence = int(student["student_absence_days"])
    if absence >= 7:
        factors.append(
            f"Absence is {absence} days (paper-highlighted behavioural risk signal)."
        )
    elif absence == 0:
        factors.append("No recorded absence days — a favourable attendance signal.")
    else:
        factors.append(f"Absence is {absence} days — keep attendance consistent.")

    engagement = np.mean(
        [
            student["raised_hands"],
            student["visited_resources"],
            student["announcements_view"],
            student["discussion"],
        ]
    )
    if engagement < 40:
        factors.append("LMS engagement (hands / resources / announcements / discussion) is low.")
    elif engagement > 70:
        factors.append("LMS engagement is strong across the four activity counts.")
    else:
        factors.append("LMS engagement is moderate; extra resource use may still help.")

    if student["parent_answering_survey"] == "No":
        factors.append("Parent has not answered the school survey (parental involvement flag).")
    if student["parent_school_satisfaction"] == "Bad":
        factors.append("Parent school satisfaction is recorded as Bad.")
    if student["relation"] == "Mum":
        factors.append("Responsible parent relation is Mum (often linked with closer follow-up in the paper).")

    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:3]
    top_names = ", ".join(name.replace("_", " ") for name, _ in ranked)
    factors.append(f"Strongest Random Forest features in this project: {top_names}.")
    return factors[:5]


def _recommendation(predicted: str, student: dict) -> str:
    if predicted == "Low":
        return (
            "Prioritise attendance recovery and daily LMS activity: visit course "
            "resources, open announcements, and join discussion. Involve the "
            "responsible parent in a short follow-up plan."
        )
    if predicted == "Medium":
        return (
            "Performance is in the middle band. Increase visited resources and "
            "announcement views, and keep absence days low to move toward High."
        )
    return (
        "Predicted performance is High. Maintain current attendance, resource "
        "visits and parental communication so the trajectory stays stable."
    )


def predict_student_performance(student: dict, bundle: dict | None = None) -> dict:
    """Return the class predicted by the saved Random Forest, plus extras."""
    bundle = bundle or load_model_bundle()
    features = encode_student_row(student, bundle)
    model = bundle["model"]
    class_index = int(model.predict(features)[0])
    predicted = CLASS_ORDER[class_index]

    confidence = None
    class_probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        for index, label in enumerate(CLASS_ORDER):
            class_probabilities[label] = round(float(proba[index]) * 100, 1)
        confidence = class_probabilities[predicted]

    importances = bundle.get("feature_importances", {})
    return {
        "predicted_performance": predicted,
        "risk_level": RISK_BY_CLASS[predicted],
        "confidence_percent": confidence,
        "class_probabilities": class_probabilities,
        "important_factors": _factor_messages(student, importances),
        "recommendation": _recommendation(predicted, student),
        "disclaimer": (
            "Predictions are analytical estimates and should support, "
            "not replace, educator judgement."
        ),
    }


# Kept so CATEGORICAL_COLUMNS import is not flagged as unused by some linters.
FORM_CATEGORICAL_FIELDS = CATEGORICAL_COLUMNS
