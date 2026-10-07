"""
Trainer for Student Academic Risk Classification Model.
Model: RandomForestClassifier
Target: Academic Risk Category ('Low', 'Moderate', 'High')

Strict Integrity Rule:
Trained only on real student academic profiles.
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from ml.feature_engineering import extract_student_academic_history

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
RISK_METRICS_PATH = os.path.join(MODEL_DIR, "risk_model_metrics.json")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def train_risk_classification_model() -> Dict[str, Any]:
    """Train RandomForestClassifier on real students."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    db = get_db()
    try:
        students = db.execute("SELECT id FROM students").fetchall()
        if len(students) < 5:
            return {
                "trained": False,
                "message": f"Insufficient historical student records ({len(students)} found, minimum 5 required) for ML risk classifier."
            }

        rows = []
        for s in students:
            hist = extract_student_academic_history(s["id"])
            # Derive risk target from established institutional benchmarks
            cgpa = hist["cgpa"]
            backlogs = hist["backlog_count"]
            att = hist["attendance_percentage"]

            if cgpa >= 7.5 and backlogs == 0 and att >= 80:
                risk = "Low"
            elif cgpa >= 5.5 and backlogs <= 1 and att >= 70:
                risk = "Moderate"
            else:
                risk = "High"

            feats = {
                "cgpa": cgpa,
                "prev_sgpa": hist["prev_sgpa"],
                "sgpa_trend": hist["sgpa_trend"],
                "backlog_count": float(backlogs),
                "attendance_percentage": att,
                "target_risk": risk
            }
            rows.append(feats)

        df = pd.DataFrame(rows)
        feature_cols = ["cgpa", "prev_sgpa", "sgpa_trend", "backlog_count", "attendance_percentage"]
        X = df[feature_cols]
        y = df["target_risk"]

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = float(accuracy_score(y, preds))

        joblib.dump({
            "model": clf,
            "feature_columns": feature_cols,
            "version": "risk_model_v1"
        }, RISK_MODEL_PATH)

        metrics = {
            "version": "risk_model_v1",
            "accuracy": round(acc, 3),
            "sample_count": len(df)
        }
        with open(RISK_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        return {
            "trained": True,
            "metrics": metrics,
            "message": "Risk model trained successfully."
        }
    finally:
        db.close()


if __name__ == "__main__":
    print(train_risk_classification_model())
