from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.models import Attendance, CurrentSemesterMark, MLModel, Student, StudentSemester, StudentSubject
from app.services.academic import DEFAULT_SETTINGS, get_group, grade_for_total

FEATURE_COLUMNS = [
    "ia1",
    "ia2",
    "internal_performance",
    "ia_trend",
    "credits",
    "attendance_percentage",
    "cgpa",
    "prev_sgpa",
    "backlog_count",
    "current_semester",
    "has_prev_subject",
    "prev_subject_marks",
]

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def attendance_pct(db: Session, student_id: str) -> float:
    rows = db.query(Attendance).filter(Attendance.student_id == student_id).all()
    if not rows:
        return 0.0
    present = sum(1 for r in rows if r.status == "Present")
    return round(100.0 * present / len(rows), 2)


def academic_history(db: Session, student: Student) -> dict[str, float]:
    sems = (
        db.query(StudentSemester)
        .filter(StudentSemester.student_id == student.id, StudentSemester.sgpa > 0)
        .order_by(StudentSemester.semester_number)
        .all()
    )
    sgpas = [float(s.sgpa) for s in sems]
    cgpa = round(sum(sgpas) / len(sgpas), 2) if sgpas else 0.0
    prev = sgpas[-1] if sgpas else 0.0
    fail = 0
    for s in sems:
        fail += sum(1 for m in s.subject_marks if (m.result or "").upper() == "F")
    return {
        "cgpa": cgpa,
        "prev_sgpa": prev,
        "backlog_count": float(fail),
        "attendance_percentage": attendance_pct(db, student.id),
        "current_semester": float(student.current_semester or 1),
    }


def build_features(
    db: Session, student: Student, subject_code: str, ia1: float, ia2: float, credits: float
) -> dict[str, float]:
    hist = academic_history(db, student)
    prev = (
        db.query(StudentSubject)
        .join(StudentSemester, StudentSubject.semester_id == StudentSemester.id)
        .filter(StudentSemester.student_id == student.id, StudentSubject.subject_code == subject_code)
        .first()
    )
    prev_marks = float(prev.total_marks) if prev and prev.total_marks else 0.0
    ia1v, ia2v = float(ia1 or 0), float(ia2 or 0)
    return {
        "ia1": ia1v,
        "ia2": ia2v,
        "internal_performance": round((ia1v + ia2v) / 2.0, 2),
        "ia_trend": round(ia2v - ia1v, 2),
        "credits": float(credits or 3),
        "attendance_percentage": hist["attendance_percentage"],
        "cgpa": hist["cgpa"],
        "prev_sgpa": hist["prev_sgpa"],
        "backlog_count": hist["backlog_count"],
        "current_semester": hist["current_semester"],
        "has_prev_subject": 1.0 if prev_marks else 0.0,
        "prev_subject_marks": prev_marks,
    }


def rule_based_estimate(feats: dict[str, float], academic: dict | None = None) -> float:
    """Transparent baseline: scale internals toward a 100-mark paper and blend history."""
    cfg = academic or DEFAULT_SETTINGS["academic"]
    ia_max = float(cfg.get("ia_max", 50)) or 50
    internal = (feats["ia1"] + feats["ia2"]) / 2.0
    scaled = (internal / ia_max) * 100.0 if ia_max else internal
    hist = feats["prev_sgpa"] * 10.0 if feats["prev_sgpa"] else scaled
    att = feats["attendance_percentage"]
    att_adj = 0.0
    if att and att < 75:
        att_adj = -4.0
    elif att and att < 85:
        att_adj = -1.5
    pred = 0.55 * scaled + 0.30 * hist + 0.15 * (feats["prev_subject_marks"] or scaled)
    pred += att_adj - 1.5 * feats["backlog_count"]
    return float(np.clip(round(pred, 2), 0, 100))


def active_model(db: Session) -> MLModel | None:
    return (
        db.query(MLModel)
        .filter(MLModel.is_active.is_(True), MLModel.file_path.isnot(None))
        .order_by(MLModel.trained_at.desc())
        .first()
    )


def collect_training_frame(db: Session) -> list[dict[str, float]]:
    rows = (
        db.query(StudentSubject, StudentSemester, Student)
        .join(StudentSemester, StudentSubject.semester_id == StudentSemester.id)
        .join(Student, StudentSemester.student_id == Student.id)
        .filter(StudentSubject.total_marks.isnot(None), StudentSubject.total_marks > 0)
        .all()
    )
    data = []
    for sub, _sem, student in rows:
        internal = float(sub.internal_marks or 0)
        feats = build_features(db, student, sub.subject_code, internal, internal, float(sub.credits or 3))
        feats["target"] = float(sub.total_marks)
        data.append(feats)
    return data


def train_models(db: Session) -> dict[str, Any]:
    cfg = get_group(db, "ml")
    min_samples = int(cfg.get("min_samples", 10))
    data = collect_training_frame(db)
    if len(data) < min_samples:
        return {
            "ok": False,
            "message": f"Not enough historical data for reliable ML prediction. Need at least {min_samples} labelled subject totals, found {len(data)}.",
            "training_size": len(data),
        }
    import pandas as pd

    df = pd.DataFrame(data)
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=120, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=120, random_state=42),
    }
    results = []
    best_name, best_model, best_r2 = None, None, -1e9
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        results.append({"algorithm": name, "mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3)})
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    path = MODEL_DIR / f"subject_model_{version}.joblib"
    joblib.dump({"model": best_model, "features": FEATURE_COLUMNS, "algorithm": best_name}, path)
    metrics = next(r for r in results if r["algorithm"] == best_name)
    db.query(MLModel).update({MLModel.is_active: False})
    rec = MLModel(
        name="subject_final_marks",
        version=version,
        algorithm=best_name or "RandomForestRegressor",
        metrics_json=json.dumps({"selected": metrics, "all": results}),
        feature_list=json.dumps(FEATURE_COLUMNS),
        file_path=str(path),
        training_size=len(data),
        is_active=True,
        trained_at=datetime.now(timezone.utc),
    )
    db.add(rec)
    db.commit()
    return {"ok": True, "model": {"version": version, "algorithm": best_name, **metrics, "training_size": len(data), "comparisons": results}}


def predict_subject(db: Session, student: Student, mark: CurrentSemesterMark) -> dict[str, Any]:
    academic = get_group(db, "academic")
    feats = build_features(db, student, mark.subject_code, mark.ia1, mark.ia2, mark.credits)
    method = "RULE"
    model_meta = None
    rec = active_model(db)
    predicted = None
    if rec and rec.file_path and Path(rec.file_path).exists():
        try:
            bundle = joblib.load(rec.file_path)
            model = bundle["model"]
            cols = bundle.get("features", FEATURE_COLUMNS)
            X = np.array([[feats[c] for c in cols]], dtype=float)
            predicted = float(np.clip(model.predict(X)[0], 0, 100))
            method = "ML"
            model_meta = {
                "version": rec.version,
                "algorithm": rec.algorithm,
                "training_size": rec.training_size,
                "metrics": json.loads(rec.metrics_json),
                "trained_at": rec.trained_at.isoformat() if rec.trained_at else None,
            }
        except Exception:
            predicted = None
    if predicted is None:
        predicted = rule_based_estimate(feats, academic)
        method = "RULE"
    letter, gp = grade_for_total(predicted, academic)
    reasons = []
    recs = []
    if feats["ia2"] < feats["ia1"]:
        reasons.append("IA2 performance is below IA1.")
        recs.append("Revise units assessed in IA2 and practise recent VTU questions.")
    if feats["ia1"] < 20 and feats["ia2"] < 20:
        reasons.append("Both internals are below 20.")
        recs.append(f"Increase weekly study time for {mark.subject_name}.")
    if feats["attendance_percentage"] and feats["attendance_percentage"] < 75:
        reasons.append(f"Attendance is {feats['attendance_percentage']}% (below 75%).")
        recs.append("Attend remaining classes regularly where possible.")
    if feats["prev_sgpa"] and feats["prev_sgpa"] < 6:
        reasons.append("Previous SGPA is in a lower range.")
    if not reasons:
        reasons.append("Prediction used IA performance, historical SGPA/CGPA, attendance, and credits.")
    if predicted < 45 or (feats["ia1"] < 20 and feats["ia2"] < 20) or (
        feats["attendance_percentage"] and feats["attendance_percentage"] < 75
    ):
        risk = "High Risk"
    elif predicted < 55:
        risk = "Needs Attention"
    elif predicted < 75:
        risk = "Stable"
    else:
        risk = "Strong"
    recs.append(f"Suggested weekly practice: {2 if risk in ('Needs Attention', 'High Risk') else 1} hours on core units.")
    recs.append("This is study guidance, separate from the numeric estimate.")
    return {
        "subject_code": mark.subject_code,
        "subject_name": mark.subject_name,
        "ia1": mark.ia1,
        "ia2": mark.ia2,
        "credits": mark.credits,
        "current_internal": round((float(mark.ia1) + float(mark.ia2)) / 2.0, 2),
        "predicted_final_marks": round(predicted, 2),
        "expected_grade": letter,
        "expected_grade_point": gp,
        "risk_level": risk,
        "method": method,
        "method_label": "ML Prediction" if method == "ML" else "Rule-based Estimate",
        "low_confidence": method != "ML",
        "factors": [
            "IA1 performance",
            "IA2 performance",
            "Previous semester performance",
            "Attendance",
            "Historical subject marks where available",
        ],
        "why_flagged": reasons,
        "recommendations": recs,
        "model": model_meta,
        "disclaimer": "AI predictions are estimates and should not be treated as guaranteed examination results.",
        "features_used": feats,
    }
