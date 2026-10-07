"""
EduPredict AI - Academic Prediction & Guidance Engine

Features:
1. Final VTU subject-mark prediction
2. ML prediction using trained model
3. Safe rule-based fallback
4. Predicted result / grade point
5. Subject attention analysis
6. Personalized improvement suggestions
7. Predicted SGPA
8. Projected CGPA
9. Prediction history storage
10. Prediction explanation
11. What-if simulation support

IMPORTANT:
- Predictions are guidance estimates, not official VTU results.
- Actual marks entered by the student are never modified by this module.
"""

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import joblib
import numpy as np

from ml.feature_engineering import (
    build_subject_prediction_features,
    extract_student_academic_history,
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_FILE = os.path.join(
    APP_DIR,
    "edupredict.db"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "subject_final_marks_model.pkl"
)

RISK_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "risk_model.pkl"
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Open SQLite database using Row objects so columns
    can be accessed by name.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# VTU GRADE POINT
# ============================================================

def grade_point(total: Optional[float]) -> int:
    """
    Convert total marks into a VTU-style 10-point grade point.

    90+ -> 10
    80-89 -> 9
    70-79 -> 8
    60-69 -> 7
    55-59 -> 6
    50-54 -> 5
    40-49 -> 4
    <40 -> 0
    """

    if total is None:
        return 0

    try:
        t = float(total)
    except (TypeError, ValueError):
        return 0

    if t >= 90:
        return 10
    elif t >= 80:
        return 9
    elif t >= 70:
        return 8
    elif t >= 60:
        return 7
    elif t >= 55:
        return 6
    elif t >= 50:
        return 5
    elif t >= 40:
        return 4

    return 0


def grade_letter(total: Optional[float]) -> str:
    """
    Human-readable grade representation.
    """

    if total is None:
        return "F"

    try:
        t = float(total)
    except (TypeError, ValueError):
        return "F"

    if t >= 90:
        return "O"
    elif t >= 80:
        return "A+"
    elif t >= 70:
        return "A"
    elif t >= 60:
        return "B+"
    elif t >= 55:
        return "B"
    elif t >= 50:
        return "C"
    elif t >= 40:
        return "P"

    return "F"


# ============================================================
# ATTENTION ANALYSIS
# ============================================================

def determine_attention_level(
    predicted_final: float,
    ia1: float,
    ia2: float,
    attendance_pct: float
) -> str:
    """
    Categorize academic attention level.

    Categories:
    - Strong
    - Stable
    - Needs Attention
    - High Attention

    This is an academic guidance indicator, not a diagnosis.
    """

    predicted_final = float(predicted_final or 0.0)
    ia1 = float(ia1 or 0.0)
    ia2 = float(ia2 or 0.0)
    attendance_pct = float(attendance_pct or 0.0)

    if (
        predicted_final < 45
        or (ia1 < 20 and ia2 < 20)
        or attendance_pct < 75.0
    ):
        return "High Attention"

    if (
        predicted_final < 55
        or attendance_pct < 85.0
        or (ia2 < ia1 and ia2 < 25)
    ):
        return "Needs Attention"

    if predicted_final < 75:
        return "Stable"

    return "Strong"


# ============================================================
# IMPROVEMENT SUGGESTIONS
# ============================================================

def generate_improvement_suggestions(
    subject_name: str,
    ia1: float,
    ia2: float,
    internal_perf: float,
    predicted_final: float,
    attendance_pct: float
) -> List[str]:
    """
    Generate evidence-based academic guidance.

    No unsupported causal claims are made.
    """

    suggestions = []

    subject_name = subject_name or "this subject"

    ia1 = float(ia1 or 0.0)
    ia2 = float(ia2 or 0.0)
    internal_perf = float(internal_perf or 0.0)
    predicted_final = float(predicted_final or 0.0)
    attendance_pct = float(attendance_pct or 0.0)

    # --------------------------------------------------------
    # IA PERFORMANCE
    # --------------------------------------------------------

    if ia1 < 20 and ia2 < 20:
        suggestions.append(
            f"Both IA1 ({ia1:.1f}) and IA2 ({ia2:.1f}) "
            f"are relatively low. Give {subject_name} additional "
            f"preparation time and practice previous VTU question papers."
        )

    elif ia2 > ia1:
        suggestions.append(
            f"IA2 ({ia2:.1f}) is higher than IA1 ({ia1:.1f}). "
            f"Continue the study approach that supported this improvement."
        )

    elif ia2 < ia1:
        suggestions.append(
            f"IA2 ({ia2:.1f}) is lower than IA1 ({ia1:.1f}). "
            f"Review important concepts and practice questions before the final examination."
        )

    else:
        suggestions.append(
            f"IA1 and IA2 are currently similar "
            f"({ia1:.1f} and {ia2:.1f}). "
            f"Maintain consistent preparation for {subject_name}."
        )

    # --------------------------------------------------------
    # PREDICTED MARKS
    # --------------------------------------------------------

    if predicted_final < 40:
        suggestions.append(
            "The predicted final mark is below the usual passing "
            "threshold used in this application. Prioritize core "
            "syllabus units and discuss difficult topics with your subject teacher."
        )

    elif predicted_final < 50:
        suggestions.append(
            "The predicted mark is close to the passing range. "
            "Prioritize revision of important syllabus units, "
            "previous VTU question papers, and frequently practiced problems."
        )

    elif predicted_final < 60:
        suggestions.append(
            "The predicted performance is in a moderate range. "
            "Focus on strengthening weaker concepts to improve the final score."
        )

    elif predicted_final >= 75:
        suggestions.append(
            "The predicted performance is strong. "
            "Continue regular revision and practice to maintain consistency."
        )

    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    if attendance_pct < 75.0:
        suggestions.append(
            f"Attendance is currently {attendance_pct:.1f}%, "
            "which is below the 75% threshold used by this application. "
            "Attend remaining classes regularly where possible."
        )

    elif attendance_pct < 85.0:
        suggestions.append(
            f"Attendance is {attendance_pct:.1f}%. "
            "Maintaining regular attendance can provide more classroom exposure."
        )

    # --------------------------------------------------------
    # INTERNAL PERFORMANCE
    # --------------------------------------------------------

    if internal_perf >= 35:
        suggestions.append(
            "Internal assessment performance is relatively strong. "
            "Focus on maintaining this level during final examination preparation."
        )

    elif internal_perf < 25:
        suggestions.append(
            "Internal assessment performance has room for improvement. "
            "Review concepts covered in both internal assessments."
        )

    return suggestions


# ============================================================
# PREDICTION EXPLANATION
# ============================================================

def generate_prediction_explanation(
    prediction_method: str,
    ia1: float,
    ia2: float,
    credits: float,
    attendance_pct: float,
    cgpa: float,
    prev_sgpa: float,
    backlog_count: int
) -> List[str]:
    """
    Explain which available inputs contribute to the prediction.

    This is an explanation of the inputs used by the engine,
    not a claim that any one variable causally determines marks.
    """

    explanation = []

    if prediction_method == "ML":
        explanation.append(
            "The final-mark estimate was generated using the trained "
            "subject prediction model."
        )
    else:
        explanation.append(
            "The trained ML model was unavailable or could not be used, "
            "so the application used its transparent rule-based estimate."
        )

    explanation.append(
        f"IA1: {float(ia1):.1f}"
    )

    explanation.append(
        f"IA2: {float(ia2):.1f}"
    )

    explanation.append(
        f"Internal performance: {(float(ia1) + float(ia2)) / 2.0:.1f}"
    )

    explanation.append(
        f"Attendance used by the engine: {float(attendance_pct):.1f}%"
    )

    explanation.append(
        f"Current academic CGPA history: {float(cgpa):.2f}"
    )

    explanation.append(
        f"Previous SGPA reference: {float(prev_sgpa):.2f}"
    )

    explanation.append(
        f"Recorded backlog count: {int(backlog_count)}"
    )

    explanation.append(
        f"Subject credits: {float(credits):.1f}"
    )

    return explanation


# ============================================================
# FINAL MARK PREDICTION
# ============================================================

def predict_subject_final_marks(
    student_id: str,
    subject_code: str,
    subject_name: str,
    ia1: float = 0.0,
    ia2: float = 0.0,
    credits: float = 3.0,
    ia_avg: Optional[float] = None
) -> Dict[str, Any]:
    """
    Predict FUTURE/FINAL VTU marks for a subject based on Internal Assessment Average.

    ML model:
        subject_final_marks_model.pkl

    Fallback:
        Transparent rule-based estimate scaled from IA average (CIE) + projected SEE.
    """

    # --------------------------------------------------------
    # SAFE INPUT CONVERSION
    # --------------------------------------------------------

    try:
        ia1_val = float(ia1 if ia1 is not None else 0.0)
    except (TypeError, ValueError):
        ia1_val = 0.0

    try:
        ia2_val = float(ia2 if ia2 is not None else 0.0)
    except (TypeError, ValueError):
        ia2_val = 0.0

    try:
        cr_val = float(credits if credits is not None else 3.0)
    except (TypeError, ValueError):
        cr_val = 3.0

    # Keep values inside reasonable ranges (0 - 50 per IA)
    ia1_val = float(np.clip(ia1_val, 0.0, 50.0))
    ia2_val = float(np.clip(ia2_val, 0.0, 50.0))
    cr_val = max(0.0, cr_val)

    if ia_avg is not None:
        try:
            internal_perf = round(float(np.clip(float(ia_avg), 0.0, 50.0)), 2)
        except (TypeError, ValueError):
            internal_perf = round((ia1_val + ia2_val) / 2.0, 2)
    else:
        internal_perf = round(
            (ia1_val + ia2_val) / 2.0,
            2
        )


    # --------------------------------------------------------
    # HISTORICAL DATA
    # --------------------------------------------------------

    hist = extract_student_academic_history(student_id)

    attendance_pct = float(
        hist.get("attendance_percentage", 0.0) or 0.0
    )

    cgpa = float(
        hist.get("cgpa", 0.0) or 0.0
    )

    prev_sgpa = float(
        hist.get("prev_sgpa", 0.0) or 0.0
    )

    backlog_count = int(
        hist.get("backlog_count", 0) or 0
    )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    predicted_final = None

    prediction_method = "RULE_BASED_ESTIMATE"
    model_version = "vtu_rule_scaling_v2"

    model_error = None

    # ========================================================
    # 1. ML MODEL
    # ========================================================

    if os.path.exists(MODEL_PATH):

        try:

            bundle = joblib.load(MODEL_PATH)

            # Support both:
            # {"model": model, "feature_columns": [...]}
            # and a direct model object.

            if isinstance(bundle, dict):

                model = bundle.get("model")

                cols = bundle.get(
                    "feature_columns"
                )

                model_version_from_file = bundle.get(
                    "version",
                    "subject_model_v1"
                )

            else:

                model = bundle
                cols = None
                model_version_from_file = "subject_model_v1"

            if model is None:
                raise ValueError(
                    "Prediction model was not found inside model bundle."
                )

            # ------------------------------------------------
            # Build features
            # ------------------------------------------------

            feats = build_subject_prediction_features(
                student_id=student_id,
                subject_code=subject_code,
                ia1=ia1_val,
                ia2=ia2_val,
                credits=cr_val
            )

            # ------------------------------------------------
            # Feature column compatibility
            # ------------------------------------------------

            if not cols:
                cols = list(feats.keys())

            X_df = pd.DataFrame([[float(feats.get(column, 0.0) or 0.0) for column in cols]], columns=cols)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            pred = model.predict(X_df)[0]

            predicted_final = float(
                np.clip(
                    float(pred),
                    0.0,
                    100.0
                )
            )

            predicted_final = round(
                predicted_final,
                2
            )

            prediction_method = "ML"

            model_version = str(
                model_version_from_file
            )

        except Exception as exc:

            model_error = str(exc)

            predicted_final = None

    # ========================================================
    # 2. RULE-BASED FALLBACK
    # ========================================================

    if predicted_final is None:

        prediction_method = "RULE_BASED_ESTIMATE"
        model_version = "vtu_rule_scaling_v2"

        # ----------------------------------------------------
        # Transparent projection
        #
        # Internal performance represents current assessment
        # performance.
        #
        # Projected SEE is estimated from internal performance
        # and attendance.
        # ----------------------------------------------------

        attendance_bonus = (
            attendance_pct / 100.0
        ) * 2.5

        projected_see = (
            internal_perf * 1.25
        ) + attendance_bonus

        total_projected = (
            internal_perf
            + projected_see
        )

        predicted_final = round(
            float(
                np.clip(
                    total_projected,
                    0.0,
                    100.0
                )
            ),
            2
        )

    # ========================================================
    # RESULT
    # ========================================================

    expected_result = (
        "PASS"
        if predicted_final >= 40.0
        else "FAIL"
    )

    gp = grade_point(
        predicted_final
    )

    letter = grade_letter(
        predicted_final
    )

    # ========================================================
    # ATTENTION
    # ========================================================

    attention = determine_attention_level(
        predicted_final=predicted_final,
        ia1=ia1_val,
        ia2=ia2_val,
        attendance_pct=attendance_pct
    )

    # ========================================================
    # SUGGESTIONS
    # ========================================================

    suggestions = generate_improvement_suggestions(
        subject_name=subject_name,
        ia1=ia1_val,
        ia2=ia2_val,
        internal_perf=internal_perf,
        predicted_final=predicted_final,
        attendance_pct=attendance_pct
    )

    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = generate_prediction_explanation(
        prediction_method=prediction_method,
        ia1=ia1_val,
        ia2=ia2_val,
        credits=cr_val,
        attendance_pct=attendance_pct,
        cgpa=cgpa,
        prev_sgpa=prev_sgpa,
        backlog_count=backlog_count
    )

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    save_prediction(
        student_id=student_id,
        prediction_type="FINAL_MARK",
        predicted_value=predicted_final,
        model_version=model_version,
        prediction_method=prediction_method,
        subject_code=subject_code
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "subject_code": subject_code,
        "subject_name": subject_name,

        "credits": cr_val,

        "ia1": ia1_val,
        "ia2": ia2_val,

        "internal_performance": internal_perf,

        "predicted_final": predicted_final,

        "expected_result": expected_result,

        "grade_point": gp,
        "grade_letter": letter,

        "attention_level": attention,

        "suggestions": suggestions,

        "prediction_method": prediction_method,
        "model_version": model_version,

        "attendance_percentage": round(
            attendance_pct,
            2
        ),

        "historical_cgpa": round(
            cgpa,
            2
        ),

        "previous_sgpa": round(
            prev_sgpa,
            2
        ),

        "backlog_count": backlog_count,

        "explanation": explanation,

        # Useful for debugging but not required
        # to be displayed to students.
        "model_error": model_error
    }


# ============================================================
# PREDICTED SGPA
# ============================================================

def calculate_predicted_sgpa(
    subject_predictions: List[Dict[str, Any]]
) -> float:
    """
    Calculate predicted semester SGPA using:

        SGPA = Σ(Grade Point × Credits) / Σ(Credits)
    """

    weighted = 0.0
    total_credits = 0.0

    for subject in subject_predictions:

        try:
            credits = float(
                subject.get("credits", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            credits = 0.0

        if credits <= 0:
            continue

        try:
            predicted_marks = float(
                subject.get(
                    "predicted_final",
                    0.0
                ) or 0.0
            )
        except (TypeError, ValueError):
            predicted_marks = 0.0

        gp = grade_point(
            predicted_marks
        )

        weighted += gp * credits
        total_credits += credits

    if total_credits <= 0:
        return 0.0

    return round(
        weighted / total_credits,
        2
    )


# ============================================================
# PROJECTED CGPA
# ============================================================

def calculate_projected_cgpa(
    student_id: str,
    predicted_sgpa: float
) -> float:
    """
    Project CGPA by incorporating the predicted current
    semester SGPA with previous recorded semester SGPAs.
    """

    db = get_db()

    try:

        rows = db.execute(
            """
            SELECT sgpa
            FROM semesters
            WHERE student_id = ?
              AND sgpa > 0
            ORDER BY semester_number ASC
            """,
            (student_id,)
        ).fetchall()

        past_sgpas = []

        for row in rows:

            try:
                value = float(
                    row["sgpa"]
                )

                if value > 0:
                    past_sgpas.append(
                        value
                    )

            except (
                TypeError,
                ValueError
            ):
                continue

        try:
            predicted_value = float(
                predicted_sgpa or 0.0
            )
        except (
            TypeError,
            ValueError
        ):
            predicted_value = 0.0

        if predicted_value > 0:
            past_sgpas.append(
                predicted_value
            )

        if not past_sgpas:
            return round(
                predicted_value,
                2
            )

        return round(
            sum(past_sgpas)
            / len(past_sgpas),
            2
        )

    finally:
        db.close()


# ============================================================
# WHAT-IF SIMULATION
# ============================================================

def simulate_subject_prediction(
    student_id: str,
    subject_code: str,
    subject_name: str,
    ia1: float,
    ia2: float,
    credits: float
) -> Dict[str, Any]:
    """
    Run a prediction without modifying student records.

    This is intended for the future "What-If Simulator".

    Example:
        "What if my IA2 improves from 18 to 30?"

    No prediction is saved to the academic prediction history.
    """

    try:
        ia1_val = float(
            ia1 or 0.0
        )
    except (
        TypeError,
        ValueError
    ):
        ia1_val = 0.0

    try:
        ia2_val = float(
            ia2 or 0.0
        )
    except (
        TypeError,
        ValueError
    ):
        ia2_val = 0.0

    try:
        cr_val = float(
            credits or 3.0
        )
    except (
        TypeError,
        ValueError
    ):
        cr_val = 3.0

    # --------------------------------------------------------
    # Get historical data
    # --------------------------------------------------------

    hist = extract_student_academic_history(
        student_id
    )

    attendance_pct = float(
        hist.get(
            "attendance_percentage",
            0.0
        ) or 0.0
    )

    predicted_final = None
    prediction_method = "RULE_BASED_SIMULATION"

    # --------------------------------------------------------
    # Try ML model
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        try:

            bundle = joblib.load(
                MODEL_PATH
            )

            if isinstance(bundle, dict):

                model = bundle.get(
                    "model"
                )

                cols = bundle.get(
                    "feature_columns"
                )

            else:

                model = bundle
                cols = None

            if model is None:
                raise ValueError(
                    "Model unavailable."
                )

            features = build_subject_prediction_features(
                student_id=student_id,
                subject_code=subject_code,
                ia1=ia1_val,
                ia2=ia2_val,
                credits=cr_val
            )

            if not cols:
                cols = list(
                    features.keys()
                )

            X_vec = [[
                float(
                    features.get(
                        c,
                        0.0
                    ) or 0.0
                )
                for c in cols
            ]]

            predicted_final = float(
                model.predict(X_vec)[0]
            )

            predicted_final = round(
                float(
                    np.clip(
                        predicted_final,
                        0.0,
                        100.0
                    )
                ),
                2
            )

            prediction_method = "ML_SIMULATION"

        except Exception:
            predicted_final = None

    # --------------------------------------------------------
    # Rule fallback
    # --------------------------------------------------------

    if predicted_final is None:

        internal_perf = (
            ia1_val + ia2_val
        ) / 2.0

        attendance_bonus = (
            attendance_pct / 100.0
        ) * 2.5

        projected_see = (
            internal_perf * 1.25
        ) + attendance_bonus

        predicted_final = round(
            float(
                np.clip(
                    internal_perf
                    + projected_see,
                    0.0,
                    100.0
                )
            ),
            2
        )

    return {
        "subject_code": subject_code,
        "subject_name": subject_name,
        "credits": cr_val,

        "ia1": ia1_val,
        "ia2": ia2_val,

        "predicted_final": predicted_final,

        "expected_result": (
            "PASS"
            if predicted_final >= 40
            else "FAIL"
        ),

        "grade_point": grade_point(
            predicted_final
        ),

        "grade_letter": grade_letter(
            predicted_final
        ),

        "prediction_method": prediction_method,

        "is_simulation": True
    }


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    student_id: str,
    prediction_type: str,
    predicted_value: float,
    model_version: str,
    prediction_method: str,
    semester_id: Optional[str] = None,
    subject_code: Optional[str] = None
):
    """
    Save a prediction to academic_predictions.

    Database errors are deliberately handled so a prediction
    display failure does not crash the Student Portal.
    """

    db = None

    try:

        db = get_db()

        # ----------------------------------------------------
        # Verify table exists
        # ----------------------------------------------------

        table = db.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'academic_predictions'
            """
        ).fetchone()

        if not table:
            return

        db.execute(
            """
            INSERT INTO academic_predictions (
                id,
                student_id,
                semester_id,
                subject_id,
                prediction_type,
                predicted_value,
                prediction_date,
                model_version,
                prediction_method
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                student_id,
                semester_id or "",
                subject_code or "",
                prediction_type,
                float(predicted_value),
                datetime.now().isoformat(),
                model_version,
                prediction_method
            )
        )

        db.commit()

    except Exception:
        # Prediction should still work even if history
        # storage is unavailable.
        pass

    finally:

        if db is not None:
            db.close()


# ============================================================
# PREDICTION HISTORY
# ============================================================

def get_prediction_history(
    student_id: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Retrieve recent prediction history for a student.
    """

    db = None

    try:

        db = get_db()

        rows = db.execute(
            """
            SELECT
                id,
                student_id,
                semester_id,
                subject_id,
                prediction_type,
                predicted_value,
                prediction_date,
                model_version,
                prediction_method
            FROM academic_predictions
            WHERE student_id = ?
            ORDER BY prediction_date DESC
            LIMIT ?
            """,
            (
                student_id,
                int(limit)
            )
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    except Exception:
        return []

    finally:

        if db is not None:
            db.close()


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status() -> Dict[str, Any]:
    """
    Return information about the currently available
    prediction model.
    """

    exists = os.path.exists(
        MODEL_PATH
    )

    status = {
        "model_available": exists,
        "model_path": MODEL_PATH,
        "model_version": None,
        "error": None
    }

    if not exists:
        status["error"] = (
            "Trained subject prediction model not found."
        )
        return status

    try:

        bundle = joblib.load(
            MODEL_PATH
        )

        if isinstance(bundle, dict):

            status["model_version"] = bundle.get(
                "version",
                "subject_model_v1"
            )

            status["feature_columns"] = bundle.get(
                "feature_columns",
                []
            )

        else:

            status["model_version"] = (
                "subject_model_v1"
            )

    except Exception as exc:

        status["model_available"] = False
        status["error"] = str(exc)

    return status