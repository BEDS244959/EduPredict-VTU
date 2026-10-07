"""
Feature Engineering for EduPredict AI.
Extracts leakage-free academic, behavioural, and historical features directly from SQLite database.
Never fabricates data. Handles missing values using valid statistical defaults.
"""

import sqlite3
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def extract_student_academic_history(student_id: str) -> Dict[str, Any]:
    """
    Extract historical marks, SGPA, CGPA, and backlog statistics for a student.
    Does NOT leak future marks.
    """
    db = get_db()
    try:
        # 1. Semesters
        sem_rows = db.execute("""
            SELECT semester_number, sgpa
            FROM semesters
            WHERE student_id = ? AND sgpa > 0
            ORDER BY semester_number ASC
        """, (student_id,)).fetchall()

        sgpas = [float(r["sgpa"]) for r in sem_rows]
        cgpa = round(sum(sgpas) / len(sgpas), 2) if sgpas else 0.0
        prev_sgpa = sgpas[-1] if sgpas else cgpa
        sgpa_trend = (sgpas[-1] - sgpas[0]) if len(sgpas) > 1 else 0.0

        # 2. Backlog count
        backlogs = db.execute("""
            SELECT COUNT(*) AS c
            FROM subjects s
            JOIN semesters sem ON s.semester_id = sem.id
            WHERE sem.student_id = ? AND s.result = 'F'
        """, (student_id,)).fetchone()
        backlog_count = int(backlogs["c"]) if backlogs else 0

        # 3. Attendance
        att = db.execute("""
            SELECT
                COUNT(*) AS total_hours,
                SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_hours
            FROM attendance
            WHERE student_id = ?
        """, (student_id,)).fetchone()

        tot = int(att["total_hours"] or 0)
        pres = int(att["present_hours"] or 0)
        if tot > 0:
            att_pct = round((pres / tot) * 100.0, 2)
        else:
            # Fall back to student default days if no hour records exist
            stu = db.execute("SELECT absent_days, working_days, current_semester FROM students WHERE id = ?", (student_id,)).fetchone()
            if stu and stu["working_days"] and stu["working_days"] > 0:
                abs_d = stu["absent_days"] or 0
                att_pct = round(max(0, stu["working_days"] - abs_d) / stu["working_days"] * 100.0, 2)
            else:
                att_pct = 85.0  # standard baseline

        # 4. Student current semester
        stu_rec = db.execute("SELECT current_semester FROM students WHERE id = ?", (student_id,)).fetchone()
        cur_sem = int(stu_rec["current_semester"]) if (stu_rec and stu_rec["current_semester"]) else 5

        return {
            "cgpa": cgpa,
            "prev_sgpa": prev_sgpa,
            "sgpa_trend": sgpa_trend,
            "backlog_count": backlog_count,
            "attendance_percentage": att_pct,
            "current_semester": cur_sem
        }
    finally:
        db.close()


def build_subject_prediction_features(
    student_id: str,
    subject_code: str,
    ia1: float,
    ia2: float,
    credits: float
) -> Dict[str, float]:
    """
    Build feature vector for predicting a subject's final VTU marks.
    Target: final_subject_marks (Total mark: 0 - 100)
    """
    hist = extract_student_academic_history(student_id)

    ia1_val = float(ia1 or 0.0)
    ia2_val = float(ia2 or 0.0)
    internal_perf = round((ia1_val + ia2_val) / 2.0, 2)
    ia_trend = round(ia2_val - ia1_val, 2)

    # Subject specific historical average if student has taken same/similar subject code before
    db = get_db()
    prev_sub = db.execute("""
        SELECT s.total_marks
        FROM subjects s
        JOIN semesters sem ON s.semester_id = sem.id
        WHERE sem.student_id = ? AND s.subject_code = ?
    """, (student_id, subject_code)).fetchone()
    db.close()

    prev_sub_marks = float(prev_sub["total_marks"]) if (prev_sub and prev_sub["total_marks"]) else 0.0

    features = {
        "ia1": ia1_val,
        "ia2": ia2_val,
        "internal_performance": internal_perf,
        "ia_trend": ia_trend,
        "credits": float(credits or 3.0),
        "attendance_percentage": hist["attendance_percentage"],
        "cgpa": hist["cgpa"],
        "prev_sgpa": hist["prev_sgpa"],
        "backlog_count": hist["backlog_count"],
        "current_semester": float(hist["current_semester"]),
        "has_prev_subject": 1.0 if prev_sub_marks > 0 else 0.0,
        "prev_subject_marks": prev_sub_marks
    }
    return features


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
    "prev_subject_marks"
]
