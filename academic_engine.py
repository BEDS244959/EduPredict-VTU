"""
Shared Academic, Segregation, Automated Data Entry & Placement Engine for EduPredict AI.

Provides a single source of truth for:
- Configurable VTU institutional rules (low attendance, max backlogs, min credits)
- Automated VTU marksheet saving, grade/SGPA calculation, and backlog sync
- Faculty individual and bulk data entry (VTU results, IA marks, attendance, master subjects)
- Student segregation by SGPA, Marks, Credits, Next-Sem Eligibility, Low Attendance, and Placements
- Placement tracking (individual CRUD, offer letter OCR mapping, and Excel/CSV bulk import)
"""

import math
import os
import sqlite3
import uuid
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from db_migrations import get_db_connection
from ml.predict import grade_letter, grade_point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "edupredict.db")

PLACED_STATUSES = ("Placed", "Offer Received", "Internship + PPO")
ALL_PLACEMENT_STATUSES = (
    "Placed",
    "Offer Received",
    "Internship + PPO",
    "Shortlisted",
    "Pending",
    "Not Placed",
    "Higher Studies",
    "Opted Out",
)

DEFAULT_RULES: Dict[str, Any] = {
    "low_attendance_threshold": 75.0,
    "condonable_min_attendance_pct": 65.0,
    "exam_min_attendance_pct": 75.0,
    "exam_min_cie_pct": 40.0,
    "max_backlogs_next_sem": 4,
    "min_credits_next_sem": 16.0,
    "enforce_credit_pct_mode": "threshold",
    "min_credit_pct_next_sem": 50.0,
    "vtu_scheme": "VTU 2022 Scheme",
    "enforce_year_back_rule": "1",
}


def calculate_classes_needed(present: int, total: int, target_pct: float = 75.0) -> int:
    """Compute consecutive classes a student must attend to reach target attendance percentage."""
    if total <= 0 or target_pct <= 0.0:
        return 0
    if target_pct >= 100.0:
        return 0 if present == total else 999
    if (present / total) * 100.0 >= target_pct:
        return 0
    p = target_pct / 100.0
    needed = (p * total - present) / (1.0 - p)
    return max(0, int(math.ceil(needed)))


def _get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open SQLite connection with row factory enabled."""
    return get_db_connection(db_path or DEFAULT_DB_PATH)


def clear_streamlit_caches() -> None:
    """Safely clear Streamlit @st.cache_data caches after DB mutations."""
    try:
        import streamlit as st
        st.cache_data.clear()
    except Exception:
        pass


def is_pass_result(result_val: Any, total_marks: Optional[float] = None) -> bool:
    """Normalize pass status across legacy ('P'/'F') and extended ('PASS'/'FAIL') rows."""
    res = str(result_val or "").strip().upper()
    if res in ("P", "PASS"):
        return True
    if res in ("F", "FAIL", "AB", "W", "ABSENT", "WITHHELD"):
        return False
    if total_marks is not None:
        return float(total_marks) >= 40.0
    return False


def normalize_result_code(result_val: Any, total_marks: Optional[float] = None) -> str:
    """Return canonical 'P' or 'F' string."""
    return "P" if is_pass_result(result_val, total_marks) else "F"


# ============================================================
# CONFIGURABLE INSTITUTIONAL ACADEMIC RULES
# ============================================================

def get_academic_rules(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Load institutional segregation & promotion thresholds from academic_rules table."""
    rules = dict(DEFAULT_RULES)
    db = _get_conn(db_path)
    try:
        rows = db.execute("SELECT rule_key, rule_value FROM academic_rules").fetchall()
        float_keys = {
            "low_attendance_threshold",
            "condonable_min_attendance_pct",
            "exam_min_attendance_pct",
            "exam_min_cie_pct",
            "min_credits_next_sem",
            "min_credit_pct_next_sem",
        }
        for r in rows:
            k = r["rule_key"]
            v = r["rule_value"]
            if k in float_keys:
                try:
                    rules[k] = float(v)
                except Exception:
                    pass
            elif k == "max_backlogs_next_sem":
                try:
                    rules[k] = int(float(v))
                except Exception:
                    pass
            elif k in ("enforce_credit_pct_mode", "vtu_scheme", "enforce_year_back_rule"):
                rules[k] = str(v).strip() or str(DEFAULT_RULES.get(k, ""))
    except Exception:
        pass
    finally:
        db.close()
    return rules


def save_academic_rules(
    rules: Dict[str, Any],
    updated_by: str = "faculty",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist institutional segregation & next-semester eligibility thresholds."""
    merged = get_academic_rules(db_path)
    merged.update(rules or {})
    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    try:
        for k in DEFAULT_RULES:
            val_str = str(merged.get(k, DEFAULT_RULES[k]))
            db.execute(
                """
                INSERT INTO academic_rules (rule_key, rule_value, description, updated_by, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(rule_key) DO UPDATE SET
                    rule_value = excluded.rule_value,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (k, val_str, f"Rule setting for {k}", updated_by, now),
            )
        db.commit()
    finally:
        db.close()
    clear_streamlit_caches()
    return merged


# ============================================================
# AUTOMATED VTU RESULT SAVE & BACKLOG SYNC
# ============================================================

def sync_student_backlogs_for_subjects(
    db: sqlite3.Connection,
    student_id: str,
    sem_num: int,
    subjects_list: List[Dict[str, Any]],
    now: str,
) -> Dict[str, int]:
    """Automatically activate backlogs for failed subjects and clear re-passed backlogs."""
    activated = 0
    cleared = 0
    existing_rows = db.execute(
        "SELECT id, subject_code, status FROM backlogs WHERE student_id = ?",
        (student_id,),
    ).fetchall()
    existing_map = {str(r["subject_code"]).strip().upper(): dict(r) for r in existing_rows}

    for sub in subjects_list:
        code = str(sub.get("subject_code", "")).strip().upper()
        name = str(sub.get("subject_name", "")).strip() or code
        credits = float(sub.get("credits") or 3.0)
        if not code:
            continue
        passed = is_pass_result(sub.get("result"), sub.get("total_marks"))
        rec = existing_map.get(code)
        if not passed:
            if rec:
                db.execute(
                    """
                    UPDATE backlogs
                    SET status = 'Active', subject_name = ?, semester_number = ?, previous_result = 'FAIL', credits = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, sem_num, credits, now, rec["id"]),
                )
            else:
                db.execute(
                    """
                    INSERT INTO backlogs (
                        id, student_id, semester_number, subject_code, subject_name,
                        previous_result, rewritten_exam, updated_result, credits, status, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'FAIL', 0, '', ?, 'Active', ?)
                    """,
                    (uuid.uuid4().hex, student_id, sem_num, code, name, credits, now),
                )
            activated += 1
        elif rec and str(rec.get("status", "")).strip().lower() != "cleared":
            db.execute(
                """
                UPDATE backlogs
                SET status = 'Cleared', rewritten_exam = 'Regular / Re-Attempt', updated_result = 'PASS', updated_at = ?
                WHERE id = ?
                """,
                (now, rec["id"]),
            )
            cleared += 1

    return {"activated": activated, "cleared": cleared}


def _compute_semester_metrics(subjects: List[Dict[str, Any]], official_sgpa: Optional[float] = None) -> Dict[str, Any]:
    """Normalize subject rows and calculate credits, marks, and VTU SGPA."""
    normalized: List[Dict[str, Any]] = []
    credits_registered = 0.0
    credits_earned = 0.0
    weighted_pts = 0.0
    total_marks_obtained = 0.0
    total_max_marks = 0.0
    passed_count = 0
    failed_count = 0

    for raw in subjects:
        code = str(raw.get("subject_code", "")).strip().upper()
        name = str(raw.get("subject_name", "")).strip() or code
        if not code:
            continue
        internal = float(raw.get("internal_marks") or 0.0)
        external = float(raw.get("external_marks") or 0.0)
        tot_raw = raw.get("total_marks")
        total = float(tot_raw) if tot_raw is not None and float(tot_raw) > 0 else (internal + external)
        credits = float(raw.get("credits") if raw.get("credits") is not None else 4.0)
        res_code = normalize_result_code(raw.get("result"), total)

        if res_code == "F":
            gp = 0
            gr = "F"
            failed_count += 1
        else:
            gp = grade_point(total)
            gr = grade_letter(total)
            passed_count += 1
            credits_earned += credits

        credits_registered += credits
        weighted_pts += gp * credits
        total_marks_obtained += total
        total_max_marks += 100.0

        normalized.append(
            {
                "subject_code": code,
                "subject_name": name,
                "internal_marks": round(internal, 2),
                "external_marks": round(external, 2),
                "total_marks": round(total, 2),
                "credits": round(credits, 2),
                "result": res_code,
                "grade": gr,
                "grade_point": gp,
            }
        )

    calc_sgpa = round(weighted_pts / credits_registered, 2) if credits_registered > 0 else 0.0
    final_sgpa = round(float(official_sgpa), 2) if (official_sgpa is not None and float(official_sgpa) > 0) else calc_sgpa

    return {
        "subjects": normalized,
        "sgpa": final_sgpa,
        "calculated_sgpa": calc_sgpa,
        "credits_registered": round(credits_registered, 2),
        "credits_earned": round(credits_earned, 2),
        "total_marks_obtained": round(total_marks_obtained, 2),
        "total_max_marks": round(total_max_marks, 2),
        "passed_subjects": passed_count,
        "failed_subjects": failed_count,
    }


def save_student_semester_result(
    student_id: str,
    sem_num: int,
    subjects: List[Dict[str, Any]],
    official_sgpa: Optional[float] = None,
    marksheet_path: str = "",
    verification_status: str = "Pending",
    actor_type: str = "student",
    actor_id: str = "",
    notify_faculty: bool = True,
    notify_student: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Atomically save a student's VTU semester subjects & SGPA, sync backlogs,
    send automatic notifications, log audit, and clear caches for instant Faculty segregation.
    """
    from auth import log_audit_action, send_notification

    computed = _compute_semester_metrics(subjects, official_sgpa=official_sgpa)
    if not computed["subjects"]:
        raise ValueError("No valid subjects supplied to save.")

    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    stu_usn = student_id
    stu_name = ""
    try:
        stu = db.execute("SELECT name, usn FROM students WHERE id = ?", (student_id,)).fetchone()
        if stu:
            stu_usn = stu["usn"]
            stu_name = stu["name"]

        existing_sem = db.execute(
            "SELECT id, sgpa, marksheet_path FROM semesters WHERE student_id = ? AND semester_number = ?",
            (student_id, int(sem_num)),
        ).fetchone()

        effective_path = marksheet_path or (existing_sem["marksheet_path"] if existing_sem else "") or ""

        if existing_sem:
            sem_id = existing_sem["id"]
            old_sgpa = float(existing_sem["sgpa"] or 0.0)
            old_subs = db.execute("SELECT credits, result, total_marks FROM subjects WHERE semester_id = ?", (sem_id,)).fetchall()
            old_cred_earned = round(sum(float(x["credits"] or 0.0) for x in old_subs if is_pass_result(x["result"], x["total_marks"])), 2)
            if abs(old_sgpa - computed["sgpa"]) >= 0.01 or abs(old_cred_earned - computed["credits_earned"]) >= 0.1:
                try:
                    db.execute(
                        """
                        INSERT INTO marksheet_revisions (
                            id, student_id, semester_number, old_sgpa, new_sgpa,
                            old_credits_earned, new_credits_earned, actor_type, actor_id, notes, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            student_id,
                            int(sem_num),
                            round(old_sgpa, 2),
                            computed["sgpa"],
                            old_cred_earned,
                            computed["credits_earned"],
                            actor_type,
                            actor_id or student_id,
                            f"Sem {sem_num} updated ({old_sgpa:.2f} -> {computed['sgpa']:.2f})",
                            now,
                        ),
                    )
                except Exception:
                    pass
            db.execute("DELETE FROM subjects WHERE semester_id = ?", (sem_id,))
            db.execute(
                """
                UPDATE semesters
                SET semester_name = ?, sgpa = ?, marksheet_path = ?, credits_total = ?, verification_status = ?
                WHERE id = ?
                """,
                (
                    f"Semester {sem_num}",
                    computed["sgpa"],
                    effective_path,
                    computed["credits_registered"],
                    verification_status,
                    sem_id,
                ),
            )
        else:
            sem_id = uuid.uuid4().hex
            db.execute(
                """
                INSERT INTO semesters (id, student_id, semester_number, semester_name, sgpa, marksheet_path, is_current, credits_total, verification_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    sem_id,
                    student_id,
                    int(sem_num),
                    f"Semester {sem_num}",
                    computed["sgpa"],
                    effective_path,
                    computed["credits_registered"],
                    verification_status,
                    now,
                ),
            )

        for sub in computed["subjects"]:
            db.execute(
                """
                INSERT INTO subjects (id, semester_id, subject_code, subject_name, internal_marks, external_marks, total_marks, credits, result, grade, grade_point)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    sem_id,
                    sub["subject_code"],
                    sub["subject_name"],
                    sub["internal_marks"],
                    sub["external_marks"],
                    sub["total_marks"],
                    sub["credits"],
                    sub["result"],
                    sub["grade"],
                    sub["grade_point"],
                ),
            )

        backlog_stats = sync_student_backlogs_for_subjects(db, student_id, int(sem_num), computed["subjects"], now)
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()

    log_audit_action(
        actor_type,
        actor_id or student_id,
        "SAVE_MARKSHEET",
        f"Saved Sem {sem_num} VTU Result for {stu_usn}: SGPA {computed['sgpa']:.2f}, Earned Credits {computed['credits_earned']:g}/{computed['credits_registered']:g}, Failed {computed['failed_subjects']}",
        target_id=student_id,
    )

    if notify_faculty:
        send_notification(
            "staff",
            "all",
            f"New VTU Result Uploaded: {stu_usn} (Sem {sem_num})",
            f"{stu_name} ({stu_usn}) uploaded Semester {sem_num} results. SGPA: {computed['sgpa']:.2f} | Earned Credits: {computed['credits_earned']:g} | Active Backlogs Synced: {backlog_stats['activated']}. Dashboard segregation updated automatically.",
            category="VTU Result",
        )

    if notify_student:
        send_notification(
            "student",
            student_id,
            f"Semester {sem_num} VTU Result Recorded",
            f"Faculty recorded your Semester {sem_num} VTU result (SGPA: {computed['sgpa']:.2f}, Credits Earned: {computed['credits_earned']:g}).",
            category="VTU Result",
        )

    return {
        "semester_id": sem_id,
        "sgpa": computed["sgpa"],
        "calculated_sgpa": computed["calculated_sgpa"],
        "credits_registered": computed["credits_registered"],
        "credits_earned": computed["credits_earned"],
        "total_marks_obtained": computed["total_marks_obtained"],
        "total_max_marks": computed["total_max_marks"],
        "passed_subjects": computed["passed_subjects"],
        "failed_subjects": computed["failed_subjects"],
        "backlogs_activated": backlog_stats["activated"],
        "backlogs_cleared": backlog_stats["cleared"],
    }


def save_student_ia_subject_mark(
    student_id: str,
    semester_number: int,
    subject_code: str,
    subject_name: str,
    credits: float,
    ia1: float,
    ia2: float,
    ia3: float = 0.0,
    assignment_1: float = 0.0,
    assignment_2: float = 0.0,
    assignment_marks: Optional[float] = None,
    cie_total: Optional[float] = None,
    actor_id: str = "faculty",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a student's individual current-semester IA tests & assignments record."""
    from auth import log_audit_action

    code = subject_code.strip().upper()
    if not code:
        raise ValueError("Subject code is required.")
    ia1_v = float(ia1 or 0.0)
    ia2_v = float(ia2 or 0.0)
    ia3_v = float(ia3 or 0.0)
    a1_v = float(assignment_1 or 0.0)
    a2_v = float(assignment_2 or 0.0)

    # Compute assignment component
    eff_assign = float(assignment_marks) if assignment_marks is not None else round(a1_v + a2_v, 2)

    # Compute test component (best 2 if ia3 > 0, else average of ia1 and ia2)
    if ia3_v > 0.0:
        top2 = sorted([ia1_v, ia2_v, ia3_v], reverse=True)[:2]
        test_comp = round(sum(top2) / 2.0, 2)
    else:
        test_comp = round((ia1_v + ia2_v) / 2.0, 2)

    # Calculate total CIE out of 50
    if cie_total is not None and float(cie_total) > 0.0:
        final_cie = round(float(cie_total), 2)
    else:
        final_cie = round(test_comp + eff_assign, 2)

    cie_eligible = bool(final_cie >= 20.0)
    now = datetime.now().isoformat()

    db = _get_conn(db_path)
    try:
        existing = db.execute(
            """
            SELECT id FROM current_semester_records
            WHERE student_id = ? AND semester_number = ? AND UPPER(subject_code) = ?
            """,
            (student_id, int(semester_number), code),
        ).fetchone()
        if existing:
            db.execute(
                """
                UPDATE current_semester_records
                SET subject_name = ?, credits = ?, ia1 = ?, ia2 = ?, ia3 = ?,
                    assignment_1 = ?, assignment_2 = ?, assignment_marks = ?,
                    cie_total = ?, internal_performance = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    subject_name.strip() or code,
                    float(credits or 4.0),
                    ia1_v,
                    ia2_v,
                    ia3_v,
                    a1_v,
                    a2_v,
                    eff_assign,
                    final_cie,
                    final_cie,
                    now,
                    existing["id"],
                ),
            )
        else:
            db.execute(
                """
                INSERT INTO current_semester_records (
                    id, student_id, semester_number, subject_code, subject_name,
                    credits, ia1, ia2, ia3, assignment_1, assignment_2,
                    assignment_marks, cie_total, internal_performance, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    student_id,
                    int(semester_number),
                    code,
                    subject_name.strip() or code,
                    float(credits or 4.0),
                    ia1_v,
                    ia2_v,
                    ia3_v,
                    a1_v,
                    a2_v,
                    eff_assign,
                    final_cie,
                    final_cie,
                    now,
                ),
            )
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()
    log_audit_action(
        "staff",
        actor_id,
        "FACULTY_IA_ENTRY",
        f"Saved IA & Assignments for {code} (Sem {semester_number}): IA1={ia1_v}, IA2={ia2_v}, IA3={ia3_v}, Assign={eff_assign}, CIE={final_cie}",
        target_id=student_id,
    )
    return {
        "cie_total": final_cie,
        "test_component": test_comp,
        "assignment_component": eff_assign,
        "cie_eligible": cie_eligible,
    }


def bulk_save_class_ia_and_assignments(
    class_id: str,
    semester_number: int,
    subject_code: str,
    subject_name: str,
    credits: float,
    records: List[Dict[str, Any]],
    actor_id: str = "faculty",
    db_path: Optional[str] = None,
) -> int:
    """Batch save IA tests and assignment marks for an entire class of students."""
    count = 0
    for r in records:
        sid = r.get("student_id")
        if not sid:
            continue
        save_student_ia_subject_mark(
            student_id=sid,
            semester_number=int(semester_number),
            subject_code=subject_code,
            subject_name=subject_name,
            credits=float(credits or 4.0),
            ia1=float(r.get("ia1") or 0.0),
            ia2=float(r.get("ia2") or 0.0),
            ia3=float(r.get("ia3") or 0.0),
            assignment_1=float(r.get("assignment_1") or 0.0),
            assignment_2=float(r.get("assignment_2") or 0.0),
            assignment_marks=float(r["assignment_marks"]) if r.get("assignment_marks") is not None else None,
            cie_total=float(r["cie_total"]) if r.get("cie_total") is not None else None,
            actor_id=actor_id,
            db_path=db_path,
        )
        count += 1
    return count


def recalculate_all_student_records(
    class_id: Optional[str] = None,
    actor_id: str = "faculty",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Force complete end-to-end recalculation of academic metrics, credits, and segregations."""
    from auth import log_audit_action

    c_ids = [class_id] if class_id else None
    df = get_comprehensive_student_analytics(class_ids=c_ids, db_path=db_path)
    seg = segregate_students(df)
    clear_streamlit_caches()
    log_audit_action("staff", actor_id, "RECALCULATE_ALL_RECORDS", f"Recalculated records for {len(df)} students")
    return {
        "total_students": len(df),
        "passed_count": seg["summary"].get("passed_count", 0),
        "not_passed_count": seg["summary"].get("not_passed_count", 0),
        "high_credits_count": seg["summary"].get("high_credits_pass_count", 0),
        "border_pass_count": seg["summary"].get("border_pass_count", 0),
        "eligible_count": seg["summary"].get("eligible_count", 0),
    }


def update_student_attendance_summary(
    student_id: str,
    working_days: int,
    absent_days: int,
    current_semester: Optional[int] = None,
    is_lateral_entry: Optional[int] = None,
    actor_id: str = "faculty",
    db_path: Optional[str] = None,
) -> None:
    """Update student summary attendance fields and optional semester/lateral profile flags."""
    from auth import log_audit_action

    db = _get_conn(db_path)
    try:
        fields = ["working_days = ?", "absent_days = ?"]
        params: List[Any] = [int(working_days), int(absent_days)]
        if current_semester is not None:
            fields.append("current_semester = ?")
            params.append(int(current_semester))
        if is_lateral_entry is not None:
            fields.append("is_lateral_entry = ?")
            params.append(int(is_lateral_entry))
        params.append(student_id)
        db.execute(f"UPDATE students SET {', '.join(fields)} WHERE id = ?", params)
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()
    log_audit_action("staff", actor_id, "FACULTY_STUDENT_UPDATE", f"Updated attendance summary ({working_days} working, {absent_days} absent)", target_id=student_id)


# ============================================================
# COMPREHENSIVE ANALYTICS & AUTOMATED SEGREGATION ENGINE
# ============================================================

def _sgpa_tier_label(sgpa: float, sem_count: int) -> str:
    if sem_count <= 0:
        return "No Result"
    if sgpa >= 9.0:
        return "Distinction (9.0 - 10.0)"
    if sgpa >= 7.75:
        return "First Class Exemplary (7.75 - 8.99)"
    if sgpa >= 6.75:
        return "First Class (6.75 - 7.74)"
    if sgpa >= 5.0:
        return "Second Class (5.0 - 6.74)"
    return "At-Risk (< 5.0)"


def _marks_tier_label(marks_pct: float, total_subjects: int) -> str:
    if total_subjects <= 0:
        return "No Marks"
    if marks_pct >= 75.0:
        return "Distinction (>= 75%)"
    if marks_pct >= 60.0:
        return "First Class (60% - 74.9%)"
    if marks_pct >= 50.0:
        return "Second Class (50% - 59.9%)"
    if marks_pct >= 40.0:
        return "Pass / Borderline (40% - 49.9%)"
    return "Needs Attention (< 40%)"


def _evaluate_next_sem_eligibility(
    credits_earned: float,
    credits_registered: float,
    active_backlogs: int,
    sem_count: int,
    rules: Dict[str, Any],
    current_semester: int = 1,
    year_back_blockers: int = 0,
) -> Tuple[bool, str, str]:
    """Return (is_eligible_bool, status_label, human_reason)."""
    max_bk = int(rules.get("max_backlogs_next_sem", 4))
    min_cr = float(rules.get("min_credits_next_sem", 16.0))
    mode = str(rules.get("enforce_credit_pct_mode", "threshold")).lower()
    min_pct = float(rules.get("min_credit_pct_next_sem", 50.0))
    check_yb = str(rules.get("enforce_year_back_rule", "1")).strip() in ("1", "true", "True", "yes")

    if sem_count <= 0 and credits_registered <= 0:
        return False, "Awaiting VTU Result", "No VTU semester results recorded yet"

    reasons: List[str] = []
    if active_backlogs > max_bk:
        reasons.append(f"{active_backlogs} active backlogs (> max {max_bk})")

    if check_yb and year_back_blockers > 0 and current_semester >= 4:
        reasons.append(f"{year_back_blockers} earlier-year active backlog(s) under VTU year-back rule")

    if mode == "percentage":
        earned_pct = (credits_earned / credits_registered * 100.0) if credits_registered > 0 else 0.0
        if earned_pct < min_pct:
            reasons.append(f"{earned_pct:.1f}% credits earned (< {min_pct:.0f}% required)")
    else:
        if credits_earned < min_cr:
            reasons.append(f"{credits_earned:g} credits earned (< {min_cr:g} min required)")

    if reasons:
        return False, "Not Eligible", "; ".join(reasons)

    return True, "Eligible", f"{credits_earned:g} credits earned, {active_backlogs} active backlog(s)"


def _compute_placement_readiness(
    cgpa: float,
    active_backlogs: int,
    attendance_pct: float,
    has_att_data: bool,
    low_att_thresh: float,
    skills: List[str],
    verified_certs: int,
) -> Tuple[float, str, str]:
    """Score placement readiness 0-100 and produce personalized skill-gap guidance."""
    acad_pts = min(40.0, (cgpa / 10.0) * 40.0) if cgpa > 0 else 0.0
    bk_pts = 25.0 if active_backlogs == 0 else max(0.0, 15.0 - active_backlogs * 5.0)
    att_pts = 15.0 if (not has_att_data or attendance_pct >= low_att_thresh) else round((attendance_pct / max(1.0, low_att_thresh)) * 15.0, 1)
    skill_pts = min(12.0, len(skills) * 3.0)
    cert_pts = min(8.0, verified_certs * 4.0)
    score = round(min(100.0, acad_pts + bk_pts + att_pts + skill_pts + cert_pts), 1)

    gaps: List[str] = []
    if active_backlogs > 0:
        gaps.append(f"Clear {active_backlogs} active backlog(s)")
    if cgpa > 0 and cgpa < 6.75:
        gaps.append("Target CGPA >= 6.75 for Tier-1 drives")
    if has_att_data and attendance_pct < low_att_thresh:
        gaps.append(f"Raise attendance to >= {low_att_thresh:g}%")
    if len(skills) < 3:
        gaps.append("Add >= 3 verified technical skills in Skills & Portfolio")
    if verified_certs == 0:
        gaps.append("Upload & verify 1+ certification/internship proof")

    if active_backlogs > 0 or score < 55.0:
        tier = "Needs Action"
    elif score >= 80.0:
        tier = "Drive Ready (80+)"
    else:
        tier = "Developing (55-79)"

    advice = "; ".join(gaps) if gaps else "Strong profile: ready for campus placement drives"
    return score, tier, advice


def get_comprehensive_student_analytics(
    class_ids: Optional[List[str]] = None,
    semester_filter: Optional[int] = None,
    rules: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build one unified per-student DataFrame covering:
    SGPA, CGPA, Marks Obtained/Max/%, Credits Registered/Earned/Deficit,
    Active Backlogs, Attendance %, Next-Semester Eligibility, and Placement Status.
    """
    eff_rules = dict(get_academic_rules(db_path))
    if rules:
        eff_rules.update(rules)

    low_att_thresh = float(eff_rules.get("low_attendance_threshold", 75.0))
    cond_att_thresh = float(eff_rules.get("condonable_min_attendance_pct", 65.0))
    exam_att_thresh = float(eff_rules.get("exam_min_attendance_pct", low_att_thresh))
    exam_cie_pct = float(eff_rules.get("exam_min_cie_pct", 40.0))
    min_cie_marks = round(exam_cie_pct * 0.5, 2)

    db = _get_conn(db_path)
    try:
        stu_sql = """
            SELECT
                st.id AS student_id,
                st.usn,
                st.name AS student_name,
                st.class_id,
                COALESCE(c.class_name, 'Unassigned') AS class_name,
                c.section AS class_section,
                st.admission_year,
                st.current_semester,
                st.is_lateral_entry,
                COALESCE(st.working_days, 0) AS working_days,
                COALESCE(st.absent_days, 0) AS absent_days,
                st.email,
                st.phone
            FROM students st
            LEFT JOIN classes c ON st.class_id = c.id
        """
        params: List[Any] = []
        if class_ids is not None:
            if not class_ids:
                return pd.DataFrame()
            placeholders = ",".join("?" for _ in class_ids)
            stu_sql += f" WHERE st.class_id IN ({placeholders})"
            params.extend(class_ids)
        stu_sql += " ORDER BY st.usn ASC"
        students = db.execute(stu_sql, params).fetchall()
        if not students:
            return pd.DataFrame()

        sem_rows = db.execute(
            "SELECT id, student_id, semester_number, sgpa, credits_total, verification_status FROM semesters ORDER BY semester_number ASC"
        ).fetchall()
        sub_rows = db.execute(
            """
            SELECT sem.student_id, sem.semester_number, s.subject_code, s.subject_name,
                   s.internal_marks, s.external_marks, s.total_marks, s.credits, s.result
            FROM subjects s
            JOIN semesters sem ON s.semester_id = sem.id
            """
        ).fetchall()
        bk_rows = db.execute(
            "SELECT student_id, subject_code, semester_number, status FROM backlogs"
        ).fetchall()
        att_rows = db.execute(
            """
            SELECT student_id,
                   COUNT(*) AS total_hours,
                   SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_hours
            FROM attendance
            GROUP BY student_id
            """
        ).fetchall()
        pl_rows = db.execute(
            """
            SELECT student_id, company_name, job_role, package_lpa, offer_status,
                   COALESCE(verification_status, 'Verified') AS verification_status,
                   drive_date, location, offer_letter_path
            FROM placements
            ORDER BY COALESCE(package_lpa, 0) DESC, created_at DESC
            """
        ).fetchall()
        csr_rows = db.execute(
            "SELECT student_id, semester_number, subject_code, ia1, ia2, internal_performance FROM current_semester_records"
        ).fetchall()
        skill_rows = db.execute(
            "SELECT student_id, skill_name FROM student_skills"
        ).fetchall()
        cert_rows = db.execute(
            "SELECT student_id, verification_status FROM certificates"
        ).fetchall()
    finally:
        db.close()

    sem_by_stu: Dict[str, List[Dict[str, Any]]] = {}
    for r in sem_rows:
        sem_by_stu.setdefault(r["student_id"], []).append(dict(r))

    sub_by_stu: Dict[str, List[Dict[str, Any]]] = {}
    for r in sub_rows:
        sub_by_stu.setdefault(r["student_id"], []).append(dict(r))

    bk_active_by_stu: Dict[str, int] = {}
    bk_any_by_stu: Dict[str, int] = {}
    bk_sem_list_by_stu: Dict[str, List[int]] = {}
    for r in bk_rows:
        sid = r["student_id"]
        bk_any_by_stu[sid] = bk_any_by_stu.get(sid, 0) + 1
        if str(r["status"] or "Active").strip().lower() == "active":
            bk_active_by_stu[sid] = bk_active_by_stu.get(sid, 0) + 1
            if r["semester_number"] is not None:
                bk_sem_list_by_stu.setdefault(sid, []).append(int(r["semester_number"]))

    att_by_stu = {r["student_id"]: dict(r) for r in att_rows}
    pl_by_stu: Dict[str, List[Dict[str, Any]]] = {}
    for r in pl_rows:
        pl_by_stu.setdefault(r["student_id"], []).append(dict(r))

    csr_by_stu: Dict[str, List[Dict[str, Any]]] = {}
    for r in csr_rows:
        csr_by_stu.setdefault(r["student_id"], []).append(dict(r))

    skills_by_stu: Dict[str, List[str]] = {}
    for r in skill_rows:
        if r["skill_name"]:
            skills_by_stu.setdefault(r["student_id"], []).append(str(r["skill_name"]).strip())

    verified_certs_by_stu: Dict[str, int] = {}
    for r in cert_rows:
        if str(r["verification_status"] or "").strip().lower() == "verified":
            verified_certs_by_stu[r["student_id"]] = verified_certs_by_stu.get(r["student_id"], 0) + 1

    records: List[Dict[str, Any]] = []
    for st_row in students:
        sid = st_row["student_id"]
        all_sems = sem_by_stu.get(sid, [])
        all_subs = sub_by_stu.get(sid, [])

        scoped_sems = [s for s in all_sems if semester_filter is None or int(s["semester_number"]) == int(semester_filter)]
        scoped_subs = [s for s in all_subs if semester_filter is None or int(s["semester_number"]) == int(semester_filter)]

        cgpa = round(sum(float(s["sgpa"] or 0.0) for s in all_sems) / len(all_sems), 2) if all_sems else 0.0
        if scoped_sems:
            latest_sgpa = round(float(scoped_sems[-1]["sgpa"] or 0.0), 2)
            latest_sem_num = int(scoped_sems[-1]["semester_number"])
        else:
            latest_sgpa = 0.0
            latest_sem_num = 0

        if scoped_subs:
            total_subjects = len(scoped_subs)
            total_marks_obtained = round(sum(float(s["total_marks"] or 0.0) for s in scoped_subs), 2)
            total_max_marks = float(total_subjects * 100)
            marks_pct = round((total_marks_obtained / total_max_marks) * 100.0, 2) if total_max_marks > 0 else 0.0
            credits_reg = round(sum(float(s["credits"] or 0.0) for s in scoped_subs), 2)
            credits_earned = round(
                sum(float(s["credits"] or 0.0) for s in scoped_subs if is_pass_result(s["result"], s["total_marks"])),
                2,
            )
            passed_subs = sum(1 for s in scoped_subs if is_pass_result(s["result"], s["total_marks"]))
            failed_subs = total_subjects - passed_subs
        else:
            total_subjects = 0
            total_marks_obtained = 0.0
            total_max_marks = 0.0
            marks_pct = 0.0
            credits_reg = round(sum(float(s["credits_total"] or 0.0) for s in scoped_sems), 2)
            credits_earned = credits_reg if latest_sgpa >= 4.0 and scoped_sems else 0.0
            passed_subs = 0
            failed_subs = 0

        # Active backlogs: use backlogs table when populated, else fallback to failed subjects count
        if sid in bk_any_by_stu:
            active_backlogs = bk_active_by_stu.get(sid, 0)
        else:
            active_backlogs = sum(1 for s in all_subs if not is_pass_result(s["result"], s["total_marks"]))

        # Cumulative credits across all uploaded semesters for promotion eligibility check
        if all_subs:
            cum_credits_reg = round(sum(float(s["credits"] or 0.0) for s in all_subs), 2)
            cum_credits_earned = round(
                sum(float(s["credits"] or 0.0) for s in all_subs if is_pass_result(s["result"], s["total_marks"])),
                2,
            )
        else:
            cum_credits_reg = round(sum(float(s["credits_total"] or 0.0) for s in all_sems), 2)
            cum_credits_earned = cum_credits_reg

        att_info = att_by_stu.get(sid)
        if att_info and int(att_info["total_hours"]) > 0:
            total_hours = int(att_info["total_hours"])
            present_hours = int(att_info["present_hours"] or 0)
            attendance_pct = round((present_hours / total_hours) * 100.0, 2)
            has_att_data = True
        elif int(st_row["working_days"] or 0) > 0:
            w_days = int(st_row["working_days"])
            a_days = min(int(st_row["absent_days"] or 0), w_days)
            total_hours = w_days
            present_hours = w_days - a_days
            attendance_pct = round((present_hours / w_days) * 100.0, 2)
            has_att_data = True
        else:
            total_hours = 0
            present_hours = 0
            attendance_pct = 0.0
            has_att_data = False

        is_low_att = bool(has_att_data and attendance_pct < low_att_thresh)
        att_status = (
            "No Attendance Data"
            if not has_att_data
            else (f"Low Attendance (< {low_att_thresh:g}%)" if is_low_att else f"Satisfactory (>= {low_att_thresh:g}%)")
        )
        classes_needed = calculate_classes_needed(present_hours, total_hours, exam_att_thresh) if has_att_data else 0
        if not has_att_data:
            cond_band = "No Attendance Data"
        elif attendance_pct >= exam_att_thresh:
            cond_band = f"Eligible (>= {exam_att_thresh:g}%)"
        elif attendance_pct >= cond_att_thresh:
            cond_band = f"Condonable ({cond_att_thresh:g}-{exam_att_thresh:g}%)"
        else:
            cond_band = f"Shortage (< {cond_att_thresh:g}%)"

        # Current semester CIE shortage check
        cur_sem_num = int(st_row["current_semester"] or 1)
        cur_csr = [x for x in csr_by_stu.get(sid, []) if int(x["semester_number"] or 0) == cur_sem_num]
        cie_shortages = [
            x["subject_code"]
            for x in cur_csr
            if float(x["internal_performance"] or ((float(x["ia1"] or 0.0) + float(x["ia2"] or 0.0)) / 2.0)) < min_cie_marks
        ]

        if not has_att_data and not cur_csr:
            exam_eligible = True
            exam_status = "Pending Data"
            exam_reason = "Attendance & IA records not uploaded yet"
        elif has_att_data and attendance_pct < cond_att_thresh:
            exam_eligible = False
            exam_status = "Detained (Shortage)"
            exam_reason = f"Attendance {attendance_pct:.1f}% < condonable {cond_att_thresh:g}%"
        elif len(cie_shortages) > 0:
            exam_eligible = False
            exam_status = "CIE Shortage"
            exam_reason = f"Low CIE (< {min_cie_marks:g}/50) in {', '.join(cie_shortages[:3])}"
        elif has_att_data and attendance_pct < exam_att_thresh:
            exam_eligible = True
            exam_status = "Condonable Attendance"
            exam_reason = f"Needs {classes_needed} class(es) to reach {exam_att_thresh:g}% or medical condonation"
        else:
            exam_eligible = True
            exam_status = "Hall Ticket Eligible"
            exam_reason = f"Attendance {attendance_pct:.1f}% & CIE compliant"

        # Evaluate next-semester eligibility against cumulative earned credits + optional VTU year-back rule
        elig_credits = credits_earned if semester_filter is not None else cum_credits_earned
        elig_reg = credits_reg if semester_filter is not None else cum_credits_reg
        elig_sem_cnt = len(scoped_sems) if semester_filter is not None else len(all_sems)
        yb_cutoff = cur_sem_num - 3 if cur_sem_num >= 4 else 0
        year_back_blockers = sum(1 for bs in bk_sem_list_by_stu.get(sid, []) if bs <= yb_cutoff) if yb_cutoff > 0 else 0
        next_eligible, elig_label, elig_reason = _evaluate_next_sem_eligibility(
            elig_credits, elig_reg, active_backlogs, elig_sem_cnt, eff_rules,
            current_semester=cur_sem_num, year_back_blockers=year_back_blockers,
        )

        # Placements aggregation (only Verified offers count toward official Placed totals)
        pls = pl_by_stu.get(sid, [])
        placed_offers = [
            p for p in pls
            if p["offer_status"] in PLACED_STATUSES
            and str(p.get("verification_status") or "Verified").strip().lower() == "verified"
        ]
        pending_offers = [
            p for p in pls
            if str(p.get("verification_status") or "Verified").strip().lower() == "pending verification"
        ]
        is_placed = len(placed_offers) > 0
        if is_placed:
            pl_status = "Placed"
            companies = ", ".join(dict.fromkeys(p["company_name"] for p in placed_offers if p["company_name"]))
            highest_pkg = max(float(p["package_lpa"] or 0.0) for p in placed_offers)
            latest_role = placed_offers[0].get("job_role") or ""
            latest_offer_path = placed_offers[0].get("offer_letter_path") or ""
        else:
            pl_status = "Not Placed"
            companies = "-"
            highest_pkg = 0.0
            latest_role = pls[0].get("offer_status", "") if pls else ""
            latest_offer_path = ""

        stu_skills = skills_by_stu.get(sid, [])
        verif_certs = verified_certs_by_stu.get(sid, 0)
        readiness_score, readiness_tier, gap_advice = _compute_placement_readiness(
            cgpa, active_backlogs, attendance_pct, has_att_data, low_att_thresh, stu_skills, verif_certs
        )

        salary_tier = (
            "Super Dream" if highest_pkg >= 10.0
            else "Dream" if highest_pkg >= 6.0
            else "Standard" if highest_pkg > 0.0
            else "Not Placed"
        )

        # Passed vs Not Passed & Borderline Pass Analysis
        has_results = bool(scoped_subs or scoped_sems)
        is_passed = bool(has_results and active_backlogs == 0 and failed_subs == 0)

        border_subs_details = []
        if is_passed and scoped_subs:
            for s in scoped_subs:
                ext = float(s.get("external_marks") or 0.0)
                tot = float(s.get("total_marks") or 0.0)
                code = str(s.get("subject_code") or "").strip()
                if (18.0 <= ext <= 22.0) or (40.0 <= tot <= 45.0):
                    border_subs_details.append(f"{code} (Tot {tot:.0f}, Ext {ext:.0f})")

        is_border_sgpa = bool(is_passed and (5.0 <= latest_sgpa < 5.75))
        is_border_marks = bool(is_passed and (40.0 <= marks_pct < 50.0))
        is_border_pass = bool(is_passed and (len(border_subs_details) > 0 or is_border_sgpa or is_border_marks))

        is_high_credits_pass = bool(
            is_passed
            and not is_border_pass
            and (credits_earned >= 20.0 or (credits_reg > 0 and credits_earned >= 0.9 * credits_reg))
            and (latest_sgpa >= 7.5 or cgpa >= 7.5)
        )

        if not has_results:
            pass_classification = "Result Pending"
            pass_status_label = "Pending Result"
        elif not is_passed:
            pass_classification = f"Not Passed ({active_backlogs} Backlog{'s' if active_backlogs > 1 else ''})"
            pass_status_label = "Not Passed"
        elif is_border_pass:
            pass_classification = "Just Border Pass (Risk Margin <= 5m / < 5.75 SGPA)"
            pass_status_label = "Border Pass"
        elif is_high_credits_pass:
            pass_classification = "Passed on High Credits (Distinction)"
            pass_status_label = "High Credits Pass"
        else:
            pass_classification = "Clear Pass (First Class)"
            pass_status_label = "Passed"

        records.append(
            {
                "student_id": sid,
                "usn": st_row["usn"],
                "student_name": st_row["student_name"],
                "class_id": st_row["class_id"],
                "class_name": st_row["class_name"],
                "class_section": st_row["class_section"] or "",
                "admission_year": int(st_row["admission_year"] or 2024),
                "current_semester": cur_sem_num,
                "is_lateral_entry": int(st_row["is_lateral_entry"] or 0),
                "working_days": int(st_row["working_days"] or 90),
                "absent_days": int(st_row["absent_days"] or 0),
                "is_passed": is_passed,
                "is_border_pass": is_border_pass,
                "is_high_credits_pass": is_high_credits_pass,
                "pass_classification": pass_classification,
                "pass_status_label": pass_status_label,
                "borderline_subjects_count": len(border_subs_details),
                "borderline_subjects_list": ", ".join(border_subs_details),
                "email": st_row["email"] or "",
                "phone": st_row["phone"] or "",
                "semesters_uploaded": len(all_sems),
                "latest_sem_num": latest_sem_num,
                "latest_sgpa": latest_sgpa,
                "cgpa": cgpa,
                "sgpa_tier": _sgpa_tier_label(latest_sgpa, len(scoped_sems)),
                "total_subjects": total_subjects,
                "passed_subjects": passed_subs,
                "failed_subjects": failed_subs,
                "total_marks_obtained": total_marks_obtained,
                "total_max_marks": total_max_marks,
                "marks_pct": marks_pct,
                "marks_tier": _marks_tier_label(marks_pct, total_subjects),
                "credits_registered": credits_reg if semester_filter is not None else cum_credits_reg,
                "credits_earned": credits_earned if semester_filter is not None else cum_credits_earned,
                "credit_deficit": round(
                    max(0.0, (credits_reg - credits_earned) if semester_filter is not None else (cum_credits_reg - cum_credits_earned)),
                    2,
                ),
                "active_backlogs": active_backlogs,
                "total_hours": total_hours,
                "present_hours": present_hours,
                "attendance_pct": attendance_pct,
                "has_attendance_data": has_att_data,
                "is_low_attendance": is_low_att,
                "attendance_status": att_status,
                "classes_to_recover_75": classes_needed,
                "att_condonation_band": cond_band,
                "cie_shortage_count": len(cie_shortages),
                "exam_eligible": exam_eligible,
                "exam_status": exam_status,
                "exam_reason": exam_reason,
                "next_sem_eligible": next_eligible,
                "eligibility_status": elig_label,
                "eligibility_reason": elig_reason,
                "is_placed": is_placed,
                "placement_status": pl_status,
                "offers_count": len(placed_offers),
                "pending_offers_count": len(pending_offers),
                "placement_companies": companies,
                "highest_package_lpa": round(highest_pkg, 2),
                "salary_tier": salary_tier,
                "placement_detail": latest_role,
                "offer_letter_path": latest_offer_path,
                "skills_list": ", ".join(stu_skills),
                "skills_count": len(stu_skills),
                "verified_certs_count": verif_certs,
                "placement_readiness_score": readiness_score,
                "placement_readiness_tier": readiness_tier,
                "placement_gap_advice": gap_advice,
            }
        )

    return pd.DataFrame(records)


def segregate_students(
    df: pd.DataFrame,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce all automatic faculty segregation views and summary KPIs from student analytics DF."""
    if df is None or df.empty:
        empty = pd.DataFrame()
        return {
            "all_df": empty,
            "passed_df": empty,
            "not_passed_df": empty,
            "pass_high_credits_df": empty,
            "border_pass_df": empty,
            "sgpa_df": empty,
            "marks_df": empty,
            "credits_df": empty,
            "eligible_df": empty,
            "not_eligible_df": empty,
            "eligible_credits_df": empty,
            "not_eligible_credits_df": empty,
            "low_attendance_df": empty,
            "placed_df": empty,
            "not_placed_df": empty,
            "summary": {
                "total_students": 0,
                "with_vtu_results": 0,
                "passed_count": 0,
                "not_passed_count": 0,
                "high_credits_pass_count": 0,
                "border_pass_count": 0,
                "eligible_count": 0,
                "not_eligible_count": 0,
                "low_attendance_count": 0,
                "placed_count": 0,
                "not_placed_count": 0,
                "avg_sgpa": 0.0,
                "avg_marks_pct": 0.0,
                "avg_credits_earned": 0.0,
                "pass_percentage": 0.0,
            },
        }

    sgpa_df = df.sort_values(by=["latest_sgpa", "cgpa", "usn"], ascending=[False, False, True]).reset_index(drop=True)
    marks_df = df.sort_values(by=["marks_pct", "total_marks_obtained", "usn"], ascending=[False, False, True]).reset_index(drop=True)
    credits_df = df.sort_values(by=["credits_earned", "active_backlogs", "usn"], ascending=[False, True, True]).reset_index(drop=True)
    eligible_df = df[df["next_sem_eligible"] == True].sort_values(by=["credits_earned", "latest_sgpa"], ascending=[False, False]).reset_index(drop=True)
    not_eligible_df = df[df["next_sem_eligible"] == False].sort_values(by=["active_backlogs", "credits_earned"], ascending=[False, True]).reset_index(drop=True)
    low_att_df = df[df["is_low_attendance"] == True].sort_values(by=["attendance_pct", "usn"], ascending=[True, True]).reset_index(drop=True)
    placed_df = df[df["is_placed"] == True].sort_values(by=["highest_package_lpa", "latest_sgpa"], ascending=[False, False]).reset_index(drop=True)
    not_placed_df = df[df["is_placed"] == False].sort_values(by=["cgpa", "active_backlogs", "usn"], ascending=[False, True, True]).reset_index(drop=True)

    passed_df = df[df["is_passed"] == True].sort_values(by=["latest_sgpa", "credits_earned", "usn"], ascending=[False, False, True]).reset_index(drop=True)
    not_passed_df = df[df["is_passed"] == False].sort_values(by=["active_backlogs", "credits_earned", "usn"], ascending=[False, True, True]).reset_index(drop=True)
    pass_high_credits_df = df[df["is_high_credits_pass"] == True].sort_values(by=["latest_sgpa", "credits_earned", "usn"], ascending=[False, False, True]).reset_index(drop=True)
    border_pass_df = df[df["is_border_pass"] == True].sort_values(by=["latest_sgpa", "total_marks_obtained", "usn"], ascending=[True, True, True]).reset_index(drop=True)

    with_res = df[df["semesters_uploaded"] > 0]
    summary = {
        "total_students": int(len(df)),
        "with_vtu_results": int(len(with_res)),
        "passed_count": int(len(passed_df)),
        "not_passed_count": int(len(not_passed_df)),
        "high_credits_pass_count": int(len(pass_high_credits_df)),
        "border_pass_count": int(len(border_pass_df)),
        "eligible_count": int(len(eligible_df)),
        "not_eligible_count": int(len(not_eligible_df)),
        "low_attendance_count": int(len(low_att_df)),
        "placed_count": int(len(placed_df)),
        "not_placed_count": int(len(not_placed_df)),
        "avg_sgpa": round(float(with_res["latest_sgpa"].mean()), 2) if not with_res.empty else 0.0,
        "avg_marks_pct": round(float(with_res["marks_pct"].mean()), 2) if not with_res.empty else 0.0,
        "avg_credits_earned": round(float(with_res["credits_earned"].mean()), 1) if not with_res.empty else 0.0,
        "pass_percentage": round(len(passed_df) / len(with_res) * 100.0, 1) if not with_res.empty else 0.0,
    }

    return {
        "all_df": df,
        "passed_df": passed_df,
        "not_passed_df": not_passed_df,
        "pass_high_credits_df": pass_high_credits_df,
        "border_pass_df": border_pass_df,
        "sgpa_df": sgpa_df,
        "marks_df": marks_df,
        "credits_df": credits_df,
        "eligible_df": eligible_df,
        "not_eligible_df": not_eligible_df,
        "eligible_credits_df": eligible_df,
        "not_eligible_credits_df": not_eligible_df,
        "low_attendance_df": low_att_df,
        "placed_df": placed_df,
        "not_placed_df": not_placed_df,
        "summary": summary,
    }


# ============================================================
# PLACEMENT MANAGEMENT & AUTOMATED IMPORT
# ============================================================

def upsert_placement_record(
    student_id: str,
    company_name: str,
    job_role: str = "",
    package_lpa: float = 0.0,
    offer_status: str = "Placed",
    drive_date: str = "",
    location: str = "",
    offer_letter_path: str = "",
    remarks: str = "",
    entered_by: str = "placement",
    entered_role: str = "placement",
    placement_id: Optional[str] = None,
    notify_student: bool = True,
    verification_status: str = "Verified",
    source: str = "faculty",
    db_path: Optional[str] = None,
) -> str:
    """Create or update an individual student placement record and notify student/placement cell."""
    from auth import log_audit_action, send_notification

    comp = (company_name or "").strip()
    if not comp and offer_status in ("Not Placed", "Opted Out", "Higher Studies"):
        comp = offer_status
    if not comp:
        raise ValueError("Company name is required for placement records.")

    now = datetime.now().isoformat()
    rec_id = placement_id or uuid.uuid4().hex
    pkg = round(float(package_lpa or 0.0), 2)
    db = _get_conn(db_path)
    stu_usn = student_id
    stu_name = ""
    try:
        stu = db.execute("SELECT usn, name FROM students WHERE id = ?", (student_id,)).fetchone()
        if stu:
            stu_usn = stu["usn"]
            stu_name = stu["name"]

        existing = db.execute("SELECT id, offer_letter_path FROM placements WHERE id = ?", (rec_id,)).fetchone()
        eff_letter = offer_letter_path or (existing["offer_letter_path"] if existing else "") or ""

        if existing:
            db.execute(
                """
                UPDATE placements
                SET student_id = ?, company_name = ?, job_role = ?, package_lpa = ?,
                    offer_status = ?, drive_date = ?, location = ?, offer_letter_path = ?,
                    remarks = ?, entered_by = ?, entered_role = ?,
                    verification_status = ?, source = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    student_id,
                    comp,
                    (job_role or "").strip(),
                    pkg,
                    offer_status,
                    (drive_date or "").strip(),
                    (location or "").strip(),
                    eff_letter,
                    (remarks or "").strip(),
                    entered_by,
                    entered_role,
                    verification_status,
                    source,
                    now,
                    rec_id,
                ),
            )
        else:
            db.execute(
                """
                INSERT INTO placements (
                    id, student_id, company_name, job_role, package_lpa, offer_status,
                    drive_date, location, offer_letter_path, remarks, entered_by, entered_role,
                    verification_status, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id,
                    student_id,
                    comp,
                    (job_role or "").strip(),
                    pkg,
                    offer_status,
                    (drive_date or "").strip(),
                    (location or "").strip(),
                    eff_letter,
                    (remarks or "").strip(),
                    entered_by,
                    entered_role,
                    verification_status,
                    source,
                    now,
                    now,
                ),
            )
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()
    log_audit_action(
        entered_role or "staff",
        entered_by,
        "PLACEMENT_UPSERT",
        f"[{verification_status}] {offer_status} at {comp} ({pkg:.2f} LPA) for {stu_usn}",
        target_id=student_id,
    )
    if notify_student and source != "student":
        pkg_msg = f" with {pkg:.2f} LPA package" if pkg > 0 else ""
        send_notification(
            "student",
            student_id,
            f"Placement Update: {comp} ({offer_status})",
            f"Placement record updated: {comp} | Role: {job_role or 'N/A'} | Status: {offer_status}{pkg_msg}.",
            category="Placement",
        )
    elif source == "student":
        send_notification(
            "staff",
            "all",
            f"Off-Campus Placement Offer Submitted: {stu_usn}",
            f"{stu_name} ({stu_usn}) submitted an offer from {comp} ({pkg:.2f} LPA) for Placement Officer verification.",
            category="Placement Approval",
        )
    return rec_id


def delete_placement_record(
    placement_id: str,
    actor_id: str = "placement",
    db_path: Optional[str] = None,
) -> bool:
    """Remove a placement record by ID."""
    from auth import log_audit_action

    db = _get_conn(db_path)
    try:
        cur = db.execute("DELETE FROM placements WHERE id = ?", (placement_id,))
        db.commit()
        deleted = cur.rowcount > 0
    finally:
        db.close()
    if deleted:
        clear_streamlit_caches()
        log_audit_action("staff", actor_id, "PLACEMENT_DELETE", f"Deleted placement record {placement_id}")
    return deleted


def get_student_placements(student_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all placement records for one student (for Student Portal & individual lookup)."""
    db = _get_conn(db_path)
    try:
        rows = db.execute(
            """
            SELECT * FROM placements
            WHERE student_id = ?
            ORDER BY COALESCE(package_lpa, 0) DESC, updated_at DESC, created_at DESC
            """,
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


def get_all_placements(
    class_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch all placement records joined with student and class metadata."""
    db = _get_conn(db_path)
    try:
        sql = """
            SELECT
                p.id,
                p.student_id,
                st.usn,
                st.name AS student_name,
                st.class_id,
                COALESCE(c.class_name, 'Unassigned') AS class_name,
                c.section AS class_section,
                p.company_name,
                p.job_role,
                p.package_lpa,
                p.offer_status,
                p.drive_date,
                p.location,
                p.offer_letter_path,
                p.remarks,
                p.entered_by,
                p.entered_role,
                p.updated_at
            FROM placements p
            JOIN students st ON p.student_id = st.id
            LEFT JOIN classes c ON st.class_id = c.id
        """
        params: List[Any] = []
        if class_ids is not None:
            if not class_ids:
                return pd.DataFrame()
            placeholders = ",".join("?" for _ in class_ids)
            sql += f" WHERE st.class_id IN ({placeholders})"
            params.extend(class_ids)
        sql += " ORDER BY COALESCE(p.package_lpa, 0) DESC, p.updated_at DESC"
        rows = db.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        db.close()


def _find_col(cols_map: Dict[str, str], candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in cols_map:
            return cols_map[c]
    for low, orig in cols_map.items():
        if any(c in low for c in candidates):
            return orig
    return None


def validate_placement_excel(df: pd.DataFrame, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate Excel/CSV dataframe for automated bulk placement import."""
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    usn_col = _find_col(cols_map, ("usn", "student usn", "university seat number"))
    comp_col = _find_col(cols_map, ("company", "company_name", "employer", "organization"))
    role_col = _find_col(cols_map, ("role", "job_role", "designation", "position"))
    pkg_col = _find_col(cols_map, ("package", "package_lpa", "ctc", "lpa", "salary"))
    status_col = _find_col(cols_map, ("status", "offer_status", "placement_status"))
    date_col = _find_col(cols_map, ("drive_date", "date", "offer_date"))
    loc_col = _find_col(cols_map, ("location", "city"))
    rem_col = _find_col(cols_map, ("remarks", "notes"))

    if not usn_col or not comp_col:
        return {
            "is_valid": False,
            "error": "Spreadsheet must contain at least 'USN' and 'Company' columns.",
            "valid_rows": [],
            "invalid_rows": [],
        }

    db = _get_conn(db_path)
    try:
        stu_map = {str(r["usn"]).strip().upper(): dict(r) for r in db.execute("SELECT id, usn, name FROM students").fetchall()}
    finally:
        db.close()

    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        usn = str(row.get(usn_col, "")).strip().upper()
        comp = str(row.get(comp_col, "")).strip()
        if not usn or usn == "NAN":
            continue
        stu = stu_map.get(usn)
        if not stu:
            invalid_rows.append({"row": idx + 2, "usn": usn, "errors": ["USN not found in students database"]})
            continue
        if not comp or comp.upper() == "NAN":
            invalid_rows.append({"row": idx + 2, "usn": usn, "errors": ["Missing company name"]})
            continue

        raw_pkg = row.get(pkg_col, 0.0) if pkg_col else 0.0
        try:
            pkg = float(str(raw_pkg).replace("LPA", "").replace("lpa", "").strip() or 0.0)
            if pd.isna(pkg):
                pkg = 0.0
        except Exception:
            pkg = 0.0

        status_val = str(row.get(status_col, "Placed")).strip() if status_col else "Placed"
        if not status_val or status_val.upper() == "NAN":
            status_val = "Placed"

        valid_rows.append(
            {
                "student_id": stu["id"],
                "usn": usn,
                "student_name": stu["name"],
                "company_name": comp,
                "job_role": str(row.get(role_col, "")).strip() if role_col and pd.notna(row.get(role_col)) else "",
                "package_lpa": round(pkg, 2),
                "offer_status": status_val,
                "drive_date": str(row.get(date_col, "")).strip() if date_col and pd.notna(row.get(date_col)) else "",
                "location": str(row.get(loc_col, "")).strip() if loc_col and pd.notna(row.get(loc_col)) else "",
                "remarks": str(row.get(rem_col, "")).strip() if rem_col and pd.notna(row.get(rem_col)) else "",
            }
        )

    return {
        "is_valid": len(invalid_rows) == 0 and len(valid_rows) > 0,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }


def execute_placement_bulk_import(
    valid_records: List[Dict[str, Any]],
    actor_id: str = "placement",
    actor_role: str = "placement",
    db_path: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """Insert bulk placement records with student notifications."""
    count = 0
    imported_usns: List[str] = []
    for rec in valid_records:
        upsert_placement_record(
            student_id=rec["student_id"],
            company_name=rec["company_name"],
            job_role=rec.get("job_role", ""),
            package_lpa=rec.get("package_lpa", 0.0),
            offer_status=rec.get("offer_status", "Placed"),
            drive_date=rec.get("drive_date", ""),
            location=rec.get("location", ""),
            remarks=rec.get("remarks", ""),
            entered_by=actor_id,
            entered_role=actor_role,
            notify_student=True,
            db_path=db_path,
        )
        count += 1
        imported_usns.append(rec["usn"])
    return count, imported_usns


# ============================================================
# AUTOMATED BULK DATA ENTRY (MARKS, ATTENDANCE, MASTER SUBJECTS, BATCH PDF)
# ============================================================

def seed_vtu_master_subjects(
    class_id: str,
    semester_number: int,
    db_path: Optional[str] = None,
) -> int:
    """Auto-populate VTU master_subjects for a class/semester using canonical dictionary."""
    from ocr_engine import VTU_KNOWN_SUBJECTS

    sem_digit = str(int(semester_number))
    matches: List[Tuple[str, str, float]] = []
    for code, (name, credits) in VTU_KNOWN_SUBJECTS.items():
        # Typical VTU code patterns: e.g. BCS401, BDS501, 21CS51, 21CSL55, 22MATS31
        digits_in_code = "".join(ch for ch in code if ch.isdigit())
        if len(digits_in_code) >= 3 and sem_digit in digits_in_code[-3:-1]:
            matches.append((code, name, float(credits)))
        elif code.startswith(("BCS", "BDS", "BEX", "BEC", "BMAT", "BPOP", "BCHE", "BPHY")) and f"{sem_digit}0" in code:
            matches.append((code, name, float(credits)))

    if not matches:
        return 0

    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    inserted = 0
    try:
        existing = {
            str(r["subject_code"]).strip().upper()
            for r in db.execute(
                "SELECT subject_code FROM master_subjects WHERE class_id = ? AND semester_number = ?",
                (class_id, int(semester_number)),
            ).fetchall()
        }
        for code, name, credits in matches[:12]:
            if code.upper() in existing:
                continue
            db.execute(
                """
                INSERT INTO master_subjects (id, class_id, semester_number, subject_code, subject_name, credits, internal_max, external_max, total_max, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 50.0, 50.0, 100.0, ?)
                """,
                (uuid.uuid4().hex, class_id, int(semester_number), code, name, credits, now),
            )
            inserted += 1
        db.commit()
    finally:
        db.close()
    clear_streamlit_caches()
    return inserted


def validate_marks_excel(
    df: pd.DataFrame,
    mode: str = "final_vtu",
    default_semester: int = 1,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate Excel/CSV dataframe for automated bulk VTU final marks or current IA marks."""
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    usn_col = _find_col(cols_map, ("usn", "student usn"))
    sub_code_col = _find_col(cols_map, ("subject_code", "subject code", "code"))
    sub_name_col = _find_col(cols_map, ("subject_name", "subject name", "subject"))
    sem_col = _find_col(cols_map, ("semester", "sem", "semester_number"))
    cred_col = _find_col(cols_map, ("credits", "credit"))

    if mode == "final_vtu":
        int_col = _find_col(cols_map, ("internal", "internal_marks", "ia", "cie"))
        ext_col = _find_col(cols_map, ("external", "external_marks", "see"))
        tot_col = _find_col(cols_map, ("total", "total_marks"))
        res_col = _find_col(cols_map, ("result", "status"))
        sgpa_col = _find_col(cols_map, ("sgpa", "official_sgpa"))
        if not usn_col or not sub_code_col or (not int_col and not tot_col):
            return {
                "is_valid": False,
                "error": "Final VTU marks sheet needs 'USN', 'Subject Code', and 'Internal'+'External' (or 'Total') columns.",
                "valid_rows": [],
                "invalid_rows": [],
            }
    else:
        ia1_col = _find_col(cols_map, ("ia1", "ia 1", "test1"))
        ia2_col = _find_col(cols_map, ("ia2", "ia 2", "test2"))
        if not usn_col or not sub_code_col or (not ia1_col and not ia2_col):
            return {
                "is_valid": False,
                "error": "IA marks sheet needs 'USN', 'Subject Code', and 'IA1'/'IA2' columns.",
                "valid_rows": [],
                "invalid_rows": [],
            }

    db = _get_conn(db_path)
    try:
        stu_map = {str(r["usn"]).strip().upper(): dict(r) for r in db.execute("SELECT id, usn, name, current_semester FROM students").fetchall()}
    finally:
        db.close()

    from ocr_engine import get_canonical_subject_info, normalize_subject_code

    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        usn = str(row.get(usn_col, "")).strip().upper()
        if not usn or usn == "NAN":
            continue
        stu = stu_map.get(usn)
        if not stu:
            invalid_rows.append({"row": idx + 2, "usn": usn, "errors": ["USN not found in students table"]})
            continue
        code = str(row.get(sub_code_col, "")).strip().upper()
        if not code or code == "NAN":
            invalid_rows.append({"row": idx + 2, "usn": usn, "errors": ["Missing Subject Code"]})
            continue

        raw_name = str(row.get(sub_name_col, "")).strip() if sub_name_col and pd.notna(row.get(sub_name_col)) else ""
        can_code = normalize_subject_code(code)
        can_name, can_cred = get_canonical_subject_info(can_code, raw_name)
        sem_num = int(row.get(sem_col)) if sem_col and pd.notna(row.get(sem_col)) else int(stu.get("current_semester") or default_semester)
        credits = float(row.get(cred_col)) if cred_col and pd.notna(row.get(cred_col)) else float(can_cred)

        if mode == "final_vtu":
            internal = float(row.get(int_col) or 0.0) if int_col and pd.notna(row.get(int_col)) else 0.0
            external = float(row.get(ext_col) or 0.0) if ext_col and pd.notna(row.get(ext_col)) else 0.0
            total = float(row.get(tot_col)) if tot_col and pd.notna(row.get(tot_col)) else (internal + external)
            raw_res = str(row.get(res_col, "")).strip() if res_col and pd.notna(row.get(res_col)) else ""
            res_code = normalize_result_code(raw_res, total)
            sgpa_val = float(row.get(sgpa_col)) if sgpa_col and pd.notna(row.get(sgpa_col)) else None
            valid_rows.append(
                {
                    "student_id": stu["id"],
                    "usn": usn,
                    "student_name": stu["name"],
                    "semester_number": sem_num,
                    "subject_code": can_code,
                    "subject_name": raw_name or can_name,
                    "credits": credits,
                    "internal_marks": round(internal, 2),
                    "external_marks": round(external, 2),
                    "total_marks": round(total, 2),
                    "result": res_code,
                    "official_sgpa": sgpa_val,
                }
            )
        else:
            ia1 = float(row.get(ia1_col) or 0.0) if ia1_col and pd.notna(row.get(ia1_col)) else 0.0
            ia2 = float(row.get(ia2_col) or 0.0) if ia2_col and pd.notna(row.get(ia2_col)) else 0.0
            valid_rows.append(
                {
                    "student_id": stu["id"],
                    "usn": usn,
                    "student_name": stu["name"],
                    "semester_number": sem_num,
                    "subject_code": can_code,
                    "subject_name": raw_name or can_name,
                    "credits": credits,
                    "ia1": round(ia1, 2),
                    "ia2": round(ia2, 2),
                }
            )

    return {
        "is_valid": len(invalid_rows) == 0 and len(valid_rows) > 0,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }


def execute_marks_bulk_import(
    valid_records: List[Dict[str, Any]],
    mode: str = "final_vtu",
    actor_id: str = "faculty",
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """Commit validated bulk VTU or IA marks records into database."""
    if mode == "final_vtu":
        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for rec in valid_records:
            grouped.setdefault((rec["student_id"], int(rec["semester_number"])), []).append(rec)
        sem_saved = 0
        for (stu_id, sem_num), subs in grouped.items():
            sgpas = [r["official_sgpa"] for r in subs if r.get("official_sgpa")]
            off_sgpa = sgpas[0] if sgpas else None
            save_student_semester_result(
                student_id=stu_id,
                sem_num=sem_num,
                subjects=subs,
                official_sgpa=off_sgpa,
                verification_status="Verified",
                actor_type="staff",
                actor_id=actor_id,
                notify_faculty=False,
                notify_student=True,
                db_path=db_path,
            )
            sem_saved += 1
        return {"semesters_saved": sem_saved, "subjects_saved": len(valid_records)}

    for rec in valid_records:
        save_student_ia_subject_mark(
            student_id=rec["student_id"],
            semester_number=int(rec["semester_number"]),
            subject_code=rec["subject_code"],
            subject_name=rec["subject_name"],
            credits=rec["credits"],
            ia1=rec["ia1"],
            ia2=rec["ia2"],
            actor_id=actor_id,
            db_path=db_path,
        )
    return {"semesters_saved": 0, "subjects_saved": len(valid_records)}


def validate_attendance_excel(
    df: pd.DataFrame,
    default_subject: str = "General",
    default_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate Excel/CSV dataframe for automated hourly attendance logs or working/absent summary."""
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    usn_col = _find_col(cols_map, ("usn", "student usn"))
    status_col = _find_col(cols_map, ("status", "present/absent", "att_status"))
    sub_col = _find_col(cols_map, ("subject", "subject_name"))
    date_col = _find_col(cols_map, ("date", "att_date"))
    hour_col = _find_col(cols_map, ("hour", "hour_slot", "slot"))
    work_col = _find_col(cols_map, ("working_days", "total_classes", "total_hours"))
    abs_col = _find_col(cols_map, ("absent_days", "absent_hours", "absent"))

    if not usn_col or (not status_col and not work_col):
        return {
            "is_valid": False,
            "error": "Spreadsheet needs 'USN' and either 'Status' (Present/Absent) or 'Working Days' + 'Absent Days'.",
            "mode": "hourly",
            "valid_rows": [],
            "invalid_rows": [],
        }

    mode = "hourly" if status_col else "summary"
    db = _get_conn(db_path)
    try:
        stu_map = {str(r["usn"]).strip().upper(): dict(r) for r in db.execute("SELECT id, class_id, usn, name FROM students").fetchall()}
    finally:
        db.close()

    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    fallback_date = default_date or datetime.now().strftime("%Y-%m-%d")

    for idx, row in df.iterrows():
        usn = str(row.get(usn_col, "")).strip().upper()
        if not usn or usn == "NAN":
            continue
        stu = stu_map.get(usn)
        if not stu:
            invalid_rows.append({"row": idx + 2, "usn": usn, "errors": ["USN not found in students table"]})
            continue
        if mode == "hourly":
            raw_st = str(row.get(status_col, "")).strip().upper()
            norm_st = "Present" if raw_st in ("P", "PRESENT", "1", "YES") else "Absent"
            valid_rows.append(
                {
                    "student_id": stu["id"],
                    "class_id": stu.get("class_id"),
                    "usn": usn,
                    "subject_name": str(row.get(sub_col, default_subject)).strip() if sub_col and pd.notna(row.get(sub_col)) else default_subject,
                    "attendance_date": str(row.get(date_col, fallback_date)).strip()[:10] if date_col and pd.notna(row.get(date_col)) else fallback_date,
                    "hour_number": int(row.get(hour_col) or 1) if hour_col and pd.notna(row.get(hour_col)) else 1,
                    "status": norm_st,
                }
            )
        else:
            w_days = int(float(row.get(work_col) or 0))
            a_days = int(float(row.get(abs_col) or 0)) if abs_col and pd.notna(row.get(abs_col)) else 0
            valid_rows.append(
                {
                    "student_id": stu["id"],
                    "usn": usn,
                    "working_days": w_days,
                    "absent_days": min(a_days, w_days),
                }
            )

    return {
        "is_valid": len(invalid_rows) == 0 and len(valid_rows) > 0,
        "mode": mode,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }


def execute_attendance_bulk_import(
    valid_records: List[Dict[str, Any]],
    mode: str = "hourly",
    actor_id: str = "faculty",
    db_path: Optional[str] = None,
) -> int:
    """Commit validated hourly attendance or summary attendance records."""
    from auth import log_audit_action

    if mode == "summary":
        for rec in valid_records:
            update_student_attendance_summary(
                student_id=rec["student_id"],
                working_days=rec["working_days"],
                absent_days=rec["absent_days"],
                actor_id=actor_id,
                db_path=db_path,
            )
        return len(valid_records)

    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    try:
        for rec in valid_records:
            db.execute(
                """
                INSERT INTO attendance (id, class_id, student_id, attendance_date, subject_name, hour_number, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(student_id, attendance_date, subject_name, hour_number)
                DO UPDATE SET status = excluded.status
                """,
                (
                    uuid.uuid4().hex,
                    rec.get("class_id"),
                    rec["student_id"],
                    rec["attendance_date"],
                    rec["subject_name"],
                    rec["hour_number"],
                    rec["status"],
                    now,
                ),
            )
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()
    log_audit_action("staff", actor_id, "BULK_ATTENDANCE_IMPORT", f"Imported {len(valid_records)} attendance logs")
    return len(valid_records)


def batch_parse_vtu_marksheet_files(
    file_entries: List[Tuple[str, str]],
    default_sem: Optional[int] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse multiple uploaded VTU marksheet files automatically:
    extracts USN, semester number, official SGPA, and subjects, and matches to DB students.
    file_entries: list of (original_filename, saved_path)
    """
    from ocr_engine import extract_vtu_metadata, process_marksheet_file

    db = _get_conn(db_path)
    try:
        stu_map = {
            str(r["usn"]).strip().upper(): dict(r)
            for r in db.execute("SELECT id, usn, name, current_semester FROM students").fetchall()
        }
    finally:
        db.close()

    results: List[Dict[str, Any]] = []
    for fname, fpath in file_entries:
        raw_text, parsed_subs, is_low = process_marksheet_file(fpath)
        meta = extract_vtu_metadata(fpath, raw_text=raw_text)
        usn = meta.get("usn") or ""
        if not usn:
            # Fallback to filename USN token if OCR missed header
            stem = os.path.splitext(fname)[0].upper()
            for token in stem.replace("-", "_").split("_"):
                if token in stu_map:
                    usn = token
                    break

        stu = stu_map.get(usn)
        sem_num = meta.get("semester_number") or default_sem or (int(stu["current_semester"]) if stu else 1)
        computed = _compute_semester_metrics(parsed_subs, official_sgpa=meta.get("official_sgpa")) if parsed_subs else {
            "subjects": [],
            "sgpa": meta.get("official_sgpa") or 0.0,
            "credits_registered": 0.0,
            "credits_earned": 0.0,
            "failed_subjects": 0,
        }

        results.append(
            {
                "filename": fname,
                "saved_path": fpath,
                "detected_usn": usn,
                "matched": stu is not None,
                "student_id": stu["id"] if stu else None,
                "student_name": stu["name"] if stu else (meta.get("student_name") or "Unmatched"),
                "semester_number": int(sem_num),
                "sgpa": computed["sgpa"],
                "credits_earned": computed["credits_earned"],
                "credits_registered": computed["credits_registered"],
                "failed_subjects": computed["failed_subjects"],
                "subject_count": len(computed["subjects"]),
                "subjects": computed["subjects"],
                "is_low_confidence": is_low or len(computed["subjects"]) == 0,
            }
        )

    return results


# ============================================================
# 5. FUTURE-IMPROVEMENT AUTOMATIONS (ZIP, DRIVES, PROMOTION, MENTORING)
# ============================================================

def expand_marksheet_uploads_with_zip(
    saved_entries: List[Tuple[str, str]],
    extract_dir: str,
) -> List[Tuple[str, str]]:
    """Expand any .zip bundles in saved batch uploads safely into PDF/image marksheet entries."""
    os.makedirs(extract_dir, exist_ok=True)
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg"}
    expanded: List[Tuple[str, str]] = []

    for orig_name, fpath in saved_entries:
        if os.path.splitext(orig_name)[1].lower() == ".zip":
            try:
                with zipfile.ZipFile(fpath, "r") as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        base = os.path.basename(member.filename)
                        ext = os.path.splitext(base)[1].lower()
                        if not base or ext not in allowed_exts:
                            continue
                        dest = os.path.join(extract_dir, f"{uuid.uuid4().hex[:6]}_{base}")
                        with zf.open(member) as src, open(dest, "wb") as out:
                            out.write(src.read())
                        expanded.append((base, dest))
            except Exception:
                continue
        else:
            expanded.append((orig_name, fpath))

    return expanded


def review_placement_offer(
    placement_id: str,
    decision: str = "Verified",
    reviewer_id: str = "placement",
    remarks: str = "",
    db_path: Optional[str] = None,
) -> bool:
    """Approve ('Verified') or Reject ('Rejected') a student-submitted placement offer."""
    from auth import log_audit_action, send_notification

    norm_dec = "Verified" if str(decision).strip().lower() in ("verified", "approve", "approved") else "Rejected"
    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    try:
        row = db.execute("SELECT student_id, company_name, package_lpa FROM placements WHERE id = ?", (placement_id,)).fetchone()
        if not row:
            return False
        db.execute(
            """
            UPDATE placements
            SET verification_status = ?,
                remarks = CASE WHEN ? != '' THEN ? ELSE remarks END,
                updated_at = ?
            WHERE id = ?
            """,
            (norm_dec, remarks.strip(), remarks.strip(), now, placement_id),
        )
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()
    log_audit_action("staff", reviewer_id, "PLACEMENT_REVIEW", f"{norm_dec} offer {row['company_name']} for {row['student_id']}")
    send_notification(
        "student",
        row["student_id"],
        f"Placement Offer {norm_dec}: {row['company_name']}",
        f"Your submitted placement offer at {row['company_name']} has been marked '{norm_dec}' by the Placement Cell.",
        category="Placement",
    )
    return True


def save_placement_drive(
    company_name: str,
    job_role: str = "",
    package_lpa: float = 0.0,
    min_cgpa: float = 6.0,
    max_active_backlogs: int = 0,
    min_attendance_pct: float = 75.0,
    required_skills: str = "",
    exclude_already_placed: bool = False,
    require_next_sem_eligible: bool = True,
    drive_date: str = "",
    created_by: str = "placement",
    db_path: Optional[str] = None,
) -> str:
    """Save a campus placement drive and its eligibility filter specification."""
    drive_id = uuid.uuid4().hex
    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    try:
        db.execute(
            """
            INSERT INTO placement_drives (
                id, company_name, job_role, package_lpa, min_cgpa, max_active_backlogs,
                min_attendance_pct, required_skills, exclude_already_placed,
                require_next_sem_eligible, drive_date, created_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                drive_id,
                company_name.strip(),
                job_role.strip(),
                round(float(package_lpa or 0.0), 2),
                round(float(min_cgpa or 0.0), 2),
                int(max_active_backlogs or 0),
                round(float(min_attendance_pct or 0.0), 1),
                required_skills.strip(),
                1 if exclude_already_placed else 0,
                1 if require_next_sem_eligible else 0,
                drive_date.strip(),
                created_by,
                now,
            ),
        )
        db.commit()
    finally:
        db.close()
    return drive_id


def get_placement_drives(db_path: Optional[str] = None) -> pd.DataFrame:
    """Fetch saved campus placement drives sorted newest first."""
    db = _get_conn(db_path)
    try:
        rows = db.execute("SELECT * FROM placement_drives ORDER BY created_at DESC").fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()
    finally:
        db.close()


def match_students_for_drive(
    min_cgpa: float = 6.0,
    max_active_backlogs: int = 0,
    min_attendance_pct: float = 0.0,
    required_skills_csv: str = "",
    exclude_already_placed: bool = False,
    require_next_sem_eligible: bool = True,
    class_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Filter cohort automatically for a campus placement drive shortlist."""
    df = get_comprehensive_student_analytics(class_ids=class_ids, db_path=db_path)
    if df.empty:
        return df

    filtered = df[
        (df["cgpa"] >= float(min_cgpa))
        & (df["active_backlogs"] <= int(max_active_backlogs))
    ].copy()

    if float(min_attendance_pct) > 0:
        filtered = filtered[(filtered["has_attendance_data"] == False) | (filtered["attendance_pct"] >= float(min_attendance_pct))]
    if exclude_already_placed:
        filtered = filtered[filtered["is_placed"] == False]
    if require_next_sem_eligible:
        filtered = filtered[filtered["next_sem_eligible"] == True]

    skill_tokens = [s.strip().lower() for s in (required_skills_csv or "").split(",") if s.strip()]
    if skill_tokens:
        filtered = filtered[
            filtered["skills_list"].fillna("").str.lower().apply(lambda txt: any(tok in txt for tok in skill_tokens))
        ]

    return filtered.sort_values(by=["cgpa", "placement_readiness_score", "usn"], ascending=[False, False, True]).reset_index(drop=True)


def notify_shortlisted_students(
    student_ids: List[str],
    company_name: str,
    job_role: str = "",
    drive_date: str = "",
    actor_id: str = "placement",
    db_path: Optional[str] = None,
) -> int:
    """Send 1-click campus drive invitation notifications to shortlisted students."""
    from auth import log_audit_action, send_notification

    sent = 0
    for sid in student_ids:
        send_notification(
            "student",
            sid,
            f"Campus Placement Drive Invite: {company_name}",
            f"You are shortlisted/eligible for {company_name} ({job_role or 'Campus Drive'}) on {drive_date or 'Upcoming Date'}. Check Placement Cell announcements.",
            category="Placement Drive",
        )
        sent += 1
    log_audit_action("staff", actor_id, "DRIVE_SHORTLIST_NOTIFY", f"Invited {sent} students for {company_name}")
    return sent


def execute_semester_batch_promotion(
    student_ids: List[str],
    actor_id: str = "faculty",
    lock_previous_sem: bool = True,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote eligible students to current_semester + 1 (max 8) and lock completed semesters."""
    from auth import log_audit_action, send_notification

    if not student_ids:
        return {"promoted": 0, "skipped": 0}

    db = _get_conn(db_path)
    promoted = 0
    skipped = 0
    promoted_meta: List[Tuple[str, int]] = []
    try:
        for sid in student_ids:
            row = db.execute("SELECT current_semester FROM students WHERE id = ?", (sid,)).fetchone()
            if not row:
                skipped += 1
                continue
            cur_sem = int(row["current_semester"] or 1)
            if cur_sem >= 8:
                skipped += 1
                continue
            new_sem = cur_sem + 1
            db.execute("UPDATE students SET current_semester = ? WHERE id = ?", (new_sem, sid))
            if lock_previous_sem:
                db.execute(
                    "UPDATE semesters SET is_locked = 1 WHERE student_id = ? AND semester_number <= ?",
                    (sid, cur_sem),
                )
            promoted += 1
            promoted_meta.append((sid, new_sem))
        db.commit()
    finally:
        db.close()

    clear_streamlit_caches()
    log_audit_action("staff", actor_id, "BATCH_PROMOTION", f"Promoted {promoted} eligible students (skipped {skipped})")
    for sid, n_sem in promoted_meta:
        send_notification(
            "student",
            sid,
            f"Promoted to Semester {n_sem}",
            f"Congratulations! Based on VTU credit & backlog eligibility, your profile has been promoted to Semester {n_sem}.",
            category="Promotion",
        )
    return {"promoted": promoted, "skipped": skipped}


def add_mentoring_log(
    student_id: str,
    faculty_id: str,
    faculty_name: str,
    category: str,
    notes: str,
    action_taken: str = "",
    parent_notified: bool = False,
    notify_student: bool = True,
    db_path: Optional[str] = None,
) -> str:
    """Save a faculty mentoring/counseling entry for low attendance or academic intervention."""
    from auth import log_audit_action, send_notification

    if not (notes or "").strip():
        raise ValueError("Mentoring notes cannot be empty.")

    log_id = uuid.uuid4().hex
    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    try:
        db.execute(
            """
            INSERT INTO mentoring_logs (id, student_id, faculty_id, faculty_name, category, notes, action_taken, parent_notified, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                student_id,
                faculty_id,
                faculty_name,
                category.strip() or "Academic",
                notes.strip(),
                action_taken.strip(),
                1 if parent_notified else 0,
                now,
            ),
        )
        db.commit()
    finally:
        db.close()

    log_audit_action("staff", faculty_id, "MENTORING_LOG", f"[{category}] {action_taken or notes[:60]}", target_id=student_id)
    if notify_student:
        send_notification(
            "student",
            student_id,
            f"Faculty Mentoring Note ({category})",
            f"{faculty_name} recorded a mentoring note: {notes.strip()}" + (f" | Action: {action_taken.strip()}" if action_taken.strip() else ""),
            category="Mentoring",
        )
    return log_id


def get_mentoring_logs(student_id: Optional[str] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve faculty mentoring logs optionally filtered to one student."""
    db = _get_conn(db_path)
    try:
        sql = """
            SELECT m.*, st.usn, st.name AS student_name
            FROM mentoring_logs m
            JOIN students st ON m.student_id = st.id
        """
        params: List[Any] = []
        if student_id:
            sql += " WHERE m.student_id = ?"
            params.append(student_id)
        sql += " ORDER BY m.created_at DESC"
        rows = db.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()
    finally:
        db.close()


def get_marksheet_revisions(student_id: Optional[str] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve VTU marksheet revision / revaluation SGPA history."""
    db = _get_conn(db_path)
    try:
        sql = """
            SELECT r.*, st.usn, st.name AS student_name
            FROM marksheet_revisions r
            JOIN students st ON r.student_id = st.id
        """
        params: List[Any] = []
        if student_id:
            sql += " WHERE r.student_id = ?"
            params.append(student_id)
        sql += " ORDER BY r.created_at DESC"
        rows = db.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()
    finally:
        db.close()


def apply_for_placement_drive(
    drive_id: str,
    student_id: str,
    resume_headline: str = "",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate student eligibility and register application for a campus drive."""
    from auth import log_audit_action, send_notification
    db = _get_conn(db_path)
    try:
        drive = db.execute("SELECT * FROM placement_drives WHERE id = ?", (drive_id,)).fetchone()
        if not drive:
            return {"success": False, "message": "Placement drive not found."}
        drive = dict(drive)

        existing = db.execute(
            "SELECT id, application_stage FROM drive_applications WHERE drive_id = ? AND student_id = ?",
            (drive_id, student_id),
        ).fetchone()
        if existing:
            return {"success": False, "message": f"Already applied. Current stage: {existing['application_stage']}."}

        # Check eligibility via comprehensive analytics
        ana_df = get_comprehensive_student_analytics(db_path=db_path)
        m = ana_df[ana_df["student_id"] == student_id] if not ana_df.empty else pd.DataFrame()
        if m.empty:
            return {"success": False, "message": "Student academic profile not found."}
        sr = m.iloc[0]

        # Verify criteria
        reasons = []
        if float(sr.get("cgpa", 0.0) or 0.0) < float(drive.get("min_cgpa") or 0.0):
            reasons.append(f"CGPA {sr.get('cgpa', 0.0):.2f} < required {float(drive.get('min_cgpa') or 0.0):.2f}")
        if int(sr.get("active_backlogs", 0) or 0) > int(drive.get("max_active_backlogs") or 0):
            reasons.append(f"{int(sr.get('active_backlogs', 0))} backlogs > max allowed {int(drive.get('max_active_backlogs') or 0)}")
        if float(sr.get("attendance_pct", 0.0) or 0.0) < float(drive.get("min_attendance_pct") or 0.0):
            reasons.append(f"Attendance {sr.get('attendance_pct', 0.0):.1f}% < required {float(drive.get('min_attendance_pct') or 0.0):.1f}%")
        if drive.get("require_next_sem_eligible") and not sr.get("next_sem_eligible", True):
            reasons.append("Credit promotion ineligible")
        if drive.get("exclude_already_placed") and sr.get("is_placed"):
            reasons.append("Drive excludes already placed candidates")

        if reasons:
            return {"success": False, "message": "Eligibility criteria not met: " + "; ".join(reasons)}

        app_id = uuid.uuid4().hex
        now = datetime.now().isoformat()
        db.execute(
            """
            INSERT INTO drive_applications (id, drive_id, student_id, application_stage, stage_feedback, resume_headline, applied_at, updated_at)
            VALUES (?, ?, ?, 'Applied', '', ?, ?, ?)
            """,
            (app_id, drive_id, student_id, resume_headline.strip(), now, now),
        )
        db.commit()

        log_audit_action("student", student_id, "DRIVE_APPLY", f"Applied for {drive['company_name']}", target_id=drive_id)
        send_notification(
            "student",
            student_id,
            f"Applied: {drive['company_name']}",
            f"Your application for {drive['company_name']} ({drive.get('job_role', 'Drive')}) was successfully registered with the Placement Cell.",
            category="Placement",
        )
        return {"success": True, "application_id": app_id, "message": "Application submitted successfully!"}
    finally:
        db.close()


def get_drive_applications(
    drive_id: Optional[str] = None,
    student_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve drive applications joined with student and drive info."""
    db = _get_conn(db_path)
    try:
        sql = """
            SELECT da.*,
                   st.name AS student_name, st.usn, st.current_semester,
                   c.class_name,
                   pd.company_name, pd.job_role, pd.package_lpa, pd.drive_date
            FROM drive_applications da
            JOIN students st ON da.student_id = st.id
            LEFT JOIN classes c ON st.class_id = c.id
            JOIN placement_drives pd ON da.drive_id = pd.id
        """
        clauses = []
        params = []
        if drive_id:
            clauses.append("da.drive_id = ?")
            params.append(drive_id)
        if student_id:
            clauses.append("da.student_id = ?")
            params.append(student_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY da.updated_at DESC, da.applied_at DESC"
        rows = db.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()
    finally:
        db.close()


def update_drive_application_stage(
    app_id: str,
    new_stage: str,
    feedback: str = "",
    package_lpa: Optional[float] = None,
    reviewer_id: str = "placement",
    db_path: Optional[str] = None,
) -> bool:
    """Advance candidate through recruitment stages (Applied -> Shortlisted -> Assessment -> Technical -> HR -> Selected/Rejected)."""
    from auth import log_audit_action, send_notification
    db = _get_conn(db_path)
    try:
        row = db.execute(
            """
            SELECT da.*, pd.company_name, pd.job_role, pd.package_lpa AS default_lpa, st.name AS student_name
            FROM drive_applications da
            JOIN placement_drives pd ON da.drive_id = pd.id
            JOIN students st ON da.student_id = st.id
            WHERE da.id = ?
            """,
            (app_id,),
        ).fetchone()
        if not row:
            return False
        row = dict(row)
        now = datetime.now().isoformat()
        db.execute(
            "UPDATE drive_applications SET application_stage = ?, stage_feedback = ?, updated_at = ? WHERE id = ?",
            (new_stage, feedback.strip(), now, app_id),
        )
        db.commit()

        # If Selected, automatically record into placements table
        if new_stage == "Selected":
            eff_pkg = float(package_lpa if package_lpa is not None else (row.get("default_lpa") or 0.0))
            upsert_placement_record(
                student_id=row["student_id"],
                company_name=row["company_name"],
                job_role=row.get("job_role") or "Graduate Trainee",
                package_lpa=eff_pkg,
                offer_status="Placed",
                remarks=f"Selected through campus drive. Feedback: {feedback.strip()}",
                entered_by=reviewer_id,
                entered_role="placement",
                verification_status="Verified",
                source="campus_drive",
                db_path=db_path,
            )

        log_audit_action("staff", reviewer_id, "DRIVE_STAGE_UPDATE", f"{row['student_name']} -> {new_stage} ({row['company_name']})", target_id=app_id)
        send_notification(
            "student",
            row["student_id"],
            f"Drive Status: {row['company_name']}",
            f"Your application for {row['company_name']} is now: {new_stage}." + (f" Note: {feedback}" if feedback else ""),
            category="Placement",
        )
        return True
    finally:
        db.close()


def dispatch_parent_communication(
    student_id: str,
    faculty_id: str,
    faculty_name: str,
    comm_type: str,
    channel: str = "Official Notice",
    subject: str = "",
    message_body: str = "",
    db_path: Optional[str] = None,
) -> str:
    """Record an official intimation / warning / congratulation dispatched to parents."""
    from auth import log_audit_action, send_notification
    comm_id = uuid.uuid4().hex
    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    try:
        stu = db.execute("SELECT name, usn, phone, email FROM students WHERE id = ?", (student_id,)).fetchone()
        stu_dict = dict(stu) if stu else {}
        p_phone = stu_dict.get("phone") or ""
        p_email = stu_dict.get("email") or ""
        eff_subject = subject.strip() or f"VTU Official Notice: {comm_type}"
        eff_body = message_body.strip() or f"Official academic notice concerning student {stu_dict.get('name', student_id)} ({stu_dict.get('usn', '')})."

        db.execute(
            """
            INSERT INTO parent_communications (id, student_id, faculty_id, faculty_name, communication_type, channel, parent_phone, parent_email, subject, message_body, status, sent_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Dispatched', ?)
            """,
            (comm_id, student_id, faculty_id, faculty_name, comm_type, channel, p_phone, p_email, eff_subject, eff_body, now),
        )
        db.commit()
    finally:
        db.close()

    log_audit_action("staff", faculty_id, "PARENT_COMMUNICATION", f"[{comm_type}] {eff_subject}", target_id=student_id)
    send_notification(
        "student",
        student_id,
        f"Parent Notice Dispatched ({comm_type})",
        f"An official {comm_type} notice has been logged for parent intimation: {eff_subject}",
        category="Notice",
    )
    return comm_id


def get_parent_communications(
    student_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve parent communications optionally filtered by student."""
    db = _get_conn(db_path)
    try:
        sql = """
            SELECT pc.*, st.name AS student_name, st.usn
            FROM parent_communications pc
            JOIN students st ON pc.student_id = st.id
        """
        params = []
        if student_id:
            sql += " WHERE pc.student_id = ?"
            params.append(student_id)
        sql += " ORDER BY pc.sent_at DESC"
        rows = db.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()
    finally:
        db.close()


def generate_official_parent_notice_pdf(
    student_id: str,
    comm_type: str = "Attendance Warning",
    custom_remarks: str = "",
    db_path: Optional[str] = None,
) -> bytes:
    """Generate an official stamped institutional parent notice PDF using PyMuPDF."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    db = _get_conn(db_path)
    try:
        stu = db.execute("""
            SELECT s.*, c.class_name, c.class_teacher
            FROM students s
            LEFT JOIN classes c ON s.class_id = c.id
            WHERE s.id = ?
        """, (student_id,)).fetchone()
        if not stu:
            raise ValueError("Student not found.")
        stu = dict(stu)

        ana_df = get_comprehensive_student_analytics(db_path=db_path)
        m = ana_df[ana_df["student_id"] == student_id] if not ana_df.empty else pd.DataFrame()
        sr = m.iloc[0].to_dict() if not m.empty else {}
    finally:
        db.close()

    doc = fitz.open()
    page = doc.new_page()
    y = 50

    # Header / Institution Title
    page.insert_text((50, y), "T. JOHN INSTITUTE OF TECHNOLOGY", fontsize=15, fontname="helv", fill=(0.08, 0.18, 0.45))
    y += 18
    page.insert_text((50, y), "Affiliated to Visvesvaraya Technological University (VTU), Belagavi", fontsize=9.5, fontname="helv", fill=(0.4, 0.4, 0.4))
    y += 14
    page.insert_text((50, y), "DEPARTMENT OF ACADEMIC AFFAIRS & STUDENT MONITORING CELL", fontsize=10, fontname="helv", fill=(0.2, 0.2, 0.2))
    y += 12
    page.draw_line((50, y), (550, y), color=(0.08, 0.18, 0.45), width=1.5)
    y += 24

    # Notice Type Banner
    today_str = datetime.now().strftime("%d-%B-%Y")
    page.insert_text((50, y), f"Ref No: TJIT/ACAD/{datetime.now().year}/{stu['usn']}", fontsize=9, fontname="helv", fill=(0.3, 0.3, 0.3))
    page.insert_text((420, y), f"Date: {today_str}", fontsize=9, fontname="helv", fill=(0.3, 0.3, 0.3))
    y += 25

    banner_text = f"OFFICIAL INTIMATION: {comm_type.upper()}"
    page.draw_rect(fitz.Rect(50, y, 550, y + 26), color=(0.85, 0.2, 0.2), fill=(0.98, 0.94, 0.94))
    page.insert_text((60, y + 18), banner_text, fontsize=11, fontname="helv", fill=(0.6, 0.05, 0.05))
    y += 45

    # Salutation
    page.insert_text((50, y), f"To the Parent / Guardian of: {stu['name']} (USN: {stu['usn']})", fontsize=11, fontname="helv")
    y += 16
    page.insert_text((50, y), f"Program: B.E. / B.Tech | Class: {stu.get('class_name', 'N/A')} | Current Semester: {stu.get('current_semester', 'N/A')}", fontsize=9.5, fontname="helv", fill=(0.3, 0.3, 0.3))
    y += 26

    # Body
    att_pct = float(sr.get("attendance_pct", 0.0) or 0.0)
    recov = int(sr.get("classes_to_recover_75", 0) or 0)
    cgpa = float(sr.get("cgpa", 0.0) or 0.0)
    backlogs = int(sr.get("active_backlogs", 0) or 0)
    band = sr.get("att_condonation_band", "N/A")

    if "Attendance" in comm_type:
        body_lines = [
            f"This is an urgent official notice regarding the attendance record of your ward, {stu['name']}.",
            f"As per VTU Academic Regulations, a minimum attendance of 75.0% in each subject is mandatory",
            f"to be eligible to appear for the University Semester End Examinations (SEE).",
            "",
            f"• Current Attendance Rate: {att_pct:.1f}% ({band})",
            f"• Required Attendance Recovery: Must attend next {recov} consecutive classes (100% presence)",
            f"• Cumulative Academic CGPA: {cgpa:.2f} | Active Backlogs: {backlogs}",
            "",
            "Failure to make up the attendance deficit will lead to exam detention under VTU rules.",
            "You are advised to counsel your ward immediately and contact the department mentor.",
        ]
    elif "CIE" in comm_type or "Shortage" in comm_type:
        body_lines = [
            f"This is an official notice regarding Continuous Internal Evaluation (CIE) of your ward, {stu['name']}.",
            "VTU regulations require a minimum of 40% in internal evaluations to be eligible for hall tickets.",
            f"Status: {sr.get('exam_status', 'CIE Shortage')} — {sr.get('exam_reason', 'Low CIE marks')}",
            "",
            f"• Overall CGPA: {cgpa:.2f} | Attendance: {att_pct:.1f}%",
            "Please ensure your ward attends improvement sessions and internal assessments diligently.",
        ]
    elif "Placement" in comm_type:
        body_lines = [
            f"The Department and Placement Cell are delighted to congratulate your ward, {stu['name']},",
            "on securing a prestigious campus / off-campus placement offer!",
            f"Placement Status: {sr.get('placement_status', 'Placed')} ({sr.get('highest_package_lpa', 0.0):.2f} LPA peak).",
            "",
            "We commend their dedication and wish them continued success in their corporate journey.",
        ]
    else:
        body_lines = [
            f"Official institutional communication regarding your ward, {stu['name']} (USN: {stu['usn']}).",
            f"Status Summary: Attendance {att_pct:.1f}% | CGPA {cgpa:.2f} | Active Backlogs: {backlogs}.",
        ]

    for line in body_lines:
        page.insert_text((50, y), line, fontsize=10, fontname="helv", fill=(0.15, 0.15, 0.15))
        y += 16

    if custom_remarks.strip():
        y += 10
        page.insert_text((50, y), f"Faculty / HOD Remarks: {custom_remarks.strip()}", fontsize=9.5, fontname="helv", fill=(0.1, 0.2, 0.5))
        y += 20

    y += 40
    # Signature block
    page.insert_text((50, y), "_______________________", fontsize=10, fontname="helv")
    page.insert_text((380, y), "_______________________", fontsize=10, fontname="helv")
    y += 16
    page.insert_text((50, y), "Class Teacher / Mentor", fontsize=9.5, fontname="helv")
    page.insert_text((380, y), "Head of Department (HOD)", fontsize=9.5, fontname="helv")
    y += 14
    page.insert_text((50, y), f"{stu.get('class_teacher') or 'Faculty In-Charge'}", fontsize=8.5, fontname="helv", fill=(0.4, 0.4, 0.4))
    page.insert_text((380, y), "Department of Computer Science & Engg", fontsize=8.5, fontname="helv", fill=(0.4, 0.4, 0.4))

    return doc.tobytes()


def simulate_what_if_cgpa(
    student_id: str,
    projected_sgpas: Dict[int, float],
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Simulate future semester SGPAs to predict cumulative CGPA trajectory and unlocked campus drives."""
    db = _get_conn(db_path)
    try:
        sems = db.execute(
            "SELECT semester_number, sgpa FROM semesters WHERE student_id = ? ORDER BY semester_number ASC",
            (student_id,),
        ).fetchall()
        past_sems = {int(r["semester_number"]): float(r["sgpa"] or 0.0) for r in sems if r["sgpa"]}
        current_cgpa = round(sum(past_sems.values()) / len(past_sems), 2) if past_sems else 0.0

        all_sgpas = dict(past_sems)
        for s_num, s_val in projected_sgpas.items():
            all_sgpas[int(s_num)] = float(s_val)

        sim_cgpa = round(sum(all_sgpas.values()) / len(all_sgpas), 2) if all_sgpas else current_cgpa

        if sim_cgpa >= 8.0:
            sim_tier = "Super Dream Tier (>= 8.0 CGPA)"
        elif sim_cgpa >= 7.0:
            sim_tier = "Dream Tier (7.0 - 7.99 CGPA)"
        elif sim_cgpa >= 6.0:
            sim_tier = "Standard / Service Tier (6.0 - 6.99 CGPA)"
        else:
            sim_tier = "Remedial Track (< 6.0 CGPA)"

        drives_df = get_placement_drives(db_path=db_path)
        unlocked = []
        if not drives_df.empty:
            for _, d in drives_df.iterrows():
                min_c = float(d.get("min_cgpa") or 0.0)
                if current_cgpa < min_c <= sim_cgpa:
                    unlocked.append(f"{d['company_name']} ({min_c:.2f} CGPA cutoff, {float(d.get('package_lpa') or 0.0):.2f} LPA)")

        return {
            "current_cgpa": current_cgpa,
            "projected_cgpa": sim_cgpa,
            "delta_cgpa": round(sim_cgpa - current_cgpa, 2),
            "projected_tier": sim_tier,
            "unlocked_drives": unlocked,
            "semester_map": all_sgpas,
        }
    finally:
        db.close()


def bulk_log_fast_attendance(
    class_id: str,
    subject_name: str,
    attendance_date: str,
    hour_number: int,
    present_student_ids: List[str],
    absent_student_ids: List[str],
    db_path: Optional[str] = None,
) -> int:
    """Rapid 1-click batch attendance logger for an entire class hour."""
    now = datetime.now().isoformat()
    db = _get_conn(db_path)
    count = 0
    try:
        for sid in present_student_ids:
            att_id = uuid.uuid4().hex
            db.execute(
                """
                INSERT INTO attendance (id, class_id, student_id, attendance_date, subject_name, hour_number, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'Present', ?)
                """,
                (att_id, class_id, sid, attendance_date, subject_name.strip(), hour_number, now),
            )
            count += 1
        for sid in absent_student_ids:
            att_id = uuid.uuid4().hex
            db.execute(
                """
                INSERT INTO attendance (id, class_id, student_id, attendance_date, subject_name, hour_number, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'Absent', ?)
                """,
                (att_id, class_id, sid, attendance_date, subject_name.strip(), hour_number, now),
            )
            count += 1
        db.commit()
    finally:
        db.close()
    return count
