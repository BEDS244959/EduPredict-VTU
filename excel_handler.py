"""
Excel Handler Module for EduPredict AI.
Handles:
1. 4-Sheet Consolidated Class Excel Export (Pandas + OpenPyXL):
   - Sheet 1: Class Current Semester Marks
   - Sheet 2: Subject-wise Analysis
   - Sheet 3: Class Summary
   - Sheet 4: Submission Status
2. Student Bulk Excel Import:
   - Flexible column header mapping (Name, USN, Admission Year, Semester, Class, Email, Phone)
   - Validates columns, types, and values
   - Detects duplicate USNs in file and database
   - Generates pre-import preview and error diagnostics
   - Preserves existing students and auto-provisions logins
3. Class Marks Bulk Import (IA1, IA2, Final Marks)
"""

import io
import os
import re
import uuid
import sqlite3
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def normalize_str(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


# ============================================================
# 4-SHEET CONSOLIDATED CLASS EXCEL EXPORT
# ============================================================

def generate_class_consolidated_excel(
    class_name: str,
    semester: int,
    students_data: List[Dict[str, Any]],
    subjects_list: List[Dict[str, str]],
    subject_analysis_rows: List[Dict[str, Any]],
    class_summary_data: Dict[str, Any],
    submission_rows: List[Dict[str, Any]]
) -> bytes:
    """
    Generate the official 4-sheet EduPredict Class Excel report.
    Returns bytes of the .xlsx file.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Class Current Semester Marks
        sheet1_rows = []
        for s in students_data:
            row = {
                "USN": s.get("usn", ""),
                "Student Name": s.get("name", ""),
                "Class": class_name,
                "Semester": semester
            }
            for sub in subjects_list:
                code = sub.get("code") or sub.get("subject_code", "")
                sub_info = s.get("subjects", {}).get(code, {})
                row[f"{code} IA1"] = sub_info.get("ia1", "-")
                row[f"{code} IA2"] = sub_info.get("ia2", "-")
                row[f"{code} Internal"] = sub_info.get("internal_performance", "-")
                row[f"{code} Predicted Final"] = sub_info.get("predicted_final", "-")
                row[f"{code} Grade"] = sub_info.get("expected_result", "-")
                row[f"{code} Risk"] = sub_info.get("attention_level", "-")

            row["Predicted SGPA"] = s.get("predicted_sgpa", "-")
            row["Projected CGPA"] = s.get("expected_cgpa", "-")
            row["Risk Subjects Count"] = s.get("attention_count", 0)
            sheet1_rows.append(row)

        df_sheet1 = pd.DataFrame(sheet1_rows) if sheet1_rows else pd.DataFrame(columns=["USN", "Student Name", "Class", "Semester", "Predicted SGPA"])
        df_sheet1.to_excel(writer, sheet_name="Current Semester Marks", index=False)

        # Sheet 2: Subject-wise Analysis
        df_sheet2 = pd.DataFrame(subject_analysis_rows) if subject_analysis_rows else pd.DataFrame(
            columns=["Subject", "USN", "Student Name", "Credits", "IA1", "IA2", "Internal Performance", "Predicted Final Marks", "Expected Result", "Attention Level"]
        )
        df_sheet2.to_excel(writer, sheet_name="Subject-wise Analysis", index=False)

        # Sheet 3: Class Summary
        summary_meta = [
            {"Metric": "Class Name", "Value": class_name},
            {"Metric": "Semester", "Value": semester},
            {"Metric": "Total Enrolled Students", "Value": class_summary_data.get("total_students", 0)},
            {"Metric": "Submitted Students", "Value": class_summary_data.get("submitted_students", 0)},
            {"Metric": "Pending Students", "Value": class_summary_data.get("pending_students", 0)},
            {"Metric": "Class Average Predicted SGPA", "Value": class_summary_data.get("predicted_avg_sgpa", 0.0)},
        ]
        df_summary_meta = pd.DataFrame(summary_meta)
        df_summary_meta.to_excel(writer, sheet_name="Class Summary", index=False, startrow=0)

        sub_breakdown = class_summary_data.get("subject_breakdown", [])
        if sub_breakdown:
            df_sub_breakdown = pd.DataFrame(sub_breakdown)
            df_sub_breakdown.to_excel(writer, sheet_name="Class Summary", index=False, startrow=len(summary_meta) + 3)

        # Sheet 4: Submission Status
        df_sheet4 = pd.DataFrame(submission_rows) if submission_rows else pd.DataFrame(
            columns=["USN", "Student Name", "Submission Status", "Subjects Entered", "Last Updated"]
        )
        df_sheet4.to_excel(writer, sheet_name="Submission Status", index=False)

    output.seek(0)
    return output.getvalue()


# ============================================================
# COMPREHENSIVE DATA EXPORTERS (STUDENTS, ATTENDANCE, MARKS, MASTER)
# ============================================================

def export_students_roster_excel(class_id: Optional[str] = None) -> bytes:
    """Generate comprehensive Student Master Roster Excel (.xlsx)."""
    db = get_db()
    query = """
        SELECT s.id, s.name, s.usn, c.class_name, s.admission_year, s.current_semester, s.is_lateral_entry,
               s.email, s.phone, s.created_at,
               (SELECT COUNT(*) FROM backlogs b WHERE b.student_id = s.id AND b.status = 'Active') as active_backlogs,
               (SELECT sgpa FROM semesters sem WHERE sem.student_id = s.id ORDER BY sem.semester_number DESC LIMIT 1) as latest_sgpa,
               (SELECT COUNT(*) FROM attendance a WHERE a.student_id = s.id) as att_total,
               (SELECT COUNT(*) FROM attendance a WHERE a.student_id = s.id AND a.status = 'Present') as att_present
        FROM students s
        LEFT JOIN classes c ON s.class_id = c.id
    """
    params = []
    if class_id:
        query += " WHERE s.class_id = ?"
        params.append(class_id)
    query += " ORDER BY c.class_name ASC, s.name ASC"

    rows = db.execute(query, params).fetchall()
    db.close()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        roster_data = []
        for r in rows:
            tot = r["att_total"] or 0
            pres = r["att_present"] or 0
            att_pct = round((pres / tot * 100), 1) if tot > 0 else 100.0
            roster_data.append({
                "USN": r["usn"] or "N/A",
                "Student Name": r["name"],
                "Class": r["class_name"] or "Unassigned",
                "Current Semester": r["current_semester"] or 1,
                "Admission Year": r["admission_year"] or "",
                "Entry Category": "🚀 Lateral Entry" if r["is_lateral_entry"] == 1 else "Regular",
                "Email": r["email"] or "N/A",
                "Phone": r["phone"] or "N/A",
                "Latest Verified SGPA": r["latest_sgpa"] if r["latest_sgpa"] is not None else "N/A",
                "Attendance %": f"{att_pct}%",
                "Active Backlogs": r["active_backlogs"] or 0,
                "Registered Date": (r["created_at"] or "")[:10]
            })

        df_roster = pd.DataFrame(roster_data) if roster_data else pd.DataFrame(columns=["USN", "Student Name", "Class", "Current Semester", "Entry Category", "Email", "Phone"])
        df_roster.to_excel(writer, sheet_name="Student Master Roster", index=False)

        # Sheet 2: Class Distribution Summary
        if roster_data:
            df_sum = df_roster.groupby("Class").agg(
                Total_Students=("USN", "count"),
                Active_Backlogs=("Active Backlogs", "sum")
            ).reset_index()
            df_sum.to_excel(writer, sheet_name="Class Summary", index=False)

    output.seek(0)
    return output.getvalue()


def export_attendance_dataset_excel(class_id: Optional[str] = None) -> bytes:
    """Generate multi-sheet Attendance Report (.xlsx) with logs and compliance stats."""
    db = get_db()
    query_log = """
        SELECT a.attendance_date, a.hour_number, a.subject_name, s.usn, s.name, c.class_name, a.status, a.created_at
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        LEFT JOIN classes c ON s.class_id = c.id
    """
    params = []
    if class_id:
        query_log += " WHERE a.class_id = ?"
        params.append(class_id)
    query_log += " ORDER BY a.attendance_date DESC, a.hour_number ASC, s.name ASC"
    att_logs = db.execute(query_log, params).fetchall()

    # Compliance summary
    query_comp = """
        SELECT s.usn, s.name, c.class_name,
               COUNT(a.id) as total_hours,
               SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_hours
        FROM students s
        LEFT JOIN classes c ON s.class_id = c.id
        LEFT JOIN attendance a ON s.id = a.student_id
    """
    if class_id:
        query_comp += " WHERE s.class_id = ?"
    query_comp += " GROUP BY s.id ORDER BY c.class_name ASC, s.name ASC"
    comp_rows = db.execute(query_comp, params).fetchall()

    # Subject-wise attendance
    query_sub = """
        SELECT s.usn, s.name, a.subject_name,
               COUNT(a.id) as sub_total,
               SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as sub_pres
        FROM students s
        JOIN attendance a ON s.id = a.student_id
    """
    if class_id:
        query_sub += " WHERE s.class_id = ?"
    query_sub += " GROUP BY s.id, a.subject_name ORDER BY s.name ASC, a.subject_name ASC"
    sub_rows = db.execute(query_sub, params).fetchall()
    db.close()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Master Log
        log_data = []
        for r in att_logs:
            log_data.append({
                "Date": r["attendance_date"],
                "Hour / Period": f"Hour {r['hour_number']}",
                "Subject": r["subject_name"],
                "USN": r["usn"],
                "Student Name": r["name"],
                "Class": r["class_name"] or "N/A",
                "Status": r["status"]
            })
        df_log = pd.DataFrame(log_data) if log_data else pd.DataFrame(columns=["Date", "Hour", "Subject", "USN", "Student Name", "Class", "Status"])
        df_log.to_excel(writer, sheet_name="Hourly Attendance Log", index=False)

        # Sheet 2: Compliance Summary
        comp_data = []
        for r in comp_rows:
            tot = r["total_hours"] or 0
            pres = r["present_hours"] or 0
            pct = round((pres / tot * 100), 1) if tot > 0 else 100.0
            status_desc = "🚨 Critical (<75%)" if pct < 75.0 else ("⚠️ Warning (75-84%)" if pct < 85.0 else "✅ Satisfactory (≥85%)")
            comp_data.append({
                "USN": r["usn"],
                "Student Name": r["name"],
                "Class": r["class_name"] or "N/A",
                "Total Hours Conducted": tot,
                "Present Hours": pres,
                "Attendance %": f"{pct}%",
                "Compliance Status": status_desc
            })
        df_comp = pd.DataFrame(comp_data) if comp_data else pd.DataFrame(columns=["USN", "Student Name", "Class", "Total Hours Conducted", "Present Hours", "Attendance %", "Compliance Status"])
        df_comp.to_excel(writer, sheet_name="Student Compliance Roster", index=False)

        # Sheet 3: Subject Breakdown
        sub_data = []
        for r in sub_rows:
            tot = r["sub_total"] or 0
            pres = r["sub_pres"] or 0
            pct = round((pres / tot * 100), 1) if tot > 0 else 100.0
            sub_data.append({
                "USN": r["usn"],
                "Student Name": r["name"],
                "Subject": r["subject_name"],
                "Held": tot,
                "Attended": pres,
                "Subject %": f"{pct}%"
            })
        df_sub = pd.DataFrame(sub_data) if sub_data else pd.DataFrame(columns=["USN", "Student Name", "Subject", "Held", "Attended", "Subject %"])
        df_sub.to_excel(writer, sheet_name="Subject Attendance Breakdown", index=False)

    output.seek(0)
    return output.getvalue()


def export_master_institution_backup_excel() -> bytes:
    """Generate complete Institutional Database Multi-Sheet Master Workbook."""
    db = get_db()
    students = db.execute("SELECT s.*, c.class_name FROM students s LEFT JOIN classes c ON s.class_id = c.id ORDER BY s.name ASC").fetchall()
    classes = db.execute("SELECT * FROM classes ORDER BY class_name ASC").fetchall()
    subjects = db.execute("SELECT sub.*, c.class_name FROM master_subjects sub LEFT JOIN classes c ON sub.class_id = c.id ORDER BY sub.subject_code ASC").fetchall()
    marks = db.execute("""
        SELECT csr.*, s.name as student_name, s.usn, c.class_name
        FROM current_semester_records csr
        JOIN students s ON csr.student_id = s.id
        LEFT JOIN classes c ON s.class_id = c.id
        ORDER BY s.name ASC, csr.subject_code ASC
    """).fetchall()
    attendance = db.execute("""
        SELECT a.*, s.name as student_name, s.usn, c.class_name
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        LEFT JOIN classes c ON a.class_id = c.id
        ORDER BY a.attendance_date DESC, a.hour_number ASC
        LIMIT 5000
    """).fetchall()
    certs = db.execute("""
        SELECT c.*, s.name as student_name, s.usn
        FROM certificates c
        JOIN students s ON c.student_id = s.id
        ORDER BY c.uploaded_at DESC
    """).fetchall()
    letters = db.execute("""
        SELECT l.*, s.name as student_name, s.usn, c.class_name
        FROM student_letters l
        JOIN students s ON l.student_id = s.id
        LEFT JOIN classes c ON s.class_id = c.id
        ORDER BY l.submission_date DESC
    """).fetchall()
    db.close()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([dict(r) for r in students]).to_excel(writer, sheet_name="1. Students Directory", index=False)
        pd.DataFrame([dict(r) for r in classes]).to_excel(writer, sheet_name="2. Classes & Batches", index=False)
        pd.DataFrame([dict(r) for r in subjects]).to_excel(writer, sheet_name="3. Subject Master", index=False)
        pd.DataFrame([dict(r) for r in marks]).to_excel(writer, sheet_name="4. Academic Marks", index=False)
        pd.DataFrame([dict(r) for r in attendance]).to_excel(writer, sheet_name="5. Attendance Log", index=False)
        pd.DataFrame([dict(r) for r in certs]).to_excel(writer, sheet_name="6. Certificates & Forms", index=False)
        pd.DataFrame([dict(r) for r in letters]).to_excel(writer, sheet_name="7. Letters & Applications", index=False)

    output.seek(0)
    return output.getvalue()


# ============================================================
# STUDENT BULK EXCEL IMPORT
# ============================================================

COLUMN_ALIASES = {
    "name": ["name", "student_name", "student name", "full_name", "full name", "candidate name"],
    "usn": ["usn", "university_seat_number", "seat no", "roll_no", "roll no", "reg_no", "reg no", "registration number"],
    "admission_year": ["admission_year", "admission year", "adm_year", "adm year", "batch", "joining year", "year of admission"],
    "current_semester": ["current_semester", "current semester", "semester", "sem", "current sem"],
    "class": ["class", "class_name", "class name", "section", "branch", "dept", "department"],
    "email": ["email", "e-mail", "email_id", "email id", "student_email"],
    "phone": ["phone", "mobile", "contact", "phone_number", "mobile_number", "contact_no"],
    "is_lateral_entry": ["is_lateral_entry", "lateral_entry", "lateral entry", "is_lateral", "lateral", "entry_type", "diploma_entry", "entry category"]
}


def find_matched_column(df_columns: List[str], target_field: str) -> Optional[str]:
    """Find matching column name in DataFrame for target field using aliases."""
    normalized_cols = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in df_columns}
    for alias in COLUMN_ALIASES.get(target_field, []):
        norm_alias = re.sub(r'[^a-z0-9]', '', alias.lower())
        if norm_alias in normalized_cols:
            return normalized_cols[norm_alias]
    return None


def validate_student_excel(file_bytes_or_path, default_class_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate uploaded Excel workbook for bulk student import.
    Returns preview dataframe, valid rows, invalid rows, duplicate USNs, and errors.
    """
    try:
        df = pd.read_excel(file_bytes_or_path)
    except Exception as e:
        return {"valid": False, "error": f"Failed to read Excel file: {e}"}

    if df.empty:
        return {"valid": False, "error": "The uploaded Excel sheet contains no data rows."}

    # Detect matched columns
    name_col = find_matched_column(df.columns, "name")
    usn_col = find_matched_column(df.columns, "usn")
    adm_col = find_matched_column(df.columns, "admission_year")
    sem_col = find_matched_column(df.columns, "current_semester")
    class_col = find_matched_column(df.columns, "class")
    email_col = find_matched_column(df.columns, "email")
    phone_col = find_matched_column(df.columns, "phone")
    lat_col = find_matched_column(df.columns, "is_lateral_entry")

    if not name_col or not usn_col:
        return {
            "valid": False,
            "error": f"Required columns 'Name' and 'USN' could not be identified. Found columns: {list(df.columns)}"
        }

    db = get_db()
    try:
        # Load existing USNs to prevent duplicates
        existing_usn_rows = db.execute("SELECT usn FROM students WHERE usn IS NOT NULL").fetchall()
        existing_usns = {r["usn"].strip().upper() for r in existing_usn_rows if r["usn"]}

        # Load available classes
        classes = db.execute("SELECT id, class_name FROM classes").fetchall()
        class_name_map = {c["class_name"].strip().lower(): c["id"] for c in classes}
        class_id_map = {c["id"]: c["class_name"] for c in classes}
    finally:
        db.close()

    valid_records = []
    errors = []
    seen_file_usns = set()

    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel 1-based index (header is row 1)
        raw_name = normalize_str(row.get(name_col))
        raw_usn = normalize_str(row.get(usn_col)).upper()
        raw_adm = row.get(adm_col) if adm_col else None
        raw_sem = row.get(sem_col) if sem_col else None
        raw_cls = normalize_str(row.get(class_col)) if class_col else ""
        raw_email = normalize_str(row.get(email_col)) if email_col else ""
        raw_phone = normalize_str(row.get(phone_col)) if phone_col else ""
        raw_lat = normalize_str(row.get(lat_col)).lower() if lat_col else ""

        # Validate Name & USN
        if not raw_name:
            errors.append(f"Row {row_num}: Student Name is missing.")
            continue
        if not raw_usn:
            errors.append(f"Row {row_num}: USN is missing for '{raw_name}'.")
            continue

        if raw_usn in seen_file_usns:
            errors.append(f"Row {row_num}: Duplicate USN '{raw_usn}' found inside the Excel file.")
            continue
        seen_file_usns.add(raw_usn)

        if raw_usn in existing_usns:
            errors.append(f"Row {row_num}: USN '{raw_usn}' already exists in the database. (Skipped to prevent overwrite).")
            continue

        # Parse Lateral Entry
        is_lat_entry = 1 if raw_lat in ["1", "true", "yes", "y", "lateral", "lateral entry", "diploma"] else 0

        # Parse Year and Semester
        try:
            adm_year = int(float(raw_adm)) if raw_adm is not None and not pd.isna(raw_adm) else datetime.now().year
        except Exception:
            adm_year = datetime.now().year

        try:
            current_sem = int(float(raw_sem)) if raw_sem is not None and not pd.isna(raw_sem) else (3 if is_lat_entry else 1)
            if current_sem < 1 or current_sem > 8:
                current_sem = 3 if is_lat_entry else 1
        except Exception:
            current_sem = 3 if is_lat_entry else 1

        # Determine Class ID
        target_class_id = None
        if raw_cls and raw_cls.lower() in class_name_map:
            target_class_id = class_name_map[raw_cls.lower()]
        elif default_class_id:
            target_class_id = default_class_id
        elif classes:
            target_class_id = classes[0]["id"]
        else:
            errors.append(f"Row {row_num}: No class found or specified for '{raw_name}'. Please create a Class first.")
            continue

        valid_records.append({
            "name": raw_name,
            "usn": raw_usn,
            "class_id": target_class_id,
            "class_name": class_id_map.get(target_class_id, "Standard Class"),
            "admission_year": adm_year,
            "current_semester": current_sem,
            "is_lateral_entry": is_lat_entry,
            "email": raw_email,
            "phone": raw_phone
        })

    return {
        "valid": True,
        "total_rows": len(df),
        "valid_count": len(valid_records),
        "error_count": len(errors),
        "errors": errors,
        "valid_records": valid_records,
        "columns_matched": {
            "name": name_col,
            "usn": usn_col,
            "admission_year": adm_col,
            "current_semester": sem_col,
            "class": class_col,
            "email": email_col,
            "phone": phone_col,
            "is_lateral_entry": lat_col
        }
    }


def execute_student_bulk_import(valid_records: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """Insert validated student records into database and provision logins."""
    from auth import ensure_student_login, log_audit_action

    db = get_db()
    inserted_count = 0
    provisioned_passwords = []

    try:
        now = datetime.now().isoformat()
        for rec in valid_records:
            stu_id = uuid.uuid4().hex
            lat_flag = rec.get("is_lateral_entry", 0)
            db.execute("""
                INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, is_lateral_entry, email, phone, absent_days, working_days, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 90, ?)
            """, (stu_id, rec["class_id"], rec["name"], rec["usn"], rec["admission_year"], rec["current_semester"], lat_flag, rec["email"], rec["phone"], now))
            inserted_count += 1

        db.commit()

        # Auto provision logins
        for rec in valid_records:
            row = db.execute("SELECT id FROM students WHERE usn = ?", (rec["usn"],)).fetchone()
            if row:
                temp_pwd = ensure_student_login(row["id"], default_password="Student@123")
                provisioned_passwords.append(f"{rec['usn']}: Student@123")

        log_audit_action("staff", "bulk_import", "EXCEL_STUDENT_IMPORT", f"Successfully imported {inserted_count} student records from Excel")
        return inserted_count, provisioned_passwords
    finally:
        db.close()
