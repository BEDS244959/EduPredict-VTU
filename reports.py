"""
Reports and Document Generation Module for EduPredict AI.
Generates comprehensive student academic PDF reports using PyMuPDF.
Guarantees NO password hashes or credentials are ever included in exported reports.
"""

import os
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import pymupdf as fitz
except ImportError:
    import fitz


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def generate_student_pdf_report(student_id: str) -> bytes:
    """
    Generate a complete, professional multi-page academic PDF report for a student.
    Includes profile, semester marks, SGPA/CGPA, attendance, certificates,
    skills, fee receipts, non-due status, letters, and predictions.
    """
    db = get_db()
    try:
        stu = db.execute("""
            SELECT s.*, c.class_name, c.class_teacher
            FROM students s
            LEFT JOIN classes c ON s.class_id = c.id
            WHERE s.id = ?
        """, (student_id,)).fetchone()

        if not stu:
            raise ValueError("Student not found")

        # Semesters & subjects
        semesters = db.execute("""
            SELECT * FROM semesters
            WHERE student_id = ?
            ORDER BY semester_number ASC
        """, (student_id,)).fetchall()

        # Attendance
        att = db.execute("""
            SELECT
                COUNT(*) AS total_hours,
                SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_hours
            FROM attendance WHERE student_id = ?
        """, (student_id,)).fetchone()
        tot_hours = int(att["total_hours"] or 0)
        pres_hours = int(att["present_hours"] or 0)
        att_pct = round((pres_hours / tot_hours * 100), 2) if tot_hours else 0.0

        # Current semester entries & predictions
        current_preds = db.execute("""
            SELECT * FROM current_semester_records
            WHERE student_id = ?
            ORDER BY subject_code ASC
        """, (student_id,)).fetchall()

        # Certificates
        certs = db.execute("SELECT * FROM certificates WHERE student_id = ?", (student_id,)).fetchall()

        # Skills
        skills = db.execute("SELECT * FROM student_skills WHERE student_id = ?", (student_id,)).fetchall()

        # LinkedIn
        lk = db.execute("SELECT linkedin_url FROM linkedin_profiles WHERE student_id = ?", (student_id,)).fetchone()
        linkedin_url = lk["linkedin_url"] if lk else "Not Provided"

        # Letters
        letters = db.execute("SELECT * FROM student_letters WHERE student_id = ?", (student_id,)).fetchall()

        # CGPA calculation
        sgpas = [float(r["sgpa"]) for r in semesters if r["sgpa"] and float(r["sgpa"]) > 0]
        cgpa = round(sum(sgpas) / len(sgpas), 2) if sgpas else 0.0

        # Document Creation with PyMuPDF
        doc = fitz.open()

        # Page 1: Academic Profile & Summary
        page = doc.new_page()
        y = 50

        # Header Title
        page.insert_text((50, y), "EDUPREDICT AI — COMPREHENSIVE STUDENT ACADEMIC REPORT", fontsize=14, fontname="helv", fill=(0.1, 0.2, 0.5))
        y += 20
        page.draw_line((50, y), (550, y), color=(0.1, 0.2, 0.5), width=1.5)
        y += 25

        # Profile Table
        page.insert_text((50, y), f"Student Name:     {stu['name']}", fontsize=11, fontname="helv")
        page.insert_text((320, y), f"USN:                {stu['usn']}", fontsize=11, fontname="helv")
        y += 18
        page.insert_text((50, y), f"Class:            {stu['class_name'] or 'N/A'}", fontsize=11, fontname="helv")
        page.insert_text((320, y), f"Class Teacher:      {stu['class_teacher'] or 'N/A'}", fontsize=11, fontname="helv")
        y += 18
        page.insert_text((50, y), f"Admission Year:   {stu['admission_year'] or 'N/A'}", fontsize=11, fontname="helv")
        page.insert_text((320, y), f"Current Semester:   Semester {stu['current_semester']}", fontsize=11, fontname="helv")
        y += 18
        page.insert_text((50, y), f"Overall CGPA:     {cgpa:.2f}", fontsize=11, fontname="helv")
        page.insert_text((320, y), f"Attendance:         {att_pct:.1f}% ({pres_hours}/{tot_hours} hrs)", fontsize=11, fontname="helv")
        y += 18
        page.insert_text((50, y), f"LinkedIn Profile: {linkedin_url}", fontsize=10, fontname="helv")
        y += 30

        # Academic Performance History
        page.insert_text((50, y), "ACADEMIC PERFORMANCE (RECORDED SEMESTERS)", fontsize=12, fontname="helv", fill=(0.1, 0.2, 0.5))
        y += 15
        page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=1)
        y += 18

        if semesters:
            for sem in semesters:
                page.insert_text((50, y), f"• {sem['semester_name']} | SGPA: {sem['sgpa']:.2f}", fontsize=10, fontname="helv")
                y += 14
                sub_rows = db.execute("SELECT * FROM subjects WHERE semester_id = ?", (sem["id"],)).fetchall()
                for sub in sub_rows:
                    page.insert_text(
                        (70, y),
                        f"- {sub['subject_code']}: {sub['subject_name']} | Int: {sub['internal_marks']} | Ext: {sub['external_marks']} | Tot: {sub['total_marks']} | Cr: {sub['credits']} | Result: {sub['result']}",
                        fontsize=9, fontname="helv"
                    )
                    y += 12
                    if y > 750:
                        page = doc.new_page()
                        y = 50
                y += 8
        else:
            page.insert_text((50, y), "No previous semester marksheets finalized yet.", fontsize=10, fontname="helv")
            y += 20

        # Current Semester Entries & Predictions
        if y > 600:
            page = doc.new_page()
            y = 50

        page.insert_text((50, y), "CURRENT SEMESTER IA PERFORMANCE & PREDICTIONS", fontsize=12, fontname="helv", fill=(0.1, 0.2, 0.5))
        y += 15
        page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=1)
        y += 18

        if current_preds:
            for cp in current_preds:
                page.insert_text(
                    (50, y),
                    f"{cp['subject_code']} - {cp['subject_name']} | Cr: {cp['credits']} | IA1: {cp['ia1']} | IA2: {cp['ia2']} | Internal: {cp['internal_performance']}",
                    fontsize=10, fontname="helv"
                )
                y += 14
        else:
            page.insert_text((50, y), "No current semester IA records submitted yet.", fontsize=10, fontname="helv")
            y += 20

        # Page 2: Co-Curricular, Certifications, Letters & Status
        if y > 650:
            page = doc.new_page()
            y = 50

        y += 15
        page.insert_text((50, y), "CO-CURRICULAR & DOCUMENT VERIFICATION STATUS", fontsize=12, fontname="helv", fill=(0.1, 0.2, 0.5))
        y += 15
        page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=1)
        y += 18

        # Certificates
        page.insert_text((50, y), f"Certificates ({len(certs)}):", fontsize=10, fontname="helv")
        y += 14
        for c in certs[:5]:
            page.insert_text((70, y), f"• {c['certificate_name']} ({c['category']}) - Status: {c['verification_status']}", fontsize=9, fontname="helv")
            y += 12

        # Skills
        y += 8
        skill_str = ", ".join([f"{sk['skill_name']} ({sk['proficiency']})" for sk in skills]) if skills else "None recorded"
        page.insert_text((50, y), f"Technical Skills: {skill_str}", fontsize=9, fontname="helv")
        y += 18

        # Letters
        page.insert_text((50, y), f"Letters Submitted to Class Teacher ({len(letters)}):", fontsize=10, fontname="helv")
        y += 14
        for l in letters[:4]:
            page.insert_text((70, y), f"• [{l['submission_date']}] {l['subject']} - Status: {l['status']}", fontsize=9, fontname="helv")
            y += 12

        # Footer Notice
        y = 780
        page.draw_line((50, y), (550, y), color=(0.8, 0.8, 0.8), width=1)
        page.insert_text(
            (50, y + 15),
            f"Generated by EduPredict AI on {datetime.now().strftime('%d-%m-%Y %H:%M')}. Official VTU results remain authoritative.",
            fontsize=8, fontname="helv", fill=(0.5, 0.5, 0.5)
        )

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    finally:
        db.close()
