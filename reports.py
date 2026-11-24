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


def get_db(db_path: Optional[str] = None):
    conn = sqlite3.connect(db_path or DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def generate_student_pdf_report(student_id: str, db_path: Optional[str] = None) -> bytes:
    """
    Generate a complete, professional multi-page academic PDF report for a student.
    Includes profile, semester marks, SGPA/CGPA, attendance, certificates,
    skills, fee receipts, non-due status, letters, and predictions.
    """
    db = get_db(db_path)
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

        # Placements
        try:
            placements = db.execute(
                "SELECT * FROM placements WHERE student_id = ? ORDER BY COALESCE(package_lpa, 0) DESC, updated_at DESC",
                (student_id,),
            ).fetchall()
        except Exception:
            placements = []

        # Live eligibility & credits summary from academic_engine
        elig_summary = None
        try:
            from academic_engine import get_comprehensive_student_analytics
            ana_df = get_comprehensive_student_analytics(db_path=DB_FILE)
            if not ana_df.empty:
                m = ana_df[ana_df["student_id"] == student_id]
                if not m.empty:
                    elig_summary = m.iloc[0].to_dict()
        except Exception:
            elig_summary = None

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

        # Next-Semester Eligibility & Placements
        if y > 680:
            page = doc.new_page()
            y = 50
        y += 14
        page.insert_text((50, y), "PROMOTION ELIGIBILITY & PLACEMENT RECORDS", fontsize=12, fontname="helv", fill=(0.1, 0.2, 0.5))
        y += 14
        page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=1)
        y += 16
        if elig_summary:
            page.insert_text(
                (50, y),
                f"Credits Earned: {elig_summary.get('credits_earned', 0):g} / {elig_summary.get('credits_registered', 0):g} | Active Backlogs: {elig_summary.get('active_backlogs', 0)} | Next-Sem Eligibility: {elig_summary.get('eligibility_status', 'N/A')}",
                fontsize=9, fontname="helv",
            )
            y += 13
            page.insert_text((50, y), f"Eligibility Note: {elig_summary.get('eligibility_reason', '')}", fontsize=9, fontname="helv")
            y += 15
        page.insert_text((50, y), f"Placement Records ({len(placements)}):", fontsize=10, fontname="helv")
        y += 14
        if placements:
            for p in placements[:5]:
                pkg_str = f"{float(p['package_lpa']):.2f} LPA" if p["package_lpa"] else "Package N/A"
                page.insert_text(
                    (70, y),
                    f"• {p['company_name']} | Role: {p['job_role'] or 'N/A'} | {pkg_str} | Status: {p['offer_status']}",
                    fontsize=9, fontname="helv",
                )
                y += 12
        else:
            page.insert_text((70, y), "• No placement offers recorded yet (Status: Not Placed)", fontsize=9, fontname="helv")
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


def generate_vtu_hall_ticket_pdf(student_id: str, db_path: Optional[str] = None) -> bytes:
    """
    Generates an official Visvesvaraya Technological University (VTU) Semester End Examination (SEE)
    Admit Card / Hall Ticket with timetable, student credentials, and invigilator signature table.
    If the student has an attendance or CIE shortage, watermarks the conditional condonation status.
    """
    db = get_db(db_path)
    try:
        stu = db.execute("""
            SELECT s.*, c.class_name, c.class_teacher, c.section, c.semester AS class_semester
            FROM students s
            LEFT JOIN classes c ON s.class_id = c.id
            WHERE s.id = ?
        """, (student_id,)).fetchone()

        if not stu:
            raise ValueError(f"Student with ID '{student_id}' not found.")

        # Load current semester records (subjects, CIE, credits)
        cmr_rows = db.execute("""
            SELECT subject_code, subject_name, credits, ia1, ia2, ia3, assignment_marks, cie_total, internal_performance
            FROM current_semester_records
            WHERE student_id = ?
            ORDER BY subject_code ASC
        """, (student_id,)).fetchall()

        # Fallback to latest semester subjects if current semester records not yet finalized
        if not cmr_rows:
            cmr_rows = db.execute("""
                SELECT s.subject_code, s.subject_name, s.credits, s.internal_marks AS cie_total,
                       s.internal_marks AS internal_performance
                FROM subjects s
                JOIN semesters sem ON s.semester_id = sem.id
                WHERE sem.student_id = ? AND sem.semester_number = (
                    SELECT MAX(semester_number) FROM semesters WHERE student_id = ?
                )
                ORDER BY s.subject_code ASC
            """, (student_id, student_id)).fetchall()

        # Attendance calculation
        att_row = db.execute("""
            SELECT COUNT(*) AS total_hours,
                   SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_hours
            FROM attendance
            WHERE student_id = ?
        """, (student_id,)).fetchone()
        tot_hours = int(att_row["total_hours"] or 0) if att_row else 0
        pres_hours = int(att_row["present_hours"] or 0) if att_row else 0
        att_pct = round((pres_hours / tot_hours * 100.0), 1) if tot_hours > 0 else 80.0

        # Eligibility check: CIE >= 20/50 and Attendance >= 75%
        cie_fails = []
        for r in cmr_rows:
            cie = float(r["cie_total"] or r["internal_performance"] or 0.0)
            if cie < 20.0:
                cie_fails.append(f"{r['subject_code']} ({cie:.1f}/50)")

        is_eligible = (att_pct >= 75.0) and (len(cie_fails) == 0)
        is_condonable = (65.0 <= att_pct < 75.0) and (len(cie_fails) == 0)

        # Build PyMuPDF Document
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)

        # Header Banner
        banner_color = (0.08, 0.15, 0.35) if is_eligible else ((0.45, 0.25, 0.05) if is_condonable else (0.45, 0.08, 0.08))
        page.draw_rect(fitz.Rect(30, 25, 565, 85), color=banner_color, fill=banner_color)
        page.insert_text((110, 48), "VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI", fontsize=13, fontname="helv", fill=(1, 1, 1))
        page.insert_text((135, 64), "SEMESTER END EXAMINATION (SEE) — HALL TICKET / ADMISSION TICKET", fontsize=9.5, fontname="helv", fill=(0.85, 0.92, 1.0))
        page.insert_text((180, 78), f"Odd Semester 2026-2027 • B.E. Examination • Sem {stu['current_semester'] or 5}", fontsize=8.5, fontname="helv", fill=(0.7, 0.8, 0.95))

        # Institution Line
        page.insert_text((35, 105), "Institution : T. John Institute of Technology, Bengaluru (College Code: 1TJ)", fontsize=9.5, fontname="helv", fill=(0.15, 0.15, 0.15))
        page.insert_text((35, 118), f"Department : Dept of Computer Science & Engineering (Data Science) | Class: {stu['class_name'] or 'Data Science'}", fontsize=9, fontname="helv", fill=(0.3, 0.3, 0.3))
        page.draw_line((35, 126), (560, 126), color=(0.7, 0.7, 0.7), width=1)

        # Student Details Box
        page.draw_rect(fitz.Rect(35, 134, 465, 212), color=(0.8, 0.8, 0.8), fill=(0.97, 0.98, 1.0))
        page.insert_text((45, 150), f"Candidate Name : {stu['name']}", fontsize=9.5, fontname="helv")
        page.insert_text((270, 150), f"USN : {stu['usn']}", fontsize=10, fontname="helv", fill=(0.1, 0.2, 0.5))
        page.insert_text((45, 166), f"Current Semester : Semester {stu['current_semester'] or 5}", fontsize=9, fontname="helv")
        page.insert_text((270, 166), f"Section / Batch  : {stu['section'] or 'Section A'} (Admitted: {stu['admission_year'] or 2024})", fontsize=9, fontname="helv")
        page.insert_text((45, 182), f"Attendance %     : {att_pct:.1f}% ({pres_hours}/{tot_hours} hrs)", fontsize=9, fontname="helv")

        # Status Label
        if is_eligible:
            status_text = "ELIGIBLE FOR SEE"
            status_color = (0.0, 0.5, 0.2)
        elif is_condonable:
            status_text = "CONDONATION REQUIRED (65-75%)"
            status_color = (0.6, 0.4, 0.0)
        else:
            status_text = "EXAM DETAINED / SHORTAGE"
            status_color = (0.7, 0.1, 0.1)

        page.insert_text((270, 182), f"Eligibility Status: {status_text}", fontsize=9.5, fontname="helv", fill=status_color)
        page.insert_text((45, 198), "Examination Center: TJIT Campus, Center Code 1TJ", fontsize=9, fontname="helv")
        page.insert_text((270, 198), f"Hall Ticket ID : HT-2026-{stu['usn']}", fontsize=9, fontname="helv")

        # Photo Box
        page.draw_rect(fitz.Rect(475, 134, 560, 212), color=(0.6, 0.6, 0.6), fill=(0.94, 0.94, 0.94))
        page.insert_text((488, 170), "CANDIDATE", fontsize=7, fontname="helv", fill=(0.5, 0.5, 0.5))
        page.insert_text((498, 180), "PHOTO", fontsize=7, fontname="helv", fill=(0.5, 0.5, 0.5))

        # Subjects Table Header
        y = 228
        page.draw_rect(fitz.Rect(35, y, 560, y + 20), color=(0.1, 0.2, 0.45), fill=(0.1, 0.2, 0.45))
        page.insert_text((42, y + 14), "Sl", fontsize=8.5, fontname="helv", fill=(1, 1, 1))
        page.insert_text((65, y + 14), "Subject Code", fontsize=8.5, fontname="helv", fill=(1, 1, 1))
        page.insert_text((140, y + 14), "Subject Title", fontsize=8.5, fontname="helv", fill=(1, 1, 1))
        page.insert_text((310, y + 14), "Exam Date & Session", fontsize=8.5, fontname="helv", fill=(1, 1, 1))
        page.insert_text((435, y + 14), "CIE (50)", fontsize=8.5, fontname="helv", fill=(1, 1, 1))
        page.insert_text((485, y + 14), "Invigilator Sign", fontsize=8.5, fontname="helv", fill=(1, 1, 1))

        # Timetable mapping
        dates_map = {
            "BCS501": ("15-Dec-2026", "09:30 AM - 12:30 PM (Morning)"),
            "BCS502": ("18-Dec-2026", "09:30 AM - 12:30 PM (Morning)"),
            "BCS503": ("22-Dec-2026", "09:30 AM - 12:30 PM (Morning)"),
            "BRMK557": ("26-Dec-2026", "02:00 PM - 05:00 PM (Afternoon)"),
        }

        y += 20
        for idx, r in enumerate(cmr_rows, 1):
            code = str(r["subject_code"])
            name = str(r["subject_name"])
            dt_info = dates_map.get(code, (f"{14 + idx*3}-Dec-2026", "09:30 AM - 12:30 PM"))
            cie = float(r["cie_total"] or r["internal_performance"] or 0.0)

            page.draw_rect(fitz.Rect(35, y, 560, y + 24), color=(0.85, 0.85, 0.85))
            page.insert_text((44, y + 16), str(idx), fontsize=8.5, fontname="helv")
            page.insert_text((65, y + 16), code, fontsize=8.5, fontname="helv")
            page.insert_text((140, y + 16), name[:28], fontsize=8.5, fontname="helv")
            page.insert_text((310, y + 16), f"{dt_info[0]} | {dt_info[1][:14]}", fontsize=8, fontname="helv")

            cie_color = (0.7, 0.1, 0.1) if cie < 20.0 else (0.1, 0.1, 0.1)
            page.insert_text((438, y + 16), f"{cie:.1f} / 50", fontsize=8.5, fontname="helv", fill=cie_color)
            page.draw_line((480, y + 18), (555, y + 18), color=(0.8, 0.8, 0.8), width=0.5)
            y += 24

        # Instructions / Condonation Box
        y += 15
        if not is_eligible:
            page.draw_rect(fitz.Rect(35, y, 560, y + 105), color=(0.7, 0.2, 0.2), fill=(1.0, 0.96, 0.96))
            page.insert_text((45, y + 18), "ATTENDANCE / CIE DEFICIT WARNING NOTICE", fontsize=9.5, fontname="helv", fill=(0.7, 0.1, 0.1))
            page.insert_text((45, y + 34), f"• Current Attendance: {att_pct:.1f}% (Required: minimum 75.0% by VTU Statute 16OB.4).", fontsize=8, fontname="helv")
            if cie_fails:
                page.insert_text((45, y + 48), f"• CIE Shortage: The student scored < 20/50 in: {', '.join(cie_fails)}.", fontsize=8, fontname="helv")
            page.insert_text((45, y + 62), "• Action Required: Apply for Medical Condonation (if eligible, 65-75%) with official certificate.", fontsize=8, fontname="helv")
            page.insert_text((45, y + 76), "• Ineligible candidates will NOT be admitted to the examination hall without Principal approval.", fontsize=8, fontname="helv")
            page.insert_text((45, y + 90), "• Mandatory remedial mentoring session scheduled with Class Teacher / Faculty In-Charge.", fontsize=8, fontname="helv")
            y += 115
        else:
            page.draw_rect(fitz.Rect(35, y, 560, y + 105), color=(0.8, 0.85, 0.9), fill=(0.98, 0.99, 1.0))
            page.insert_text((45, y + 16), "IMPORTANT INSTRUCTIONS TO THE CANDIDATE", fontsize=9, fontname="helv", fill=(0.1, 0.2, 0.5))
            instructions = [
                "1. This Hall Ticket is valid only when accompanied by a valid College Identity Card.",
                "2. Candidates must occupy their designated seats 15 minutes prior to examination commencement.",
                "3. Possession of mobile phones, smartwatches, programmable calculators, or paper chits is strictly barred.",
                "4. Candidates are not permitted to leave the hall during the first 60 minutes or last 10 minutes of the exam.",
                "5. Ensure that the Room Invigilator signs in the respective column against each subject on the date of examination."
            ]
            iy = y + 32
            for inst in instructions:
                page.insert_text((45, iy), inst, fontsize=7.5, fontname="helv", fill=(0.25, 0.25, 0.25))
                iy += 13
            y += 115

        # Signatures
        y += 20
        page.draw_line((35, y), (560, y), color=(0.7, 0.7, 0.7), width=1)
        y += 35
        page.insert_text((55, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((62, y + 14), "Candidate's Signature", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        page.insert_text((240, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((248, y + 14), "Chief Superintendent", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        page.insert_text((420, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((425, y + 14), "Principal / Head of Inst.", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        # Footer
        page.insert_text((130, 810), f"VTU Examination Portal • Generated on {datetime.now().strftime('%d-%b-%Y %H:%M')} • Official Academic Record", fontsize=7.5, fontname="helv", fill=(0.5, 0.5, 0.5))

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    finally:
        db.close()


def generate_remedial_mentoring_plan_pdf(student_id: str, db_path: Optional[str] = None) -> bytes:
    """
    Generates a personalized VTU Academic Remedial & Faculty Mentoring Action Plan PDF.
    Diagnoses exact CIE and attendance deficits and prescribes concrete action milestones.
    """
    db = get_db(db_path)
    try:
        stu = db.execute("""
            SELECT s.*, c.class_name, c.class_teacher
            FROM students s
            LEFT JOIN classes c ON s.class_id = c.id
            WHERE s.id = ?
        """, (student_id,)).fetchone()

        if not stu:
            raise ValueError(f"Student with ID '{student_id}' not found.")

        # Current semester records
        cmr_rows = db.execute("""
            SELECT subject_code, subject_name, credits, ia1, ia2, ia3, assignment_marks, cie_total, internal_performance
            FROM current_semester_records
            WHERE student_id = ?
            ORDER BY subject_code ASC
        """, (student_id,)).fetchall()

        # Backlogs
        backlogs = db.execute("""
            SELECT subject_code, subject_name, semester_number, status
            FROM backlogs
            WHERE student_id = ? AND status = 'Active'
        """, (student_id,)).fetchall()

        # Attendance
        att_row = db.execute("""
            SELECT COUNT(*) AS total_hours,
                   SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_hours
            FROM attendance
            WHERE student_id = ?
        """, (student_id,)).fetchone()
        tot_hours = int(att_row["total_hours"] or 0) if att_row else 0
        pres_hours = int(att_row["present_hours"] or 0) if att_row else 0
        att_pct = round((pres_hours / tot_hours * 100.0), 1) if tot_hours > 0 else 80.0
        classes_needed = max(0, int((0.75 * tot_hours - pres_hours) / (1 - 0.75) + 0.999)) if att_pct < 75.0 else 0

        # Deficits
        weak_subjects = []
        for r in cmr_rows:
            cie = float(r["cie_total"] or r["internal_performance"] or 0.0)
            if cie < 25.0:
                deficit = max(0.0, 20.0 - cie)
                weak_subjects.append({
                    "code": r["subject_code"],
                    "name": r["subject_name"],
                    "cie": cie,
                    "deficit": deficit
                })

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)

        # Header Banner
        page.draw_rect(fitz.Rect(30, 25, 565, 80), color=(0.45, 0.08, 0.08), fill=(0.45, 0.08, 0.08))
        page.insert_text((125, 48), "T. JOHN INSTITUTE OF TECHNOLOGY, BENGALURU", fontsize=12.5, fontname="helv", fill=(1, 1, 1))
        page.insert_text((140, 64), "DEPARTMENT OF DATA SCIENCE • FACULTY MENTORING CELL", fontsize=9.5, fontname="helv", fill=(1, 0.85, 0.85))
        page.insert_text((155, 76), "Personalized Academic Remedial & Intervention Plan", fontsize=8.5, fontname="helv", fill=(1, 0.7, 0.7))

        # Student Details Box
        y = 95
        page.draw_rect(fitz.Rect(35, y, 560, y + 80), color=(0.8, 0.8, 0.8), fill=(0.99, 0.98, 0.98))
        page.insert_text((45, y + 18), f"Student Name : {stu['name']}", fontsize=9.5, fontname="helv")
        page.insert_text((300, y + 18), f"USN : {stu['usn']}", fontsize=9.5, fontname="helv", fill=(0.5, 0.1, 0.1))
        page.insert_text((45, y + 36), f"Current Semester : Semester {stu['current_semester'] or 5} ({stu['class_name'] or 'Data Science'})", fontsize=9, fontname="helv")
        page.insert_text((300, y + 36), f"Faculty Mentor : Prof. {stu['class_teacher'] or 'Sreevani'}", fontsize=9, fontname="helv")

        risk_label = "HIGH RISK / REMEDIAL MANDATORY" if (weak_subjects or backlogs or att_pct < 75.0) else "BORDERLINE / GUIDANCE PLAN"
        page.insert_text((45, y + 54), f"Risk Category : {risk_label}", fontsize=9, fontname="helv", fill=(0.7, 0.1, 0.1))
        page.insert_text((300, y + 54), f"Active Backlogs : {len(backlogs)} Subject(s)", fontsize=9, fontname="helv", fill=(0.7, 0.1, 0.1) if backlogs else (0.2, 0.5, 0.2))
        page.insert_text((45, y + 72), f"Current Attendance : {att_pct:.1f}% ({pres_hours}/{tot_hours} hrs)", fontsize=9, fontname="helv")
        page.insert_text((300, y + 72), f"Plan Date : {datetime.now().strftime('%d-%b-%Y')}", fontsize=9, fontname="helv")

        # Section 1: Deficit Diagnostics
        y = 192
        page.insert_text((35, y), "1. ACADEMIC & ATTENDANCE DEFICIT DIAGNOSTIC", fontsize=10.5, fontname="helv", fill=(0.4, 0.1, 0.1))
        page.draw_line((35, y + 4), (560, y + 4), color=(0.7, 0.7, 0.7), width=1)
        y += 20

        if weak_subjects:
            for ws in weak_subjects:
                page.insert_text((45, y), f"• Subject {ws['code']} ({ws['name']}): Current CIE is {ws['cie']:.1f}/50 (Min VTU Passing: 20.0/50).", fontsize=8.5, fontname="helv")
                y += 14
        else:
            page.insert_text((45, y), "• CIE Performance: All current semester subjects meeting basic 20/50 threshold.", fontsize=8.5, fontname="helv")
            y += 14

        if att_pct < 75.0:
            page.insert_text((45, y), f"• Attendance Shortage: Current attendance is {att_pct:.1f}%. Candidate requires {classes_needed} consecutive hours to reach 75%.", fontsize=8.5, fontname="helv", fill=(0.7, 0.1, 0.1))
            y += 14
        else:
            page.insert_text((45, y), f"• Attendance Standing: {att_pct:.1f}% (Above minimum VTU statutory requirement).", fontsize=8.5, fontname="helv")
            y += 14

        if backlogs:
            bk_str = ", ".join([f"{b['subject_code']} (Sem {b['semester_number']})" for b in backlogs])
            page.insert_text((45, y), f"• Active Backlogs: {bk_str}. Dedicated re-examination schedule required.", fontsize=8.5, fontname="helv", fill=(0.7, 0.1, 0.1))
            y += 14

        # Section 2: Prescribed Remedial Action Items
        y += 12
        page.insert_text((35, y), "2. MANDATORY REMEDIAL ACTION PLAN & TIMELINE", fontsize=10.5, fontname="helv", fill=(0.4, 0.1, 0.1))
        page.draw_line((35, y + 4), (560, y + 4), color=(0.7, 0.7, 0.7), width=1)
        y += 20

        actions = [
            ("Action 1", "Remedial Coaching Tutorials", "Attend 2 hours weekly special remedial problem-solving for flagged subjects.", "Weekly (Mon & Thu)"),
            ("Action 2", "Compensatory Attendance", "Maintain 100% attendance in all scheduled classes until final eligibility freeze.", "Immediate & Daily"),
            ("Action 3", "Assignment Re-submission", "Complete and submit revised assignment questionnaires for Units 1-3.", "Within 10 Days"),
            ("Action 4", "Improvement CIE Evaluation", "Appear for scheduled CIE Re-test to elevate continuous internal assessment marks.", "First Week of Dec"),
            ("Action 5", "Mentor Review Meetings", f"Report to Prof. {stu['class_teacher'] or 'Sreevani'} every Friday for progress verification.", "Weekly Check-in"),
        ]

        for tag, title, desc, timeline in actions:
            page.draw_rect(fitz.Rect(35, y, 560, y + 28), color=(0.85, 0.85, 0.85))
            page.insert_text((42, y + 14), tag, fontsize=8.5, fontname="helv", fill=(0.5, 0.1, 0.1))
            page.insert_text((105, y + 14), f"{title}: {desc[:55]}", fontsize=8, fontname="helv")
            page.insert_text((440, y + 14), f"Timeline: {timeline}", fontsize=7.5, fontname="helv", fill=(0.2, 0.4, 0.2))
            y += 31

        # Section 3: Candidate & Parent Undertaking
        y += 10
        page.insert_text((35, y), "3. CANDIDATE UNDERTAKING & PARENT ACKNOWLEDGEMENT", fontsize=10.5, fontname="helv", fill=(0.4, 0.1, 0.1))
        page.draw_line((35, y + 4), (560, y + 4), color=(0.7, 0.7, 0.7), width=1)
        y += 18
        page.insert_text((45, y), '"I understand that failure to fulfill this remedial plan or improve my CIE and attendance to the required', fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))
        y += 12
        page.insert_text((45, y), 'VTU thresholds will result in debarment from the Semester End Examination (SEE) as per VTU regulations."', fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        # Signatures
        y = 690
        page.draw_line((35, y), (560, y), color=(0.7, 0.7, 0.7), width=1)
        y += 35
        page.insert_text((50, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((60, y + 14), "Student's Signature", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        page.insert_text((190, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((200, y + 14), "Faculty Mentor", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        page.insert_text((330, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((345, y + 14), "Parent's Signature", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        page.insert_text((470, y), "______________________", fontsize=9, fontname="helv")
        page.insert_text((480, y + 14), "HOD / Principal", fontsize=8, fontname="helv", fill=(0.3, 0.3, 0.3))

        # Footer
        page.insert_text((160, 810), f"EduPredict AI Mentoring Cell • Generated on {datetime.now().strftime('%d-%b-%Y %H:%M')} • Official Remedial Record", fontsize=7.5, fontname="helv", fill=(0.5, 0.5, 0.5))

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    finally:
        db.close()


def generate_class_hall_tickets_zip(class_id: str, db_path: Optional[str] = None) -> bytes:
    """
    Generates a single ZIP file containing the official VTU Examination Hall Tickets
    for all students enrolled in the specified class.
    """
    import io
    import zipfile

    db = get_db(db_path)
    try:
        students = db.execute("""
            SELECT id, usn, name FROM students
            WHERE class_id = ?
            ORDER BY usn ASC
        """, (class_id,)).fetchall()

        if not students:
            students = db.execute("SELECT id, usn, name FROM students ORDER BY usn ASC").fetchall()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for s in students:
                try:
                    pdf_bytes = generate_vtu_hall_ticket_pdf(s["id"], db_path=db_path)
                    safe_name = "".join(c for c in s["name"] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
                    filename = f"VTU_HallTicket_{s['usn']}_{safe_name}.pdf"
                    zf.writestr(filename, pdf_bytes)
                except Exception as e:
                    zf.writestr(f"ERROR_{s['usn']}.txt", f"Failed to generate hall ticket for {s['usn']}: {e}")

        return zip_buffer.getvalue()
    finally:
        db.close()
