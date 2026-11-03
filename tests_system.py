import os
import sys
import unittest
import sqlite3
import pandas as pd

sys.path.insert(0, r'c:\Users\vivek\OneDrive\Desktop\EduPredict_AI')

from db_migrations import run_migrations, get_db_connection
from auth import (
    hash_password,
    verify_password,
    authenticate_student,
    authenticate_staff,
    create_first_staff,
    student_first_time_setup,
    provision_existing_students,
    reset_student_password
)
from ocr_engine import calculate_total, calculate_result, guess_credits
from ml.predict import (
    predict_subject_final_marks,
    calculate_predicted_sgpa,
    calculate_projected_cgpa,
    grade_point,
    grade_letter,
    determine_attention_level
)
from excel_handler import validate_student_excel, generate_class_consolidated_excel
from reports import generate_student_pdf_report

DB_FILE = r'c:\Users\vivek\OneDrive\Desktop\EduPredict_AI\edupredict.db'

class TestEduPredictAI(unittest.TestCase):

    def test_01_migrations(self):
        """Test database migrations."""
        succ = run_migrations(DB_FILE)
        self.assertTrue(succ)
        conn = get_db_connection(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()
        expected_tables = ['classes', 'students', 'semesters', 'subjects', 'master_subjects', 'backlogs', 'attendance', 'staff_users', 'student_users', 'notifications', 'audit_logs']
        for t in expected_tables:
            self.assertIn(t, tables, f"Missing table {t}")

    def test_02_password_hashing(self):
        """Test PBKDF2 HMAC SHA-256 password hashing with salt."""
        pwd = "SecureStudentPass@2026"
        hashed = hash_password(pwd)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPass", hashed))

    def test_03_student_auth(self):
        """Test student login authentication."""
        conn = get_db_connection(DB_FILE)
        stu = conn.execute("SELECT id, usn, name FROM students ORDER BY usn ASC LIMIT 1").fetchone()
        conn.close()
        self.assertIsNotNone(stu, "No student found in DB")
        reset_student_password(stu["id"], "Student@123")
        succ, u_dict, p_dict, msg = authenticate_student(stu["usn"], "Student@123")
        self.assertTrue(succ, f"Auth failed: {msg}")
        self.assertEqual(u_dict["usn"], stu["usn"])
        self.assertEqual(p_dict["name"], stu["name"])

    def test_04_marks_and_grade_calculations(self):
        """Test VTU marks, grades, and credit-weighted SGPA."""
        # VTU total = internal + external
        tot = calculate_total(42.0, 38.0)
        self.assertEqual(tot, 80.0)
        res = calculate_result(42.0, 38.0)
        self.assertEqual(res, "P")

        # Grade letters & points
        self.assertEqual(grade_point(95), 10)
        self.assertEqual(grade_letter(95), "O")
        self.assertEqual(grade_point(85), 9)
        self.assertEqual(grade_letter(85), "A+")
        self.assertEqual(grade_point(35), 0)
        self.assertEqual(grade_letter(35), "F")

        # Credit weighted SGPA: 4 credits of 90 (10) + 4 credits of 80 (9) = (40 + 36)/8 = 9.5
        mock_preds = [
            {"predicted_final": 90, "credits": 4.0},
            {"predicted_final": 80, "credits": 4.0}
        ]
        sgpa = calculate_predicted_sgpa(mock_preds)
        self.assertAlmostEqual(sgpa, 9.5, places=2)

    def test_05_ml_prediction_pipeline(self):
        """Test real ML model inference and attention recommendations."""
        pred = predict_subject_final_marks(
            student_id="a85ba3f65a3648e8b87393d62370eca5",
            subject_code="21CS51",
            subject_name="Database Management Systems",
            ia1=40.0,
            ia2=42.0,
            credits=4.0
        )
        self.assertIn("predicted_final", pred)
        self.assertGreaterEqual(pred["predicted_final"], 0.0)
        self.assertLessEqual(pred["predicted_final"], 100.0)
        self.assertIn("attention_level", pred)
        self.assertIn("suggestions", pred)
        self.assertGreater(len(pred["suggestions"]), 0)

    def test_06_pdf_report_generation(self):
        """Test PDF academic report generation."""
        conn = get_db_connection(DB_FILE)
        stu = conn.execute("SELECT id FROM students LIMIT 1").fetchone()
        conn.close()
        self.assertIsNotNone(stu, "No student found in DB")
        pdf_bytes = generate_student_pdf_report(stu["id"])
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(len(pdf_bytes) > 500, "PDF bytes too small")

    def test_07_excel_consolidated_export(self):
        """Test multi-sheet Excel generation."""
        students_data = [{
            "usn": "1TJ24DS028", "name": "Vivek S",
            "subjects": {"21CS51": {"ia1": 40, "ia2": 42, "internal_performance": 41, "predicted_final": 82, "expected_result": "A+", "attention_level": "Stable"}},
            "predicted_sgpa": "8.80", "expected_cgpa": "8.50", "attention_count": 0
        }]
        excel_bytes = generate_class_consolidated_excel(
            class_name="Data Science",
            semester=5,
            students_data=students_data,
            subjects_list=[{"code": "21CS51", "name": "DBMS"}],
            subject_analysis_rows=[],
            class_summary_data={"total_students": 1, "submitted_students": 1, "pending_students": 0, "predicted_avg_sgpa": 8.8},
            submission_rows=[]
        )
        self.assertIsNotNone(excel_bytes)
    def test_08_admin_and_snapshots(self):
        """Test admin metrics, database snapshot creation, and semester locking."""
        from admin_portal import get_system_stats, BACKUPS_DIR
        stats = get_system_stats()
        self.assertIn("students", stats)
        self.assertIn("classes", stats)
        self.assertGreater(stats["students"], 0)
        self.assertGreater(stats["classes"], 0)

        # Test snapshot creation
        snap_file = os.path.join(BACKUPS_DIR, "test_snapshot.db")
        import shutil
        shutil.copy2(DB_FILE, snap_file)
        self.assertTrue(os.path.exists(snap_file))
        self.assertGreater(os.path.getsize(snap_file), 1000)
        if os.path.exists(snap_file):
            os.remove(snap_file)

if __name__ == '__main__':
    unittest.main()
