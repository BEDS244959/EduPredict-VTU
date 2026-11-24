import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import zipfile
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from db_migrations import run_migrations, get_db_connection
from auth import (
    hash_password,
    verify_password,
    authenticate_student,
    authenticate_staff,
    create_first_staff,
    student_first_time_setup,
    provision_existing_students,
    reset_student_password,
)
from ocr_engine import (
    calculate_total,
    calculate_result,
    guess_credits,
    extract_vtu_metadata_from_text,
    extract_offer_letter_details_from_text,
)
from ml.predict import (
    predict_subject_final_marks,
    calculate_predicted_sgpa,
    calculate_projected_cgpa,
    grade_point,
    grade_letter,
    determine_attention_level,
)
from excel_handler import (
    validate_student_excel,
    generate_class_consolidated_excel,
    export_segregated_report_excel,
)
from reports import (
    generate_student_pdf_report,
    generate_vtu_hall_ticket_pdf,
    generate_remedial_mentoring_plan_pdf,
    generate_class_hall_tickets_zip,
    generate_executive_class_ledger_pdf,
)
import academic_engine as ae
import auth as auth_mod

DB_FILE = os.path.join(BASE_DIR, "edupredict.db")


class TestEduPredictAI(unittest.TestCase):

    def _first_student(self):
        conn = get_db_connection(DB_FILE)
        stu = conn.execute("SELECT id, usn, name, class_id FROM students ORDER BY created_at ASC LIMIT 1").fetchone()
        conn.close()
        return dict(stu) if stu else None

    def test_01_migrations(self):
        """Test database migrations including placements & academic_rules."""
        succ = run_migrations(DB_FILE)
        self.assertTrue(succ)
        conn = get_db_connection(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        cls_cols = [r[1] for r in cur.execute("PRAGMA table_info(classes)").fetchall()]
        conn.close()
        expected_tables = [
            "classes", "students", "semesters", "subjects", "master_subjects",
            "backlogs", "attendance", "staff_users", "student_users",
            "notifications", "audit_logs", "placements", "academic_rules",
        ]
        for t in expected_tables:
            self.assertIn(t, tables, f"Missing table {t}")
        self.assertIn("staff_tjit", cls_cols)

    def test_02_password_hashing(self):
        """Test PBKDF2 HMAC SHA-256 password hashing with salt."""
        pwd = "SecureStudentPass@2026"
        hashed = hash_password(pwd)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPass", hashed))

    def test_03_student_auth_isolated(self):
        """Test student password setup & authentication on an isolated temporary copy of DB."""
        stu = self._first_student()
        self.assertIsNotNone(stu)
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_db = os.path.join(tmpd, "test_auth.db")
            shutil.copy2(DB_FILE, tmp_db)
            orig_auth_db = auth_mod.DB_FILE
            auth_mod.DB_FILE = tmp_db
            try:
                reset_student_password(stu["id"], "Student@123")
                succ, u_dict, p_dict, msg = authenticate_student(stu["usn"], "Student@123")
                self.assertTrue(succ, f"Auth failed: {msg}")
                self.assertEqual(u_dict["usn"], stu["usn"])
                self.assertEqual(p_dict["name"], stu["name"])
            finally:
                auth_mod.DB_FILE = orig_auth_db

    def test_04_marks_and_grade_calculations(self):
        """Test VTU marks, grades, and credit-weighted SGPA."""
        tot = calculate_total(42.0, 38.0)
        self.assertEqual(tot, 80.0)
        res = calculate_result(42.0, 38.0)
        self.assertEqual(res, "P")

        self.assertEqual(grade_point(95), 10)
        self.assertEqual(grade_letter(95), "O")
        self.assertEqual(grade_point(85), 9)
        self.assertEqual(grade_letter(85), "A+")
        self.assertEqual(grade_point(35), 0)
        self.assertEqual(grade_letter(35), "F")

        mock_preds = [
            {"predicted_final": 90, "credits": 4.0},
            {"predicted_final": 80, "credits": 4.0},
        ]
        sgpa = calculate_predicted_sgpa(mock_preds)
        self.assertAlmostEqual(sgpa, 9.5, places=2)

    def test_05_ml_prediction_pipeline(self):
        """Test real ML model inference and attention recommendations."""
        stu = self._first_student()
        self.assertIsNotNone(stu)
        pred = predict_subject_final_marks(
            student_id=stu["id"],
            subject_code="21CS51",
            subject_name="Database Management Systems",
            ia1=40.0,
            ia2=42.0,
            credits=4.0,
        )
        self.assertIn("predicted_final", pred)
        self.assertGreaterEqual(pred["predicted_final"], 0.0)
        self.assertLessEqual(pred["predicted_final"], 100.0)
        self.assertIn("attention_level", pred)
        self.assertIn("suggestions", pred)
        self.assertGreater(len(pred["suggestions"]), 0)

    def test_06_pdf_report_generation(self):
        """Test PDF academic report generation including eligibility & placements block."""
        stu = self._first_student()
        self.assertIsNotNone(stu)
        pdf_bytes = generate_student_pdf_report(stu["id"])
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(len(pdf_bytes) > 500, "PDF bytes too small")

    def test_07_excel_consolidated_export(self):
        """Test multi-sheet Excel generation."""
        students_data = [{
            "usn": "1TJ24DS028",
            "name": "Vivek S",
            "subjects": {
                "21CS51": {
                    "ia1": 40,
                    "ia2": 42,
                    "internal_performance": 41,
                    "predicted_final": 82,
                    "expected_result": "A+",
                    "attention_level": "Stable",
                }
            },
            "predicted_sgpa": "8.80",
            "expected_cgpa": "8.50",
            "attention_count": 0,
        }]
        excel_bytes = generate_class_consolidated_excel(
            class_name="Data Science",
            semester=5,
            students_data=students_data,
            subjects_list=[{"code": "21CS51", "name": "DBMS"}],
            subject_analysis_rows=[],
            class_summary_data={
                "total_students": 1,
                "submitted_students": 1,
                "pending_students": 0,
                "predicted_avg_sgpa": 8.8,
            },
            submission_rows=[],
        )
        self.assertIsNotNone(excel_bytes)

    def test_08_admin_and_snapshots(self):
        """Test admin metrics and database snapshot creation."""
        from admin_portal import get_system_stats, BACKUPS_DIR
        os.makedirs(BACKUPS_DIR, exist_ok=True)
        stats = get_system_stats()
        self.assertIn("students", stats)
        self.assertIn("classes", stats)
        self.assertGreater(stats["students"], 0)
        self.assertGreater(stats["classes"], 0)

        snap_file = os.path.join(BACKUPS_DIR, "test_snapshot.db")
        shutil.copy2(DB_FILE, snap_file)
        self.assertTrue(os.path.exists(snap_file))
        self.assertGreater(os.path.getsize(snap_file), 1000)
        if os.path.exists(snap_file):
            os.remove(snap_file)

    def test_09_placement_faculty_role_registration_and_auth(self):
        """Test creating and authenticating a dedicated 'placement' role faculty account."""
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_db = os.path.join(tmpd, "test_place_role.db")
            run_migrations(tmp_db)
            orig_auth_db = auth_mod.DB_FILE
            auth_mod.DB_FILE = tmp_db
            try:
                succ, msg = create_first_staff(
                    "PLACE01", "Prof. Placement Head", "Place@1234",
                    email="placement@college.edu", role="placement",
                )
                self.assertTrue(succ, msg)
                ok, s_dict, auth_msg = authenticate_staff("PLACE01", "Place@1234")
                self.assertTrue(ok, auth_msg)
                self.assertEqual(s_dict["role"], "placement")
            finally:
                auth_mod.DB_FILE = orig_auth_db

    def test_10_vtu_result_save_backlog_sync_and_segregation(self):
        """Test automated VTU upload -> backlog sync -> SGPA/marks/credits/eligibility/low-att/placement segregation."""
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_db = os.path.join(tmpd, "test_seg.db")
            run_migrations(tmp_db)
            orig_auth_db = auth_mod.DB_FILE
            auth_mod.DB_FILE = tmp_db
            try:
                conn = get_db_connection(tmp_db)
                conn.execute("INSERT INTO classes (id, class_name, semester, section, created_at) VALUES ('cls1', '5th Sem DS', 5, 'A', '2026-01-01')")
                conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, working_days, absent_days, created_at) VALUES ('stu1', 'cls1', 'Topper Student', '1TJ24DS001', 2024, 5, 100, 5, '2026-01-01')")
                conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, working_days, absent_days, created_at) VALUES ('stu2', 'cls1', 'At Risk Student', '1TJ24DS002', 2024, 5, 100, 40, '2026-01-01')")
                conn.commit()
                conn.close()

                # Student 1 passes 4 high-credit subjects -> high SGPA, eligible, placed
                res1 = ae.save_student_semester_result(
                    student_id="stu1",
                    sem_num=4,
                    subjects=[
                        {"subject_code": "BCS401", "subject_name": "ADA", "internal_marks": 48, "external_marks": 45, "total_marks": 93, "credits": 4.0, "result": "P"},
                        {"subject_code": "BCS402", "subject_name": "MC", "internal_marks": 45, "external_marks": 42, "total_marks": 87, "credits": 4.0, "result": "P"},
                        {"subject_code": "BCS403", "subject_name": "DBMS", "internal_marks": 44, "external_marks": 40, "total_marks": 84, "credits": 4.0, "result": "P"},
                        {"subject_code": "BCS405", "subject_name": "DMS", "internal_marks": 46, "external_marks": 44, "total_marks": 90, "credits": 4.0, "result": "P"},
                        {"subject_code": "BCSL404", "subject_name": "ADA Lab", "internal_marks": 48, "external_marks": 47, "total_marks": 95, "credits": 1.0, "result": "P"},
                    ],
                    db_path=tmp_db,
                )
                self.assertEqual(res1["credits_earned"], 17.0)
                self.assertEqual(res1["backlogs_activated"], 0)

                # Student 2 fails 3 subjects (credits earned = 8 < 16 min) + 60% attendance
                res2 = ae.save_student_semester_result(
                    student_id="stu2",
                    sem_num=4,
                    subjects=[
                        {"subject_code": "BCS401", "subject_name": "ADA", "internal_marks": 20, "external_marks": 10, "total_marks": 30, "credits": 4.0, "result": "F"},
                        {"subject_code": "BCS402", "subject_name": "MC", "internal_marks": 18, "external_marks": 12, "total_marks": 30, "credits": 4.0, "result": "F"},
                        {"subject_code": "BCS403", "subject_name": "DBMS", "internal_marks": 35, "external_marks": 35, "total_marks": 70, "credits": 4.0, "result": "P"},
                        {"subject_code": "BCS405", "subject_name": "DMS", "internal_marks": 36, "external_marks": 36, "total_marks": 72, "credits": 4.0, "result": "P"},
                    ],
                    db_path=tmp_db,
                )
                self.assertEqual(res2["credits_earned"], 8.0)
                self.assertEqual(res2["backlogs_activated"], 2)

                # Re-attempt clears 1 backlog automatically
                res2_clear = ae.save_student_semester_result(
                    student_id="stu2",
                    sem_num=5,
                    subjects=[
                        {"subject_code": "BCS401", "subject_name": "ADA", "internal_marks": 35, "external_marks": 35, "total_marks": 70, "credits": 4.0, "result": "P"},
                    ],
                    db_path=tmp_db,
                )
                self.assertEqual(res2_clear["backlogs_cleared"], 1)

                # Place student 1
                ae.upsert_placement_record(
                    student_id="stu1",
                    company_name="Infosys",
                    job_role="System Engineer",
                    package_lpa=6.5,
                    offer_status="Placed",
                    entered_by="PLACE01",
                    db_path=tmp_db,
                )

                # Verify segregation output + Excel export
                rules = ae.save_academic_rules({"low_attendance_threshold": 75.0, "max_backlogs_next_sem": 4, "min_credits_next_sem": 16.0}, db_path=tmp_db)
                df_all = ae.get_comprehensive_student_analytics(rules=rules, db_path=tmp_db)
                seg = ae.segregate_students(df_all, rules=rules)

                self.assertEqual(seg["summary"]["total_students"], 2)
                self.assertEqual(seg["summary"]["eligible_count"], 1)
                self.assertEqual(seg["summary"]["not_eligible_count"], 1)
                self.assertEqual(seg["summary"]["low_attendance_count"], 1)
                self.assertEqual(seg["summary"]["placed_count"], 1)
                self.assertEqual(seg["summary"]["not_placed_count"], 1)
                self.assertEqual(seg["sgpa_df"].iloc[0]["usn"], "1TJ24DS001")
                self.assertEqual(seg["low_attendance_df"].iloc[0]["usn"], "1TJ24DS002")
                self.assertEqual(seg["placed_df"].iloc[0]["placement_companies"], "Infosys")

                xl = export_segregated_report_excel(seg, rules=rules)
                self.assertGreater(len(xl), 1000)

                # Verify individual placement record visible for stu1 and empty for stu2
                self.assertEqual(len(ae.get_student_placements("stu1", db_path=tmp_db)), 1)
                self.assertEqual(len(ae.get_student_placements("stu2", db_path=tmp_db)), 0)
            finally:
                auth_mod.DB_FILE = orig_auth_db

    def test_11_ocr_metadata_and_bulk_automation(self):
        """Test OCR VTU metadata/offer letter extractors and automated bulk Excel importers."""
        vtu_sample = "VISVESVARAYA TECHNOLOGICAL UNIVERSITY\nUSN: 1TJ24DS028\nSEMESTER: 4\nSGPA: 8.95\n"
        meta = extract_vtu_metadata_from_text(vtu_sample)
        self.assertEqual(meta["usn"], "1TJ24DS028")
        self.assertEqual(meta["semester_number"], 4)
        self.assertAlmostEqual(meta["official_sgpa"], 8.95, places=2)

        offer_sample = (
            "Company: TCS Digital\nUSN: 1TJ24DS028\nDesignation: Systems Engineer\n"
            "Compensation: 7.25 LPA\nLocation: Bengaluru\nDrive Date: 2026-09-15\n"
        )
        offer = extract_offer_letter_details_from_text(offer_sample)
        self.assertEqual(offer["usn"], "1TJ24DS028")
        self.assertIn("TCS", offer["company_name"])
        self.assertAlmostEqual(offer["package_lpa"], 7.25, places=2)

        with tempfile.TemporaryDirectory() as tmpd:
            tmp_db = os.path.join(tmpd, "test_bulk.db")
            run_migrations(tmp_db)
            orig_auth_db = auth_mod.DB_FILE
            auth_mod.DB_FILE = tmp_db
            try:
                conn = get_db_connection(tmp_db)
                conn.execute("INSERT INTO classes (id, class_name, semester, section, created_at) VALUES ('cls1', '5th Sem', 5, 'A', '2026-01-01')")
                conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, created_at) VALUES ('stu1', 'cls1', 'Vivek S', '1TJ24DS028', 2024, 5, '2026-01-01')")
                conn.commit()
                conn.close()

                seeded = ae.seed_vtu_master_subjects("cls1", 5, db_path=tmp_db)
                self.assertGreater(seeded, 0)

                df_marks = pd.DataFrame([
                    {"USN": "1TJ24DS028", "Semester": 5, "Subject Code": "BCS501", "Subject Name": "SEPM", "Internal": 44, "External": 41, "Credits": 4.0, "Result": "P", "SGPA": 9.0}
                ])
                vm = ae.validate_marks_excel(df_marks, mode="final_vtu", db_path=tmp_db)
                self.assertTrue(vm["is_valid"])
                res_m = ae.execute_marks_bulk_import(vm["valid_rows"], mode="final_vtu", db_path=tmp_db)
                self.assertEqual(res_m["semesters_saved"], 1)

                df_att = pd.DataFrame([
                    {"USN": "1TJ24DS028", "Subject": "SEPM", "Date": "2026-09-10", "Hour": 1, "Status": "Present"},
                    {"USN": "1TJ24DS028", "Subject": "SEPM", "Date": "2026-09-10", "Hour": 2, "Status": "Absent"},
                ])
                va = ae.validate_attendance_excel(df_att, db_path=tmp_db)
                self.assertTrue(va["is_valid"])
                cnt_a = ae.execute_attendance_bulk_import(va["valid_rows"], mode=va["mode"], db_path=tmp_db)
                self.assertEqual(cnt_a, 2)

                df_pl = pd.DataFrame([
                    {"USN": "1TJ24DS028", "Company": "Accenture", "Job Role": "Associate SWE", "Package LPA": 6.5, "Status": "Placed"}
                ])
                vp = ae.validate_placement_excel(df_pl, db_path=tmp_db)
                self.assertTrue(vp["is_valid"])
                cnt_p, _ = ae.execute_placement_bulk_import(vp["valid_rows"], db_path=tmp_db)
                self.assertEqual(cnt_p, 1)
            finally:
                auth_mod.DB_FILE = orig_auth_db

    def test_12_phase2_future_improvements(self):
        """Test Phase 2 VTU Exam eligibility, ZIP marksheet extraction, placement drives, revisions, promotion, and mentoring."""
        import academic_engine as ae
        import auth as auth_mod

        # 1. Attendance Recovery Calculator
        self.assertEqual(ae.calculate_classes_needed(6, 10, 75.0), 6)
        self.assertEqual(ae.calculate_classes_needed(8, 10, 75.0), 0)
        self.assertEqual(ae.calculate_classes_needed(0, 0, 75.0), 0)

        # 2. ZIP Archive Batch Marksheet Expansion
        with tempfile.TemporaryDirectory() as zip_tmp:
            dummy_pdf = os.path.join(zip_tmp, "sample_sem3.pdf")
            dummy_img = os.path.join(zip_tmp, "sample_sem4.png")
            with open(dummy_pdf, "w") as f:
                f.write("DUMMY_PDF_CONTENT")
            with open(dummy_img, "w") as f:
                f.write("DUMMY_IMG_CONTENT")

            archive_path = os.path.join(zip_tmp, "batch_marks.zip")
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.write(dummy_pdf, arcname="sample_sem3.pdf")
                zf.write(dummy_img, arcname="sample_sem4.png")

            extract_target = os.path.join(zip_tmp, "extracted")
            expanded = ae.expand_marksheet_uploads_with_zip([("batch_marks.zip", archive_path)], extract_dir=extract_target)
            self.assertEqual(len(expanded), 2)
            names = [entry[0] for entry in expanded]
            self.assertIn("sample_sem3.pdf", names)
            self.assertIn("sample_sem4.png", names)

        # 3. Database integration for Phase 2 workflows
        tmpd = tempfile.mkdtemp()
        tmp_db = os.path.join(tmpd, "test_phase2.db")
        run_migrations(tmp_db)
        orig_auth_db = auth_mod.DB_FILE
        auth_mod.DB_FILE = tmp_db
        try:
            conn = get_db_connection(tmp_db)
            conn.execute("INSERT INTO classes (id, class_name, semester, section, created_at) VALUES ('cls_p2', '5th Sem CSE', 5, 'A', '2026-01-01')")
            conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, created_at) VALUES ('stu_p2', 'cls_p2', 'Arjun K', '1TJ24CS001', 2024, 5, '2026-01-01')")
            conn.commit()
            conn.close()

            # Marksheet Revision / Revaluation Delta Logging
            res1 = ae.save_student_semester_result(
                student_id="stu_p2",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Management", "ia_marks": 40, "ea_marks": 35, "total_marks": 75, "credits": 4.0, "result": "P", "grade": "B"}
                ],
                official_sgpa=7.50,
                verification_status="Verified",
                db_path=tmp_db,
            )
            self.assertIn("semester_id", res1)

            # Re-upload revised marksheet (Revaluation: SGPA 7.50 -> 8.50)
            res2 = ae.save_student_semester_result(
                student_id="stu_p2",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Management", "ia_marks": 48, "ea_marks": 45, "total_marks": 93, "credits": 4.0, "result": "P", "grade": "S"}
                ],
                official_sgpa=8.50,
                verification_status="Verified",
                db_path=tmp_db,
            )
            self.assertIn("semester_id", res2)

            revisions = ae.get_marksheet_revisions(student_id="stu_p2", db_path=tmp_db)
            self.assertEqual(len(revisions), 1)
            self.assertAlmostEqual(revisions.iloc[0]["old_sgpa"], 7.50, places=2)
            self.assertAlmostEqual(revisions.iloc[0]["new_sgpa"], 8.50, places=2)
            self.assertAlmostEqual(revisions.iloc[0]["new_sgpa"] - revisions.iloc[0]["old_sgpa"], 1.00, places=2)

            # VTU Exam Eligibility & Attendance Condonation Band Testing
            # Add 10 attendance records: 7 present, 3 absent = 70.0% (in 65-75% condonation band)
            conn = get_db_connection(tmp_db)
            for h in range(1, 8):
                conn.execute("INSERT INTO attendance (id, class_id, student_id, attendance_date, subject_name, hour_number, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (f"att_{h}", "cls_p2", "stu_p2", "2026-09-01", "Management", h, "Present", "2026-09-01"))
            for h in range(8, 11):
                conn.execute("INSERT INTO attendance (id, class_id, student_id, attendance_date, subject_name, hour_number, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (f"att_{h}", "cls_p2", "stu_p2", "2026-09-02", "Management", h, "Absent", "2026-09-02"))
            # Add current semester record with low CIE (< 40% of 50 = 20)
            conn.execute("""
                INSERT INTO current_semester_records (id, student_id, semester_number, subject_code, subject_name, credits, ia1, ia2, internal_performance, updated_at)
                VALUES ('csr_cie_fail', 'stu_p2', 5, '21CS52', 'Computer Networks', 4.0, 10.0, 15.0, 12.5, '2026-09-01')
            """)
            conn.commit()
            conn.close()

            ana_df = ae.get_comprehensive_student_analytics(db_path=tmp_db)
            self.assertEqual(len(ana_df), 1)
            stu_row = ana_df.iloc[0]
            self.assertEqual(stu_row["att_condonation_band"], "Condonable (65-75%)")
            self.assertGreaterEqual(stu_row["classes_to_recover_75"], 2)
            self.assertFalse(stu_row["exam_eligible"])
            self.assertIn("CIE Shortage", stu_row["exam_status"])

            # Student Off-Campus Offer Submission -> Verification Workflow
            pid = ae.upsert_placement_record(
                student_id="stu_p2",
                company_name="Google",
                job_role="Software Engineer",
                package_lpa=24.5,
                offer_status="Placed",
                entered_role="student",
                verification_status="Pending Verification",
                source="student",
                db_path=tmp_db,
            )
            self.assertTrue(bool(pid))

            ana_pending = ae.get_comprehensive_student_analytics(db_path=tmp_db).iloc[0]
            self.assertEqual(ana_pending["pending_offers_count"], 1)
            self.assertFalse(ana_pending["is_placed"])  # Pending offers don't mark placed yet

            reviewed = ae.review_placement_offer(pid, decision="Verified", reviewer_id="prof_placement", remarks="Official Google offer verified", db_path=tmp_db)
            self.assertTrue(reviewed)

            ana_verified = ae.get_comprehensive_student_analytics(db_path=tmp_db).iloc[0]
            self.assertTrue(ana_verified["is_placed"])
            self.assertEqual(ana_verified["offers_count"], 1)
            self.assertAlmostEqual(ana_verified["highest_package_lpa"], 24.5, places=1)

            # Placement Drives & Shortlist Matching
            drive_id = ae.save_placement_drive(
                company_name="Microsoft",
                job_role="Software Engineer",
                package_lpa=20.0,
                min_cgpa=6.0,
                max_active_backlogs=0,
                min_attendance_pct=65.0,
                db_path=tmp_db,
            )
            self.assertTrue(bool(drive_id))
            drives_df = ae.get_placement_drives(db_path=tmp_db)
            self.assertEqual(len(drives_df), 1)
            self.assertEqual(drives_df.iloc[0]["company_name"], "Microsoft")

            match_df = ae.match_students_for_drive(min_cgpa=6.0, max_active_backlogs=0, min_attendance_pct=65.0, require_next_sem_eligible=False, db_path=tmp_db)
            self.assertGreaterEqual(len(match_df), 1)

            notif_count = ae.notify_shortlisted_students(["stu_p2"], "Microsoft", "Software Engineer", "2026-11-20", db_path=tmp_db)
            self.assertEqual(notif_count, 1)

            # 1-Click End-of-Semester Batch Promotion
            prom_res = ae.execute_semester_batch_promotion(["stu_p2"], actor_id="admin_user", db_path=tmp_db)
            self.assertEqual(prom_res["promoted"], 1)
            
            conn = get_db_connection(tmp_db)
            cur_sem = conn.execute("SELECT current_semester FROM students WHERE id = 'stu_p2'").fetchone()[0]
            conn.close()
            self.assertEqual(cur_sem, 6)

            # Mentoring / Counseling Log
            log_id = ae.add_mentoring_log(
                student_id="stu_p2",
                faculty_id="fac_101",
                faculty_name="Dr. Raghavan",
                category="Attendance Recovery",
                notes="Met student to discuss recovery plan for 70% attendance.",
                action_taken="Parent contacted and study plan agreed",
                parent_notified=True,
                notify_student=True,
                db_path=tmp_db,
            )
            self.assertTrue(bool(log_id))
            m_logs = ae.get_mentoring_logs(student_id="stu_p2", db_path=tmp_db)
            self.assertEqual(len(m_logs), 1)
            self.assertEqual(m_logs.iloc[0]["category"], "Attendance Recovery")
            self.assertEqual(m_logs.iloc[0]["faculty_name"], "Dr. Raghavan")
        finally:
            auth_mod.DB_FILE = orig_auth_db
            shutil.rmtree(tmpd, ignore_errors=True)

    def test_13_phase3_ats_parent_notices_and_simulator(self):
        tmpd = tempfile.mkdtemp()
        tmp_db = os.path.join(tmpd, "test_phase3.db")
        run_migrations(tmp_db)
        orig_auth_db = auth_mod.DB_FILE
        auth_mod.DB_FILE = tmp_db
        try:
            conn = get_db_connection(tmp_db)
            conn.execute("INSERT INTO classes (id, class_name, semester, section, created_at) VALUES ('cls_p3', '6th Sem CSE', 6, 'A', '2026-01-01')")
            conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, created_at) VALUES ('stu_p3_1', 'cls_p3', 'Vikram Rao', '1TJ23CS001', 2023, 6, '2026-01-01')")
            conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, created_at) VALUES ('stu_p3_2', 'cls_p3', 'Sneha M', '1TJ23CS002', 2023, 6, '2026-01-01')")
            conn.commit()
            conn.close()

            # 1. Bulk Rapid Class Attendance Logging
            fast_res = ae.bulk_log_fast_attendance(
                class_id="cls_p3",
                subject_name="Cloud Computing",
                attendance_date="2026-11-15",
                hour_number=1,
                present_student_ids=["stu_p3_1"],
                absent_student_ids=["stu_p3_2"],
                db_path=tmp_db,
            )
            self.assertEqual(fast_res, 2)

            # Give Vikram high attendance (1 hour present = 100%)
            fast_res2 = ae.bulk_log_fast_attendance(
                class_id="cls_p3",
                subject_name="Cloud Computing",
                attendance_date="2026-11-16",
                hour_number=2,
                present_student_ids=["stu_p3_1"],
                absent_student_ids=[],
                db_path=tmp_db,
            )
            self.assertEqual(fast_res2, 1)

            # Give Vikram good semester grades (SGPA 8.8)
            ae.save_student_semester_result(
                student_id="stu_p3_1",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Cloud Computing", "ia_marks": 45, "ea_marks": 48, "total_marks": 93, "credits": 4.0, "result": "P", "grade": "S"},
                    {"subject_code": "21CS52", "subject_name": "Operating Systems", "ia_marks": 42, "ea_marks": 44, "total_marks": 86, "credits": 4.0, "result": "P", "grade": "A"}
                ],
                official_sgpa=8.80,
                verification_status="Verified",
                db_path=tmp_db,
            )

            # Give Sneha lower grades (SGPA 5.5 with backlog)
            ae.save_student_semester_result(
                student_id="stu_p3_2",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Cloud Computing", "ia_marks": 20, "ea_marks": 15, "total_marks": 35, "credits": 4.0, "result": "F", "grade": "F"}
                ],
                official_sgpa=5.50,
                verification_status="Verified",
                db_path=tmp_db,
            )

            # 2. Placement Drive Creation & ATS Workflow
            drive_id = ae.save_placement_drive(
                company_name="Amazon AWS",
                job_role="Cloud DevOps Engineer",
                package_lpa=28.0,
                min_cgpa=8.0,
                max_active_backlogs=0,
                min_attendance_pct=75.0,
                require_next_sem_eligible=False,
                drive_date="2026-12-01",
                db_path=tmp_db,
            )
            self.assertTrue(bool(drive_id))

            # Student 2 (Sneha) tries to apply -> should fail eligibility check
            res_sneha = ae.apply_for_placement_drive(
                drive_id=drive_id,
                student_id="stu_p3_2",
                resume_headline="Aspiring Cloud Engineer",
                db_path=tmp_db,
            )
            self.assertFalse(res_sneha["success"])
            self.assertIn("Eligibility", res_sneha["message"])

            # Student 1 (Vikram) applies -> should succeed
            res_vikram = ae.apply_for_placement_drive(
                drive_id=drive_id,
                student_id="stu_p3_1",
                resume_headline="AWS Certified Solutions Architect Candidate",
                db_path=tmp_db,
            )
            self.assertTrue(res_vikram["success"])
            app_id = res_vikram["application_id"]
            self.assertIsNotNone(app_id)

            # Duplicate application should be prevented
            dup_res = ae.apply_for_placement_drive(
                drive_id=drive_id,
                student_id="stu_p3_1",
                db_path=tmp_db,
            )
            self.assertFalse(dup_res["success"])
            self.assertIn("Already applied", dup_res["message"])

            # Retrieve drive applications
            apps_df = ae.get_drive_applications(drive_id=drive_id, db_path=tmp_db)
            self.assertEqual(len(apps_df), 1)
            self.assertEqual(apps_df.iloc[0]["student_id"], "stu_p3_1")
            self.assertEqual(apps_df.iloc[0]["application_stage"], "Applied")

            # Update ATS stage to Shortlisted -> Interview -> Selected
            up_res1 = ae.update_drive_application_stage(app_id, "Interview", feedback="Passed online test", reviewer_id="placement_officer", db_path=tmp_db)
            self.assertTrue(up_res1)

            # When Selected: should auto-generate verified offer in placements table
            up_res2 = ae.update_drive_application_stage(
                app_id,
                "Selected",
                feedback="Top candidate in final round",
                package_lpa=28.0,
                reviewer_id="placement_officer",
                db_path=tmp_db,
            )
            self.assertTrue(up_res2)

            # Check that offer was auto-created in placements
            offers_df = ae.get_comprehensive_student_analytics(db_path=tmp_db)
            vikram_stats = offers_df[offers_df["student_id"] == "stu_p3_1"].iloc[0]
            self.assertTrue(vikram_stats["is_placed"])
            self.assertAlmostEqual(vikram_stats["highest_package_lpa"], 28.0, places=1)
            self.assertEqual(vikram_stats["salary_tier"], "Super Dream")

            # 3. Parent Communications & Official PDF Generation
            comm_id = ae.dispatch_parent_communication(
                student_id="stu_p3_2",
                faculty_id="fac_101",
                faculty_name="Dr. Raghavan",
                comm_type="Attendance Warning",
                channel="Official Notice",
                subject="Critical Attendance Shortage Notice",
                message_body="Attendance is critically low at 0%. Immediate conference requested.",
                db_path=tmp_db,
            )
            self.assertTrue(bool(comm_id))

            parent_comms = ae.get_parent_communications(student_id="stu_p3_2", db_path=tmp_db)
            self.assertEqual(len(parent_comms), 1)
            self.assertEqual(parent_comms.iloc[0]["communication_type"], "Attendance Warning")

            # Generate PyMuPDF PDF letterhead notice
            pdf_bytes = ae.generate_official_parent_notice_pdf(
                student_id="stu_p3_2",
                comm_type="Attendance Warning",
                custom_remarks="Mandatory parent conference on Monday.",
                db_path=tmp_db,
            )
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
            self.assertGreater(len(pdf_bytes), 1000)

            # 4. What-If CGPA & Career Simulator
            sim_res = ae.simulate_what_if_cgpa(
                student_id="stu_p3_1",
                projected_sgpas={6: 9.0, 7: 9.2, 8: 9.5},
                db_path=tmp_db,
            )
            self.assertAlmostEqual(sim_res["current_cgpa"], 8.80, places=2)
            self.assertGreater(sim_res["projected_cgpa"], 8.80)
            self.assertIn("Super Dream", sim_res["projected_tier"])
            self.assertIn("semester_map", sim_res)
            self.assertIn("unlocked_drives", sim_res)
        finally:
            auth_mod.DB_FILE = orig_auth_db
            shutil.rmtree(tmpd, ignore_errors=True)

    def test_14_passed_border_pass_ia_assignments_and_recalc(self):
        tmpd = tempfile.mkdtemp()
        tmp_db = os.path.join(tmpd, "test_phase4.db")
        run_migrations(tmp_db)
        orig_auth_db = auth_mod.DB_FILE
        auth_mod.DB_FILE = tmp_db
        try:
            conn = get_db_connection(tmp_db)
            conn.execute("INSERT INTO classes (id, class_name, semester, section, created_at) VALUES ('cls_p4', '5th Sem CSE', 5, 'A', '2026-01-01')")
            # stu1: High credits distinction pass
            conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, working_days, absent_days, created_at) VALUES ('stu_h1', 'cls_p4', 'High Achiever', '1TJ24CS001', 2024, 5, 90, 5, '2026-01-01')")
            # stu2: Borderline pass (external mark 19 in subject)
            conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, working_days, absent_days, created_at) VALUES ('stu_b1', 'cls_p4', 'Border Passer', '1TJ24CS002', 2024, 5, 90, 10, '2026-01-01')")
            # stu3: Not passed (1 active backlog)
            conn.execute("INSERT INTO students (id, class_id, name, usn, admission_year, current_semester, working_days, absent_days, created_at) VALUES ('stu_f1', 'cls_p4', 'Failed Student', '1TJ24CS003', 2024, 5, 90, 20, '2026-01-01')")
            conn.commit()
            conn.close()

            # 1. IA Tests & Assignment Marks calculation
            ia_res = ae.save_student_ia_subject_mark(
                student_id="stu_h1",
                semester_number=5,
                subject_code="21CS51",
                subject_name="Management",
                credits=4.0,
                ia1=25.0,
                ia2=27.0,
                ia3=29.0,
                assignment_1=8.5,
                assignment_2=8.5,
                db_path=tmp_db,
            )
            # Best 2 of (25, 27, 29) is (27+29)/2 = 28.0 + (8.5+8.5) = 17.0 -> 45.0
            self.assertEqual(ia_res["test_component"], 28.0)
            self.assertEqual(ia_res["assignment_component"], 17.0)
            self.assertEqual(ia_res["cie_total"], 45.0)
            self.assertTrue(ia_res["cie_eligible"])

            # 2. Batch class-wide IA & Assignment Entry
            batch_ia_cnt = ae.bulk_save_class_ia_and_assignments(
                class_id="cls_p4",
                semester_number=5,
                subject_code="21CS52",
                subject_name="Computer Networks",
                credits=4.0,
                records=[
                    {"student_id": "stu_h1", "ia1": 30.0, "ia2": 30.0, "ia3": 0.0, "assignment_1": 10.0, "assignment_2": 10.0},
                    {"student_id": "stu_b1", "ia1": 15.0, "ia2": 15.0, "ia3": 0.0, "assignment_1": 5.0, "assignment_2": 5.0},
                ],
                db_path=tmp_db,
            )
            self.assertEqual(batch_ia_cnt, 2)

            # 3. Save VTU Results to test segregation:
            # stu_h1: High credits pass (SGPA 8.5, 20 credits, 0 backlogs)
            ae.save_student_semester_result(
                student_id="stu_h1",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Management", "ia_marks": 45, "ea_marks": 45, "total_marks": 90, "credits": 4.0, "result": "P", "grade": "S"},
                    {"subject_code": "21CS52", "subject_name": "CN", "ia_marks": 40, "ea_marks": 40, "total_marks": 80, "credits": 4.0, "result": "P", "grade": "A"},
                    {"subject_code": "21CS53", "subject_name": "DBMS", "ia_marks": 40, "ea_marks": 40, "total_marks": 80, "credits": 4.0, "result": "P", "grade": "A"},
                    {"subject_code": "21CS54", "subject_name": "ATC", "ia_marks": 40, "ea_marks": 40, "total_marks": 80, "credits": 4.0, "result": "P", "grade": "A"},
                    {"subject_code": "21CS55", "subject_name": "EVS", "ia_marks": 40, "ea_marks": 40, "total_marks": 80, "credits": 4.0, "result": "P", "grade": "A"},
                ],
                official_sgpa=8.50,
                verification_status="Verified",
                db_path=tmp_db,
            )

            # stu_b1: Borderline pass (external mark 19 in Management -> near border 18)
            ae.save_student_semester_result(
                student_id="stu_b1",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Management", "ia_marks": 22, "ea_marks": 19, "total_marks": 41, "credits": 4.0, "result": "P", "grade": "E"},
                    {"subject_code": "21CS52", "subject_name": "CN", "ia_marks": 25, "ea_marks": 30, "total_marks": 55, "credits": 4.0, "result": "P", "grade": "C"},
                ],
                official_sgpa=5.50,
                verification_status="Verified",
                db_path=tmp_db,
            )

            # stu_f1: Not passed (external mark 12 -> failed with active backlog)
            ae.save_student_semester_result(
                student_id="stu_f1",
                sem_num=5,
                subjects=[
                    {"subject_code": "21CS51", "subject_name": "Management", "ia_marks": 20, "ea_marks": 12, "total_marks": 32, "credits": 4.0, "result": "F", "grade": "F"},
                ],
                official_sgpa=4.00,
                verification_status="Verified",
                db_path=tmp_db,
            )

            # 4. Check analytics & working_days presence
            ana_df = ae.get_comprehensive_student_analytics(db_path=tmp_db)
            self.assertEqual(len(ana_df), 3)
            self.assertIn("working_days", ana_df.columns)
            self.assertIn("absent_days", ana_df.columns)
            self.assertIn("is_passed", ana_df.columns)
            self.assertIn("is_border_pass", ana_df.columns)
            self.assertIn("is_high_credits_pass", ana_df.columns)

            h1_row = ana_df[ana_df["student_id"] == "stu_h1"].iloc[0]
            b1_row = ana_df[ana_df["student_id"] == "stu_b1"].iloc[0]
            f1_row = ana_df[ana_df["student_id"] == "stu_f1"].iloc[0]

            self.assertTrue(h1_row["is_passed"])
            self.assertTrue(h1_row["is_high_credits_pass"])
            self.assertFalse(h1_row["is_border_pass"])

            self.assertTrue(b1_row["is_passed"])
            self.assertTrue(b1_row["is_border_pass"])
            self.assertFalse(b1_row["is_high_credits_pass"])

            self.assertFalse(f1_row["is_passed"])

            # 5. Check Segregation dictionary
            seg = ae.segregate_students(ana_df)
            self.assertEqual(len(seg["passed_df"]), 2)
            self.assertEqual(len(seg["pass_high_credits_df"]), 1)
            self.assertEqual(len(seg["border_pass_df"]), 1)
            self.assertEqual(len(seg["not_passed_df"]), 1)
            self.assertEqual(seg["summary"]["passed_count"], 2)
            self.assertEqual(seg["summary"]["border_pass_count"], 1)
            self.assertEqual(seg["summary"]["high_credits_pass_count"], 1)
            self.assertEqual(seg["summary"]["not_passed_count"], 1)

            # 6. Recalculate All Student Records
            recalc_res = ae.recalculate_all_student_records(db_path=tmp_db)
            self.assertEqual(recalc_res["total_students"], 3)
            self.assertEqual(recalc_res["passed_count"], 2)
            self.assertEqual(recalc_res["border_pass_count"], 1)
            self.assertEqual(recalc_res["high_credits_count"], 1)
            self.assertEqual(recalc_res["not_passed_count"], 1)
        finally:
            auth_mod.DB_FILE = orig_auth_db
            shutil.rmtree(tmpd, ignore_errors=True)

    def test_15_vtu_hall_tickets_and_remedial_plans(self):
        """Test generation of official VTU Hall Ticket PDF, Remedial Mentoring Plan PDF, and batch class ZIP."""
        stu = self._first_student()
        self.assertIsNotNone(stu, "Should have at least 1 student in test DB")
        student_id = stu["id"]

        # 1. Test VTU Hall Ticket PDF
        ht_pdf = generate_vtu_hall_ticket_pdf(student_id, db_path=DB_FILE)
        self.assertIsInstance(ht_pdf, bytes)
        self.assertGreater(len(ht_pdf), 1000)
        self.assertTrue(ht_pdf.startswith(b"%PDF"))

        # 2. Test Remedial Action Plan PDF
        rem_pdf = generate_remedial_mentoring_plan_pdf(student_id, db_path=DB_FILE)
        self.assertIsInstance(rem_pdf, bytes)
        self.assertGreater(len(rem_pdf), 1000)
        self.assertTrue(rem_pdf.startswith(b"%PDF"))

        # 3. Test Batch Class Hall Tickets ZIP
        target_class_id = stu.get("class_id") or "all"
        zip_bytes = generate_class_hall_tickets_zip(target_class_id, db_path=DB_FILE)
        self.assertIsInstance(zip_bytes, bytes)
        self.assertGreater(len(zip_bytes), 2000)
        self.assertTrue(zip_bytes.startswith(b"PK\x03\x04"))

        # Verify ZIP archive contents
        import io
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            self.assertGreaterEqual(len(namelist), 1)
            pdf_entries = [name for name in namelist if name.endswith(".pdf")]
            self.assertGreaterEqual(len(pdf_entries), 1)

        # 4. Test Executive Class Ledger PDF
        ledger_pdf = generate_executive_class_ledger_pdf(target_class_id, db_path=DB_FILE)
        self.assertIsInstance(ledger_pdf, bytes)
        self.assertGreater(len(ledger_pdf), 2000)
        self.assertTrue(ledger_pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
