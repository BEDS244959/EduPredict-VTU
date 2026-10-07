"""
Database migration module for EduPredict AI.
Ensures all required tables and columns exist in edupredict.db using safe, non-destructive migrations.
Preserves existing classes, students, semesters, subjects, and attendance records.
"""

import sqlite3
import os
import uuid
from datetime import datetime


def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _add_column_if_not_exists(cur: sqlite3.Cursor, table_name: str, col_name: str, col_def: str):
    """Safely add a column to an existing table if it does not already exist."""
    cur.execute(f"PRAGMA table_info(`{table_name}`)")
    columns = [row[1] for row in cur.fetchall()]
    if col_name not in columns:
        cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_def}")


def run_migrations(db_path: str):
    """
    Executes safe non-destructive schema migrations.
    Uses CREATE TABLE IF NOT EXISTS and safe column additions.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Classes Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            class_teacher TEXT,
            created_at TEXT
        )
    """)
    _add_column_if_not_exists(cur, "classes", "section", "TEXT")
    _add_column_if_not_exists(cur, "classes", "academic_year", "TEXT")
    _add_column_if_not_exists(cur, "classes", "semester", "INTEGER DEFAULT 1")
    _add_column_if_not_exists(cur, "classes", "is_locked", "INTEGER DEFAULT 0")

    # 2. Students Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            class_id TEXT NOT NULL,
            name TEXT NOT NULL,
            usn TEXT UNIQUE,
            admission_year INTEGER,
            current_semester INTEGER,
            photo_path TEXT,
            absent_days INTEGER DEFAULT 0,
            working_days INTEGER DEFAULT 90,
            FOREIGN KEY(class_id) REFERENCES classes(id)
        )
    """)
    _add_column_if_not_exists(cur, "students", "email", "TEXT")
    _add_column_if_not_exists(cur, "students", "phone", "TEXT")
    _add_column_if_not_exists(cur, "students", "created_at", "TEXT")
    _add_column_if_not_exists(cur, "students", "is_lateral_entry", "INTEGER DEFAULT 0")

    # 3. Semesters Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS semesters (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            semester_number INTEGER NOT NULL,
            semester_name TEXT,
            sgpa REAL DEFAULT 0,
            marksheet_path TEXT,
            is_current INTEGER DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    _add_column_if_not_exists(cur, "semesters", "credits_total", "REAL DEFAULT 0")
    _add_column_if_not_exists(cur, "semesters", "verification_status", "TEXT DEFAULT 'Verified'")
    _add_column_if_not_exists(cur, "semesters", "staff_remarks", "TEXT")
    _add_column_if_not_exists(cur, "semesters", "is_locked", "INTEGER DEFAULT 0")

    # 4. Subjects (historical/semester-wise)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id TEXT PRIMARY KEY,
            semester_id TEXT NOT NULL,
            subject_code TEXT,
            subject_name TEXT,
            internal_marks REAL,
            external_marks REAL,
            total_marks REAL,
            credits REAL DEFAULT 0,
            result TEXT,
            FOREIGN KEY(semester_id) REFERENCES semesters(id)
        )
    """)
    _add_column_if_not_exists(cur, "subjects", "grade", "TEXT")
    _add_column_if_not_exists(cur, "subjects", "grade_point", "INTEGER DEFAULT 0")

    # 5. Master / Dynamic Subjects (Unlimited subjects configured per class/semester)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS master_subjects (
            id TEXT PRIMARY KEY,
            class_id TEXT,
            semester_number INTEGER NOT NULL,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            credits REAL DEFAULT 3.0,
            internal_max REAL DEFAULT 50.0,
            external_max REAL DEFAULT 50.0,
            total_max REAL DEFAULT 100.0,
            created_at TEXT,
            FOREIGN KEY(class_id) REFERENCES classes(id),
            UNIQUE(class_id, semester_number, subject_code)
        )
    """)

    # 6. Backlog Marksheets & Backlogs Tracking Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS backlog_marksheets (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            semester_number INTEGER,
            subject_name TEXT,
            file_path TEXT,
            uploaded_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS backlogs (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            semester_number INTEGER NOT NULL,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            previous_result TEXT DEFAULT 'F',
            rewritten_exam TEXT,
            updated_result TEXT,
            credits REAL DEFAULT 3.0,
            status TEXT DEFAULT 'Active',
            file_path TEXT,
            updated_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 7. Attendance
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id TEXT PRIMARY KEY,
            class_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            hour_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT,
            UNIQUE(student_id, attendance_date, subject_name, hour_number)
        )
    """)

    # 8. Authentication and User Management Tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staff_users (
            id TEXT PRIMARY KEY,
            tjit_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'staff',
            active INTEGER DEFAULT 1,
            created_at TEXT
        )
    """)
    _add_column_if_not_exists(cur, "staff_users", "email", "TEXT")
    _add_column_if_not_exists(cur, "staff_users", "phone", "TEXT")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_users (
            id TEXT PRIMARY KEY,
            student_id TEXT UNIQUE NOT NULL,
            usn TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS login_activity (
            id TEXT PRIMARY KEY,
            user_type TEXT NOT NULL,
            user_id TEXT,
            login_time TEXT NOT NULL,
            success INTEGER NOT NULL,
            ip_info TEXT
        )
    """)

    # 9. Current Semester Records
    cur.execute("""
        CREATE TABLE IF NOT EXISTS current_semester_records (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            semester_number INTEGER NOT NULL,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            credits REAL NOT NULL,
            ia1 REAL DEFAULT 0,
            ia2 REAL DEFAULT 0,
            internal_performance REAL DEFAULT 0,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id),
            UNIQUE(student_id, semester_number, subject_code)
        )
    """)
    _add_column_if_not_exists(cur, "current_semester_records", "final_marks", "REAL")
    _add_column_if_not_exists(cur, "current_semester_records", "predicted_final", "REAL")
    _add_column_if_not_exists(cur, "current_semester_records", "grade", "TEXT")
    _add_column_if_not_exists(cur, "current_semester_records", "result", "TEXT")

    # 10. Academic Predictions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS academic_predictions (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            semester_id TEXT,
            subject_id TEXT,
            prediction_type TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            prediction_date TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction_method TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 11. Certificates
    cur.execute("""
        CREATE TABLE IF NOT EXISTS certificates (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            certificate_name TEXT NOT NULL,
            category TEXT NOT NULL,
            organization TEXT,
            issue_date TEXT,
            description TEXT,
            file_path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            verification_status TEXT DEFAULT 'Pending',
            staff_comment TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 12. Student Skills
    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_skills (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            category TEXT,
            proficiency TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id),
            UNIQUE(student_id, skill_name)
        )
    """)

    # 13. LinkedIn Profiles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS linkedin_profiles (
            id TEXT PRIMARY KEY,
            student_id TEXT UNIQUE NOT NULL,
            linkedin_url TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 14. Extracurricular Activities
    cur.execute("""
        CREATE TABLE IF NOT EXISTS extracurricular_activities (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            activity_name TEXT NOT NULL,
            category TEXT NOT NULL,
            organization_event TEXT,
            date TEXT,
            achievement TEXT,
            description TEXT,
            certificate_path TEXT,
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 15. Fee Receipts
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fee_receipts (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            academic_year TEXT NOT NULL,
            semester INTEGER NOT NULL,
            receipt_number TEXT NOT NULL,
            amount REAL NOT NULL,
            receipt_path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            verification_status TEXT DEFAULT 'Pending',
            staff_comment TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 16. Non-Due Form Documents
    cur.execute("""
        CREATE TABLE IF NOT EXISTS non_due_documents (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            academic_year TEXT NOT NULL,
            semester INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            verification_status TEXT DEFAULT 'Pending',
            staff_comment TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # 17. Student Letters / Applications
    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_letters (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            recipient TEXT NOT NULL,
            subject TEXT NOT NULL,
            letter_text TEXT NOT NULL,
            attachment_path TEXT,
            submission_date TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            staff_comment TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    _add_column_if_not_exists(cur, "student_letters", "application_type", "TEXT DEFAULT 'General Application'")

    # 18. Notifications System
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            recipient_type TEXT NOT NULL,
            recipient_id TEXT,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            is_read INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # 18b. Notification Reads per user
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notification_reads (
            id TEXT PRIMARY KEY,
            notification_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            read_at TEXT NOT NULL,
            UNIQUE(notification_id, user_id),
            FOREIGN KEY(notification_id) REFERENCES notifications(id)
        )
    """)

    # 19. Audit Logs System
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            user_type TEXT NOT NULL,
            user_id TEXT,
            action TEXT NOT NULL,
            details TEXT,
            target_id TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    # 20. Essential Indexes for Performance & Search
    cur.execute("CREATE INDEX IF NOT EXISTS idx_student_users_usn ON student_users(usn);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_staff_users_tjit ON staff_users(tjit_number);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_students_class ON students(class_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_students_usn ON students(usn);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_semesters_student ON semesters(student_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_subjects_semester ON subjects(semester_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_current_sem_rec ON current_semester_records(student_id, semester_number);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_master_subjects ON master_subjects(class_id, semester_number);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_recip ON notifications(recipient_type, recipient_id, is_read);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_time ON audit_logs(timestamp);")

    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edupredict.db")
    run_migrations(db_file)
    print("Database migrations ran successfully.")
