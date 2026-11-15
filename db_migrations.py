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
    _add_column_if_not_exists(cur, "classes", "staff_tjit", "TEXT")

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

    # 21. Placements Tracking & Career Records
    cur.execute("""
    CREATE TABLE IF NOT EXISTS placements (
        id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        company_name TEXT NOT NULL,
        job_role TEXT,
        package_lpa REAL DEFAULT 0.0,
        offer_status TEXT DEFAULT 'Placed',
        drive_date TEXT,
        location TEXT,
        offer_letter_path TEXT,
        remarks TEXT,
        entered_by TEXT,
        entered_role TEXT DEFAULT 'placement',
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_placements_student ON placements(student_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_placements_status ON placements(offer_status);")
    _add_column_if_not_exists(cur, "placements", "verification_status", "TEXT DEFAULT 'Verified'")
    _add_column_if_not_exists(cur, "placements", "source", "TEXT DEFAULT 'faculty'")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_placements_verif ON placements(verification_status);")

    # 22. Configurable Academic & Eligibility Rules
    cur.execute("""
    CREATE TABLE IF NOT EXISTS academic_rules (
        rule_key TEXT PRIMARY KEY,
        rule_value TEXT NOT NULL,
        description TEXT,
        updated_by TEXT,
        updated_at TEXT
    );
    """)
    default_rules = [
        ("low_attendance_threshold", "75.0", "Minimum attendance percentage required before flagging Low Attendance"),
        ("condonable_min_attendance_pct", "65.0", "Minimum attendance percentage eligible for medical/condonation"),
        ("exam_min_attendance_pct", "75.0", "Minimum attendance percentage required for VTU Hall Ticket / SEE exam"),
        ("exam_min_cie_pct", "40.0", "Minimum Internal/CIE percentage per subject required for VTU exam eligibility"),
        ("max_backlogs_next_sem", "4", "Maximum active backlogs allowed for Next Semester eligibility"),
        ("min_credits_next_sem", "16.0", "Minimum earned credits required for Next Semester eligibility"),
        ("enforce_credit_pct_mode", "threshold", "Credit eligibility check mode: threshold or percentage"),
        ("min_credit_pct_next_sem", "50.0", "Minimum earned credit percentage when percentage mode is active"),
        ("vtu_scheme", "VTU 2022 Scheme", "Active institutional VTU academic scheme"),
        ("enforce_year_back_rule", "1", "Enforce VTU odd-to-even year-back rule for 1st/2nd year pending backlogs"),
    ]
    for r_key, r_val, r_desc in default_rules:
        cur.execute("""
            INSERT OR IGNORE INTO academic_rules (rule_key, rule_value, description, updated_by, updated_at)
            VALUES (?, ?, ?, 'system', datetime('now'))
        """, (r_key, r_val, r_desc))

    # 23. Faculty Mentoring & Counseling Action Logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mentoring_logs (
        id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        faculty_id TEXT,
        faculty_name TEXT,
        category TEXT DEFAULT 'Attendance / Academic',
        notes TEXT NOT NULL,
        action_taken TEXT,
        parent_notified INTEGER DEFAULT 0,
        created_at TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mentoring_student ON mentoring_logs(student_id);")

    # 24. VTU Marksheet Revision & Revaluation Audit History
    cur.execute("""
    CREATE TABLE IF NOT EXISTS marksheet_revisions (
        id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        semester_number INTEGER NOT NULL,
        old_sgpa REAL,
        new_sgpa REAL,
        old_credits_earned REAL,
        new_credits_earned REAL,
        actor_type TEXT,
        actor_id TEXT,
        notes TEXT,
        created_at TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_revisions_student ON marksheet_revisions(student_id);")

    # 25. Placement Drives & Recruiter Shortlists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS placement_drives (
        id TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        job_role TEXT,
        package_lpa REAL DEFAULT 0.0,
        min_cgpa REAL DEFAULT 6.0,
        max_active_backlogs INTEGER DEFAULT 0,
        min_attendance_pct REAL DEFAULT 75.0,
        required_skills TEXT,
        exclude_already_placed INTEGER DEFAULT 0,
        require_next_sem_eligible INTEGER DEFAULT 1,
        drive_date TEXT,
        created_by TEXT,
        created_at TEXT
    );
    """)

    # 26. Campus Placement Drive Applications & ATS Pipeline
    cur.execute("""
    CREATE TABLE IF NOT EXISTS drive_applications (
        id TEXT PRIMARY KEY,
        drive_id TEXT NOT NULL,
        student_id TEXT NOT NULL,
        application_stage TEXT DEFAULT 'Applied',
        stage_feedback TEXT DEFAULT '',
        resume_headline TEXT DEFAULT '',
        applied_at TEXT,
        updated_at TEXT,
        FOREIGN KEY(drive_id) REFERENCES placement_drives(id) ON DELETE CASCADE,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_drive_apps_drive ON drive_applications(drive_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_drive_apps_student ON drive_applications(student_id);")

    # 27. Official Parent Communications & Institutional Warning Notices
    cur.execute("""
    CREATE TABLE IF NOT EXISTS parent_communications (
        id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        faculty_id TEXT,
        faculty_name TEXT,
        communication_type TEXT NOT NULL,
        channel TEXT DEFAULT 'Official Notice',
        parent_phone TEXT,
        parent_email TEXT,
        subject TEXT NOT NULL,
        message_body TEXT NOT NULL,
        status TEXT DEFAULT 'Dispatched',
        sent_at TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_parent_comm_student ON parent_communications(student_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_parent_comm_type ON parent_communications(communication_type);")

    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edupredict.db")
    run_migrations(db_file)
    print("Database migrations ran successfully.")
