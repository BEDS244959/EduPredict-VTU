"""
Authentication, User Management, and Security Auditing Module for EduPredict AI.
Implements:
- PBKDF2 HMAC SHA-256 password hashing with 16-byte cryptographic salt
- Role-based authentication (STUDENT and STAFF)
- USN and TJIT normalization
- First-time staff setup detection & account creation
- Student login self-registration & auto-provisioning
- Secure password change & reset workflows
- Streamlit session state management & secure logout
- Comprehensive audit logging and real-time notifications
"""

import os
import re
import hashlib
import secrets
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ============================================================
# PASSWORD SECURITY & NORMALIZATION
# ============================================================

def hash_password(password: str) -> str:
    """Hash password using PBKDF2 HMAC SHA-256 with a 16-byte random salt."""
    if not password:
        raise ValueError("Password cannot be empty")
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations=100000
    )
    return f"{salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hash format."""
    try:
        if not password or not stored_hash or "$" not in stored_hash:
            return False
        salt_hex, key_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        candidate_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations=100000
        )
        return secrets.compare_digest(candidate_key, expected_key)
    except Exception:
        return False


def normalize_usn(usn: str) -> str:
    """Normalize USN: strip whitespace and convert to uppercase."""
    return (usn or "").strip().upper()


def normalize_tjit(tjit: str) -> str:
    """Normalize TJIT/Staff number: strip whitespace and convert to uppercase."""
    return (tjit or "").strip().upper()


# ============================================================
# AUDIT LOGGING & NOTIFICATIONS
# ============================================================

def log_audit_action(user_type: str, user_id: str, action: str, details: str = "", target_id: str = ""):
    """Record staff/student action to audit_logs table."""
    try:
        db = get_db()
        db.execute("""
            INSERT INTO audit_logs (id, user_type, user_id, action, details, target_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (uuid.uuid4().hex, user_type, str(user_id or ""), action, details, str(target_id or ""), datetime.now().isoformat()))
        db.commit()
        db.close()
    except Exception:
        pass


def log_login_activity(user_type: str, user_id: str, success: int, ip_info: str = ""):
    """Log login attempt into login_activity."""
    try:
        db = get_db()
        db.execute("""
            INSERT INTO login_activity (id, user_type, user_id, login_time, success, ip_info)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (uuid.uuid4().hex, user_type, str(user_id), datetime.now().isoformat(), success, ip_info))
        db.commit()
        db.close()
    except Exception:
        pass


def send_notification(recipient_type: str, recipient_id: str, title: str, message: str, category: str = "General"):
    """Send a notification to a specific student or staff member."""
    try:
        db = get_db()
        db.execute("""
            INSERT INTO notifications (id, recipient_type, recipient_id, title, message, category, is_read, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
        """, (uuid.uuid4().hex, recipient_type, recipient_id, title, message, category, datetime.now().isoformat()))
        db.commit()
        db.close()
    except Exception:
        pass


def mark_notification_as_read(notification_id: str, user_id: str) -> bool:
    """Mark a notification as read for a specific user."""
    try:
        db = get_db()
        now = datetime.now().isoformat()
        db.execute("""
            INSERT INTO notification_reads (id, notification_id, user_id, read_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(notification_id, user_id) DO UPDATE SET read_at = excluded.read_at
        """, (uuid.uuid4().hex, notification_id, str(user_id), now))
        
        # If this was a direct notification, also mark the main record
        db.execute("UPDATE notifications SET is_read = 1 WHERE id = ? AND recipient_id = ?", (notification_id, str(user_id)))
        db.commit()
        db.close()
        return True
    except Exception:
        return False


def mark_notification_as_unread(notification_id: str, user_id: str) -> bool:
    """Mark a notification as unread for a specific user."""
    try:
        db = get_db()
        db.execute("DELETE FROM notification_reads WHERE notification_id = ? AND user_id = ?", (notification_id, str(user_id)))
        db.execute("UPDATE notifications SET is_read = 0 WHERE id = ? AND recipient_id = ?", (notification_id, str(user_id)))
        db.commit()
        db.close()
        return True
    except Exception:
        return False


def mark_all_notifications_as_read(user_id: str, recipient_type: str = "student") -> int:
    """Mark all unread notifications for this user as read."""
    try:
        db = get_db()
        now = datetime.now().isoformat()
        
        # Get all notifications addressed to this user or all
        rows = db.execute("""
            SELECT id FROM notifications
            WHERE (recipient_id = ? OR recipient_type = 'all' OR (recipient_type = ? AND recipient_id = 'all'))
              AND id NOT IN (SELECT notification_id FROM notification_reads WHERE user_id = ?)
        """, (str(user_id), recipient_type, str(user_id))).fetchall()
        
        count = 0
        for r in rows:
            db.execute("""
                INSERT INTO notification_reads (id, notification_id, user_id, read_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(notification_id, user_id) DO NOTHING
            """, (uuid.uuid4().hex, r["id"], str(user_id), now))
            count += 1
            
        db.execute("UPDATE notifications SET is_read = 1 WHERE recipient_id = ?", (str(user_id),))
        db.commit()
        db.close()
        return count
    except Exception:
        return 0


def get_user_notifications(user_id: str, recipient_type: str = "student") -> Dict[str, Any]:
    """Retrieve all notifications categorized into unread and read for the user."""
    db = get_db()
    try:
        # Load all applicable notifications
        query = """
            SELECT n.*,
                   CASE WHEN nr.id IS NOT NULL THEN 1 ELSE n.is_read END as user_is_read,
                   nr.read_at
            FROM notifications n
            LEFT JOIN notification_reads nr ON n.id = nr.notification_id AND nr.user_id = ?
            WHERE (n.recipient_id = ? OR n.recipient_type = 'all' OR (n.recipient_type = ? AND n.recipient_id = 'all'))
            ORDER BY n.created_at DESC
        """
        rows = db.execute(query, (str(user_id), str(user_id), recipient_type)).fetchall()
        
        unread_list = []
        read_list = []
        for r in rows:
            d = dict(r)
            if d.get("user_is_read") == 1:
                read_list.append(d)
            else:
                unread_list.append(d)
                
        return {
            "unread": unread_list,
            "read": read_list,
            "unread_count": len(unread_list),
            "total_count": len(rows)
        }
    finally:
        db.close()


# ============================================================
# STAFF AUTHENTICATION & SETUP
# ============================================================

def count_staff_users() -> int:
    """Return count of active staff users."""
    db = get_db()
    try:
        row = db.execute("SELECT COUNT(*) AS c FROM staff_users WHERE active = 1").fetchone()
        return row["c"] if row else 0
    finally:
        db.close()


def create_first_staff(tjit_number: str, name: str, password: str, email: str = "", phone: str = "") -> Tuple[bool, str]:
    """Create the initial or additional staff user."""
    tjit = normalize_tjit(tjit_number)
    name = (name or "").strip()

    if not tjit:
        return False, "Staff / TJIT ID is required."
    if not name:
        return False, "Staff Name is required."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."

    db = get_db()
    try:
        existing = db.execute("SELECT id FROM staff_users WHERE tjit_number = ?", (tjit,)).fetchone()
        if existing:
            return False, f"Staff ID '{tjit}' is already registered. Please log in."

        staff_id = uuid.uuid4().hex
        pwd_hash = hash_password(password)
        created_at = datetime.now().isoformat()

        db.execute("""
            INSERT INTO staff_users (id, tjit_number, name, password_hash, role, active, created_at, email, phone)
            VALUES (?, ?, ?, ?, 'staff', 1, ?, ?, ?)
        """, (staff_id, tjit, name, pwd_hash, created_at, email.strip(), phone.strip()))
        db.commit()

        log_audit_action("staff", staff_id, "STAFF_REGISTRATION", f"Staff account created for {name} ({tjit})")
        return True, "Staff account created successfully! You can now log in."
    except sqlite3.IntegrityError:
        return False, "Staff ID already registered."
    except Exception as e:
        return False, f"Failed to create staff account: {e}"
    finally:
        db.close()


def authenticate_staff(tjit_number: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Verify staff credentials."""
    tjit = normalize_tjit(tjit_number)
    if not tjit or not password:
        return False, None, "Please enter both Staff ID and password."

    db = get_db()
    try:
        user = db.execute("""
            SELECT * FROM staff_users
            WHERE tjit_number = ? AND active = 1
        """, (tjit,)).fetchone()

        if not user:
            log_login_activity("staff", tjit, 0, "Invalid Staff ID")
            return False, None, "Invalid Staff ID or password."

        if not verify_password(password, user["password_hash"]):
            log_login_activity("staff", user["id"], 0, "Invalid Password")
            return False, None, "Invalid Staff ID or password."

        log_login_activity("staff", user["id"], 1, "Success")
        log_audit_action("staff", user["id"], "STAFF_LOGIN", f"Staff {user['name']} logged in successfully")
        return True, dict(user), "Login successful."
    finally:
        db.close()


# ============================================================
# STUDENT AUTHENTICATION & PROVISIONING
# ============================================================

def ensure_student_login(student_id: str, default_password: Optional[str] = None) -> Optional[str]:
    """Ensure student_users record exists for student_id."""
    db = get_db()
    try:
        stu = db.execute("SELECT id, name, usn FROM students WHERE id = ?", (student_id,)).fetchone()
        if not stu or not stu["usn"]:
            return None

        normalized = normalize_usn(stu["usn"])
        existing = db.execute("SELECT id FROM student_users WHERE student_id = ?", (student_id,)).fetchone()
        if existing:
            return None

        temp_pwd = default_password if default_password else f"Edu@{secrets.token_hex(3)}"
        pwd_hash = hash_password(temp_pwd)
        uid = uuid.uuid4().hex
        now = datetime.now().isoformat()

        db.execute("""
            INSERT OR REPLACE INTO student_users (id, student_id, usn, password_hash, active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
        """, (uid, student_id, normalized, pwd_hash, now))
        db.commit()
        return temp_pwd
    finally:
        db.close()


def provision_existing_students(default_password: str = "Student@123") -> int:
    """Provision logins for all existing students in the database who do not have one."""
    db = get_db()
    count = 0
    try:
        students = db.execute("""
            SELECT s.id, s.usn
            FROM students s
            LEFT JOIN student_users u ON s.id = u.student_id
            WHERE u.id IS NULL AND s.usn IS NOT NULL AND TRIM(s.usn) != ''
        """).fetchall()

        for stu in students:
            normalized = normalize_usn(stu["usn"])
            uid = uuid.uuid4().hex
            pwd_hash = hash_password(default_password)
            now = datetime.now().isoformat()
            db.execute("""
                INSERT OR IGNORE INTO student_users (id, student_id, usn, password_hash, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (uid, stu["id"], normalized, pwd_hash, now))
            count += 1
        db.commit()
        return count
    finally:
        db.close()


def student_first_time_setup(usn: str, new_password: str, confirm_password: str) -> Tuple[bool, str]:
    """Allow an enrolled student to set their password for first-time login."""
    norm_usn = normalize_usn(usn)
    if not norm_usn:
        return False, "USN is required."
    if not new_password or len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    if new_password != confirm_password:
        return False, "Passwords do not match."

    db = get_db()
    try:
        stu = db.execute("SELECT * FROM students WHERE usn = ?", (norm_usn,)).fetchone()
        if not stu:
            return False, f"USN '{norm_usn}' is not registered in the student directory. Please contact your Class Teacher."

        pwd_hash = hash_password(new_password)
        now = datetime.now().isoformat()

        existing_user = db.execute("SELECT id FROM student_users WHERE student_id = ? OR usn = ?", (stu["id"], norm_usn)).fetchone()
        if existing_user:
            db.execute("""
                UPDATE student_users
                SET password_hash = ?, active = 1
                WHERE id = ?
            """, (pwd_hash, existing_user["id"]))
        else:
            db.execute("""
                INSERT INTO student_users (id, student_id, usn, password_hash, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (uuid.uuid4().hex, stu["id"], norm_usn, pwd_hash, now))

        db.commit()
        log_audit_action("student", stu["id"], "PASSWORD_SET", f"Student {norm_usn} set password successfully")
        return True, "Password configured successfully! You can now log in with your USN and new password."
    except Exception as e:
        return False, f"Failed to set password: {e}"
    finally:
        db.close()


def authenticate_student(usn: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    Authenticate a student by USN and password.
    Returns (success, student_user_dict, student_profile_dict, message)
    """
    norm_usn = normalize_usn(usn)
    if not norm_usn or not password:
        return False, None, None, "Please enter both USN and password."

    db = get_db()
    try:
        user = db.execute("""
            SELECT * FROM student_users
            WHERE usn = ? AND active = 1
        """, (norm_usn,)).fetchone()

        if not user:
            # Check if student exists in students table
            stu_rec = db.execute("SELECT id, name, usn FROM students WHERE usn = ?", (norm_usn,)).fetchone()
            if stu_rec:
                if password == "Student@123":
                    # Auto-provision login with default password immediately
                    uid = uuid.uuid4().hex
                    pwd_hash = hash_password("Student@123")
                    now = datetime.now().isoformat()
                    db.execute("""
                        INSERT OR REPLACE INTO student_users (id, student_id, usn, password_hash, active, created_at)
                        VALUES (?, ?, ?, ?, 1, ?)
                    """, (uid, stu_rec["id"], norm_usn, pwd_hash, now))
                    db.commit()
                    user = db.execute("SELECT * FROM student_users WHERE student_id = ?", (stu_rec["id"],)).fetchone()
                else:
                    return False, None, None, "Invalid password. Default password is 'Student@123', or you can activate in 'First-time Student Setup'."
            else:
                log_login_activity("student", norm_usn, 0, "Unknown USN")
                return False, None, None, "USN not found. Please contact your Class Teacher to enroll your USN."

        if not user or not verify_password(password, user["password_hash"]):
            log_login_activity("student", user["id"] if user else norm_usn, 0, "Invalid Password")
            return False, None, None, "Invalid USN or password. (Default password: Student@123)"

        # Fetch corresponding profile
        profile = db.execute("""
            SELECT s.*, c.class_name, c.class_teacher
            FROM students s
            LEFT JOIN classes c ON s.class_id = c.id
            WHERE s.id = ?
        """, (user["student_id"],)).fetchone()

        if not profile:
            return False, None, None, "Student academic profile record not found."

        log_login_activity("student", user["id"], 1, "Success")
        log_audit_action("student", user["student_id"], "STUDENT_LOGIN", f"Student {profile['name']} ({norm_usn}) logged in")
        return True, dict(user), dict(profile), "Login successful."
    finally:
        db.close()


# ============================================================
# PASSWORD CHANGE & RESET
# ============================================================

def change_user_password(user_role: str, user_id: str, old_pwd: str, new_pwd: str, confirm_pwd: str) -> Tuple[bool, str]:
    """Change password for authenticated user."""
    if not new_pwd or len(new_pwd) < 6:
        return False, "New password must be at least 6 characters long."
    if new_pwd != confirm_pwd:
        return False, "New passwords do not match."

    db = get_db()
    try:
        if user_role == "staff":
            user = db.execute("SELECT password_hash FROM staff_users WHERE id = ?", (user_id,)).fetchone()
            if not user or not verify_password(old_pwd, user["password_hash"]):
                return False, "Incorrect current password."
            new_hash = hash_password(new_pwd)
            db.execute("UPDATE staff_users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        elif user_role == "student":
            user = db.execute("SELECT password_hash FROM student_users WHERE student_id = ? OR id = ?", (user_id, user_id)).fetchone()
            if not user or not verify_password(old_pwd, user["password_hash"]):
                return False, "Incorrect current password."
            new_hash = hash_password(new_pwd)
            db.execute("UPDATE student_users SET password_hash = ? WHERE student_id = ? OR id = ?", (new_hash, user_id, user_id))
        else:
            return False, "Invalid user role."

        db.commit()
        log_audit_action(user_role, user_id, "CHANGE_PASSWORD", "User successfully changed password")
        return True, "Password changed successfully!"
    except Exception as e:
        return False, f"Failed to change password: {e}"
    finally:
        db.close()


def reset_student_password(student_id: str, new_password: str) -> Tuple[bool, str]:
    """Staff resets student password securely."""
    if not new_password or len(new_password) < 6:
        return False, "Password must be at least 6 characters."

    db = get_db()
    try:
        pwd_hash = hash_password(new_password)
        db.execute("""
            UPDATE student_users
            SET password_hash = ?
            WHERE student_id = ?
        """, (pwd_hash, student_id))
        db.commit()
        log_audit_action("staff", "staff_admin", "RESET_STUDENT_PASSWORD", f"Reset password for student ID {student_id}")
        return True, "Student password reset successfully."
    except Exception as e:
        return False, f"Password reset failed: {e}"
    finally:
        db.close()


# ============================================================
# SESSION MANAGEMENT (STREAMLIT)
# ============================================================

def init_session_state(st_module):
    """Initialize standard session state variables."""
    keys = {
        "authenticated": False,
        "user_role": None,       # "student" or "staff"
        "user_id": None,         # staff_users.id or student_users.id
        "student_id": None,      # students.id (ONLY set when user_role == "student")
        "username": None,        # Display name
        "usn": None,             # Student USN
        "tjit_number": None,     # Staff TJIT number
        "temp_student_pwd": None,# For displaying temporary password once
        "staff_role": "staff"
    }
    for k, v in keys.items():
        if k not in st_module.session_state:
            st_module.session_state[k] = v


def logout_user(st_module):
    """Securely log out and clear all session tokens."""
    st_module.session_state["authenticated"] = False
    st_module.session_state["user_role"] = None
    st_module.session_state["user_id"] = None
    st_module.session_state["student_id"] = None
    st_module.session_state["username"] = None
    st_module.session_state["usn"] = None
    st_module.session_state["tjit_number"] = None
    st_module.session_state["temp_student_pwd"] = None
    st_module.rerun()
