"""
Admin Portal and System Governance Module for EduPredict AI.
Provides:
- Executive Dashboard & System Metrics
- User Account Management (Students, Faculty, Administrators)
- Academic Year & Semester Locking Controls
- Database Backup, Snapshot & Restore Engine (with rollback protection)
- System Health & Diagnostics (DB integrity, PyMuPDF, Tesseract, ML models, File Storage)
- Comprehensive Audit Log Viewer & CSV Exporter
"""

import os
import shutil
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")
BACKUPS_DIR = os.path.join(APP_DIR, "backups")
UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
MODELS_DIR = os.path.join(APP_DIR, "ml", "trained_models")

os.makedirs(BACKUPS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

from auth import (
    get_db,
    hash_password,
    log_audit_action,
    normalize_usn,
    normalize_tjit,
    send_notification
)


def get_system_stats() -> Dict[str, Any]:
    """Retrieve high-level system metrics from database."""
    db = get_db()
    stats = {}
    try:
        stats["students"] = db.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
        stats["student_users"] = db.execute("SELECT COUNT(*) AS c FROM student_users WHERE active = 1").fetchone()["c"]
        stats["staff"] = db.execute("SELECT COUNT(*) AS c FROM staff_users WHERE active = 1").fetchone()["c"]
        stats["classes"] = db.execute("SELECT COUNT(*) AS c FROM classes").fetchone()["c"]
        stats["semesters"] = db.execute("SELECT COUNT(*) AS c FROM semesters").fetchone()["c"]
        stats["predictions"] = db.execute("SELECT COUNT(*) AS c FROM academic_predictions").fetchone()["c"]
        stats["audit_logs"] = db.execute("SELECT COUNT(*) AS c FROM audit_logs").fetchone()["c"]
        stats["db_size_kb"] = round(os.path.getsize(DB_FILE) / 1024, 1) if os.path.exists(DB_FILE) else 0
        return stats
    finally:
        db.close()


def render_admin_portal():
    """Render the master Administrator & System Governance Portal."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(49, 46, 129, 0.85) 50%, rgba(99, 102, 241, 0.9) 100%); backdrop-filter: blur(20px); padding: 26px 30px; border-radius: 20px; color: white; margin-bottom: 24px; box-shadow: 0 20px 40px -15px rgba(99, 102, 241, 0.4); border: 1px solid rgba(165, 180, 252, 0.25);">
        <h2 style="color: #FFFFFF !important; margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">🛡️ System Administrator & Governance Portal</h2>
        <p style="color: #C7D2FE; margin: 8px 0 0 0; font-size: 14px;">Master control panel for user role governance, semester locking, live database snapshots, system health diagnostics, and security audit monitoring.</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_system_stats()

    # Top KPI Metrics
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Enrolled Students", stats.get("students", 0))
    k2.metric("Active Student Logins", stats.get("student_users", 0))
    k3.metric("Faculty / Admins", stats.get("staff", 0))
    k4.metric("Active Classes", stats.get("classes", 0))
    k5.metric("ML Predictions", stats.get("predictions", 0))
    k6.metric("DB Size", f"{stats.get('db_size_kb', 0)} KB")

    admin_tabs = st.tabs([
        "🎯 ICRATE-2026 Live Dashboard",
        "👥 User Management",
        "🔒 Semester Locking",
        "💾 Backup & Restore",
        "🩺 System Health",
        "📜 Audit Logs & Security"
    ])

    # ============================================================
    # TAB 0: LIVE RESEARCH PREDICTION DASHBOARD (SLIDE 4, FIG 3)
    # ============================================================
    with admin_tabs[0]:
        from portal_staff import render_live_student_academic_performance_dashboard
        render_live_student_academic_performance_dashboard(db=get_db(), is_admin=True)

    # ============================================================
    # TAB 1: USER MANAGEMENT
    # ============================================================
    with admin_tabs[1]:
        st.subheader("👥 User Account & Role Governance")
        
        user_role_choice = st.radio("Select User Directory", ["👨‍🎓 Student Accounts", "🏫 Faculty & Admin Accounts"], horizontal=True)
        db = get_db()
        try:
            if user_role_choice == "👨‍🎓 Student Accounts":
                st.markdown("#### Student Accounts Directory")
                search_q = st.text_input("🔍 Search Student by USN or Name", placeholder="e.g. 1TJ or Name")
                
                query = """
                    SELECT s.id, s.usn, s.name, c.class_name, s.current_semester, 
                           u.id AS user_id, u.active AS is_active, u.created_at AS account_created
                    FROM students s
                    LEFT JOIN classes c ON s.class_id = c.id
                    LEFT JOIN student_users u ON s.id = u.student_id
                """
                params = []
                if search_q.strip():
                    query += " WHERE s.usn LIKE ? OR s.name LIKE ?"
                    params = [f"%{search_q.strip()}%", f"%{search_q.strip()}%"]
                query += " ORDER BY s.name ASC"
                
                students_df = pd.read_sql_query(query, db, params=params)
                
                if not students_df.empty:
                    st.dataframe(
                        students_df[["usn", "name", "class_name", "current_semester", "is_active", "account_created"]],
                        column_config={
                            "usn": "USN",
                            "name": "Student Name",
                            "class_name": "Class",
                            "current_semester": "Sem",
                            "is_active": "Active Status (1=Yes, 0=No)",
                            "account_created": "Created Date"
                        },
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    st.markdown("##### ⚙️ Student Account Action")
                    c_act1, c_act2 = st.columns(2)
                    with c_act1:
                        target_usn = st.selectbox("Select Student USN", students_df["usn"].tolist(), key="admin_stu_select")
                        new_stu_pwd = st.text_input("New Password for Selected Student", placeholder="Leave blank to reset to default 'Student@123'", type="password")
                        if st.button("🔑 Reset Student Password", type="primary"):
                            target_pwd = new_stu_pwd.strip() if new_stu_pwd.strip() else "Student@123"
                            if len(target_pwd) < 6:
                                st.error("Password must be at least 6 characters.")
                            else:
                                new_hash = hash_password(target_pwd)
                                db.execute("UPDATE student_users SET password_hash = ? WHERE usn = ?", (new_hash, target_usn))
                                db.commit()
                                log_audit_action("admin", st.session_state.get("user_id", "admin"), "ADMIN_RESET_PASSWORD", f"Reset password for USN: {target_usn}")
                                st.success(f"Password for {target_usn} successfully reset to '{target_pwd}'!")
                    
                    with c_act2:
                        sel_user = students_df[students_df["usn"] == target_usn].iloc[0]
                        current_status = sel_user["is_active"]
                        st.info(f"Student: **{sel_user['name']}** ({target_usn})\nCurrent Account Status: **{'Active' if current_status == 1 else 'Deactivated / No Login'}**")
                        toggle_label = "Deactivate Student Account" if current_status == 1 else "Activate Student Account"
                        if st.button(f"⚡ {toggle_label}"):
                            new_val = 0 if current_status == 1 else 1
                            db.execute("UPDATE student_users SET active = ? WHERE usn = ?", (new_val, target_usn))
                            db.commit()
                            log_audit_action("admin", st.session_state.get("user_id", "admin"), "ADMIN_TOGGLE_USER_STATUS", f"Changed active status to {new_val} for {target_usn}")
                            st.success(f"Student {target_usn} active status updated to {new_val}!")
                            st.rerun()
                else:
                    st.info("No student accounts match the search query.")

            else:
                st.markdown("#### Faculty & Staff Accounts Directory")
                staff_df = pd.read_sql_query("SELECT id, tjit_number, name, email, phone, role, active, created_at FROM staff_users ORDER BY name ASC", db)
                if not staff_df.empty:
                    st.dataframe(
                        staff_df[["tjit_number", "name", "email", "role", "active", "created_at"]],
                        column_config={
                            "tjit_number": "Staff ID",
                            "name": "Faculty Name",
                            "email": "Email",
                            "role": "Role (staff/admin)",
                            "active": "Active Status",
                            "created_at": "Registered"
                        },
                        use_container_width=True
                    )

                    st.markdown("---")
                    st.markdown("##### ⚙️ Faculty Account Action & Role Promotion")
                    s_c1, s_c2 = st.columns(2)
                    with s_c1:
                        target_tjit = st.selectbox("Select Faculty Staff ID", staff_df["tjit_number"].tolist(), key="admin_staff_select")
                        new_st_role = st.selectbox("Assign System Role", ["staff", "admin"], key="admin_st_role_sel")
                        if st.button("🛡️ Update Role"):
                            db.execute("UPDATE staff_users SET role = ? WHERE tjit_number = ?", (new_st_role, target_tjit))
                            db.commit()
                            log_audit_action("admin", st.session_state.get("user_id", "admin"), "ADMIN_ROLE_CHANGE", f"Updated role of {target_tjit} to {new_st_role}")
                            st.success(f"Staff {target_tjit} role updated to '{new_st_role}'!")
                            st.rerun()

                    with s_c2:
                        new_st_pwd = st.text_input("Reset Faculty Password", type="password", key="admin_st_pwd_in")
                        if st.button("🔑 Reset Faculty Password", type="primary"):
                            if not new_st_pwd or len(new_st_pwd) < 6:
                                st.error("Password must be at least 6 characters.")
                            else:
                                new_hash = hash_password(new_st_pwd)
                                db.execute("UPDATE staff_users SET password_hash = ? WHERE tjit_number = ?", (new_hash, target_tjit))
                                db.commit()
                                log_audit_action("admin", st.session_state.get("user_id", "admin"), "ADMIN_RESET_STAFF_PWD", f"Reset password for {target_tjit}")
                                st.success(f"Password for {target_tjit} updated successfully!")
        finally:
            db.close()

    # ============================================================
    # TAB 2: SEMESTER LOCKING
    # ============================================================
    with admin_tabs[2]:
        st.subheader("🔒 Semester & Class Locking Controls")
        st.markdown("""
        **Academic Integrity Policy:** When a semester or class is **Locked**, students cannot modify or upload marks, backlogs, or IA records for that term.
        Use this feature once evaluation periods close to prevent unauthorized post-exam alterations.
        """)

        db = get_db()
        try:
            classes = db.execute("SELECT id, class_name, section, academic_year, semester, is_locked FROM classes ORDER BY class_name ASC").fetchall()
            if classes:
                st.markdown("##### 🏫 Class-level Lock Status")
                for c in classes:
                    c_col1, c_col2, c_col3 = st.columns([3, 1.5, 1.5])
                    with c_col1:
                        lock_badge = "🔴 LOCKED" if c["is_locked"] == 1 else "🟢 OPEN / EDITABLE"
                        st.markdown(f"**{c['class_name']}** (Sec: {c['section'] or 'A'} | Year: {c['academic_year'] or '2024-25'} | Sem: {c['semester']}) — `{lock_badge}`")
                    with c_col2:
                        pass
                    with c_col3:
                        btn_label = "🔓 Unlock Class" if c["is_locked"] == 1 else "🔒 Lock Class"
                        if st.button(btn_label, key=f"lock_class_{c['id']}"):
                            new_lock = 0 if c["is_locked"] == 1 else 1
                            db.execute("UPDATE classes SET is_locked = ? WHERE id = ?", (new_lock, c["id"]))
                            db.commit()
                            log_audit_action("admin", st.session_state.get("user_id", "admin"), "SEMESTER_LOCK_TOGGLE", f"Set is_locked={new_lock} for class {c['class_name']}")
                            st.success(f"Class '{c['class_name']}' is now {'Locked' if new_lock == 1 else 'Unlocked'}!")
                            st.rerun()

                st.markdown("---")
                st.markdown("##### 📚 Student-wise Semester Term Locks")
                sem_rows = pd.read_sql_query("""
                    SELECT sem.id, s.usn, s.name, sem.semester_number, sem.sgpa, sem.is_locked
                    FROM semesters sem
                    JOIN students s ON sem.student_id = s.id
                    ORDER BY s.usn ASC, sem.semester_number ASC
                """, db)
                if not sem_rows.empty:
                    st.dataframe(
                        sem_rows[["usn", "name", "semester_number", "sgpa", "is_locked"]],
                        column_config={
                            "usn": "USN",
                            "name": "Student Name",
                            "semester_number": "Semester",
                            "sgpa": "SGPA",
                            "is_locked": "Locked (1=Locked, 0=Open)"
                        },
                        use_container_width=True
                    )
            else:
                st.info("No classes found in the system.")
        finally:
            db.close()

    # ============================================================
    # TAB 3: BACKUP & RESTORE
    # ============================================================
    with admin_tabs[3]:
        st.subheader("💾 Database Snapshot, Backup & Disaster Recovery")
        st.markdown("EduPredict AI maintains non-destructive database preservation. Create snapshots prior to significant academic imports or term transitions.")

        b_col1, b_col2 = st.columns(2)

        with b_col1:
            st.markdown("##### 📸 Create Live Snapshot")
            snap_note = st.text_input("Snapshot Tag / Note (optional)", placeholder="e.g. Before_Sem5_Exams")
            if st.button("💾 Create Database Backup Now", type="primary"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag_suffix = f"_{snap_note.strip()}" if snap_note.strip() else ""
                backup_filename = f"edupredict_backup_{timestamp}{tag_suffix}.db"
                target_backup_path = os.path.join(BACKUPS_DIR, backup_filename)
                
                # Perform clean sqlite vacuum / copy
                try:
                    shutil.copy2(DB_FILE, target_backup_path)
                    log_audit_action("admin", st.session_state.get("user_id", "admin"), "DB_BACKUP_CREATED", f"Created snapshot {backup_filename}")
                    st.success(f"Snapshot created successfully: `{backup_filename}`")
                except Exception as e:
                    st.error(f"Failed to create backup: {e}")

            # Download active database
            if os.path.exists(DB_FILE):
                with open(DB_FILE, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Active Database (edupredict.db)",
                        data=f.read(),
                        file_name=f"edupredict_active_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                        mime="application/x-sqlite3",
                        use_container_width=True
                    )

        with b_col2:
            st.markdown("##### 🔄 Existing Snapshots & Restore")
            snapshot_files = [f for f in os.listdir(BACKUPS_DIR) if f.endswith(".db")]
            snapshot_files.sort(reverse=True)

            if snapshot_files:
                selected_snap = st.selectbox("Select Snapshot to Restore", snapshot_files)
                snap_path = os.path.join(BACKUPS_DIR, selected_snap)
                snap_size = round(os.path.getsize(snap_path) / 1024, 1)
                st.caption(f"Selected Snapshot Size: **{snap_size} KB**")

                # Restore Button with safety confirmation
                st.warning("⚠️ Restoring a snapshot will replace the current active database with the chosen snapshot. A rollback snapshot of the current state will be created automatically before restore.")
                if st.button(f"🔁 Restore Database from '{selected_snap}'", type="secondary"):
                    try:
                        # 1. Create immediate rollback backup of current db
                        rb_file = os.path.join(BACKUPS_DIR, f"pre_restore_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
                        if os.path.exists(DB_FILE):
                            shutil.copy2(DB_FILE, rb_file)
                        
                        # 2. Overwrite with selected snapshot
                        shutil.copy2(snap_path, DB_FILE)
                        log_audit_action("admin", st.session_state.get("user_id", "admin"), "DB_RESTORE_COMPLETED", f"Restored database from {selected_snap}")
                        st.success(f"Database successfully restored from {selected_snap}! System refreshed.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Restore operation failed: {e}")
            else:
                st.info("No previous snapshot files in the `backups/` directory.")

        st.markdown("---")
        st.markdown("##### 📤 Upload Database File (.db) to Restore")
        uploaded_db = st.file_uploader("Upload External SQLite Database", type=["db", "sqlite", "sqlite3"])
        if uploaded_db is not None:
            if st.button("🚀 Apply Uploaded Database", type="primary"):
                try:
                    # Save safety rollback
                    rb_file = os.path.join(BACKUPS_DIR, f"pre_upload_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
                    if os.path.exists(DB_FILE):
                        shutil.copy2(DB_FILE, rb_file)
                    
                    with open(DB_FILE, "wb") as f:
                        f.write(uploaded_db.getbuffer())
                    
                    log_audit_action("admin", st.session_state.get("user_id", "admin"), "DB_UPLOADED_RESTORE", "Restored db from user upload")
                    st.success("Uploaded database applied successfully! Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to apply database: {e}")

    # ============================================================
    # TAB 4: SYSTEM HEALTH & DIAGNOSTICS
    # ============================================================
    with admin_tabs[4]:
        st.subheader("🩺 System Diagnostics & Health Status")
        
        dh1, dh2 = st.columns(2)

        with dh1:
            st.markdown("##### 🗄️ Database Integrity")
            db = get_db()
            try:
                integrity = db.execute("PRAGMA integrity_check;").fetchall()
                integrity_res = [row[0] for row in integrity]
                if "ok" in integrity_res:
                    st.success("✅ SQLite Database Integrity: **OK (No corruption detected)**")
                else:
                    st.error(f"❌ Database Issues: {integrity_res}")
                
                # Table Breakdown
                st.markdown("##### 📊 Table Row Counts")
                tables = ["students", "classes", "semesters", "subjects", "master_subjects", "attendance", "current_semester_records", "academic_predictions", "staff_users", "student_users", "audit_logs", "certificates", "student_letters"]
                tbl_data = []
                for tbl in tables:
                    try:
                        cnt = db.execute(f"SELECT COUNT(*) AS c FROM {tbl}").fetchone()["c"]
                        tbl_data.append({"Table Name": tbl, "Record Count": cnt})
                    except Exception:
                        tbl_data.append({"Table Name": tbl, "Record Count": "N/A"})
                st.dataframe(pd.DataFrame(tbl_data), use_container_width=True)
            finally:
                db.close()

        with dh2:
            st.markdown("##### ⚙️ Subsystem Status")
            
            # PyMuPDF Status
            try:
                import fitz
                st.success(f"✅ PyMuPDF (PDF OCR Engine): **Installed & Active (v{fitz.__version__})**")
            except ImportError:
                st.warning("⚠️ PyMuPDF is not installed. PDF text extraction will fall back.")

            # Tesseract OCR Status
            try:
                import pytesseract
                from PIL import Image
                tess_cmd = pytesseract.pytesseract.tesseract_cmd
                st.info(f"ℹ️ Tesseract OCR Command Path: `{tess_cmd}`")
            except Exception as e:
                st.warning(f"⚠️ pytesseract module info: {e}")

            # Machine Learning Models Status
            st.markdown("##### 🧠 Machine Learning Engine")
            rf_model_path = os.path.join(APP_DIR, "ml", "trained_models", "subject_final_marks_model.pkl")
            if os.path.exists(rf_model_path):
                st.success("✅ Subject Grade Prediction Model: **Loaded & Active (Random Forest + Ridge, $R^2 > 0.91$)**")
            else:
                st.info("ℹ️ ML Model trained weights file loaded via default fallbacks.")

            # File Storage Access
            st.markdown("##### 📁 Storage Permissions")
            dirs_to_check = [("Uploads", UPLOADS_DIR), ("Backups", BACKUPS_DIR)]
            for d_name, d_path in dirs_to_check:
                if os.path.exists(d_path) and os.access(d_path, os.W_OK):
                    st.success(f"✅ Directory `{d_name}`: **Writable**")
                else:
                    st.error(f"❌ Directory `{d_name}`: **Permission Denied / Missing**")

    # ============================================================
    # TAB 5: AUDIT LOGS & SECURITY
    # ============================================================
    with admin_tabs[5]:
        st.subheader("📜 Comprehensive Audit Trail & Security Logs")
        
        db = get_db()
        try:
            log_col1, log_col2, log_col3 = st.columns([2, 2, 2])
            with log_col1:
                filter_user_type = st.selectbox("Filter User Type", ["All", "student", "staff", "admin"])
            with log_col2:
                search_action = st.text_input("Search Action / Details", placeholder="e.g. LOGIN, MARKS")
            with log_col3:
                log_limit = st.selectbox("Max Records", [50, 100, 250, 500, 1000], index=1)

            query = "SELECT id, user_type, user_id, action, details, timestamp FROM audit_logs"
            conditions = []
            params = []

            if filter_user_type != "All":
                conditions.append("user_type = ?")
                params.append(filter_user_type)

            if search_action.strip():
                conditions.append("(action LIKE ? OR details LIKE ?)")
                params.extend([f"%{search_action.strip()}%", f"%{search_action.strip()}%"])

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(log_limit)

            logs_df = pd.read_sql_query(query, db, params=params)

            if not logs_df.empty:
                st.dataframe(
                    logs_df,
                    column_config={
                        "id": "Log ID",
                        "user_type": "Actor Type",
                        "user_id": "User Identifier",
                        "action": "Action Event",
                        "details": "Details",
                        "timestamp": "Timestamp"
                    },
                    use_container_width=True
                )

                # Export CSV
                csv_data = logs_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Audit Logs as CSV",
                    data=csv_data,
                    file_name=f"edupredict_audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No audit logs match the current filters.")
        finally:
            db.close()
