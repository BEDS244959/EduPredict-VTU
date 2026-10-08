"""
EduPredict AI — Comprehensive Student Portal.
Implements complete academic, portfolio, predictive guidance, and persistent storage:
1. Student Dashboard (Real-time metrics, SGPA/CGPA, profile %, attendance gauge, backlogs, Plotly charts)
2. My Profile (Personal, academic, career, photo upload/view, contact info)
3. My Academic Records (Semesters 1-8 history, credit-weighted SGPA, CGPA)
4. My Current Semester (Unlimited dynamic subjects, IA1, IA2, internal calculation, SGPA preview)
5. My Marksheet Upload (PyMuPDF/Tesseract OCR, editable review/confirmation table before saving, manual entry)
6. My Certificates (Portfolio certificate upload, category, verification tracking)
7. My Skills (Technical skills, programming languages, tools, frameworks, soft skills)
8. My LinkedIn (Validated LinkedIn profile link with direct launcher)
9. My Extracurricular Activities (Activities, achievements, events, supporting certificates)
10. My Fee Receipts (Semester fee receipt uploads, amount, verification status)
11. My Non-Due Form (Semester clearance document uploads, staff feedback)
12. Submit Letter / Applications (Leave, Bonafide, Permission, Academic requests, status tracking)
13. My Verification Status (Live 360-degree verification status across all documents)
14. My Attendance (Hourly logs, subject-wise attendance %, low attendance alert)
15. My Backlogs (Dedicated backlog tracking, rewritten exams, cleared/active status)
16. My AI Prediction & Focus (ML predictions, weak subject detection, educational advice, what-if simulator)
17. Reports (Downloadable student academic PDF and summary report)
18. Notifications (Real-time alerts on marks, verifications, applications)
19. Settings (Change student password, update contact details)
"""

import os
import re
import uuid
import base64
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List, Optional

from auth import (
    log_audit_action,
    change_user_password,
    get_user_notifications,
    mark_notification_as_read,
    mark_notification_as_unread,
    mark_all_notifications_as_read
)
from ocr_engine import (
    process_marksheet_file,
    safe_float,
    normalize_subject_code,
    VTU_KNOWN_SUBJECTS,
    get_canonical_subject_info
)
from reports import generate_student_pdf_report
from ml.predict import (
    predict_subject_final_marks,
    calculate_predicted_sgpa,
    calculate_projected_cgpa,
    determine_attention_level,
    grade_point,
    grade_letter
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
PHOTO_DIR = os.path.join(UPLOAD_DIR, "photos")
os.makedirs(PHOTO_DIR, exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def save_student_file(uploaded_file, subfolder: str, student_id: str) -> str:
    """Save an uploaded file safely into an isolated student directory."""
    if uploaded_file is None:
        return ""
    folder = os.path.join(UPLOAD_DIR, subfolder, str(student_id))
    os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        ext = ".pdf"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def render_student_portal():
    """Render the full Student Portal."""
    student_id = st.session_state.get("student_id")
    if not student_id:
        st.error("Authentication session missing. Please log in.")
        return

    db = get_db()
    student_row = db.execute("""
        SELECT s.*, c.class_name, c.class_teacher
        FROM students s
        LEFT JOIN classes c ON s.class_id = c.id
        WHERE s.id = ?
    """, (student_id,)).fetchone()
    db.close()

    if not student_row:
        st.error("Student profile record not found.")
        return

    student = dict(student_row)
    is_lat_student = bool(student.get("is_lateral_entry") == 1)

    # Student Sidebar Navigation
    with st.sidebar:
        lat_pill = "<div style='font-size: 10.5px; color: #7C3AED; background: #EDE9FE; padding: 2px 8px; border-radius: 12px; display: inline-block; margin-top: 3px; font-weight: 700;'>🚀 Lateral Entry</div>" if is_lat_student else ""
        stu_photo = student.get("photo_path") or ""
        if stu_photo and (stu_photo.startswith("data:image/") or stu_photo.startswith("http") or os.path.exists(stu_photo)):
            avatar_html = f'<div style="text-align: center; margin-bottom: 8px;"><img src="{stu_photo}" style="width: 76px; height: 76px; border-radius: 50%; object-fit: cover; border: 3px solid #3B82F6; box-shadow: 0 4px 12px rgba(59,130,246,0.25);" /></div>'
        else:
            avatar_html = '<div style="font-size: 38px; text-align: center; margin-bottom: 4px;">👨‍🎓</div>'

        st.markdown(f"""
        <div style="padding: 10px 0; text-align: center;">
            {avatar_html}
            <div style="font-weight: 700; font-size: 16px; color: #1E293B;">{student['name']}</div>
            <div style="font-size: 12px; color: #64748B; font-weight: 500;">USN: {student['usn']}</div>
            <div style="font-size: 11px; color: #3B82F6; background: #EFF6FF; padding: 2px 8px; border-radius: 12px; display: inline-block; margin-top: 4px;">Sem {student['current_semester']} &bull; {student['class_name'] or 'Class N/A'}</div>
            {lat_pill}
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        nav_options = [
            "Dashboard",
            "My Profile",
            "Academic Records",
            "Marksheets Upload",
            "Current Semester",
            "Attendance",
            "Backlogs",
            "Skills & Portfolio",
            "Extracurriculars & LinkedIn",
            "Fee Receipts & Non-Due",
            "Submit Letter / Applications",
            "Document Verification Status",
            "AI Prediction & Focus",
            "Reports",
            "Notifications",
            "Settings"
        ]
        cur_nav = st.session_state.get("student_selected_page", "Dashboard")
        if cur_nav not in nav_options:
            cur_nav = "Dashboard"
        nav_idx = nav_options.index(cur_nav)

        dyn_key = f"student_nav_radio_{st.session_state.get('student_nav_counter', 0)}"
        selected_page = st.radio("Navigation Menu", nav_options, index=nav_idx, key=dyn_key)
        st.session_state["student_selected_page"] = selected_page

    def navigate_student_to(page_name: str):
        st.session_state["student_selected_page"] = page_name
        st.session_state["student_nav_counter"] = st.session_state.get("student_nav_counter", 0) + 1
        st.rerun()

    # -------------------------------------------------------------
    # 1. STUDENT DASHBOARD
    # -------------------------------------------------------------
    if selected_page == "Dashboard":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 22px 26px; border-radius: 14px; color: white; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);">
            <h1 style="color: white !important; font-size: 26px; font-weight: 700; margin: 0;">🎓 Welcome back, {student['name']}!</h1>
            <p style="color: #DBEAFE; margin: 6px 0 0 0; font-size: 14px;">VTU Academic Intelligence & Automated Grade Prediction Hub</p>
        </div>
        """, unsafe_allow_html=True)

        col_pic, col_info = st.columns([1, 3])
        with col_pic:
            stu_photo = student.get("photo_path") or ""
            if stu_photo and (stu_photo.startswith("data:image/") or stu_photo.startswith("http") or os.path.exists(stu_photo)):
                st.image(stu_photo, width=140)
            else:
                st.markdown("""
                <div style="width: 140px; height: 140px; border-radius: 12px; background: #F1F5F9; display: flex; align-items: center; justify-content: center; border: 2px dashed #CBD5E1; color: #64748B; font-size: 13px; text-align: center;">
                    Upload Photo<br>Below
                </div>
                """, unsafe_allow_html=True)
            with st.expander("📸 Change Photo", expanded=False):
                d_photo = st.file_uploader("Upload New Passport Photo", type=["jpg", "jpeg", "png"], key="dash_instant_photo_up")
                if d_photo is not None:
                    if st.button("💾 Save Photo", type="primary", key="btn_save_dash_photo", use_container_width=True):
                        p_bytes = d_photo.getvalue()
                        ext_m = "jpeg" if d_photo.name.lower().endswith((".jpg", ".jpeg")) else "png"
                        b64_str = base64.b64encode(p_bytes).decode("utf-8")
                        photo_uri = f"data:image/{ext_m};base64,{b64_str}"
                        db = get_db()
                        db.execute("UPDATE students SET photo_path = ? WHERE id = ?", (photo_uri, student_id))
                        db.commit()
                        db.close()
                        st.success("✅ Photo saved permanently!")
                        st.rerun()

        with col_info:
            entry_badge = '<span style="color: #8B5CF6; font-weight: 700;">🚀 Lateral Entry (Direct Sem 3)</span>' if is_lat_student else '<span style="color: #10B981; font-weight: 600;">Regular 4-Year B.E.</span>'
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(16px); padding: 18px 22px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 14px; color: #E2E8F0;">
                    <div><span style="color: #94A3B8;">USN:</span> <code style="color: #60A5FA; font-weight: 700; background: rgba(59, 130, 246, 0.15); padding: 2px 6px; border-radius: 6px;">{student['usn']}</code></div>
                    <div><span style="color: #94A3B8;">Enrolled Class:</span> <strong style="color: #F8FAFC;">{student['class_name'] or 'Not Assigned'}</strong></div>
                    <div><span style="color: #94A3B8;">Class Teacher:</span> <strong style="color: #F8FAFC;">{student['class_teacher'] or 'Prof. Faculty'}</strong></div>
                    <div><span style="color: #94A3B8;">Admission Year:</span> <strong style="color: #F8FAFC;">{student['admission_year'] or '2024'}</strong></div>
                    <div><span style="color: #94A3B8;">Active Semester:</span> <strong style="color: #A78BFA;">Semester {student['current_semester']}</strong></div>
                    <div><span style="color: #94A3B8;">Admission Type:</span> {entry_badge}</div>
                    <div><span style="color: #94A3B8;">Email:</span> <span style="color: #CBD5E1;">{student['email'] or 'Not Provided'}</span></div>
                    <div><span style="color: #94A3B8;">Phone:</span> <span style="color: #CBD5E1;">{student['phone'] or 'Not Provided'}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Academic metrics from Database
        db = get_db()
        sem_records = db.execute("SELECT semester_number, sgpa FROM semesters WHERE student_id = ? AND sgpa > 0 ORDER BY semester_number ASC", (student_id,)).fetchall()
        sgpas = [float(r["sgpa"]) for r in sem_records]
        cur_sgpa = f"{sgpas[-1]:.2f}" if sgpas else "N/A"
        cur_cgpa = f"{(sum(sgpas) / len(sgpas)):.2f}" if sgpas else "N/A"

        att_row = db.execute("""
            SELECT COUNT(*) AS tot, SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS pres
            FROM attendance WHERE student_id = ?
        """, (student_id,)).fetchone()
        tot_h = att_row["tot"] or 0
        pres_h = att_row["pres"] or 0
        att_pct = round((pres_h / tot_h * 100), 1) if tot_h > 0 else 88.0

        backlog_row = db.execute("SELECT COUNT(*) AS c FROM backlogs WHERE student_id = ? AND status = 'Active'", (student_id,)).fetchone()
        active_backlogs = backlog_row["c"] if backlog_row else 0

        pend_doc = db.execute("""
            SELECT (SELECT COUNT(*) FROM certificates WHERE student_id = ? AND verification_status = 'Pending') +
                   (SELECT COUNT(*) FROM fee_receipts WHERE student_id = ? AND verification_status = 'Pending') +
                   (SELECT COUNT(*) FROM non_due_documents WHERE student_id = ? AND verification_status = 'Pending') +
                   (SELECT COUNT(*) FROM semesters WHERE student_id = ? AND verification_status = 'Pending') AS c
        """, (student_id, student_id, student_id, student_id)).fetchone()["c"]

        pend_let = db.execute("SELECT COUNT(*) AS c FROM student_letters WHERE student_id = ? AND (status = 'Pending' OR status = 'Submitted')", (student_id,)).fetchone()["c"]
        db.close()

        notif_data = get_user_notifications(student_id, "student")
        unread_notif = notif_data["unread_count"]

        # Metric Cards Layout
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recent SGPA", cur_sgpa)
        m2.metric("Overall CGPA", cur_cgpa)
        m3.metric("Attendance Rate", f"{att_pct}%", delta="Compliant (≥75%)" if att_pct >= 75.0 else "-Deficit (<75%)", delta_color="normal" if att_pct >= 75.0 else "inverse")
        m4.metric("Active Backlogs", active_backlogs, delta="Clear" if active_backlogs == 0 else "Action Required", delta_color="normal" if active_backlogs == 0 else "inverse")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Current Semester", f"Sem {student['current_semester']}")
        m6.metric("Pending Verifications", pend_doc)
        m7.metric("Pending Letters", pend_let)
        m8.metric("Unread Alerts", unread_notif)

        if unread_notif > 0:
            c_notif_alert, c_notif_btn = st.columns([4, 1.2])
            with c_notif_alert:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 10px 16px; margin-top: 10px;">
                    <strong style="color: #991B1B; font-size: 14px;">🔔 You have {unread_notif} unread notification(s) waiting for your review.</strong>
                </div>
                """, unsafe_allow_html=True)
            with c_notif_btn:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if st.button("📬 Open Notifications", key="dash_open_notifs", use_container_width=True, type="primary"):
                    navigate_student_to("Notifications")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Plotly Charts Section
        c_chart1, c_chart2 = st.columns([2, 1])
        with c_chart1:
            st.subheader("📈 SGPA Progression Trend")
            if sgpas:
                df_sgpa = pd.DataFrame({
                    "Semester": [f"Sem {r['semester_number']}" for r in sem_records],
                    "SGPA": sgpas
                })
                fig = px.line(df_sgpa, x="Semester", y="SGPA", markers=True, text="SGPA", range_y=[0, 10])
                fig.update_traces(line_color="#3B82F6", line_width=3, marker=dict(size=10, color="#1E3A8A"))
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=260)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload semester marksheets to visualize your SGPA progression graph.")

        with c_chart2:
            st.subheader("⏱️ Attendance Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=att_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981" if att_pct >= 75.0 else "#EF4444"},
                    'steps': [
                        {'range': [0, 75], 'color': "#FEE2E2"},
                        {'range': [75, 100], 'color': "#DCFCE7"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75.0
                    }
                }
            ))
            fig_gauge.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("⚡ Quick Shortcuts")
        q1, q2, q3, q4 = st.columns(4)
        if q1.button("📑 Marksheet OCR Upload", use_container_width=True, key="stu_q_upload"):
            navigate_student_to("Marksheets Upload")
        if q2.button("🎯 AI Grade Prediction", use_container_width=True, key="stu_q_ai"):
            navigate_student_to("AI Prediction & Focus")
        if q3.button("⏱️ View Hourly Attendance", use_container_width=True, key="stu_q_att"):
            navigate_student_to("Attendance")
        if q4.button("📩 Submit Leave/Bonafide Letter", use_container_width=True, key="stu_q_let"):
            navigate_student_to("Submit Letter / Applications")

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 2. MY PROFILE
    # -------------------------------------------------------------
    elif selected_page == "My Profile":
        st.title("👤 My Student Profile")
        st.write("Manage your personal details, official contact info, admission category, and passport photograph.")

        # Dedicated Photo Upload & Management Card
        st.subheader("📸 Passport Photograph & ID Avatar")
        prof_pic_col1, prof_pic_col2 = st.columns([1, 2.5])
        with prof_pic_col1:
            stu_cur_photo = student.get("photo_path") or ""
            if stu_cur_photo and (stu_cur_photo.startswith("data:image/") or stu_cur_photo.startswith("http") or os.path.exists(stu_cur_photo)):
                st.image(stu_cur_photo, width=130, caption="Current Saved Photo")
            else:
                st.markdown("""
                <div style="width: 130px; height: 130px; border-radius: 12px; background: #F1F5F9; display: flex; align-items: center; justify-content: center; border: 2px dashed #CBD5E1; color: #64748B; font-size: 13px; text-align: center;">
                    No Photo<br>Uploaded
                </div>
                """, unsafe_allow_html=True)
        with prof_pic_col2:
            new_photo_file = st.file_uploader("Upload / Replace Passport Photo (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="direct_stu_photo_uploader")
            if new_photo_file is not None:
                if st.button("📤 Upload & Save Photo Permanently", type="primary", key="btn_save_photo_direct", use_container_width=True):
                    photo_bytes = new_photo_file.getvalue()
                    ext_m = "jpeg" if new_photo_file.name.lower().endswith((".jpg", ".jpeg")) else "png"
                    b64_str = base64.b64encode(photo_bytes).decode("utf-8")
                    photo_uri = f"data:image/{ext_m};base64,{b64_str}"
                    db = get_db()
                    db.execute("UPDATE students SET photo_path = ? WHERE id = ?", (photo_uri, student_id))
                    db.commit()
                    db.close()
                    try:
                        save_student_file(new_photo_file, "photos", student_id)
                    except Exception:
                        pass
                    log_audit_action("student", student_id, "UPDATE_PHOTO", f"Updated passport photo for {student['usn']}")
                    st.success("🎉 Passport photograph permanently saved to your account in database!")
                    st.rerun()

        st.divider()
        st.subheader("📝 Personal & Academic Details")
        with st.form("profile_update_form"):
            c1, c2 = st.columns(2)
            with c1:
                name_val = st.text_input("Full Name", value=student["name"])
                st.text_input("USN (University Seat Number)", value=student["usn"], disabled=True, help="Official USN is locked to your university registration.")
                st.text_input("Class Assigned", value=student["class_name"] or "N/A", disabled=True)
                st.text_input("Class Teacher", value=student["class_teacher"] or "N/A", disabled=True)
                cur_sem_edit = st.number_input("Current Semester", min_value=1, max_value=8, value=int(student["current_semester"] or 5))
            with c2:
                adm_year_edit = st.number_input("Admission Year", min_value=2015, max_value=2030, value=int(student.get("admission_year") or 2024))
                is_lat_edit = st.checkbox("🚀 Lateral Entry Student (Direct 2nd Year / Sem 3)", value=is_lat_student, help="Check if you joined directly in Semester 3 via Diploma / Lateral Entry")
                email_val = st.text_input("Email Address", value=student["email"] or "")
                phone_val = st.text_input("Phone Number", value=student["phone"] or "")

            sub_prof = st.form_submit_button("💾 Save Personal Details Permanently", type="primary", use_container_width=True)
            if sub_prof:
                if not name_val.strip():
                    st.error("Full Name cannot be empty.")
                else:
                    lat_val = 1 if is_lat_edit else 0
                    db = get_db()
                    db.execute("""
                        UPDATE students
                        SET name = ?, email = ?, phone = ?, admission_year = ?, current_semester = ?, is_lateral_entry = ?
                        WHERE id = ?
                    """, (name_val.strip(), email_val.strip(), phone_val.strip(), adm_year_edit, cur_sem_edit, lat_val, student_id))
                    db.commit()
                    db.close()
                    st.session_state["username"] = name_val.strip()
                    log_audit_action("student", student_id, "UPDATE_PROFILE", f"Updated student profile for {student['usn']} [Name: {name_val.strip()}, Lateral: {lat_val}]")
                    st.success("🎉 Personal details updated and permanently saved in database!")
                    st.rerun()

    # -------------------------------------------------------------
    # 3. ACADEMIC RECORDS
    # -------------------------------------------------------------
    elif selected_page == "Academic Records":
        st.title("📑 Semester Academic Records")
        st.write("Verified historical semester course results, credits earned, and credit-weighted SGPA/CGPA.")

        if is_lat_student:
            st.info("ℹ️ **Lateral Entry Candidate Notice**: Semesters 1 and 2 are exempted under VTU diploma lateral entry regulations. Your CGPA is computed from Semester 3 onward.")

        db = get_db()
        semesters = db.execute("SELECT * FROM semesters WHERE student_id = ? ORDER BY semester_number ASC", (student_id,)).fetchall()

        if semesters:
            sgpa_list = []
            for s_row in semesters:
                s = dict(s_row)
                sem_id = s["id"]
                subjects = db.execute("SELECT * FROM subjects WHERE semester_id = ? ORDER BY subject_code ASC", (sem_id,)).fetchall()

                sgpa_fmt = f"{float(s['sgpa']):.2f}" if s.get("sgpa") is not None and str(s.get("sgpa")).strip() != "" else "N/A"
                with st.expander(f"📘 Semester {s['semester_number']} — SGPA: **{sgpa_fmt}** &bull; Status: `{s.get('verification_status') or 'Verified'}`", expanded=True):
                    if subjects:
                        df_s = pd.DataFrame([dict(sub) for sub in subjects])[["subject_code", "subject_name", "internal_marks", "external_marks", "total_marks", "credits", "result"]]
                        st.dataframe(df_s.rename(columns={
                            "subject_code": "Subject Code", "subject_name": "Subject Name",
                            "internal_marks": "Internal Marks", "external_marks": "External Exam",
                            "total_marks": "Total (100)", "credits": "Credits", "result": "Result"
                        }), use_container_width=True, hide_index=True)
                    if s["sgpa"] and float(s["sgpa"]) > 0:
                        sgpa_list.append(float(s["sgpa"]))

            if sgpa_list:
                cgpa_val = sum(sgpa_list) / len(sgpa_list)
                st.success(f"🏆 **Cumulative Grade Point Average (CGPA):** {cgpa_val:.2f} (Calculated across {len(sgpa_list)} completed semesters)")
        else:
            st.info("No semester academic records found yet. Upload your marksheet under 'Marksheets Upload'.")
        db.close()

    # -------------------------------------------------------------
    # 4. MARKSHEETS UPLOAD (OCR + CONFIRMATION TABLE)
    # -------------------------------------------------------------
    elif selected_page == "Marksheets Upload":
        st.title("📄 Marksheets Upload & OCR Engine")
        st.write("Upload previous semester VTU marksheets (PDF, JPG, PNG). The OCR engine automatically parses subject codes, marks, and credits. **Review, correct, and confirm to save permanently.**")

        sem_num = st.selectbox("Select Marksheet Semester", [1, 2, 3, 4, 5, 6, 7, 8], index=0)
        uploaded_ms = st.file_uploader(f"Upload Semester {sem_num} Marksheet Document (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"], key=f"ms_up_{sem_num}")

        ocr_key = f"ocr_parsed_{student_id}_{sem_num}"
        table_key = f"edit_table_{student_id}_{sem_num}"
        last_file_key = f"last_proc_file_{student_id}_{sem_num}"

        def _clear_ms_keys(snum):
            for k in list(st.session_state.keys()):
                if any(k.startswith(f"{pfx}_{snum}_") for pfx in ["ms_code", "ms_name", "ms_int", "ms_ext", "ms_tot", "ms_res", "ms_crd"]):
                    st.session_state.pop(k, None)

        def _sync_ms_row_state(snum, i):
            code_k = f"ms_code_{snum}_{i}"
            name_k = f"ms_name_{snum}_{i}"
            int_k = f"ms_int_{snum}_{i}"
            ext_k = f"ms_ext_{snum}_{i}"
            tot_k = f"ms_tot_{snum}_{i}"
            res_k = f"ms_res_{snum}_{i}"
            crd_k = f"ms_crd_{snum}_{i}"

            raw_code = str(st.session_state.get(code_k, "")).strip().upper()
            norm_code = normalize_subject_code(raw_code)
            st.session_state[code_k] = norm_code

            if norm_code in VTU_KNOWN_SUBJECTS:
                st.session_state[name_k] = VTU_KNOWN_SUBJECTS[norm_code][0]
                st.session_state[crd_k] = VTU_KNOWN_SUBJECTS[norm_code][1]

            int_val = float(st.session_state.get(int_k, 0.0) or 0.0)
            ext_val = float(st.session_state.get(ext_k, 0.0) or 0.0)

            # VTU Passing & Total Rules:
            # 1. External Exam < 18 is strictly FAIL (F) in VTU
            # 2. Total Aggregate < 40 is strictly FAIL (F)
            if ext_val > 0.0:
                tot_val = round(int_val + ext_val, 2)
                pass_status = (tot_val >= 40.0 and ext_val >= 18.0)
            elif int_val > 50.0 and ext_val == 0.0:
                # 100-mark Continuous Internal Evaluation course (e.g. Yoga/NSS)
                tot_val = int_val
                pass_status = (tot_val >= 40.0)
            else:
                tot_val = round(int_val + ext_val, 2)
                is_non_exam = any(k in norm_code for k in ["PEK", "YOK", "NSS", "AUDIT"])
                pass_status = (tot_val >= 40.0) if is_non_exam else False

            st.session_state[tot_k] = min(200.0, max(0.0, float(tot_val)))
            st.session_state[res_k] = "PASS (P)" if pass_status else "FAIL (F)"

        if uploaded_ms:
            saved_ms_path = save_student_file(uploaded_ms, "marksheets", student_id)
            st.session_state[f"saved_ms_path_{sem_num}"] = saved_ms_path

            up_c1, up_c2 = st.columns([3, 1])
            with up_c2:
                re_extract_btn = st.button("⚡ Re-Scan Marksheet", key=f"btn_rescan_{sem_num}", type="primary", use_container_width=True, help="Force re-scan marksheet document with all OCR engines")

            # Always extract when file is newly uploaded, session missing, or user clicks re-scan
            if (st.session_state.get(last_file_key) != uploaded_ms.name) or (ocr_key not in st.session_state) or re_extract_btn:
                with st.spinner(f"Extracting Semester {sem_num} marksheet via Multi-Tier VTU OCR Engine..."):
                    try:
                        _clear_ms_keys(sem_num)
                        raw_text, detected_subjects, is_ocr = process_marksheet_file(saved_ms_path)
                        st.session_state[ocr_key] = detected_subjects
                        st.session_state[table_key] = [dict(x) for x in detected_subjects]
                        st.session_state[last_file_key] = uploaded_ms.name
                        if detected_subjects:
                            st.success(f"✅ Successfully extracted all **{len(detected_subjects)} subjects** from `{uploaded_ms.name}`!")
                            st.rerun()
                        else:
                            st.warning("⚠️ No subjects auto-detected. You can manually enter subjects below or click 'Add Subject Row'.")
                    except Exception as e:
                        st.error(f"OCR Parser note: {e}. You can manually enter or edit the subjects below.")
                        st.session_state[ocr_key] = []
                        st.session_state[table_key] = []
                        st.session_state[last_file_key] = uploaded_ms.name

        # Load detected OCR rows or load existing records from DB if available
        db = get_db()
        existing_sem_subs = db.execute("""
            SELECT sub.* FROM subjects sub
            JOIN semesters sem ON sub.semester_id = sem.id
            WHERE sem.student_id = ? AND sem.semester_number = ?
            ORDER BY sub.subject_code ASC
        """, (student_id, sem_num)).fetchall()
        db.close()

        if table_key not in st.session_state:
            if ocr_key in st.session_state and st.session_state[ocr_key]:
                st.session_state[table_key] = [dict(x) for x in st.session_state[ocr_key]]
            elif existing_sem_subs:
                st.session_state[table_key] = [{
                    "subject_code": r["subject_code"],
                    "subject_name": r["subject_name"],
                    "internal": float(r["internal_marks"]),
                    "external": float(r["external_marks"]),
                    "total": float(r["total_marks"]),
                    "credits": float(r["credits"]),
                    "result": str(r["result"])
                } for r in existing_sem_subs]
            else:
                st.session_state[table_key] = []

        st.divider()
        st.subheader(f"🔍 Review & Edit Marksheet Confirmation Table (Semester {sem_num})")
        st.caption("Total marks and Pass/Fail results recalculate dynamically in real-time. Any code, name, marks, or credits can be directly edited before saving.")

        btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.4, 1.2])
        if btn_col1.button("➕ Add Another Subject Row", key=f"add_row_ms_{sem_num}", use_container_width=True):
            st.session_state[table_key].append({
                "subject_code": f"SUB{len(st.session_state[table_key])+1}",
                "subject_name": "",
                "internal": 0.0,
                "external": 0.0,
                "total": 0.0,
                "credits": 3.0,
                "result": "P"
            })
            st.rerun()

        if btn_col2.button("🔄 Fresh Re-Scan from Document", key=f"reset_ms_{sem_num}", use_container_width=True):
            _clear_ms_keys(sem_num)
            saved_p = st.session_state.get(f"saved_ms_path_{sem_num}")
            if saved_p and os.path.exists(saved_p):
                with st.spinner("Re-analyzing marksheet document..."):
                    raw_text, detected_subjects, is_ocr = process_marksheet_file(saved_p)
                    st.session_state[ocr_key] = detected_subjects
                    st.session_state[table_key] = [dict(x) for x in detected_subjects]
                    st.rerun()
            elif existing_sem_subs:
                st.session_state[table_key] = [{
                    "subject_code": r["subject_code"],
                    "subject_name": r["subject_name"],
                    "internal": float(r["internal_marks"]),
                    "external": float(r["external_marks"]),
                    "total": float(r["total_marks"]),
                    "credits": float(r["credits"]),
                    "result": str(r["result"])
                } for r in existing_sem_subs]
                st.rerun()

        if btn_col3.button("🗑️ Clear All Table Rows", key=f"clear_ms_{sem_num}", type="secondary", use_container_width=True):
            _clear_ms_keys(sem_num)
            st.session_state[table_key] = []
            st.rerun()

        if not st.session_state[table_key]:
            st.info(f"💡 Upload your **Semester {sem_num} marksheet** document above to auto-extract subjects, or click **➕ Add Another Subject Row** to enter manually.")

        # Table Header Row
        h_code, h_name, h_int, h_ext, h_tot, h_res, h_crd, h_del = st.columns([1.5, 2.7, 0.9, 0.9, 0.9, 1.1, 0.8, 0.9])
        h_code.markdown("**Subject Code**")
        h_name.markdown("**Subject Name**")
        h_int.markdown("**Internal**")
        h_ext.markdown("**External**")
        h_tot.markdown("**Total**")
        h_res.markdown("**Result (P/F)**")
        h_crd.markdown("**Credits**")
        h_del.markdown("**Action**")

        edited_rows = []
        del_row_idx = None
        for idx, item in enumerate(st.session_state[table_key]):
            code_k = f"ms_code_{sem_num}_{idx}"
            name_k = f"ms_name_{sem_num}_{idx}"
            int_k = f"ms_int_{sem_num}_{idx}"
            ext_k = f"ms_ext_{sem_num}_{idx}"
            tot_k = f"ms_tot_{sem_num}_{idx}"
            res_k = f"ms_res_{sem_num}_{idx}"
            crd_k = f"ms_crd_{sem_num}_{idx}"

            # Initialize widget keys from item if first render
            if code_k not in st.session_state:
                st.session_state[code_k] = item.get("subject_code", "")
            if name_k not in st.session_state:
                st.session_state[name_k] = item.get("subject_name", "")
            if int_k not in st.session_state:
                st.session_state[int_k] = min(100.0, max(0.0, float(item.get("internal", 0.0) or 0.0)))
            if ext_k not in st.session_state:
                st.session_state[ext_k] = min(100.0, max(0.0, float(item.get("external", 0.0) or 0.0)))
            if crd_k not in st.session_state:
                st.session_state[crd_k] = min(10.0, max(0.0, float(item.get("credits", 3.0) or 3.0)))

            # Auto calculate current Total & Result
            cur_code = normalize_subject_code(str(st.session_state[code_k]).strip().upper())
            cur_int = float(st.session_state[int_k])
            cur_ext = float(st.session_state[ext_k])

            # VTU Passing & Total Rules:
            # 1. External Exam < 18 is strictly FAIL (F) in VTU
            # 2. Total Aggregate < 40 is strictly FAIL (F)
            if cur_ext > 0.0:
                calc_tot = round(cur_int + cur_ext, 2)
                calc_pass = (calc_tot >= 40.0 and cur_ext >= 18.0)
            elif cur_int > 50.0 and cur_ext == 0.0:
                # 100-mark Continuous Internal Evaluation course (e.g. Yoga/NSS)
                calc_tot = cur_int
                calc_pass = (calc_tot >= 40.0)
            else:
                calc_tot = round(cur_int + cur_ext, 2)
                is_non_exam = any(k in cur_code for k in ["PEK", "YOK", "NSS", "AUDIT"])
                calc_pass = (calc_tot >= 40.0) if is_non_exam else False

            st.session_state[tot_k] = min(200.0, max(0.0, float(calc_tot)))
            st.session_state[res_k] = "PASS (P)" if calc_pass else "FAIL (F)"

            with st.container():
                c_code, c_name, c_int, c_ext, c_tot, c_res, c_crd, c_del = st.columns([1.5, 2.7, 0.9, 0.9, 0.9, 1.1, 0.8, 0.9])
                s_code = c_code.text_input("Code", key=code_k, on_change=_sync_ms_row_state, args=(sem_num, idx), label_visibility="collapsed", placeholder="Code").strip().upper()
                s_name = c_name.text_input("Subject Name", key=name_k, label_visibility="collapsed", placeholder="Subject Name").strip()
                s_int = c_int.number_input("Internal", min_value=0.0, max_value=100.0, step=1.0, key=int_k, on_change=_sync_ms_row_state, args=(sem_num, idx), label_visibility="collapsed")
                s_ext = c_ext.number_input("External", min_value=0.0, max_value=100.0, step=1.0, key=ext_k, on_change=_sync_ms_row_state, args=(sem_num, idx), label_visibility="collapsed")
                s_tot = c_tot.number_input("Total", min_value=0.0, max_value=200.0, step=1.0, key=tot_k, label_visibility="collapsed", help="Combined Internal + External Marks")
                res_choice = c_res.selectbox("Result", ["PASS (P)", "FAIL (F)"], key=res_k, label_visibility="collapsed")
                s_res = "P" if "PASS" in res_choice else "F"
                s_crd = c_crd.number_input("Credits", min_value=0.0, max_value=10.0, step=0.5, key=crd_k, label_visibility="collapsed")

                if c_del.button("🗑️ Delete", key=f"del_ms_row_{sem_num}_{idx}", type="secondary", help="Remove this subject row", use_container_width=True):
                    del_row_idx = idx

                edited_rows.append({
                    "subject_code": s_code,
                    "subject_name": s_name,
                    "internal_marks": s_int,
                    "external_marks": s_ext,
                    "total_marks": s_tot,
                    "credits": s_crd,
                    "result": s_res
                })

        if del_row_idx is not None:
            _clear_ms_keys(sem_num)
            st.session_state[table_key].pop(del_row_idx)
            st.rerun()


        # Calculate SGPA & Result Preview (VTU Standard: Failed subjects earn 0 grade points)
        tot_creds = sum(r["credits"] for r in edited_rows if r["subject_code"])
        weighted_pts = 0.0
        for r in edited_rows:
            if not r["subject_code"]:
                continue
            c = float(r["credits"] or 0.0)
            t_m = float(r["total_marks"] or 0.0)
            res_val = str(r["result"]).upper()
            if "F" in res_val or t_m < 40.0:
                gp_val = 0
            else:
                gp_val = grade_point(t_m)
            weighted_pts += gp_val * c

        calc_sgpa = round(weighted_pts / tot_creds, 2) if tot_creds > 0 else 0.0
        
        pass_count = sum(1 for r in edited_rows if r["subject_code"] and r["result"] == "P")
        fail_count = sum(1 for r in edited_rows if r["subject_code"] and r["result"] == "F")
        sem_res_badge = "🟢 ALL SUBJECTS PASSED" if (fail_count == 0 and len(edited_rows) > 0) else f"🔴 {fail_count} FAILED / BACKLOG"

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        c_res1, c_res2, c_res3 = st.columns([1.4, 2.0, 1.4])
        c_res1.markdown(f"""
        <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="color: #64748B; font-size: 12px; font-weight: 700; text-transform: uppercase;">Total Evaluated Credits</div>
            <div style="color: #0284C7; font-weight: 800; font-size: 24px; margin-top: 2px;">{tot_creds:g}</div>
        </div>
        """, unsafe_allow_html=True)

        c_res2.markdown(f"""
        <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="color: #64748B; font-size: 12px; font-weight: 700; text-transform: uppercase;">Semester Result Status</div>
            <div style="font-weight: 800; font-size: 15px; margin-top: 5px; color: {'#059669' if fail_count == 0 else '#DC2626'};">{sem_res_badge} ({pass_count}/{len(edited_rows)} Passed)</div>
        </div>
        """, unsafe_allow_html=True)

        # Track previous calculated SGPA so when credits/marks change, number_input updates dynamically
        prev_calc_key = f"prev_calc_sgpa_{student_id}_{sem_num}"
        curr_sgpa_key = f"sgpa_inp_{sem_num}"
        if prev_calc_key not in st.session_state or st.session_state[prev_calc_key] != calc_sgpa:
            st.session_state[prev_calc_key] = calc_sgpa
            st.session_state[curr_sgpa_key] = float(calc_sgpa)

        safe_sgpa_val = min(10.0, max(0.0, float(calc_sgpa or 0.0)))
        final_sgpa = c_res3.number_input("Official SGPA (2 Decimals)", min_value=0.0, max_value=10.0, value=safe_sgpa_val, step=0.01, format="%.2f", key=curr_sgpa_key, help="Calculated automatically from credits and grades. You can also fine-tune or enter the exact 2-decimal SGPA.")

        if st.button("💾 Confirm & Permanently Save Marksheet to Database", type="primary", use_container_width=True, key=f"save_ms_btn_{sem_num}"):
            valid_subs = [r for r in edited_rows if r["subject_code"] and r["subject_name"]]
            if not valid_subs:
                st.error("Please enter at least one subject with code and name.")
            else:
                db = get_db()
                # Enforce Semester and Class Locking
                cls_id = (student.get("class_id") if isinstance(student, dict) else (student["class_id"] if "class_id" in student.keys() else None)) if student else None
                cls_row = db.execute("SELECT is_locked FROM classes WHERE id = ?", (cls_id,)).fetchone() if cls_id else None
                sem_row = db.execute("SELECT is_locked FROM semesters WHERE student_id = ? AND semester_number = ?", (student_id, sem_num)).fetchone()
                
                is_cls_locked = bool(cls_row and (cls_row["is_locked"] == 1 if "is_locked" in cls_row.keys() else dict(cls_row).get("is_locked") == 1))
                is_sem_locked = bool(sem_row and (sem_row["is_locked"] == 1 if "is_locked" in sem_row.keys() else dict(sem_row).get("is_locked") == 1))
                
                if is_cls_locked or is_sem_locked:
                    db.close()
                    st.error("🔒 This semester is locked by college administration. Changes cannot be submitted.")
                    return

                now = datetime.now().isoformat()
                saved_path = st.session_state.get(f"saved_ms_path_{sem_num}", "")
                saved_sgpa = round(float(final_sgpa), 2)

                existing_sem = db.execute("SELECT id FROM semesters WHERE student_id = ? AND semester_number = ?", (student_id, sem_num)).fetchone()
                if existing_sem:
                    sem_id = existing_sem["id"]
                    db.execute("DELETE FROM subjects WHERE semester_id = ?", (sem_id,))
                    db.execute("""
                        UPDATE semesters SET semester_name = ?, sgpa = ?, marksheet_path = ?, credits_total = ?, verification_status = 'Pending'
                        WHERE id = ?
                    """, (f"Semester {sem_num}", saved_sgpa, saved_path, tot_creds, sem_id))
                else:
                    sem_id = uuid.uuid4().hex
                    db.execute("""
                        INSERT INTO semesters (id, student_id, semester_number, semester_name, sgpa, marksheet_path, is_current, credits_total, verification_status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, 0, ?, 'Pending', ?)
                    """, (sem_id, student_id, sem_num, f"Semester {sem_num}", saved_sgpa, saved_path, tot_creds, now))

                for sub in valid_subs:
                    sub_res = str(sub["result"]).upper()
                    sub_tot = float(sub["total_marks"] or 0.0)
                    if "F" in sub_res or sub_tot < 40.0:
                        sub_gp = 0
                        sub_gr = "F"
                    else:
                        sub_gp = grade_point(sub_tot)
                        sub_gr = grade_letter(sub_tot)

                    db.execute("""
                        INSERT INTO subjects (id, semester_id, subject_code, subject_name, internal_marks, external_marks, total_marks, credits, result, grade, grade_point)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (uuid.uuid4().hex, sem_id, sub["subject_code"], sub["subject_name"], sub["internal_marks"], sub["external_marks"], sub["total_marks"], sub["credits"], sub["result"], sub_gr, sub_gp))

                db.commit()
                db.close()
                log_audit_action("student", student_id, "SAVE_MARKSHEET", f"Saved Semester {sem_num} marksheet (SGPA {saved_sgpa:.2f})")
                st.success(f"🎉 Semester {sem_num} marksheet permanently saved with Official SGPA {saved_sgpa:.2f}!")
                st.session_state.pop(ocr_key, None)
                st.session_state.pop(table_key, None)
                st.rerun()

        # Database Management / Delete Semester Records Option
        if existing_sem_subs:
            st.markdown("---")
            with st.expander(f"⚙️ Manage Saved Semester {sem_num} Database Records ({len(existing_sem_subs)} Subjects Recorded)", expanded=False):
                st.warning(f"This semester is currently stored in the database with SGPA records. If you need to completely remove it and re-upload, use the option below.")
                if st.button(f"🗑️ Delete Semester {sem_num} Records & Marksheet Permanently", type="secondary", key=f"del_db_sem_btn_{sem_num}", use_container_width=True):
                    db = get_db()
                    sem_rec = db.execute("SELECT id FROM semesters WHERE student_id = ? AND semester_number = ?", (student_id, sem_num)).fetchone()
                    if sem_rec:
                        db.execute("DELETE FROM subjects WHERE semester_id = ?", (sem_rec["id"],))
                        db.execute("DELETE FROM semesters WHERE id = ?", (sem_rec["id"],))
                        db.commit()
                    db.close()
                    st.session_state.pop(table_key, None)
                    st.session_state.pop(ocr_key, None)
                    log_audit_action("student", student_id, "DELETE_SEMESTER", f"Deleted Semester {sem_num} records")
                    st.success(f"Semester {sem_num} records deleted successfully from database.")
                    st.rerun()

    # -------------------------------------------------------------
    # 5. CURRENT SEMESTER (UNLIMITED DYNAMIC SUBJECTS)
    # -------------------------------------------------------------
    elif selected_page == "Current Semester":
        st.title("📘 Current Semester IA Marks Entry")
        st.write("Manage dynamic current-semester subjects, enter IA1 & IA2 marks, and automatically generate predictions.")

        cur_sem = int(student["current_semester"] or 5)
        st.subheader(f"Active Semester: **Semester {cur_sem}**")

        db = get_db()
        existing_records = db.execute("SELECT * FROM current_semester_records WHERE student_id = ? AND semester_number = ? ORDER BY subject_code ASC", (student_id, cur_sem)).fetchall()
        master_subs = db.execute("SELECT * FROM master_subjects WHERE semester_number = ? AND (class_id = ? OR class_id IS NULL)", (cur_sem, student["class_id"])).fetchall()
        db.close()

        state_key = f"cur_subs_{student_id}_{cur_sem}"
        if state_key not in st.session_state:
            if existing_records:
                st.session_state[state_key] = [dict(r) for r in existing_records]
            elif master_subs:
                st.session_state[state_key] = [{"subject_code": m["subject_code"], "subject_name": m["subject_name"], "credits": m["credits"], "ia1": 0.0, "ia2": 0.0} for m in master_subs]
            else:
                st.session_state[state_key] = [
                    {"subject_code": "21CS51", "subject_name": "Database Management Systems", "credits": 4.0, "ia1": 0.0, "ia2": 0.0},
                    {"subject_code": "21CS52", "subject_name": "Computer Networks", "credits": 4.0, "ia1": 0.0, "ia2": 0.0},
                    {"subject_code": "21CS53", "subject_name": "Software Engineering", "credits": 3.0, "ia1": 0.0, "ia2": 0.0}
                ]

        btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1.2, 1.2])
        if btn_c1.button("➕ Add Another Subject", use_container_width=True):
            st.session_state[state_key].append({
                "subject_code": f"SUB{len(st.session_state[state_key])+1}",
                "subject_name": "",
                "credits": 3.0,
                "ia1": 0.0,
                "ia2": 0.0
            })
            st.rerun()

        if master_subs and btn_c2.button("📥 Load Prescribed Class Subjects", use_container_width=True):
            st.session_state[state_key] = [{"subject_code": m["subject_code"], "subject_name": m["subject_name"], "credits": m["credits"], "ia1": 0.0, "ia2": 0.0} for m in master_subs]
            st.rerun()

        if btn_c3.button("🗑️ Clear All Rows", key=f"clear_cur_{cur_sem}", type="secondary", use_container_width=True):
            st.session_state[state_key] = []
            st.rerun()

        st.divider()

        # Header Row
        sh_code, sh_name, sh_cred, sh_ia1, sh_ia2, sh_avg, sh_del = st.columns([1.6, 2.8, 0.9, 0.9, 0.9, 1.0, 0.9])
        sh_code.markdown("**Subject Code**")
        sh_name.markdown("**Subject Name**")
        sh_cred.markdown("**Credits**")
        sh_ia1.markdown("**IA1 (50)**")
        sh_ia2.markdown("**IA2 (50)**")
        sh_avg.markdown("**IA Avg (50)**")
        sh_del.markdown("**Action**")

        cur_edited = []
        rows_to_del = []
        for idx, sub in enumerate(st.session_state[state_key]):
            with st.container():
                sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns([1.6, 2.8, 0.9, 0.9, 0.9, 1.0, 0.9])
                code_v = sc1.text_input("Code", value=sub.get("subject_code", ""), key=f"cur_code_{idx}", label_visibility="collapsed", placeholder="Code").strip().upper()
                name_v = sc2.text_input("Subject Name", value=sub.get("subject_name", ""), key=f"cur_name_{idx}", label_visibility="collapsed", placeholder="Subject Name").strip()
                cred_val = min(10.0, max(0.0, float(sub.get("credits", 3.0) or 3.0)))
                ia1_val = min(50.0, max(0.0, float(sub.get("ia1", 0.0) or 0.0)))
                ia2_val = min(50.0, max(0.0, float(sub.get("ia2", 0.0) or 0.0)))
                cred_v = sc3.number_input("Credits", min_value=0.0, max_value=10.0, value=cred_val, step=0.5, key=f"cur_cred_{idx}", label_visibility="collapsed")
                ia1_v = sc4.number_input("IA1 (50)", min_value=0.0, max_value=50.0, value=ia1_val, step=0.5, key=f"cur_ia1_{idx}", label_visibility="collapsed")
                ia2_v = sc5.number_input("IA2 (50)", min_value=0.0, max_value=50.0, value=ia2_val, step=0.5, key=f"cur_ia2_{idx}", label_visibility="collapsed")

                int_perf = round((ia1_v + ia2_v) / 2.0, 2)
                sc6.markdown(f"<div style='background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 7px 10px; text-align: center; font-weight: 700; color: #1E40AF; font-size: 14px;'>{int_perf:.1f}</div>", unsafe_allow_html=True)

                if sc7.button("🗑️ Delete", key=f"del_cur_row_{idx}", type="secondary", help="Remove this subject", use_container_width=True):
                    rows_to_del.append(idx)

                cur_edited.append({
                    "subject_code": code_v,
                    "subject_name": name_v,
                    "credits": cred_v,
                    "ia1": ia1_v,
                    "ia2": ia2_v,
                    "internal_performance": int_perf
                })

        if rows_to_del:
            for d in reversed(rows_to_del):
                st.session_state[state_key].pop(d)
            st.rerun()

        st.divider()
        if st.button("💾 Permanently Save Current Semester IA Marks", type="primary", use_container_width=True):
            valid_cur = [c for c in cur_edited if c["subject_code"] and c["subject_name"]]
            if not valid_cur:
                st.error("Please enter at least one subject with code and name.")
            else:
                db = get_db()
                cls_id = (student.get("class_id") if isinstance(student, dict) else (student["class_id"] if "class_id" in student.keys() else None)) if student else None
                cls_row = db.execute("SELECT is_locked FROM classes WHERE id = ?", (cls_id,)).fetchone() if cls_id else None
                if cls_row and (cls_row["is_locked"] == 1 if "is_locked" in cls_row.keys() else dict(cls_row).get("is_locked") == 1):
                    db.close()
                    st.error("🔒 This academic semester is locked by college administration. Updates are disabled.")
                    return

                now = datetime.now().isoformat()
                saved_codes = [c["subject_code"] for c in valid_cur]
                
                # Delete removed codes
                placeholders = ','.join(['?'] * len(saved_codes))
                db.execute(f"DELETE FROM current_semester_records WHERE student_id = ? AND semester_number = ? AND subject_code NOT IN ({placeholders})", [student_id, cur_sem] + saved_codes)

                for c in valid_cur:
                    db.execute("""
                        INSERT INTO current_semester_records (id, student_id, semester_number, subject_code, subject_name, credits, ia1, ia2, internal_performance, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(student_id, semester_number, subject_code)
                        DO UPDATE SET subject_name = excluded.subject_name, credits = excluded.credits, ia1 = excluded.ia1, ia2 = excluded.ia2, internal_performance = excluded.internal_performance, updated_at = excluded.updated_at
                    """, (uuid.uuid4().hex, student_id, cur_sem, c["subject_code"], c["subject_name"], c["credits"], c["ia1"], c["ia2"], c["internal_performance"], now))

                db.commit()
                db.close()
                st.session_state[state_key] = valid_cur
                log_audit_action("student", student_id, "SAVE_CURRENT_MARKS", f"Saved {len(valid_cur)} subjects for Semester {cur_sem}")
                st.success("🎉 Current semester IA marks saved permanently! Head to 'AI Prediction & Focus' to see your predicted VTU marks.")
                st.rerun()

    # -------------------------------------------------------------
    # 6. ATTENDANCE
    # -------------------------------------------------------------
    elif selected_page == "Attendance":
        st.title("⏱️ My Class Attendance Log")
        st.write("View-only real-time institutional attendance records logged by faculty.")

        db = get_db()
        att_rows = db.execute("SELECT * FROM attendance WHERE student_id = ? ORDER BY attendance_date DESC, hour_number ASC", (student_id,)).fetchall()
        sub_att = db.execute("""
            SELECT subject_name, COUNT(*) as total_hours,
                   SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_hours
            FROM attendance WHERE student_id = ?
            GROUP BY subject_name
        """, (student_id,)).fetchall()
        db.close()

        if sub_att:
            st.subheader("Subject-wise Attendance Breakdown")
            df_sub = []
            for r in sub_att:
                tot = r["total_hours"]
                pres = r["present_hours"]
                pct = round(pres / tot * 100, 1) if tot else 0.0
                status_icon = "🚨 Critical (<75%)" if pct < 75.0 else ("⚠️ Caution" if pct < 85.0 else "✅ Good")
                df_sub.append({
                    "Subject": r["subject_name"],
                    "Total Hours": tot,
                    "Present Hours": pres,
                    "Attendance %": pct,
                    "Status": status_icon
                })
            st.dataframe(pd.DataFrame(df_sub), use_container_width=True, hide_index=True)

        if att_rows:
            st.divider()
            st.subheader("Daily Hourly Attendance History")
            st.dataframe(pd.DataFrame([dict(r) for r in att_rows])[["attendance_date", "subject_name", "hour_number", "status"]].rename(
                columns={"attendance_date": "Date", "subject_name": "Subject", "hour_number": "Hour / Period", "status": "Attendance"}
            ), use_container_width=True, hide_index=True)
        else:
            st.info("No hourly attendance has been logged by faculty for your account yet.")

    # -------------------------------------------------------------
    # 7. BACKLOGS
    # -------------------------------------------------------------
    elif selected_page == "Backlogs":
        st.title("⚠️ Backlog Management & Remediation")
        st.write("Track active and cleared backlogs, rewritten examination attempts, and update cleared status.")

        tab_view, tab_add = st.tabs(["📋 My Backlogs", "➕ Record / Add Backlog"])

        with tab_add:
            with st.form("add_backlog_form", clear_on_submit=True):
                st.subheader("Add Backlog Record")
                b_sem = st.number_input("Semester", min_value=1, max_value=8, value=1)
                b_code = st.text_input("Subject Code (e.g. 21MAT11) *").strip().upper()
                b_name = st.text_input("Subject Name *").strip()
                b_crd = st.number_input("Credits", min_value=0.0, max_value=10.0, value=4.0, step=0.5)
                b_status = st.selectbox("Status", ["Active", "Cleared"])
                b_file = st.file_uploader("Upload Re-evaluation / Result Sheet", type=["pdf", "jpg", "png"])
                b_sub = st.form_submit_button("Save Backlog Entry Permanently", type="primary")

                if b_sub:
                    if not b_code or not b_name:
                        st.error("Subject Code and Name are required.")
                    else:
                        file_p = save_student_file(b_file, "backlogs", student_id) if b_file else None
                        db = get_db()
                        db.execute("""
                            INSERT INTO backlogs (id, student_id, semester_number, subject_code, subject_name, credits, status, file_path, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (uuid.uuid4().hex, student_id, b_sem, b_code, b_name, b_crd, b_status, file_p, datetime.now().isoformat()))
                        db.commit()
                        db.close()
                        st.success("🎉 Backlog record permanently saved!")
                        st.rerun()

        with tab_view:
            db = get_db()
            backlogs = db.execute("SELECT * FROM backlogs WHERE student_id = ? ORDER BY semester_number ASC", (student_id,)).fetchall()
            db.close()

            if backlogs:
                for b in backlogs:
                    status_badge = "🔴 Active Backlog" if b["status"] == "Active" else "🟢 Cleared"
                    with st.expander(f"⚠️ {b['subject_code']} - {b['subject_name']} (Sem {b['semester_number']}) — {status_badge}", expanded=True):
                        st.write(f"**Credits:** {b['credits']}")
                        st.write(f"**Current Status:** `{b['status']}`")
                        new_stat = st.selectbox(f"Update Status for {b['subject_code']}", ["Active", "Cleared"], index=0 if b["status"] == "Active" else 1, key=f"bstat_{b['id']}")
                        if st.button("💾 Save Status Change", key=f"btn_b_{b['id']}"):
                            db = get_db()
                            db.execute("UPDATE backlogs SET status = ?, updated_at = ? WHERE id = ?", (new_stat, datetime.now().isoformat(), b["id"]))
                            db.commit()
                            db.close()
                            st.success("Backlog status permanently updated!")
                            st.rerun()
            else:
                st.success("🎉 No active or historical backlogs recorded! Keep up the great academic work.")

    # -------------------------------------------------------------
    # 8. SKILLS & PORTFOLIO
    # -------------------------------------------------------------
    elif selected_page == "Skills & Portfolio":
        st.title("💼 Skills & Portfolio")
        st.write("Showcase your technical skills, programming languages, and industry certifications.")

        tab_skills, tab_certs = st.tabs(["💡 Technical Skills", "📜 Certifications"])

        with tab_skills:
            with st.form("add_skill_form", clear_on_submit=True):
                st.subheader("Add New Skill")
                c1, c2, c3 = st.columns(3)
                s_name = c1.text_input("Skill Name (e.g. Python, SQL, React) *")
                s_cat = c2.selectbox("Category", ["Programming Language", "Database / SQL", "Web Development", "Machine Learning / AI", "Cloud / DevOps", "Core Engineering", "Soft Skills"])
                s_prof = c3.selectbox("Proficiency", ["Beginner", "Intermediate", "Advanced", "Expert"])
                s_desc = st.text_input("Brief Project Application / Experience")
                add_sk = st.form_submit_button("Save Skill Permanently", type="primary")

                if add_sk:
                    if not s_name.strip():
                        st.error("Skill name is required.")
                    else:
                        db = get_db()
                        try:
                            db.execute("""
                                INSERT INTO student_skills (id, student_id, skill_name, category, proficiency, description, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (uuid.uuid4().hex, student_id, s_name.strip(), s_cat, s_prof, s_desc.strip(), datetime.now().isoformat()))
                            db.commit()
                            st.success(f"🎉 Skill '{s_name}' permanently added!")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error(f"Skill '{s_name}' is already listed.")
                        finally:
                            db.close()

            db = get_db()
            skills = db.execute("SELECT * FROM student_skills WHERE student_id = ? ORDER BY category ASC, skill_name ASC", (student_id,)).fetchall()
            db.close()

            if skills:
                st.subheader(f"My Skills Directory ({len(skills)} Total)")
                st.dataframe(pd.DataFrame([dict(s) for s in skills])[["skill_name", "category", "proficiency", "description"]].rename(
                    columns={"skill_name": "Skill", "category": "Category", "proficiency": "Proficiency", "description": "Details"}
                ), use_container_width=True, hide_index=True)
            else:
                st.info("No skills added yet.")

        with tab_certs:
            with st.form("upload_cert_form", clear_on_submit=True):
                st.subheader("Upload Certification")
                c1, c2 = st.columns(2)
                c_name = c1.text_input("Certificate Name *")
                c_cat = c1.selectbox("Category", ["Technical", "Online Course", "Workshop", "Internship", "Hackathon", "Academic Excellence"])
                c_org = c2.text_input("Issuing Organization (e.g. Coursera, AWS, NPTEL, TJIT)")
                c_date = c2.date_input("Issue Date", datetime.now()).strftime("%Y-%m-%d")
                c_file = st.file_uploader("Certificate File (PDF, JPG, PNG) *", type=["pdf", "jpg", "jpeg", "png"])
                c_sub = st.form_submit_button("Submit Certificate Permanently", type="primary")

                if c_sub:
                    if not c_name.strip() or not c_file:
                        st.error("Certificate Name and File are required.")
                    else:
                        file_p = save_student_file(c_file, "certificates", student_id)
                        db = get_db()
                        db.execute("""
                            INSERT INTO certificates (id, student_id, certificate_name, category, organization, issue_date, file_path, uploaded_at, verification_status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending')
                        """, (uuid.uuid4().hex, student_id, c_name.strip(), c_cat, c_org.strip(), c_date, file_p, datetime.now().isoformat()))
                        db.commit()
                        db.close()
                        st.success(f"🎉 Certificate '{c_name}' uploaded and permanently saved for faculty verification!")
                        st.rerun()

            db = get_db()
            certs = db.execute("SELECT * FROM certificates WHERE student_id = ? ORDER BY uploaded_at DESC", (student_id,)).fetchall()
            db.close()
            if certs:
                st.subheader("Submitted Certificates")
                st.dataframe(pd.DataFrame([dict(c) for c in certs])[["certificate_name", "category", "organization", "issue_date", "verification_status", "staff_comment"]].rename(
                    columns={"certificate_name": "Certificate", "category": "Category", "organization": "Issuer", "issue_date": "Date", "verification_status": "Status", "staff_comment": "Faculty Remark"}
                ), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 9. EXTRACURRICULARS & LINKEDIN
    # -------------------------------------------------------------
    elif selected_page == "Extracurriculars & LinkedIn":
        st.title("🌟 Extracurriculars & Professional Links")
        st.write("Record extracurricular achievements, sports, cultural events, and your LinkedIn profile.")

        tab_lk, tab_extra = st.tabs(["🔗 LinkedIn Profile", "🏆 Extracurricular Activities"])

        with tab_lk:
            db = get_db()
            lk_row = db.execute("SELECT linkedin_url FROM linkedin_profiles WHERE student_id = ?", (student_id,)).fetchone()
            db.close()
            cur_lk = lk_row["linkedin_url"] if lk_row else ""

            with st.form("linkedin_form"):
                st.subheader("Update LinkedIn URL")
                lk_input = st.text_input("LinkedIn Profile URL", value=cur_lk, placeholder="https://www.linkedin.com/in/username")
                sub_lk = st.form_submit_button("Save LinkedIn Link Permanently", type="primary")

                if sub_lk:
                    if lk_input.strip() and not re.match(r"^https?://(www\.)?linkedin\.com/.*", lk_input.strip()):
                        st.error("Please enter a valid LinkedIn URL starting with https://linkedin.com or https://www.linkedin.com")
                    else:
                        db = get_db()
                        db.execute("""
                            INSERT INTO linkedin_profiles (id, student_id, linkedin_url, updated_at)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(student_id) DO UPDATE SET linkedin_url = excluded.linkedin_url, updated_at = excluded.updated_at
                        """, (uuid.uuid4().hex, student_id, lk_input.strip(), datetime.now().isoformat()))
                        db.commit()
                        db.close()
                        st.success("🎉 LinkedIn profile permanently updated!")
                        st.rerun()

            if cur_lk:
                st.markdown(f"🔗 **Clickable Profile:** [{cur_lk}]({cur_lk})")

        with tab_extra:
            with st.form("add_extra_form", clear_on_submit=True):
                st.subheader("Add Extracurricular Activity")
                c1, c2 = st.columns(2)
                a_name = c1.text_input("Activity / Event Name *")
                a_cat = c1.selectbox("Category", ["Sports", "Cultural", "Robotics Club", "IEEE / Technical Society", "Volunteering / NSS", "Leadership", "Other"])
                a_org = c2.text_input("Organization / Venue")
                a_date = c2.date_input("Date", datetime.now()).strftime("%Y-%m-%d")
                a_ach = st.text_input("Achievement / Role (e.g. Winner, Organizer, Captain)")
                a_desc = st.text_area("Description")
                a_sub = st.form_submit_button("Save Activity Permanently", type="primary")

                if a_sub:
                    if not a_name.strip():
                        st.error("Activity name is required.")
                    else:
                        db = get_db()
                        db.execute("""
                            INSERT INTO extracurricular_activities (id, student_id, activity_name, category, organization_event, date, achievement, description, uploaded_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (uuid.uuid4().hex, student_id, a_name.strip(), a_cat, a_org.strip(), a_date, a_ach.strip(), a_desc.strip(), datetime.now().isoformat()))
                        db.commit()
                        db.close()
                        st.success("🎉 Extracurricular activity permanently saved!")
                        st.rerun()

            db = get_db()
            extras = db.execute("SELECT * FROM extracurricular_activities WHERE student_id = ? ORDER BY date DESC", (student_id,)).fetchall()
            db.close()
            if extras:
                st.dataframe(pd.DataFrame([dict(e) for e in extras])[["activity_name", "category", "organization_event", "date", "achievement", "description"]], use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 10. FEE RECEIPTS & NON-DUE
    # -------------------------------------------------------------
    elif selected_page == "Fee Receipts & Non-Due":
        st.title("🧾 Fee Receipts & Non-Due Clearance")
        st.write("Upload fee payment receipts and semester clearance non-due documents.")

        tab_fees, tab_nondue = st.tabs(["🧾 Fee Receipts", "📄 Non-Due Form"])

        with tab_fees:
            with st.form("fee_form", clear_on_submit=True):
                st.subheader("Upload Fee Receipt")
                c1, c2 = st.columns(2)
                f_yr = c1.text_input("Academic Year (e.g. 2024-2025)", value="2024-2025")
                f_sem = c1.number_input("Semester", min_value=1, max_value=8, value=int(student["current_semester"] or 5))
                f_num = c2.text_input("Receipt Number *")
                f_amt = c2.number_input("Amount Paid (₹) *", min_value=0.0, value=50000.0, step=1000.0)
                f_file = st.file_uploader("Upload Receipt File (PDF, JPG, PNG) *", type=["pdf", "jpg", "png"])
                sub_f = st.form_submit_button("Submit Fee Receipt Permanently", type="primary")

                if sub_f:
                    if not f_num.strip() or not f_file:
                        st.error("Receipt Number and File are required.")
                    else:
                        file_p = save_student_file(f_file, "fee_receipts", student_id)
                        db = get_db()
                        db.execute("""
                            INSERT INTO fee_receipts (id, student_id, academic_year, semester, receipt_number, amount, receipt_path, uploaded_at, verification_status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending')
                        """, (uuid.uuid4().hex, student_id, f_yr.strip(), f_sem, f_num.strip(), f_amt, file_p, datetime.now().isoformat()))
                        db.commit()
                        db.close()
                        st.success("🎉 Fee receipt submitted and saved permanently!")
                        st.rerun()

            db = get_db()
            fees = db.execute("SELECT * FROM fee_receipts WHERE student_id = ? ORDER BY uploaded_at DESC", (student_id,)).fetchall()
            db.close()
            if fees:
                st.dataframe(pd.DataFrame([dict(f) for f in fees])[["receipt_number", "academic_year", "semester", "amount", "verification_status", "staff_comment"]], use_container_width=True, hide_index=True)

        with tab_nondue:
            with st.form("nondue_form", clear_on_submit=True):
                st.subheader("Upload Non-Due Clearance Form")
                nd_yr = st.text_input("Academic Year", value="2024-2025")
                nd_sem = st.number_input("Semester Clearance", min_value=1, max_value=8, value=int(student["current_semester"] or 5))
                nd_file = st.file_uploader("Non-Due Document (PDF, JPG, PNG) *", type=["pdf", "jpg", "png"])
                sub_nd = st.form_submit_button("Submit Non-Due Form Permanently", type="primary")

                if sub_nd:
                    if not nd_file:
                        st.error("Document file is required.")
                    else:
                        file_p = save_student_file(nd_file, "non_due", student_id)
                        db = get_db()
                        db.execute("""
                            INSERT INTO non_due_documents (id, student_id, academic_year, semester, file_path, uploaded_at, verification_status)
                            VALUES (?, ?, ?, ?, ?, ?, 'Pending')
                        """, (uuid.uuid4().hex, student_id, nd_yr.strip(), nd_sem, file_p, datetime.now().isoformat()))
                        db.commit()
                        db.close()
                        st.success("🎉 Non-due clearance form submitted and saved permanently!")
                        st.rerun()

            db = get_db()
            nd_rows = db.execute("SELECT * FROM non_due_documents WHERE student_id = ? ORDER BY uploaded_at DESC", (student_id,)).fetchall()
            db.close()
            if nd_rows:
                st.dataframe(pd.DataFrame([dict(n) for n in nd_rows])[["semester", "academic_year", "uploaded_at", "verification_status", "staff_comment"]], use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 11. SUBMIT LETTER / APPLICATIONS
    # -------------------------------------------------------------
    elif selected_page == "Submit Letter / Applications":
        st.title("📩 Student Applications & Letters")
        st.write("Submit formal applications, leave requests, bonafide requests, and permission letters to your Class Teacher.")

        with st.form("submit_letter_form", clear_on_submit=True):
            st.subheader("Write Application to Class Teacher")
            c1, c2 = st.columns(2)
            l_type = c1.selectbox("Application Type", ["Leave Request", "Bonafide Certificate Request", "Internal Assessment Re-test Request", "Event Attendance Permission", "General Grievance / Application"])
            l_sub = c2.text_input("Subject *")
            l_body = st.text_area("Application Content / Details *", height=150)
            l_file = st.file_uploader("Supporting Document (Optional)", type=["pdf", "jpg", "png"])
            sub_let = st.form_submit_button("Submit Application Permanently", type="primary")

            if sub_let:
                if not l_sub.strip() or not l_body.strip():
                    st.error("Subject and application content are required.")
                else:
                    file_p = save_student_file(l_file, "applications", student_id) if l_file else None
                    db = get_db()
                    db.execute("""
                        INSERT INTO student_letters (id, student_id, recipient, subject, letter_text, attachment_path, submission_date, status, application_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'Pending', ?)
                    """, (uuid.uuid4().hex, student_id, student["class_teacher"] or "Class Teacher", l_sub.strip(), l_body.strip(), file_p, datetime.now().isoformat(), l_type))
                    db.commit()
                    db.close()
                    st.success("🎉 Application submitted and saved permanently to your Class Teacher!")
                    st.rerun()

        st.divider()
        st.subheader("My Previous Applications")
        db = get_db()
        letters = db.execute("SELECT * FROM student_letters WHERE student_id = ? ORDER BY submission_date DESC", (student_id,)).fetchall()
        db.close()

        if letters:
            for l_row in letters:
                l = dict(l_row)
                status_badge = "🟡 Pending" if l["status"] in ["Pending", "Submitted"] else ("🟢 Approved" if l["status"] == "Approved" else "🔴 Rejected")
                with st.expander(f"📩 {l['subject']} ({l.get('application_type') or 'Letter'}) — {status_badge} ({l['submission_date'][:10]})"):
                    st.write(f"**Content:**\n\n{l['letter_text']}")
                    if l["staff_comment"]:
                        st.info(f"**Faculty Remark:** {l['staff_comment']}")
        else:
            st.info("No previous applications submitted.")

    # -------------------------------------------------------------
    # 12. DOCUMENT VERIFICATION STATUS
    # -------------------------------------------------------------
    elif selected_page == "Document Verification Status":
        st.title("🔍 360° Document Verification Tracking")
        st.write("Live status of all academic marksheets, certificates, fee receipts, and non-due forms submitted for faculty review.")

        db = get_db()
        c_rows = db.execute("SELECT certificate_name AS name, 'Certificate' AS type, verification_status AS status, staff_comment AS comment, uploaded_at FROM certificates WHERE student_id = ?", (student_id,)).fetchall()
        f_rows = db.execute("SELECT 'Receipt #' || receipt_number AS name, 'Fee Receipt' AS type, verification_status AS status, staff_comment AS comment, uploaded_at FROM fee_receipts WHERE student_id = ?", (student_id,)).fetchall()
        n_rows = db.execute("SELECT 'Semester ' || semester || ' Non-Due' AS name, 'Non-Due Form' AS type, verification_status AS status, staff_comment AS comment, uploaded_at FROM non_due_documents WHERE student_id = ?", (student_id,)).fetchall()
        m_rows = db.execute("SELECT 'Semester ' || semester_number || ' Marksheet' AS name, 'Marksheet' AS type, verification_status AS status, staff_remarks AS comment, created_at AS uploaded_at FROM semesters WHERE student_id = ?", (student_id,)).fetchall()
        db.close()

        all_docs = [dict(r) for r in (list(c_rows) + list(f_rows) + list(n_rows) + list(m_rows))]
        if all_docs:
            st.dataframe(pd.DataFrame(all_docs)[["name", "type", "status", "comment", "uploaded_at"]].rename(
                columns={"name": "Document Name", "type": "Category", "status": "Verification Status", "comment": "Faculty Remark", "uploaded_at": "Date Uploaded"}
            ), use_container_width=True, hide_index=True)
        else:
            st.info("No documents uploaded for verification yet.")

    # -------------------------------------------------------------
    # 13. AI PREDICTION & FOCUS
    # -------------------------------------------------------------
    elif selected_page == "AI Prediction & Focus":
        st.title("🎯 AI Academic Performance Prediction & Guidance")
        st.write("Machine learning projections and personalized VTU academic guidance based on your internal assessment (IA1/IA2) performance.")

        cur_sem = int(student["current_semester"] or 5)
        db = get_db()
        current_records = db.execute("SELECT * FROM current_semester_records WHERE student_id = ? AND semester_number = ? ORDER BY subject_code ASC", (student_id, cur_sem)).fetchall()
        db.close()

        if not current_records:
            st.warning("Please add your current semester subjects and IA marks under 'Current Semester' first.")
        else:
            predictions = []
            for r in current_records:
                pred = predict_subject_final_marks(
                    student_id=student_id,
                    subject_code=r["subject_code"],
                    subject_name=r["subject_name"],
                    ia1=r["ia1"],
                    ia2=r["ia2"],
                    credits=r["credits"]
                )
                predictions.append(pred)

            pred_sgpa = calculate_predicted_sgpa(predictions)
            proj_cgpa = calculate_projected_cgpa(student_id, pred_sgpa)

            c1, c2 = st.columns(2)
            c1.metric("Predicted Semester SGPA", f"{pred_sgpa:.2f}")
            c2.metric("Projected Overall CGPA", f"{proj_cgpa:.2f}")

            st.divider()
            st.subheader("📋 Subject-Level Predictions Matrix")

            pred_df = []
            for p in predictions:
                pred_df.append({
                    "Subject Code": p["subject_code"],
                    "Subject Name": p["subject_name"],
                    "Credits": p["credits"],
                    "IA1": p["ia1"],
                    "IA2": p["ia2"],
                    "Internal Avg": p["internal_performance"],
                    "Predicted Final VTU Mark": p["predicted_final"],
                    "Expected Grade": p["expected_result"],
                    "Risk Level": p["attention_level"]
                })
            st.dataframe(pd.DataFrame(pred_df), use_container_width=True, hide_index=True)

            # Interactive Plotly Bar Chart for Predictions
            df_plot_pred = pd.DataFrame(pred_df)
            fig_bar = px.bar(
                df_plot_pred,
                x="Subject Code",
                y="Predicted Final VTU Mark",
                color="Risk Level",
                text="Predicted Final VTU Mark",
                color_discrete_map={"Strong": "#10B981", "Stable": "#3B82F6", "Needs Attention": "#F59E0B", "High Attention": "#EF4444"},
                range_y=[0, 100]
            )
            fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()
            st.subheader("🔍 Weak Subject Detection & Improvement Guidance")
            for p in predictions:
                badge_color = "red" if p["attention_level"] == "High Attention" else ("orange" if p["attention_level"] == "Needs Attention" else "green")
                with st.expander(f"{p['subject_code']} - {p['subject_name']} — :{badge_color}[{p['attention_level']}]", expanded=(p["attention_level"] in ["High Attention", "Needs Attention"])):
                    st.write(f"**Predicted Total Marks:** {p['predicted_final']} / 100")
                    st.write(f"**Expected Result:** {p['expected_result']}")
                    for sug in p["suggestions"]:
                        st.info(f"💡 {sug}")

            # What-if Simulation Tool
            st.divider()
            st.subheader("🧪 What-If IA Performance Simulator (Based on IA Average)")
            st.caption("Simulate how improving your IA marks and Average Internal Score directly drives your predicted final VTU marks and grade.")
            
            sim_sub = st.selectbox("Select Subject to Simulate", [p["subject_code"] + " - " + p["subject_name"] for p in predictions], key="what_if_sub_select")
            sim_code = sim_sub.split(" - ")[0]
            orig_p = next(p for p in predictions if p["subject_code"] == sim_code)
            orig_avg = float(orig_p.get("internal_performance", round((orig_p["ia1"] + orig_p["ia2"])/2, 1)))

            sim_mode = st.radio("Simulation Mode", ["Target IA Average", "Individual IA1 & IA2"], horizontal=True, key=f"sim_mode_{sim_code}")

            if sim_mode == "Target IA Average":
                sim_avg_val = st.slider(
                    f"Target IA Average for {sim_code} (Current Avg: {orig_avg:.1f} / 50)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(orig_avg),
                    step=0.5,
                    key=f"sim_avg_slider_{sim_code}"
                )
                sim_pred = predict_subject_final_marks(
                    student_id=student_id,
                    subject_code=orig_p["subject_code"],
                    subject_name=orig_p["subject_name"],
                    ia1=sim_avg_val,
                    ia2=sim_avg_val,
                    credits=orig_p["credits"],
                    ia_avg=sim_avg_val
                )
                used_sim_avg = sim_avg_val
            else:
                c_s1, c_s2 = st.columns(2)
                sim_ia1 = c_s1.slider(f"Simulate IA1", 0.0, 50.0, float(orig_p["ia1"]), 0.5, key=f"s_ia1_{sim_code}")
                sim_ia2 = c_s2.slider(f"Simulate IA2", 0.0, 50.0, float(orig_p["ia2"]), 0.5, key=f"s_ia2_{sim_code}")
                used_sim_avg = round((sim_ia1 + sim_ia2) / 2.0, 1)
                sim_pred = predict_subject_final_marks(
                    student_id=student_id,
                    subject_code=orig_p["subject_code"],
                    subject_name=orig_p["subject_name"],
                    ia1=sim_ia1,
                    ia2=sim_ia2,
                    credits=orig_p["credits"],
                    ia_avg=used_sim_avg
                )

            diff_marks = round(sim_pred["predicted_final"] - orig_p["predicted_final"], 1)
            diff_avg = round(used_sim_avg - orig_avg, 1)

            st.markdown("#### 📊 Simulation Comparison")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Original IA Avg", f"{orig_avg:.1f} / 50")
            sc2.metric("Simulated IA Avg", f"{used_sim_avg:.1f} / 50", delta=f"{diff_avg:+.1f} avg" if diff_avg != 0 else "0.0")
            sc3.metric("Original Final", f"{orig_p['predicted_final']} / 100")
            sc4.metric("Simulated Final", f"{sim_pred['predicted_final']} / 100", delta=f"{diff_marks:+.1f} marks" if diff_marks != 0 else "0.0 (Baseline)")

            if diff_marks > 0:
                st.success(f"🚀 **Targeted Improvement:** Increasing your IA Average from **{orig_avg:.1f}** to **{used_sim_avg:.1f}** raises your predicted final marks by **+{diff_marks} marks** (Expected Grade: **{sim_pred.get('expected_result', 'PASS')}**).")
            elif diff_marks < 0:
                st.warning(f"⚠️ A lower IA Average ({used_sim_avg:.1f}) drops your predicted final marks by **{diff_marks} marks**.")
            else:
                st.info("💡 **Try it:** Drag the slider above to increase your IA Average (e.g. from 22 to 38+) and observe the live predicted final score increase!")

    # -------------------------------------------------------------
    # 14. REPORTS
    # -------------------------------------------------------------
    elif selected_page == "Reports":
        st.title("📑 Academic Reports & Exports")
        st.write("Generate and download your official comprehensive academic profile report in PDF format.")

        if st.button("📄 Generate & Download Student PDF Report", type="primary", use_container_width=True):
            try:
                pdf_data = generate_student_pdf_report(student_id)
                st.download_button(
                    label="📥 Download Academic Report (PDF)",
                    data=pdf_data,
                    file_name=f"EduPredict_{student['usn']}_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")

    # -------------------------------------------------------------
    # 15. NOTIFICATIONS
    # -------------------------------------------------------------
    elif selected_page == "Notifications":
        st.title("🔔 Notifications & Announcements")
        st.write("Live institutional updates, document verification feedback, and faculty messages.")

        notif_info = get_user_notifications(student_id, "student")
        unread_items = notif_info["unread"]
        read_items = notif_info["read"]
        all_items = unread_items + read_items
        all_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Header metric overview
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("📬 Total Notifications", len(all_items))
        col_m2.metric("🟡 Unread Alerts", len(unread_items))
        col_m3.metric("✅ Marked as Read", len(read_items))

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        tab_unread, tab_read, tab_all = st.tabs([
            f"📬 Unread Alerts ({len(unread_items)})",
            f"✅ Marked as Read ({len(read_items)})",
            f"📋 All Notifications ({len(all_items)})"
        ])

        with tab_unread:
            if unread_items:
                col_u_hdr, col_u_btn = st.columns([3.5, 1.5])
                with col_u_hdr:
                    st.caption(f"You have **{len(unread_items)}** unread notification(s). Mark them as read once reviewed:")
                with col_u_btn:
                    if st.button("✨ Mark All as Read", key="btn_mark_all_read_stu", type="secondary", use_container_width=True):
                        cnt = mark_all_notifications_as_read(student_id, "student")
                        st.success(f"🎉 Marked {cnt} notification(s) as read!")
                        st.rerun()

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                for n in unread_items:
                    with st.container():
                        st.markdown(f"""
                        <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #3B82F6; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; font-size: 15px; color: #1E293B;">📢 {n['title']}</span>
                                <div>
                                    <span style="font-size: 11px; color: #B45309; background: #FEF3C7; padding: 3px 10px; border-radius: 12px; font-weight: 600; margin-right: 6px;">🟡 Unread</span>
                                    <span style="font-size: 11px; color: #2563EB; background: #EFF6FF; padding: 3px 10px; border-radius: 12px; font-weight: 600;">{n.get('category') or 'General'}</span>
                                </div>
                            </div>
                            <div style="font-size: 13.5px; color: #334155; margin-top: 8px; line-height: 1.5;">{n['message']}</div>
                            <div style="font-size: 11.5px; color: #94A3B8; margin-top: 8px;">Received on: {n['created_at'][:16].replace('T', ' ')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_dummy, c_mark_btn = st.columns([4, 1.3])
                        with c_mark_btn:
                            if st.button("✅ Mark as Read", key=f"mark_read_stu_{n['id']}", use_container_width=True, type="primary"):
                                mark_notification_as_read(n["id"], student_id)
                                st.rerun()
            else:
                st.success("✨ All caught up! You have zero unread notifications.")

        with tab_read:
            if read_items:
                st.caption(f"Showing **{len(read_items)}** notification(s) marked as read:")
                for n in read_items:
                    with st.container():
                        st.markdown(f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid #10B981; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 600; font-size: 14.5px; color: #475569;">📢 {n['title']}</span>
                                <div>
                                    <span style="font-size: 11px; color: #047857; background: #D1FAE5; padding: 3px 10px; border-radius: 12px; font-weight: 600; margin-right: 6px;">✅ Marked as Read</span>
                                    <span style="font-size: 11px; color: #64748B; background: #F1F5F9; padding: 3px 8px; border-radius: 12px; font-weight: 500;">{n.get('category') or 'General'}</span>
                                </div>
                            </div>
                            <div style="font-size: 13px; color: #64748B; margin-top: 6px; line-height: 1.4;">{n['message']}</div>
                            <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">Received: {n['created_at'][:16].replace('T', ' ')} {('• Marked Read: ' + n['read_at'][:16].replace('T', ' ')) if n.get('read_at') else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_dummy, c_unmark_btn = st.columns([4, 1.3])
                        with c_unmark_btn:
                            if st.button("↩️ Mark as Unread", key=f"unmark_read_stu_{n['id']}", use_container_width=True):
                                mark_notification_as_unread(n["id"], student_id)
                                st.rerun()
            else:
                st.info("No past read notifications recorded.")

        with tab_all:
            if all_items:
                st.caption(f"Showing all **{len(all_items)}** total notifications with active status:")
                for n in all_items:
                    is_n_read = (n.get("user_is_read") == 1)
                    badge_html = '<span style="font-size: 11px; color: #047857; background: #D1FAE5; padding: 3px 10px; border-radius: 12px; font-weight: 600;">✅ Marked as Read</span>' if is_n_read else '<span style="font-size: 11px; color: #B45309; background: #FEF3C7; padding: 3px 10px; border-radius: 12px; font-weight: 600;">🟡 Unread</span>'
                    border_color = "#10B981" if is_n_read else "#3B82F6"
                    bg_color = "#F8FAFC" if is_n_read else "#FFFFFF"

                    with st.container():
                        st.markdown(f"""
                        <div style="background: {bg_color}; border: 1px solid #CBD5E1; border-left: 5px solid {border_color}; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; font-size: 15px; color: #1E293B;">📢 {n['title']}</span>
                                <div>
                                    {badge_html}
                                    <span style="font-size: 11px; color: #64748B; background: #F1F5F9; padding: 3px 8px; border-radius: 12px; font-weight: 500; margin-left: 6px;">{n.get('category') or 'General'}</span>
                                </div>
                            </div>
                            <div style="font-size: 13.5px; color: #334155; margin-top: 8px; line-height: 1.5;">{n['message']}</div>
                            <div style="font-size: 11.5px; color: #94A3B8; margin-top: 8px;">Received: {n['created_at'][:16].replace('T', ' ')} {('• Marked Read: ' + n['read_at'][:16].replace('T', ' ')) if is_n_read and n.get('read_at') else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_dummy, c_btn = st.columns([4, 1.3])
                        with c_btn:
                            if is_n_read:
                                if st.button("↩️ Mark as Unread", key=f"all_unmark_{n['id']}", use_container_width=True):
                                    mark_notification_as_unread(n["id"], student_id)
                                    st.rerun()
                            else:
                                if st.button("✅ Mark as Read", key=f"all_mark_{n['id']}", use_container_width=True, type="primary"):
                                    mark_notification_as_read(n["id"], student_id)
                                    st.rerun()
            else:
                st.info("No notifications found.")

    # -------------------------------------------------------------
    # 16. SETTINGS
    # -------------------------------------------------------------
    elif selected_page == "Settings":
        st.title("⚙️ Account Settings")
        st.write("Manage your password and security credentials.")

        with st.form("student_change_password_form"):
            st.subheader("Change Account Password")
            old_p = st.text_input("Current Password", type="password")
            new_p = st.text_input("New Password (min 6 characters)", type="password")
            cnf_p = st.text_input("Confirm New Password", type="password")
            sub_cp = st.form_submit_button("Update Password Permanently", type="primary")

            if sub_cp:
                succ, msg = change_user_password("student", student_id, old_p, new_p, cnf_p)
                if succ:
                    st.success(msg)
                else:
                    st.error(msg)