import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
import re

# ============================================================
# EDUPREDICT AI
# Student Performance Prediction & Academic Analytics
# ============================================================

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(80,80,180,0.12), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(0,180,180,0.08), transparent 30%),
        #070b18;
    color: #f5f7ff;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1023 0%, #080b17 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(135deg, #171d42, #10162e);
    border: 1px solid rgba(130,140,255,0.25);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    color: #aab4d0;
    font-size: 15px;
}

.card {
    background: linear-gradient(145deg, #11182e, #0c1224);
    border: 1px solid rgba(150,160,255,0.15);
    border-radius: 18px;
    padding: 22px;
    min-height: 130px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.card-title {
    color: #9ca8c8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.card-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.card-sub {
    color: #8490af;
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 12px;
}

.student-card {
    background: linear-gradient(135deg, #151c38, #0e1428);
    border: 1px solid rgba(120,140,255,0.2);
    border-radius: 20px;
    padding: 20px;
}

.photo-box {
    border-radius: 14px;
    border: 2px solid rgba(120,140,255,0.3);
    padding: 5px;
}

.risk-high {
    color: #ff7070;
    font-weight: 800;
}

.risk-medium {
    color: #ffc857;
    font-weight: 800;
}

.risk-low {
    color: #49e0a5;
    font-weight: 800;
}

.small-muted {
    color: #8994b2;
    font-size: 12px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #11182e, #0d1325);
    border: 1px solid rgba(130,140,255,0.15);
    padding: 16px;
    border-radius: 16px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 650;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "students": [],
    "selected_student": None,
    "class_name": "",
    "class_teacher": "",
    "branch": "",
    "college": "",
    "academic_year": "",
    "current_semester": 1,
    "previous_documents": {},
    "backlog_documents": {},
    "current_subjects": {},
    "history": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except:
        return default


def calculate_grade_point(mark):
    """
    Simplified VTU-style 10-point mapping.
    The mapping is kept transparent so it can be adjusted
    according to the student's applicable VTU scheme.
    """
    mark = safe_float(mark)

    if mark >= 90:
        return 10
    elif mark >= 80:
        return 9
    elif mark >= 70:
        return 8
    elif mark >= 60:
        return 7
    elif mark >= 55:
        return 6
    elif mark >= 50:
        return 5
    elif mark >= 40:
        return 4
    else:
        return 0


def calculate_sgpa(subjects):
    if not subjects:
        return 0.0

    total_credits = 0
    weighted_points = 0

    for subject in subjects:
        credits = safe_float(subject.get("credits", 0))
        final_mark = safe_float(subject.get("final", 0))

        gp = calculate_grade_point(final_mark)

        total_credits += credits
        weighted_points += gp * credits

    if total_credits == 0:
        return 0.0

    return round(weighted_points / total_credits, 2)


def calculate_cgpa(semester_data):
    total_credits = 0
    weighted_sgpa = 0

    for sem in semester_data:
        sgpa = safe_float(sem.get("sgpa", 0))
        credits = safe_float(sem.get("credits", 0))

        total_credits += credits
        weighted_sgpa += sgpa * credits

    if total_credits == 0:
        return 0.0

    return round(weighted_sgpa / total_credits, 2)


def performance_band(value):
    value = safe_float(value)

    if value >= 8.5:
        return "Excellent"
    elif value >= 7.0:
        return "Good"
    elif value >= 5.5:
        return "Average"
    else:
        return "Needs Improvement"


def prediction_from_data(
    previous_sgpa,
    current_sgpa,
    attendance_absent,
    previous_absent,
    ia_average,
    current_final_average
):
    """
    Transparent academic-risk scoring.
    This is intended as a project-level predictive demonstration.
    """

    score = 50.0

    score += (previous_sgpa - 5.0) * 7
    score += (current_sgpa - 5.0) * 8
    score += (ia_average - 40) * 0.25
    score += (current_final_average - 40) * 0.18

    # Current absenteeism
    if attendance_absent > previous_absent:
        score -= min((attendance_absent - previous_absent) * 1.5, 20)

    if attendance_absent >= 10:
        score -= 10
    elif attendance_absent >= 5:
        score -= 5

    score = max(0, min(100, score))

    if score >= 75:
        risk = "LOW"
        prediction = "Student is predicted to maintain a strong academic performance."
    elif score >= 50:
        risk = "MEDIUM"
        prediction = "Student may require academic monitoring and timely support."
    else:
        risk = "HIGH"
        prediction = "Student shows indicators of possible academic difficulty."

    return round(score, 1), risk, prediction


def read_uploaded_table(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Could not read file: {e}")

    return None


def extract_pdf_text(uploaded_file):
    try:
        from pypdf import PdfReader

        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        return text.strip()

    except ImportError:
        return None

    except Exception as e:
        return f"PDF extraction error: {e}"


def validate_photo(uploaded_file):
    if uploaded_file is None:
        return False, "No photo selected."

    max_size = 1 * 1024 * 1024

    if uploaded_file.size > max_size:
        return False, "Photo must be 1 MB or smaller."

    allowed = [".jpg", ".jpeg", ".png"]

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in allowed:
        return False, "Only JPG, JPEG and PNG images are allowed."

    return True, "Valid photo."


def calculate_average_ia(subjects):
    values = []

    for subject in subjects:
        ia1 = safe_float(subject.get("ia1", 0))
        ia2 = safe_float(subject.get("ia2", 0))

        values.append((ia1 + ia2) / 2)

    if not values:
        return 0

    return round(np.mean(values), 2)


def calculate_final_average(subjects):
    values = [
        safe_float(s.get("final", 0))
        for s in subjects
        if s.get("final") not in ("", None)
    ]

    if not values:
        return 0

    return round(np.mean(values), 2)


def get_student_history(student_id):
    return st.session_state.history.get(student_id, [])


def get_current_subjects(student_id):
    return st.session_state.current_subjects.get(student_id, [])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:8px 0 22px 0;">
        <div style="font-size:25px;font-weight:800;">🎓 EduPredict AI</div>
        <div style="color:#8994b2;font-size:13px;">
            Student Academic Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Student Profile",
            "📚 Academic History",
            "📝 Current Semester",
            "📊 AI Analysis",
            "📈 SGPA & CGPA",
            "📄 Documents",
        ]
    )

    st.divider()

    st.markdown("### Class Information")

    st.session_state.class_name = st.text_input(
        "Class Name",
        value=st.session_state.class_name,
        placeholder="Example: 5th Sem CSE-A"
    )

    st.session_state.class_teacher = st.text_input(
        "Class Teacher",
        value=st.session_state.class_teacher,
        placeholder="Enter teacher name"
    )

    st.session_state.branch = st.text_input(
        "Branch / Department",
        value=st.session_state.branch,
        placeholder="Example: Computer Science"
    )

    st.session_state.college = st.text_input(
        "College Name",
        value=st.session_state.college,
        placeholder="Engineering College"
    )

    st.session_state.academic_year = st.text_input(
        "Academic Year",
        value=st.session_state.academic_year,
        placeholder="2026-27"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">EduPredict AI</div>
    <div class="hero-subtitle">
        Student Performance Prediction using Academic History,
        Current Semester Performance and Ensemble-style Analytics
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("## 📊 Academic Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Class",
            st.session_state.class_name or "Not Set"
        )

    with col2:
        st.metric(
            "Students",
            len(st.session_state.students)
        )

    with col3:
        st.metric(
            "Current Semester",
            st.session_state.current_semester
        )

    with col4:
        st.metric(
            "Class Teacher",
            st.session_state.class_teacher or "Not Set"
        )

    st.markdown("### 👨‍🎓 Student Overview")

    if not st.session_state.students:
        st.info(
            "No students added yet. Go to **Student Profile** to add students."
        )
    else:

        rows = []

        for student in st.session_state.students:

            student_id = student["id"]

            history = get_student_history(student_id)
            current = get_current_subjects(student_id)

            previous_sgpa = (
                history[-1]["sgpa"]
                if history else 0
            )

            current_sgpa = calculate_sgpa(current)

            rows.append({
                "Student": student["name"],
                "USN": student["id"],
                "Previous SGPA": previous_sgpa,
                "Current SGPA": current_sgpa,
                "Current CGPA": calculate_cgpa(history),
                "Current Absentees": student.get(
                    "current_absent", 0
                ),
            })

        df_dashboard = pd.DataFrame(rows)

        st.dataframe(
            df_dashboard,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# STUDENT PROFILE
# ============================================================

elif page == "👤 Student Profile":

    st.markdown("## 👤 Student Profile")

    st.info(
        "Add students manually. Passport-size photos must be "
        "JPG/PNG and 1 MB or smaller."
    )

    col1, col2 = st.columns(2)

    with col1:
        student_name = st.text_input(
            "Student Name",
            placeholder="Enter full name"
        )

        student_id = st.text_input(
            "USN / Student ID",
            placeholder="Example: 1TJ24DS001"
        )

        admission_year = st.number_input(
            "Admission Year",
            min_value=2000,
            max_value=2100,
            value=2024
        )

    with col2:

        current_sem = st.number_input(
            "Current Semester",
            min_value=1,
            max_value=8,
            value=1
        )

        previous_absent = st.number_input(
            "Previous Semester Absentee Days",
            min_value=0,
            value=0
        )

        current_absent = st.number_input(
            "Current Semester Absentee Days",
            min_value=0,
            value=0
        )

    photo = st.file_uploader(
        "Passport-size Photo — Maximum 1 MB",
        type=["jpg", "jpeg", "png"],
        key="student_photo"
    )

    if photo:
        valid, message = validate_photo(photo)

        if valid:
            st.success(message)

            photo_col1, photo_col2 = st.columns([1, 3])

            with photo_col1:
                st.image(
                    photo,
                    caption="Student Photo",
                    width=150
                )
        else:
            st.error(message)
            photo = None

    if st.button(
        "➕ Add Student",
        type="primary",
        use_container_width=True
    ):

        if not student_name.strip():
            st.error("Please enter student name.")

        elif not student_id.strip():
            st.error("Please enter USN / Student ID.")

        elif any(
            s["id"].lower() == student_id.lower()
            for s in st.session_state.students
        ):
            st.error("A student with this USN already exists.")

        else:

            photo_bytes = None

            if photo:
                photo_bytes = photo.getvalue()

            new_student = {
                "id": student_id.strip(),
                "name": student_name.strip(),
                "admission_year": admission_year,
                "current_semester": current_sem,
                "previous_absent": previous_absent,
                "current_absent": current_absent,
                "photo": photo_bytes
            }

            st.session_state.students.append(new_student)

            st.success(
                f"{student_name} added successfully."
            )

    st.markdown("### 👥 Added Students")

    if st.session_state.students:

        for student in st.session_state.students:

            c1, c2, c3 = st.columns([1, 3, 1])

            with c1:

                if student.get("photo"):
                    st.image(
                        student["photo"],
                        width=80
                    )
                else:
                    st.markdown("👤")

            with c2:

                st.markdown(
                    f"**{student['name']}**"
                )

                st.caption(
                    f"USN: {student['id']} | "
                    f"Semester: {student['current_semester']}"
                )

            with c3:

                if st.button(
                    "Select",
                    key=f"select_{student['id']}"
                ):
                    st.session_state.selected_student = student["id"]
                    st.success("Student selected.")

            st.divider()


# ============================================================
# ACADEMIC HISTORY
# ============================================================

elif page == "📚 Academic History":

    st.markdown("## 📚 Previous Academic History")

    if not st.session_state.students:

        st.warning(
            "Add students first from Student Profile."
        )

    else:

        student_ids = [
            f"{s['name']} — {s['id']}"
            for s in st.session_state.students
        ]

        selected = st.selectbox(
            "Select Student",
            student_ids
        )

        student_id = selected.split(" — ")[-1]

        student = next(
            s for s in st.session_state.students
            if s["id"] == student_id
        )

        st.markdown(
            f"### Academic History — {student['name']}"
        )

        semester = st.number_input(
            "Semester to Add",
            min_value=1,
            max_value=8,
            value=1,
            key=f"history_sem_{student_id}"
        )

        st.markdown("### 📄 Upload Previous Marksheet")

        uploaded_marksheet = st.file_uploader(
            "PDF / Excel / CSV",
            type=["pdf", "csv", "xlsx", "xls"],
            key=f"marksheet_{student_id}_{semester}"
        )

        if uploaded_marksheet:

            file_name = uploaded_marksheet.name.lower()

            if file_name.endswith(
                (".csv", ".xlsx", ".xls")
            ):

                imported_df = read_uploaded_table(
                    uploaded_marksheet
                )

                if imported_df is not None:

                    st.success(
                        "Academic data imported successfully."
                    )

                    st.dataframe(
                        imported_df,
                        use_container_width=True
                    )

                    st.caption(
                        "Review the imported data before using it."
                    )

            elif file_name.endswith(".pdf"):

                extracted = extract_pdf_text(
                    uploaded_marksheet
                )

                if extracted is None:

                    st.warning(
                        "Install pypdf to enable PDF text extraction."
                    )

                else:

                    st.success(
                        "PDF uploaded successfully."
                    )

                    st.text_area(
                        "Extracted Marksheet Text",
                        extracted,
                        height=250
                    )

        st.markdown("### 🧮 Enter / Verify Semester Data")

        number_subjects = st.number_input(
            "Number of Subjects",
            min_value=1,
            max_value=20,
            value=5,
            key=f"num_subjects_{student_id}_{semester}"
        )

        subjects = []

        for i in range(int(number_subjects)):

            st.markdown(
                f"#### Subject {i + 1}"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                subject_name = st.text_input(
                    "Subject Name",
                    key=f"hist_subject_{student_id}_{semester}_{i}"
                )

            with c2:
                credits = st.number_input(
                    "Credits",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    key=f"hist_credit_{student_id}_{semester}_{i}"
                )

            with c3:
                final_mark = st.number_input(
                    "VTU Final Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"hist_final_{student_id}_{semester}_{i}"
                )

            ia1, ia2 = st.columns(2)

            with ia1:
                ia1_mark = st.number_input(
                    "IA-1",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"hist_ia1_{student_id}_{semester}_{i}"
                )

            with ia2:
                ia2_mark = st.number_input(
                    "IA-2",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"hist_ia2_{student_id}_{semester}_{i}"
                )

            subjects.append({
                "subject": subject_name,
                "credits": credits,
                "ia1": ia1_mark,
                "ia2": ia2_mark,
                "final": final_mark
            })

        calculated_sgpa = calculate_sgpa(subjects)

        st.metric(
            "Calculated Semester SGPA",
            calculated_sgpa
        )

        if st.button(
            "💾 Save Semester History",
            type="primary",
            key=f"save_history_{student_id}_{semester}"
        ):

            semester_record = {
                "semester": semester,
                "subjects": subjects,
                "sgpa": calculated_sgpa,
                "credits": sum(
                    safe_float(x["credits"])
                    for x in subjects
                )
            }

            if student_id not in st.session_state.history:
                st.session_state.history[student_id] = []

            existing = [
                x for x in st.session_state.history[student_id]
                if x["semester"] != semester
            ]

            existing.append(semester_record)

            existing.sort(
                key=lambda x: x["semester"]
            )

            st.session_state.history[student_id] = existing

            st.success(
                f"Semester {semester} saved."
            )

        st.markdown("### 🎓 Semester History")

        history = get_student_history(student_id)

        if history:

            history_rows = []

            for record in history:

                history_rows.append({
                    "Semester": record["semester"],
                    "Credits": record["credits"],
                    "SGPA": record["sgpa"],
                    "Performance":
                        performance_band(record["sgpa"])
                })

            st.dataframe(
                pd.DataFrame(history_rows),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No previous semester data saved yet."
            )


# ============================================================
# CURRENT SEMESTER
# ============================================================

elif page == "📝 Current Semester":

    st.markdown("## 📝 Current Semester")

    if not st.session_state.students:

        st.warning(
            "Add students first."
        )

    else:

        student_options = [
            f"{s['name']} — {s['id']}"
            for s in st.session_state.students
        ]

        selected = st.selectbox(
            "Select Student",
            student_options,
            key="current_student_select"
        )

        student_id = selected.split(" — ")[-1]

        student = next(
            s for s in st.session_state.students
            if s["id"] == student_id
        )

        current_semester = student["current_semester"]

        st.markdown(
            f"### Semester {current_semester} — {student['name']}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            previous_absent = st.number_input(
                "Previous Semester Absentee Days",
                min_value=0,
                value=int(student.get("previous_absent", 0)),
                key=f"prev_abs_{student_id}"
            )

        with c2:

            current_absent = st.number_input(
                "Current Semester Absentee Days",
                min_value=0,
                value=int(student.get("current_absent", 0)),
                key=f"curr_abs_{student_id}"
            )

        with c3:

            total_working_days = st.number_input(
                "Current Semester Working Days",
                min_value=1,
                value=90,
                key=f"working_days_{student_id}"
            )

        attendance = max(
            0,
            min(
                100,
                ((total_working_days - current_absent)
                 / total_working_days) * 100
            )
        )

        st.metric(
            "Current Attendance",
            f"{attendance:.1f}%"
        )

        number_subjects = st.number_input(
            "Number of Current Semester Subjects",
            min_value=1,
            max_value=20,
            value=5,
            key=f"current_subject_count_{student_id}"
        )

        subjects = []

        for i in range(int(number_subjects)):

            st.markdown(
                f"### 📘 Subject {i + 1}"
            )

            c1, c2 = st.columns(2)

            with c1:

                subject_name = st.text_input(
                    "Subject Name",
                    key=f"current_sub_{student_id}_{i}"
                )

            with c2:

                credits = st.number_input(
                    "Credits",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    key=f"current_credit_{student_id}_{i}"
                )

            c1, c2, c3 = st.columns(3)

            with c1:

                ia1 = st.number_input(
                    "IA-1 Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"current_ia1_{student_id}_{i}"
                )

            with c2:

                ia2 = st.number_input(
                    "IA-2 Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"current_ia2_{student_id}_{i}"
                )

            with c3:

                final_mark = st.number_input(
                    "Final / VTU Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"current_final_{student_id}_{i}"
                )

            subjects.append({
                "subject": subject_name,
                "credits": credits,
                "ia1": ia1,
                "ia2": ia2,
                "final": final_mark
            })

        current_sgpa = calculate_sgpa(subjects)

        st.markdown("### 📊 Current Semester Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Current SGPA",
                current_sgpa
            )

        with c2:
            st.metric(
                "Average IA",
                calculate_average_ia(subjects)
            )

        with c3:
            st.metric(
                "Final Marks Average",
                calculate_final_average(subjects)
            )

        if st.button(
            "💾 Save Current Semester",
            type="primary",
            use_container_width=True
        ):

            st.session_state.current_subjects[
                student_id
            ] = subjects

            student["current_absent"] = current_absent
            student["previous_absent"] = previous_absent

            st.success(
                "Current semester data saved successfully."
            )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "📊 AI Analysis":

    st.markdown("## 🤖 AI Academic Analysis")

    if not st.session_state.students:

        st.warning("Add students first.")

    else:

        student_options = [
            f"{s['name']} — {s['id']}"
            for s in st.session_state.students
        ]

        selected = st.selectbox(
            "Select Student",
            student_options,
            key="analysis_student"
        )

        student_id = selected.split(" — ")[-1]

        student = next(
            s for s in st.session_state.students
            if s["id"] == student_id
        )

        history = get_student_history(student_id)
        current = get_current_subjects(student_id)

        previous_sgpa = (
            history[-1]["sgpa"]
            if history else 0
        )

        current_sgpa = calculate_sgpa(current)

        current_ia = calculate_average_ia(current)

        current_final = calculate_final_average(current)

        score, risk, prediction = prediction_from_data(
            previous_sgpa,
            current_sgpa,
            student.get("current_absent", 0),
            student.get("previous_absent", 0),
            current_ia,
            current_final
        )

        st.markdown(
            f"""
            <div class="student-card">
                <h2>{student['name']}</h2>
                <div class="small-muted">
                    USN: {student['id']}
                </div>
                <br>
                <b>AI Academic Prediction</b>
                <br><br>
                {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📈 Prediction Indicators")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Prediction Score",
                f"{score}/100"
            )

        with c2:
            st.metric(
                "Previous SGPA",
                previous_sgpa
            )

        with c3:
            st.metric(
                "Current SGPA",
                current_sgpa
            )

        with c4:
            st.metric(
                "Current Absentees",
                student.get("current_absent", 0)
            )

        if risk == "HIGH":
            st.error(
                "🔴 HIGH RISK — Academic intervention recommended."
            )

        elif risk == "MEDIUM":
            st.warning(
                "🟡 MEDIUM RISK — Student should be monitored."
            )

        else:
            st.success(
                "🟢 LOW RISK — Student is currently performing well."
            )

        st.markdown("### 📚 Subject-wise Analysis")

        if current:

            subject_rows = []

            for subject in current:

                ia_avg = (
                    safe_float(subject["ia1"]) +
                    safe_float(subject["ia2"])
                ) / 2

                final_mark = safe_float(
                    subject["final"]
                )

                subject_score = (
                    ia_avg * 0.4 +
                    final_mark * 0.6
                )

                if subject_score < 40:
                    subject_risk = "High"

                elif subject_score < 60:
                    subject_risk = "Medium"

                else:
                    subject_risk = "Low"

                subject_rows.append({
                    "Subject": subject["subject"],
                    "Credits": subject["credits"],
                    "IA Average": round(ia_avg, 2),
                    "Final": final_mark,
                    "Performance Score":
                        round(subject_score, 2),
                    "Risk": subject_risk
                })

            df_subject = pd.DataFrame(subject_rows)

            st.dataframe(
                df_subject,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Enter current-semester subject data first."
            )

        st.markdown("### 🧠 Academic Recommendations")

        recommendations = []

        if student.get("current_absent", 0) > student.get(
            "previous_absent", 0
        ):
            recommendations.append(
                "Attendance has decreased compared with the previous semester."
            )

        if current_ia < 40:
            recommendations.append(
                "IA performance indicates that additional internal-assessment preparation may be useful."
            )

        if current_sgpa < 6:
            recommendations.append(
                "Current semester performance requires closer academic monitoring."
            )

        if not recommendations:
            recommendations.append(
                "Current indicators are stable. Continue consistent preparation."
            )

        for item in recommendations:
            st.write("•", item)


# ============================================================
# SGPA / CGPA
# ============================================================

elif page == "📈 SGPA & CGPA":

    st.markdown("## 📈 SGPA & CGPA Analysis")

    if not st.session_state.students:

        st.warning("Add students first.")

    else:

        student_options = [
            f"{s['name']} — {s['id']}"
            for s in st.session_state.students
        ]

        selected = st.selectbox(
            "Select Student",
            student_options,
            key="cgpa_student"
        )

        student_id = selected.split(" — ")[-1]

        history = get_student_history(student_id)

        if not history:

            st.info(
                "Add previous semester academic data first."
            )

        else:

            rows = []

            for sem in history:

                rows.append({
                    "Semester":
                        f"Semester {sem['semester']}",
                    "Credits":
                        sem["credits"],
                    "SGPA":
                        sem["sgpa"],
                    "Performance":
                        performance_band(sem["sgpa"])
                })

            df = pd.DataFrame(rows)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            total_cgpa = calculate_cgpa(history)

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Current CGPA
                    </div>
                    <div class="card-value">
                        {total_cgpa}
                    </div>
                    <div class="card-sub">
                        Based on entered semester academic history
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📊 SGPA Trend")

            chart_df = pd.DataFrame({
                "Semester": [
                    f"S{x['semester']}"
                    for x in history
                ],
                "SGPA": [
                    x["sgpa"]
                    for x in history
                ]
            })

            st.line_chart(
                chart_df.set_index("Semester")
            )


# ============================================================
# DOCUMENTS
# ============================================================

elif page == "📄 Documents":

    st.markdown("## 📄 Academic Documents")

    if not st.session_state.students:

        st.warning("Add students first.")

    else:

        student_options = [
            f"{s['name']} — {s['id']}"
            for s in st.session_state.students
        ]

        selected = st.selectbox(
            "Select Student",
            student_options,
            key="document_student"
        )

        student_id = selected.split(" — ")[-1]

        student = next(
            s for s in st.session_state.students
            if s["id"] == student_id
        )

        st.markdown(
            f"### Documents — {student['name']}"
        )

        tab1, tab2, tab3 = st.tabs([
            "📄 Previous Marksheet",
            "⚠️ Backlog Cleared",
            "🖼️ Student Photo"
        ])

        with tab1:

            sem = st.number_input(
                "Semester",
                min_value=1,
                max_value=8,
                value=1,
                key=f"doc_sem_{student_id}"
            )

            document = st.file_uploader(
                "Upload Previous Semester Marksheet",
                type=["pdf", "jpg", "jpeg", "png", "csv", "xlsx"],
                key=f"prev_doc_{student_id}_{sem}"
            )

            if document:

                st.session_state.previous_documents[
                    f"{student_id}_S{sem}"
                ] = document.name

                st.success(
                    f"Uploaded: {document.name}"
                )

                if document.name.lower().endswith(
                    ".pdf"
                ):

                    extracted = extract_pdf_text(document)

                    if extracted:

                        st.text_area(
                            "Extracted Text Preview",
                            extracted,
                            height=200
                        )

        with tab2:

            st.markdown(
                "Upload proof/result for a backlog that has been cleared."
            )

            backlog_subject = st.text_input(
                "Backlog Subject",
                key=f"backlog_subject_{student_id}"
            )

            cleared_document = st.file_uploader(
                "Backlog Cleared Result — PDF/JPG/PNG",
                type=["pdf", "jpg", "jpeg", "png"],
                key=f"backlog_doc_{student_id}"
            )

            if cleared_document:

                if backlog_subject.strip():

                    st.session_state.backlog_documents[
                        f"{student_id}_{backlog_subject}"
                    ] = {
                        "file": cleared_document.name,
                        "status": "CLEARED"
                    }

                    st.success(
                        f"{backlog_subject} marked as CLEARED."
                    )

                else:

                    st.warning(
                        "Enter the backlog subject name first."
                    )

        with tab3:

            if student.get("photo"):

                st.image(
                    student["photo"],
                    caption=student["name"],
                    width=180
                )

                st.success(
                    "Passport photo is stored for this student."
                )

            else:

                st.info(
                    "No passport photo uploaded."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#68738f;font-size:12px;">
        EduPredict AI • Student Performance Prediction & Academic Analytics
        <br>
        Historical Academic Data + Current Semester Data + Attendance Indicators
    </div>
    """,
    unsafe_allow_html=True
)
