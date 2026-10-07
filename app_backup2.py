import streamlit as st
import sqlite3
import uuid
import os
import re
import json
import math
from datetime import datetime

import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")

UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
PHOTO_DIR = os.path.join(UPLOAD_DIR, "photos")
MARKSHEET_DIR = os.path.join(UPLOAD_DIR, "marksheets")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(MARKSHEET_DIR, exist_ok=True)


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():

    db = get_db()

    db.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            class_teacher TEXT,
            created_at TEXT
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            class_id TEXT NOT NULL,
            name TEXT NOT NULL,
            usn TEXT,
            admission_year INTEGER,
            current_semester INTEGER,
            photo_path TEXT,
            absent_days INTEGER DEFAULT 0,
            working_days INTEGER DEFAULT 90,
            FOREIGN KEY(class_id) REFERENCES classes(id)
        )
    """)

    db.execute("""
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

    db.execute("""
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

    db.execute("""
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

    db.commit()
    db.close()


init_database()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def safe_float(value):
    try:
        if value is None:
            return None

        value = str(value).strip()

        if value in ["", "-", "—", "NA", "N/A"]:
            return None

        value = value.replace(",", "")

        match = re.search(r"\d+(?:\.\d+)?", value)

        if match:
            return float(match.group())

    except Exception:
        pass

    return None


def save_uploaded_file(uploaded_file, folder):
    if uploaded_file is None:
        return None

    extension = os.path.splitext(uploaded_file.name)[1].lower()

    filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex
        + extension
    )

    path = os.path.join(folder, filename)

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path


# ============================================================
# OCR FUNCTIONS
# ============================================================

def extract_pdf_text(pdf_path):
    """
    First try normal PDF text extraction.
    This works for digitally generated VTU PDFs.
    """

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += "\n" + page_text

        doc.close()

    except Exception:
        return ""

    return text


def extract_pdf_ocr(pdf_path):
    """
    OCR fallback for scanned/image-based PDFs.
    """

    all_text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2.5, 2.5),
                alpha=False
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            page_text = pytesseract.image_to_string(
                img,
                config="--psm 6"
            )

            all_text += "\n" + page_text

        doc.close()

    except Exception as e:
        st.warning(f"OCR error: {e}")

    return all_text


def get_marksheet_text(pdf_path):

    text = extract_pdf_text(pdf_path)

    # If PDF text is too short, use OCR.
    if len(text.strip()) < 100:
        text = extract_pdf_ocr(pdf_path)

    return text


# ============================================================
# VTU SUBJECT DETECTION
# ============================================================

# VTU subject codes generally look like:
# BMATS201
# BPHYS202
# BPOPS203
# BESCK204C
# BETCK205H
# BPLCK105A
#
# We deliberately use a flexible pattern because VTU codes
# change between schemes/semesters.

SUBJECT_CODE_PATTERN = re.compile(
    r"\b[A-Z]{2,8}\d{2,4}[A-Z]{0,3}\b"
)


def is_probable_subject_code(code):

    if not code:
        return False

    code = code.upper()

    # Avoid treating normal words as subject codes.
    if len(code) < 5:
        return False

    if not re.search(r"\d", code):
        return False

    if not re.match(r"^[A-Z]{2,8}\d{2,4}[A-Z]{0,3}$", code):
        return False

    return True


def extract_numbers_from_end(text):

    """
    Extract marks from the end of a VTU row.

    Typical:
        ... 44 18 62 P 2025-08-19

    Returns:
        internal, external, total
    """

    # Remove date
    text = re.sub(
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        " ",
        text
    )

    # Remove result letters
    text = re.sub(
        r"\b(PASS|FAIL|ABSENT|WITHHELD|P|F|A|W|X|NE)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    numbers = re.findall(
        r"(?<![A-Za-z])\d{1,3}(?:\.\d+)?(?![A-Za-z])",
        text
    )

    values = []

    for n in numbers:
        try:
            value = float(n)

            # Marks normally fall within this range.
            if 0 <= value <= 100:
                values.append(value)
        except Exception:
            pass

    if len(values) >= 3:
        return values[-3], values[-2], values[-1]

    if len(values) == 2:
        return values[-2], values[-1], None

    if len(values) == 1:
        return values[-1], None, None

    return None, None, None


def normalize_ocr_code(code):

    """
    Correct common OCR mistakes in subject codes.
    """

    code = code.upper().strip()

    replacements = {
        "0": "0",
        "O": "O",
        "I": "I"
    }

    # Remove punctuation around code.
    code = re.sub(r"[^A-Z0-9]", "", code)

    return code


def parse_vtu_subjects(text):

    """
    Main VTU parser.

    It identifies every subject code and uses the following
    subject-code occurrence as the boundary of the next row.
    """

    if not text:
        return []

    # Normalize line endings.
    text = text.replace("\r", "\n")

    raw_lines = text.split("\n")

    lines = []

    for line in raw_lines:
        line = clean_text(line)

        if line:
            lines.append(line)

    subjects = []

    current = None

    for line in lines:

        matches = list(SUBJECT_CODE_PATTERN.finditer(line))

        # ----------------------------------------------------
        # New subject starts
        # ----------------------------------------------------

        if matches:

            # Save previous subject.
            if current is not None:
                subjects.append(current)

            match = matches[0]

            code = normalize_ocr_code(match.group())

            if not is_probable_subject_code(code):
                current = None
                continue

            remainder = line[match.end():].strip()

            current = {
                "subject_code": code,
                "subject_name": remainder,
                "internal": None,
                "external": None,
                "total": None,
                "credits": 0,
                "result": ""
            }

            internal, external, total = extract_numbers_from_end(line)

            # Numbers appearing in the same row.
            if internal is not None:
                current["internal"] = internal

            if external is not None:
                current["external"] = external

            if total is not None:
                current["total"] = total

            # Result
            result_match = re.search(
                r"\b(P|F|A|W|X|NE)\b",
                line,
                flags=re.IGNORECASE
            )

            if result_match:
                current["result"] = result_match.group().upper()

            continue

        # ----------------------------------------------------
        # Continuation line of current subject
        # ----------------------------------------------------

        if current is not None:

            # Ignore headings/notes.
            upper_line = line.upper()

            ignored = [
                "NOMENCLATURE",
                "ABBREVIATIONS",
                "REGISTRAR",
                "NOTE",
                "RESULTS OF SOME",
                "SUBJECT CODE",
                "SUBJECT NAME",
                "INTERNAL MARKS",
                "EXTERNAL MARKS",
                "ANNOUNCED",
                "UNIVERSITY SEAT NUMBER",
                "STUDENT NAME",
                "SEMESTER"
            ]

            if any(word in upper_line for word in ignored):
                continue

            # Look for additional marks.
            internal, external, total = extract_numbers_from_end(line)

            # If this continuation line contains marks,
            # don't add the marks to the subject name.
            if internal is not None or external is not None or total is not None:

                if current["internal"] is None:
                    current["internal"] = internal

                if current["external"] is None:
                    current["external"] = external

                if current["total"] is None:
                    current["total"] = total

                result_match = re.search(
                    r"\b(P|F|A|W|X|NE)\b",
                    line,
                    flags=re.IGNORECASE
                )

                if result_match:
                    current["result"] = result_match.group().upper()

            else:
                # Subject names sometimes wrap to the next line.
                if len(line) > 1:
                    current["subject_name"] += " " + line

    # Save final subject.
    if current is not None:
        subjects.append(current)

    # --------------------------------------------------------
    # Clean subjects
    # --------------------------------------------------------

    cleaned = []

    for subject in subjects:

        name = clean_text(subject["subject_name"])

        # Remove trailing numeric marks accidentally included.
        name = re.sub(
            r"\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}\s*$",
            "",
            name
        )

        # Remove result/date fragments.
        name = re.sub(
            r"\b(P|F|A|W|X|NE)\s+\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            "",
            name,
            flags=re.IGNORECASE
        )

        name = clean_text(name)

        if not name:
            name = "Detected Subject"

        subject["subject_name"] = name

        # If total isn't detected but internal and external are,
        # calculate total.
        if (
            subject["total"] is None
            and subject["internal"] is not None
            and subject["external"] is not None
        ):
            subject["total"] = (
                subject["internal"] + subject["external"]
            )

        cleaned.append(subject)

    # Remove duplicate subject codes.
    unique = []
    seen = set()

    for subject in cleaned:

        key = (
            subject["subject_code"],
            subject["subject_name"].upper()
        )

        if key not in seen:
            seen.add(key)
            unique.append(subject)

    return unique


# ============================================================
# OCR WRAPPER
# ============================================================

def process_marksheet(pdf_path):

    text = get_marksheet_text(pdf_path)

    subjects = parse_vtu_subjects(text)

    return text, subjects


# ============================================================
# SGPA / CGPA
# ============================================================

def grade_point(total):

    if total is None:
        return 0

    total = float(total)

    if total >= 90:
        return 10
    elif total >= 80:
        return 9
    elif total >= 70:
        return 8
    elif total >= 60:
        return 7
    elif total >= 55:
        return 6
    elif total >= 50:
        return 5
    elif total >= 40:
        return 4
    else:
        return 0


def calculate_sgpa(subjects):

    weighted = 0
    credits = 0

    for subject in subjects:

        c = safe_float(subject.get("credits"))

        if c is None or c <= 0:
            continue

        total = safe_float(subject.get("total"))

        if total is None:
            internal = safe_float(subject.get("internal"))
            external = safe_float(subject.get("external"))

            if internal is not None and external is not None:
                total = internal + external

        if total is None:
            continue

        gp = grade_point(total)

        weighted += gp * c
        credits += c

    if credits == 0:
        return 0

    return round(weighted / credits, 2)


def calculate_cgpa(student_id):

    db = get_db()

    semesters = db.execute("""
        SELECT sgpa
        FROM semesters
        WHERE student_id = ?
        AND sgpa > 0
        ORDER BY semester_number
    """, (student_id,)).fetchall()

    db.close()

    if not semesters:
        return 0

    total = sum(float(row["sgpa"]) for row in semesters)

    return round(total / len(semesters), 2)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_classes():

    db = get_db()

    rows = db.execute("""
        SELECT *
        FROM classes
        ORDER BY class_name
    """).fetchall()

    db.close()

    return rows


def get_students(class_id):

    db = get_db()

    rows = db.execute("""
        SELECT *
        FROM students
        WHERE class_id = ?
        ORDER BY name
    """, (class_id,)).fetchall()

    db.close()

    return rows


def get_student(student_id):

    db = get_db()

    row = db.execute("""
        SELECT *
        FROM students
        WHERE id = ?
    """, (student_id,)).fetchone()

    db.close()

    return row


def get_semesters(student_id):

    db = get_db()

    rows = db.execute("""
        SELECT *
        FROM semesters
        WHERE student_id = ?
        ORDER BY semester_number
    """, (student_id,)).fetchall()

    db.close()

    return rows


def get_subjects(semester_id):

    db = get_db()

    rows = db.execute("""
        SELECT *
        FROM subjects
        WHERE semester_id = ?
        ORDER BY rowid
    """, (semester_id,)).fetchall()

    db.close()

    return rows


# ============================================================
# DELETE CLASS
# ============================================================

def delete_class(class_id):

    db = get_db()

    students = db.execute("""
        SELECT id
        FROM students
        WHERE class_id = ?
    """, (class_id,)).fetchall()

    for student in students:

        student_id = student["id"]

        semesters = db.execute("""
            SELECT id
            FROM semesters
            WHERE student_id = ?
        """, (student_id,)).fetchall()

        for semester in semesters:

            db.execute("""
                DELETE FROM subjects
                WHERE semester_id = ?
            """, (semester["id"],))

        db.execute("""
            DELETE FROM semesters
            WHERE student_id = ?
        """, (student_id,))

        db.execute("""
            DELETE FROM backlog_marksheets
            WHERE student_id = ?
        """, (student_id,))

    db.execute("""
        DELETE FROM students
        WHERE class_id = ?
    """, (class_id,))

    db.execute("""
        DELETE FROM classes
        WHERE id = ?
    """, (class_id,))

    db.commit()
    db.close()


# ============================================================
# SESSION STATE
# ============================================================

if "selected_class" not in st.session_state:
    st.session_state.selected_class = None

if "selected_student" not in st.session_state:
    st.session_state.selected_student = None

if "ocr_subjects" not in st.session_state:
    st.session_state.ocr_subjects = []

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if "ocr_filename" not in st.session_state:
    st.session_state.ocr_filename = ""


# ============================================================
# HEADER
# ============================================================

st.title("🎓 EduPredict AI")

st.subheader(
    "Student Academic Performance Prediction & Analytics"
)

st.caption(
    "VTU-focused academic record management, OCR marksheet processing "
    "and student performance analysis."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏫 Classes",
            "👨‍🎓 Students",
            "📚 Previous Semesters",
            "📘 Current Semester",
            "📊 Student Analytics"
        ]
    )


# ============================================================
# PAGE 1 - CLASSES
# ============================================================

def select_student_for_page():

    classes = get_classes()

    if not classes:

        st.warning("Create a class first.")

        return None

    class_options = {
        c["class_name"]: c["id"]
        for c in classes
    }

    selected_class_name = st.selectbox(
        "Select Class",
        list(class_options.keys()),
        key="page_class_selector"
    )

    class_id = class_options[selected_class_name]

    students_list = get_students(class_id)

    if not students_list:

        st.warning(
            "No students found in this class."
        )

        return None

    student_options = {
        f"{s['name']} — {s['usn']}": s["id"]
        for s in students_list
    }

    selected_student_label = st.selectbox(
        "Select Student",
        list(student_options.keys()),
        key="page_student_selector"
    )

    student_id = student_options[
        selected_student_label
    ]

    st.session_state.selected_student = student_id

    return student_id



if page == "🏫 Classes":

    st.header("🏫 Class Management")

    st.write(
        "Create multiple classes and assign a class teacher."
    )

    with st.form("create_class_form"):

        class_name = st.text_input(
            "Class Name",
            placeholder="Example: 3DS-A"
        )

        teacher_name = st.text_input(
            "Class Teacher Name",
            placeholder="Enter class teacher name"
        )

        create = st.form_submit_button(
            "➕ Create Class"
        )

        if create:

            if not class_name.strip():

                st.error("Please enter the class name.")

            else:

                db = get_db()

                db.execute("""
                    INSERT INTO classes
                    (id, class_name, class_teacher, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    uuid.uuid4().hex,
                    class_name.strip(),
                    teacher_name.strip(),
                    datetime.now().isoformat()
                ))

                db.commit()
                db.close()

                st.success(
                    f"Class '{class_name}' created successfully."
                )

                st.rerun()

    st.divider()

    classes = get_classes()

    if not classes:

        st.info("No classes created yet.")

    else:

        st.subheader("📋 Existing Classes")

        for cls in classes:

            students_count = len(
                get_students(cls["id"])
            )

            col1, col2, col3, col4 = st.columns(
                [3, 3, 2, 1]
            )

            with col1:
                st.write(
                    f"**{cls['class_name']}**"
                )

            with col2:
                st.write(
                    f"Teacher: {cls['class_teacher'] or 'Not entered'}"
                )

            with col3:
                st.write(
                    f"Students: {students_count}"
                )

            with col4:

                if st.button(
                    "🗑️",
                    key=f"delete_class_{cls['id']}"
                ):

                    delete_class(cls["id"])

                    if (
                        st.session_state.selected_class
                        == cls["id"]
                    ):
                        st.session_state.selected_class = None

                    st.success("Class deleted.")

                    st.rerun()


# ============================================================
# PAGE 2 - STUDENTS
# ============================================================

elif page == "👨‍🎓 Students":

    st.header("👨‍🎓 Student Management")

    classes = get_classes()

    if not classes:

        st.warning(
            "Create a class first from Class Management."
        )

    else:

        class_options = {
            f"{c['class_name']} — "
            f"{c['class_teacher'] or 'No teacher'}": c["id"]
            for c in classes
        }

        selected_label = st.selectbox(
            "Select Class",
            list(class_options.keys())
        )

        cid = class_options[selected_label]

        st.session_state.selected_class = cid

        st.divider()

        st.subheader("➕ Add Student")

        with st.form("student_form"):

            col1, col2 = st.columns(2)

            with col1:

                name = st.text_input(
                    "Student Name"
                )

                usn = st.text_input(
                    "University Seat Number (USN)"
                )

                admission_year = st.number_input(
                    "Admission Year",
                    min_value=2000,
                    max_value=2100,
                    value=2026
                )

            with col2:

                current_semester = st.number_input(
                    "Current Semester",
                    min_value=1,
                    max_value=8,
                    value=1
                )

                photo = st.file_uploader(
                    "Passport Size Photo "
                    "(JPG/PNG, preferably ≤ 1 MB)",
                    type=["jpg", "jpeg", "png"]
                )

            save_student = st.form_submit_button(
                "💾 Save Student"
            )

            if save_student:

                if not name.strip():

                    st.error("Enter student name.")

                else:

                    db = get_db()

                    try:

                        existing = db.execute("""
                            SELECT id
                            FROM students
                            WHERE class_id = ?
                            AND usn = ?
                        """, (
                            cid,
                            usn.strip()
                        )).fetchone()

                        if existing:

                            st.error(
                                "USN already exists in this class."
                            )

                        else:

                            photo_path = None

                            if photo is not None:

                                if photo.size > 1024 * 1024:

                                    st.error(
                                        "Photo must be 1 MB or below."
                                    )

                                else:

                                    photo_path = save_uploaded_file(
                                        photo,
                                        PHOTO_DIR
                                    )

                            if photo is None or photo.size <= 1024 * 1024:

                                db.execute("""
                                    INSERT INTO students
                                    (
                                        id,
                                        class_id,
                                        name,
                                        usn,
                                        admission_year,
                                        current_semester,
                                        photo_path,
                                        absent_days,
                                        working_days
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    uuid.uuid4().hex,
                                    cid,
                                    name.strip(),
                                    usn.strip(),
                                    admission_year,
                                    current_semester,
                                    photo_path,
                                    0,
                                    90
                                ))

                                db.commit()

                                st.success(
                                    "Student saved permanently."
                                )

                    finally:
                        db.close()

        st.divider()

        students_list = get_students(cid)

        st.subheader("📋 Students")

        if students_list:

            data = []

            for student in students_list:

                data.append({
                    "Name": student["name"],
                    "USN": student["usn"],
                    "Semester": student["current_semester"]
                })

            st.dataframe(
                pd.DataFrame(data),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No students in this class yet.")


# ============================================================
# STUDENT SELECTOR USED BY OTHER PAGES
# ============================================================

# ============================================================
# PAGE 3 - PREVIOUS SEMESTERS
# ============================================================

elif page == "📚 Previous Semesters":

    st.header("📚 Previous Semester Records")

    st.info(
        "Upload the VTU marksheet. The system will automatically "
        "detect subjects, subject codes, internal marks, external "
        "marks and total marks. Normally, you only need to enter "
        "credits."
    )

    student_id = select_student_for_page()

    if student_id:

        student = get_student(student_id)

        st.divider()

        st.subheader("📘 Add Previous Semester")

        semester_number = st.number_input(
            "Semester Number",
            min_value=1,
            max_value=8,
            value=1,
            step=1
        )

        semester_name = st.text_input(
            "Semester Name",
            value=f"Semester {semester_number}"
        )

        uploaded_marksheet = st.file_uploader(
            "📄 Upload VTU Previous Semester Marksheet",
            type=["pdf"],
            key=f"previous_marksheet_{student_id}_{semester_number}"
        )

        if uploaded_marksheet is not None:

            file_signature = (
                uploaded_marksheet.name
                + str(uploaded_marksheet.size)
            )

            if (
                st.session_state.ocr_filename
                != file_signature
            ):

                path = save_uploaded_file(
                    uploaded_marksheet,
                    MARKSHEET_DIR
                )

                with st.spinner(
                    "🔎 Reading VTU marksheet using OCR..."
                ):

                    text, detected_subjects = process_marksheet(
                        path
                    )

                st.session_state.ocr_subjects = detected_subjects
                st.session_state.ocr_text = text
                st.session_state.ocr_filename = file_signature
                st.session_state.ocr_path = path

            subjects = st.session_state.ocr_subjects

            if subjects:

                st.success(
                    f"Automatically detected "
                    f"{len(subjects)} subject(s)."
                )

                st.subheader(
                    "📋 Automatically Detected Subjects"
                )

                st.caption(
                    "Subject information was detected automatically. "
                    "Enter credits only. You can edit any OCR result."
                )

                edited_subjects = []

                for index, subject in enumerate(subjects):

                    st.markdown(
                        f"### Subject {index + 1}"
                    )

                    c1, c2 = st.columns([1, 2])

                    with c1:

                        code = st.text_input(
                            "Subject Code",
                            value=subject.get(
                                "subject_code", ""
                            ),
                            key=f"code_{student_id}_{semester_number}_{index}"
                        )

                    with c2:

                        subject_name = st.text_input(
                            "Subject Name",
                            value=subject.get(
                                "subject_name", ""
                            ),
                            key=f"name_{student_id}_{semester_number}_{index}"
                        )

                    c3, c4, c5, c6 = st.columns(4)

                    with c3:

                        internal = st.number_input(
                            "Internal Marks",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(
                                subject["internal"]
                                if subject["internal"] is not None
                                else 0
                            ),
                            key=f"internal_{student_id}_{semester_number}_{index}"
                        )

                    with c4:

                        external = st.number_input(
                            "External / VTU Marks",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(
                                subject["external"]
                                if subject["external"] is not None
                                else 0
                            ),
                            key=f"external_{student_id}_{semester_number}_{index}"
                        )

                    with c5:

                        detected_total = subject.get(
                            "total"
                        )

                        if detected_total is None:
                            detected_total = (
                                internal + external
                            )

                        total = st.number_input(
                            "Total Marks",
                            min_value=0.0,
                            max_value=200.0,
                            value=float(
                                detected_total
                            ),
                            key=f"total_{student_id}_{semester_number}_{index}"
                        )

                    with c6:

                        credits = st.number_input(
                            "Credits",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(
                                subject.get(
                                    "credits", 0
                                )
                            ),
                            step=0.5,
                            key=f"credits_{student_id}_{semester_number}_{index}"
                        )

                    edited_subjects.append({
                        "subject_code": code,
                        "subject_name": subject_name,
                        "internal": internal,
                        "external": external,
                        "total": total,
                        "credits": credits,
                        "result": subject.get(
                            "result", ""
                        )
                    })

                st.divider()

                st.subheader(
                    "➕ Add Missing Subject"
                )

                with st.expander(
                    "Click here only if OCR missed a subject"
                ):

                    mc1, mc2 = st.columns(2)

                    with mc1:

                        manual_code = st.text_input(
                            "Subject Code",
                            key="manual_subject_code"
                        )

                        manual_name = st.text_input(
                            "Subject Name",
                            key="manual_subject_name"
                        )

                    with mc2:

                        manual_internal = st.number_input(
                            "Internal",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            key="manual_internal"
                        )

                        manual_external = st.number_input(
                            "External",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            key="manual_external"
                        )

                        manual_credits = st.number_input(
                            "Credits",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.5,
                            key="manual_credits"
                        )

                    if st.button(
                        "➕ Add Subject",
                        key="add_missing_subject"
                    ):

                        new_subject = {
                            "subject_code": manual_code,
                            "subject_name": manual_name,
                            "internal": manual_internal,
                            "external": manual_external,
                            "total": (
                                manual_internal
                                + manual_external
                            ),
                            "credits": manual_credits,
                            "result": ""
                        }

                        st.session_state.ocr_subjects.append(
                            new_subject
                        )

                        st.rerun()

                st.divider()

                calculated_sgpa = calculate_sgpa(
                    edited_subjects
                )

                total_credits = sum(
                    safe_float(x["credits"]) or 0
                    for x in edited_subjects
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Subjects Detected",
                        len(edited_subjects)
                    )

                with c2:

                    st.metric(
                        "Total Credits",
                        round(total_credits, 2)
                    )

                with c3:

                    st.metric(
                        "Calculated SGPA",
                        calculated_sgpa
                    )

                st.divider()

                if st.button(
                    "💾 Save Previous Semester",
                    type="primary",
                    key=f"save_semester_{student_id}_{semester_number}"
                ):

                    db = get_db()

                    try:

                        # Remove existing semester with same number.
                        old_sem = db.execute("""
                            SELECT id
                            FROM semesters
                            WHERE student_id = ?
                            AND semester_number = ?
                        """, (
                            student_id,
                            semester_number
                        )).fetchone()

                        if old_sem:

                            db.execute("""
                                DELETE FROM subjects
                                WHERE semester_id = ?
                            """, (
                                old_sem["id"],
                            ))

                            db.execute("""
                                DELETE FROM semesters
                                WHERE id = ?
                            """, (
                                old_sem["id"],
                            ))

                        semester_id = uuid.uuid4().hex

                        db.execute("""
                            INSERT INTO semesters
                            (
                                id,
                                student_id,
                                semester_number,
                                semester_name,
                                sgpa,
                                marksheet_path,
                                is_current,
                                created_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            semester_id,
                            student_id,
                            semester_number,
                            semester_name,
                            calculated_sgpa,
                            st.session_state.get(
                                "ocr_path", ""
                            ),
                            0,
                            datetime.now().isoformat()
                        ))

                        for subject in edited_subjects:

                            db.execute("""
                                INSERT INTO subjects
                                (
                                    id,
                                    semester_id,
                                    subject_code,
                                    subject_name,
                                    internal_marks,
                                    external_marks,
                                    total_marks,
                                    credits,
                                    result
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                uuid.uuid4().hex,
                                semester_id,
                                subject["subject_code"],
                                subject["subject_name"],
                                subject["internal"],
                                subject["external"],
                                subject["total"],
                                subject["credits"],
                                subject["result"]
                            ))

                        db.commit()

                        st.success(
                            f"{semester_name} saved permanently."
                        )

                        st.session_state.ocr_subjects = []
                        st.session_state.ocr_text = ""
                        st.session_state.ocr_filename = ""

                    finally:
                        db.close()

            else:

                st.error(
                    "⚠️ No subjects could be detected automatically."
                )

                st.info(
                    "You can enter the subjects manually below."
                )

                manual_count = st.number_input(
                    "Number of Subjects",
                    min_value=1,
                    max_value=30,
                    value=1,
                    step=1,
                    key=f"manual_subject_count_{student_id}_{semester_number}"
                )

                manual_subjects = []

                for i in range(manual_count):

                    st.markdown(
                        f"### Manual Subject {i + 1}"
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        code = st.text_input(
                            "Subject Code",
                            key=f"fallback_code_{student_id}_{semester_number}_{i}"
                        )

                        name = st.text_input(
                            "Subject Name",
                            key=f"fallback_name_{student_id}_{semester_number}_{i}"
                        )

                    with c2:

                        internal = st.number_input(
                            "Internal",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            key=f"fallback_internal_{student_id}_{semester_number}_{i}"
                        )

                        external = st.number_input(
                            "External",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            key=f"fallback_external_{student_id}_{semester_number}_{i}"
                        )

                        total = st.number_input(
                            "Total",
                            min_value=0.0,
                            max_value=200.0,
                            value=0.0,
                            key=f"fallback_total_{student_id}_{semester_number}_{i}"
                        )

                        credits = st.number_input(
                            "Credits",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.5,
                            key=f"fallback_credits_{student_id}_{semester_number}_{i}"
                        )

                    manual_subjects.append({
                        "subject_code": code,
                        "subject_name": name,
                        "internal": internal,
                        "external": external,
                        "total": total,
                        "credits": credits,
                        "result": ""
                    })

                manual_sgpa = calculate_sgpa(
                    manual_subjects
                )

                st.metric(
                    "Calculated SGPA",
                    manual_sgpa
                )

                if st.button(
                    "💾 Save Manual Semester",
                    type="primary"
                ):

                    db = get_db()

                    try:

                        semester_id = uuid.uuid4().hex

                        db.execute("""
                            INSERT INTO semesters
                            (
                                id,
                                student_id,
                                semester_number,
                                semester_name,
                                sgpa,
                                marksheet_path,
                                is_current,
                                created_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            semester_id,
                            student_id,
                            semester_number,
                            semester_name,
                            manual_sgpa,
                            st.session_state.get(
                                "ocr_path", ""
                            ),
                            0,
                            datetime.now().isoformat()
                        ))

                        for subject in manual_subjects:

                            db.execute("""
                                INSERT INTO subjects
                                (
                                    id,
                                    semester_id,
                                    subject_code,
                                    subject_name,
                                    internal_marks,
                                    external_marks,
                                    total_marks,
                                    credits,
                                    result
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                uuid.uuid4().hex,
                                semester_id,
                                subject["subject_code"],
                                subject["subject_name"],
                                subject["internal"],
                                subject["external"],
                                subject["total"],
                                subject["credits"],
                                subject["result"]
                            ))

                        db.commit()

                        st.success(
                            "Manual semester saved permanently."
                        )

                    finally:
                        db.close()

        # ----------------------------------------------------
        # BACKLOG / RE-ATTEMPT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🔄 Backlog / Re-attempt"
        )

        backlog = st.radio(
            "Does this student have a backlog/re-attempt?",
            ["No", "Yes"],
            horizontal=True,
            key=f"backlog_{student_id}"
        )

        if backlog == "Yes":

            backlog_semester = st.number_input(
                "Backlog Semester",
                min_value=1,
                max_value=8,
                value=1,
                key=f"backlog_sem_{student_id}"
            )

            backlog_subject = st.text_input(
                "Backlog Subject Name",
                key=f"backlog_subject_{student_id}"
            )

            backlog_file = st.file_uploader(
                "Upload Re-attempt / Backlog Marksheet",
                type=["pdf", "jpg", "jpeg", "png"],
                key=f"backlog_file_{student_id}"
            )

            if backlog_file is not None:

                if st.button(
                    "💾 Save Backlog Marksheet",
                    key=f"save_backlog_{student_id}"
                ):

                    path = save_uploaded_file(
                        backlog_file,
                        MARKSHEET_DIR
                    )

                    db = get_db()

                    db.execute("""
                        INSERT INTO backlog_marksheets
                        (
                            id,
                            student_id,
                            semester_number,
                            subject_name,
                            file_path,
                            uploaded_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        uuid.uuid4().hex,
                        student_id,
                        backlog_semester,
                        backlog_subject,
                        path,
                        datetime.now().isoformat()
                    ))

                    db.commit()
                    db.close()

                    st.success(
                        "Backlog/re-attempt marksheet saved."
                    )

        # ----------------------------------------------------
        # EXISTING SEMESTERS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📚 Saved Previous Semesters"
        )

        semesters = get_semesters(student_id)

        if semesters:

            for sem in semesters:

                with st.expander(
                    f"Semester {sem['semester_number']} "
                    f"— SGPA {sem['sgpa']}"
                ):

                    subjects = get_subjects(
                        sem["id"]
                    )

                    table = []

                    for subject in subjects:

                        table.append({
                            "Subject Code": subject["subject_code"],
                            "Subject Name": subject["subject_name"],
                            "Internal": subject["internal_marks"],
                            "External": subject["external_marks"],
                            "Total": subject["total_marks"],
                            "Credits": subject["credits"],
                            "Result": subject["result"]
                        })

                    if table:

                        st.dataframe(
                            pd.DataFrame(table),
                            use_container_width=True,
                            hide_index=True
                        )

                    st.write(
                        f"**SGPA:** {sem['sgpa']}"
                    )

        else:

            st.info(
                "No previous semesters saved yet."
            )


# ============================================================
# PAGE 4 - CURRENT SEMESTER
# ============================================================

elif page == "📘 Current Semester":

    st.header("📘 Current Semester")

    st.info(
        "Current semester marks are entered manually because "
        "the semester is still in progress."
    )

    student_id = select_student_for_page()

    if student_id:

        student = get_student(student_id)

        semester = student["current_semester"]

        st.subheader(
            f"Semester {semester}"
        )

        col1, col2 = st.columns(2)

        with col1:

            absent_days = st.number_input(
                "Number of Absent Days",
                min_value=0,
                max_value=300,
                value=int(
                    student["absent_days"] or 0
                )
            )

        with col2:

            working_days = st.number_input(
                "Working Days",
                min_value=1,
                max_value=365,
                value=int(
                    student["working_days"] or 90
                )
            )

        st.divider()

        subject_count = st.number_input(
            "Number of Current Semester Subjects",
            min_value=1,
            max_value=30,
            value=1
        )

        current_subjects = []

        for i in range(subject_count):

            st.markdown(
                f"### 📘 Subject {i + 1}"
            )

            c1, c2 = st.columns(2)

            with c1:

                subject_code = st.text_input(
                    "Subject Code",
                    key=f"current_code_{student_id}_{i}"
                )

                subject_name = st.text_input(
                    "Subject Name",
                    key=f"current_name_{student_id}_{i}"
                )

                credits = st.number_input(
                    "Credits",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    key=f"current_credit_{student_id}_{i}"
                )

            with c2:

                ia1 = st.number_input(
                    "IA-1 Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"current_ia1_{student_id}_{i}"
                )

                ia2 = st.number_input(
                    "IA-2 Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"current_ia2_{student_id}_{i}"
                )

                final_marks = st.number_input(
                    "Final / VTU Marks",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    key=f"current_final_{student_id}_{i}"
                )

            combined_internal = (
                ia1 + ia2
            ) / 2

            total = (
                combined_internal
                + final_marks
            )

            st.info(
                f"Combined Internal: "
                f"{combined_internal:.2f} | "
                f"Estimated Total: {total:.2f}"
            )

            current_subjects.append({
                "subject_code": subject_code,
                "subject_name": subject_name,
                "internal": combined_internal,
                "external": final_marks,
                "total": total,
                "credits": credits,
                "result": ""
            })

        current_sgpa = calculate_sgpa(
            current_subjects
        )

        st.divider()

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Subjects",
                subject_count
            )

        with c2:

            st.metric(
                "Attendance",
                f"{max(0, working_days - absent_days) / working_days * 100:.1f}%"
            )

        with c3:

            st.metric(
                "Estimated SGPA",
                current_sgpa
            )

        if st.button(
            "💾 Save Current Semester",
            type="primary"
        ):

            db = get_db()

            try:

                existing = db.execute("""
                    SELECT id
                    FROM semesters
                    WHERE student_id = ?
                    AND semester_number = ?
                """, (
                    student_id,
                    semester
                )).fetchone()

                if existing:

                    semester_id = existing["id"]

                    db.execute("""
                        DELETE FROM subjects
                        WHERE semester_id = ?
                    """, (
                        semester_id,
                    ))

                    db.execute("""
                        UPDATE semesters
                        SET semester_name = ?,
                            sgpa = ?,
                            is_current = 1
                        WHERE id = ?
                    """, (
                        f"Semester {semester}",
                        current_sgpa,
                        semester_id
                    ))

                else:

                    semester_id = uuid.uuid4().hex

                    db.execute("""
                        INSERT INTO semesters
                        (
                            id,
                            student_id,
                            semester_number,
                            semester_name,
                            sgpa,
                            marksheet_path,
                            is_current,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        semester_id,
                        student_id,
                        semester,
                        f"Semester {semester}",
                        current_sgpa,
                        "",
                        1,
                        datetime.now().isoformat()
                    ))

                for subject in current_subjects:

                    db.execute("""
                        INSERT INTO subjects
                        (
                            id,
                            semester_id,
                            subject_code,
                            subject_name,
                            internal_marks,
                            external_marks,
                            total_marks,
                            credits,
                            result
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        uuid.uuid4().hex,
                        semester_id,
                        subject["subject_code"],
                        subject["subject_name"],
                        subject["internal"],
                        subject["external"],
                        subject["total"],
                        subject["credits"],
                        ""
                    ))

                db.execute("""
                    UPDATE students
                    SET absent_days = ?,
                        working_days = ?
                    WHERE id = ?
                """, (
                    absent_days,
                    working_days,
                    student_id
                ))

                db.commit()

                st.success(
                    "Current semester saved permanently."
                )

            finally:
                db.close()


# ============================================================
# PAGE 5 - ANALYTICS
# ============================================================

elif page == "📊 Student Analytics":

    st.header("📊 Student Analytics")

    student_id = select_student_for_page()

    if student_id:

        student = get_student(student_id)

        semesters = get_semesters(
            student_id
        )

        cgpa = calculate_cgpa(
            student_id
        )

        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 3])

        with col1:

            if (
                student["photo_path"]
                and os.path.exists(
                    student["photo_path"]
                )
            ):

                st.image(
                    student["photo_path"],
                    width=150
                )

        with col2:

            st.subheader(
                student["name"]
            )

            st.write(
                f"**USN:** {student['usn']}"
            )

            st.write(
                f"**Current Semester:** "
                f"{student['current_semester']}"
            )

            st.metric(
                "Current CGPA",
                cgpa
            )

        # ----------------------------------------------------
        # SEMESTER PERFORMANCE
        # ----------------------------------------------------

        if semesters:

            st.subheader(
                "📈 Semester Performance"
            )

            performance = []

            for sem in semesters:

                performance.append({
                    "Semester":
                        sem["semester_number"],
                    "SGPA":
                        sem["sgpa"]
                })

            df = pd.DataFrame(
                performance
            )

            st.line_chart(
                df.set_index("Semester")
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # ATTENDANCE
        # ----------------------------------------------------

        working_days = student[
            "working_days"
        ] or 90

        absent_days = student[
            "absent_days"
        ] or 0

        attendance = (
            max(
                0,
                working_days - absent_days
            )
            / working_days
            * 100
        )

        st.subheader(
            "🕒 Attendance"
        )

        st.metric(
            "Attendance Percentage",
            f"{attendance:.2f}%"
        )

        # ----------------------------------------------------
        # SIMPLE PERFORMANCE PREDICTION
        # ----------------------------------------------------

        st.subheader(
            "🤖 Academic Performance Prediction"
        )

        if cgpa >= 8.5:

            risk = "Low Risk"
            recommendation = (
                "Student is currently maintaining "
                "strong academic performance."
            )

        elif cgpa >= 7:

            risk = "Moderate Risk"
            recommendation = (
                "Student is performing reasonably well. "
                "Consistent preparation can improve the CGPA."
            )

        elif cgpa >= 5.5:

            risk = "Needs Attention"
            recommendation = (
                "Additional academic support and "
                "regular monitoring are recommended."
            )

        else:

            risk = "High Academic Risk"
            recommendation = (
                "The student may require focused academic "
                "support and closer monitoring."
            )

        st.metric(
            "Academic Risk Level",
            risk
        )

        st.info(
            recommendation
        )

        # ----------------------------------------------------
        # SUBJECT PERFORMANCE
        # ----------------------------------------------------

        st.subheader(
            "📚 Subject-wise Performance"
        )

        all_subjects = []

        for sem in semesters:

            subjects = get_subjects(
                sem["id"]
            )

            for subject in subjects:

                all_subjects.append({
                    "Semester":
                        sem["semester_number"],
                    "Subject Code":
                        subject["subject_code"],
                    "Subject":
                        subject["subject_name"],
                    "Internal":
                        subject["internal_marks"],
                    "External":
                        subject["external_marks"],
                    "Total":
                        subject["total_marks"],
                    "Credits":
                        subject["credits"],
                    "Result":
                        subject["result"]
                })

        if all_subjects:

            st.dataframe(
                pd.DataFrame(all_subjects),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No subject records available yet."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EduPredict AI • VTU Academic Performance Management System"
)