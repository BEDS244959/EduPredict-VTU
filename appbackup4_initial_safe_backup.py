import streamlit as st
import sqlite3
import uuid
import os
import re
from datetime import datetime

import pandas as pd
import fitz
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

    db.execute("""
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

        match = re.search(
            r"\d+(?:\.\d+)?",
            value
        )

        if match:
            return float(match.group())

    except Exception:
        pass

    return None


def save_uploaded_file(uploaded_file, folder):

    if uploaded_file is None:
        return None

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex
        + extension
    )

    path = os.path.join(
        folder,
        filename
    )

    with open(path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    return path


# ============================================================
# OCR
# ============================================================

def extract_pdf_text(pdf_path):

    text = ""

    try:

        doc = fitz.open(pdf_path)

        for page in doc:

            page_text = page.get_text(
                "text"
            )

            if page_text:
                text += "\n" + page_text

        doc.close()

    except Exception:

        return ""

    return text


def extract_pdf_ocr(pdf_path):

    all_text = ""

    try:

        doc = fitz.open(pdf_path)

        for page in doc:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    2.5,
                    2.5
                ),
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

        st.warning(
            f"OCR error: {e}"
        )

    return all_text


def get_marksheet_text(pdf_path):

    text = extract_pdf_text(
        pdf_path
    )

    if len(text.strip()) < 100:

        text = extract_pdf_ocr(
            pdf_path
        )

    return text


# ============================================================
# VTU SUBJECT DETECTION
# ============================================================

SUBJECT_CODE_PATTERN = re.compile(
    r"\b[A-Z]{2,8}\d{2,4}[A-Z]{0,3}\b"
)


def is_probable_subject_code(code):

    if not code:
        return False

    code = code.upper()

    if len(code) < 5:
        return False

    if not re.search(
        r"\d",
        code
    ):
        return False

    return bool(
        re.match(
            r"^[A-Z]{2,8}\d{2,4}[A-Z]{0,3}$",
            code
        )
    )


def extract_numbers_from_end(text):
    """Extract a likely Internal / External / Total triple from one OCR line."""
    if not text:
        return None, None, None

    text = clean_text(text)
    text = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", text)
    text = re.sub(
        r"\b(PASS|FAIL|ABSENT|WITHHELD|P|F|A|W|X|NE)\b",
        " ", text, flags=re.IGNORECASE
    )

    raw = re.findall(r"(?<![A-Za-z])\d{1,3}(?:\.\d+)?(?![A-Za-z])", text)
    values = []
    for value in raw:
        try:
            number = float(value)
            if 0 <= number <= 100:
                values.append(number)
        except (ValueError, TypeError):
            pass

    # Prefer a mathematically consistent triple: Internal + External = Total.
    for i in range(len(values) - 2):
        a, b, c = values[i:i + 3]
        if abs((a + b) - c) <= 1:
            return a, b, c

    if len(values) >= 3:
        a, b, c = values[-3:]
        # If the third value looks like a total, use it; otherwise calculate it.
        total = c if abs((a + b) - c) <= 2 else a + b
        return a, b, total

    if len(values) == 2:
        return values[0], values[1], values[0] + values[1]

    if len(values) == 1:
        return values[0], None, None

    return None, None, None


def normalize_mark_value(value):
    """Return a numeric mark in the valid 0-100 range."""
    value = safe_float(value)
    if value is None:
        return None
    if value < 0 or value > 100:
        return None
    return value


def extract_marks_from_pdf_layout(pdf_path):
    """
    Try to read Internal / External / Total using PDF word coordinates.
    This is more reliable than taking the last three numbers from OCR text
    because it uses the actual column positions when the PDF contains text.
    """
    subjects = []

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            words = page.get_text("words")
            if not words:
                continue

            # word tuple: x0, y0, x1, y1, text, block, line, word_no
            header_x = {
                "internal": None,
                "external": None,
                "total": None,
            }

            for w in words:
                text = clean_text(w[4]).upper()
                x_center = (w[0] + w[2]) / 2

                if "INTERNAL" in text or text in {"INT", "CIE"}:
                    header_x["internal"] = x_center
                elif "EXTERNAL" in text or text in {"EXT", "SEE", "UNIVERSITY"}:
                    header_x["external"] = x_center
                elif text == "TOTAL" or "TOTAL" in text:
                    header_x["total"] = x_center

            # Without useful headers, coordinate matching is unsafe.
            if header_x["internal"] is None or header_x["external"] is None:
                continue

            # Group words by approximate visual line.
            rows = {}
            for w in words:
                y = round(w[1] / 3) * 3
                rows.setdefault(y, []).append(w)

            for y, row_words in rows.items():
                row_words = sorted(row_words, key=lambda w: w[0])
                row_text = " ".join(clean_text(w[4]) for w in row_words)

                code_match = SUBJECT_CODE_PATTERN.search(row_text.upper())
                if not code_match:
                    continue

                code = normalize_ocr_code(code_match.group())
                if not is_probable_subject_code(code):
                    continue

                code_word = None
                for w in row_words:
                    if normalize_ocr_code(w[4]) == code:
                        code_word = w
                        break

                if code_word is None:
                    continue

                # Collect numeric words from the same visual row.
                numeric_words = []
                for w in row_words:
                    raw = clean_text(w[4]).replace("O", "0")
                    if re.fullmatch(r"\d{1,3}(?:\.\d+)?", raw):
                        try:
                            number = float(raw)
                            if 0 <= number <= 100:
                                numeric_words.append((number, (w[0] + w[2]) / 2))
                        except ValueError:
                            pass

                if len(numeric_words) < 2:
                    continue

                def closest(target):
                    return min(numeric_words, key=lambda item: abs(item[1] - target))

                internal_item = closest(header_x["internal"])
                external_item = closest(header_x["external"])

                internal = internal_item[0]
                external = external_item[0]
                total = None

                if header_x["total"] is not None:
                    total = closest(header_x["total"])[0]

                calculated = internal + external
                if total is None or abs(total - calculated) > 1:
                    total = calculated

                # Avoid duplicate subject codes from repeated page/header content.
                if not any(x["subject_code"] == code for x in subjects):
                    name_parts = []
                    for w in row_words:
                        t = clean_text(w[4])
                        if not t:
                            continue
                        if t.upper() == code:
                            continue
                        if re.fullmatch(r"\d{1,3}(?:\.\d+)?", t):
                            continue
                        name_parts.append(t)

                    name = clean_text(" ".join(name_parts))
                    subjects.append({
                        "subject_code": code,
                        "subject_name": name or "Detected Subject",
                        "internal": internal,
                        "external": external,
                        "total": total,
                        "credits": 0,
                        "result": ""
                    })

        doc.close()

    except Exception:
        return []

    return subjects

def normalize_ocr_code(code):

    code = code.upper().strip()

    code = re.sub(
        r"[^A-Z0-9]",
        "",
        code
    )

    return code


def parse_vtu_subjects(text):

    if not text:
        return []

    text = text.replace(
        "\r",
        "\n"
    )

    raw_lines = text.split(
        "\n"
    )

    lines = []

    for line in raw_lines:

        line = clean_text(
            line
        )

        if line:
            lines.append(line)

    subjects = []

    current = None

    for line in lines:

        matches = list(
            SUBJECT_CODE_PATTERN.finditer(
                line
            )
        )

        if matches:

            if current is not None:
                subjects.append(
                    current
                )

            match = matches[0]

            code = normalize_ocr_code(
                match.group()
            )

            if not is_probable_subject_code(
                code
            ):

                current = None
                continue

            remainder = line[
                match.end():
            ].strip()

            current = {
                "subject_code": code,
                "subject_name": remainder,
                "internal": None,
                "external": None,
                "total": None,
                "credits": 0,
                "result": ""
            }

            internal, external, total = (
                extract_numbers_from_end(
                    line
                )
            )

            current["internal"] = internal
            current["external"] = external
            current["total"] = total

            result_match = re.search(
                r"\b(P|F|A|W|X|NE)\b",
                line,
                flags=re.IGNORECASE
            )

            if result_match:

                current["result"] = (
                    result_match.group()
                    .upper()
                )

            continue

        if current is not None:

            upper_line = line.upper()

            ignored = [
                "NOMENCLATURE",
                "ABBREVIATIONS",
                "REGISTRAR",
                "NOTE",
                "SUBJECT CODE",
                "SUBJECT NAME",
                "INTERNAL MARKS",
                "EXTERNAL MARKS",
                "ANNOUNCED",
                "UNIVERSITY SEAT NUMBER",
                "STUDENT NAME",
                "SEMESTER"
            ]

            if any(
                word in upper_line
                for word in ignored
            ):
                continue

            internal, external, total = (
                extract_numbers_from_end(
                    line
                )
            )

            if (
                internal is not None
                or external is not None
                or total is not None
            ):

                if current["internal"] is None:
                    current["internal"] = internal

                if current["external"] is None:
                    current["external"] = external

                if current["total"] is None:
                    current["total"] = total

            else:

                if len(line) > 1:

                    current[
                        "subject_name"
                    ] += " " + line

    if current is not None:
        subjects.append(
            current
        )

    cleaned = []

    for subject in subjects:

        name = clean_text(
            subject["subject_name"]
        )

        name = re.sub(
            r"\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}\s*$",
            "",
            name
        )

        name = clean_text(
            name
        )

        if not name:
            name = "Detected Subject"

        subject["subject_name"] = name

        if (
            subject["total"] is None
            and subject["internal"] is not None
            and subject["external"] is not None
        ):

            subject["total"] = (
                subject["internal"]
                + subject["external"]
            )

        cleaned.append(
            subject
        )

    unique = []

    seen = set()

    for subject in cleaned:

        key = (
            subject["subject_code"],
            subject["subject_name"].upper()
        )

        if key not in seen:

            seen.add(key)

            unique.append(
                subject
            )

    return unique


def process_marksheet(pdf_path):
    """Read a marksheet with layout-aware extraction first, then OCR/text fallback."""
    text = get_marksheet_text(pdf_path)

    # First attempt: use PDF coordinates/columns.
    subjects = extract_marks_from_pdf_layout(pdf_path)

    # Fallback: text/OCR parser.
    if not subjects:
        subjects = parse_vtu_subjects(text)

    # Always make Total authoritative from Internal + External.
    for subject in subjects:
        internal = normalize_mark_value(subject.get("internal"))
        external = normalize_mark_value(subject.get("external"))
        subject["internal"] = internal
        subject["external"] = external

        if internal is not None and external is not None:
            subject["total"] = round(internal + external, 2)
        else:
            subject["total"] = normalize_mark_value(subject.get("total"))

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

    return 0


def calculate_sgpa(subjects):

    weighted = 0
    credits = 0

    for subject in subjects:

        c = safe_float(
            subject.get("credits")
        )

        if c is None or c <= 0:
            continue

        total = safe_float(
            subject.get("total")
        )

        if total is None:

            internal = safe_float(
                subject.get("internal")
            )

            external = safe_float(
                subject.get("external")
            )

            if (
                internal is not None
                and external is not None
            ):

                total = (
                    internal
                    + external
                )

        if total is None:
            continue

        gp = grade_point(
            total
        )

        weighted += gp * c
        credits += c

    if credits == 0:
        return 0

    return round(
        weighted / credits,
        2
    )


def calculate_cgpa(student_id):

    db = get_db()

    semesters = db.execute("""
        SELECT sgpa
        FROM semesters
        WHERE student_id = ?
        AND sgpa > 0
        ORDER BY semester_number
    """, (
        student_id,
    )).fetchall()

    db.close()

    if not semesters:
        return 0

    return round(
        sum(
            float(row["sgpa"])
            for row in semesters
        ) / len(semesters),
        2
    )


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
    """, (
        class_id,
    )).fetchall()

    db.close()

    return rows


def get_student(student_id):

    db = get_db()

    row = db.execute("""
        SELECT *
        FROM students
        WHERE id = ?
    """, (
        student_id,
    )).fetchone()

    db.close()

    return row


def get_semesters(student_id):

    db = get_db()

    rows = db.execute("""
        SELECT *
        FROM semesters
        WHERE student_id = ?
        ORDER BY semester_number
    """, (
        student_id,
    )).fetchall()

    db.close()

    return rows


def get_subjects(semester_id):

    db = get_db()

    rows = db.execute("""
        SELECT *
        FROM subjects
        WHERE semester_id = ?
        ORDER BY rowid
    """, (
        semester_id,
    )).fetchall()

    db.close()

    return rows


# ============================================================
# DELETE STUDENT
# ============================================================

def delete_student(student_id):

    db = get_db()

    student = db.execute(
        """
        SELECT photo_path
        FROM students
        WHERE id = ?
        """,
        (student_id,)
    ).fetchone()

    semester_rows = db.execute(
        """
        SELECT id, marksheet_path
        FROM semesters
        WHERE student_id = ?
        """,
        (student_id,)
    ).fetchall()

    backlog_rows = db.execute(
        """
        SELECT file_path
        FROM backlog_marksheets
        WHERE student_id = ?
        """,
        (student_id,)
    ).fetchall()

    for semester in semester_rows:

        db.execute(
            """
            DELETE FROM subjects
            WHERE semester_id = ?
            """,
            (semester["id"],)
        )

    db.execute(
        """
        DELETE FROM semesters
        WHERE student_id = ?
        """,
        (student_id,)
    )

    db.execute(
        """
        DELETE FROM backlog_marksheets
        WHERE student_id = ?
        """,
        (student_id,)
    )

    db.execute(
        """
        DELETE FROM attendance
        WHERE student_id = ?
        """,
        (student_id,)
    )

    db.execute(
        """
        DELETE FROM students
        WHERE id = ?
        """,
        (student_id,)
    )

    db.commit()
    db.close()

    paths = []

    if student and student["photo_path"]:
        paths.append(
            student["photo_path"]
        )

    paths.extend(
        row["marksheet_path"]
        for row in semester_rows
        if row["marksheet_path"]
    )

    paths.extend(
        row["file_path"]
        for row in backlog_rows
        if row["file_path"]
    )

    for path in paths:

        try:

            if path and os.path.isfile(path):
                os.remove(path)

        except OSError:
            pass


# ============================================================
# DELETE CLASS
# ============================================================

def delete_class(class_id):

    db = get_db()

    students = db.execute(
        """
        SELECT id
        FROM students
        WHERE class_id = ?
        """,
        (class_id,)
    ).fetchall()

    for student in students:

        student_id = student["id"]

        semesters = db.execute(
            """
            SELECT id
            FROM semesters
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchall()

        for semester in semesters:

            db.execute(
                """
                DELETE FROM subjects
                WHERE semester_id = ?
                """,
                (semester["id"],)
            )

        db.execute(
            """
            DELETE FROM semesters
            WHERE student_id = ?
            """,
            (student_id,)
        )

        db.execute(
            """
            DELETE FROM backlog_marksheets
            WHERE student_id = ?
            """,
            (student_id,)
        )

    db.execute(
        """
        DELETE FROM attendance
        WHERE class_id = ?
        """,
        (class_id,)
    )

    db.execute(
        """
        DELETE FROM students
        WHERE class_id = ?
        """,
        (class_id,)
    )

    db.execute(
        """
        DELETE FROM classes
        WHERE id = ?
        """,
        (class_id,)
    )

    db.commit()
    db.close()


# ============================================================
# ATTENDANCE FUNCTIONS
# ============================================================

def attendance_summary(
    student_id,
    month=None,
    year=None
):

    db = get_db()

    query = """
        SELECT
            COUNT(*) AS total,
            SUM(
                CASE
                    WHEN status = 'Present'
                    THEN 1
                    ELSE 0
                END
            ) AS present,
            SUM(
                CASE
                    WHEN status = 'Absent'
                    THEN 1
                    ELSE 0
                END
            ) AS absent
        FROM attendance
        WHERE student_id = ?
    """

    params = [
        student_id
    ]

    if (
        month is not None
        and year is not None
    ):

        query += """
            AND strftime(
                '%m',
                attendance_date
            ) = ?
            AND strftime(
                '%Y',
                attendance_date
            ) = ?
        """

        params.extend([
            f"{int(month):02d}",
            str(int(year))
        ])

    row = db.execute(
        query,
        params
    ).fetchone()

    db.close()

    total = int(
        row["total"] or 0
    )

    present = int(
        row["present"] or 0
    )

    absent = int(
        row["absent"] or 0
    )

    percentage = (
        round(
            (present / total) * 100,
            2
        )
        if total
        else 0
    )

    return (
        total,
        present,
        absent,
        percentage
    )


def subject_attendance(student_id):

    db = get_db()

    rows = db.execute("""
        SELECT
            subject_name,
            COUNT(*) AS total,
            SUM(
                CASE
                    WHEN status = 'Present'
                    THEN 1
                    ELSE 0
                END
            ) AS present,
            SUM(
                CASE
                    WHEN status = 'Absent'
                    THEN 1
                    ELSE 0
                END
            ) AS absent
        FROM attendance
        WHERE student_id = ?
        GROUP BY subject_name
        ORDER BY subject_name
    """, (
        student_id,
    )).fetchall()

    db.close()

    result = []

    for row in rows:

        total = int(
            row["total"] or 0
        )

        present = int(
            row["present"] or 0
        )

        absent = int(
            row["absent"] or 0
        )

        result.append({
            "Subject":
                row["subject_name"],
            "Total Hours":
                total,
            "Present":
                present,
            "Absent":
                absent,
            "Attendance %":
                round(
                    (present / total) * 100,
                    2
                )
                if total
                else 0
        })

    return result


def monthly_attendance(student_id):

    db = get_db()

    rows = db.execute("""
        SELECT
            strftime(
                '%Y-%m',
                attendance_date
            ) AS month,
            COUNT(*) AS total,
            SUM(
                CASE
                    WHEN status = 'Present'
                    THEN 1
                    ELSE 0
                END
            ) AS present,
            SUM(
                CASE
                    WHEN status = 'Absent'
                    THEN 1
                    ELSE 0
                END
            ) AS absent
        FROM attendance
        WHERE student_id = ?
        GROUP BY strftime(
            '%Y-%m',
            attendance_date
        )
        ORDER BY month
    """, (
        student_id,
    )).fetchall()

    db.close()

    result = []

    for row in rows:

        total = int(
            row["total"] or 0
        )

        present = int(
            row["present"] or 0
        )

        absent = int(
            row["absent"] or 0
        )

        result.append({
            "Month":
                row["month"],
            "Total Hours":
                total,
            "Present":
                present,
            "Absent":
                absent,
            "Attendance %":
                round(
                    (present / total) * 100,
                    2
                )
                if total
                else 0
        })

    return result


def class_attendance_report(
    class_id,
    month,
    year
):

    db = get_db()

    rows = db.execute("""
        SELECT
            s.id AS student_id,
            s.name,
            s.usn,
            a.subject_name,
            COUNT(*) AS total_hours,
            SUM(
                CASE
                    WHEN a.status = 'Present'
                    THEN 1
                    ELSE 0
                END
            ) AS present_hours,
            SUM(
                CASE
                    WHEN a.status = 'Absent'
                    THEN 1
                    ELSE 0
                END
            ) AS absent_hours
        FROM attendance a
        JOIN students s
            ON s.id = a.student_id
        WHERE a.class_id = ?
        AND strftime(
            '%m',
            a.attendance_date
        ) = ?
        AND strftime(
            '%Y',
            a.attendance_date
        ) = ?
        GROUP BY
            s.id,
            s.name,
            s.usn,
            a.subject_name
        ORDER BY
            s.name,
            a.subject_name
    """, (
        class_id,
        f"{int(month):02d}",
        str(int(year))
    )).fetchall()

    db.close()

    data = []

    for row in rows:

        total = int(
            row["total_hours"] or 0
        )

        present = int(
            row["present_hours"] or 0
        )

        absent = int(
            row["absent_hours"] or 0
        )

        percentage = (
            round(
                present / total * 100,
                2
            )
            if total
            else 0
        )

        data.append({
            "Student":
                row["name"],
            "USN":
                row["usn"],
            "Subject":
                row["subject_name"],
            "Total Hours":
                total,
            "Present":
                present,
            "Absent":
                absent,
            "Attendance %":
                percentage
        })

    return pd.DataFrame(data)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "selected_class": None,
    "selected_student": None,
    "ocr_subjects": [],
    "ocr_text": "",
    "ocr_filename": "",
    "ocr_path": ""
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.title("🎓 EduPredict AI")

st.subheader(
    "Student Academic Performance Prediction & Analytics"
)

st.caption(
    "VTU-focused academic record management, "
    "OCR marksheet processing, attendance monitoring "
    "and student performance analytics."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "🏫 Classes",
            "👨‍🎓 Students",
            "📚 Previous Semesters",
            "📘 Current Semester",
            "📅 Daily Attendance",
            "📊 Attendance Dashboard",
            "📈 Student Analytics"
        ]
    )

    st.divider()

    st.caption(
        "EduPredict AI"
    )

    st.caption(
        "Academic + Attendance Intelligence"
    )


# ============================================================
# COMMON STUDENT SELECTOR
# ============================================================

def select_student_for_page():

    classes = get_classes()

    if not classes:

        st.warning(
            "Create a class first."
        )

        return None

    class_options = {
        c["class_name"]: c["id"]
        for c in classes
    }

    selected_class_name = st.selectbox(
        "Select Class",
        list(
            class_options.keys()
        ),
        key=f"{page}_class_selector"
    )

    class_id = class_options[
        selected_class_name
    ]

    students_list = get_students(
        class_id
    )

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
        list(
            student_options.keys()
        ),
        key=f"{page}_student_selector"
    )

    student_id = student_options[
        selected_student_label
    ]

    st.session_state.selected_student = (
        student_id
    )

    return student_id


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 EduPredict AI Dashboard")

    st.write(
        "Central overview of students, academic performance "
        "and attendance."
    )

    classes = get_classes()

    db = get_db()

    total_students = db.execute(
        "SELECT COUNT(*) AS c FROM students"
    ).fetchone()["c"]

    total_semesters = db.execute(
        "SELECT COUNT(*) AS c FROM semesters"
    ).fetchone()["c"]

    total_attendance = db.execute(
        "SELECT COUNT(*) AS c FROM attendance"
    ).fetchone()["c"]

    present_attendance = db.execute("""
        SELECT COUNT(*) AS c
        FROM attendance
        WHERE status = 'Present'
    """).fetchone()["c"]

    db.close()

    overall_attendance = (
        present_attendance
        / total_attendance
        * 100
        if total_attendance
        else 0
    )

    # --------------------------------------------------------
    # MAIN CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏫 Total Classes",
        len(classes)
    )

    c2.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    c3.metric(
        "📚 Semester Records",
        total_semesters
    )

    c4.metric(
        "🕒 Overall Attendance",
        f"{overall_attendance:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # ATTENDANCE STATUS
    # --------------------------------------------------------

    st.subheader(
        "📅 Attendance Overview"
    )

    if total_attendance:

        a, b, c = st.columns(3)

        a.metric(
            "Total Attendance Entries",
            total_attendance
        )

        b.metric(
            "Present",
            present_attendance
        )

        c.metric(
            "Absent",
            total_attendance
            - present_attendance
        )

    else:

        st.info(
            "No attendance records yet. "
            "Start using Daily Attendance."
        )

    st.divider()

    # --------------------------------------------------------
    # CLASS SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "🏫 Class-wise Summary"
    )

    if classes:

        class_rows = []

        for cls in classes:

            students = get_students(
                cls["id"]
            )

            class_total = 0
            class_present = 0

            db = get_db()

            attendance_rows = db.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(
                        CASE
                            WHEN status = 'Present'
                            THEN 1
                            ELSE 0
                        END
                    ) AS present
                FROM attendance
                WHERE class_id = ?
            """, (
                cls["id"],
            )).fetchone()

            db.close()

            class_total = int(
                attendance_rows["total"] or 0
            )

            class_present = int(
                attendance_rows["present"] or 0
            )

            class_percentage = (
                class_present
                / class_total
                * 100
                if class_total
                else 0
            )

            class_rows.append({
                "Class":
                    cls["class_name"],
                "Class Teacher":
                    cls["class_teacher"]
                    or "Not assigned",
                "Students":
                    len(students),
                "Attendance Entries":
                    class_total,
                "Attendance %":
                    round(
                        class_percentage,
                        2
                    )
            })

        st.dataframe(
            pd.DataFrame(class_rows),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # LOW ATTENDANCE ALERTS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🚨 Attendance Alerts"
    )

    threshold = st.slider(
        "Attendance Alert Threshold",
        min_value=50,
        max_value=90,
        value=75,
        step=1
    )

    low_students = []

    for cls in classes:

        students = get_students(
            cls["id"]
        )

        for student in students:

            total, present, absent, percentage = (
                attendance_summary(
                    student["id"]
                )
            )

            if total and percentage < threshold:

                low_students.append({
                    "Student":
                        student["name"],
                    "USN":
                        student["usn"],
                    "Class":
                        cls["class_name"],
                    "Total Hours":
                        total,
                    "Present":
                        present,
                    "Absent":
                        absent,
                    "Attendance %":
                        percentage
                })

    if low_students:

        st.warning(
            f"{len(low_students)} student(s) "
            f"are below {threshold}% attendance."
        )

        st.dataframe(
            pd.DataFrame(
                low_students
            ).sort_values(
                "Attendance %"
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            f"No students are below {threshold}% "
            "based on recorded attendance."
        )

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "⚡ Quick Actions"
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.info(
        "🏫 Create and manage classes"
    )

    q2.info(
        "👨‍🎓 Add or delete students"
    )

    q3.info(
        "📅 Record daily attendance"
    )

    q4.info(
        "📊 Analyze student performance"
    )


# ============================================================
# PAGE 2 - CLASSES
# ============================================================

elif page == "🏫 Classes":

    st.header(
        "🏫 Class Management"
    )

    st.write(
        "Create multiple classes and assign "
        "a class teacher."
    )

    with st.form(
        "create_class_form"
    ):

        class_name = st.text_input(
            "Class Name",
            placeholder="Example: 3DS-A"
        )

        teacher_name = st.text_input(
            "Class Teacher Name"
        )

        create = st.form_submit_button(
            "➕ Create Class"
        )

        if create:

            if not class_name.strip():

                st.error(
                    "Please enter the class name."
                )

            else:

                db = get_db()

                db.execute("""
                    INSERT INTO classes
                    (
                        id,
                        class_name,
                        class_teacher,
                        created_at
                    )
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
                    f"Class '{class_name}' "
                    "created successfully."
                )

                st.rerun()

    st.divider()

    classes = get_classes()

    if not classes:

        st.info(
            "No classes created yet."
        )

    else:

        st.subheader(
            "📋 Existing Classes"
        )

        for cls in classes:

            students_count = len(
                get_students(
                    cls["id"]
                )
            )

            c1, c2, c3, c4 = st.columns(
                [3, 3, 2, 1]
            )

            with c1:

                st.write(
                    f"**{cls['class_name']}**"
                )

            with c2:

                st.write(
                    "Teacher: "
                    + (
                        cls["class_teacher"]
                        or "Not entered"
                    )
                )

            with c3:

                st.write(
                    f"Students: {students_count}"
                )

            with c4:

                if st.button(
                    "🗑️",
                    key=f"delete_class_{cls['id']}"
                ):

                    delete_class(
                        cls["id"]
                    )

                    st.success(
                        "Class deleted."
                    )

                    st.rerun()


# ============================================================
# PAGE 3 - STUDENTS
# ============================================================

elif page == "👨‍🎓 Students":

    st.header(
        "👨‍🎓 Student Management"
    )

    classes = get_classes()

    if not classes:

        st.warning(
            "Create a class first."
        )

    else:

        class_options = {
            f"{c['class_name']} — "
            f"{c['class_teacher'] or 'No teacher'}":
            c["id"]
            for c in classes
        }

        selected_label = st.selectbox(
            "Select Class",
            list(
                class_options.keys()
            )
        )

        cid = class_options[
            selected_label
        ]

        st.session_state.selected_class = cid

        st.divider()

        st.subheader(
            "➕ Add Student"
        )

        with st.form(
            "student_form"
        ):

            c1, c2 = st.columns(2)

            with c1:

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

            with c2:

                current_semester = st.number_input(
                    "Current Semester",
                    min_value=1,
                    max_value=8,
                    value=1
                )

                photo = st.file_uploader(
                    "Passport Size Photo",
                    type=[
                        "jpg",
                        "jpeg",
                        "png"
                    ]
                )

            save_student = st.form_submit_button(
                "💾 Save Student"
            )

            if save_student:

                if not name.strip():

                    st.error(
                        "Enter student name."
                    )

                else:

                    db = get_db()

                    try:

                        existing = db.execute(
                            """
                            SELECT id
                            FROM students
                            WHERE class_id = ?
                            AND usn = ?
                            """,
                            (
                                cid,
                                usn.strip()
                            )
                        ).fetchone()

                        if existing:

                            st.error(
                                "USN already exists "
                                "in this class."
                            )

                        else:

                            photo_path = None

                            if photo is not None:

                                if photo.size > 1024 * 1024:

                                    st.error(
                                        "Photo must be "
                                        "1 MB or below."
                                    )

                                else:

                                    photo_path = (
                                        save_uploaded_file(
                                            photo,
                                            PHOTO_DIR
                                        )
                                    )

                            if (
                                photo is None
                                or photo.size <= 1024 * 1024
                            ):

                                db.execute(
                                    """
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
                                    VALUES
                                    (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        uuid.uuid4().hex,
                                        cid,
                                        name.strip(),
                                        usn.strip(),
                                        admission_year,
                                        current_semester,
                                        photo_path,
                                        0,
                                        90
                                    )
                                )

                                db.commit()

                                st.success(
                                    "Student saved permanently."
                                )

                    finally:

                        db.close()

        st.divider()

        students_list = get_students(
            cid
        )

        st.subheader(
            "📋 Students"
        )

        if students_list:

            data = []

            for student in students_list:

                data.append({
                    "Name":
                        student["name"],
                    "USN":
                        student["usn"],
                    "Semester":
                        student["current_semester"]
                })

            st.dataframe(
                pd.DataFrame(data),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No students in this class yet."
            )

        # DELETE
        if students_list:

            st.divider()

            st.subheader(
                "🗑️ Delete Individual Student"
            )

            st.warning(
                "Deleting a student permanently removes "
                "the profile, academic records, marksheets "
                "and attendance."
            )

            delete_options = {
                f"{s['name']} — "
                f"{s['usn'] or 'No USN'}":
                s["id"]
                for s in students_list
            }

            delete_label = st.selectbox(
                "Select student to delete",
                list(
                    delete_options.keys()
                ),
                key="delete_student_selector"
            )

            delete_student_id = (
                delete_options[
                    delete_label
                ]
            )

            confirm_delete = st.checkbox(
                "I understand that all data for this student will be permanently deleted.",
                key="confirm_delete_student"
            )

            if st.button(
                "🗑️ Permanently Delete Student",
                type="primary",
                disabled=not confirm_delete
            ):

                delete_student(
                    delete_student_id
                )

                if (
                    st.session_state.selected_student
                    == delete_student_id
                ):

                    st.session_state.selected_student = None

                st.success(
                    "Student and all linked records "
                    "were deleted."
                )

                st.rerun()


# ============================================================
# PAGE 4 - DAILY ATTENDANCE
# ============================================================

elif page == "📅 Daily Attendance":

    st.header(
        "📅 Daily Attendance"
    )

    st.info(
        "Select class, date, subject and hour. "
        "All students in the selected class "
        "will appear automatically."
    )

    classes = get_classes()

    if not classes:

        st.warning(
            "Create a class first."
        )

    else:

        class_options = {
            c["class_name"]: c["id"]
            for c in classes
        }

        selected_class_name = st.selectbox(
            "🏫 Select Class",
            list(
                class_options.keys()
            ),
            key="attendance_class"
        )

        class_id = class_options[
            selected_class_name
        ]

        students_list = get_students(
            class_id
        )

        if not students_list:

            st.warning(
                "No students found."
            )

        else:

            c1, c2, c3 = st.columns(3)

            with c1:

                attendance_date = st.date_input(
                    "📅 Date",
                    value=datetime.now().date()
                )

            with c2:

                subject_name = st.text_input(
                    "📚 Subject Name",
                    placeholder="Example: Data Warehousing"
                ).strip()

            with c3:

                hour_number = st.number_input(
                    "⏰ Hour / Period",
                    min_value=1,
                    max_value=15,
                    value=1
                )

            if subject_name:

                date_string = (
                    attendance_date.isoformat()
                )

                db = get_db()

                existing = db.execute(
                    """
                    SELECT student_id, status
                    FROM attendance
                    WHERE class_id = ?
                    AND attendance_date = ?
                    AND subject_name = ?
                    AND hour_number = ?
                    """,
                    (
                        class_id,
                        date_string,
                        subject_name,
                        hour_number
                    )
                ).fetchall()

                db.close()

                existing_status = {
                    r["student_id"]:
                    r["status"]
                    for r in existing
                }

                st.divider()

                st.subheader(
                    f"👥 {selected_class_name} "
                    f"— {attendance_date.strftime('%d %B %Y')}"
                )

                st.caption(
                    f"📚 {subject_name} "
                    f"| ⏰ Hour {hour_number}"
                )

                attendance_values = {}

                for index, student in enumerate(
                    students_list
                ):

                    c1, c2, c3 = st.columns(
                        [0.5, 4, 2]
                    )

                    with c1:
                        st.write(
                            index + 1
                        )

                    with c2:

                        st.write(
                            f"**{student['name']}**"
                        )

                        st.caption(
                            student["usn"]
                            or "No USN"
                        )

                    with c3:

                        default = (
                            existing_status.get(
                                student["id"],
                                "Present"
                            )
                        )

                        attendance_values[
                            student["id"]
                        ] = st.radio(
                            "Status",
                            [
                                "Present",
                                "Absent"
                            ],
                            index=(
                                0
                                if default == "Present"
                                else 1
                            ),
                            horizontal=True,
                            key=(
                                f"att_"
                                f"{date_string}_"
                                f"{subject_name}_"
                                f"{hour_number}_"
                                f"{student['id']}"
                            ),
                            label_visibility="collapsed"
                        )

                present = sum(
                    value == "Present"
                    for value
                    in attendance_values.values()
                )

                absent = (
                    len(attendance_values)
                    - present
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Students",
                    len(students_list)
                )

                b.metric(
                    "Present",
                    present
                )

                c.metric(
                    "Absent",
                    absent
                )

                if st.button(
                    "💾 Save Daily Attendance",
                    type="primary",
                    use_container_width=True
                ):

                    db = get_db()

                    try:

                        for student_id, status in (
                            attendance_values.items()
                        ):

                            db.execute(
                                """
                                INSERT INTO attendance
                                (
                                    id,
                                    class_id,
                                    student_id,
                                    attendance_date,
                                    subject_name,
                                    hour_number,
                                    status,
                                    created_at
                                )
                                VALUES
                                (?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT
                                (
                                    student_id,
                                    attendance_date,
                                    subject_name,
                                    hour_number
                                )
                                DO UPDATE SET
                                    status =
                                    excluded.status,
                                    class_id =
                                    excluded.class_id,
                                    created_at =
                                    excluded.created_at
                                """,
                                (
                                    uuid.uuid4().hex,
                                    class_id,
                                    student_id,
                                    date_string,
                                    subject_name,
                                    hour_number,
                                    status,
                                    datetime.now().isoformat()
                                )
                            )

                        db.commit()

                        st.success(
                            "Attendance saved for all students."
                        )

                    except Exception as e:

                        db.rollback()

                        st.error(
                            f"Could not save attendance: {e}"
                        )

                    finally:

                        db.close()


# ============================================================
# PAGE 5 - ATTENDANCE DASHBOARD
# ============================================================

elif page == "📊 Attendance Dashboard":

    st.header(
        "📊 Attendance Dashboard"
    )

    st.info(
        "View monthly attendance, subject-wise attendance, "
        "student attendance and low-attendance alerts."
    )

    classes = get_classes()

    if not classes:

        st.warning(
            "Create a class first."
        )

    else:

        class_options = {
            c["class_name"]: c["id"]
            for c in classes
        }

        c1, c2, c3 = st.columns(3)

        with c1:

            selected_class_name = st.selectbox(
                "🏫 Class",
                list(
                    class_options.keys()
                ),
                key="att_dash_class"
            )

        with c2:

            selected_month = st.selectbox(
                "📅 Month",
                list(
                    range(1, 13)
                ),
                index=datetime.now().month - 1,
                format_func=lambda x:
                    datetime(
                        2026,
                        x,
                        1
                    ).strftime("%B")
            )

        with c3:

            selected_year = st.number_input(
                "📆 Year",
                min_value=2020,
                max_value=2100,
                value=datetime.now().year,
                step=1
            )

        class_id = class_options[
            selected_class_name
        ]

        df = class_attendance_report(
            class_id,
            selected_month,
            selected_year
        )

        st.divider()

        # ----------------------------------------------------
        # CLASS MONTHLY SUMMARY
        # ----------------------------------------------------

        st.subheader(
            f"📅 {selected_class_name} — "
            f"{datetime(selected_year, selected_month, 1).strftime('%B %Y')}"
        )

        if df.empty:

            st.info(
                "No attendance records found "
                "for this month."
            )

        else:

            total_hours = int(
                df["Total Hours"].sum()
            )

            total_present = int(
                df["Present"].sum()
            )

            total_absent = int(
                df["Absent"].sum()
            )

            overall_percentage = (
                total_present
                / total_hours
                * 100
                if total_hours
                else 0
            )

            a, b, c, d = st.columns(4)

            a.metric(
                "Total Hours",
                total_hours
            )

            b.metric(
                "Present",
                total_present
            )

            c.metric(
                "Absent",
                total_absent
            )

            d.metric(
                "Attendance %",
                f"{overall_percentage:.2f}%"
            )

            # ------------------------------------------------
            # STUDENT SUMMARY
            # ------------------------------------------------

            st.subheader(
                "👨‍🎓 Student-wise Attendance"
            )

            student_summary = (
                df.groupby(
                    [
                        "Student",
                        "USN"
                    ],
                    as_index=False
                )
                .agg({
                    "Total Hours": "sum",
                    "Present": "sum",
                    "Absent": "sum"
                })
            )

            student_summary[
                "Attendance %"
            ] = (
                student_summary["Present"]
                / student_summary["Total Hours"]
                * 100
            ).round(2)

            st.dataframe(
                student_summary,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # LOW ATTENDANCE
            # ------------------------------------------------

            st.subheader(
                "🚨 Low Attendance Students"
            )

            threshold = st.slider(
                "Alert below",
                50,
                95,
                75,
                key="monthly_threshold"
            )

            low_df = student_summary[
                student_summary["Attendance %"]
                < threshold
            ].copy()

            if low_df.empty:

                st.success(
                    "No students are below "
                    f"{threshold}%."
                )

            else:

                st.warning(
                    f"{len(low_df)} student(s) "
                    f"below {threshold}%."
                )

                st.dataframe(
                    low_df.sort_values(
                        "Attendance %"
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # SUBJECT SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📚 Subject-wise Attendance"
            )

            subject_summary = (
                df.groupby(
                    "Subject",
                    as_index=False
                )
                .agg({
                    "Total Hours": "sum",
                    "Present": "sum",
                    "Absent": "sum"
                })
            )

            subject_summary[
                "Attendance %"
            ] = (
                subject_summary["Present"]
                / subject_summary["Total Hours"]
                * 100
            ).round(2)

            st.dataframe(
                subject_summary,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # CHART
            # ------------------------------------------------

            st.subheader(
                "📈 Subject Attendance Chart"
            )

            chart_df = subject_summary[
                [
                    "Subject",
                    "Attendance %"
                ]
            ].set_index(
                "Subject"
            )

            st.bar_chart(
                chart_df
            )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.subheader(
                "📥 Export Attendance"
            )

            csv_data = df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                "📥 Download Detailed Attendance CSV",
                data=csv_data,
                file_name=(
                    f"{selected_class_name}_"
                    f"{selected_month}_"
                    f"{selected_year}_attendance.csv"
                ),
                mime="text/csv"
            )


# ============================================================
# PAGE 6 - PREVIOUS SEMESTERS
# ============================================================

elif page == "📚 Previous Semesters":

    st.header(
        "📚 Previous Semester Records"
    )

    st.info(
        "Upload the VTU marksheet. Subjects, codes, "
        "internal, external and total marks are detected "
        "automatically."
    )

    student_id = select_student_for_page()

    if student_id:

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
            "📄 Upload VTU Marksheet",
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
                    "🔎 Reading marksheet..."
                ):

                    text, detected_subjects = (
                        process_marksheet(
                            path
                        )
                    )

                st.session_state.ocr_subjects = (
                    detected_subjects
                )

                st.session_state.ocr_text = (
                    text
                )

                st.session_state.ocr_filename = (
                    file_signature
                )

                st.session_state.ocr_path = (
                    path
                )

            subjects = (
                st.session_state.ocr_subjects
            )

            if subjects:

                st.success(
                    f"Automatically detected "
                    f"{len(subjects)} subject(s)."
                )

                st.subheader(
                    "📋 Detected Subjects"
                )

                st.caption(
                    "Check the detected values. "
                    "Normally you only need to enter Credits."
                )

                edited_subjects = []

                # TABLE-LIKE DISPLAY
                st.markdown(
                    """
                    **Subject Code | Subject Name | Internal | External | Total | Credits**
                    """
                )

                for index, subject in enumerate(
                    subjects
                ):

                    c1, c2, c3, c4, c5, c6 = st.columns(
                        [1.2, 3, 1.2, 1.2, 1.2, 1]
                    )

                    with c1:

                        code = st.text_input(
                            "Code",
                            value=subject.get(
                                "subject_code",
                                ""
                            ),
                            key=f"code_{student_id}_{semester_number}_{index}"
                        )

                    with c2:

                        subject_name = st.text_input(
                            "Subject",
                            value=subject.get(
                                "subject_name",
                                ""
                            ),
                            key=f"name_{student_id}_{semester_number}_{index}"
                        )

                    with c3:

                        internal = st.number_input(
                            "Internal",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(
                                subject["internal"]
                                if subject["internal"]
                                is not None
                                else 0
                            ),
                            key=f"internal_{student_id}_{semester_number}_{index}"
                        )

                    with c4:

                        external = st.number_input(
                            "External",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(
                                subject["external"]
                                if subject["external"]
                                is not None
                                else 0
                            ),
                            key=f"external_{student_id}_{semester_number}_{index}"
                        )

                    # Total is ALWAYS calculated. It is not an independent input.
                    total = round(
                        float(internal) + float(external),
                        2
                    )

                    with c5:
                        st.metric(
                            "Total",
                            f"{total:g}"
                        )

                    with c6:

                        credits = st.number_input(
                            "Credits",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(
                                subject.get(
                                    "credits",
                                    0
                                )
                            ),
                            step=0.5,
                            key=f"credits_{student_id}_{semester_number}_{index}"
                        )

                    edited_subjects.append({
                        "subject_code":
                            code,
                        "subject_name":
                            subject_name,
                        "internal":
                            internal,
                        "external":
                            external,
                        "total":
                            round(
                                float(internal) + float(external),
                                2
                            ),
                        "credits":
                            credits,
                        "result":
                            subject.get(
                                "result",
                                ""
                            )
                    })

                st.divider()

                calculated_sgpa = (
                    calculate_sgpa(
                        edited_subjects
                    )
                )

                total_credits = sum(
                    safe_float(
                        x["credits"]
                    ) or 0
                    for x in edited_subjects
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Subjects",
                    len(edited_subjects)
                )

                c2.metric(
                    "Total Credits",
                    round(
                        total_credits,
                        2
                    )
                )

                c3.metric(
                    "Calculated SGPA",
                    calculated_sgpa
                )

                st.divider()

                if st.button(
                    "💾 Save Previous Semester",
                    type="primary"
                ):

                    db = get_db()

                    try:

                        old_sem = db.execute(
                            """
                            SELECT id
                            FROM semesters
                            WHERE student_id = ?
                            AND semester_number = ?
                            """,
                            (
                                student_id,
                                semester_number
                            )
                        ).fetchone()

                        if old_sem:

                            db.execute(
                                """
                                DELETE FROM subjects
                                WHERE semester_id = ?
                                """,
                                (
                                    old_sem["id"],
                                )
                            )

                            db.execute(
                                """
                                DELETE FROM semesters
                                WHERE id = ?
                                """,
                                (
                                    old_sem["id"],
                                )
                            )

                        semester_id = (
                            uuid.uuid4().hex
                        )

                        db.execute(
                            """
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
                            VALUES
                            (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                semester_id,
                                student_id,
                                semester_number,
                                semester_name,
                                calculated_sgpa,
                                st.session_state.get(
                                    "ocr_path",
                                    ""
                                ),
                                0,
                                datetime.now().isoformat()
                            )
                        )

                        for subject in edited_subjects:

                            subject["total"] = round(
                                float(subject["internal"]) + float(subject["external"]),
                                2
                            )

                            db.execute(
                                """
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
                                VALUES
                                (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    uuid.uuid4().hex,
                                    semester_id,
                                    subject["subject_code"],
                                    subject["subject_name"],
                                    subject["internal"],
                                    subject["external"],
                                    subject["total"],
                                    subject["credits"],
                                    subject["result"]
                                )
                            )

                        db.commit()

                        st.success(
                            "Semester saved permanently."
                        )

                        st.session_state.ocr_subjects = []
                        st.session_state.ocr_filename = ""

                    finally:

                        db.close()

            else:

                st.error(
                    "No subjects detected."
                )

                st.info(
                    "Use manual entry if required."
                )

                count = st.number_input(
                    "Number of Subjects",
                    1,
                    30,
                    1
                )

                manual_subjects = []

                for i in range(count):

                    st.markdown(
                        f"### Subject {i + 1}"
                    )

                    code = st.text_input(
                        "Subject Code",
                        key=f"manual_code_{i}"
                    )

                    name = st.text_input(
                        "Subject Name",
                        key=f"manual_name_{i}"
                    )

                    internal = st.number_input(
                        "Internal",
                        0.0,
                        100.0,
                        0.0,
                        key=f"manual_int_{i}"
                    )

                    external = st.number_input(
                        "External",
                        0.0,
                        100.0,
                        0.0,
                        key=f"manual_ext_{i}"
                    )

                    total = (
                        internal
                        + external
                    )

                    credits = st.number_input(
                        "Credits",
                        0.0,
                        10.0,
                        3.0,
                        step=0.5,
                        key=f"manual_cred_{i}"
                    )

                    manual_subjects.append({
                        "subject_code":
                            code,
                        "subject_name":
                            name,
                        "internal":
                            internal,
                        "external":
                            external,
                        "total":
                            total,
                        "credits":
                            credits,
                        "result":
                            ""
                    })

                sgpa = calculate_sgpa(
                    manual_subjects
                )

                st.metric(
                    "Calculated SGPA",
                    sgpa
                )

        # BACKLOG
        st.divider()

        st.subheader(
            "🔄 Backlog / Re-attempt"
        )

        backlog_subject = st.text_input(
            "Backlog Subject Name"
        )

        backlog_semester = st.number_input(
            "Backlog Semester",
            1,
            8,
            1
        )

        backlog_file = st.file_uploader(
            "Backlog Marksheet",
            type=[
                "pdf",
                "jpg",
                "jpeg",
                "png"
            ]
        )

        if backlog_file:

            if st.button(
                "💾 Save Backlog"
            ):

                path = save_uploaded_file(
                    backlog_file,
                    MARKSHEET_DIR
                )

                db = get_db()

                db.execute(
                    """
                    INSERT INTO backlog_marksheets
                    (
                        id,
                        student_id,
                        semester_number,
                        subject_name,
                        file_path,
                        uploaded_at
                    )
                    VALUES
                    (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        student_id,
                        backlog_semester,
                        backlog_subject,
                        path,
                        datetime.now().isoformat()
                    )
                )

                db.commit()
                db.close()

                st.success(
                    "Backlog saved."
                )

        # SAVED SEMESTERS
        st.divider()

        st.subheader(
            "📚 Saved Semesters"
        )

        semesters = get_semesters(
            student_id
        )

        for sem in semesters:

            with st.expander(
                f"Semester "
                f"{sem['semester_number']} "
                f"— SGPA {sem['sgpa']}"
            ):

                subjects = get_subjects(
                    sem["id"]
                )

                table = []

                for subject in subjects:

                    table.append({
                        "Subject Code":
                            subject["subject_code"],
                        "Subject Name":
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

                if table:

                    st.dataframe(
                        pd.DataFrame(table),
                        use_container_width=True,
                        hide_index=True
                    )


# ============================================================
# PAGE 7 - CURRENT SEMESTER
# ============================================================

elif page == "📘 Current Semester":

    st.header(
        "📘 Current Semester"
    )

    st.info(
        "Enter current semester marks manually."
    )

    student_id = select_student_for_page()

    if student_id:

        student = get_student(
            student_id
        )

        semester = student[
            "current_semester"
        ]

        absent_days = st.number_input(
            "Absent Days",
            0,
            300,
            int(
                student["absent_days"]
                or 0
            )
        )

        working_days = st.number_input(
            "Working Days",
            1,
            365,
            int(
                student["working_days"]
                or 90
            )
        )

        subject_count = st.number_input(
            "Number of Subjects",
            1,
            30,
            1
        )

        current_subjects = []

        for i in range(
            subject_count
        ):

            st.markdown(
                f"### 📘 Subject {i + 1}"
            )

            c1, c2 = st.columns(2)

            with c1:

                code = st.text_input(
                    "Subject Code",
                    key=f"cur_code_{i}"
                )

                name = st.text_input(
                    "Subject Name",
                    key=f"cur_name_{i}"
                )

                credits = st.number_input(
                    "Credits",
                    0.0,
                    10.0,
                    3.0,
                    step=0.5,
                    key=f"cur_credit_{i}"
                )

            with c2:

                ia1 = st.number_input(
                    "IA-1",
                    0.0,
                    100.0,
                    0.0,
                    key=f"cur_ia1_{i}"
                )

                ia2 = st.number_input(
                    "IA-2",
                    0.0,
                    100.0,
                    0.0,
                    key=f"cur_ia2_{i}"
                )

                final = st.number_input(
                    "Final / VTU",
                    0.0,
                    100.0,
                    0.0,
                    key=f"cur_final_{i}"
                )

            internal = (
                ia1 + ia2
            ) / 2

            total = (
                internal
                + final
            )

            st.info(
                f"Internal: {internal:.2f} "
                f"| Estimated Total: {total:.2f}"
            )

            current_subjects.append({
                "subject_code":
                    code,
                "subject_name":
                    name,
                "internal":
                    internal,
                "external":
                    final,
                "total":
                    total,
                "credits":
                    credits,
                "result":
                    ""
            })

        current_sgpa = calculate_sgpa(
            current_subjects
        )

        total, present, absent, attendance = (
            attendance_summary(
                student_id
            )
        )

        if total:

            attendance_text = (
                f"{attendance:.2f}%"
            )

        else:

            attendance_text = (
                f"{max(0, working_days - absent_days) / working_days * 100:.2f}%"
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Subjects",
            subject_count
        )

        c2.metric(
            "Attendance",
            attendance_text
        )

        c3.metric(
            "Estimated SGPA",
            current_sgpa
        )

        if st.button(
            "💾 Save Current Semester",
            type="primary"
        ):

            db = get_db()

            try:

                existing = db.execute(
                    """
                    SELECT id
                    FROM semesters
                    WHERE student_id = ?
                    AND semester_number = ?
                    """,
                    (
                        student_id,
                        semester
                    )
                ).fetchone()

                if existing:

                    semester_id = (
                        existing["id"]
                    )

                    db.execute(
                        """
                        DELETE FROM subjects
                        WHERE semester_id = ?
                        """,
                        (
                            semester_id,
                        )
                    )

                    db.execute(
                        """
                        UPDATE semesters
                        SET semester_name = ?,
                            sgpa = ?,
                            is_current = 1
                        WHERE id = ?
                        """,
                        (
                            f"Semester {semester}",
                            current_sgpa,
                            semester_id
                        )
                    )

                else:

                    semester_id = (
                        uuid.uuid4().hex
                    )

                    db.execute(
                        """
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
                        VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            semester_id,
                            student_id,
                            semester,
                            f"Semester {semester}",
                            current_sgpa,
                            "",
                            1,
                            datetime.now().isoformat()
                        )
                    )

                for subject in current_subjects:

                    db.execute(
                        """
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
                        VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            semester_id,
                            subject["subject_code"],
                            subject["subject_name"],
                            subject["internal"],
                            subject["external"],
                            subject["total"],
                            subject["credits"],
                            ""
                        )
                    )

                db.execute(
                    """
                    UPDATE students
                    SET absent_days = ?,
                        working_days = ?
                    WHERE id = ?
                    """,
                    (
                        absent_days,
                        working_days,
                        student_id
                    )
                )

                db.commit()

                st.success(
                    "Current semester saved."
                )

            finally:

                db.close()


# ============================================================
# PAGE 8 - STUDENT ANALYTICS
# ============================================================

elif page == "📈 Student Analytics":

    st.header(
        "📈 Student Analytics"
    )

    student_id = select_student_for_page()

    if student_id:

        student = get_student(
            student_id
        )

        semesters = get_semesters(
            student_id
        )

        cgpa = calculate_cgpa(
            student_id
        )

        c1, c2 = st.columns(
            [1, 3]
        )

        with c1:

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

        with c2:

            st.subheader(
                student["name"]
            )

            st.write(
                f"**USN:** {student['usn']}"
            )

            st.write(
                f"**Semester:** "
                f"{student['current_semester']}"
            )

            st.metric(
                "CGPA",
                cgpa
            )

        st.divider()

        # ATTENDANCE
        st.subheader(
            "🕒 Attendance"
        )

        total, present, absent, percentage = (
            attendance_summary(
                student_id
            )
        )

        if total:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total Hours",
                total
            )

            c2.metric(
                "Present",
                present
            )

            c3.metric(
                "Absent",
                absent
            )

            c4.metric(
                "Attendance",
                f"{percentage:.2f}%"
            )

            if percentage < 75:

                st.error(
                    "🚨 Attendance is below 75%."
                )

            elif percentage < 85:

                st.warning(
                    "⚠️ Attendance requires attention."
                )

            else:

                st.success(
                    "✅ Attendance is currently above 85%."
                )

            st.subheader(
                "📚 Subject-wise Attendance"
            )

            subject_rows = subject_attendance(
                student_id
            )

            if subject_rows:

                st.dataframe(
                    pd.DataFrame(
                        subject_rows
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "📅 Monthly Attendance"
            )

            monthly_rows = monthly_attendance(
                student_id
            )

            if monthly_rows:

                monthly_df = pd.DataFrame(
                    monthly_rows
                )

                st.dataframe(
                    monthly_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.line_chart(
                    monthly_df.set_index(
                        "Month"
                    )[[
                        "Attendance %"
                    ]]
                )

        else:

            st.info(
                "No hour-wise attendance has "
                "been recorded yet."
            )

        # ACADEMIC PERFORMANCE
        st.divider()

        st.subheader(
            "📚 Academic Performance"
        )

        if semesters:

            performance = []

            for sem in semesters:

                performance.append({
                    "Semester":
                        sem["semester_number"],
                    "SGPA":
                        sem["sgpa"]
                })

            performance_df = pd.DataFrame(
                performance
            )

            st.line_chart(
                performance_df.set_index(
                    "Semester"
                )
            )

            st.dataframe(
                performance_df,
                use_container_width=True,
                hide_index=True
            )

        # RISK
        st.divider()

        st.subheader(
            "🤖 Academic Risk Analysis"
        )

        if cgpa >= 8.5:

            risk = "Low Risk"

        elif cgpa >= 7:

            risk = "Moderate Risk"

        elif cgpa >= 5.5:

            risk = "Needs Attention"

        else:

            risk = "High Academic Risk"

        st.metric(
            "Academic Risk Level",
            risk
        )

        # SUBJECT PERFORMANCE
        st.subheader(
            "📖 Subject-wise Performance"
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
                pd.DataFrame(
                    all_subjects
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No subject records available."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 EduPredict AI • "
    "VTU Academic Performance Management System • "
    "Academic + Attendance Intelligence"
)