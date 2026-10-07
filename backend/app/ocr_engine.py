"""
Enhanced Multi-Layer VTU OCR Engine for EduPredict AI.

Supports:
- VTU 2022/2023/2024 NEP schemes
- VTU 2018/2021 CBCS schemes
- Native/vector PDFs
- Scanned PDFs
- JPG/JPEG/PNG marksheets
- Multi-line result layouts
- Single-line/tabular layouts
- OCR fallback
- Manual verification workflow

Important rule:
    TOTAL = INTERNAL + EXTERNAL

The OCR engine never treats OCR-detected Total as authoritative.
Total is always recalculated from Internal + External.
"""

import os
import re
from typing import List, Dict, Tuple, Any, Optional

import fitz
import pytesseract

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
)


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH
    )


# ============================================================
# VTU SUBJECT CODE
# ============================================================

VTU_CODE_PATTERN = re.compile(
    r"\b("
    r"[A-Z]{2,8}\d{2,4}[A-Z0-9]?"
    r"|"
    r"\d{2}[A-Z]{2,6}\d{2,4}[A-Z0-9]?"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# COMMON NON-SUBJECT TERMS
# ============================================================

HEADER_TERMS = {
    "USN",
    "NAME",
    "SEMESTER",
    "RESULT",
    "RESULT SHEET",
    "VTU",
    "UNIVERSITY",
    "EXAMINATION",
    "EXAM",
    "MARKS",
    "MARKSHEET",
    "PROVISIONAL",
    "UPDATED",
    "PAGE",
    "DATE",
    "REGISTRAR",
    "NOMENCLATURE",
    "ABBREVIATIONS",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def normalize_line(value: Any) -> str:

    text = clean_text(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:

        text = str(value).strip()

        if not text:
            return None

        text = text.replace(",", "")

        value_float = float(text)

        if 0 <= value_float <= 1000:
            return value_float

    except (
        ValueError,
        TypeError,
    ):
        return None

    return None


# ============================================================
# INTEGER-LIKE NUMBER
# ============================================================

def is_number_line(text: str) -> bool:

    return bool(
        re.fullmatch(
            r"\d{1,3}(?:\.\d+)?",
            text.strip(),
        )
    )


def extract_numbers(text: str) -> List[float]:

    matches = re.findall(
        r"(?<![A-Za-z])"
        r"\d{1,3}(?:\.\d+)?"
        r"(?![A-Za-z])",
        text,
    )

    values = []

    for match in matches:

        value = safe_float(match)

        if value is not None:
            values.append(value)

    return values


# ============================================================
# CREDIT GUESS
# ============================================================

def guess_credits(
    code: str,
    name: str,
    total: float,
) -> float:

    code_upper = (
        code or ""
    ).upper()

    name_upper = (
        name or ""
    ).upper()

    if any(
        term in name_upper
        for term in [
            "LAB",
            "LABORATORY",
            "PRACTICAL",
        ]
    ):

        return 1.0

    if "L" in code_upper[-3:]:

        return 1.0

    if any(
        term in name_upper
        for term in [
            "PHYSICAL",
            "KANNADA",
            "SOCIAL",
            "ENGLISH",
            "VALUES",
            "YOGA",
            "CONSTITUTION",
        ]
    ):

        return 1.0

    if any(
        term in name_upper
        for term in [
            "PROJECT",
            "MINI PROJECT",
            "INTERNSHIP",
        ]
    ):

        return 2.0

    if any(
        term in name_upper
        for term in [
            "MATHEMATICS",
            "MATH",
            "DATA STRUCTURES",
            "ALGORITHMS",
        ]
    ):

        return 4.0

    return 3.0


# ============================================================
# RESULT CALCULATION
# ============================================================

def calculate_total(
    internal: float,
    external: float,
) -> float:

    internal = safe_float(internal) or 0.0
    external = safe_float(external) or 0.0

    return round(
        internal + external,
        2,
    )


def calculate_result(
    internal: float,
    external: float,
) -> str:

    total = calculate_total(
        internal,
        external,
    )

    return (
        "P"
        if total >= 40 and external >= 18
        else "F"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image_for_ocr(
    image: Image.Image,
) -> Image.Image:

    gray = image.convert("L")

    contrast = ImageEnhance.Contrast(
        gray
    ).enhance(2.2)

    sharp = contrast.filter(
        ImageFilter.SHARPEN
    )

    return sharp


# ============================================================
# SUBJECT CODE DETECTION
# ============================================================

def find_subject_code(
    line: str,
) -> Optional[str]:

    if not line:
        return None

    matches = list(
        VTU_CODE_PATTERN.finditer(
            line
        )
    )

    for match in matches:

        code = match.group().upper()

        if len(code) < 5:
            continue

        upper_line = line.upper()

        if any(
            term in upper_line
            for term in HEADER_TERMS
        ):
            continue

        return code

    return None


# ============================================================
# SUBJECT NAME CLEANING
# ============================================================

def clean_subject_name(
    value: str,
    code: str,
) -> str:

    if not value:
        return f"Subject {code}"

    text = clean_text(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip(
        " -|:;,"
    )

    if not text:
        return f"Subject {code}"

    return text


# ============================================================
# MULTI-LINE PARSER
# ============================================================

def parse_vtu_multi_line_text(
    text: str,
) -> List[Dict[str, Any]]:

    lines = [
        normalize_line(line)
        for line in text.split("\n")
        if normalize_line(line)
    ]

    subjects = []

    seen_codes = set()

    index = 0

    while index < len(lines):

        line = lines[index]

        code = find_subject_code(line)

        if not code:

            index += 1
            continue

        if code in seen_codes:

            index += 1
            continue

        seen_codes.add(code)

        # ----------------------------------------------------
        # If code and subject name occur on same line
        # ----------------------------------------------------

        remainder = (
            line[
                line.upper().find(code.upper())
                + len(code):
            ].strip()
        )

        index += 1

        name_parts = []

        if remainder:

            if not is_number_line(
                remainder
            ):

                name_parts.append(
                    remainder
                )

        # ----------------------------------------------------
        # Collect subject name
        # ----------------------------------------------------

        while index < len(lines):

            current = lines[index]

            if find_subject_code(
                current
            ):

                break

            if is_number_line(
                current
            ):

                break

            upper = current.upper()

            if upper in [
                "P",
                "F",
                "PASS",
                "FAIL",
                "A",
                "W",
            ]:

                break

            if any(
                marker in upper
                for marker in [
                    "INTERNAL",
                    "EXTERNAL",
                    "TOTAL",
                    "RESULT",
                ]
            ):

                index += 1
                continue

            if re.match(
                r"^\d{1,2}/\d{1,2}/\d{2,4}$",
                current,
            ):

                index += 1
                continue

            if len(current) <= 100:

                name_parts.append(
                    current
                )

            index += 1

            if len(name_parts) >= 4:
                break

        subject_name = clean_subject_name(
            " ".join(name_parts),
            code,
        )

        # ----------------------------------------------------
        # Collect marks
        # ----------------------------------------------------

        numeric_values = []

        result_from_text = None

        mark_scan_count = 0

        while (
            index < len(lines)
            and mark_scan_count < 10
        ):

            current = lines[index]

            # next subject
            if find_subject_code(
                current
            ):

                break

            upper = current.upper()

            if upper in [
                "P",
                "PASS",
                "F",
                "FAIL",
                "A",
                "W",
            ]:

                if upper in [
                    "P",
                    "PASS",
                ]:
                    result_from_text = "P"

                elif upper in [
                    "F",
                    "FAIL",
                    "A",
                    "W",
                ]:
                    result_from_text = "F"

                index += 1
                mark_scan_count += 1
                continue

            if is_number_line(
                current
            ):

                value = safe_float(
                    current
                )

                if (
                    value is not None
                    and 0 <= value <= 100
                ):

                    numeric_values.append(
                        value
                    )

                index += 1
                mark_scan_count += 1
                continue

            # Sometimes values appear like:
            # Internal 35
            # External 44

            if any(
                marker in upper
                for marker in [
                    "INTERNAL",
                    "EXTERNAL",
                    "TOTAL",
                ]
            ):

                nums = extract_numbers(
                    current
                )

                for number in nums:

                    if (
                        0 <= number <= 100
                    ):
                        numeric_values.append(
                            number
                        )

                index += 1
                mark_scan_count += 1
                continue

            # Stop on obvious next sections

            if any(
                marker in upper
                for marker in [
                    "NOMENCLATURE",
                    "ABBREVIATIONS",
                    "REGISTRAR",
                    "NOTE :",
                    "NOTE:",
                ]
            ):

                break

            index += 1
            mark_scan_count += 1

        # ----------------------------------------------------
        # Determine marks
        # ----------------------------------------------------

        if len(numeric_values) >= 2:

            internal = numeric_values[0]
            external = numeric_values[1]

        elif len(numeric_values) == 1:

            internal = numeric_values[0]
            external = 0.0

        else:

            internal = 0.0
            external = 0.0

        total = calculate_total(
            internal,
            external,
        )

        calculated_result = (
            calculate_result(
                internal,
                external,
            )
        )

        if result_from_text in [
            "P",
            "F",
        ]:

            result = result_from_text

        else:

            result = calculated_result

        credits = guess_credits(
            code,
            subject_name,
            total,
        )

        subjects.append(
            {
                "subject_code": code,
                "subject_name": subject_name,
                "internal": internal,
                "external": external,
                "total": total,
                "credits": credits,
                "result": result,
            }
        )

    return subjects


# ============================================================
# SINGLE-LINE PARSER
# ============================================================

def parse_vtu_single_line_text(
    text: str,
) -> List[Dict[str, Any]]:

    lines = [
        normalize_line(line)
        for line in text.split("\n")
        if normalize_line(line)
    ]

    subjects = []

    seen_codes = set()

    for line in lines:

        code = find_subject_code(
            line
        )

        if not code:
            continue

        if code in seen_codes:
            continue

        seen_codes.add(code)

        code_match = re.search(
            re.escape(code),
            line,
            re.IGNORECASE,
        )

        if not code_match:
            continue

        remainder = line[
            code_match.end():
        ].strip()

        # Remove result words from name
        remainder = re.sub(
            r"\b(PASS|FAIL|P|F)\b",
            " ",
            remainder,
            flags=re.IGNORECASE,
        )

        nums = extract_numbers(
            remainder
        )

        nums = [
            value
            for value in nums
            if 0 <= value <= 100
        ]

        if len(nums) >= 2:

            internal = nums[0]
            external = nums[1]

        elif len(nums) == 1:

            internal = nums[0]
            external = 0.0

        else:

            internal = 0.0
            external = 0.0

        total = calculate_total(
            internal,
            external,
        )

        # Name = everything before first number
        name_match = re.search(
            r"\d",
            remainder,
        )

        if name_match:

            subject_name = remainder[
                :name_match.start()
            ].strip()

        else:

            subject_name = remainder.strip()

        subject_name = clean_subject_name(
            subject_name,
            code,
        )

        explicit_result = None

        result_match = re.search(
            r"\b(PASS|FAIL|P|F)\b",
            line,
            re.IGNORECASE,
        )

        if result_match:

            explicit = (
                result_match.group(1)
                .upper()
            )

            explicit_result = (
                "P"
                if explicit in [
                    "P",
                    "PASS",
                ]
                else "F"
            )

        result = (
            explicit_result
            if explicit_result
            else calculate_result(
                internal,
                external,
            )
        )

        credits = guess_credits(
            code,
            subject_name,
            total,
        )

        subjects.append(
            {
                "subject_code": code,
                "subject_name": subject_name,
                "internal": internal,
                "external": external,
                "total": total,
                "credits": credits,
                "result": result,
            }
        )

    return subjects


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_subjects(
    subjects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    result = []

    seen = set()

    for subject in subjects:

        code = (
            subject.get(
                "subject_code",
                "",
            )
            .strip()
            .upper()
        )

        if not code:
            continue

        if code in seen:
            continue

        seen.add(code)

        internal = (
            safe_float(
                subject.get(
                    "internal",
                    0,
                )
            )
            or 0.0
        )

        external = (
            safe_float(
                subject.get(
                    "external",
                    0,
                )
            )
            or 0.0
        )

        total = calculate_total(
            internal,
            external,
        )

        subject["subject_code"] = code
        subject["internal"] = internal
        subject["external"] = external
        subject["total"] = total

        subject["result"] = (
            calculate_result(
                internal,
                external,
            )
        )

        result.append(subject)

    return result


# ============================================================
# PDF NATIVE TEXT
# ============================================================

def extract_pdf_text(
    file_path: str,
) -> str:

    raw_text = ""

    doc = fitz.open(
        file_path
    )

    try:

        for page in doc:

            page_text = page.get_text(
                "text"
            )

            if page_text:

                raw_text += (
                    page_text
                    + "\n"
                )

    finally:

        doc.close()

    return raw_text


# ============================================================
# PDF OCR
# ============================================================

def extract_pdf_ocr_text(
    file_path: str,
) -> str:

    raw_text = ""

    doc = fitz.open(
        file_path
    )

    try:

        for page_index in range(
            len(doc)
        ):

            page = doc[
                page_index
            ]

            pix = page.get_pixmap(
                dpi=250,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                [
                    pix.width,
                    pix.height,
                ],
                pix.samples,
            )

            processed = (
                preprocess_image_for_ocr(
                    image
                )
            )

            page_text = (
                pytesseract.image_to_string(
                    processed,
                    config="--psm 6",
                )
            )

            raw_text += (
                page_text
                + "\n"
            )

    finally:

        doc.close()

    return raw_text


# ============================================================
# IMAGE OCR
# ============================================================

def extract_image_ocr_text(
    file_path: str,
) -> str:

    image = Image.open(
        file_path
    )

    processed = (
        preprocess_image_for_ocr(
            image
        )
    )

    return pytesseract.image_to_string(
        processed,
        config="--psm 6",
    )


# ============================================================
# MAIN PROCESSOR
# ============================================================

def process_marksheet_file(
    file_path: str,
) -> Tuple[
    str,
    List[Dict[str, Any]],
    bool,
]:

    extension = os.path.splitext(
        file_path
    )[1].lower()

    raw_text = ""

    subjects = []

    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        try:

            raw_text = extract_pdf_text(
                file_path
            )

            if len(raw_text.strip()) >= 50:

                subjects = (
                    parse_vtu_multi_line_text(
                        raw_text
                    )
                )

                if not subjects:

                    subjects = (
                        parse_vtu_single_line_text(
                            raw_text
                        )
                    )

            # OCR fallback
            if not subjects:

                raw_text = (
                    extract_pdf_ocr_text(
                        file_path
                    )
                )

                subjects = (
                    parse_vtu_multi_line_text(
                        raw_text
                    )
                )

                if not subjects:

                    subjects = (
                        parse_vtu_single_line_text(
                            raw_text
                        )
                    )

        except Exception as exc:

            raw_text += (
                f"\nPDF Processing Error: {exc}"
            )

    # ========================================================
    # IMAGE
    # ========================================================

    elif extension in [
        ".jpg",
        ".jpeg",
        ".png",
    ]:

        try:

            raw_text = (
                extract_image_ocr_text(
                    file_path
                )
            )

            subjects = (
                parse_vtu_multi_line_text(
                    raw_text
                )
            )

            if not subjects:

                subjects = (
                    parse_vtu_single_line_text(
                        raw_text
                    )
                )

        except Exception as exc:

            raw_text += (
                f"\nImage Processing Error: {exc}"
            )

    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    subjects = deduplicate_subjects(
        subjects
    )

    for subject in subjects:

        internal = (
            safe_float(
                subject.get(
                    "internal",
                    0,
                )
            )
            or 0.0
        )

        external = (
            safe_float(
                subject.get(
                    "external",
                    0,
                )
            )
            or 0.0
        )

        subject["internal"] = internal
        subject["external"] = external

        # STRICT FORMULA
        subject["total"] = calculate_total(
            internal,
            external,
        )

        subject["result"] = (
            calculate_result(
                internal,
                external,
            )
        )

        if not subject.get(
            "credits"
        ):

            subject["credits"] = (
                guess_credits(
                    subject.get(
                        "subject_code",
                        "",
                    ),
                    subject.get(
                        "subject_name",
                        "",
                    ),
                    subject["total"],
                )
            )

    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    low_confidence = (
        len(subjects) == 0
        or any(
            (
                s["internal"] == 0
                and s["external"] == 0
            )
            for s in subjects
        )
    )

    return (
        raw_text,
        subjects,
        low_confidence,
    )