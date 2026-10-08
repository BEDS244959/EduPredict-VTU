"""
EduPredict AI - High-Accuracy VTU Marksheet OCR & Parsing Engine
Supports:
1. Standard VTU Digital PDFs (Table extraction & Stream parsing)
2. Scanned / Image PDFs (Adaptive thresholded high-speed OCR)
3. Image Marksheets (.PNG, .JPG, .JPEG)
4. Multi-scheme VTU codes (2022 Scheme: BCS/BMATS/BPHYS/BSCK/BPEK/etc., 2021/2018 Scheme: 21CS/18CS/etc.)
5. Multi-line subject names & accurate mark-triplet parsing
"""

from __future__ import annotations
import os
import re
from typing import Any, Tuple, List, Dict

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from PIL import Image, ImageOps, ImageFilter
import pytesseract

# Windows Tesseract Executable Path
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(DEFAULT_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

# Comprehensive VTU Subject Code Pattern
# Matches:
# 2022 Scheme: BMATS101, BCHES102, BCEDK103, BCS301, BCS302, BCS303, BCSL305, BSCK307, BPEK359,
#              BCS401, BDS402, BCS403, BCSL404, BCS405A, BDSL456B, BBOC407, BUHK408, BPEK459, etc.
# 2021/2018 Scheme: 21CS41, 21CS42, 21CSL46, 21MAT31, 18CS51, 18CSL57, 21CIP37, 21KSK37
# Diploma & Lateral: 21MATDIP31, 18MATDIP41, 21CIP37, 21EGH18
# General: CS41, MAT31, etc.
SUBJECT_CODE_RE = re.compile(
    r"\b("
    r"B[A-Z0-9]{2,7}\d{1,4}[A-Z0-9]?"
    r"|"
    r"\d{2}[A-Z]{2,6}\d{1,4}[A-Z0-9]?"
    r"|"
    r"[A-Z]{2,6}\d{2,4}[A-Z0-9]?"
    r")\b",
    re.IGNORECASE,
)

# Reserved words / USN pattern that should NOT be treated as subject codes
NOISE_WORDS = {
    "VTU", "USN", "RESULT", "SEMESTER", "SEM", "MARKS", "TOTAL", "INTERNAL", "EXTERNAL",
    "CREDITS", "GRADE", "GRADE_POINT", "PASS", "FAIL", "BELAGAVI", "REGISTRAR", "EVALUATION",
    "PROVISIONAL", "EXAMINATION", "NOMENCLATURE", "ABBREVIATIONS", "NOTE", "STUDENT", "NAME",
    "UNIVERSITY", "SEAT", "NUMBER", "COLLEGE", "BRANCH", "CAMPUS", "BACHELOR", "MASTER",
    "ENGINEERING", "TECHNOLOGY", "SCIENCE", "DATE", "TIME", "PAGE", "ANNOUNCED", "UPDATED",
    "STATUS", "WITHHELD", "ABSENT", "ELIGIBLE", "REVALUATION", "ANNUAL", "SUPPLEMENTARY",
    "CIE", "SEE", "MAX", "MIN", "OBT", "PERCENTAGE", "REMARKS"
}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None or value == "":
            return default
        cleaned = str(value).replace(",", "").strip()
        return float(cleaned)
    except (TypeError, ValueError):
        return default


def calculate_total(internal: float, external: float) -> float:
    """Calculate combined total marks (VTU standard: Internal + External)."""
    return round(safe_float(internal) + safe_float(external), 2)


def calculate_result(internal: float, external: float) -> str:
    """Calculate P (Pass) or F (Fail) according to VTU evaluation criteria."""
    tot = calculate_total(internal, external)
    ext = safe_float(external)
    # VTU criteria: Total >= 40 and External >= 18 (or ext==0 for internal-only subjects like PE)
    if tot >= 40.0 and (ext >= 18.0 or ext == 0.0):
        return "P"
    return "F"


def guess_credits(subject_code: str, subject_name: str = "") -> float:
    """Infer VTU course credits based on subject code scheme and name."""
    code = str(subject_code).upper().strip()
    name = str(subject_name).upper().strip()

    # 0-credit audit / mandatory non-credit courses (Physical Education / PEK, Yoga, NSS, Audit)
    if any(k in code for k in ["PEK", "BPEK"]) or any(k in name for k in ["PHYSICAL EDUCATION", "YOGA", "NSS", "AUDIT"]):
        return 0.0

    # Laboratory courses (usually 1.0 or 1.5 credits)
    if "L" in code[3:] or "LAB" in name or "PRACTICAL" in name:
        return 1.5

    # 1-credit or 2-credit mandatory non-technical / skill / ability enhancement courses
    if any(k in code for k in ["SFH", "CIP", "ICO", "WSK", "KSK", "IDT", "UHK", "SCK", "ENG", "CPH"]):
        return 1.0

    # Foundational Sciences / Core Mathematics in Sem 1 & 2 (typically 4.0 credits)
    if any(code.startswith(k) for k in ["BMATS1", "BMATS2", "BPHYS1", "BPHYS2", "BCHES1", "BCHES2", "21MAT1", "21MAT2"]):
        return 4.0

    # Core Engineering / Theory / Electives default to 3.0 credits
    return 3.0


def is_usn_or_noise(code: str) -> bool:
    """Check if token is a Student USN (e.g. 1TJ24DS028) or reserved noise keyword."""
    if not code:
        return True
    c = code.upper().strip()
    # Student USN format: 1[A-Z]{2}\d{2}[A-Z]{2}\d{3}
    if re.fullmatch(r"1[A-Z]{2}\d{2}[A-Z]{2}\d{3}", c):
        return True
    # 4-digit years (e.g. 2024, 2025, 2026)
    if re.fullmatch(r"\d{4}", c):
        return True
    if c in NOISE_WORDS:
        return True
    if len(c) < 3 or len(c) > 12:
        return True
    return False


def clean_subject_name(name: str) -> str:
    """Clean subject title text from OCR artifacts, dates, and noise words."""
    if not name:
        return ""
    n = name.replace("\n", " ").replace("\r", " ")
    # Remove noise words and footnote patterns
    noise_patterns = [
        r"\b(moabedea|Sofedironed|Nomenclature|Abbreviations|Note|REGISTRAR|EVALUATION|Announced|Updated|Result)\b.*$",
        r"P\s*->\s*PASS.*$",
        r"F\s*->\s*FAIL.*$",
        r"\d{4}[-/]\d{2}[-/]\d{2}",
        r"[|:;_~\\/]+"
    ]
    for np in noise_patterns:
        n = re.sub(np, " ", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+", " ", n).strip(" -.,")
    return n


def dedupe_subjects(subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate subject codes, preserving the highest-quality detected record."""
    seen = {}
    for sub in subjects:
        code = str(sub.get("subject_code", "")).upper().strip()
        if not code or is_usn_or_noise(code):
            continue
        if code not in seen:
            seen[code] = dict(sub)
        else:
            old = seen[code]
            # Prefer longer, more descriptive name
            if len(str(sub.get("subject_name", ""))) > len(str(old.get("subject_name", ""))):
                old["subject_name"] = sub.get("subject_name", "")
            # Prefer non-zero marks
            if safe_float(old.get("total")) == 0 and safe_float(sub.get("total")) > 0:
                old["total"] = sub.get("total")
                old["internal"] = sub.get("internal")
                old["external"] = sub.get("external")
                old["result"] = sub.get("result")
    return list(seen.values())


# ============================================================
# STRATEGY 1: PDF Table Extraction via PyMuPDF
# ============================================================

def _extract_from_pdf_tables(path: str) -> List[Dict[str, Any]]:
    """Extract structured subject table rows from digital VTU PDFs."""
    subjects = []
    doc = fitz.open(path)
    try:
        for page in doc:
            finder = page.find_tables()
            for t in finder.tables:
                rows = t.extract()
                for r in rows:
                    if not r or len(r) < 2:
                        continue
                    row_str = [str(x).strip() if x is not None else "" for x in r]
                    
                    # Scan across the first 4 columns to find where subject code is located
                    code_cand = None
                    code_idx = -1
                    for ci in range(min(4, len(row_str))):
                        val = re.sub(r"[^A-Za-z0-9]", "", row_str[ci]).upper()
                        if not val or is_usn_or_noise(val):
                            continue
                        if re.match(r"^(B[A-Z0-9]{2,7}\d{1,4}[A-Z0-9]?|\d{2}[A-Z]{2,6}\d{1,4}[A-Z0-9]?|[A-Z]{2,6}\d{1,4}[A-Z0-9]?)$", val):
                            code_cand = val
                            code_idx = ci
                            break

                    if not code_cand:
                        continue

                    # Subject Name is typically the next column
                    name = ""
                    if code_idx + 1 < len(row_str):
                        name = clean_subject_name(row_str[code_idx + 1])
                    if not name:
                        name = f"Subject {code_cand}"

                    # Extract numerical marks from subsequent columns
                    nums = []
                    res_candidate = None
                    for col in row_str[code_idx + 2:]:
                        # Exclude date strings
                        if re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", col):
                            continue
                        if col.upper() in ["P", "PASS", "F", "FAIL"]:
                            res_candidate = "P" if "P" in col.upper() else "F"
                        m = re.findall(r"\b\d{1,3}\b", col)
                        for num_val in m:
                            f_val = safe_float(num_val)
                            if 0.0 <= f_val <= 100.0:
                                nums.append(f_val)

                    internal, external, total = 0.0, 0.0, 0.0
                    if len(nums) >= 3:
                        internal, external, total = nums[0], nums[1], nums[2]
                    elif len(nums) == 2:
                        internal, external = nums[0], nums[1]
                        total = calculate_total(internal, external)
                    elif len(nums) == 1:
                        total = nums[0]

                    res = res_candidate if res_candidate else (calculate_result(internal, external) if (internal > 0 or external > 0) else ("P" if total >= 40.0 else "F"))
                    crd = guess_credits(code_cand, name)

                    subjects.append({
                        "subject_code": code_cand,
                        "subject_name": name,
                        "internal": internal,
                        "external": external,
                        "total": total if total > 0 else calculate_total(internal, external),
                        "credits": crd,
                        "result": res
                    })
    finally:
        doc.close()

    return dedupe_subjects(subjects)


# ============================================================
# STRATEGY 2: Selectable Text Stream Parsing
# ============================================================

def _extract_from_text_stream(text: str) -> List[Dict[str, Any]]:
    """Parse subject rows from raw selectable PDF text."""
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    subjects = []
    i = 0
    while i < len(lines):
        line = lines[i]
        code_match = SUBJECT_CODE_RE.search(line)
        if code_match:
            code_cand = code_match.group(0).upper().strip(" .|:;")
            if not is_usn_or_noise(code_cand):
                numeric_sequence = []
                subject_name_parts = []

                # Line remainder after code
                first_remainder = line[code_match.end():].strip(" .|:;_-\t")
                first_no_date = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}", "", first_remainder)
                first_clean_name = re.sub(r"\b\d{1,3}\b.*$", "", first_no_date).strip(" .|:;_-\t")
                if first_clean_name:
                    subject_name_parts.append(first_clean_name)

                # Collect numbers from line 1
                for m in re.findall(r"\b\d{1,3}\b", first_no_date):
                    f_val = safe_float(m)
                    if 0.0 <= f_val <= 100.0:
                        numeric_sequence.append(f_val)

                k = i + 1
                while k < len(lines) and k < i + 8:
                    cand = lines[k]
                    # Stop if another subject code starts
                    if re.search(r"\b(B[A-Z0-9]{2,7}\d{1,4}[A-Z0-9]?|\d{2}[A-Z]{2,6}\d{1,4}[A-Z0-9]?)\b", cand, re.IGNORECASE):
                        break
                    if any(term in cand.upper() for term in ["NOMENCLATURE", "ABBREVIATIONS", "NOTE:", "REGISTRAR", "SD/-"]):
                        break
                    cand_no_date = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}", "", cand)
                    nums = [safe_float(x) for x in re.findall(r"\b\d{1,3}\b", cand_no_date) if 0.0 <= safe_float(x) <= 100.0]
                    if nums:
                        numeric_sequence.extend(nums)
                    cand_text = re.sub(r"\b\d{1,3}\b", "", cand_no_date).strip(" .|:;_-\t")
                    if cand_text and not is_usn_or_noise(cand_text):
                        subject_name_parts.append(cand_text)
                    k += 1

                full_name = clean_subject_name(" ".join(subject_name_parts))
                if not full_name:
                    full_name = f"Subject {code_cand}"

                internal, external, total = 0.0, 0.0, 0.0
                if len(numeric_sequence) >= 3:
                    internal, external, total = numeric_sequence[0], numeric_sequence[1], numeric_sequence[2]
                elif len(numeric_sequence) == 2:
                    internal, external = numeric_sequence[0], numeric_sequence[1]
                    total = calculate_total(internal, external)
                elif len(numeric_sequence) == 1:
                    total = numeric_sequence[0]

                res = calculate_result(internal, external) if (internal > 0 or external > 0) else ("P" if total >= 40.0 else "F")
                crd = guess_credits(code_cand, full_name)

                subjects.append({
                    "subject_code": code_cand,
                    "subject_name": full_name,
                    "internal": internal,
                    "external": external,
                    "total": total if total > 0 else calculate_total(internal, external),
                    "credits": crd,
                    "result": res
                })
                i = k - 1
        i += 1

    return dedupe_subjects(subjects)


# ============================================================
# STRATEGY 3: High-Accuracy Image & Scanned PDF OCR (Tesseract)
# ============================================================

def _ocr_image_or_scanned_pdf(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Perform preprocessed adaptive-threshold OCR on images or scanned PDFs."""
    images = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        finally:
            doc.close()
    else:
        img = Image.open(path).convert("RGB")
        images.append(img)

    all_raw_text = []
    detected_subjects = []

    for img in images:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)
        thresh = gray.point(lambda p: 255 if p > 165 else 0)

        # Run OCR with multiple PSM configurations
        text = ""
        try:
            text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
        except Exception:
            try:
                text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
            except Exception:
                text = ""

        # Fallback to PSM 4 if text is sparse
        if len(text.strip()) < 50:
            try:
                text = pytesseract.image_to_string(img, config="--oem 3 --psm 4")
            except Exception:
                pass

        all_raw_text.append(text)

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            m = SUBJECT_CODE_RE.search(line)
            if m:
                code_cand = m.group(0).upper().strip(" .|:;")
                if not is_usn_or_noise(code_cand):
                    rest_of_line = line[m.end():].strip(" .|:;_-\t")
                    rest_no_date = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}", "", rest_of_line)
                    mark_matches = re.findall(r"\b\d{1,3}\b", rest_no_date)
                    mark_nums = [safe_float(x) for x in mark_matches if 0.0 <= safe_float(x) <= 100.0]

                    name_p1 = re.sub(r"\b\d{1,3}\b.*$", "", rest_no_date).strip(" .|:;_-\t")
                    continuation = []
                    k = i + 1
                    while k < len(lines) and k < i + 6:
                        cand_line = lines[k]
                        if SUBJECT_CODE_RE.search(cand_line):
                            break
                        if any(term in cand_line.upper() for term in ["NOMENCLATURE", "ABBREVIATIONS", "NOTE:", "REGISTRAR", "SD/-"]):
                            break
                        cand_clean = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}", "", cand_line)
                        cand_nums = [safe_float(x) for x in re.findall(r"\b\d{1,3}\b", cand_clean) if 0.0 <= safe_float(x) <= 100.0]
                        if cand_nums and len(mark_nums) < 3:
                            mark_nums.extend(cand_nums)
                        cand_name_only = re.sub(r"\b\d{1,3}\b", "", cand_clean).strip(" .|:;_-\t")
                        if cand_name_only and not is_usn_or_noise(cand_name_only):
                            continuation.append(cand_name_only)
                        k += 1

                    full_name = clean_subject_name((name_p1 + " " + " ".join(continuation)).strip())
                    if not full_name:
                        full_name = f"Subject {code_cand}"

                    internal, external, total = 0.0, 0.0, 0.0
                    if len(mark_nums) >= 3:
                        internal, external, total = mark_nums[0], mark_nums[1], mark_nums[2]
                    elif len(mark_nums) == 2:
                        internal, external = mark_nums[0], mark_nums[1]
                        total = calculate_total(internal, external)
                    elif len(mark_nums) == 1:
                        total = mark_nums[0]

                    res = calculate_result(internal, external) if (internal > 0 or external > 0) else ("P" if total >= 40.0 else "F")
                    crd = guess_credits(code_cand, full_name)

                    detected_subjects.append({
                        "subject_code": code_cand,
                        "subject_name": full_name,
                        "internal": internal,
                        "external": external,
                        "total": total if total > 0 else calculate_total(internal, external),
                        "credits": crd,
                        "result": res
                    })
                    i = k - 1
            i += 1

    return "\n".join(all_raw_text), dedupe_subjects(detected_subjects)


# ============================================================
# MAIN PUBLIC API
# ============================================================

def process_marksheet_file(path: str) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Unified multi-tier entry point to parse VTU Marksheets from PDF or Image files.
    Combines table extraction, stream text parsing, and OCR into a unified, complete list of subjects.
    Returns: (raw_text, detected_subjects, is_low_confidence)
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Marksheet file not found at: {path}")

    ext = os.path.splitext(path)[1].lower()
    all_subjects = []
    combined_texts = []

    if ext == ".pdf":
        # 1. First run Table Parser
        try:
            table_subs = _extract_from_pdf_tables(path)
            if table_subs:
                all_subjects.extend(table_subs)
                combined_texts.append(f"Table Parser found {len(table_subs)} subjects")
        except Exception:
            pass

        # 2. Run Selectable Text Stream Parser
        try:
            doc = fitz.open(path)
            text_chunks = [page.get_text("text") for page in doc if page.get_text("text")]
            doc.close()
            full_pdf_text = "\n".join(text_chunks)
            if full_pdf_text.strip():
                combined_texts.append(full_pdf_text)
                stream_subs = _extract_from_text_stream(full_pdf_text)
                all_subjects.extend(stream_subs)
        except Exception:
            pass

    # 3. If image or if fewer than 6 subjects found from digital stream, run Tesseract OCR
    if ext != ".pdf" or len(dedupe_subjects(all_subjects)) < 6:
        try:
            raw_ocr, ocr_subs = _ocr_image_or_scanned_pdf(path)
            if raw_ocr:
                combined_texts.append(raw_ocr)
            if ocr_subs:
                all_subjects.extend(ocr_subs)
        except Exception:
            pass

    final_subjects = dedupe_subjects(all_subjects)
    is_low = len(final_subjects) == 0
    return "\n\n".join(combined_texts), final_subjects, is_low