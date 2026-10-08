"""
EduPredict AI - High-Accuracy VTU Marksheet OCR & Parsing Engine
Supports:
1. Standard VTU Digital PDFs (Table extraction & Stream parsing)
2. Scanned / Image PDFs (Adaptive thresholded high-speed OCR)
3. Image Marksheets (.PNG, .JPG, .JPEG)
4. Multi-scheme VTU codes (2022 Scheme: BCS/BDS/BMATS/BPHYS/BSCK/BPEK/etc., 2021/2018 Scheme: 21CS/18CS/etc.)
5. Multi-line subject names & accurate mark-triplet parsing
"""

from __future__ import annotations
import os
import re
from typing import Any, Tuple, List, Dict, Optional

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


def resolve_marks_triplet(nums: List[float], res_cand: Optional[str] = None) -> Tuple[float, float, float, str]:
    """
    Intelligently deduce (Internal, External, Total, Result) using arithmetic verification.
    """
    if not nums:
        return 0.0, 0.0, 0.0, "F"

    if len(nums) >= 3:
        n1, n2, n3 = nums[0], nums[1], nums[2]
        # Standard: Internal + External = Total
        if abs((n1 + n2) - n3) <= 1.0:
            int_m, ext_m, tot_m = n1, n2, n3
        elif n1 == n2 and n3 == n1:
            int_m, ext_m, tot_m = n1, 0.0, n1
        elif n2 == 0.0 and n1 == n3:
            int_m, ext_m, tot_m = n1, 0.0, n1
        elif n1 == n3 and n2 > 0:
            int_m, ext_m, tot_m = n1, 0.0, n1
        elif abs((n1 + n3) - n2) <= 1.0:
            int_m, ext_m, tot_m = n1, n3, n2
        else:
            int_m, ext_m, tot_m = n1, n2, n3
    elif len(nums) == 2:
        n1, n2 = nums[0], nums[1]
        if n1 == n2:
            int_m, ext_m, tot_m = n1, 0.0, n1
        elif n1 > 50 and n2 == 0:
            int_m, ext_m, tot_m = n1, 0.0, n1
        elif n1 <= 50 and n2 <= 50:
            int_m, ext_m, tot_m = n1, n2, round(n1 + n2, 2)
        elif n2 > n1 and n2 <= 100:
            int_m, ext_m, tot_m = n1, round(n2 - n1, 2), n2
        else:
            int_m, ext_m, tot_m = n1, n2, round(n1 + n2, 2)
    elif len(nums) == 1:
        tot_m = nums[0]
        int_m, ext_m = (tot_m, 0.0) if tot_m > 50 else (round(tot_m/2, 2), round(tot_m/2, 2))
    else:
        int_m, ext_m, tot_m = 0.0, 0.0, 0.0

    # VTU Result Logic
    if res_cand and res_cand.upper() in ["P", "PASS", "F", "FAIL"]:
        final_res = "P" if "P" in res_cand.upper() else "F"
    else:
        # If external exam exists and is < 18, it's FAIL in VTU
        if ext_m > 0 and ext_m < 18.0:
            final_res = "F"
        elif tot_m < 40.0:
            final_res = "F"
        else:
            final_res = "P"

    return int_m, ext_m, tot_m, final_res


def guess_credits(subject_code: str, subject_name: str = "") -> float:
    """Infer VTU course credits based on subject code scheme and name."""
    code = str(subject_code).upper().strip()
    name = str(subject_name).upper().strip()

    # 0-credit audit / mandatory non-credit courses
    if any(k in code for k in ["PEK", "BPEK"]) or any(k in name for k in ["PHYSICAL EDUCATION", "YOGA", "NSS", "AUDIT"]):
        return 0.0

    # Laboratory courses (usually 1.5 credits)
    if "L" in code[3:] or "LAB" in name or "PRACTICAL" in name:
        return 1.5

    # 1-credit mandatory non-technical / ability enhancement courses
    if any(k in code for k in ["SFH", "CIP", "ICO", "WSK", "KSK", "IDT", "UHK", "SCK", "ENG", "CPH"]):
        return 1.0

    # Foundational Sciences / Core Mathematics in Sem 1 & 2 (typically 4.0 credits)
    if any(code.startswith(k) for k in ["BMATS1", "BMATS2", "BPHYS1", "BPHYS2", "BCHES1", "BCHES2", "21MAT1", "21MAT2"]):
        return 4.0

    # Core Engineering / Theory / Electives default to 3.0 credits
    return 3.0


def is_usn_or_noise(code: str) -> bool:
    """Check if token is a Student USN or reserved noise keyword."""
    if not code:
        return True
    c = code.upper().strip()
    if re.fullmatch(r"1[A-Z]{2}\d{2}[A-Z]{2}\d{3}", c):
        return True
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
    noise_patterns = [
        r"\b(moabedea|Sofedironed|Nomenclature|Abbreviations|Note|REGISTRAR|EVALUATION|Announced|Updated|Result)\b.*$",
        r"P\s*->\s*PASS.*$",
        r"F\s*->\s*FAIL.*$",
        r"\d{4}[-/]\d{2}[-/]\d{2}",
        r"[|:;_~\\/]+"
    ]
    for np in noise_patterns:
        n = re.sub(np, " ", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+", " ", n).strip(" -.,}")
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
            if len(str(sub.get("subject_name", ""))) > len(str(old.get("subject_name", ""))):
                old["subject_name"] = sub.get("subject_name", "")
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

                    name = ""
                    if code_idx + 1 < len(row_str):
                        name = clean_subject_name(row_str[code_idx + 1])
                    if not name:
                        name = f"Subject {code_cand}"

                    nums = []
                    res_candidate = None
                    for col in row_str[code_idx + 2:]:
                        if re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", col):
                            continue
                        if col.upper() in ["P", "PASS", "F", "FAIL"]:
                            res_candidate = "P" if "P" in col.upper() else "F"
                        m = re.findall(r"\b\d{1,3}\b", col)
                        for num_val in m:
                            f_val = safe_float(num_val)
                            if 0.0 <= f_val <= 100.0:
                                nums.append(f_val)

                    internal, external, total, res = resolve_marks_triplet(nums, res_candidate)
                    crd = guess_credits(code_cand, name)

                    subjects.append({
                        "subject_code": code_cand,
                        "subject_name": name,
                        "internal": internal,
                        "external": external,
                        "total": total,
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
                res_cand = None

                first_remainder = line[code_match.end():].strip(" .|:;_-\t")
                first_no_date = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}", "", first_remainder)
                first_clean_name = re.sub(r"\b\d{1,3}\b.*$", "", first_no_date).strip(" .|:;_-\t")
                if first_clean_name:
                    subject_name_parts.append(first_clean_name)

                for m in re.findall(r"\b\d{1,3}\b", first_no_date):
                    f_val = safe_float(m)
                    if 0.0 <= f_val <= 100.0:
                        numeric_sequence.append(f_val)

                k = i + 1
                while k < len(lines) and k < i + 8:
                    cand = lines[k]
                    if re.search(r"\b(B[A-Z0-9]{2,7}\d{1,4}[A-Z0-9]?|\d{2}[A-Z]{2,6}\d{1,4}[A-Z0-9]?)\b", cand, re.IGNORECASE):
                        break
                    if any(term in cand.upper() for term in ["NOMENCLATURE", "ABBREVIATIONS", "NOTE:", "REGISTRAR", "SD/-"]):
                        break
                    if "PASS" in cand.upper() or " P " in f" {cand.upper()} ":
                        res_cand = "P"
                    elif "FAIL" in cand.upper() or " F " in f" {cand.upper()} ":
                        res_cand = "F"

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

                internal, external, total, res = resolve_marks_triplet(numeric_sequence, res_cand)
                crd = guess_credits(code_cand, full_name)

                subjects.append({
                    "subject_code": code_cand,
                    "subject_name": full_name,
                    "internal": internal,
                    "external": external,
                    "total": total,
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
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
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
        # Background watermark filter
        thresh = gray.point(lambda p: 255 if p > 155 else 0)

        text = ""
        try:
            text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
        except Exception:
            try:
                text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
            except Exception:
                text = ""

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
                    res_cand = None
                    k = i + 1
                    while k < len(lines) and k < i + 6:
                        cand_line = lines[k]
                        if SUBJECT_CODE_RE.search(cand_line):
                            break
                        if any(term in cand_line.upper() for term in ["NOMENCLATURE", "ABBREVIATIONS", "NOTE:", "REGISTRAR", "SD/-"]):
                            break
                        if "PASS" in cand_line.upper() or " P " in f" {cand_line.upper()} ":
                            res_cand = "P"
                        elif "FAIL" in cand_line.upper() or " F " in f" {cand_line.upper()} ":
                            res_cand = "F"

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

                    internal, external, total, res = resolve_marks_triplet(mark_nums, res_cand)
                    crd = guess_credits(code_cand, full_name)

                    detected_subjects.append({
                        "subject_code": code_cand,
                        "subject_name": full_name,
                        "internal": internal,
                        "external": external,
                        "total": total,
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
        try:
            table_subs = _extract_from_pdf_tables(path)
            if table_subs:
                all_subjects.extend(table_subs)
                combined_texts.append(f"Table Parser found {len(table_subs)} subjects")
        except Exception:
            pass

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