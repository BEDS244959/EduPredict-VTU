"""
EduPredict AI - High-Accuracy VTU Marksheet OCR & Parsing Engine
Supports:
1. Standard VTU Digital PDFs (Table extraction & Stream parsing)
2. Scanned / Image PDFs (Multi-candidate Row-Strip scanning with arithmetic verification)
3. Image Marksheets (.PNG, .JPG, .JPEG)
4. Multi-scheme VTU codes (2022 Scheme: BCS/BDS/BMATS/BPHYS/BSCK/BPEK/etc., 2021/2018 Scheme: 21CS/18CS/etc.)
5. Knowledge-based canonical mapping for clean subject names & official credits
"""

from __future__ import annotations
import os
import re
from typing import Any, Tuple, List, Dict, Optional

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

# Windows Tesseract Executable Paths
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Users\vivek\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for p in TESSERACT_PATHS:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break

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

# Reserved words / noise patterns that should NOT be treated as subject codes
NOISE_WORDS = {
    "VTU", "USN", "RESULT", "SEMESTER", "SEM", "MARKS", "TOTAL", "INTERNAL", "EXTERNAL",
    "CREDITS", "GRADE", "GRADE_POINT", "PASS", "FAIL", "BELAGAVI", "REGISTRAR", "EVALUATION",
    "PROVISIONAL", "EXAMINATION", "NOMENCLATURE", "ABBREVIATIONS", "NOTE", "STUDENT", "NAME",
    "UNIVERSITY", "SEAT", "NUMBER", "COLLEGE", "BRANCH", "CAMPUS", "BACHELOR", "MASTER",
    "ENGINEERING", "TECHNOLOGY", "SCIENCE", "DATE", "TIME", "PAGE", "ANNOUNCED", "UPDATED",
    "STATUS", "WITHHELD", "ABSENT", "ELIGIBLE", "REVALUATION", "ANNUAL", "SUPPLEMENTARY",
    "CIE", "SEE", "MAX", "MIN", "OBT", "PERCENTAGE", "REMARKS", "BHOWO", "BUIBWO", "POST",
    "PFO", "NOU", "UB30", "UB30909", "32NN7"
}

# Comprehensive VTU Standard Subject Knowledge Base (Code -> (Clean Standard Name, Standard Credits))
VTU_KNOWN_SUBJECTS = {
    # ---------------- 1st & 2nd Semesters (2022 Scheme) ----------------
    "BMATS101": ("MATHEMATICS-I FOR CSE STREAM", 4.0),
    "BMATE101": ("MATHEMATICS-I FOR ECE STREAM", 4.0),
    "BMATC101": ("MATHEMATICS-I FOR CIVIL STREAM", 4.0),
    "BMATM101": ("MATHEMATICS-I FOR MECH STREAM", 4.0),
    "BCHES102": ("CHEMISTRY FOR CSE STREAM", 4.0),
    "BPHYS102": ("PHYSICS FOR CSE STREAM", 4.0),
    "BPHYE102": ("PHYSICS FOR ECE STREAM", 4.0),
    "BCHEC102": ("CHEMISTRY FOR CIVIL STREAM", 4.0),
    "BCEDK103": ("COMPUTER-AIDED ENGINEERING DRAWING", 3.0),
    "BCEDK203": ("COMPUTER-AIDED ENGINEERING DRAWING", 3.0),
    "BESCK104A": ("INTRODUCTION TO CIVIL ENGINEERING", 3.0),
    "BESCK104B": ("INTRODUCTION TO ELECTRICAL ENGINEERING", 3.0),
    "BESCK104C": ("INTRODUCTION TO ELECTRONICS ENGINEERING", 3.0),
    "BESCK104D": ("INTRODUCTION TO MECHANICAL ENGINEERING", 3.0),
    "BESCK104E": ("INTRODUCTION TO C PROGRAMMING", 3.0),
    "BESCK204A": ("INTRODUCTION TO CIVIL ENGINEERING", 3.0),
    "BESCK204B": ("INTRODUCTION TO ELECTRICAL ENGINEERING", 3.0),
    "BESCK204C": ("INTRODUCTION TO ELECTRONICS ENGINEERING", 3.0),
    "BESCK204D": ("INTRODUCTION TO MECHANICAL ENGINEERING", 3.0),
    "BESCK204E": ("INTRODUCTION TO PYTHON PROGRAMMING", 3.0),
    "BPLCK105A": ("INTRODUCTION TO C PROGRAMMING", 3.0),
    "BPLCK105B": ("INTRODUCTION TO PYTHON PROGRAMMING", 3.0),
    "BPLCK105C": ("INTRODUCTION TO WEB PROGRAMMING", 3.0),
    "BPLCK105D": ("INTRODUCTION TO JAVA PROGRAMMING", 3.0),
    "BPLCK205A": ("INTRODUCTION TO C PROGRAMMING", 3.0),
    "BPLCK205B": ("INTRODUCTION TO PYTHON PROGRAMMING", 3.0),
    "BPLCK205C": ("INTRODUCTION TO WEB PROGRAMMING", 3.0),
    "BPLCK205D": ("INTRODUCTION TO JAVA PROGRAMMING", 3.0),
    "BETCK105H": ("INTRODUCTION TO INTERNET OF THINGS", 3.0),
    "BETCK205H": ("INTRODUCTION TO INTERNET OF THINGS", 3.0),
    "BENGK106": ("COMMUNICATIVE ENGLISH", 1.0),
    "BPWSK206": ("PROFESSIONAL WRITING SKILLS IN ENGLISH", 1.0),
    "BICOK107": ("INDIAN CONSTITUTION", 1.0),
    "BICOK207": ("INDIAN CONSTITUTION", 1.0),
    "BKSKK107": ("BALAKE KANNADA", 1.0),
    "BKSKK207": ("BALAKE KANNADA", 1.0),
    "BKBKK107": ("SAMSKRUTIKA KANNADA", 1.0),
    "BKBKK207": ("SAMSKRUTIKA KANNADA", 1.0),
    "BSFHK158": ("SCIENTIFIC FOUNDATIONS OF HEALTH", 1.0),
    "BSFHK258": ("SCIENTIFIC FOUNDATIONS OF HEALTH", 1.0),
    "BIDTK158": ("INNOVATION AND DESIGN THINKING", 1.0),
    "BIDTK258": ("INNOVATION AND DESIGN THINKING", 1.0),
    "BMATS201": ("MATHEMATICS-II FOR CSE STREAM", 4.0),
    "BMATE201": ("MATHEMATICS-II FOR ECE STREAM", 4.0),
    "BMATC201": ("MATHEMATICS-II FOR CIVIL STREAM", 4.0),
    "BMATM201": ("MATHEMATICS-II FOR MECH STREAM", 4.0),
    "BPHYS202": ("PHYSICS FOR CSE STREAM", 4.0),
    "BCHES202": ("CHEMISTRY FOR CSE STREAM", 4.0),
    "BPOPS203": ("PRINCIPLES OF PROGRAMMING USING C", 3.0),

    # ---------------- 3rd Semester (2022 Scheme) ----------------
    "BDS301": ("MATHEMATICS FOR DATA SCIENCE", 4.0),
    "BCS301": ("MATHEMATICS FOR COMPUTER SCIENCE", 4.0),
    "BMATS301": ("MATHEMATICS FOR CS / DS STREAM", 4.0),
    "BAID301": ("MATHEMATICS FOR AI & ML STREAM", 4.0),
    "BCS302": ("DIGITAL DESIGN & COMPUTER ORGANIZATION", 3.0),
    "BDS302": ("DIGITAL DESIGN & COMPUTER ORGANIZATION", 3.0),
    "BCS303": ("OPERATING SYSTEMS", 3.0),
    "BDS303": ("OPERATING SYSTEMS", 3.0),
    "BCS304": ("DATA STRUCTURES AND APPLICATIONS", 3.0),
    "BDS304": ("DATA STRUCTURES AND APPLICATIONS", 3.0),
    "BCSL305": ("DATA STRUCTURES LAB", 1.5),
    "BDSL305": ("DATA STRUCTURES LAB", 1.5),
    "BCS306A": ("OBJECT ORIENTED PROGRAMMING WITH JAVA", 3.0),
    "BCS306B": ("OBJECT ORIENTED PROGRAMMING WITH C++", 3.0),
    "BCS306C": ("OBJECT ORIENTED PROGRAMMING WITH PYTHON", 3.0),
    "BCS358A": ("DATA ANALYTICS WITH EXCEL", 3.0),
    "BCS358B": ("R PROGRAMMING", 3.0),
    "BCS358C": ("PROJECT MANAGEMENT WITH GIT", 3.0),
    "BCS358D": ("DATA VISUALIZATION WITH PYTHON", 3.0),
    "BSCK307": ("SOCIAL CONNECT AND RESPONSIBILITY", 1.0),
    "BCSK307": ("SOCIAL CONNECT AND RESPONSIBILITY", 1.0),
    "BPEK359": ("PHYSICAL EDUCATION", 0.0),
    "BYOK359": ("YOGA", 0.0),
    "BNSS359": ("NSS", 0.0),

    # ---------------- 4th Semester (2022 Scheme) ----------------
    "BCS401": ("ANALYSIS & DESIGN OF ALGORITHMS", 3.0),
    "BDS401": ("ANALYSIS & DESIGN OF ALGORITHMS", 3.0),
    "BCS402": ("COMPUTER NETWORKS", 3.0),
    "BDS402": ("COMPUTER NETWORKS & SECURITY", 3.0),
    "BCS403": ("DATABASE MANAGEMENT SYSTEMS", 3.0),
    "BDS403": ("DATABASE MANAGEMENT SYSTEMS", 3.0),
    "BCSL404": ("ANALYSIS & DESIGN OF ALGORITHMS LAB", 1.5),
    "BDSL404": ("DATABASE MANAGEMENT SYSTEMS LAB", 1.5),
    "BCS405A": ("DISCRETE MATHEMATICAL STRUCTURES", 3.0),
    "BCS405B": ("GRAPH THEORY", 3.0),
    "BCS405C": ("OPTIMIZATION TECHNIQUES", 3.0),
    "BCS405D": ("LINEAR ALGEBRA", 3.0),
    "BDSL456A": ("DATA WRANGLING WITH PYTHON", 1.5),
    "BDSL456B": ("MONGODB BASICS", 1.5),
    "BDSL456C": ("SQL FOR DATA SCIENCE", 1.5),
    "BBOC407": ("BIOLOGY FOR COMPUTER ENGINEERS", 3.0),
    "BUHK408": ("UNIVERSAL HUMAN VALUES COURSE", 1.0),
    "BPEK459": ("PHYSICAL EDUCATION", 0.0),
    "BYOK459": ("YOGA", 0.0),
    "BNSS459": ("NSS", 0.0),

    # ---------------- 5th & 6th Semesters (2022 Scheme) ----------------
    "BCS501": ("SOFTWARE ENGINEERING & PROJECT MANAGEMENT", 3.0),
    "BCS502": ("COMPUTER NETWORKS", 3.0),
    "BCS503": ("THEORY OF COMPUTATION", 3.0),
    "BCS504": ("ARTIFICIAL INTELLIGENCE & MACHINE LEARNING", 3.0),
    "BCSL505": ("COMPUTER NETWORKS LAB", 1.5),
    "BCS601": ("CLOUD COMPUTING", 3.0),
    "BCS602": ("FULL STACK DEVELOPMENT", 3.0),
    "BCS603": ("CYBER SECURITY & DIGITAL FORENSICS", 3.0),
    "BCSL604": ("CLOUD & FULL STACK LAB", 1.5),

    # ---------------- 2021 & 2018 Schemes ----------------
    "21MAT31": ("TRANSFORM CALCULUS, FOURIER SERIES AND NUMERICAL TECHNIQUES", 3.0),
    "21CS32": ("DATA STRUCTURES AND APPLICATIONS", 4.0),
    "21CS33": ("ANALOG AND DIGITAL ELECTRONICS", 4.0),
    "21CS34": ("COMPUTER ORGANIZATION AND ARCHITECTURE", 3.0),
    "21CSL35": ("OBJECT ORIENTED PROGRAMMING WITH JAVA LAB", 1.0),
    "21MAT41": ("MATHEMATICAL FOUNDATIONS FOR COMPUTING", 3.0),
    "21CS42": ("DESIGN AND ANALYSIS OF ALGORITHMS", 4.0),
    "21CS43": ("MICROCONTROLLER AND EMBEDDED SYSTEMS", 4.0),
    "21CS44": ("OPERATING SYSTEMS", 3.0),
    "21CSL45": ("DESIGN AND ANALYSIS OF ALGORITHMS LAB", 1.0),
    "21CS51": ("MANAGEMENT, ENTREPRENEURSHIP FOR IT INDUSTRY", 3.0),
    "21CS52": ("COMPUTER NETWORKS AND SECURITY", 4.0),
    "21CS53": ("DATABASE MANAGEMENT SYSTEMS", 4.0),
    "21CS54": ("AUTOMATA THEORY AND COMPUTABILITY", 3.0),
    "21CSL55": ("DATABASE MANAGEMENT SYSTEMS LAB", 1.0),
    "18MAT31": ("TRANSFORM CALCULUS, FOURIER SERIES AND NUMERICAL TECHNIQUES", 3.0),
    "18CS32": ("DATA STRUCTURES AND APPLICATIONS", 4.0),
    "18CS33": ("ANALOG AND DIGITAL ELECTRONICS", 4.0),
    "18CS34": ("COMPUTER ORGANIZATION", 3.0),
    "18CS35": ("SOFTWARE ENGINEERING", 3.0),
    "18CS36": ("DISCRETE MATHEMATICAL STRUCTURES", 3.0),
    "18CSL37": ("ANALOG AND DIGITAL ELECTRONICS LAB", 1.5),
    "18CSL38": ("DATA STRUCTURES LABORATORY", 1.5),
    "18CPC39": ("CONSTITUTION OF INDIA, PROFESSIONAL ETHICS & CYBER LAW", 1.0),
}


def normalize_subject_code(raw_code: str) -> str:
    """Normalize and fix OCR misread subject codes."""
    if not raw_code:
        return ""
    c = str(raw_code).upper().strip(" .|:;-_,/\\}")
    c = re.sub(r"\s+", "", c)

    # Prefix replacement
    prefix_map = {
        "8CSL": "BCSL", "8DSL": "BDSL", "8CS": "BCS", "8DS": "BDS",
        "RCS": "BCS", "RDS": "BDS",
        "8SCK": "BSCK", "8PEK": "BPEK", "8MAT": "BMAT", "8AID": "BAID",
        "8ES": "BES", "8PL": "BPL", "8ET": "BET", "8IC": "BIC",
        "8SF": "BSFH", "8UH": "BUHK", "8CHE": "BCHE", "8PHY": "BPHY",
        "PEKG=9": "BPEK359", "PEK359": "BPEK359", "SCK307": "BSCK307"
    }
    for bad_p, good_p in prefix_map.items():
        if c.startswith(bad_p):
            c = good_p + c[len(bad_p):]
            break

    # Fix letter 'O' vs digit '0' in numeric portion
    m = re.match(r"^([A-Z]{2,7}|\d{2}[A-Z]{2,5})([0-9O]{2,4})([A-Z]?)$", c)
    if m:
        pref, num_part, suf = m.group(1), m.group(2), m.group(3)
        num_fixed = num_part.replace("O", "0")
        c = f"{pref}{num_fixed}{suf}"

    return c


def get_canonical_subject_info(code: str, extracted_name: str = "") -> Tuple[str, float]:
    """Retrieve authoritative clean standard subject name and credits from VTU knowledge base."""
    norm_code = normalize_subject_code(code)
    if norm_code in VTU_KNOWN_SUBJECTS:
        return VTU_KNOWN_SUBJECTS[norm_code][0], VTU_KNOWN_SUBJECTS[norm_code][1]

    # Fallback to cleaned extracted name if not in knowledge base
    clean_n = clean_subject_name(extracted_name)
    crd = guess_credits(norm_code, clean_n)
    if not clean_n:
        clean_n = f"Subject {norm_code}"
    return clean_n, crd


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None or value == "":
            return default
        cleaned = str(value).replace(",", "").strip()
        return float(cleaned)
    except (TypeError, ValueError):
        return default


def resolve_marks_triplet(nums: List[float], subject_code: str = "", res_cand: Optional[str] = None) -> Tuple[float, float, float, str]:
    """
    Intelligently deduce (Internal, External, Total, Result) using arithmetic verification and VTU scheme rules.
    VTU Marksheet Column order is always: [Internal Marks, External Marks, Total Marks].
    VTU Passing Rule:
    - Minimum External (SEE) = 18.0 (out of 50)
    - Minimum Aggregate Total = 40.0 (out of 100)
    - If External < 18.0 or Total < 40.0 -> Result is FAIL (F).
    """
    code_up = normalize_subject_code(subject_code)

    if not nums:
        return 0.0, 0.0, 0.0, "F"

    # 1. 3-value extracted rows: [Internal, External, Total]
    if len(nums) >= 3:
        n1, n2, n3 = nums[0], nums[1], nums[2]
        if abs((n1 + n2) - n3) <= 1.0:
            int_m, ext_m, tot_m = n1, n2, n3
        elif abs((n1 + n3) - n2) <= 1.0:
            int_m, ext_m, tot_m = n1, n3, n2
        elif 40.0 <= n3 <= 100.0 and 10.0 <= n2 <= 50.0 and n1 < 10.0:
            int_m, ext_m, tot_m = round(n3 - n2, 2), n2, n3
        elif 40.0 <= n3 <= 100.0 and 10.0 <= n1 <= 50.0 and n2 < 10.0:
            int_m, ext_m, tot_m = n1, round(n3 - n1, 2), n3
        elif n1 <= 50.0 and n2 <= 50.0:
            int_m, ext_m, tot_m = n1, n2, round(n1 + n2, 2)
        else:
            int_m, ext_m, tot_m = n1, n2, round(n1 + n2, 2)

    # 2. 2-value extracted rows: [Internal, External] or [Internal, Total]
    elif len(nums) == 2:
        n1, n2 = nums[0], nums[1]
        if n1 <= 50.0 and n2 <= 50.0:
            int_m, ext_m, tot_m = n1, n2, round(n1 + n2, 2)
        elif n1 <= 50.0 and 50.0 < n2 <= 100.0:
            int_m, ext_m, tot_m = n1, round(n2 - n1, 2), n2
        else:
            int_m, ext_m, tot_m = n1, n2, round(n1 + n2, 2)

    # 3. 1-value extracted row (e.g. 100-mark Continuous Internal Evaluation course)
    elif len(nums) == 1:
        tot_m = nums[0]
        int_m, ext_m = tot_m, 0.0
    else:
        int_m, ext_m, tot_m = 0.0, 0.0, 0.0

    # VTU Pass/Fail Evaluation Criteria:
    # 1. External exam < 18 is an automatic FAIL in VTU
    # 2. Total < 40 is FAIL
    # 3. Both External >= 18 and Total >= 40 required to PASS
    if ext_m > 0.0:
        if ext_m < 18.0 or tot_m < 40.0:
            final_res = "F"
        else:
            final_res = "P"
    elif tot_m >= 40.0 and int_m > 50.0:
        # Pure 100-mark continuous internal evaluation course (e.g. 85/100)
        final_res = "P"
    elif tot_m >= 40.0 and any(k in code_up for k in ["PEK", "YOK", "NSS", "AUDIT"]):
        final_res = "P"
    else:
        final_res = "F" if tot_m < 40.0 else ("P" if any(k in code_up for k in ["PEK", "YOK", "NSS"]) else "F")

    return int_m, ext_m, tot_m, final_res


def guess_credits(subject_code: str, subject_name: str = "") -> float:
    """Infer VTU course credits based on subject code scheme and name."""
    code = normalize_subject_code(subject_code)
    name = str(subject_name).upper().strip()

    if code in VTU_KNOWN_SUBJECTS:
        return VTU_KNOWN_SUBJECTS[code][1]

    # 0-credit audit / mandatory non-credit courses
    if any(k in code for k in ["PEK", "BPEK", "YOK", "NSS"]) or any(k in name for k in ["PHYSICAL EDUCATION", "YOGA", "NSS", "AUDIT"]):
        return 0.0

    # Laboratory courses (usually 1.5 credits)
    if "L" in code[3:] or "LAB" in name or "PRACTICAL" in name:
        return 1.5

    # 1-credit mandatory non-technical / ability enhancement courses
    if any(k in code for k in ["SFH", "CIP", "ICO", "WSK", "KSK", "IDT", "UHK", "SCK", "ENG", "CPH"]):
        return 1.0

    # Foundational Sciences / Core Mathematics in Sem 1, 2 & 3 (typically 4.0 credits)
    if any(code.startswith(k) for k in ["BMATS1", "BMATS2", "BMATS3", "BDS301", "BCS301", "BPHYS1", "BPHYS2", "BCHES1", "BCHES2", "21MAT1", "21MAT2", "21MAT3"]):
        return 4.0

    # Core Engineering / Theory / Electives default to 3.0 credits
    return 3.0


def is_usn_or_noise(code: str) -> bool:
    """Check if token is a Student USN or reserved noise keyword."""
    if not code:
        return True
    c = str(code).upper().strip()
    if re.fullmatch(r"1[A-Z]{2}\d{2}[A-Z]{2}\d{3}", c):
        return True
    if re.fullmatch(r"\d{4}", c):
        return True
    if c in NOISE_WORDS:
        return True
    if len(c) < 4 or len(c) > 10:
        return True
    # Must contain at least 2 alphabet letters and at least 2 digits
    alpha_count = sum(1 for ch in c if ch.isalpha())
    digit_count = sum(1 for ch in c if ch.isdigit())
    if alpha_count < 2 or digit_count < 2:
        return True
    return False


def clean_subject_name(name: str) -> str:
    """Clean subject title text from OCR artifacts, dates, repeated words, and noise words."""
    if not name:
        return ""
    n = name.replace("\n", " ").replace("\r", " ")
    noise_patterns = [
        r"\b(moabedea|Sofedironed|Nomenclature|Abbreviations|Note|REGISTRAR|EVALUATION|Announced|Updated|Result)\b.*$",
        r"P\s*->\s*PASS.*$",
        r"F\s*->\s*FAIL.*$",
        r"\d{4}[-/]\d{2}[-/]\d{2}",
        r"[|:;_~\\/=><]+"
    ]
    for np in noise_patterns:
        n = re.sub(np, " ", n, flags=re.IGNORECASE)
    
    # Remove consecutive duplicate words
    words = [w.strip(" ,.-") for w in n.split() if w.strip(" ,.-")]
    deduped_words = []
    for w in words:
        if not deduped_words or w.upper() != deduped_words[-1].upper():
            deduped_words.append(w)
    n = " ".join(deduped_words)

    n = re.sub(r"\s+", " ", n).strip(" -.,}")
    return n


def dedupe_subjects(subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate subject codes, preserving the highest-quality detected record."""
    seen = {}
    for sub in subjects:
        raw_code = str(sub.get("subject_code", "")).upper().strip()
        code = normalize_subject_code(raw_code)
        if not code or is_usn_or_noise(code):
            continue

        std_name, std_crd = get_canonical_subject_info(code, sub.get("subject_name", ""))
        sub["subject_code"] = code
        sub["subject_name"] = std_name
        sub["credits"] = std_crd

        # Quality scoring
        int_m = safe_float(sub.get("internal"))
        ext_m = safe_float(sub.get("external"))
        tot_m = safe_float(sub.get("total"))

        if abs(int_m + ext_m - tot_m) <= 1.0 and tot_m >= 40.0 and int_m > 0:
            score = 200 + tot_m
        elif tot_m > 0 and (int_m > 0 or ext_m > 0):
            score = 100 + tot_m
        elif tot_m > 0:
            score = 50 + tot_m
        else:
            score = 1

        sub["_score"] = score

        if code not in seen:
            seen[code] = dict(sub)
        else:
            old = seen[code]
            if score > old.get("_score", 0):
                seen[code] = dict(sub)
                
    for k in seen:
        seen[k].pop("_score", None)
        
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
                        val = normalize_subject_code(re.sub(r"[^A-Za-z0-9]", "", row_str[ci]))
                        if not val or is_usn_or_noise(val):
                            continue
                        if re.match(r"^(B[A-Z0-9]{2,7}\d{1,4}[A-Z0-9]?|\d{2}[A-Z]{2,6}\d{1,4}[A-Z0-9]?|[A-Z]{2,6}\d{1,4}[A-Z0-9]?)$", val):
                            code_cand = val
                            code_idx = ci
                            break

                    if not code_cand:
                        continue

                    raw_name = ""
                    if code_idx + 1 < len(row_str):
                        raw_name = row_str[code_idx + 1]
                    name, crd = get_canonical_subject_info(code_cand, raw_name)

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

                    internal, external, total, res = resolve_marks_triplet(nums, code_cand, res_candidate)

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
            code_cand = normalize_subject_code(code_match.group(0))
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

                raw_full_name = " ".join(subject_name_parts)
                name, crd = get_canonical_subject_info(code_cand, raw_full_name)
                internal, external, total, res = resolve_marks_triplet(numeric_sequence, code_cand, res_cand)

                subjects.append({
                    "subject_code": code_cand,
                    "subject_name": name,
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
# STRATEGY 3: High-Accuracy Multi-Candidate Row-Strip Scanning
# ============================================================

def _ocr_image_or_scanned_pdf(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Perform sliding row-strip scanning and arithmetic evaluation on scanned documents and images."""
    images = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        finally:
            doc.close()
    else:
        img = Image.open(path).convert("RGB")
        images.append(img)

    all_raw_text = []
    detected_candidates: List[Dict[str, Any]] = []

    for img in images:
        y_step = 25
        y_range = range(800, min(img.height - 70, 2300), y_step)
        
        for y in y_range:
            code_crop = img.crop((40, y, 450, min(img.height, y + 70)))
            c_gray = ImageOps.grayscale(code_crop)
            c_bw = ImageEnhance.Contrast(c_gray).enhance(2.5).point(lambda p: 255 if p > 150 else 0)
            
            try:
                txt = pytesseract.image_to_string(c_bw, config="--oem 3 --psm 6").strip()
                m = SUBJECT_CODE_RE.search(txt)
                if m:
                    code_cand = normalize_subject_code(m.group(0))
                    if not is_usn_or_noise(code_cand):
                        # Extract marks horizontal strip at this exact y
                        marks_crop = img.crop((950, max(0, y - 12), min(img.width, 1760), min(img.height, y + 68)))
                        m_gray = ImageOps.grayscale(marks_crop)
                        m_bw = ImageEnhance.Contrast(m_gray).enhance(3.0).point(lambda p: 255 if p > 150 else 0)
                        
                        marks_raw = pytesseract.image_to_string(m_bw, config="--oem 3 --psm 6").strip()
                        all_raw_text.append(f"[{code_cand} at Y={y}] {marks_raw}")
                        
                        marks_no_date = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}.*$", "", marks_raw)
                        
                        # Fix common OCR number artifacts
                        cleaned_str = marks_no_date.replace("Al", "41").replace("AL", "41").replace("al", "41")
                        cleaned_str = re.sub(r"R35|B35", "85", cleaned_str)
                        cleaned_str = re.sub(r"B32|R32|B82", "82", cleaned_str)
                        
                        raw_nums = re.findall(r"\b\d{1,3}\b", cleaned_str)
                        nums = [safe_float(n) for n in raw_nums if 0.0 <= safe_float(n) <= 100.0]
                        
                        name, crd = get_canonical_subject_info(code_cand)
                        internal, external, total, res = resolve_marks_triplet(nums, code_cand)

                        detected_candidates.append({
                            "subject_code": code_cand,
                            "subject_name": name,
                            "internal": internal,
                            "external": external,
                            "total": total,
                            "credits": crd,
                            "result": res,
                            "y": y
                        })
            except Exception:
                pass

    final_subjects = dedupe_subjects(detected_candidates)
    return "\n".join(all_raw_text), final_subjects


# ============================================================
# MAIN PUBLIC API
# ============================================================

def process_marksheet_file(path: str) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Unified multi-tier entry point to parse VTU Marksheets from PDF or Image files.
    Combines table extraction, stream text parsing, and row-strip OCR into a complete list of subjects.
    Returns: (raw_text, detected_subjects, is_low_confidence)
    """
    if not path:
        raise FileNotFoundError("Marksheet file path is empty.")
    if not os.path.exists(path):
        norm = str(path).replace("\\", "/")
        if "uploads" in norm:
            rel = "uploads/" + norm.split("uploads", 1)[1].lstrip("/")
            if os.path.exists(rel):
                path = rel
            elif os.path.exists(os.path.join(os.getcwd(), rel)):
                path = os.path.join(os.getcwd(), rel)
    if not os.path.exists(path):
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