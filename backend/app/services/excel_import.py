from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

CANON = {
    "usn": ["usn", "student usn", "university seat number"],
    "name": ["name", "student name", "student"],
    "subject_code": ["subject code", "code", "sub code"],
    "subject_name": ["subject name", "subject", "course"],
    "credits": ["credits", "credit"],
    "ia1": ["ia1", "ia 1", "internal 1", "cie1"],
    "ia2": ["ia2", "ia 2", "internal 2", "cie2"],
    "internal": ["internal", "internal marks", "cie"],
    "external": ["external", "see", "external marks"],
    "total": ["total", "total marks"],
    "semester": ["semester", "sem"],
}


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def detect_mapping(columns: list[str]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {k: None for k in CANON}
    norms = {_norm(c): c for c in columns}
    for field, aliases in CANON.items():
        for a in aliases:
            if a in norms:
                mapping[field] = norms[a]
                break
    return mapping


def validate_row(row: dict[str, Any], known_usns: set[str], ia_max: float, ext_max: float) -> list[str]:
    errors = []
    usn = str(row.get("usn") or "").strip().upper()
    if not usn:
        errors.append("USN is required")
    elif known_usns and usn not in known_usns:
        errors.append("Unknown USN")
    if not str(row.get("subject_code") or "").strip():
        errors.append("Subject code is required")
    if not str(row.get("subject_name") or "").strip():
        errors.append("Subject name is required")
    for key, mx in (("ia1", ia_max), ("ia2", ia_max), ("internal", ia_max), ("external", ext_max), ("total", ia_max + ext_max)):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
            continue
        try:
            n = float(val)
        except (TypeError, ValueError):
            errors.append(f"{key} is not a number")
            continue
        if n < 0 or n > mx:
            errors.append(f"{key} must be between 0 and {mx}")
    return errors


def mapped_records(df: pd.DataFrame, mapping: dict[str, str | None]) -> list[dict[str, Any]]:
    out = []
    for _, raw in df.iterrows():
        rec = {}
        for field, col in mapping.items():
            if not col:
                rec[field] = None
            else:
                val = raw.get(col)
                rec[field] = None if pd.isna(val) else val
        if rec.get("usn"):
            rec["usn"] = str(rec["usn"]).strip().upper()
        out.append(rec)
    return out


def error_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b"row,errors\n"
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
