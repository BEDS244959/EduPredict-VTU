from __future__ import annotations

import json
from typing import Any

from app.ocr_engine import process_marksheet_file, safe_float


def extract_marksheet(path: str) -> dict[str, Any]:
    try:
        raw_text, subjects, ok = process_marksheet_file(path)
    except Exception as exc:
        return {
            "ok": False,
            "message": "Could not process this file. Please enter marks manually.",
            "error": str(exc),
            "subjects": [],
            "raw_excerpt": "",
        }
    rows = []
    for s in subjects or []:
        internal = safe_float(s.get("internal") if isinstance(s, dict) else None)
        if internal is None:
            internal = safe_float(s.get("internal_marks") if isinstance(s, dict) else None)
        external = safe_float(s.get("external") if isinstance(s, dict) else None)
        if external is None:
            external = safe_float(s.get("external_marks") if isinstance(s, dict) else None)
        total = None
        if internal is not None or external is not None:
            total = round(float(internal or 0) + float(external or 0), 2)
        code = (s.get("subject_code") or s.get("code") or "").strip() if isinstance(s, dict) else ""
        name = (s.get("subject_name") or s.get("name") or "").strip() if isinstance(s, dict) else ""
        credits = safe_float(s.get("credits")) if isinstance(s, dict) else None
        uncertain = []
        if not code:
            uncertain.append("subject_code")
        if not name:
            uncertain.append("subject_name")
        if internal is None:
            uncertain.append("internal")
        if external is None:
            uncertain.append("external")
        rows.append(
            {
                "subject_code": code,
                "subject_name": name,
                "internal": internal,
                "external": external,
                "total": total,
                "credits": credits or 3,
                "grade": s.get("grade") if isinstance(s, dict) else None,
                "result": s.get("result") if isinstance(s, dict) else None,
                "uncertain_fields": uncertain,
                "manual_needed": bool(uncertain),
                "hint": "Could not confidently read this field. Please enter it manually." if uncertain else "",
            }
        )
    return {
        "ok": bool(rows),
        "message": None if rows else "Could not confidently extract subjects. Please enter them manually.",
        "subjects": rows,
        "raw_excerpt": (raw_text or "")[:4000],
        "engine_ok": bool(ok),
    }
