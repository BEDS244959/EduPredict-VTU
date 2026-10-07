from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.models import Setting

DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "academic": {
        "ia_max": 50,
        "external_max": 100,
        "total_max": 100,
        "passing_marks": 40,
        "internal_method": "average",
        "internal_weight": 0.5,
        "credits_default": 3,
        "grade_boundaries": [
            {"min": 90, "letter": "O", "point": 10},
            {"min": 80, "letter": "A+", "point": 9},
            {"min": 70, "letter": "A", "point": 8},
            {"min": 60, "letter": "B+", "point": 7},
            {"min": 55, "letter": "B", "point": 6},
            {"min": 50, "letter": "C", "point": 5},
            {"min": 40, "letter": "P", "point": 4},
            {"min": 0, "letter": "F", "point": 0},
        ],
    },
    "attendance": {"min_percent": 75, "warning_percent": 85},
    "files": {
        "max_mb": 15,
        "allowed_ext": [".pdf", ".jpg", ".jpeg", ".png", ".xlsx", ".xls", ".csv"],
    },
    "ml": {"min_samples": 10, "allow_rule_fallback": True},
}


def get_group(db: Session, group: str) -> dict[str, Any]:
    row = db.query(Setting).filter(Setting.key == group).first()
    if not row:
        return dict(DEFAULT_SETTINGS.get(group, {}))
    try:
        return json.loads(row.value)
    except json.JSONDecodeError:
        return dict(DEFAULT_SETTINGS.get(group, {}))


def set_group(db: Session, group: str, value: dict[str, Any]) -> dict[str, Any]:
    row = db.query(Setting).filter(Setting.key == group).first()
    payload = json.dumps(value)
    if row:
        row.value = payload
        row.group = group
    else:
        db.add(Setting(key=group, value=payload, group=group))
    db.commit()
    return value


def grade_for_total(total: float | None, academic: dict[str, Any] | None = None) -> tuple[str, float]:
    if total is None:
        return "NA", 0.0
    try:
        t = float(total)
    except (TypeError, ValueError):
        return "NA", 0.0
    bounds = (academic or DEFAULT_SETTINGS["academic"])["grade_boundaries"]
    for b in bounds:
        if t >= float(b["min"]):
            return str(b["letter"]), float(b["point"])
    return "F", 0.0


def combine_internal(ia1: float, ia2: float, academic: dict[str, Any] | None = None) -> float:
    cfg = academic or DEFAULT_SETTINGS["academic"]
    method = cfg.get("internal_method", "average")
    a, b = float(ia1 or 0), float(ia2 or 0)
    if method == "best":
        return round(max(a, b), 2)
    if method == "sum":
        return round(min(a + b, float(cfg.get("ia_max", 50))), 2)
    return round((a + b) / 2.0, 2)


def compute_total(internal: float | None, external: float | None) -> float | None:
    if internal is None and external is None:
        return None
    return round(float(internal or 0) + float(external or 0), 2)


def result_for_total(total: float | None, academic: dict[str, Any] | None = None) -> str:
    if total is None:
        return "NA"
    passing = float((academic or DEFAULT_SETTINGS["academic"]).get("passing_marks", 40))
    return "P" if float(total) >= passing else "F"


def sgpa_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    credit_points = 0.0
    credits = 0.0
    breakdown = []
    for r in rows:
        cr = float(r.get("credits") or 0)
        gp = float(r.get("grade_point") or 0)
        cp = cr * gp
        credit_points += cp
        credits += cr
        breakdown.append(
            {
                "subject_code": r.get("subject_code"),
                "subject_name": r.get("subject_name"),
                "credits": cr,
                "grade": r.get("grade"),
                "grade_point": gp,
                "credit_points": round(cp, 2),
            }
        )
    sgpa = round(credit_points / credits, 2) if credits else 0.0
    return {
        "sgpa": sgpa,
        "total_credits": credits,
        "total_credit_points": round(credit_points, 2),
        "breakdown": breakdown,
    }


def cgpa_from_semesters(semesters: list[dict[str, Any]]) -> dict[str, Any]:
    weighted = 0.0
    credits = 0.0
    for s in semesters:
        cr = float(s.get("total_credits") or 0)
        sg = float(s.get("sgpa") or 0)
        if cr <= 0:
            continue
        weighted += sg * cr
        credits += cr
    return {
        "cgpa": round(weighted / credits, 2) if credits else 0.0,
        "total_credits": credits,
    }
