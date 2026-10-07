from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.auth import get_current_user, hash_password, password_issues, require_roles, write_audit
from app.database import get_db
from app.models import (
    AcademicYear,
    Admin,
    AuditLog,
    ClassRoom,
    Department,
    Staff,
    Student,
    SubjectCatalog,
    User,
)
from app.services.academic import DEFAULT_SETTINGS, get_group, set_group
from app.services.ml_engine import active_model, train_models
from app.services.reports import csv_bytes

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/dashboard")
def dashboard(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    return {
        "stats": {
            "students": db.query(Student).count(),
            "staff": db.query(Staff).count(),
            "classes": db.query(ClassRoom).count(),
            "departments": db.query(Department).count(),
            "subjects": db.query(SubjectCatalog).count(),
        },
        "empty": db.query(Student).count() == 0,
        "message": "No students yet" if db.query(Student).count() == 0 else None,
    }


@router.get("/departments")
def departments(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = db.query(Department).all()
    return {"items": [{"id": r.id, "name": r.name, "code": r.code, "description": r.description} for r in rows], "empty": not rows}


@router.post("/departments")
def add_dept(payload: dict, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rec = Department(name=payload["name"], code=str(payload["code"]).upper(), description=payload.get("description"))
    db.add(rec)
    db.commit()
    return {"id": rec.id}


@router.put("/departments/{did}")
def edit_dept(did: str, payload: dict, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rec = db.query(Department).filter(Department.id == did).first()
    if not rec:
        raise HTTPException(404, "Not found")
    rec.name = payload.get("name", rec.name)
    rec.code = str(payload.get("code", rec.code)).upper()
    rec.description = payload.get("description", rec.description)
    db.commit()
    return {"message": "Updated"}


@router.delete("/departments/{did}")
def del_dept(did: str, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rec = db.query(Department).filter(Department.id == did).first()
    if rec:
        db.delete(rec)
        db.commit()
    return {"message": "Deleted"}


@router.get("/subjects")
def subjects(user: User = Depends(require_roles("admin", "staff")), db: Session = Depends(get_db)):
    rows = db.query(SubjectCatalog).order_by(SubjectCatalog.semester, SubjectCatalog.code).all()
    return {"items": [{"id": r.id, "code": r.code, "name": r.name, "credits": r.credits, "semester": r.semester, "branch": r.branch} for r in rows], "empty": not rows, "message": "No subjects in catalogue" if not rows else None}


@router.post("/subjects")
def add_subject(payload: dict, user: User = Depends(require_roles("admin", "staff")), db: Session = Depends(get_db)):
    rec = SubjectCatalog(
        code=str(payload["code"]).upper(),
        name=payload["name"],
        credits=float(payload.get("credits") or 3),
        semester=int(payload.get("semester") or 1),
        branch=payload.get("branch"),
    )
    db.add(rec)
    db.commit()
    return {"id": rec.id}


@router.put("/subjects/{sid}")
def edit_subject(sid: str, payload: dict, user: User = Depends(require_roles("admin", "staff")), db: Session = Depends(get_db)):
    rec = db.query(SubjectCatalog).filter(SubjectCatalog.id == sid).first()
    if not rec:
        raise HTTPException(404, "Not found")
    for f in ("code", "name", "credits", "semester", "branch"):
        if f in payload:
            setattr(rec, f, payload[f] if f != "code" else str(payload[f]).upper())
    db.commit()
    return {"message": "Updated"}


@router.delete("/subjects/{sid}")
def del_subject(sid: str, user: User = Depends(require_roles("admin", "staff")), db: Session = Depends(get_db)):
    rec = db.query(SubjectCatalog).filter(SubjectCatalog.id == sid).first()
    if rec:
        db.delete(rec)
        db.commit()
    return {"message": "Deleted"}


@router.get("/students")
def students(q: str = "", user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    query = db.query(Student)
    if q:
        query = query.filter(Student.full_name.ilike(f"%{q}%") | Student.usn.ilike(f"%{q}%"))
    rows = query.order_by(Student.usn).limit(500).all()
    return {"items": [{"id": r.id, "usn": r.usn, "full_name": r.full_name, "branch": r.branch, "section": r.section, "semester": r.current_semester, "active": True} for r in rows], "empty": not rows}


@router.post("/students/{sid}/toggle")
def toggle_student(sid: str, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    st = db.query(Student).filter(Student.id == sid).first()
    if not st:
        raise HTTPException(404, "Not found")
    u = db.query(User).filter(User.id == st.user_id).first()
    u.is_active = not u.is_active
    db.commit()
    return {"active": u.is_active}


@router.get("/staff")
def staff_list(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = db.query(Staff).all()
    return {"items": [{"id": r.id, "staff_number": r.staff_number, "full_name": r.full_name, "designation": r.designation} for r in rows], "empty": not rows}


@router.get("/classes")
def classes(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = db.query(ClassRoom).all()
    return {"items": [{"id": r.id, "class_name": r.class_name, "branch": r.branch, "semester": r.semester, "section": r.section, "academic_year": r.academic_year, "staff_id": r.staff_id} for r in rows]}


@router.get("/years")
def years(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = db.query(AcademicYear).all()
    return {"items": [{"id": r.id, "name": r.name, "is_current": r.is_current} for r in rows]}


@router.post("/years")
def add_year(payload: dict, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rec = AcademicYear(name=payload["name"], is_current=bool(payload.get("is_current")))
    if rec.is_current:
        db.query(AcademicYear).update({AcademicYear.is_current: False})
    db.add(rec)
    db.commit()
    return {"id": rec.id}


@router.get("/settings/{group}")
def get_settings(group: str, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    return get_group(db, group)


@router.put("/settings/{group}")
def put_settings(group: str, payload: dict, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    if group not in DEFAULT_SETTINGS and group not in ("system", "ml", "academic", "attendance", "files"):
        raise HTTPException(400, "Unknown settings group")
    set_group(db, group, payload)
    write_audit(db, user.id, "SETTINGS_UPDATED", group, "")
    return {"message": "Settings saved", "value": payload}


@router.get("/audit")
def audit(q: str = "", user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    query = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if q:
        query = query.filter(AuditLog.action.ilike(f"%{q}%") | AuditLog.detail.ilike(f"%{q}%"))
    rows = query.limit(300).all()
    return {"items": [{"id": r.id, "action": r.action, "entity": r.entity, "detail": r.detail, "user_id": r.user_id, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows], "empty": not rows}


@router.post("/ml/train")
def train(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    result = train_models(db)
    write_audit(db, user.id, "ML_TRAIN", "ml", json.dumps({k: result.get(k) for k in ("ok", "message")}))
    return result


@router.get("/ml/models")
def models(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rec = active_model(db)
    from app.models import MLModel

    rows = db.query(MLModel).order_by(MLModel.trained_at.desc()).all()
    return {
        "active": {"version": rec.version, "algorithm": rec.algorithm, "metrics": json.loads(rec.metrics_json), "training_size": rec.training_size, "trained_at": rec.trained_at.isoformat() if rec and rec.trained_at else None} if rec else None,
        "items": [{"id": r.id, "version": r.version, "algorithm": r.algorithm, "training_size": r.training_size, "is_active": r.is_active} for r in rows],
        "empty": not rows,
        "message": "No trained models yet" if not rows else None,
    }


@router.get("/export/students.csv")
def export_students(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = [[s.usn, s.full_name, s.branch, s.section, s.current_semester] for s in db.query(Student).all()]
    data = csv_bytes(["USN", "Name", "Branch", "Section", "Semester"], rows)
    return Response(data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=students.csv"})


@router.get("/export/audit.csv")
def export_audit(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = [[r.created_at, r.action, r.entity, r.detail, r.user_id] for r in db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(2000)]
    data = csv_bytes(["Time", "Action", "Entity", "Detail", "User"], rows)
    return Response(data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit.csv"})


@router.post("/backup")
def backup(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    from app.config import settings
    from pathlib import Path
    import shutil

    src = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
    if not src.exists():
        src = Path("edupredict.db")
    dest = Path("backups")
    dest.mkdir(exist_ok=True)
    out = dest / f"edupredict_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.db"
    if src.exists():
        shutil.copy(src, out)
    write_audit(db, user.id, "BACKUP", "database", str(out))
    return {"message": "Backup created", "path": str(out)}
