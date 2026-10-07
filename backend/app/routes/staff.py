from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.auth import assert_staff_student, get_current_user, require_roles, staff_of, write_audit
from app.database import get_db
from app.models import (
    Application,
    Attendance,
    Certificate,
    ClassRoom,
    CurrentSemesterMark,
    Document,
    Extracurricular,
    FeeReceipt,
    ImportJob,
    InterventionNote,
    LinkedInProfile,
    Marksheet,
    NonDueForm,
    Skill,
    Staff,
    Student,
    StudentSemester,
    StudentSubject,
    SubjectCatalog,
    User,
)
from app.services.academic import combine_internal, compute_total, get_group, grade_for_total, result_for_total
from app.services.excel_import import detect_mapping, error_csv_bytes, mapped_records, read_table, validate_row
from app.services.files import save_upload
from app.services.ml_engine import academic_history, attendance_pct, predict_subject
from app.services.notify import notify
from app.services.reports import csv_bytes, pdf_bytes
import pandas as pd

router = APIRouter(prefix="/staff", tags=["staff"])


def _staff(user: User, db: Session) -> Staff:
    if user.role == "admin":
        rec = db.query(Staff).first()
        if rec:
            return rec
        raise HTTPException(400, "No staff profile available")
    return staff_of(user, db)


def _classes(db: Session, staff: Staff, user: User):
    if user.role == "admin":
        return db.query(ClassRoom).all()
    return db.query(ClassRoom).filter(ClassRoom.staff_id == staff.id).all()


@router.get("/dashboard")
def dashboard(user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    students = db.query(Student).filter(Student.class_id.in_([c.id for c in classes] or ["__none__"])).all()
    atts = []
    sgpases = []
    risk = 0
    backlogs = 0
    for st in students:
        hist = academic_history(db, st)
        sgpases.append(hist["cgpa"])
        atts.append(hist["attendance_percentage"])
        backlogs += int(hist["backlog_count"])
        if hist["attendance_percentage"] and hist["attendance_percentage"] < 75:
            risk += 1
        elif hist["cgpa"] and hist["cgpa"] < 6:
            risk += 1
    avg_sgpa = round(sum(x for x in sgpases if x) / max(len([x for x in sgpases if x]), 1), 2) if sgpases else 0
    avg_att = round(sum(atts) / max(len(atts), 1), 2) if atts else 0
    return {
        "staff": {"full_name": staff.full_name, "staff_number": staff.staff_number, "designation": staff.designation},
        "classes": [{"id": c.id, "class_name": c.class_name, "semester": c.semester, "section": c.section, "branch": c.branch} for c in classes],
        "stats": {
            "total_students": len(students),
            "average_sgpa": avg_sgpa,
            "average_attendance": avg_att,
            "needing_attention": risk,
            "backlogs": backlogs,
            "classes": len(classes),
        },
        "empty": not students,
        "message": "No students yet" if not students else None,
    }


@router.get("/classes")
def list_classes(user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    out = []
    for c in classes:
        n = db.query(Student).filter(Student.class_id == c.id).count()
        out.append(
            {
                "id": c.id,
                "class_name": c.class_name,
                "department_id": c.department_id,
                "branch": c.branch,
                "semester": c.semester,
                "section": c.section,
                "academic_year": c.academic_year,
                "student_count": n,
            }
        )
    return {"items": out, "empty": not out, "message": "No classes yet" if not out else None}


@router.post("/classes")
def create_class(payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    rec = ClassRoom(
        class_name=payload.get("class_name") or "Untitled class",
        branch=payload.get("branch") or "CSE",
        semester=int(payload.get("semester") or 1),
        section=str(payload.get("section") or "A").upper(),
        academic_year=payload.get("academic_year") or "2025-26",
        staff_id=staff.id,
        department_id=payload.get("department_id"),
    )
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Class created"}


@router.put("/classes/{cid}")
def update_class(cid: str, payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    rec = db.query(ClassRoom).filter(ClassRoom.id == cid).first()
    if not rec:
        raise HTTPException(404, "Class not found")
    for f in ("class_name", "branch", "semester", "section", "academic_year", "department_id"):
        if f in payload:
            setattr(rec, f, payload[f])
    db.commit()
    return {"message": "Class updated"}


@router.post("/classes/{cid}/students")
def assign_students(cid: str, payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    rec = db.query(ClassRoom).filter(ClassRoom.id == cid).first()
    if not rec:
        raise HTTPException(404, "Class not found")
    ids = payload.get("student_ids") or []
    usns = [u.strip().upper() for u in (payload.get("usns") or [])]
    students = db.query(Student).filter((Student.id.in_(ids)) | (Student.usn.in_(usns))).all()
    for st in students:
        st.class_id = rec.id
        st.section = rec.section
        st.current_semester = rec.semester
        st.branch = rec.branch
    db.commit()
    return {"message": f"Assigned {len(students)} students"}


@router.get("/students")
def list_students(
    q: str = "",
    class_id: Optional[str] = None,
    semester: Optional[int] = None,
    section: Optional[str] = None,
    performance: Optional[str] = None,
    attendance: Optional[str] = None,
    backlog: Optional[str] = None,
    verification: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    user: User = Depends(require_roles("staff", "admin")),
    db: Session = Depends(get_db),
):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    allowed = [c.id for c in classes]
    query = db.query(Student).filter(Student.class_id.in_(allowed or ["__none__"]))
    if class_id:
        query = query.filter(Student.class_id == class_id)
    if semester:
        query = query.filter(Student.current_semester == semester)
    if section:
        query = query.filter(Student.section == section.upper())
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Student.full_name.ilike(like), Student.usn.ilike(like)))
    if verification:
        query = query.filter(Student.personal_verification == verification)
    total = query.count()
    rows = query.order_by(Student.usn).offset((page - 1) * page_size).limit(page_size).all()
    items = []
    for st in rows:
        hist = academic_history(db, st)
        if performance == "low" and (hist["cgpa"] or 0) >= 6.5:
            continue
        if attendance == "low" and hist["attendance_percentage"] >= 75:
            continue
        if backlog == "yes" and hist["backlog_count"] <= 0:
            continue
        items.append(
            {
                "id": st.id,
                "usn": st.usn,
                "full_name": st.full_name,
                "section": st.section,
                "semester": st.current_semester,
                "cgpa": hist["cgpa"],
                "attendance": hist["attendance_percentage"],
                "backlogs": int(hist["backlog_count"]),
                "verification": st.personal_verification,
            }
        )
    return {"items": items, "total": total, "page": page, "page_size": page_size, "empty": not items, "message": "No students yet" if not items else None}


@router.get("/students/{sid}")
def student_profile(sid: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    st = assert_staff_student(db, user, sid)
    u = db.query(User).filter(User.id == st.user_id).first()
    hist = academic_history(db, st)
    sems = db.query(StudentSemester).filter(StudentSemester.student_id == st.id).all()
    current = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == st.id).all()
    return {
        "overview": {
            "full_name": st.full_name,
            "usn": st.usn,
            "email": u.email if u else None,
            "phone": u.phone if u else None,
            "semester": st.current_semester,
            "branch": st.branch,
            "section": st.section,
            "photo_path": st.photo_path,
            "cgpa": hist["cgpa"],
            "attendance": hist["attendance_percentage"],
            "backlogs": int(hist["backlog_count"]),
        },
        "personal": {"address": st.address, "dob": str(st.date_of_birth) if st.date_of_birth else None, "admission_year": st.admission_year, "verification": st.personal_verification},
        "academics": [
            {"semester": s.semester_number, "sgpa": s.sgpa, "subjects": [{"code": m.subject_code, "name": m.subject_name, "total": m.total_marks, "grade": m.grade} for m in s.subject_marks]}
            for s in sems
        ],
        "current": [{"id": m.id, "code": m.subject_code, "name": m.subject_name, "ia1": m.ia1, "ia2": m.ia2, "credits": m.credits} for m in current],
        "certificates": [{"id": r.id, "name": r.name, "status": r.verification_status} for r in db.query(Certificate).filter(Certificate.student_id == st.id)],
        "skills": [{"id": r.id, "name": r.skill_name, "level": r.level} for r in db.query(Skill).filter(Skill.student_id == st.id)],
        "linkedin": (lambda r: {"url": r.url, "status": r.verification_status} if r else None)(db.query(LinkedInProfile).filter(LinkedInProfile.student_id == st.id).first()),
        "activities": [{"id": r.id, "activity": r.activity, "category": r.category} for r in db.query(Extracurricular).filter(Extracurricular.student_id == st.id)],
        "applications": [{"id": r.id, "type": r.application_type, "subject": r.subject, "status": r.status} for r in db.query(Application).filter(Application.student_id == st.id)],
        "documents": [{"id": r.id, "type": r.document_type, "title": r.title, "status": r.verification_status} for r in db.query(Document).filter(Document.student_id == st.id)],
        "marksheets": [{"id": r.id, "semester": r.semester_number, "status": r.verification_status} for r in db.query(Marksheet).filter(Marksheet.student_id == st.id)],
        "fees": [{"id": r.id, "receipt": r.receipt_number, "status": r.verification_status} for r in db.query(FeeReceipt).filter(FeeReceipt.student_id == st.id)],
        "nondue": [{"id": r.id, "year": r.academic_year, "status": r.verification_status} for r in db.query(NonDueForm).filter(NonDueForm.student_id == st.id)],
    }


@router.get("/marks")
def class_marks(class_id: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    students = db.query(Student).filter(Student.class_id == class_id).order_by(Student.usn).all()
    codes = set()
    matrix = []
    for st in students:
        marks = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == st.id).all()
        cell = {m.subject_code: m for m in marks}
        for m in marks:
            codes.add(m.subject_code)
        hist = academic_history(db, st)
        matrix.append({"student_id": st.id, "usn": st.usn, "full_name": st.full_name, "marks": {k: {"ia1": v.ia1, "ia2": v.ia2, "external": v.external_marks, "total": v.total_marks, "predicted": v.predicted_final} for k, v in cell.items()}, "sgpa": hist["prev_sgpa"]})
    return {"subjects": sorted(codes), "rows": matrix, "empty": not students, "message": "No students yet" if not students else None}


@router.put("/marks/{mark_id}")
def update_mark(mark_id: str, payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    rec = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.id == mark_id).first()
    if not rec:
        raise HTTPException(404, "Mark record not found")
    assert_staff_student(db, user, rec.student_id)
    academic = get_group(db, "academic")
    for f in ("ia1", "ia2", "credits", "external_marks", "subject_name"):
        if f in payload:
            setattr(rec, f, payload[f])
    rec.internal_combined = combine_internal(rec.ia1, rec.ia2, academic)
    rec.total_marks = compute_total(rec.internal_combined, rec.external_marks) if rec.external_marks is not None else None
    db.commit()
    write_audit(db, user.id, "MARKS_MODIFIED", "marks", rec.subject_code)
    st = db.query(Student).filter(Student.id == rec.student_id).first()
    if st:
        notify(db, st.user_id, "Academic", "Marks changed", f"{rec.subject_code} was updated by faculty.")
    return {"message": "Updated"}


@router.post("/excel/upload")
def excel_upload(file: UploadFile = File(...), user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    path, orig, _ = save_upload(file, f"imports/{staff.id}", {".xlsx", ".xls", ".csv"})
    df = read_table(path)
    mapping = detect_mapping(list(df.columns))
    job = ImportJob(staff_id=staff.id, file_path=path, status="MAP", mapping_json=json.dumps(mapping), summary_json=json.dumps({"columns": list(df.columns), "rows": int(len(df))}))
    db.add(job)
    db.commit()
    preview = df.head(8).fillna("").to_dict(orient="records")
    return {"job_id": job.id, "columns": list(df.columns), "mapping": mapping, "preview": preview, "row_count": int(len(df))}


@router.post("/excel/{job_id}/map")
def excel_map(job_id: str, payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    job = db.query(ImportJob).filter(ImportJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Import job not found")
    job.mapping_json = json.dumps(payload.get("mapping") or {})
    job.status = "VALIDATED"
    df = read_table(job.file_path)
    recs = mapped_records(df, json.loads(job.mapping_json))
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    students = db.query(Student).filter(Student.class_id.in_([c.id for c in classes] or ["__none__"])).all()
    usns = {st.usn for st in students}
    academic = get_group(db, "academic")
    errors = []
    valid = []
    for i, rec in enumerate(recs, start=2):
        errs = validate_row(rec, usns, float(academic.get("ia_max", 50)), float(academic.get("external_max", 100)))
        if errs:
            errors.append({"row": i, "usn": rec.get("usn"), "errors": "; ".join(errs), **{k: rec.get(k) for k in rec}})
        else:
            valid.append(rec)
    job.summary_json = json.dumps({"valid": len(valid), "errors": len(errors), "total": len(recs)})
    db.commit()
    return {"valid_count": len(valid), "error_count": len(errors), "errors": errors[:50], "preview_valid": valid[:12]}


@router.post("/excel/{job_id}/import")
def excel_import(job_id: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    job = db.query(ImportJob).filter(ImportJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Import job not found")
    df = read_table(job.file_path)
    recs = mapped_records(df, json.loads(job.mapping_json))
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    students = {st.usn: st for st in db.query(Student).filter(Student.class_id.in_([c.id for c in classes] or ["__none__"])).all()}
    academic = get_group(db, "academic")
    imported = 0
    errors = []
    for i, rec in enumerate(recs, start=2):
        errs = validate_row(rec, set(students), float(academic.get("ia_max", 50)), float(academic.get("external_max", 100)))
        if errs:
            errors.append({"row": i, "usn": rec.get("usn"), "errors": "; ".join(errs)})
            continue
        st = students[str(rec["usn"]).upper()]
        sem = int(rec.get("semester") or st.current_semester)
        code = str(rec["subject_code"]).strip().upper()
        ia1 = float(rec["ia1"] or rec.get("internal") or 0)
        ia2 = float(rec["ia2"] or rec.get("internal") or 0)
        combined = combine_internal(ia1, ia2, academic)
        ext = rec.get("external")
        existing = (
            db.query(CurrentSemesterMark)
            .filter(CurrentSemesterMark.student_id == st.id, CurrentSemesterMark.semester_number == sem, CurrentSemesterMark.subject_code == code)
            .first()
        )
        if existing:
            existing.ia1, existing.ia2, existing.internal_combined = ia1, ia2, combined
            existing.subject_name = str(rec.get("subject_name"))
            existing.credits = float(rec.get("credits") or existing.credits)
            if ext not in (None, ""):
                existing.external_marks = float(ext)
                existing.total_marks = compute_total(combined, existing.external_marks)
        else:
            db.add(
                CurrentSemesterMark(
                    student_id=st.id,
                    semester_number=sem,
                    subject_code=code,
                    subject_name=str(rec.get("subject_name")),
                    credits=float(rec.get("credits") or 3),
                    ia1=ia1,
                    ia2=ia2,
                    internal_combined=combined,
                    external_marks=float(ext) if ext not in (None, "") else None,
                    total_marks=compute_total(combined, float(ext)) if ext not in (None, "") else None,
                )
            )
        imported += 1
        notify(db, st.user_id, "Academic", "Marks uploaded", f"{code} marks were imported by faculty.")
    from app.config import settings

    err_path = None
    if errors:
        p = settings.upload_path / f"imports/{staff.id}" / f"errors_{job.id}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(error_csv_bytes(errors))
        err_path = str(p)
    job.status = "DONE"
    job.error_csv_path = err_path
    job.summary_json = json.dumps({"imported": imported, "errors": len(errors), "total": len(recs)})
    db.commit()
    write_audit(db, user.id, "MARKS_UPLOADED", "excel", job.id)
    return {"imported": imported, "errors": len(errors), "total": len(recs), "error_download": bool(err_path), "message": f"{len(recs)} rows · {imported} imported · {len(errors)} errors"}


@router.get("/excel/{job_id}/errors.csv")
def excel_errors(job_id: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    job = db.query(ImportJob).filter(ImportJob.id == job_id).first()
    if not job or not job.error_csv_path:
        raise HTTPException(404, "No error file")
    data = open(job.error_csv_path, "rb").read()
    return Response(data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=import_errors.csv"})


@router.get("/attendance")
def get_attendance(class_id: str, attendance_date: str, subject_name: str, hour_number: int, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    students = db.query(Student).filter(Student.class_id == class_id).order_by(Student.usn).all()
    existing = {
        r.student_id: r
        for r in db.query(Attendance).filter(
            Attendance.class_id == class_id,
            Attendance.attendance_date == attendance_date,
            Attendance.subject_name == subject_name,
            Attendance.hour_number == hour_number,
        )
    }
    return {
        "students": [
            {"student_id": st.id, "usn": st.usn, "full_name": st.full_name, "status": existing[st.id].status if st.id in existing else "Present"}
            for st in students
        ],
        "empty": not students,
        "message": "No students yet" if not students else None,
    }


@router.post("/attendance")
def save_attendance(payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    class_id = payload["class_id"]
    d = payload["attendance_date"]
    subject = payload["subject_name"]
    hour = int(payload["hour_number"])
    for row in payload.get("records") or []:
        rec = (
            db.query(Attendance)
            .filter(
                Attendance.student_id == row["student_id"],
                Attendance.attendance_date == d,
                Attendance.subject_name == subject,
                Attendance.hour_number == hour,
            )
            .first()
        )
        if rec:
            rec.status = row["status"]
        else:
            db.add(Attendance(class_id=class_id, student_id=row["student_id"], attendance_date=d, subject_name=subject, hour_number=hour, status=row["status"]))
        if row["status"] == "Absent":
            st = db.query(Student).filter(Student.id == row["student_id"]).first()
            if st:
                pct = attendance_pct(db, st.id)
                if pct < 75:
                    notify(db, st.user_id, "Attendance", "Attendance warning", f"Your attendance is {pct}%.")
    db.commit()
    write_audit(db, user.id, "ATTENDANCE_SAVED", "attendance", f"{class_id} {d}")
    return {"message": "Attendance saved"}


@router.get("/risk")
def risk(class_id: Optional[str] = None, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    allowed = [c.id for c in classes if (not class_id or c.id == class_id)]
    students = db.query(Student).filter(Student.class_id.in_(allowed or ["__none__"])).all()
    items = []
    for st in students:
        hist = academic_history(db, st)
        factors = []
        if hist["attendance_percentage"] and hist["attendance_percentage"] < 75:
            factors.append(f"Attendance {hist['attendance_percentage']}% is below 75%")
        if hist["cgpa"] and hist["cgpa"] < 6:
            factors.append(f"CGPA {hist['cgpa']} is below 6.0")
        if hist["backlog_count"]:
            factors.append(f"{int(hist['backlog_count'])} backlog records")
        marks = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == st.id).all()
        weak = []
        for m in marks:
            if m.ia1 < 20 or m.ia2 < 20:
                weak.append(m.subject_name)
                factors.append(f"Low IA in {m.subject_code}")
        if not factors:
            continue
        notes = db.query(InterventionNote).filter(InterventionNote.student_id == st.id).all()
        items.append(
            {
                "student_id": st.id,
                "usn": st.usn,
                "full_name": st.full_name,
                "factors": factors,
                "weak_subjects": weak,
                "attendance": hist["attendance_percentage"],
                "cgpa": hist["cgpa"],
                "predicted_sgpa": None,
                "intervention": notes[-1].note if notes else None,
            }
        )
    return {"items": items, "empty": not items, "message": "No students currently flagged" if not items else None}


@router.post("/risk/{sid}/note")
def add_note(sid: str, payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    assert_staff_student(db, user, sid)
    db.add(InterventionNote(student_id=sid, staff_id=staff.id, note=payload.get("note") or ""))
    db.commit()
    return {"message": "Intervention note saved"}


@router.get("/applications")
def apps(user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    students = db.query(Student).filter(Student.class_id.in_([c.id for c in classes] or ["__none__"])).all()
    ids = [s.id for s in students]
    rows = db.query(Application).filter(Application.student_id.in_(ids or ["__none__"])).order_by(Application.created_at.desc()).all()
    smap = {s.id: s for s in students}
    return {
        "items": [
            {"id": r.id, "student": smap[r.student_id].full_name, "usn": smap[r.student_id].usn, "type": r.application_type, "subject": r.subject, "message": r.message, "status": r.status, "staff_remark": r.staff_remark}
            for r in rows
        ],
        "empty": not rows,
        "message": "No applications" if not rows else None,
    }


@router.post("/applications/{aid}")
def process_app(aid: str, payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    rec = db.query(Application).filter(Application.id == aid).first()
    if not rec:
        raise HTTPException(404, "Application not found")
    assert_staff_student(db, user, rec.student_id)
    rec.status = payload.get("status") or rec.status
    rec.staff_remark = payload.get("remark")
    rec.processed_by = user.id
    db.commit()
    st = db.query(Student).filter(Student.id == rec.student_id).first()
    notify(db, st.user_id, "Applications", f"Application {rec.status.lower()}", rec.staff_remark or rec.subject)
    write_audit(db, user.id, "APPLICATION_PROCESSED", "application", aid)
    return {"message": "Updated"}


VERIFY_MODELS = {
    "marksheet": Marksheet,
    "certificate": Certificate,
    "document": Document,
    "fee": FeeReceipt,
    "nondue": NonDueForm,
    "linkedin": LinkedInProfile,
    "skill": Skill,
    "activity": Extracurricular,
}


@router.get("/verification")
def verification_queue(user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    students = {s.id: s for s in db.query(Student).filter(Student.class_id.in_([c.id for c in classes] or ["__none__"])).all()}
    items = []
    for key, model in VERIFY_MODELS.items():
        rows = db.query(model).filter(model.student_id.in_(list(students) or ["__none__"])).all()
        for r in rows:
            st = students.get(r.student_id)
            status = getattr(r, "verification_status", None)
            items.append({"entity_type": key, "id": r.id, "status": status, "student": st.full_name if st else "", "usn": st.usn if st else "", "label": getattr(r, "name", None) or getattr(r, "title", None) or getattr(r, "url", None) or key})
    return {"items": items, "empty": not items, "message": "Nothing pending verification" if not items else None}


@router.post("/verification")
def verify_item(payload: dict, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    model = VERIFY_MODELS.get(payload.get("entity_type"))
    if not model:
        raise HTTPException(400, "Unknown entity type")
    rec = db.query(model).filter(model.id == payload.get("id")).first()
    if not rec:
        raise HTTPException(404, "Record not found")
    assert_staff_student(db, user, rec.student_id)
    status = payload.get("status") or "VERIFIED"
    rec.verification_status = status
    if hasattr(rec, "staff_remark"):
        rec.staff_remark = payload.get("remark")
    db.commit()
    st = db.query(Student).filter(Student.id == rec.student_id).first()
    notify(db, st.user_id, "Documents", "Document verification", f"Status: {status}. {payload.get('remark') or ''}")
    write_audit(db, user.id, "VERIFICATION", payload.get("entity_type"), rec.id)
    return {"message": "Verification updated"}


@router.get("/documents")
def search_docs(
    q: str = "",
    document_type: Optional[str] = None,
    status: Optional[str] = None,
    user: User = Depends(require_roles("staff", "admin")),
    db: Session = Depends(get_db),
):
    staff = _staff(user, db)
    classes = _classes(db, staff, user)
    students = {s.id: s for s in db.query(Student).filter(Student.class_id.in_([c.id for c in classes] or ["__none__"])).all()}
    query = db.query(Document).filter(Document.student_id.in_(list(students) or ["__none__"]))
    if document_type:
        query = query.filter(Document.document_type == document_type)
    if status:
        query = query.filter(Document.verification_status == status)
    rows = query.order_by(Document.created_at.desc()).limit(200).all()
    items = []
    for r in rows:
        st = students.get(r.student_id)
        if q and q.lower() not in (st.full_name + st.usn + r.title).lower():
            continue
        items.append({"id": r.id, "title": r.title, "type": r.document_type, "status": r.verification_status, "usn": st.usn if st else "", "student": st.full_name if st else ""})
    return {"items": items, "empty": not items, "message": "No documents match" if not items else None}


@router.get("/reports/class.csv")
def report_csv(class_id: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    students = db.query(Student).filter(Student.class_id == class_id).all()
    rows = []
    for st in students:
        hist = academic_history(db, st)
        rows.append([st.usn, st.full_name, st.section, st.current_semester, hist["cgpa"], hist["attendance_percentage"], int(hist["backlog_count"])])
    data = csv_bytes(["USN", "Name", "Section", "Semester", "CGPA", "Attendance", "Backlogs"], rows)
    return Response(data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=class_report.csv"})


@router.get("/reports/class.pdf")
def report_pdf(class_id: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    classroom = db.query(ClassRoom).filter(ClassRoom.id == class_id).first()
    students = db.query(Student).filter(Student.class_id == class_id).all()
    table = [["USN", "Name", "CGPA", "Attendance"]]
    for st in students:
        hist = academic_history(db, st)
        table.append([st.usn, st.full_name, hist["cgpa"], hist["attendance_percentage"]])
    data = pdf_bytes(f"Class Report - {classroom.class_name if classroom else class_id}", ["Faculty class analytics export"], [table])
    return Response(data, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=class_report.pdf"})


@router.post("/predictions/{sid}")
def pred_student(sid: str, user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    st = assert_staff_student(db, user, sid)
    marks = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == st.id).all()
    if not marks:
        return {"empty": True, "message": "Not enough data for prediction"}
    return {"subjects": [predict_subject(db, st, m) for m in marks], "disclaimer": "AI predictions are estimates and should not be treated as guaranteed examination results."}
