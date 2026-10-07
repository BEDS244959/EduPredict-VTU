from __future__ import annotations

import json
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.auth import get_current_user, student_of, write_audit
from app.database import get_db
from app.models import (
    Application,
    Attendance,
    Backlog,
    Certificate,
    CurrentSemesterMark,
    Document,
    Extracurricular,
    FeeReceipt,
    LinkedInProfile,
    Marksheet,
    NonDueForm,
    Notification,
    Prediction,
    PredictionResult,
    Skill,
    Student,
    StudentSemester,
    StudentSubject,
    StudyTask,
    User,
)
from app.services.academic import (
    combine_internal,
    compute_total,
    grade_for_total,
    result_for_total,
    sgpa_from_rows,
    cgpa_from_semesters,
    get_group,
)
from app.services.files import save_upload, safe_resolve
from app.services.ml_engine import academic_history, attendance_pct, predict_subject, train_models
from app.services.notify import notify
from app.services.ocr_service import extract_marksheet
from app.services.reports import csv_bytes, pdf_bytes

router = APIRouter(prefix="/students", tags=["students"])

CERT_CATS = ["Technical", "Sports", "Cultural", "Workshop", "Internship", "Competition", "Hackathon", "Other"]
ACTIVITY_CATS = ["Sports", "Cultural", "NSS", "NCC", "Clubs", "Volunteering", "Hackathons", "Competitions", "Other"]


def _student(user: User, db: Session) -> Student:
    return student_of(user, db)


@router.get("/dashboard")
def dashboard(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    hist = academic_history(db, s)
    att = attendance_pct(db, s.id)
    sems = db.query(StudentSemester).filter(StudentSemester.student_id == s.id).order_by(StudentSemester.semester_number).all()
    sgpa_trend = [{"semester": x.semester_number, "sgpa": x.sgpa} for x in sems]
    current = (
        db.query(StudentSemester)
        .filter(StudentSemester.student_id == s.id, StudentSemester.semester_number == s.current_semester)
        .first()
    )
    subjects = []
    if current:
        subjects = [
            {
                "code": m.subject_code,
                "name": m.subject_name,
                "internal": m.internal_marks,
                "external": m.external_marks,
                "total": m.total_marks,
                "grade": m.grade,
            }
            for m in current.subject_marks
        ]
    ia_trend = [
        {"subject": m.subject_code, "ia1": m.ia1, "ia2": m.ia2}
        for m in db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == s.id).all()
    ]
    last_pred = (
        db.query(Prediction)
        .filter(Prediction.student_id == s.id)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    return {
        "profile": {
            "full_name": s.full_name,
            "usn": s.usn,
            "photo_path": s.photo_path,
            "current_semester": s.current_semester,
            "branch": s.branch,
            "section": s.section,
            "class_id": s.class_id,
        },
        "summary": {
            "current_sgpa": hist["prev_sgpa"] or None,
            "cgpa": hist["cgpa"] or None,
            "previous_sgpa": hist["prev_sgpa"] or None,
            "attendance_percentage": att,
            "backlogs": int(hist["backlog_count"]),
            "predicted_sgpa": last_pred.predicted_sgpa if last_pred else None,
            "status": "On Track" if (hist["cgpa"] or 0) >= 6.5 else "Needs Focus" if hist["cgpa"] else "No data yet",
        },
        "counts": {
            "skills": db.query(Skill).filter(Skill.student_id == s.id).count(),
            "documents": db.query(Document).filter(Document.student_id == s.id).count(),
            "certificates": db.query(Certificate).filter(Certificate.student_id == s.id).count(),
        },
        "charts": {"sgpa_trend": sgpa_trend, "subjects": subjects, "ia_trend": ia_trend},
        "empty": not sems and not ia_trend,
    }


@router.get("/profile")
def get_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    return {
        "full_name": s.full_name,
        "usn": s.usn,
        "email": user.email,
        "phone": user.phone,
        "date_of_birth": str(s.date_of_birth) if s.date_of_birth else None,
        "address": s.address,
        "admission_year": s.admission_year,
        "current_semester": s.current_semester,
        "branch": s.branch,
        "section": s.section,
        "photo_path": s.photo_path,
        "personal_verification": s.personal_verification,
    }


@router.put("/profile")
def update_profile(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    for field in ("full_name", "address", "branch", "section"):
        if field in payload and payload[field] is not None:
            setattr(s, field, str(payload[field]).strip())
    if "phone" in payload:
        user.phone = str(payload["phone"]).strip()
    if "current_semester" in payload:
        sem = int(payload["current_semester"])
        if sem < 1 or sem > 8:
            raise HTTPException(400, "Semester must be 1-8")
        s.current_semester = sem
    if "admission_year" in payload and payload["admission_year"]:
        s.admission_year = int(payload["admission_year"])
    if payload.get("date_of_birth"):
        s.date_of_birth = date.fromisoformat(payload["date_of_birth"])
    db.commit()
    write_audit(db, user.id, "PROFILE_UPDATED", "student", s.usn)
    return {"message": "Profile saved"}


@router.post("/profile/photo")
def upload_photo(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    path, orig, _ = save_upload(file, f"photos/{s.id}", {".jpg", ".jpeg", ".png"})
    s.photo_path = path
    db.commit()
    return {"message": "Photo updated", "photo_path": path, "original": orig}


@router.delete("/profile/photo")
def delete_photo(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    s.photo_path = None
    db.commit()
    return {"message": "Photo removed"}


@router.get("/academics")
def academics(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    academic = get_group(db, "academic")
    sems = db.query(StudentSemester).filter(StudentSemester.student_id == s.id).order_by(StudentSemester.semester_number).all()
    out = []
    packed = []
    for sem in sems:
        rows = [
            {
                "id": m.id,
                "subject_code": m.subject_code,
                "subject_name": m.subject_name,
                "credits": m.credits,
                "internal_marks": m.internal_marks,
                "external_marks": m.external_marks,
                "total_marks": m.total_marks,
                "grade": m.grade,
                "grade_point": m.grade_point,
                "result": m.result,
                "status": m.status,
            }
            for m in sem.subject_marks
        ]
        calc = sgpa_from_rows(rows)
        packed.append({"sgpa": calc["sgpa"], "total_credits": calc["total_credits"]})
        out.append({"id": sem.id, "semester_number": sem.semester_number, "sgpa": calc["sgpa"], "is_current": sem.is_current, "subjects": rows, "calculation": calc})
    cg = cgpa_from_semesters(packed)
    return {"semesters": out, "cgpa": cg, "rules": academic, "empty": not out}


@router.post("/academics/subjects")
def add_subject(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    academic = get_group(db, "academic")
    sem_n = int(payload.get("semester_number") or s.current_semester)
    if sem_n < 1 or sem_n > 8:
        raise HTTPException(400, "Semester must be 1-8")
    sem = db.query(StudentSemester).filter(StudentSemester.student_id == s.id, StudentSemester.semester_number == sem_n).first()
    if not sem:
        sem = StudentSemester(student_id=s.id, semester_number=sem_n, semester_name=f"Semester {sem_n}")
        db.add(sem)
        db.flush()
    internal = payload.get("internal_marks")
    external = payload.get("external_marks")
    total = compute_total(internal, external)
    letter, gp = grade_for_total(total, academic)
    rec = StudentSubject(
        semester_id=sem.id,
        subject_code=str(payload.get("subject_code") or "").strip().upper(),
        subject_name=str(payload.get("subject_name") or "").strip(),
        credits=float(payload.get("credits") or 3),
        internal_marks=internal,
        external_marks=external,
        total_marks=total,
        grade=letter,
        grade_point=gp,
        result=result_for_total(total, academic),
        status=payload.get("status") or "RECORDED",
    )
    if not rec.subject_code or not rec.subject_name:
        raise HTTPException(400, "Subject code and name are required")
    db.add(rec)
    db.flush()
    rows = [{"credits": m.credits, "grade_point": m.grade_point, "subject_code": m.subject_code, "subject_name": m.subject_name, "grade": m.grade} for m in sem.subject_marks]
    sem.sgpa = sgpa_from_rows(rows)["sgpa"]
    db.commit()
    return {"message": "Subject added", "id": rec.id, "sgpa": sem.sgpa}


@router.put("/academics/subjects/{sid}")
def edit_subject(sid: str, payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    academic = get_group(db, "academic")
    rec = db.query(StudentSubject).filter(StudentSubject.id == sid).first()
    if not rec or rec.semester.student_id != s.id:
        raise HTTPException(404, "Subject not found")
    for f in ("subject_code", "subject_name", "credits", "internal_marks", "external_marks", "status"):
        if f in payload:
            setattr(rec, f, payload[f] if f != "subject_code" else str(payload[f]).upper())
    rec.total_marks = compute_total(rec.internal_marks, rec.external_marks)
    rec.grade, rec.grade_point = grade_for_total(rec.total_marks, academic)
    rec.result = result_for_total(rec.total_marks, academic)
    rows = [{"credits": m.credits, "grade_point": m.grade_point, "subject_code": m.subject_code, "subject_name": m.subject_name, "grade": m.grade} for m in rec.semester.subject_marks]
    rec.semester.sgpa = sgpa_from_rows(rows)["sgpa"]
    db.commit()
    write_audit(db, user.id, "MARKS_MODIFIED", "subject", rec.subject_code)
    return {"message": "Updated", "sgpa": rec.semester.sgpa}


@router.delete("/academics/subjects/{sid}")
def delete_subject(sid: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = db.query(StudentSubject).filter(StudentSubject.id == sid).first()
    if not rec or rec.semester.student_id != s.id:
        raise HTTPException(404, "Subject not found")
    sem = rec.semester
    db.delete(rec)
    db.flush()
    rows = [{"credits": m.credits, "grade_point": m.grade_point, "subject_code": m.subject_code, "subject_name": m.subject_name, "grade": m.grade} for m in sem.subject_marks]
    sem.sgpa = sgpa_from_rows(rows)["sgpa"]
    db.commit()
    return {"message": "Deleted"}


@router.post("/marksheets/upload")
def upload_marksheet(
    semester_number: int = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _student(user, db)
    if semester_number < 1 or semester_number > 8:
        raise HTTPException(400, "Semester must be 1-8")
    path, orig, _ = save_upload(file, f"marksheets/{s.id}", {".pdf", ".jpg", ".jpeg", ".png"})
    extracted = extract_marksheet(path)
    rec = Marksheet(
        student_id=s.id,
        semester_number=semester_number,
        file_path=path,
        original_name=orig,
        extraction_json=json.dumps(extracted),
        confirmed=False,
    )
    db.add(rec)
    db.commit()
    write_audit(db, user.id, "DOCUMENT_UPLOADED", "marksheet", orig)
    return {"id": rec.id, "extraction": extracted, "message": "Review extracted marks before saving."}


@router.get("/marksheets")
def list_marksheets(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = db.query(Marksheet).filter(Marksheet.student_id == s.id).order_by(Marksheet.semester_number).all()
    if not rows:
        return {"items": [], "empty": True, "message": "No marksheets uploaded"}
    return {
        "items": [
            {
                "id": r.id,
                "semester_number": r.semester_number,
                "original_name": r.original_name,
                "confirmed": r.confirmed,
                "verification_status": r.verification_status,
                "staff_remark": r.staff_remark,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


@router.post("/marksheets/{mid}/confirm")
def confirm_marksheet(mid: str, payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = db.query(Marksheet).filter(Marksheet.id == mid, Marksheet.student_id == s.id).first()
    if not rec:
        raise HTTPException(404, "Marksheet not found")
    academic = get_group(db, "academic")
    subjects = payload.get("subjects") or []
    sem = db.query(StudentSemester).filter(StudentSemester.student_id == s.id, StudentSemester.semester_number == rec.semester_number).first()
    if not sem:
        sem = StudentSemester(student_id=s.id, semester_number=rec.semester_number, semester_name=f"Semester {rec.semester_number}", marksheet_path=rec.file_path)
        db.add(sem)
        db.flush()
    for row in subjects:
        internal = row.get("internal")
        external = row.get("external")
        total = compute_total(internal, external)
        letter, gp = grade_for_total(total, academic)
        db.add(
            StudentSubject(
                semester_id=sem.id,
                subject_code=str(row.get("subject_code") or "").strip().upper(),
                subject_name=str(row.get("subject_name") or "").strip(),
                credits=float(row.get("credits") or 3),
                internal_marks=internal,
                external_marks=external,
                total_marks=total,
                grade=letter,
                grade_point=gp,
                result=result_for_total(total, academic),
                ocr_confidence="confirmed",
            )
        )
    db.flush()
    rows = [{"credits": m.credits, "grade_point": m.grade_point, "subject_code": m.subject_code, "subject_name": m.subject_name, "grade": m.grade} for m in sem.subject_marks]
    sem.sgpa = sgpa_from_rows(rows)["sgpa"]
    rec.confirmed = True
    rec.extraction_json = json.dumps({"subjects": subjects})
    db.commit()
    notify(db, user.id, "Academic", "Marksheet confirmed", f"Semester {rec.semester_number} marks were saved after review.")
    write_audit(db, user.id, "MARKS_UPLOADED", "marksheet", str(rec.semester_number))
    return {"message": "Marks saved", "sgpa": sem.sgpa}


@router.get("/current-semester")
def current_semester(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == s.id, CurrentSemesterMark.semester_number == s.current_semester).all()
    academic = get_group(db, "academic")
    return {
        "semester": s.current_semester,
        "rules": academic,
        "subjects": [
            {
                "id": r.id,
                "subject_code": r.subject_code,
                "subject_name": r.subject_name,
                "credits": r.credits,
                "ia1": r.ia1,
                "ia2": r.ia2,
                "internal_combined": r.internal_combined,
                "external_marks": r.external_marks,
                "total_marks": r.total_marks,
            }
            for r in rows
        ],
        "empty": not rows,
        "message": "No current semester subjects yet" if not rows else None,
    }


@router.post("/current-semester")
def upsert_current(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    academic = get_group(db, "academic")
    ia_max = float(academic.get("ia_max", 50))
    code = str(payload.get("subject_code") or "").strip().upper()
    name = str(payload.get("subject_name") or "").strip()
    if not code or not name:
        raise HTTPException(400, "Subject code and name are required")
    ia1 = float(payload.get("ia1") or 0)
    ia2 = float(payload.get("ia2") or 0)
    if ia1 < 0 or ia2 < 0 or ia1 > ia_max or ia2 > ia_max:
        raise HTTPException(400, f"IA marks must be between 0 and {ia_max}")
    rec = (
        db.query(CurrentSemesterMark)
        .filter(CurrentSemesterMark.student_id == s.id, CurrentSemesterMark.semester_number == s.current_semester, CurrentSemesterMark.subject_code == code)
        .first()
    )
    combined = combine_internal(ia1, ia2, academic)
    ext = payload.get("external_marks")
    total = compute_total(combined, ext) if ext is not None else None
    if rec:
        rec.subject_name = name
        rec.credits = float(payload.get("credits") or rec.credits)
        rec.ia1, rec.ia2, rec.internal_combined, rec.external_marks, rec.total_marks = ia1, ia2, combined, ext, total
    else:
        rec = CurrentSemesterMark(
            student_id=s.id,
            semester_number=s.current_semester,
            subject_code=code,
            subject_name=name,
            credits=float(payload.get("credits") or 3),
            ia1=ia1,
            ia2=ia2,
            internal_combined=combined,
            external_marks=ext,
            total_marks=total,
        )
        db.add(rec)
    db.commit()
    write_audit(db, user.id, "MARKS_MODIFIED", "current_semester", code)
    notify(db, user.id, "Academic", "Marks updated", f"{code} internals were saved.")
    return {"message": "Saved", "id": rec.id, "internal_combined": combined}


@router.delete("/current-semester/{sid}")
def delete_current(sid: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.id == sid, CurrentSemesterMark.student_id == s.id).first()
    if not rec:
        raise HTTPException(404, "Record not found")
    db.delete(rec)
    db.commit()
    return {"message": "Deleted"}


@router.post("/predictions/run")
def run_predictions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    marks = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == s.id, CurrentSemesterMark.semester_number == s.current_semester).all()
    if not marks:
        return {
            "empty": True,
            "message": "Not enough historical data for reliable ML prediction.",
            "hint": "Add current semester IA1/IA2 subjects first.",
            "disclaimer": "AI predictions are estimates and should not be treated as guaranteed examination results.",
        }
    results = [predict_subject(db, s, m) for m in marks]
    methods = {r["method"] for r in results}
    method = "ML" if methods == {"ML"} else "MIXED" if "ML" in methods else "RULE"
    credit_pts = 0.0
    credits = 0.0
    for r, m in zip(results, marks):
        credit_pts += float(m.credits) * float(r["expected_grade_point"])
        credits += float(m.credits)
        m.predicted_final = r["predicted_final_marks"]
    pred_sgpa = round(credit_pts / credits, 2) if credits else None
    pred = Prediction(
        student_id=s.id,
        semester_number=s.current_semester,
        method=method,
        predicted_sgpa=pred_sgpa,
        summary_json=json.dumps({"results": results}),
        model_version=(results[0].get("model") or {}).get("version") if results else None,
    )
    db.add(pred)
    db.flush()
    for r in results:
        db.add(
            PredictionResult(
                prediction_id=pred.id,
                student_id=s.id,
                subject_code=r["subject_code"],
                subject_name=r["subject_name"],
                predicted_marks=r["predicted_final_marks"],
                expected_grade=r["expected_grade"],
                risk_level=r["risk_level"],
                method=r["method"],
                reasons_json=json.dumps(r["why_flagged"]),
                recommendations_json=json.dumps(r["recommendations"]),
            )
        )
    db.commit()
    notify(db, user.id, "Academic", "Prediction updated", "New subject-wise estimates are available.")
    weak = [r for r in results if r["risk_level"] in ("Needs Attention", "High Risk")]
    return {
        "prediction_id": pred.id,
        "predicted_sgpa": pred_sgpa,
        "method": method,
        "subjects": results,
        "weak_subjects": weak,
        "disclaimer": "AI predictions are estimates and should not be treated as guaranteed examination results.",
        "empty": False,
    }


@router.get("/predictions")
def get_predictions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    last = db.query(Prediction).filter(Prediction.student_id == s.id).order_by(Prediction.created_at.desc()).first()
    if not last:
        return {"empty": True, "message": "No predictions available"}
    return json.loads(last.summary_json) | {"predicted_sgpa": last.predicted_sgpa, "method": last.method, "created_at": last.created_at.isoformat() if last.created_at else None}


@router.get("/study-plan")
def list_plan(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    tasks = db.query(StudyTask).filter(StudyTask.student_id == s.id).order_by(StudyTask.completed, StudyTask.deadline).all()
    return {
        "tasks": [
            {"id": t.id, "subject": t.subject, "title": t.title, "priority": t.priority, "deadline": t.deadline, "weekday": t.weekday, "hours": t.hours, "completed": t.completed}
            for t in tasks
        ],
        "empty": not tasks,
        "message": "No study plan yet. Generate one from predictions." if not tasks else None,
    }


@router.post("/study-plan/generate")
def generate_plan(payload: dict | None = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    hours = float((payload or {}).get("available_hours") or 12)
    last = db.query(Prediction).filter(Prediction.student_id == s.id).order_by(Prediction.created_at.desc()).first()
    weak = []
    if last:
        data = json.loads(last.summary_json)
        weak = [r for r in data.get("results", data.get("subjects", [])) if r.get("risk_level") in ("Needs Attention", "High Risk")]
    if not weak:
        marks = db.query(CurrentSemesterMark).filter(CurrentSemesterMark.student_id == s.id).all()
        weak = [{"subject_name": m.subject_name, "subject_code": m.subject_code} for m in marks]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    created = []
    per = max(1.0, hours / max(len(weak), 1) / 2)
    for i, w in enumerate(weak or [{"subject_name": "General revision", "subject_code": "GEN"}]):
        t = StudyTask(
            student_id=s.id,
            subject=w.get("subject_name") or w.get("subject_code"),
            title=f"Revise core units + 10 practice questions ({w.get('subject_code')})",
            priority="High" if i == 0 else "Medium",
            weekday=days[i % len(days)],
            hours=round(per, 1),
        )
        db.add(t)
        created.append(t)
    db.commit()
    return {"message": "Weekly plan generated. You can edit tasks.", "count": len(created)}


@router.post("/study-plan")
def add_task(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    t = StudyTask(
        student_id=s.id,
        subject=payload.get("subject") or "General",
        title=payload.get("title") or "Study task",
        priority=payload.get("priority") or "Medium",
        deadline=payload.get("deadline"),
        weekday=payload.get("weekday"),
        hours=float(payload.get("hours") or 1),
    )
    db.add(t)
    db.commit()
    return {"id": t.id, "message": "Task added"}


@router.put("/study-plan/{tid}")
def update_task(tid: str, payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    t = db.query(StudyTask).filter(StudyTask.id == tid, StudyTask.student_id == s.id).first()
    if not t:
        raise HTTPException(404, "Task not found")
    if "completed" in payload:
        t.completed = bool(payload["completed"])
    for f in ("title", "subject", "priority", "deadline", "weekday", "hours"):
        if f in payload:
            setattr(t, f, payload[f])
    db.commit()
    return {"message": "Updated"}


@router.delete("/study-plan/{tid}")
def delete_task(tid: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    t = db.query(StudyTask).filter(StudyTask.id == tid, StudyTask.student_id == s.id).first()
    if not t:
        raise HTTPException(404, "Task not found")
    db.delete(t)
    db.commit()
    return {"message": "Deleted"}


@router.get("/attendance")
def my_attendance(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = db.query(Attendance).filter(Attendance.student_id == s.id).all()
    if not rows:
        return {"empty": True, "message": "No attendance records", "overall": 0, "by_subject": [], "monthly": []}
    overall = round(100.0 * sum(1 for r in rows if r.status == "Present") / len(rows), 2)
    by_sub: dict[str, list] = {}
    monthly: dict[str, list] = {}
    for r in rows:
        by_sub.setdefault(r.subject_name, []).append(r)
        monthly.setdefault(r.attendance_date[:7], []).append(r)
    warning = overall < float(get_group(db, "attendance").get("min_percent", 75))
    return {
        "overall": overall,
        "warning": warning,
        "by_subject": [
            {"subject": k, "percent": round(100 * sum(1 for x in v if x.status == "Present") / len(v), 2), "hours": len(v)}
            for k, v in by_sub.items()
        ],
        "monthly": [
            {"month": k, "percent": round(100 * sum(1 for x in v if x.status == "Present") / len(v), 2)}
            for k, v in sorted(monthly.items())
        ],
        "empty": False,
    }


@router.get("/backlogs")
def backlogs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = db.query(Backlog).filter(Backlog.student_id == s.id).all()
    failed = (
        db.query(StudentSubject)
        .join(StudentSemester)
        .filter(StudentSemester.student_id == s.id, StudentSubject.result == "F")
        .all()
    )
    return {
        "items": [
            {
                "id": r.id,
                "subject_code": r.subject_code,
                "subject_name": r.subject_name,
                "semester_number": r.semester_number,
                "original_marks": r.original_marks,
                "rewritten_marks": r.rewritten_marks,
                "result": r.result,
                "attempt": r.attempt,
                "status": r.status,
            }
            for r in rows
        ],
        "from_marksheets": [{"code": f.subject_code, "name": f.subject_name, "total": f.total_marks} for f in failed],
        "empty": not rows and not failed,
        "message": "No backlog records" if not rows and not failed else None,
    }


@router.post("/backlogs")
def add_backlog(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = Backlog(
        student_id=s.id,
        semester_number=int(payload.get("semester_number") or 1),
        subject_code=str(payload.get("subject_code") or "").upper(),
        subject_name=payload.get("subject_name") or "",
        credits=float(payload.get("credits") or 3),
        original_marks=payload.get("original_marks"),
        rewritten_marks=payload.get("rewritten_marks"),
        result=payload.get("result") or "F",
        attempt=int(payload.get("attempt") or 1),
        status=payload.get("status") or "ACTIVE",
    )
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Backlog recorded"}


def _list_simple(db, model, student_id):
    rows = db.query(model).filter(model.student_id == student_id).order_by(model.created_at.desc()).all()
    return rows


@router.get("/certificates")
def certificates(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, Certificate, s.id)
    return {
        "categories": CERT_CATS,
        "items": [
            {"id": r.id, "name": r.name, "category": r.category, "organization": r.organization, "issue_date": r.issue_date, "description": r.description, "verification_status": r.verification_status, "staff_remark": r.staff_remark, "file_path": r.file_path}
            for r in rows
        ],
        "empty": not rows,
        "message": "No certificates uploaded" if not rows else None,
    }


@router.post("/certificates")
def add_cert(
    name: str = Form(...),
    category: str = Form(...),
    organization: str = Form(""),
    issue_date: str = Form(""),
    description: str = Form(""),
    file: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _student(user, db)
    path = None
    if file and file.filename:
        path, _, _ = save_upload(file, f"certificates/{s.id}", {".pdf", ".jpg", ".jpeg", ".png"})
    rec = Certificate(student_id=s.id, name=name, category=category, organization=organization, issue_date=issue_date or None, description=description, file_path=path)
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Certificate uploaded and pending verification"}


@router.delete("/certificates/{cid}")
def del_cert(cid: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = db.query(Certificate).filter(Certificate.id == cid, Certificate.student_id == s.id).first()
    if not rec:
        raise HTTPException(404, "Not found")
    db.delete(rec)
    db.commit()
    return {"message": "Deleted"}


@router.get("/skills")
def skills(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, Skill, s.id)
    return {
        "items": [{"id": r.id, "skill_name": r.skill_name, "level": r.level, "experience_months": r.experience_months, "verification_status": r.verification_status} for r in rows],
        "empty": not rows,
        "message": "No skills added" if not rows else None,
    }


@router.post("/skills")
def add_skill(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = Skill(student_id=s.id, skill_name=payload.get("skill_name"), level=payload.get("level") or "Beginner", experience_months=int(payload.get("experience_months") or 0))
    db.add(rec)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(400, "Skill already exists")
    return {"id": rec.id, "message": "Skill added"}


@router.delete("/skills/{sid}")
def del_skill(sid: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = db.query(Skill).filter(Skill.id == sid, Skill.student_id == s.id).first()
    if rec:
        db.delete(rec)
        db.commit()
    return {"message": "Deleted"}


@router.get("/linkedin")
def get_linkedin(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = db.query(LinkedInProfile).filter(LinkedInProfile.student_id == s.id).first()
    if not rec:
        return {"empty": True, "url": None, "message": "No LinkedIn profile saved"}
    return {"url": rec.url, "verification_status": rec.verification_status, "staff_remark": rec.staff_remark, "empty": False}


@router.put("/linkedin")
def set_linkedin(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    url = str(payload.get("url") or "").strip()
    if not url.startswith("https://") or "linkedin.com" not in url.lower():
        raise HTTPException(400, "Enter a valid LinkedIn URL starting with https://")
    rec = db.query(LinkedInProfile).filter(LinkedInProfile.student_id == s.id).first()
    if rec:
        rec.url = url
        rec.verification_status = "PENDING"
    else:
        rec = LinkedInProfile(student_id=s.id, url=url)
        db.add(rec)
    db.commit()
    return {"message": "LinkedIn URL saved", "url": url}


@router.get("/extracurricular")
def extras(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, Extracurricular, s.id)
    return {
        "categories": ACTIVITY_CATS,
        "items": [
            {"id": r.id, "activity": r.activity, "category": r.category, "role": r.role, "organization": r.organization, "activity_date": r.activity_date, "description": r.description, "verification_status": r.verification_status}
            for r in rows
        ],
        "empty": not rows,
        "message": "No activities added" if not rows else None,
    }


@router.post("/extracurricular")
def add_extra(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rec = Extracurricular(
        student_id=s.id,
        activity=payload.get("activity") or "",
        category=payload.get("category") or "Other",
        role=payload.get("role"),
        organization=payload.get("organization"),
        activity_date=payload.get("activity_date"),
        description=payload.get("description"),
    )
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Activity added"}


@router.get("/fee-receipts")
def fees(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, FeeReceipt, s.id)
    return {
        "items": [
            {"id": r.id, "academic_year": r.academic_year, "semester": r.semester, "receipt_number": r.receipt_number, "amount": r.amount, "receipt_date": r.receipt_date, "verification_status": r.verification_status, "staff_remark": r.staff_remark}
            for r in rows
        ],
        "empty": not rows,
        "message": "No fee receipts uploaded" if not rows else None,
    }


@router.post("/fee-receipts")
def add_fee(
    academic_year: str = Form(...),
    semester: int = Form(...),
    receipt_number: str = Form(...),
    amount: float = Form(...),
    receipt_date: str = Form(""),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _student(user, db)
    path, _, _ = save_upload(file, f"fees/{s.id}", {".pdf", ".jpg", ".jpeg", ".png"})
    rec = FeeReceipt(student_id=s.id, academic_year=academic_year, semester=semester, receipt_number=receipt_number, amount=amount, receipt_date=receipt_date or None, file_path=path)
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Fee receipt uploaded and pending verification"}


@router.get("/non-due")
def nondue(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, NonDueForm, s.id)
    return {
        "items": [
            {"id": r.id, "academic_year": r.academic_year, "semester": r.semester, "verification_status": r.verification_status, "staff_remark": r.staff_remark, "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "empty": not rows,
        "message": "No non-due form uploaded" if not rows else None,
    }


@router.post("/non-due")
def add_nondue(
    academic_year: str = Form(...),
    semester: int = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _student(user, db)
    path, _, _ = save_upload(file, f"nondue/{s.id}", {".pdf", ".jpg", ".jpeg", ".png"})
    rec = NonDueForm(student_id=s.id, academic_year=academic_year, semester=semester, file_path=path)
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Non-due form uploaded"}


@router.get("/applications")
def applications(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, Application, s.id)
    return {
        "items": [
            {"id": r.id, "application_type": r.application_type, "subject": r.subject, "message": r.message, "status": r.status, "staff_remark": r.staff_remark, "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "empty": not rows,
        "message": "No applications submitted" if not rows else None,
    }


@router.post("/applications")
def add_app(
    application_type: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
    file: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _student(user, db)
    path = None
    if file and file.filename:
        path, _, _ = save_upload(file, f"applications/{s.id}", {".pdf", ".jpg", ".jpeg", ".png"})
    rec = Application(student_id=s.id, application_type=application_type, subject=subject, message=message, attachment_path=path, status="SUBMITTED")
    db.add(rec)
    db.commit()
    if s.classroom and s.classroom.teacher:
        notify(db, s.classroom.teacher.user_id, "Applications", "New application", f"{s.full_name} submitted {application_type}.")
    return {"id": rec.id, "message": "Application submitted"}


@router.get("/documents")
def documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    rows = _list_simple(db, Document, s.id)
    return {
        "items": [
            {"id": r.id, "document_type": r.document_type, "title": r.title, "semester_number": r.semester_number, "original_name": r.original_name, "verification_status": r.verification_status, "staff_remark": r.staff_remark, "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "empty": not rows,
        "message": "No documents uploaded" if not rows else None,
    }


@router.post("/documents")
def add_doc(
    document_type: str = Form(...),
    title: str = Form(...),
    semester_number: Optional[int] = Form(None),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _student(user, db)
    path, orig, _ = save_upload(file, f"documents/{s.id}", {".pdf", ".jpg", ".jpeg", ".png"})
    rec = Document(student_id=s.id, document_type=document_type, title=title, semester_number=semester_number, file_path=path, original_name=orig)
    db.add(rec)
    db.commit()
    return {"id": rec.id, "message": "Document uploaded"}


@router.get("/verification")
def verification(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    buckets = {
        "personal": s.personal_verification,
        "academic": s.academic_verification,
        "marksheets": [(r.id, r.verification_status, r.staff_remark) for r in db.query(Marksheet).filter(Marksheet.student_id == s.id)],
        "certificates": [(r.id, r.verification_status, r.staff_remark) for r in db.query(Certificate).filter(Certificate.student_id == s.id)],
        "skills": [(r.id, r.verification_status, None) for r in db.query(Skill).filter(Skill.student_id == s.id)],
        "linkedin": [(r.id, r.verification_status, r.staff_remark) for r in db.query(LinkedInProfile).filter(LinkedInProfile.student_id == s.id)],
        "extracurricular": [(r.id, r.verification_status, None) for r in db.query(Extracurricular).filter(Extracurricular.student_id == s.id)],
        "fee_receipts": [(r.id, r.verification_status, r.staff_remark) for r in db.query(FeeReceipt).filter(FeeReceipt.student_id == s.id)],
        "non_due": [(r.id, r.verification_status, r.staff_remark) for r in db.query(NonDueForm).filter(NonDueForm.student_id == s.id)],
        "applications": [(r.id, r.status, r.staff_remark) for r in db.query(Application).filter(Application.student_id == s.id)],
        "documents": [(r.id, r.verification_status, r.staff_remark) for r in db.query(Document).filter(Document.student_id == s.id)],
    }
    return buckets


@router.get("/reports/academic.pdf")
def academic_pdf(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    hist = academic_history(db, s)
    lines = [
        f"Student: {s.full_name} ({s.usn})",
        f"Branch / Section: {s.branch} / {s.section}",
        f"CGPA: {hist['cgpa']}",
        f"Attendance: {hist['attendance_percentage']}%",
        "AI predictions are estimates and should not be treated as guaranteed examination results.",
    ]
    tables = []
    for sem in db.query(StudentSemester).filter(StudentSemester.student_id == s.id).all():
        table = [["Code", "Subject", "Int", "Ext", "Total", "GP", "Result"]]
        for m in sem.subject_marks:
            table.append([m.subject_code, m.subject_name, m.internal_marks, m.external_marks, m.total_marks, m.grade_point, m.result])
        tables.append(table)
    data = pdf_bytes(f"Academic Report - {s.usn}", lines, tables)
    return Response(data, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={s.usn}_academic.pdf"})


@router.get("/reports/profile.pdf")
def profile_pdf(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _student(user, db)
    hist = academic_history(db, s)
    skills = db.query(Skill).filter(Skill.student_id == s.id).all()
    certs = db.query(Certificate).filter(Certificate.student_id == s.id).all()
    extras = db.query(Extracurricular).filter(Extracurricular.student_id == s.id).all()
    last = db.query(Prediction).filter(Prediction.student_id == s.id).order_by(Prediction.created_at.desc()).first()
    lines = [
        f"{s.full_name} | {s.usn} | Sem {s.current_semester} | {s.branch}-{s.section}",
        f"CGPA {hist['cgpa']} | Attendance {hist['attendance_percentage']}% | Backlogs {int(hist['backlog_count'])}",
        f"Skills: {', '.join(x.skill_name for x in skills) or 'None'}",
        f"Certificates: {len(certs)} | Activities: {len(extras)}",
        f"Predicted SGPA: {last.predicted_sgpa if last else 'N/A'}",
        "AI predictions are estimates and should not be treated as guaranteed examination results.",
    ]
    data = pdf_bytes(f"Complete Profile - {s.usn}", lines)
    return Response(data, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={s.usn}_profile.pdf"})
