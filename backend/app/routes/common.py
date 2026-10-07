from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.auth import get_current_user, require_roles
from app.database import get_db
from app.models import ClassRoom, Document, Notification, Staff, Student, SubjectCatalog, User
from app.services.files import safe_resolve

router = APIRouter(tags=["common"])


@router.get("/notifications")
def list_notes(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Notification).filter(Notification.user_id == user.id).order_by(Notification.created_at.desc()).limit(100).all()
    unread = sum(1 for r in rows if not r.is_read)
    return {
        "unread": unread,
        "items": [
            {"id": r.id, "category": r.category, "title": r.title, "body": r.body, "is_read": r.is_read, "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "empty": not rows,
        "message": "No notifications" if not rows else None,
    }


@router.post("/notifications/read")
def mark_read(payload: dict | None = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ids = (payload or {}).get("ids")
    q = db.query(Notification).filter(Notification.user_id == user.id)
    if ids:
        q = q.filter(Notification.id.in_(ids))
    for n in q.all():
        n.is_read = True
    db.commit()
    return {"message": "Marked as read"}


@router.get("/search")
def search(q: str = Query("", min_length=1), user: User = Depends(require_roles("staff", "admin")), db: Session = Depends(get_db)):
    like = f"%{q}%"
    students = db.query(Student).filter(or_(Student.full_name.ilike(like), Student.usn.ilike(like))).limit(20).all()
    staff = db.query(Staff).filter(or_(Staff.full_name.ilike(like), Staff.staff_number.ilike(like))).limit(10).all()
    subjects = db.query(SubjectCatalog).filter(or_(SubjectCatalog.code.ilike(like), SubjectCatalog.name.ilike(like))).limit(10).all()
    classes = db.query(ClassRoom).filter(ClassRoom.class_name.ilike(like)).limit(10).all()
    docs = db.query(Document).filter(Document.title.ilike(like)).limit(10).all()
    return {
        "students": [{"id": s.id, "usn": s.usn, "full_name": s.full_name} for s in students],
        "staff": [{"id": s.id, "staff_number": s.staff_number, "full_name": s.full_name} for s in staff],
        "subjects": [{"id": s.id, "code": s.code, "name": s.name} for s in subjects],
        "classes": [{"id": c.id, "class_name": c.class_name} for c in classes],
        "documents": [{"id": d.id, "title": d.title, "type": d.document_type} for d in docs],
    }


@router.get("/files")
def get_file(path: str, user: User = Depends(get_current_user)):
    target = safe_resolve(path)
    if user.role == "student" and user.id not in path and getattr(user, "student", None):
        if user.student and user.student.id not in path:
            raise HTTPException(403, "Not allowed")
    return FileResponse(target)


@router.get("/settings/me")
def my_settings(user: User = Depends(get_current_user)):
    return {
        "notify_academic": user.notify_academic,
        "notify_attendance": user.notify_attendance,
        "notify_documents": user.notify_documents,
        "notify_applications": user.notify_applications,
        "privacy_hide_linkedin": user.privacy_hide_linkedin,
        "email": user.email,
        "phone": user.phone,
    }


@router.put("/settings/me")
def update_my_settings(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    for f in ("notify_academic", "notify_attendance", "notify_documents", "notify_applications", "privacy_hide_linkedin"):
        if f in payload:
            setattr(user, f, bool(payload[f]))
    if "phone" in payload:
        user.phone = payload["phone"]
    db.commit()
    return {"message": "Preferences saved"}
