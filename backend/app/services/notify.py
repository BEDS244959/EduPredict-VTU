from sqlalchemy.orm import Session

from app.models import Notification, User


def notify(db: Session, user_id: str | None, category: str, title: str, body: str):
    if not user_id:
        return
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return
    prefs = {
        "Academic": user.notify_academic,
        "Attendance": user.notify_attendance,
        "Documents": user.notify_documents,
        "Applications": user.notify_applications,
        "System": True,
        "Security": True,
    }
    if prefs.get(category, True) is False:
        return
    db.add(Notification(user_id=user_id, category=category, title=title, body=body))
    db.commit()
