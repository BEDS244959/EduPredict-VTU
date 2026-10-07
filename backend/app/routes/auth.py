from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    hash_password,
    password_issues,
    password_strength,
    verify_password,
    write_audit,
    get_current_user,
)
from app.config import settings
from app.database import get_db
from app.models import Admin, Department, PasswordResetToken, Staff, Student, User
from app.schemas import (
    ForgotIn,
    LoginIn,
    PasswordChangeIn,
    ResetIn,
    StaffRegisterIn,
    StudentRegisterIn,
)
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import re

router = APIRouter(prefix="/auth", tags=["auth"])

USN_RE = re.compile(r"^[0-9A-Z]{10,15}$")
PHONE_RE = re.compile(r"^[0-9+\- ]{8,15}$")


def _user_name(db: Session, user: User) -> str:
    if user.role == "student":
        s = db.query(Student).filter(Student.user_id == user.id).first()
        return s.full_name if s else user.login_id
    if user.role == "staff":
        s = db.query(Staff).filter(Staff.user_id == user.id).first()
        return s.full_name if s else user.login_id
    a = db.query(Admin).filter(Admin.user_id == user.id).first()
    return a.full_name if a else user.login_id


@router.post("/login")
def login(payload: LoginIn, request: Request, db: Session = Depends(get_db)):
    role = payload.role.lower().strip()
    login_id = payload.login_id.strip().upper() if role != "admin" else payload.login_id.strip()
    if role == "admin":
        login_id = payload.login_id.strip()
    user = db.query(User).filter(User.login_id == login_id, User.role == role).first()
    if not user or not verify_password(payload.password, user.password_hash):
        write_audit(db, None, "LOGIN_FAILED", role, login_id, request.client.host if request.client else "")
        raise HTTPException(401, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(403, "Account is disabled")
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    write_audit(db, user.id, "LOGIN", role, login_id, request.client.host if request.client else "")
    token = create_access_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "login_id": user.login_id,
        "full_name": _user_name(db, user),
        "user_id": user.id,
    }


@router.post("/register/student")
def register_student(payload: StudentRegisterIn, request: Request, db: Session = Depends(get_db)):
    usn = payload.usn.strip().upper()
    if not USN_RE.match(usn):
        raise HTTPException(400, "USN format looks invalid. Use 10-15 alphanumeric characters.")
    if not PHONE_RE.match(payload.phone.strip()):
        raise HTTPException(400, "Enter a valid phone number")
    if payload.password != payload.confirm_password:
        raise HTTPException(400, "Passwords do not match")
    issues = password_issues(payload.password)
    if issues:
        raise HTTPException(400, "Password requirements: " + ", ".join(issues))
    if db.query(User).filter(User.login_id == usn).first() or db.query(Student).filter(Student.usn == usn).first():
        raise HTTPException(400, "USN is already registered")
    if db.query(User).filter(User.email == str(payload.email).lower()).first():
        raise HTTPException(400, "Email is already registered")
    user = User(
        role="student",
        login_id=usn,
        email=str(payload.email).lower(),
        phone=payload.phone.strip(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.flush()
    student = Student(
        user_id=user.id,
        usn=usn,
        full_name=payload.full_name.strip(),
        date_of_birth=payload.date_of_birth,
        admission_year=payload.admission_year,
        current_semester=payload.current_semester,
        branch=payload.branch.strip().upper(),
        section=payload.section.strip().upper(),
    )
    db.add(student)
    db.commit()
    write_audit(db, user.id, "STUDENT_CREATED", "student", usn, request.client.host if request.client else "")
    return {"message": "Registration successful. You can now log in with your USN.", "usn": usn}


@router.post("/register/staff")
def register_staff(payload: StaffRegisterIn, request: Request, db: Session = Depends(get_db)):
    sid = payload.staff_number.strip().upper()
    if payload.password != payload.confirm_password:
        raise HTTPException(400, "Passwords do not match")
    issues = password_issues(payload.password)
    if issues:
        raise HTTPException(400, "Password requirements: " + ", ".join(issues))
    if db.query(User).filter(User.login_id == sid).first():
        raise HTTPException(400, "Staff / TJIT number already registered")
    if db.query(User).filter(User.email == str(payload.email).lower()).first():
        raise HTTPException(400, "Email is already registered")
    dept = None
    if payload.department_code:
        dept = db.query(Department).filter(Department.code == payload.department_code.strip().upper()).first()
    user = User(
        role="staff",
        login_id=sid,
        email=str(payload.email).lower(),
        phone=payload.phone.strip(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.flush()
    staff = Staff(
        user_id=user.id,
        staff_number=sid,
        full_name=payload.full_name.strip(),
        department_id=dept.id if dept else None,
        designation=payload.designation.strip(),
    )
    db.add(staff)
    db.commit()
    write_audit(db, user.id, "STAFF_CREATED", "staff", sid)
    return {"message": "Staff account created. You can now log in.", "staff_number": sid}


@router.get("/me")
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = {"id": user.id, "role": user.role, "login_id": user.login_id, "email": user.email, "phone": user.phone}
    if user.role == "student":
        s = db.query(Student).filter(Student.user_id == user.id).first()
        if s:
            profile.update(
                {
                    "student_id": s.id,
                    "full_name": s.full_name,
                    "usn": s.usn,
                    "current_semester": s.current_semester,
                    "branch": s.branch,
                    "section": s.section,
                    "class_id": s.class_id,
                    "photo_path": s.photo_path,
                    "admission_year": s.admission_year,
                }
            )
    elif user.role == "staff":
        s = db.query(Staff).filter(Staff.user_id == user.id).first()
        if s:
            profile.update({"staff_id": s.id, "full_name": s.full_name, "staff_number": s.staff_number, "designation": s.designation})
    else:
        a = db.query(Admin).filter(Admin.user_id == user.id).first()
        if a:
            profile.update({"admin_id": a.id, "full_name": a.full_name, "designation": a.designation})
    return profile


@router.post("/password/strength")
def strength(payload: dict):
    return password_strength(payload.get("password") or "")


@router.post("/password/change")
def change_password(payload: PasswordChangeIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    if payload.new_password != payload.confirm_password:
        raise HTTPException(400, "Passwords do not match")
    issues = password_issues(payload.new_password)
    if issues:
        raise HTTPException(400, "Password requirements: " + ", ".join(issues))
    user.password_hash = hash_password(payload.new_password)
    db.commit()
    write_audit(db, user.id, "PASSWORD_CHANGED", "user", user.login_id)
    return {"message": "Password updated"}


@router.post("/password/forgot")
def forgot(payload: ForgotIn, db: Session = Depends(get_db)):
    role = payload.role.lower()
    login_id = payload.login_id.strip().upper() if role != "admin" else payload.login_id.strip()
    user = db.query(User).filter(User.login_id == login_id, User.role == role, User.email == str(payload.email).lower()).first()
    msg = "If the account exists, a reset token was generated. Configure SMTP in production to email it."
    if not user:
        return {"message": msg}
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    db.add(
        PasswordResetToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        )
    )
    db.commit()
    extra = {"reset_token": token} if settings.DEMO_MODE else {}
    write_audit(db, user.id, "PASSWORD_RESET_REQUEST", "user", user.login_id)
    return {"message": msg, **extra}


@router.post("/password/reset")
def reset(payload: ResetIn, db: Session = Depends(get_db)):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(400, "Passwords do not match")
    issues = password_issues(payload.new_password)
    if issues:
        raise HTTPException(400, "Password requirements: " + ", ".join(issues))
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    rec = (
        db.query(PasswordResetToken)
        .filter(PasswordResetToken.token_hash == token_hash, PasswordResetToken.used.is_(False))
        .first()
    )
    if not rec or rec.expires_at < datetime.now(timezone.utc):
        raise HTTPException(400, "Reset token is invalid or expired")
    user = db.query(User).filter(User.id == rec.user_id).first()
    user.password_hash = hash_password(payload.new_password)
    rec.used = True
    db.commit()
    write_audit(db, user.id, "PASSWORD_RESET", "user", user.login_id)
    return {"message": "Password reset successful. You can log in now."}


@router.post("/logout")
def logout(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    write_audit(db, user.id, "LOGOUT", "user", user.login_id)
    return {"message": "Logged out"}
