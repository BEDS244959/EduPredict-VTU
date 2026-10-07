from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import AuditLog, Staff, Student, User

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

PASSWORD_RULE = (
    "Password must be at least 8 characters and include uppercase, lowercase, number, and special character."
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(password, hashed)
    except Exception:
        return False


def password_issues(password: str) -> list[str]:
    issues = []
    if not password or len(password) < 8:
        issues.append("At least 8 characters")
    if not any(c.isupper() for c in password or ""):
        issues.append("One uppercase letter")
    if not any(c.islower() for c in password or ""):
        issues.append("One lowercase letter")
    if not any(c.isdigit() for c in password or ""):
        issues.append("One number")
    if not any(not c.isalnum() for c in password or ""):
        issues.append("One special character")
    return issues


def password_strength(password: str) -> dict:
    issues = password_issues(password)
    score = max(0, 5 - len(issues))
    labels = {0: "Very weak", 1: "Weak", 2: "Fair", 3: "Good", 4: "Strong", 5: "Excellent"}
    return {"score": score, "label": labels[score], "issues": issues, "valid": not issues}


def create_access_token(user: User) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user.id, "role": user.role, "login_id": user.login_id, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def write_audit(db: Session, user_id: Optional[str], action: str, entity: str = "", detail: str = "", ip: str = ""):
    db.add(AuditLog(user_id=user_id, action=action, entity=entity, detail=detail, ip_info=ip))
    db.commit()


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        data = jwt.decode(creds.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = data.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please log in again.")
    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account inactive or not found")
    request.state.user = user
    return user


def require_roles(*roles: str):
    def _inner(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this resource")
        return user

    return _inner


def student_of(user: User, db: Session) -> Student:
    if user.role != "student" or not user.student:
        rec = db.query(Student).filter(Student.user_id == user.id).first()
        if not rec:
            raise HTTPException(status_code=403, detail="Student profile not found")
        return rec
    return user.student


def staff_of(user: User, db: Session) -> Staff:
    rec = db.query(Staff).filter(Staff.user_id == user.id).first()
    if not rec:
        raise HTTPException(status_code=403, detail="Staff profile not found")
    return rec


def staff_can_access_student(db: Session, staff: Staff, student: Student) -> bool:
    if student.class_id and any(c.id == student.class_id for c in staff.classes):
        return True
    return student.class_id in {c.id for c in staff.classes}


def assert_staff_student(db: Session, user: User, student_id: str) -> Student:
    if user.role == "admin":
        st = db.query(Student).filter(Student.id == student_id).first()
        if not st:
            raise HTTPException(404, "Student not found")
        return st
    staff = staff_of(user, db)
    st = db.query(Student).filter(Student.id == student_id).first()
    if not st:
        raise HTTPException(404, "Student not found")
    allowed_classes = {c.id for c in db.query(type(st.classroom)).filter_by() } if False else set()
    from app.models import ClassRoom

    allowed = {c.id for c in db.query(ClassRoom).filter(ClassRoom.staff_id == staff.id).all()}
    if st.class_id not in allowed:
        raise HTTPException(403, "This student is not in your assigned class")
    return st
