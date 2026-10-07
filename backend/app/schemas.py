from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    login_id: str
    full_name: str


class LoginIn(BaseModel):
    role: str
    login_id: str
    password: str


class StudentRegisterIn(BaseModel):
    full_name: str
    usn: str
    email: EmailStr
    phone: str
    date_of_birth: Optional[date] = None
    admission_year: int
    current_semester: int = Field(ge=1, le=8)
    branch: str
    section: str
    password: str
    confirm_password: str


class StaffRegisterIn(BaseModel):
    staff_number: str
    full_name: str
    email: EmailStr
    phone: str
    department_code: Optional[str] = None
    designation: str = "Class Teacher"
    password: str
    confirm_password: str


class PasswordChangeIn(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


class ForgotIn(BaseModel):
    role: str
    login_id: str
    email: EmailStr


class ResetIn(BaseModel):
    token: str
    new_password: str
    confirm_password: str


class GenericMsg(BaseModel):
    message: str
    extra: Optional[dict[str, Any]] = None
