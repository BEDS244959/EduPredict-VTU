import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    role: Mapped[str] = mapped_column(String(20), index=True)
    login_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notify_academic: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_attendance: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_documents: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_applications: Mapped[bool] = mapped_column(Boolean, default=True)
    privacy_hide_linkedin: Mapped[bool] = mapped_column(Boolean, default=False)

    student: Mapped["Student"] = relationship(back_populates="user", uselist=False)
    staff: Mapped["Staff"] = relationship(back_populates="user", uselist=False)
    admin: Mapped["Admin"] = relationship(back_populates="user", uselist=False)


class Department(Base, TimestampMixin):
    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    code: Mapped[str] = mapped_column(String(20), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class AcademicYear(Base, TimestampMixin):
    __tablename__ = "academic_years"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(32), unique=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)


class Admin(Base, TimestampMixin):
    __tablename__ = "admins"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True)
    full_name: Mapped[str] = mapped_column(String(160))
    designation: Mapped[str] = mapped_column(String(80), default="Administrator")
    user: Mapped[User] = relationship(back_populates="admin")


class Staff(Base, TimestampMixin):
    __tablename__ = "staff"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True)
    staff_number: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    department_id: Mapped[str | None] = mapped_column(ForeignKey("departments.id"), nullable=True)
    designation: Mapped[str] = mapped_column(String(80), default="Class Teacher")
    photo_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    user: Mapped[User] = relationship(back_populates="staff")
    department: Mapped[Department | None] = relationship()
    classes: Mapped[list["ClassRoom"]] = relationship(back_populates="teacher")


class ClassRoom(Base, TimestampMixin):
    __tablename__ = "classes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    class_name: Mapped[str] = mapped_column(String(120))
    department_id: Mapped[str | None] = mapped_column(ForeignKey("departments.id"), nullable=True)
    branch: Mapped[str] = mapped_column(String(80))
    semester: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(8))
    academic_year: Mapped[str] = mapped_column(String(32), default="2025-26")
    staff_id: Mapped[str | None] = mapped_column(ForeignKey("staff.id"), nullable=True)

    department: Mapped[Department | None] = relationship()
    teacher: Mapped[Staff | None] = relationship(back_populates="classes")
    students: Mapped[list["Student"]] = relationship(back_populates="classroom")


class Student(Base, TimestampMixin):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True)
    usn: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    date_of_birth: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    admission_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_semester: Mapped[int] = mapped_column(Integer, default=1)
    branch: Mapped[str] = mapped_column(String(80), default="CSE")
    section: Mapped[str] = mapped_column(String(8), default="A")
    department_id: Mapped[str | None] = mapped_column(ForeignKey("departments.id"), nullable=True)
    class_id: Mapped[str | None] = mapped_column(ForeignKey("classes.id"), nullable=True)
    photo_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    personal_verification: Mapped[str] = mapped_column(String(32), default="PENDING")
    academic_verification: Mapped[str] = mapped_column(String(32), default="PENDING")

    user: Mapped[User] = relationship(back_populates="student")
    department: Mapped[Department | None] = relationship()
    classroom: Mapped[ClassRoom | None] = relationship(back_populates="students")
    semesters: Mapped[list["StudentSemester"]] = relationship(back_populates="student", cascade="all, delete-orphan")


class ClassStudent(Base, TimestampMixin):
    __tablename__ = "class_students"
    __table_args__ = (UniqueConstraint("class_id", "student_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    class_id: Mapped[str] = mapped_column(ForeignKey("classes.id"))
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))


class SubjectCatalog(Base, TimestampMixin):
    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    code: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(200))
    credits: Mapped[float] = mapped_column(Float, default=3.0)
    semester: Mapped[int] = mapped_column(Integer)
    department_id: Mapped[str | None] = mapped_column(ForeignKey("departments.id"), nullable=True)
    branch: Mapped[str | None] = mapped_column(String(80), nullable=True)

    __table_args__ = (UniqueConstraint("code", "semester", "branch"),)


class StudentSemester(Base, TimestampMixin):
    __tablename__ = "semesters"
    __table_args__ = (UniqueConstraint("student_id", "semester_number"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    semester_number: Mapped[int] = mapped_column(Integer)
    semester_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    sgpa: Mapped[float] = mapped_column(Float, default=0.0)
    is_current: Mapped[int] = mapped_column(Integer, default=0)
    marksheet_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    student: Mapped[Student] = relationship(back_populates="semesters")
    subject_marks: Mapped[list["StudentSubject"]] = relationship(back_populates="semester", cascade="all, delete-orphan")


class StudentSubject(Base, TimestampMixin):
    __tablename__ = "student_subjects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    semester_id: Mapped[str] = mapped_column(ForeignKey("semesters.id"), index=True)
    subject_code: Mapped[str] = mapped_column(String(32))
    subject_name: Mapped[str] = mapped_column(String(200))
    credits: Mapped[float] = mapped_column(Float, default=3.0)
    internal_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    external_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade: Mapped[str | None] = mapped_column(String(8), nullable=True)
    grade_point: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[str] = mapped_column(String(16), default="NA")
    status: Mapped[str] = mapped_column(String(24), default="RECORDED")
    ocr_confidence: Mapped[str] = mapped_column(String(16), default="manual")

    semester: Mapped[StudentSemester] = relationship(back_populates="subject_marks")


class CurrentSemesterMark(Base, TimestampMixin):
    __tablename__ = "marks"
    __table_args__ = (UniqueConstraint("student_id", "semester_number", "subject_code"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    semester_number: Mapped[int] = mapped_column(Integer)
    subject_code: Mapped[str] = mapped_column(String(32))
    subject_name: Mapped[str] = mapped_column(String(200))
    credits: Mapped[float] = mapped_column(Float, default=3.0)
    ia1: Mapped[float] = mapped_column(Float, default=0.0)
    ia2: Mapped[float] = mapped_column(Float, default=0.0)
    internal_combined: Mapped[float] = mapped_column(Float, default=0.0)
    external_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_final: Mapped[float | None] = mapped_column(Float, nullable=True)


class Marksheet(Base, TimestampMixin):
    __tablename__ = "marksheets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    semester_number: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(500))
    original_name: Mapped[str] = mapped_column(String(255))
    extraction_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)


class Backlog(Base, TimestampMixin):
    __tablename__ = "backlogs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    semester_number: Mapped[int] = mapped_column(Integer)
    subject_code: Mapped[str] = mapped_column(String(32))
    subject_name: Mapped[str] = mapped_column(String(200))
    credits: Mapped[float] = mapped_column(Float, default=3.0)
    original_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    rewritten_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[str] = mapped_column(String(16), default="F")
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24), default="ACTIVE")
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)


class Attendance(Base, TimestampMixin):
    __tablename__ = "attendance"
    __table_args__ = (
        UniqueConstraint("student_id", "attendance_date", "subject_name", "hour_number"),
        Index("idx_attendance_class_date", "class_id", "attendance_date"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    class_id: Mapped[str] = mapped_column(ForeignKey("classes.id"))
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    attendance_date: Mapped[str] = mapped_column(String(16), index=True)
    subject_name: Mapped[str] = mapped_column(String(200))
    hour_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    document_type: Mapped[str] = mapped_column(String(40), index=True)
    semester_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    file_path: Mapped[str] = mapped_column(String(500))
    original_name: Mapped[str] = mapped_column(String(255))
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)


class Certificate(Base, TimestampMixin):
    __tablename__ = "certificates"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40))
    organization: Mapped[str | None] = mapped_column(String(160), nullable=True)
    issue_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)


class Skill(Base, TimestampMixin):
    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("student_id", "skill_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    skill_name: Mapped[str] = mapped_column(String(80))
    level: Mapped[str] = mapped_column(String(24), default="Beginner")
    experience_months: Mapped[int] = mapped_column(Integer, default=0)
    certificate_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")


class LinkedInProfile(Base, TimestampMixin):
    __tablename__ = "linkedin_profiles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), unique=True)
    url: Mapped[str] = mapped_column(String(400))
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)


class Extracurricular(Base, TimestampMixin):
    __tablename__ = "extracurriculars"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    activity: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40))
    role: Mapped[str | None] = mapped_column(String(80), nullable=True)
    organization: Mapped[str | None] = mapped_column(String(160), nullable=True)
    activity_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    certificate_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")


class FeeReceipt(Base, TimestampMixin):
    __tablename__ = "fee_receipts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    academic_year: Mapped[str] = mapped_column(String(32))
    semester: Mapped[int] = mapped_column(Integer)
    receipt_number: Mapped[str] = mapped_column(String(64))
    amount: Mapped[float] = mapped_column(Float)
    receipt_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    file_path: Mapped[str] = mapped_column(String(500))
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)


class NonDueForm(Base, TimestampMixin):
    __tablename__ = "non_due_forms"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    academic_year: Mapped[str] = mapped_column(String(32))
    semester: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(500))
    verification_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)


class Application(Base, TimestampMixin):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    application_type: Mapped[str] = mapped_column(String(40))
    subject: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    attachment_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="SUBMITTED")
    staff_remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_by: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    category: Mapped[str] = mapped_column(String(32), default="System")
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)


class MLModel(Base, TimestampMixin):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(32))
    algorithm: Mapped[str] = mapped_column(String(80))
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    feature_list: Mapped[str] = mapped_column(Text, default="[]")
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    training_size: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    semester_number: Mapped[int] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(String(24))
    predicted_sgpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)


class PredictionResult(Base, TimestampMixin):
    __tablename__ = "prediction_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), index=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    subject_code: Mapped[str] = mapped_column(String(32))
    subject_name: Mapped[str] = mapped_column(String(200))
    predicted_marks: Mapped[float] = mapped_column(Float)
    expected_grade: Mapped[str] = mapped_column(String(8))
    risk_level: Mapped[str] = mapped_column(String(32))
    method: Mapped[str] = mapped_column(String(24))
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    recommendations_json: Mapped[str] = mapped_column(Text, default="[]")


class StudyTask(Base, TimestampMixin):
    __tablename__ = "study_tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    subject: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(200))
    priority: Mapped[str] = mapped_column(String(16), default="Medium")
    deadline: Mapped[str | None] = mapped_column(String(16), nullable=True)
    weekday: Mapped[str | None] = mapped_column(String(16), nullable=True)
    hours: Mapped[float] = mapped_column(Float, default=1.0)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


class VerificationRecord(Base, TimestampMixin):
    __tablename__ = "verification_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    entity_type: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    staff_id: Mapped[str | None] = mapped_column(String(32), nullable=True)


class InterventionNote(Base, TimestampMixin):
    __tablename__ = "intervention_notes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    staff_id: Mapped[str] = mapped_column(ForeignKey("staff.id"))
    note: Mapped[str] = mapped_column(Text)


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    entity: Mapped[str | None] = mapped_column(String(80), nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    ip_info: Mapped[str | None] = mapped_column(String(120), nullable=True)


class Setting(Base, TimestampMixin):
    __tablename__ = "settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(80), unique=True)
    value: Mapped[str] = mapped_column(Text)
    group: Mapped[str] = mapped_column(String(40), default="system")


class PasswordResetToken(Base, TimestampMixin):
    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used: Mapped[bool] = mapped_column(Boolean, default=False)


class ImportJob(Base, TimestampMixin):
    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    staff_id: Mapped[str] = mapped_column(ForeignKey("staff.id"))
    file_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(24), default="UPLOADED")
    mapping_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    error_csv_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
