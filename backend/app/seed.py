from datetime import date

from sqlalchemy.orm import Session

from app.auth import hash_password
from app.database import SessionLocal
from app.models import (
    AcademicYear,
    Admin,
    Attendance,
    Certificate,
    ClassRoom,
    CurrentSemesterMark,
    Department,
    Skill,
    Staff,
    Student,
    StudentSemester,
    StudentSubject,
    SubjectCatalog,
    User,
)
from app.services.academic import DEFAULT_SETTINGS, compute_total, grade_for_total, result_for_total, set_group, sgpa_from_rows


DEMO_PASSWORD = "Demo@1234"


def seed_if_needed():
    db: Session = SessionLocal()
    try:
        if db.query(User).first():
            return
        for group, value in DEFAULT_SETTINGS.items():
            set_group(db, group, value)

        cse = Department(name="Computer Science & Engineering", code="CSE", description="CSE department")
        ise = Department(name="Information Science & Engineering", code="ISE", description="ISE department")
        db.add_all([cse, ise])
        db.flush()
        db.add(AcademicYear(name="2025-26", is_current=True))

        admin_user = User(role="admin", login_id="admin", email="admin@edupredict.local", phone="9999999999", password_hash=hash_password(DEMO_PASSWORD))
        db.add(admin_user)
        db.flush()
        db.add(Admin(user_id=admin_user.id, full_name="Campus Administrator", designation="Principal Admin"))

        staff_user = User(role="staff", login_id="TJIT001", email="staff@edupredict.local", phone="9888888888", password_hash=hash_password(DEMO_PASSWORD))
        db.add(staff_user)
        db.flush()
        staff = Staff(user_id=staff_user.id, staff_number="TJIT001", full_name="Prof. Kavya Sharma", department_id=cse.id, designation="Class Teacher")
        db.add(staff)
        db.flush()

        classroom = ClassRoom(class_name="CSE 5A", branch="CSE", semester=5, section="A", academic_year="2025-26", staff_id=staff.id, department_id=cse.id)
        db.add(classroom)
        db.flush()

        catalog = [
            ("21CS51", "Automata Theory", 3, 5),
            ("21CS52", "Computer Networks", 4, 5),
            ("21CS53", "Database Management Systems", 4, 5),
            ("21CS54", "Artificial Intelligence", 3, 5),
            ("21CSL55", "DBMS Laboratory", 1, 5),
            ("21CSL56", "CN Laboratory", 1, 5),
            ("21CS32", "Data Structures", 4, 3),
            ("21CS33", "Analog and Digital Electronics", 3, 3),
            ("21MAT31", "Transform Calculus", 3, 3),
            ("21CS42", "Design and Analysis of Algorithms", 4, 4),
        ]
        for code, name, cr, sem in catalog:
            db.add(SubjectCatalog(code=code, name=name, credits=cr, semester=sem, branch="CSE"))

        demo_students = [
            ("1TJ22CS001", "Ananya Rao", 8.2),
            ("1TJ22CS002", "Rohit Mehta", 6.4),
            ("1TJ22CS003", "Sneha Iyer", 7.1),
            ("1TJ22CS004", "Vikram Nair", 5.5),
            ("1TJ22CS005", "Meera Joshi", 9.0),
        ]
        created = []
        for usn, name, target in demo_students:
            u = User(role="student", login_id=usn, email=f"{usn.lower()}@edupredict.local", phone="9876543210", password_hash=hash_password(DEMO_PASSWORD))
            db.add(u)
            db.flush()
            st = Student(
                user_id=u.id,
                usn=usn,
                full_name=name,
                date_of_birth=date(2004, 6, 15),
                admission_year=2022,
                current_semester=5,
                branch="CSE",
                section="A",
                department_id=cse.id,
                class_id=classroom.id,
                address="Bengaluru",
            )
            db.add(st)
            db.flush()
            created.append((st, target))
            db.add(Skill(student_id=st.id, skill_name="Python", level="Intermediate", experience_months=18))
            db.add(Skill(student_id=st.id, skill_name="SQL", level="Beginner", experience_months=8))
            db.add(Certificate(student_id=st.id, name="Hackathon Participant", category="Hackathon", organization="TJIT", issue_date="2025-03-12"))

        for st, target in created:
            for sem_n, scale in ((3, 0.92), (4, 0.96)):
                sem = StudentSemester(student_id=st.id, semester_number=sem_n, semester_name=f"Semester {sem_n}")
                db.add(sem)
                db.flush()
                rows_meta = []
                for code, name, cr, ssem in catalog:
                    if ssem != sem_n:
                        continue
                    total = min(95, max(38, target * 10 + (hash(code + st.usn) % 12) - 4))
                    internal = round(total * 0.3, 1)
                    external = round(total - internal, 1)
                    letter, gp = grade_for_total(total)
                    rec = StudentSubject(
                        semester_id=sem.id,
                        subject_code=code,
                        subject_name=name,
                        credits=cr,
                        internal_marks=internal,
                        external_marks=external,
                        total_marks=compute_total(internal, external),
                        grade=letter,
                        grade_point=gp,
                        result=result_for_total(total),
                    )
                    db.add(rec)
                    rows_meta.append({"credits": cr, "grade_point": gp, "subject_code": code, "subject_name": name, "grade": letter})
                db.flush()
                sem.sgpa = sgpa_from_rows(rows_meta)["sgpa"]

            for code, name, cr, ssem in catalog:
                if ssem != 5:
                    continue
                ia1 = 20 + (hash(st.usn + code) % 20)
                ia2 = 18 + (hash(code + st.usn) % 22)
                db.add(
                    CurrentSemesterMark(
                        student_id=st.id,
                        semester_number=5,
                        subject_code=code,
                        subject_name=name,
                        credits=cr,
                        ia1=ia1,
                        ia2=ia2,
                        internal_combined=round((ia1 + ia2) / 2, 2),
                    )
                )
            for day in range(1, 11):
                db.add(
                    Attendance(
                        class_id=classroom.id,
                        student_id=st.id,
                        attendance_date=f"2025-09-{day:02d}",
                        subject_name="Computer Networks",
                        hour_number=1,
                        status="Present" if day % 5 else "Absent",
                    )
                )
        db.commit()
    finally:
        db.close()
