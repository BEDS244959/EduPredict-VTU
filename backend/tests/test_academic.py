from app.auth import password_issues, password_strength
from app.services.academic import combine_internal, compute_total, grade_for_total, result_for_total, sgpa_from_rows, cgpa_from_semesters


def test_password_rules():
    assert password_issues("short") 
    assert password_issues("nouppercase1!")
    assert not password_issues("ValidPass1!")
    assert password_strength("ValidPass1!")["valid"] is True


def test_grade_and_sgpa():
    letter, gp = grade_for_total(91)
    assert letter == "O" and gp == 10
    letter, gp = grade_for_total(39)
    assert letter == "F" and gp == 0
    assert result_for_total(40) == "P"
    assert compute_total(20, 50) == 70
    assert combine_internal(20, 30) == 25
    calc = sgpa_from_rows(
        [
            {"subject_code": "A", "subject_name": "A", "credits": 4, "grade_point": 9, "grade": "A+"},
            {"subject_code": "B", "subject_name": "B", "credits": 2, "grade_point": 8, "grade": "A"},
        ]
    )
    assert calc["sgpa"] == round((4 * 9 + 2 * 8) / 6, 2)
    cg = cgpa_from_semesters([{"sgpa": 8, "total_credits": 20}, {"sgpa": 9, "total_credits": 22}])
    assert cg["cgpa"] == round((8 * 20 + 9 * 22) / 42, 2)
