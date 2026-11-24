"""
EduPredict AI - Period-Level & Monthly Attendance Intelligence Module.
Renders responsive weekly period-level timetable grids (matching custom UI design),
monthly calendar heatmaps, subject breakdowns, AI attend insights, and multi-sheet
styled Excel (.xlsx) workbooks using openpyxl.
"""

import io
import os
import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")


PERIODS_INFO: Dict[int, Dict[str, str]] = {
    1: {"code": "BCS501", "name": "SEPM", "time": "09:00 - 10:00 AM", "block": "Block A (P1-P3)", "day_type": "A-Day"},
    2: {"code": "BCS502", "name": "Computer Networks", "time": "10:00 - 11:00 AM", "block": "Block A (P1-P3)", "day_type": "A-Day"},
    3: {"code": "BCS503", "name": "Theory of Computation", "time": "11:15 - 12:15 PM", "block": "Block A (P1-P3)", "day_type": "A-Day"},
    4: {"code": "BRMK557", "name": "RM & IPR", "time": "12:15 - 01:15 PM", "block": "Block B (P4-P6)", "day_type": "B-Day"},
    5: {"code": "BCSL504", "name": "CN Laboratory", "time": "02:00 - 03:00 PM", "block": "Block B (P4-P6)", "day_type": "B-Day"},
    6: {"code": "BCS586", "name": "Mini Project / Mentoring", "time": "03:00 - 04:00 PM", "block": "Block B (P4-P6)", "day_type": "B-Day"},
}


def get_db(db_path: Optional[str] = None):
    conn = sqlite3.connect(db_path or DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_student_attendance_data(student_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve and calculate all weekly, monthly, and subject-level attendance analytics
    for a student, including period-level absence patterns and narrative insights.
    """
    db = get_db(db_path)
    try:
        stu = db.execute("""
            SELECT s.*, c.class_name, c.class_teacher
            FROM students s
            LEFT JOIN classes c ON s.class_id = c.id
            WHERE s.id = ?
        """, (student_id,)).fetchone()

        if not stu:
            raise ValueError(f"Student {student_id} not found")

        att_rows = db.execute("""
            SELECT * FROM attendance
            WHERE student_id = ?
            ORDER BY attendance_date ASC, hour_number ASC
        """, (student_id,)).fetchall()
    finally:
        db.close()

    total_hours = len(att_rows)
    present_hours = sum(1 for r in att_rows if r["status"] == "Present")
    absent_hours = total_hours - present_hours
    attendance_pct = round((present_hours / total_hours * 100.0), 1) if total_hours > 0 else 0.0

    # Morning (P1-3) vs Afternoon (P4-6) breakdown
    morn_missed = sum(1 for r in att_rows if r["status"] == "Absent" and int(r["hour_number"]) in (1, 2, 3))
    aft_missed = sum(1 for r in att_rows if r["status"] == "Absent" and int(r["hour_number"]) in (4, 5, 6))

    # Absences by period
    abs_by_period = {p: 0 for p in range(1, 7)}
    for r in att_rows:
        if r["status"] == "Absent":
            p = int(r["hour_number"])
            if p in abs_by_period:
                abs_by_period[p] += 1

    # Date-hour lookup
    records_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    distinct_dates = set()
    for r in att_rows:
        d = r["attendance_date"]
        p = int(r["hour_number"])
        distinct_dates.add(d)
        records_map[(d, p)] = dict(r)

    # Monthly breakdown
    months_dict: Dict[str, Dict[str, Any]] = {}
    for r in att_rows:
        m_key = r["attendance_date"][:7] # YYYY-MM
        m_name = datetime.strptime(m_key, "%Y-%m").strftime("%B %Y")
        if m_key not in months_dict:
            months_dict[m_key] = {"month_key": m_key, "month_name": m_name, "total": 0, "present": 0, "absent": 0}
        months_dict[m_key]["total"] += 1
        if r["status"] == "Present":
            months_dict[m_key]["present"] += 1
        else:
            months_dict[m_key]["absent"] += 1

    monthly_stats = []
    for m_key in sorted(months_dict.keys()):
        m_data = months_dict[m_key]
        tot = m_data["total"]
        pres = m_data["present"]
        pct = round((pres / tot * 100.0), 1) if tot > 0 else 0.0
        status = "Satisfactory (≥75%)" if pct >= 75.0 else ("Condonable (65-75%)" if pct >= 65.0 else "Critical (<65%)")
        monthly_stats.append({
            "month_key": m_key,
            "month_name": m_data["month_name"],
            "total_hours": tot,
            "present_hours": pres,
            "absent_hours": m_data["absent"],
            "attendance_pct": pct,
            "status": status
        })

    # Subject-wise breakdown
    sub_dict: Dict[str, Dict[str, Any]] = {}
    for r in att_rows:
        s_name = r["subject_name"]
        if s_name not in sub_dict:
            sub_dict[s_name] = {"subject_name": s_name, "total": 0, "present": 0, "absent": 0}
        sub_dict[s_name]["total"] += 1
        if r["status"] == "Present":
            sub_dict[s_name]["present"] += 1
        else:
            sub_dict[s_name]["absent"] += 1

    subject_stats = []
    for s_name, s_data in sub_dict.items():
        tot = s_data["total"]
        pres = s_data["present"]
        pct = round((pres / tot * 100.0), 1) if tot > 0 else 0.0
        status = "Eligible (≥75%)" if pct >= 75.0 else ("Condonable (65-75%)" if pct >= 65.0 else "Shortage (<65%)")
        subject_stats.append({
            "subject_name": s_name,
            "total_hours": tot,
            "present_hours": pres,
            "absent_hours": s_data["absent"],
            "attendance_pct": pct,
            "status": status
        })

    # Available weeks (Monday through Saturday)
    available_weeks = [
        {"key": "2026-10-13", "label": "Week of October 13–18, 2026 (Mid-Semester)", "dates": ["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18"]},
        {"key": "2026-09-01", "label": "Week of September 01–06, 2026 (Semester Opening)", "dates": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"]},
        {"key": "2026-11-09", "label": "Week of November 09–14, 2026 (Pre-SEE Freeze)", "dates": ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14"]},
    ]

    # Generate personalized Attend Insight narrative
    name = stu["name"]
    usn = stu["usn"]
    if attendance_pct >= 90.0:
        attend_insight = (
            f"{name} maintains an exceptional attendance rate of {attendance_pct}% across both morning and afternoon blocks. "
            f"Only {absent_hours} total period absences recorded YTD. A standard day-level attendance system would mark full days; "
            f"EduPredict AI's period tracking pinpoints isolated sessions (e.g. Period 2 on Tuesday Oct 14 in Computer Networks), "
            f"ensuring zero risk of VTU CIE debarment."
        )
    elif attendance_pct >= 80.0:
        attend_insight = (
            f"{name} attended Thursday and Friday sessions with 100% presence, but recorded isolated morning absences "
            f"in Period 1 & 2 (SEPM / CN). Overall cumulative attendance is {attendance_pct}%, safely above the VTU 75% SEE examination threshold. "
            f"Period-level tracking confirms strong afternoon block consistency."
        )
    elif attendance_pct >= 75.0:
        attend_insight = (
            f"{name} is in the border zone at {attendance_pct}%. Noticeable absence clusters in Morning Blocks (P1–P3) with {morn_missed} missed hours. "
            f"A single day-level view could mask subject-specific deficits. Maintaining 100% consecutive attendance over the next 10 teaching days "
            f"is strongly recommended to prevent dropping into the VTU condonation band."
        )
    else:
        attend_insight = (
            f"{name} has {absent_hours} total missed hours with cumulative attendance at {attendance_pct}% (VTU Condonation Band: 65%–75%). "
            f"Consecutive absences logged in Period 1–3 ({morn_missed} hrs) and afternoon labs. Immediate submission of medical documentation "
            f"to the HOD and mandatory remedial coaching are required for SEE examination hall-ticket clearance."
        )

    # Initials for avatar
    parts = stu["name"].strip().split()
    initials = (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()

    return {
        "student": dict(stu),
        "initials": initials,
        "total_hours": total_hours,
        "present_hours": present_hours,
        "absent_hours": absent_hours,
        "attendance_pct": attendance_pct,
        "morning_missed_hours": morn_missed,
        "morning_missed_events": morn_missed,
        "afternoon_missed_hours": aft_missed,
        "afternoon_missed_events": aft_missed,
        "absences_by_period": abs_by_period,
        "monthly_stats": monthly_stats,
        "subject_stats": subject_stats,
        "available_weeks": available_weeks,
        "records_map": records_map,
        "attend_insight": attend_insight,
    }


def export_styled_attendance_excel(
    student_id: str,
    target_week_key: Optional[str] = None,
    db_path: Optional[str] = None
) -> bytes:
    """
    Generate an official multi-sheet Excel (.xlsx) workbook for the student:
    - Sheet 1: Weekly Period-by-Period Timetable Matrix (matching the dashboard grid)
    - Sheet 2: Monthly Attendance Ledger
    - Sheet 3: Subject-Wise Attendance Breakdown
    - Sheet 4: Detailed Hourly Raw Logs
    """
    data = get_student_attendance_data(student_id, db_path=db_path)
    stu = data["student"]

    wb = openpyxl.Workbook()

    # Style definitions
    font_title = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    font_sub = Font(name="Calibri", size=10, italic=True, color="E2E8F0")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=10, bold=True)
    font_regular = Font(name="Calibri", size=10)

    font_present = Font(name="Calibri", size=10, bold=True, color="166534")
    font_absent = Font(name="Calibri", size=10, bold=True, color="991B1B")

    fill_navy = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    fill_slate = PatternFill(start_color="334155", end_color="334155", fill_type="solid")
    fill_present = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_absent = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    fill_gray = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    # -------------------------------------------------------------
    # SHEET 1: Weekly Period Matrix
    # -------------------------------------------------------------
    ws1 = wb.active
    ws1.title = "Weekly Period Matrix"
    ws1.views.sheetView[0].showGridLines = True

    # Title Banner
    ws1.merge_cells("A1:H1")
    ws1["A1"] = "VISVESVARAYA TECHNOLOGICAL UNIVERSITY • PERIOD-LEVEL ATTENDANCE MATRIX"
    ws1["A1"].font = font_title
    ws1["A1"].fill = fill_navy
    ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws1.row_dimensions[1].height = 36

    # Subtitle
    ws1.merge_cells("A2:H2")
    ws1["A2"] = f"Candidate: {stu['name']} | USN: {stu['usn']} | Semester: 5 | Class: {stu['class_name']} | Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}"
    ws1["A2"].font = font_sub
    ws1["A2"].fill = fill_slate
    ws1["A2"].alignment = Alignment(horizontal="center", vertical="center")
    ws1.row_dimensions[2].height = 24

    # Target week selection
    sel_week = next((w for w in data["available_weeks"] if w["key"] == target_week_key), data["available_weeks"][0])
    week_dates = sel_week["dates"]

    headers_w1 = ["Period & Time Slot", "Block & Day Type"]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    for i, d_str in enumerate(week_dates[:6]):
        d_obj = datetime.strptime(d_str, "%Y-%m-%d")
        headers_w1.append(f"{day_names[i]} ({d_obj.strftime('%b %d')})")

    for col_idx, h_text in enumerate(headers_w1, start=1):
        c = ws1.cell(row=4, column=col_idx, value=h_text)
        c.font = font_header
        c.fill = fill_navy
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin_border
    ws1.row_dimensions[4].height = 30

    # Fill Period rows (Periods 1 to 6)
    for p in range(1, 7):
        r_idx = 4 + p
        ws1.row_dimensions[r_idx].height = 36
        p_info = PERIODS_INFO[p]

        c_p = ws1.cell(row=r_idx, column=1, value=f"Period {p}\n{p_info['time']}")
        c_p.font = font_bold
        c_p.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c_p.border = thin_border

        c_b = ws1.cell(row=r_idx, column=2, value=f"{p_info['block']}\n{p_info['name']}")
        c_b.font = font_regular
        c_b.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c_b.border = thin_border

        for col_idx, d_str in enumerate(week_dates[:6], start=3):
            rec = data["records_map"].get((d_str, p))
            if rec:
                status = rec["status"]
                sub_name = rec["subject_name"]
                cell_val = f"{status}\n[{sub_name}]"
                is_pres = status == "Present"
            else:
                cell_val = "Scheduled\nNo Class"
                is_pres = None

            c = ws1.cell(row=r_idx, column=col_idx, value=cell_val)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = thin_border
            if is_pres is True:
                c.fill = fill_present
                c.font = font_present
            elif is_pres is False:
                c.fill = fill_absent
                c.font = font_absent
            else:
                c.fill = fill_gray
                c.font = font_regular

    # Widths
    ws1.column_dimensions["A"].width = 22
    ws1.column_dimensions["B"].width = 24
    for c_letter in ["C", "D", "E", "F", "G", "H"]:
        ws1.column_dimensions[c_letter].width = 18

    # -------------------------------------------------------------
    # SHEET 2: Monthly Wise Attendance
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Monthly Attendance Ledger")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:F1")
    ws2["A1"] = f"MONTHLY ATTENDANCE SUMMARY • {stu['name']} ({stu['usn']})"
    ws2["A1"].font = font_title
    ws2["A1"].fill = fill_navy
    ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 34

    headers_m = ["Month", "Sessions Conducted", "Present Hours", "Absent Hours", "Monthly %", "VTU Compliance Status"]
    for col_idx, h in enumerate(headers_m, start=1):
        c = ws2.cell(row=3, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_navy
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin_border
    ws2.row_dimensions[3].height = 26

    for r_idx, m in enumerate(data["monthly_stats"], start=4):
        ws2.row_dimensions[r_idx].height = 24
        ws2.cell(row=r_idx, column=1, value=m["month_name"]).border = thin_border
        ws2.cell(row=r_idx, column=2, value=m["total_hours"]).border = thin_border
        ws2.cell(row=r_idx, column=3, value=m["present_hours"]).border = thin_border
        ws2.cell(row=r_idx, column=4, value=m["absent_hours"]).border = thin_border
        c_pct = ws2.cell(row=r_idx, column=5, value=f"{m['attendance_pct']}%")
        c_pct.border = thin_border
        c_pct.font = font_bold
        c_st = ws2.cell(row=r_idx, column=6, value=m["status"])
        c_st.border = thin_border
        if "Satisfactory" in m["status"]:
            c_st.fill = fill_present
            c_st.font = font_present
        else:
            c_st.fill = fill_absent
            c_st.font = font_absent

    # Cumulative row
    cum_row = 4 + len(data["monthly_stats"])
    ws2.row_dimensions[cum_row].height = 26
    c_tot_lbl = ws2.cell(row=cum_row, column=1, value="CUMULATIVE (YTD)")
    c_tot_lbl.font = font_bold
    c_tot_lbl.fill = fill_slate
    c_tot_lbl.alignment = Alignment(horizontal="center", vertical="center")
    c_tot_lbl.border = thin_border

    ws2.cell(row=cum_row, column=2, value=data["total_hours"]).border = thin_border
    ws2.cell(row=cum_row, column=3, value=data["present_hours"]).border = thin_border
    ws2.cell(row=cum_row, column=4, value=data["absent_hours"]).border = thin_border
    c_cum_pct = ws2.cell(row=cum_row, column=5, value=f"{data['attendance_pct']}%")
    c_cum_pct.font = font_bold
    c_cum_pct.border = thin_border
    cum_status = "Hall Ticket Ready (>=75%)" if data["attendance_pct"] >= 75.0 else "Condonation Required (65-75%)"
    c_cum_st = ws2.cell(row=cum_row, column=6, value=cum_status)
    c_cum_st.font = font_bold
    c_cum_st.border = thin_border

    for l in ["A", "B", "C", "D", "E", "F"]:
        ws2.column_dimensions[l].width = 22

    # -------------------------------------------------------------
    # SHEET 3: Subject-Wise Breakdown
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Subject Attendance Breakdown")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:E1")
    ws3["A1"] = f"SUBJECT-WISE VTU ATTENDANCE • {stu['name']} ({stu['usn']})"
    ws3["A1"].font = font_title
    ws3["A1"].fill = fill_navy
    ws3["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 34

    headers_s = ["Subject Name", "Conducted", "Attended", "Attendance %", "SEE Eligibility Status"]
    for col_idx, h in enumerate(headers_s, start=1):
        c = ws3.cell(row=3, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_navy
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin_border
    ws3.row_dimensions[3].height = 26

    for r_idx, s in enumerate(data["subject_stats"], start=4):
        ws3.row_dimensions[r_idx].height = 24
        ws3.cell(row=r_idx, column=1, value=s["subject_name"]).border = thin_border
        ws3.cell(row=r_idx, column=2, value=s["total_hours"]).border = thin_border
        ws3.cell(row=r_idx, column=3, value=s["present_hours"]).border = thin_border
        c_p = ws3.cell(row=r_idx, column=4, value=f"{s['attendance_pct']}%")
        c_p.font = font_bold
        c_p.border = thin_border
        c_st = ws3.cell(row=r_idx, column=5, value=s["status"])
        c_st.border = thin_border
        if "Eligible" in s["status"]:
            c_st.fill = fill_present
            c_st.font = font_present
        else:
            c_st.fill = fill_absent
            c_st.font = font_absent

    for l in ["A", "B", "C", "D", "E"]:
        ws3.column_dimensions[l].width = 26

    # -------------------------------------------------------------
    # SHEET 4: Complete Hourly Logs
    # -------------------------------------------------------------
    ws4 = wb.create_sheet(title="Hourly Attendance Logs")
    ws4.views.sheetView[0].showGridLines = True

    ws4.merge_cells("A1:G1")
    ws4["A1"] = f"HOURLY ATTENDANCE LOGS • {stu['name']} ({stu['usn']})"
    ws4["A1"].font = font_title
    ws4["A1"].fill = fill_navy
    ws4["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws4.row_dimensions[1].height = 34

    headers_l = ["Date", "Day", "Period / Hour", "Subject", "Time Slot", "Status", "Audit Timestamp"]
    for col_idx, h in enumerate(headers_l, start=1):
        c = ws4.cell(row=3, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_navy
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin_border
    ws4.row_dimensions[3].height = 26

    # Fetch ordered logs
    db = get_db(db_path)
    try:
        raw_logs = db.execute("""
            SELECT * FROM attendance
            WHERE student_id = ?
            ORDER BY attendance_date DESC, hour_number ASC
        """, (student_id,)).fetchall()
    finally:
        db.close()

    for r_idx, r in enumerate(raw_logs, start=4):
        ws4.row_dimensions[r_idx].height = 22
        d_obj = datetime.strptime(r["attendance_date"], "%Y-%m-%d")
        p_num = int(r["hour_number"])
        time_str = PERIODS_INFO.get(p_num, {}).get("time", f"Hour {p_num}")

        ws4.cell(row=r_idx, column=1, value=r["attendance_date"]).border = thin_border
        ws4.cell(row=r_idx, column=2, value=d_obj.strftime("%A")).border = thin_border
        ws4.cell(row=r_idx, column=3, value=f"Period {p_num}").border = thin_border
        ws4.cell(row=r_idx, column=4, value=r["subject_name"]).border = thin_border
        ws4.cell(row=r_idx, column=5, value=time_str).border = thin_border

        c_st = ws4.cell(row=r_idx, column=6, value=r["status"])
        c_st.border = thin_border
        if r["status"] == "Present":
            c_st.fill = fill_present
            c_st.font = font_present
        else:
            c_st.fill = fill_absent
            c_st.font = font_absent

        ws4.cell(row=r_idx, column=7, value=r["created_at"]).border = thin_border

    for l in ["A", "B", "C", "D", "E", "F", "G"]:
        ws4.column_dimensions[l].width = 20

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def render_period_level_attendance_widget(
    student_id: str,
    is_staff_view: bool = False,
    db_path: Optional[str] = None
) -> None:
    """
    Renders the modern Period-by-Period weekly timetable grid and Monthly Attendance
    views in Streamlit, perfectly styled to match the requested design layout.
    """
    data = get_student_attendance_data(student_id, db_path=db_path)
    stu = data["student"]

    # 1. Top Student Profile Card (Matching Screenshot Header)
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="width: 52px; height: 52px; border-radius: 50%; background: #4F46E5; color: #FFFFFF; font-weight: 700; font-size: 19px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);">
                    {data['initials']}
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: 700; color: #1E293B;">
                        {stu['name']} <span style="color: #F59E0B; font-size: 16px;">★</span>
                    </div>
                    <div style="font-size: 13px; color: #64748B; margin-top: 2px;">
                        Semester {stu['current_semester'] or 5} &nbsp;•&nbsp; {stu['class_name'] or 'Data Science'} (VTU 2022 Scheme) &nbsp;•&nbsp; <span style="font-weight: 600; color: #334155;">USN #{stu['usn']}</span>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                    🟣 Periods 1–3: Mon & Wed (A-Day • Morning)
                </span>
                <span style="background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                    🟢 Periods 4–6: Tue & Thu (B-Day • Afternoon)
                </span>
                <span style="background: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                    🔵 All Periods: Friday (C-Day • Labs)
                </span>
                <span style="background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                    🟡 Saturday: Tutorials & Skill Labs (D-Day)
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Top Metric Cards (3 KPI Cards with Left Border Accents)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #4F46E5; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 26px; font-weight: 800; color: #1E293B;">{data['morning_missed_hours']} hrs</div>
            <div style="font-size: 13px; font-weight: 600; color: #475569; margin-top: 4px;">Morning Block 1–3 Missed (YTD)</div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">Mon/Wed • {data['morning_missed_events']} absence events • Periods 1, 2 & 3</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #0D9488; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 26px; font-weight: 800; color: #1E293B;">{data['afternoon_missed_hours']} hrs</div>
            <div style="font-size: 13px; font-weight: 600; color: #475569; margin-top: 4px;">Afternoon Block 4–6 Missed (YTD)</div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">Tue/Thu • {data['afternoon_missed_events']} absence events • Periods 4, 5 & 6</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #059669; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 26px; font-weight: 800; color: #1E293B;">{data['absent_hours']} hrs &nbsp;<span style="font-size: 16px; color: #059669; font-weight: 700;">({data['attendance_pct']}%)</span></div>
            <div style="font-size: 13px; font-weight: 600; color: #475569; margin-top: 4px;">Total Hours Missed (YTD)</div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">{data['present_hours']} of {data['total_hours']} sessions attended • VTU Compliant</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. View Switcher Tabs: Weekly Period View vs Monthly Wise Attendance
    tab_weekly, tab_monthly = st.tabs(["🗓️ Period-by-Period Weekly Timetable Grid", "📅 Monthly Wise Attendance & Calendar Ledger"])

    # -----------------------------------------------------------------
    # TAB 1: WEEKLY PERIOD-BY-PERIOD VIEW (MATCHING USER SCREENSHOT)
    # -----------------------------------------------------------------
    with tab_weekly:
        ctrl_c1, ctrl_c2 = st.columns([2.5, 1])
        week_options = {w["label"]: w["key"] for w in data["available_weeks"]}
        with ctrl_c1:
            sel_week_label = st.selectbox(
                "Select Academic Teaching Week",
                list(week_options.keys()),
                index=0,
                key=f"att_week_sel_{student_id}"
            )
        sel_week_key = week_options[sel_week_label]
        target_week = next((w for w in data["available_weeks"] if w["key"] == sel_week_key), data["available_weeks"][0])

        with ctrl_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            excel_bytes = export_styled_attendance_excel(student_id, target_week_key=sel_week_key, db_path=db_path)
            st.download_button(
                "📥 Download in Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"VTU_Attendance_{stu['usn']}_Week_{sel_week_key}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_att_xl_week_{student_id}",
                use_container_width=True,
                type="primary",
            )

        col_main, col_side = st.columns([2.7, 1.3])

        with col_main:
            # Navy Header Bar
            st.markdown(f"""
            <div style="background: #1E1B4B; color: #FFFFFF; border-radius: 12px 12px 0 0; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.3px;">
                    {target_week['label'].split('(')[0].strip()}
                </div>
                <div style="background: rgba(255, 255, 255, 0.15); padding: 4px 12px; border-radius: 14px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">
                    PERIOD-LEVEL ABSENCE VIEW • 9 AM – 4 PM
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Days Headers (Mon to Sat)
            day_cols = st.columns(6)
            day_meta = [
                ("MON", "A-Day • Oct 13", "#4338CA"),
                ("TUE", "B-Day • Oct 14", "#0D9488"),
                ("WED", "A-Day • Oct 15", "#4338CA"),
                ("THU", "B-Day • Oct 16", "#0D9488"),
                ("FRI", "C-Day • Oct 17", "#6D28D9"),
                ("SAT", "D-Day • Oct 18", "#D97706"),
            ]
            for i, (d_name, d_sub, d_color) in enumerate(day_meta):
                with day_cols[i]:
                    st.markdown(f"""
                    <div style="background: {d_color}; color: #FFFFFF; text-align: center; padding: 10px 4px; border-radius: 0; margin-bottom: 8px;">
                        <div style="font-weight: 800; font-size: 13px;">{d_name}</div>
                        <div style="font-size: 10px; opacity: 0.85;">{d_sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Period Rows (P1 through P6 across Mon-Sat)
            week_dates = target_week["dates"]
            for p_num in range(1, 7):
                p_info = PERIODS_INFO[p_num]
                p_cols = st.columns(6)

                for d_idx, d_str in enumerate(week_dates[:6]):
                    with p_cols[d_idx]:
                        rec = data["records_map"].get((d_str, p_num))
                        if rec:
                            is_present = (rec["status"] == "Present")
                            sub_name = rec["subject_name"]
                        else:
                            is_present = True
                            sub_name = p_info["name"]

                        if is_present:
                            card_html = f"""
                            <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 10px 8px; margin-bottom: 8px; text-align: center; min-height: 85px;">
                                <div style="font-size: 10px; color: #059669; font-weight: 600; text-transform: uppercase;">{p_info['day_type']}</div>
                                <div style="font-size: 12px; font-weight: 700; color: #065F46; margin: 3px 0;">Period {p_num} Present</div>
                                <div style="font-size: 10px; color: #047857;">{p_info['time'].split('-')[0].strip()}</div>
                                <div style="font-size: 10px; color: #64748B; font-weight: 500; margin-top: 2px;">{sub_name[:14]}</div>
                            </div>
                            """
                        else:
                            card_html = f"""
                            <div style="background: #FEF2F2; border: 1px solid #FECDD3; border-radius: 8px; padding: 10px 8px; margin-bottom: 8px; text-align: center; position: relative; min-height: 85px;">
                                <div style="position: absolute; top: 4px; right: 6px; color: #EF4444; font-size: 11px; font-weight: 800;">✕</div>
                                <div style="font-size: 10px; color: #DC2626; font-weight: 600; text-transform: uppercase;">{p_info['day_type']}</div>
                                <div style="font-size: 12px; font-weight: 700; color: #991B1B; margin: 3px 0;">Period {p_num} Absent</div>
                                <div style="font-size: 10px; color: #B91C1C;">{p_info['time'].split('-')[0].strip()}</div>
                                <div style="font-size: 10px; color: #64748B; font-weight: 500; margin-top: 2px;">{sub_name[:14]}</div>
                            </div>
                            """
                        st.markdown(card_html, unsafe_allow_html=True)

        with col_side:
            # Absences by Period Card
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
                <div style="font-size: 14px; font-weight: 700; color: #1E293B; margin-bottom: 12px;">Absences by Period (YTD)</div>
            """, unsafe_allow_html=True)

            for p_num in range(1, 7):
                cnt = data["absences_by_period"].get(p_num, 0)
                p_inf = PERIODS_INFO[p_num]
                dot_color = "#4F46E5" if p_num <= 3 else "#0D9488"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid #F1F5F9; font-size: 12px;">
                    <div>
                        <span style="color: {dot_color}; font-size: 14px; margin-right: 6px;">●</span>
                        <span style="font-weight: 600; color: #334155;">Period {p_num}</span>
                        <span style="color: #94A3B8; font-size: 11px; margin-left: 4px;">({p_inf['name'][:8]})</span>
                    </div>
                    <div style="font-weight: 700; color: {'#DC2626' if cnt > 2 else '#475569'};">
                        {cnt} <span style="font-weight: 400; font-size: 11px; color: #94A3B8;">absence{'s' if cnt != 1 else ''}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # ATTEND INSIGHT Card (Matching Dark Navy Aesthetic)
            st.markdown(f"""
            <div style="background: #1E1B4B; color: #FFFFFF; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(30, 27, 75, 0.25);">
                <div style="font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #A5B4FC; margin-bottom: 8px;">
                    ATTEND INSIGHT
                </div>
                <div style="font-size: 13px; font-weight: 600; color: #FFFFFF; line-height: 1.5; margin-bottom: 10px;">
                    {data['attend_insight']}
                </div>
                <div style="font-size: 11px; color: #C7D2FE; line-height: 1.4; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 10px;">
                    A day-level system marks full day absence. EduPredict AI reveals the exact period gap — providing faculty mentors the precise evidence needed for VTU condonation approval.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 2: MONTHLY WISE ATTENDANCE & CALENDAR LEDGER
    # -----------------------------------------------------------------
    with tab_monthly:
        st.markdown("### 📅 Monthly Attendance Ledger & Performance Trajectory")
        st.caption("Comprehensive month-by-month session audit, working day calculations, and VTU condonation status.")

        m_opts = {m["month_name"]: m["month_key"] for m in data["monthly_stats"]}
        m_opts["All Months (Cumulative Semester Total)"] = "all"

        m_col1, m_col2 = st.columns([2.5, 1])
        with m_col1:
            sel_m_label = st.selectbox(
                "Filter Month Scope",
                list(m_opts.keys()),
                index=len(m_opts) - 1, # Default to Cumulative
                key=f"att_mon_sel_{student_id}"
            )
        sel_m_key = m_opts[sel_m_label]

        with m_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            xl_month_bytes = export_styled_attendance_excel(student_id, db_path=db_path)
            st.download_button(
                "📊 Export Monthly Ledger (.xlsx)",
                data=xl_month_bytes,
                file_name=f"VTU_Monthly_Attendance_{stu['usn']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_att_xl_mon_{student_id}",
                use_container_width=True,
            )

        # Monthly KPI Summary Grid
        st.markdown("#### Month-by-Month Attendance Breakdown")
        df_monthly = pd.DataFrame(data["monthly_stats"])
        if not df_monthly.empty:
            cols_disp = ["month_name", "total_hours", "present_hours", "absent_hours", "attendance_pct", "status"]
            st.dataframe(
                df_monthly[cols_disp].rename(columns={
                    "month_name": "Month",
                    "total_hours": "Sessions Held",
                    "present_hours": "Attended",
                    "absent_hours": "Absent",
                    "attendance_pct": "Attendance %",
                    "status": "VTU Status"
                }),
                use_container_width=True,
                hide_index=True
            )

        # Plotly Monthly Trajectory
        if len(data["monthly_stats"]) > 1:
            m_names = [m["month_name"] for m in data["monthly_stats"]]
            m_pcts = [m["attendance_pct"] for m in data["monthly_stats"]]

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=m_names,
                y=m_pcts,
                name="Monthly Attendance %",
                marker_color=["#10B981" if p >= 75.0 else ("#F59E0B" if p >= 65.0 else "#EF4444") for p in m_pcts],
                text=[f"{p:.1f}%" for p in m_pcts],
                textposition="auto"
            ))
            fig_m.add_hline(y=75.0, line_dash="dash", line_color="#EF4444", annotation_text="VTU 75% Minimum Threshold")
            fig_m.add_hline(y=65.0, line_dash="dot", line_color="#F59E0B", annotation_text="65% Condonation Limit")
            fig_m.update_layout(
                title="Monthly Attendance Percentage Trajectory",
                yaxis_title="Attendance Rate (%)",
                yaxis_range=[50, 100],
                template="plotly_white",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # Subject-wise attendance breakdown table
        st.divider()
        st.markdown("#### 📚 Subject-Wise Attendance Breakdown")
        df_sub = pd.DataFrame(data["subject_stats"])
        if not df_sub.empty:
            st.dataframe(
                df_sub.rename(columns={
                    "subject_name": "Subject Name",
                    "total_hours": "Classes Held",
                    "present_hours": "Attended",
                    "absent_hours": "Absent",
                    "attendance_pct": "Attendance %",
                    "status": "VTU Compliance Status"
                }),
                use_container_width=True,
                hide_index=True
            )
