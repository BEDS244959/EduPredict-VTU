"""
Modular Faculty Intelligence, Segregation, Individual Data Entry, Batch Automation & Placement UI.

Imported cleanly by portal_staff.py so Dashboard, Individual Record Entry,
Placement, Excel Import, Daily Attendance, and Subjects (Master) share
one unified, well-structured UI layer backed by academic_engine.py.
"""

import io
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from academic_engine import (
    ALL_PLACEMENT_STATUSES,
    add_mentoring_log,
    apply_for_placement_drive,
    batch_parse_vtu_marksheet_files,
    bulk_log_fast_attendance,
    bulk_save_class_ia_and_assignments,
    clear_streamlit_caches,
    delete_placement_record,
    dispatch_parent_communication,
    execute_attendance_bulk_import,
    execute_marks_bulk_import,
    execute_placement_bulk_import,
    execute_semester_batch_promotion,
    expand_marksheet_uploads_with_zip,
    generate_official_parent_notice_pdf,
    get_academic_rules,
    get_all_placements,
    get_comprehensive_student_analytics,
    get_drive_applications,
    get_marksheet_revisions,
    get_mentoring_logs,
    get_parent_communications,
    get_placement_drives,
    get_student_placements,
    match_students_for_drive,
    notify_shortlisted_students,
    recalculate_all_student_records,
    review_placement_offer,
    save_academic_rules,
    save_placement_drive,
    save_student_ia_subject_mark,
    save_student_semester_result,
    seed_vtu_master_subjects,
    segregate_students,
    simulate_what_if_cgpa,
    update_drive_application_stage,
    update_student_attendance_summary,
    upsert_placement_record,
    validate_attendance_excel,
    validate_marks_excel,
    validate_placement_excel,
)
from auth import log_audit_action, send_notification
from excel_handler import export_segregated_report_excel
from reports import (
    generate_vtu_hall_ticket_pdf,
    generate_remedial_mentoring_plan_pdf,
    generate_class_hall_tickets_zip,
)
from ocr_engine import (
    VTU_KNOWN_SUBJECTS,
    extract_offer_letter_details,
    extract_vtu_metadata,
    get_canonical_subject_info,
    normalize_subject_code,
    process_marksheet_file,
)


def _df_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes safely."""
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# 1. AUTOMATED VTU RESULT SEGREGATION & FACULTY INTELLIGENCE HUB
# ============================================================

def render_segregation_dashboard_section(
    db_getter,
    staff_classes: List[Any],
    user_id: str,
    staff_role: str = "staff",
) -> None:
    """Render automated SGPA, Marks, Credits, Next-Sem Eligibility, Low Attendance & Placement segregation on Faculty Dashboard."""
    st.markdown("## 🎯 Automated VTU Result Segregation & Student Eligibility Hub")
    st.caption(
        "Automatically segregates students by Official SGPA, Aggregate Marks, Earned Credits, "
        "Next-Semester Eligibility, Low Attendance (< threshold), and Placement Status whenever a VTU result or record is updated."
    )

    saved_rules = get_academic_rules()
    class_options = {"All Assigned Classes": None}
    for c in staff_classes:
        class_options[c["class_name"]] = [c["id"]]
    all_assigned_ids = [c["id"] for c in staff_classes] if staff_classes else None

    f_c1, f_c2, f_c3, f_c4 = st.columns([1.8, 1.5, 1.3, 1.4])
    sel_cls_label = f_c1.selectbox("Filter Cohort / Class", list(class_options.keys()), key="seg_cls_filter")
    scoped_class_ids = class_options[sel_cls_label] if class_options[sel_cls_label] is not None else all_assigned_ids

    sem_filter_opts = {"All Uploaded Semesters (Cumulative)": None}
    for s_n in range(1, 9):
        sem_filter_opts[f"Semester {s_n} Only"] = s_n
    sel_sem_label = f_c2.selectbox("Result Scope", list(sem_filter_opts.keys()), key="seg_sem_filter")
    scoped_sem = sem_filter_opts[sel_sem_label]

    with f_c3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Recalculate & Update All Records", type="primary", use_container_width=True, key="seg_refresh_btn"):
            c_target = scoped_class_ids[0] if (scoped_class_ids and len(scoped_class_ids) == 1) else None
            re_res = recalculate_all_student_records(class_id=c_target, actor_id=user_id)
            st.success(
                f"🎉 Recalculated {re_res['total_students']} student records! "
                f"Passed: {re_res['passed_count']} | High Credits: {re_res['high_credits_count']} | "
                f"Border Pass: {re_res['border_pass_count']} | Next-Sem Eligible: {re_res['eligible_count']}"
            )
            st.rerun()

    # Configurable VTU rules bar
    with st.expander("⚙️ Configure Next-Semester Eligibility & Low Attendance Threshold Rules", expanded=False):
        rc1, rc2, rc3, rc4 = st.columns(4)
        low_att_val = rc1.number_input(
            "Low Attendance Threshold (%)",
            min_value=30.0,
            max_value=95.0,
            value=float(saved_rules.get("low_attendance_threshold", 75.0)),
            step=1.0,
            key="rule_low_att",
        )
        max_bk_val = rc2.number_input(
            "Max Allowed Active Backlogs",
            min_value=0,
            max_value=12,
            value=int(saved_rules.get("max_backlogs_next_sem", 4)),
            step=1,
            key="rule_max_bk",
        )
        cred_mode = rc3.selectbox(
            "Credit Eligibility Mode",
            ["threshold", "percentage"],
            index=0 if saved_rules.get("enforce_credit_pct_mode", "threshold") == "threshold" else 1,
            format_func=lambda x: "Min Earned Credits" if x == "threshold" else "Min Earned Credits %",
            key="rule_cred_mode",
        )
        if cred_mode == "threshold":
            min_cr_val = rc4.number_input(
                "Min Earned Credits Required",
                min_value=0.0,
                max_value=200.0,
                value=float(saved_rules.get("min_credits_next_sem", 16.0)),
                step=1.0,
                key="rule_min_cred",
            )
            min_pct_val = float(saved_rules.get("min_credit_pct_next_sem", 50.0))
        else:
            min_pct_val = rc4.number_input(
                "Min Earned Credits (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(saved_rules.get("min_credit_pct_next_sem", 50.0)),
                step=5.0,
                key="rule_min_pct",
            )
            min_cr_val = float(saved_rules.get("min_credits_next_sem", 16.0))

        rc5, rc6, rc7, rc8 = st.columns(4)
        cond_att_val = rc5.number_input(
            "Condonable Attendance Min (%)",
            min_value=30.0,
            max_value=85.0,
            value=float(saved_rules.get("condonable_min_attendance_pct", 65.0)),
            step=1.0,
            key="rule_cond_att",
        )
        cie_pct_val = rc6.number_input(
            "Min CIE / IA % (for Exam)",
            min_value=20.0,
            max_value=70.0,
            value=float(saved_rules.get("exam_min_cie_pct", 40.0)),
            step=5.0,
            key="rule_cie_pct",
        )
        scheme_opts = ["VTU 2022 Scheme", "VTU 2021 Scheme", "VTU 2024 Scheme"]
        cur_scheme = saved_rules.get("vtu_scheme", "VTU 2022 Scheme")
        scheme_val = rc7.selectbox(
            "VTU Academic Scheme",
            scheme_opts,
            index=scheme_opts.index(cur_scheme) if cur_scheme in scheme_opts else 0,
            key="rule_vtu_scheme",
        )
        yb_val = rc8.checkbox(
            "Enforce VTU Year-Back Backlog Rule",
            value=str(saved_rules.get("enforce_year_back_rule", "1")) in ("1", "true", "True"),
            key="rule_year_back",
        )

        active_rules = {
            "low_attendance_threshold": low_att_val,
            "exam_min_attendance_pct": low_att_val,
            "condonable_min_attendance_pct": cond_att_val,
            "exam_min_cie_pct": cie_pct_val,
            "max_backlogs_next_sem": int(max_bk_val),
            "enforce_credit_pct_mode": cred_mode,
            "min_credits_next_sem": min_cr_val,
            "min_credit_pct_next_sem": min_pct_val,
            "vtu_scheme": scheme_val,
            "enforce_year_back_rule": "1" if yb_val else "0",
        }
        if st.button("💾 Save Rules as Institutional Default", type="primary", key="save_acad_rules_btn"):
            save_academic_rules(active_rules, updated_by=user_id)
            st.success("✅ Institutional VTU eligibility, exam & attendance thresholds saved!")
            st.rerun()
    if "rule_low_att" in st.session_state:
        active_rules = {
            "low_attendance_threshold": float(st.session_state["rule_low_att"]),
            "exam_min_attendance_pct": float(st.session_state["rule_low_att"]),
            "condonable_min_attendance_pct": float(st.session_state.get("rule_cond_att", saved_rules.get("condonable_min_attendance_pct", 65.0))),
            "exam_min_cie_pct": float(st.session_state.get("rule_cie_pct", saved_rules.get("exam_min_cie_pct", 40.0))),
            "max_backlogs_next_sem": int(st.session_state["rule_max_bk"]),
            "enforce_credit_pct_mode": st.session_state["rule_cred_mode"],
            "min_credits_next_sem": float(st.session_state.get("rule_min_cred", saved_rules.get("min_credits_next_sem", 16.0))),
            "min_credit_pct_next_sem": float(st.session_state.get("rule_min_pct", saved_rules.get("min_credit_pct_next_sem", 50.0))),
            "vtu_scheme": st.session_state.get("rule_vtu_scheme", saved_rules.get("vtu_scheme", "VTU 2022 Scheme")),
            "enforce_year_back_rule": "1" if st.session_state.get("rule_year_back", True) else "0",
        }
    else:
        active_rules = saved_rules

    df_all = get_comprehensive_student_analytics(
        class_ids=scoped_class_ids,
        semester_filter=scoped_sem,
        rules=active_rules,
    )
    seg = segregate_students(df_all, rules=active_rules)
    summary = seg["summary"]

    with f_c4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if not df_all.empty:
            wb_bytes = export_segregated_report_excel(seg, rules=active_rules)
            st.download_button(
                "📥 Export Segregated (.xlsx)",
                data=wb_bytes,
                file_name=f"EduPredict_Segregated_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_seg_excel_all",
            )

    if df_all.empty:
        st.info("No students found in the selected class scope yet.")
        return

    # KPI Summary Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Students", summary["total_students"], f"{summary['with_vtu_results']} VTU Uploaded")
    k2.metric("Passed (0 Backlogs)", summary["passed_count"], f"{summary.get('pass_percentage', 0.0):.1f}% Pass Rate")
    k3.metric("Not Passed / Backlogs", summary["not_passed_count"], f"{summary.get('border_pass_count', 0)} Just Border Pass")
    k4.metric("Next-Sem Eligible", summary["eligible_count"], f"{summary['not_eligible_count']} Not Eligible")
    k5.metric("Low Attendance", summary["low_attendance_count"], f"Threshold < {active_rules['low_attendance_threshold']:g}%")
    k6.metric("Placed Students", summary["placed_count"], f"{summary['not_placed_count']} Not Placed")

    tab_pass_fail, tab_credits, tab_sgpa, tab_marks, tab_exam, tab_att, tab_place, tab_custom = st.tabs([
        "🎯 Passed vs Not Passed & Borderline",
        "🎓 Next-Sem Credit Eligibility",
        "📈 SGPA Segregation",
        "📝 Marks Segregation",
        "🎫 Exam / CIE Eligibility",
        "⏱️ Low Attendance List",
        "💼 Placed vs Not Placed",
        "🔍 Multi-Criteria Filter",
    ])

    with tab_pass_fail:
        st.subheader("🎯 Student Result Segregation: Passed, Borderline Pass & Failed/Backlogs")
        st.caption(
            "Segregates students into Passed vs Not Passed with dedicated identification of "
            "🌟 High Credits Distinction (Earned >= 20 credits, SGPA >= 7.5) and ⚠️ Just Border Pass (Margin <= 5 marks near 18/40 cutoff or 5.0 <= SGPA < 5.75)."
        )

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("✅ Passed Students", summary["passed_count"], f"{summary.get('pass_percentage', 0.0):.1f}% of cohort")
        p2.metric("🌟 High Credits Distinction", summary["high_credits_pass_count"], "Cleared full credits")
        p3.metric("⚠️ Just Border Pass", summary["border_pass_count"], "Intervention recommended")
        p4.metric("❌ Not Passed / Backlogs", summary["not_passed_count"], "Active backlogs")
        p5.metric("Average SGPA", f"{summary['avg_sgpa']:.2f}", f"Avg Marks: {summary['avg_marks_pct']:.1f}%")

        pass_view_mode = st.radio(
            "Filter Pass Classification",
            [
                "All Evaluated Students",
                "✅ Passed Students Only (0 Backlogs)",
                "🌟 Passed on High Credits (Distinction)",
                "⚠️ Just Border Pass (High Risk Margin)",
                "❌ Not Passed (Active Backlogs / Failed)",
            ],
            horizontal=True,
            key="seg_pass_view_mode",
        )

        if pass_view_mode == "✅ Passed Students Only (0 Backlogs)":
            view_pf = seg["passed_df"]
        elif pass_view_mode == "🌟 Passed on High Credits (Distinction)":
            view_pf = seg["pass_high_credits_df"]
        elif pass_view_mode == "⚠️ Just Border Pass (High Risk Margin)":
            view_pf = seg["border_pass_df"]
        elif pass_view_mode == "❌ Not Passed (Active Backlogs / Failed)":
            view_pf = seg["not_passed_df"]
        else:
            view_pf = seg["all_df"]

        cols_pf = [
            "usn", "student_name", "class_name", "latest_sgpa", "cgpa",
            "credits_earned", "credits_registered", "active_backlogs",
            "pass_classification", "borderline_subjects_list",
        ]
        avail_cols_pf = [c for c in cols_pf if c in view_pf.columns]
        st.dataframe(view_pf[avail_cols_pf], use_container_width=True, hide_index=True)

        d_pf1, d_pf2, d_pf3, d_pf4 = st.columns(4)
        d_pf1.download_button(
            "⬇️ Download Passed Students (CSV)",
            data=_df_csv_bytes(seg["passed_df"][avail_cols_pf] if not seg["passed_df"].empty else pd.DataFrame()),
            file_name="passed_students.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_passed_csv",
        )
        d_pf2.download_button(
            "⬇️ Download High Credits (CSV)",
            data=_df_csv_bytes(seg["pass_high_credits_df"][avail_cols_pf] if not seg["pass_high_credits_df"].empty else pd.DataFrame()),
            file_name="passed_high_credits.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_high_credits_csv",
        )
        d_pf3.download_button(
            "⬇️ Download Border Pass (CSV)",
            data=_df_csv_bytes(seg["border_pass_df"][avail_cols_pf] if not seg["border_pass_df"].empty else pd.DataFrame()),
            file_name="border_pass_students.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_border_pass_csv",
        )
        d_pf4.download_button(
            "⬇️ Download Not Passed (CSV)",
            data=_df_csv_bytes(seg["not_passed_df"][avail_cols_pf] if not seg["not_passed_df"].empty else pd.DataFrame()),
            file_name="not_passed_students.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_not_passed_csv",
        )

    with tab_sgpa:
        sgpa_df = seg["sgpa_df"]
        c_s1, c_s2 = st.columns([1.4, 2.2])
        with c_s1:
            tier_counts = sgpa_df["sgpa_tier"].value_counts().reset_index()
            tier_counts.columns = ["SGPA Band", "Students"]
            fig = px.bar(tier_counts, x="SGPA Band", y="Students", color="SGPA Band", text="Students", title="Students Segregated by SGPA Band")
            fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c_s2:
            sel_tiers = st.multiselect(
                "Filter SGPA Tier(s)",
                sorted(sgpa_df["sgpa_tier"].unique().tolist()),
                default=sorted(sgpa_df["sgpa_tier"].unique().tolist()),
                key="seg_sgpa_tier_ms",
            )
            min_sgpa_f = st.slider("Minimum SGPA", 0.0, 10.0, 0.0, 0.25, key="seg_min_sgpa")
            view_sgpa = sgpa_df[(sgpa_df["sgpa_tier"].isin(sel_tiers)) & (sgpa_df["latest_sgpa"] >= min_sgpa_f)]
            cols_s = ["usn", "student_name", "class_name", "latest_sem_num", "latest_sgpa", "cgpa", "sgpa_tier", "credits_earned", "active_backlogs"]
            st.dataframe(view_sgpa[cols_s], use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download SGPA Segregated List (CSV)",
                data=_df_csv_bytes(view_sgpa[cols_s]),
                file_name="segregated_by_sgpa.csv",
                mime="text/csv",
                key="dl_sgpa_csv",
            )

    with tab_marks:
        marks_df = seg["marks_df"]
        c_m1, c_m2 = st.columns([1.4, 2.2])
        with c_m1:
            m_counts = marks_df["marks_tier"].value_counts().reset_index()
            m_counts.columns = ["Marks Band", "Students"]
            fig_m = px.bar(m_counts, x="Marks Band", y="Students", color="Marks Band", text="Students", title="Students Segregated by Marks %")
            fig_m.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_m, use_container_width=True)
        with c_m2:
            sel_mtiers = st.multiselect(
                "Filter Marks Tier(s)",
                sorted(marks_df["marks_tier"].unique().tolist()),
                default=sorted(marks_df["marks_tier"].unique().tolist()),
                key="seg_marks_tier_ms",
            )
            min_mpct = st.slider("Minimum Marks %", 0.0, 100.0, 0.0, 5.0, key="seg_min_mpct")
            view_marks = marks_df[(marks_df["marks_tier"].isin(sel_mtiers)) & (marks_df["marks_pct"] >= min_mpct)]
            cols_m = ["usn", "student_name", "class_name", "total_marks_obtained", "total_max_marks", "marks_pct", "marks_tier", "passed_subjects", "failed_subjects"]
            st.dataframe(view_marks[cols_m], use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Marks Segregated List (CSV)",
                data=_df_csv_bytes(view_marks[cols_m]),
                file_name="segregated_by_marks.csv",
                mime="text/csv",
                key="dl_marks_csv",
            )

    with tab_credits:
        elig_choice = st.radio(
            "Next-Semester Promotion View",
            ["All Students (Credits & Status)", "✅ Eligible for Next Semester", "⛔ Not Eligible / Pending"],
            horizontal=True,
            key="seg_elig_radio",
        )
        cols_cr = [
            "usn", "student_name", "class_name", "credits_earned", "credits_registered",
            "credit_deficit", "active_backlogs", "latest_sgpa", "eligibility_status", "eligibility_reason",
        ]
        if elig_choice == "✅ Eligible for Next Semester":
            view_cr = seg["eligible_df"]
        elif elig_choice == "⛔ Not Eligible / Pending":
            view_cr = seg["not_eligible_df"]
        else:
            view_cr = seg["credits_df"]

        st.dataframe(view_cr[cols_cr], use_container_width=True, hide_index=True)
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "⬇️ Download Eligible Students (CSV)",
            data=_df_csv_bytes(seg["eligible_df"][cols_cr]),
            file_name="eligible_next_semester_students.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_elig_csv",
        )
        dl2.download_button(
            "⬇️ Download Not Eligible Students (CSV)",
            data=_df_csv_bytes(seg["not_eligible_df"][cols_cr]),
            file_name="not_eligible_next_semester_students.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_not_elig_csv",
        )
        dl3.download_button(
            "⬇️ Download Credits Ranking (CSV)",
            data=_df_csv_bytes(seg["credits_df"][cols_cr]),
            file_name="segregated_by_credits.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_creds_csv",
        )

        # 1-Click End-of-Semester Batch Promotion Tool + Revaluation History
        elig_promotable = seg["eligible_df"][seg["eligible_df"]["current_semester"] < 8] if not seg["eligible_df"].empty else pd.DataFrame()
        with st.expander(f"🚀 1-Click End-of-Semester Batch Promotion ({len(elig_promotable)} Eligible Student(s))", expanded=False):
            st.caption("Promotes all credit-eligible students below Semester 8 to their next semester and optionally locks their completed semester marksheets.")
            lock_prev = st.checkbox("Lock completed semesters upon promotion", value=True, key="prom_lock_prev")
            confirm_prom = st.checkbox("I confirm promoting eligible students in this filtered scope", value=False, key="prom_confirm_chk")
            if st.button("🎓 Promote All Eligible Students to Next Semester", type="primary", disabled=(not confirm_prom or elig_promotable.empty), key="btn_batch_promote"):
                res_prom = execute_semester_batch_promotion(elig_promotable["student_id"].tolist(), actor_id=user_id, lock_previous_sem=lock_prev)
                st.success(f"🎉 Promoted **{res_prom['promoted']}** student(s) to the next semester (skipped {res_prom['skipped']})!")
                st.rerun()

        rev_df = get_marksheet_revisions()
        if not rev_df.empty:
            with st.expander(f"🔄 VTU Marksheet Revision / Revaluation Deltas ({len(rev_df)})", expanded=False):
                st.dataframe(
                    rev_df[["usn", "student_name", "semester_number", "old_sgpa", "new_sgpa", "old_credits_earned", "new_credits_earned", "actor_type", "created_at"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_exam:
        st.caption("Tracks VTU Hall Ticket / End-Semester Exam (SEE) eligibility combining attendance condonation bands and current CIE/IA scores.")
        ex_filter = st.radio(
            "Exam Eligibility Scope",
            ["All Students", "✅ Hall Ticket Eligible / Condonable", "⛔ Detained / CIE Shortage Only"],
            horizontal=True,
            key="seg_exam_radio",
        )
        all_ex = seg["all_df"]
        if ex_filter == "✅ Hall Ticket Eligible / Condonable":
            ex_view = all_ex[all_ex["exam_eligible"] == True]
        elif ex_filter == "⛔ Detained / CIE Shortage Only":
            ex_view = all_ex[all_ex["exam_eligible"] == False]
        else:
            ex_view = all_ex
        cols_ex = [
            "usn", "student_name", "class_name", "current_semester", "attendance_pct",
            "att_condonation_band", "classes_to_recover_75", "cie_shortage_count", "exam_status", "exam_reason",
        ]
        st.dataframe(ex_view[cols_ex], use_container_width=True, hide_index=True)
        col_ex_d1, col_ex_d2 = st.columns(2)
        with col_ex_d1:
            st.download_button(
                "⬇️ Download Exam / Hall Ticket Eligibility List (CSV)",
                data=_df_csv_bytes(ex_view[cols_ex]),
                file_name="exam_hall_ticket_eligibility.csv",
                mime="text/csv",
                key="dl_exam_elig_csv",
                use_container_width=True,
            )
        with col_ex_d2:
            try:
                active_cid = scoped_class_ids[0] if (scoped_class_ids and len(scoped_class_ids) == 1) else "all"
                zip_bytes = generate_class_hall_tickets_zip(active_cid)
                st.download_button(
                    "📦 Batch Download All Class Hall Tickets (ZIP)",
                    data=zip_bytes,
                    file_name=f"VTU_Hall_Tickets_Cohort_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    key="btn_batch_dl_hall_tickets",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"Batch hall tickets note: {e}")

        if not ex_view.empty:
            with st.expander("🎫 Student VTU Hall Ticket & Remedial Mentoring Plan", expanded=True):
                stu_ht_map = {f"{r['usn']} — {r['student_name']} ({r['exam_status']})": (r["student_id"], r["usn"], r.get("exam_eligible", True)) for _, r in ex_view.iterrows()}
                ht_sel = st.selectbox("Select Student for Document Generation", list(stu_ht_map.keys()), key="ht_sel_student")
                sel_sid, sel_usn, is_elig = stu_ht_map[ht_sel]
                c_ht1, c_ht2 = st.columns(2)
                try:
                    ht_pdf = generate_vtu_hall_ticket_pdf(sel_sid)
                    c_ht1.download_button(
                        f"🎫 Download VTU Hall Ticket (PDF)",
                        data=ht_pdf,
                        file_name=f"VTU_HallTicket_{sel_usn}_Sem5.pdf",
                        mime="application/pdf",
                        key=f"btn_dl_ht_{sel_sid}",
                        use_container_width=True,
                    )
                except Exception as e:
                    c_ht1.error(f"Hall ticket error: {e}")

                try:
                    rem_pdf = generate_remedial_mentoring_plan_pdf(sel_sid)
                    c_ht2.download_button(
                        f"📋 Download Remedial Plan (PDF)",
                        data=rem_pdf,
                        file_name=f"VTU_Remedial_Plan_{sel_usn}.pdf",
                        mime="application/pdf",
                        key=f"btn_dl_rem_{sel_sid}",
                        use_container_width=True,
                    )
                except Exception as e:
                    c_ht2.error(f"Remedial plan error: {e}")

            with st.expander("📜 Official VTU Exam Shortage Notice & Parent Dispatch", expanded=False):
                shortage_stu_map = {f"{r['usn']} — {r['student_name']} ({r['exam_status']})": r["student_id"] for _, r in ex_view.iterrows()}
                ex_sel_stu = st.selectbox("Select Student for Official Notice", list(shortage_stu_map.keys()), key="ex_notif_sel_stu")
                ex_stu_id = shortage_stu_map[ex_sel_stu]
                ex_remarks = st.text_input("Custom Faculty / HOD Remarks", placeholder="e.g. Must attend special remedial classes to clear CIE deficit.", key="ex_notif_remarks")
                col_ex_p1, col_ex_p2 = st.columns(2)
                try:
                    pdf_bytes = generate_official_parent_notice_pdf(ex_stu_id, comm_type="CIE Shortage Warning", custom_remarks=ex_remarks)
                    col_ex_p1.download_button(
                        "📄 Download Official Notice (PDF)",
                        data=pdf_bytes,
                        file_name=f"VTU_Notice_{ex_stu_id[:8]}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_ex_notice_{ex_stu_id}",
                    )
                except Exception as e:
                    col_ex_p1.error(f"PDF error: {e}")

                if col_ex_p2.button("📲 Dispatch Notice to Parent & Log Communication", use_container_width=True, key=f"btn_ex_disp_{ex_stu_id}"):
                    comm_id = dispatch_parent_communication(
                        student_id=ex_stu_id,
                        faculty_id=user_id,
                        faculty_name=st.session_state.get("username", "Faculty"),
                        comm_type="CIE Shortage Warning",
                        channel="Official Notice PDF",
                        subject="VTU CIE Shortage & Hall-Ticket Intimation",
                        message_body=ex_remarks or "Official warning notice regarding CIE shortage.",
                    )
                    st.success(f"✅ Dispatched notice and logged communication (Ref: {comm_id[:8]})!")

    with tab_att:
        low_att_df = seg["low_attendance_df"]
        show_all_att = st.checkbox("Also include students with Satisfactory Attendance", value=False, key="seg_show_all_att")
        att_view = seg["all_df"].sort_values(by=["attendance_pct", "usn"]) if show_all_att else low_att_df
        cols_a = [
            "usn", "student_name", "class_name", "present_hours", "total_hours",
            "attendance_pct", "classes_to_recover_75", "att_condonation_band",
            "attendance_status", "eligibility_status", "email", "phone",
        ]
        if att_view.empty:
            st.success(f"🎉 Zero students below {active_rules['low_attendance_threshold']:g}% attendance in this scope!")
        else:
            st.warning(f"⚠️ **{len(low_att_df)} student(s)** currently below **{active_rules['low_attendance_threshold']:g}%** attendance threshold (`classes_to_recover_75` shows consecutive classes needed).")
            st.dataframe(att_view[cols_a], use_container_width=True, hide_index=True)
            ac1, ac2 = st.columns(2)
            ac1.download_button(
                "⬇️ Download Low Attendance List (CSV)",
                data=_df_csv_bytes(low_att_df[cols_a]),
                file_name="low_attendance_students.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_low_att_csv",
            )
            if ac2.button("🔔 Send Low Attendance Alert + Recovery Target", use_container_width=True, key="notify_low_att_btn"):
                sent = 0
                for _, sr in low_att_df.iterrows():
                    need_cls = int(sr.get("classes_to_recover_75", 0))
                    rec_txt = f" Attend the next {need_cls} consecutive class(es) to reach {active_rules['low_attendance_threshold']:g}%." if need_cls > 0 else ""
                    send_notification(
                        "student",
                        sr["student_id"],
                        "Low Attendance Warning",
                        f"Your attendance is currently {sr['attendance_pct']:.1f}% (below required {active_rules['low_attendance_threshold']:g}%).{rec_txt} Meet your Class Teacher.",
                        category="Attendance Alert",
                    )
                    sent += 1
                log_audit_action("staff", user_id, "LOW_ATTENDANCE_ALERT", f"Sent attendance alerts to {sent} students")
                st.success(f"✅ Sent recovery-target alerts to {sent} student(s)!")

            if not low_att_df.empty:
                with st.expander("🤝 Quick Mentoring / Parent Call Logger for Low Attendance Student", expanded=False):
                    q_stu_map = {f"{r['usn']} — {r['student_name']} ({r['attendance_pct']:.1f}%)": r["student_id"] for _, r in low_att_df.iterrows()}
                    with st.form("quick_att_mentoring_form"):
                        m_stu_lbl = st.selectbox("Select Flagged Student", list(q_stu_map.keys()))
                        mc1, mc2 = st.columns(2)
                        m_action = mc1.text_input("Action Taken", placeholder="Called parent / Scheduled counseling")
                        m_parent = mc2.checkbox("Parent Contacted", value=True)
                        m_notes = st.text_area("Counseling Notes *", placeholder="Student reason and attendance recovery commitment...")
                        if st.form_submit_button("💾 Save Mentoring Log", type="primary"):
                            try:
                                add_mentoring_log(
                                    student_id=q_stu_map[m_stu_lbl],
                                    faculty_id=user_id,
                                    faculty_name=st.session_state.get("username", "Faculty"),
                                    category="Low Attendance",
                                    notes=m_notes,
                                    action_taken=m_action,
                                    parent_notified=m_parent,
                                    notify_student=True,
                                )
                                st.success("✅ Mentoring & parent-contact log saved!")
                            except ValueError as ve:
                                st.error(str(ve))

                with st.expander("📜 Official VTU Attendance Shortage Notice & Parent Dispatch", expanded=False):
                    att_stu_map = {f"{r['usn']} — {r['student_name']} ({r['attendance_pct']:.1f}%)": r["student_id"] for _, r in low_att_df.iterrows()}
                    att_sel_stu = st.selectbox("Select Student for Parent Notice", list(att_stu_map.keys()), key="att_notif_sel_stu")
                    att_stu_id = att_stu_map[att_sel_stu]
                    att_remarks = st.text_input("Custom Faculty Remarks", placeholder="e.g. Student must attend morning remedial hours without fail.", key="att_notif_remarks")
                    col_att_p1, col_att_p2 = st.columns(2)
                    try:
                        att_pdf_bytes = generate_official_parent_notice_pdf(att_stu_id, comm_type="Attendance Warning", custom_remarks=att_remarks)
                        col_att_p1.download_button(
                            "📄 Download Official Attendance Warning (PDF)",
                            data=att_pdf_bytes,
                            file_name=f"VTU_Attendance_Notice_{att_stu_id[:8]}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"dl_att_notice_{att_stu_id}",
                        )
                    except Exception as e:
                        col_att_p1.error(f"PDF error: {e}")

                    if col_att_p2.button("📲 Dispatch Attendance Notice & Log", use_container_width=True, key=f"btn_att_disp_{att_stu_id}"):
                        comm_id = dispatch_parent_communication(
                            student_id=att_stu_id,
                            faculty_id=user_id,
                            faculty_name=st.session_state.get("username", "Faculty"),
                            comm_type="Attendance Warning",
                            channel="Official Notice PDF",
                            subject="VTU Attendance Shortage Warning Notice",
                            message_body=att_remarks or "Official attendance warning notice with recovery target.",
                        )
                        st.success(f"✅ Dispatched attendance warning and logged communication (Ref: {comm_id[:8]})!")

    with tab_place:
        placed_df = seg["placed_df"]
        not_placed_df = seg["not_placed_df"]
        p_filter = st.radio(
            "Placement List Filter",
            ["All Students", "✅ Placed Students List", "📋 Not Placed Students List", "🎯 Eligible & Not Placed (Target Pool)"],
            horizontal=True,
            key="seg_place_radio",
        )
        if p_filter == "✅ Placed Students List":
            p_view = placed_df
        elif p_filter == "📋 Not Placed Students List":
            p_view = not_placed_df
        elif p_filter == "🎯 Eligible & Not Placed (Target Pool)":
            p_view = not_placed_df[not_placed_df["next_sem_eligible"] == True]
        else:
            p_view = seg["all_df"]

        cols_p = [
            "usn", "student_name", "class_name", "cgpa", "latest_sgpa",
            "credits_earned", "active_backlogs", "attendance_pct", "eligibility_status",
            "placement_status", "placement_companies", "highest_package_lpa", "offers_count",
        ]
        st.dataframe(p_view[cols_p], use_container_width=True, hide_index=True)
        pd1, pd2 = st.columns(2)
        pd1.download_button(
            "⬇️ Download Placed Students List (CSV)",
            data=_df_csv_bytes(placed_df[cols_p]),
            file_name="placed_students_list.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_placed_csv",
        )
        pd2.download_button(
            "⬇️ Download Not Placed Students List (CSV)",
            data=_df_csv_bytes(not_placed_df[cols_p]),
            file_name="not_placed_students_list.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_not_placed_csv",
        )

    with tab_custom:
        st.caption("Combine attendance, next-semester eligibility, placement status, and SGPA rules to generate exact target lists.")
        cf1, cf2, cf3, cf4 = st.columns(4)
        q_att = cf1.selectbox("Attendance Filter", ["Any", "Low Attendance Only", "Satisfactory Only"], key="cf_att")
        q_elig = cf2.selectbox("Next-Sem Eligibility", ["Any", "Eligible Only", "Not Eligible Only"], key="cf_elig")
        q_pl = cf3.selectbox("Placement Status", ["Any", "Placed Only", "Not Placed Only"], key="cf_pl")
        q_sgpa = cf4.number_input("Min CGPA", min_value=0.0, max_value=10.0, value=0.0, step=0.5, key="cf_cgpa")

        filtered = seg["all_df"].copy()
        if q_att == "Low Attendance Only":
            filtered = filtered[filtered["is_low_attendance"] == True]
        elif q_att == "Satisfactory Only":
            filtered = filtered[filtered["is_low_attendance"] == False]
        if q_elig == "Eligible Only":
            filtered = filtered[filtered["next_sem_eligible"] == True]
        elif q_elig == "Not Eligible Only":
            filtered = filtered[filtered["next_sem_eligible"] == False]
        if q_pl == "Placed Only":
            filtered = filtered[filtered["is_placed"] == True]
        elif q_pl == "Not Placed Only":
            filtered = filtered[filtered["is_placed"] == False]
        if q_sgpa > 0:
            filtered = filtered[filtered["cgpa"] >= q_sgpa]

        st.info(f"Matched **{len(filtered)}** student(s).")
        cols_cust = [
            "usn", "student_name", "class_name", "latest_sgpa", "cgpa", "marks_pct",
            "credits_earned", "active_backlogs", "attendance_pct", "eligibility_status",
            "placement_status", "placement_companies", "highest_package_lpa",
        ]
        st.dataframe(filtered[cols_cust], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Filtered Custom List (CSV)",
            data=_df_csv_bytes(filtered[cols_cust]),
            file_name="custom_segregated_students.csv",
            mime="text/csv",
            key="dl_custom_seg_csv",
        )


# ============================================================
# 2. FACULTY INDIVIDUAL STUDENT RECORD ENTRY & BATCH MARKSHEET OCR
# ============================================================

def render_individual_record_entry_page(
    db_getter,
    upload_dir: str,
    staff_classes: List[Any],
    user_id: str,
    username: str,
) -> None:
    """Allow faculty to individually enter/edit student results, IA marks, attendance & batch scan VTU PDFs."""
    st.title("✍️ Individual Student Record Entry & Automated Marksheet Scan")
    st.caption(
        "Enter or update any individual student's VTU Semester Result (manually or via PDF/Image OCR), "
        "Current IA Marks, and Attendance Summary — or batch-process multiple VTU marksheets automatically."
    )

    tab_ind, tab_batch_ia, tab_batch_pdf = st.tabs([
        "👤 Individual Student Entry",
        "👥 Class-Wide Batch IA & Assignment Entry",
        "🤖 Automated Multi-PDF VTU Marksheet Batch Import",
    ])

    with tab_batch_pdf:
        st.subheader("Automated Multi-PDF / Image VTU Marksheet Batch Processor")
        st.info(
            "Upload one or many VTU marksheet PDFs/images at once. The OCR engine automatically detects USN, "
            "Semester, Subject Marks, and Official SGPA, matches students, syncs backlogs, and updates Dashboard segregation."
        )
        bc1, bc2 = st.columns([2, 1])
        batch_files = bc1.file_uploader(
            "Upload VTU Marksheets or ZIP Archive (PDF, PNG, JPG, JPEG, ZIP)",
            type=["pdf", "png", "jpg", "jpeg", "zip"],
            accept_multiple_files=True,
            key="batch_ms_uploader",
        )
        fallback_sem = bc2.selectbox("Fallback Semester (if not found on PDF)", list(range(1, 9)), index=4, key="batch_fallback_sem")

        if batch_files and st.button("⚡ Scan & Match All Uploaded Marksheets", type="primary", key="btn_run_batch_ms"):
            ms_dir = os.path.join(upload_dir, "marksheets", "faculty_batch")
            os.makedirs(ms_dir, exist_ok=True)
            saved_entries: List[tuple] = []
            for bf in batch_files:
                safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{bf.name}"
                dest = os.path.join(ms_dir, safe_name)
                with open(dest, "wb") as f:
                    f.write(bf.getbuffer())
                saved_entries.append((bf.name, dest))

            expanded_entries = expand_marksheet_uploads_with_zip(
                saved_entries, os.path.join(ms_dir, "zip_extracted")
            )
            with st.spinner(f"Scanning {len(expanded_entries)} marksheet(s) via OCR..."):
                parsed = batch_parse_vtu_marksheet_files(expanded_entries, default_sem=fallback_sem)
            st.session_state["faculty_batch_ms_parsed"] = parsed

        batch_parsed = st.session_state.get("faculty_batch_ms_parsed", [])
        if batch_parsed:
            preview_df = pd.DataFrame([
                {
                    "Filename": r["filename"],
                    "Detected USN": r["detected_usn"] or "NOT DETECTED",
                    "Matched Student": r["student_name"],
                    "Sem": r["semester_number"],
                    "Subjects": r["subject_count"],
                    "SGPA": r["sgpa"],
                    "Earned Credits": r["credits_earned"],
                    "Ready": "✅ Yes" if (r["matched"] and r["subject_count"] > 0) else "⚠️ Needs Review",
                }
                for r in batch_parsed
            ])
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            ready_items = [r for r in batch_parsed if r["matched"] and r["subject_count"] > 0]
            if ready_items and st.button(f"🚀 Confirm & Auto-Save {len(ready_items)} Matched Result(s)", type="primary", use_container_width=True, key="btn_commit_batch_ms"):
                saved_cnt = 0
                for item in ready_items:
                    save_student_semester_result(
                        student_id=item["student_id"],
                        sem_num=item["semester_number"],
                        subjects=item["subjects"],
                        official_sgpa=item["sgpa"],
                        marksheet_path=item["saved_path"],
                        verification_status="Verified",
                        actor_type="staff",
                        actor_id=user_id,
                        notify_faculty=False,
                        notify_student=True,
                    )
                    saved_cnt += 1
                st.session_state.pop("faculty_batch_ms_parsed", None)
                st.success(f"🎉 Auto-saved {saved_cnt} VTU semester marksheet(s) and updated Dashboard segregation!")
                st.rerun()

    with tab_batch_ia:
        st.subheader("👥 Class-Wide Batch IA Tests & Assignment Marks Entry")
        st.caption("Enter or edit IA1, IA2, IA3, Assignment 1, and Assignment 2 marks for all students in a class at once.")

        if not staff_classes:
            st.info("No classes assigned to your faculty profile.")
        else:
            b_c1, b_c2, b_c3 = st.columns([1.5, 1.2, 1.8])
            cls_map = {c["class_name"]: c["id"] for c in staff_classes}
            sel_b_cls = b_c1.selectbox("Select Class", list(cls_map.keys()), key="batch_ia_cls")
            sel_b_sem = b_c2.selectbox("Semester", list(range(1, 9)), index=4, key="batch_ia_sem")
            b_cls_id = cls_map[sel_b_cls]

            db = db_getter()
            b_subs = db.execute(
                "SELECT subject_code, subject_name, credits FROM master_subjects WHERE (class_id = ? OR class_id IS NULL) AND semester_number = ? ORDER BY subject_code ASC",
                (b_cls_id, sel_b_sem),
            ).fetchall()
            b_students = db.execute("SELECT id, usn, name FROM students WHERE class_id = ? ORDER BY usn ASC", (b_cls_id,)).fetchall()
            db.close()

            sub_dict = {f"{r['subject_code']} — {r['subject_name']}": (r["subject_code"], r["subject_name"], float(r["credits"] or 3.0)) for r in b_subs}
            for k_code, (v_name, v_cred) in list(VTU_KNOWN_SUBJECTS.items())[:20]:
                lbl = f"{k_code} — {v_name}"
                if lbl not in sub_dict:
                    sub_dict[lbl] = (k_code, v_name, float(v_cred))

            sel_b_sub = b_c3.selectbox("Select Subject", list(sub_dict.keys()), key="batch_ia_sub")
            b_code, b_name, b_cred = sub_dict[sel_b_sub]

            if not b_students:
                st.warning(f"No students enrolled in class {sel_b_cls}.")
            else:
                db = db_getter()
                existing_ia = db.execute(
                    """
                    SELECT student_id, ia1, ia2,
                           COALESCE(ia3, 0) AS ia3,
                           COALESCE(assignment_1, 0) AS assignment_1,
                           COALESCE(assignment_2, 0) AS assignment_2,
                           COALESCE(cie_total, internal_performance) AS cie_total
                    FROM current_semester_records
                    WHERE semester_number = ? AND UPPER(subject_code) = ?
                    """,
                    (sel_b_sem, b_code.upper()),
                ).fetchall()
                db.close()

                exist_map = {r["student_id"]: dict(r) for r in existing_ia}
                table_rows = []
                for s in b_students:
                    ex = exist_map.get(s["id"], {})
                    table_rows.append({
                        "student_id": s["id"],
                        "USN": s["usn"],
                        "Student Name": s["name"],
                        "IA 1 (Max 50)": float(ex.get("ia1", 20.0)),
                        "IA 2 (Max 50)": float(ex.get("ia2", 20.0)),
                        "IA 3 (Optional)": float(ex.get("ia3", 0.0)),
                        "Assignment 1 (Max 10)": float(ex.get("assignment_1", 8.0)),
                        "Assignment 2 (Max 10)": float(ex.get("assignment_2", 8.0)),
                    })

                st.caption(f"Editing marks for **{len(b_students)} students** in **{sel_b_cls}** — Subject: **{b_code} ({b_name})**")
                edited_batch_df = st.data_editor(
                    pd.DataFrame(table_rows),
                    disabled=["student_id", "USN", "Student Name"],
                    use_container_width=True,
                    hide_index=True,
                    key=f"editor_batch_ia_{b_cls_id}_{b_code}",
                )

                if st.button(f"💾 Save All Students' IA & Assignment Marks ({len(b_students)} Students)", type="primary", use_container_width=True, key="btn_save_batch_ia"):
                    records_to_save = []
                    for _, row in edited_batch_df.iterrows():
                        records_to_save.append({
                            "student_id": row["student_id"],
                            "ia1": float(row.get("IA 1 (Max 50)", 0.0)),
                            "ia2": float(row.get("IA 2 (Max 50)", 0.0)),
                            "ia3": float(row.get("IA 3 (Optional)", 0.0)),
                            "assignment_1": float(row.get("Assignment 1 (Max 10)", 0.0)),
                            "assignment_2": float(row.get("Assignment 2 (Max 10)", 0.0)),
                        })
                    saved_n = bulk_save_class_ia_and_assignments(
                        class_id=b_cls_id,
                        semester_number=sel_b_sem,
                        subject_code=b_code,
                        subject_name=b_name,
                        credits=b_cred,
                        records=records_to_save,
                        actor_id=user_id,
                    )
                    st.success(f"🎉 Successfully saved IA & Assignment marks for **{saved_n} students** in **{b_code}**! CIE scores and Exam Eligibility updated automatically.")
                    st.rerun()

    with tab_ind:
        class_ids = [c["id"] for c in staff_classes] if staff_classes else None
        ana_df = get_comprehensive_student_analytics(class_ids=class_ids)
        if ana_df.empty:
            st.warning("No students available in your assigned classes yet.")
            return

        student_labels = {
            f"{r['usn']} — {r['student_name']} ({r['class_name']})": r["student_id"]
            for _, r in ana_df.iterrows()
        }
        sel_label = st.selectbox("Select Student to Enter / Edit Records", list(student_labels.keys()), key="ind_sel_student")
        target_id = student_labels[sel_label]
        s_row = ana_df[ana_df["student_id"] == target_id].iloc[0]

        # Snapshot strip
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Current Sem", int(s_row["current_semester"]))
        m2.metric("CGPA / Latest SGPA", f"{s_row['cgpa']:.2f} / {s_row['latest_sgpa']:.2f}")
        m3.metric("Credits Earned", f"{s_row['credits_earned']:g} / {s_row['credits_registered']:g}")
        m4.metric("Active Backlogs", int(s_row["active_backlogs"]))
        m5.metric("Attendance %", f"{s_row['attendance_pct']:.1f}%")
        m6.metric("Next Sem / Placement", f"{s_row['eligibility_status']}", s_row["placement_status"])

        sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs([
            "📄 Final VTU Semester Result",
            "📝 Current Semester IA Marks",
            "⏱️ Attendance & Profile Summary",
            "🤝 Mentoring & Counseling Log",
        ])

        with sub_t1:
            sc1, sc2, sc3 = st.columns([1.2, 2.0, 1.4])
            sem_num = sc1.selectbox("Semester Number", list(range(1, 9)), index=max(0, min(7, int(s_row["current_semester"]) - 1)), key="ind_vtu_sem")
            up_ms = sc2.file_uploader(
                "Optional: Upload Student VTU Marksheet (PDF/Image) to Auto-Fill",
                type=["pdf", "png", "jpg", "jpeg"],
                key="ind_vtu_file",
            )
            ind_state_key = f"ind_vtu_rows_{target_id}_{sem_num}"
            ind_sgpa_key = f"ind_vtu_sgpa_{target_id}_{sem_num}"
            ind_path_key = f"ind_vtu_path_{target_id}_{sem_num}"

            if sc3.button("📥 Load Saved / Template Subjects", use_container_width=True, key="ind_load_sub_btn"):
                db = db_getter()
                existing_subs = db.execute(
                    """
                    SELECT sub.subject_code, sub.subject_name, sub.internal_marks, sub.external_marks, sub.total_marks, sub.credits, sub.result
                    FROM subjects sub
                    JOIN semesters sem ON sub.semester_id = sem.id
                    WHERE sem.student_id = ? AND sem.semester_number = ?
                    ORDER BY sub.subject_code ASC
                    """,
                    (target_id, sem_num),
                ).fetchall()
                existing_sem = db.execute(
                    "SELECT sgpa FROM semesters WHERE student_id = ? AND semester_number = ?",
                    (target_id, sem_num),
                ).fetchone()
                master_subs = db.execute(
                    """
                    SELECT subject_code, subject_name, credits FROM master_subjects
                    WHERE (class_id = ? OR class_id IS NULL) AND semester_number = ?
                    ORDER BY subject_code ASC
                    """,
                    (s_row["class_id"], sem_num),
                ).fetchall()
                db.close()

                if existing_subs:
                    st.session_state[ind_state_key] = [
                        {
                            "subject_code": r["subject_code"],
                            "subject_name": r["subject_name"],
                            "internal_marks": float(r["internal_marks"] or 0.0),
                            "external_marks": float(r["external_marks"] or 0.0),
                            "credits": float(r["credits"] or 3.0),
                            "result": "P" if str(r["result"]).upper() in ("P", "PASS") else "F",
                        }
                        for r in existing_subs
                    ]
                    if existing_sem and existing_sem["sgpa"]:
                        st.session_state[ind_sgpa_key] = float(existing_sem["sgpa"])
                elif master_subs:
                    st.session_state[ind_state_key] = [
                        {
                            "subject_code": r["subject_code"],
                            "subject_name": r["subject_name"],
                            "internal_marks": 0.0,
                            "external_marks": 0.0,
                            "credits": float(r["credits"] or 3.0),
                            "result": "P",
                        }
                        for r in master_subs
                    ]
                else:
                    st.session_state[ind_state_key] = [
                        {"subject_code": "BCS501", "subject_name": "Software Engineering & Project Management", "internal_marks": 40.0, "external_marks": 40.0, "credits": 4.0, "result": "P"}
                    ]
                st.rerun()

            if up_ms is not None and st.session_state.get(f"ind_last_file_{target_id}_{sem_num}") != up_ms.name:
                ms_dir = os.path.join(upload_dir, "marksheets", target_id)
                os.makedirs(ms_dir, exist_ok=True)
                dest = os.path.join(ms_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{up_ms.name}")
                with open(dest, "wb") as f:
                    f.write(up_ms.getbuffer())
                raw_txt, det_subs, _ = process_marksheet_file(dest)
                meta = extract_vtu_metadata(dest, raw_text=raw_txt)
                st.session_state[ind_path_key] = dest
                st.session_state[f"ind_last_file_{target_id}_{sem_num}"] = up_ms.name
                if det_subs:
                    st.session_state[ind_state_key] = [
                        {
                            "subject_code": x.get("subject_code", ""),
                            "subject_name": x.get("subject_name", ""),
                            "internal_marks": float(x.get("internal", 0.0)),
                            "external_marks": float(x.get("external", 0.0)),
                            "credits": float(x.get("credits", 3.0)),
                            "result": "P" if str(x.get("result", "P")).upper() in ("P", "PASS") else "F",
                        }
                        for x in det_subs
                    ]
                if meta.get("official_sgpa"):
                    st.session_state[ind_sgpa_key] = float(meta["official_sgpa"])
                st.success(f"✅ OCR extracted {len(det_subs)} subject(s) from `{up_ms.name}`!")

            if ind_state_key not in st.session_state:
                st.session_state[ind_state_key] = [
                    {"subject_code": "", "subject_name": "", "internal_marks": 0.0, "external_marks": 0.0, "credits": 3.0, "result": "P"}
                ]

            st.caption("Edit marks/credits below or add rows (`+`). Totals, Grades, SGPA & Backlogs compute automatically on save.")
            edited_df = st.data_editor(
                pd.DataFrame(st.session_state[ind_state_key]),
                num_rows="dynamic",
                use_container_width=True,
                key=f"ind_editor_{target_id}_{sem_num}",
                column_config={
                    "subject_code": st.column_config.TextColumn("Subject Code"),
                    "subject_name": st.column_config.TextColumn("Subject Name"),
                    "internal_marks": st.column_config.NumberColumn("Internal", min_value=0.0, max_value=100.0, step=1.0),
                    "external_marks": st.column_config.NumberColumn("External", min_value=0.0, max_value=100.0, step=1.0),
                    "credits": st.column_config.NumberColumn("Credits", min_value=0.0, max_value=10.0, step=0.5),
                    "result": st.column_config.SelectboxColumn("Result (P/F)", options=["P", "F"]),
                },
            )
            off_sgpa_inp = st.number_input(
                "Official SGPA Override (leave 0.00 to auto-calculate from credits & marks)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get(ind_sgpa_key, 0.0)),
                step=0.01,
                format="%.2f",
                key=f"ind_sgpa_num_{target_id}_{sem_num}",
            )

            if st.button("💾 Save Student VTU Result & Update Segregation", type="primary", use_container_width=True, key="ind_save_vtu_btn"):
                clean_subs = []
                for _, er in edited_df.iterrows():
                    code = str(er.get("subject_code") or "").strip().upper()
                    if not code:
                        continue
                    can_code = normalize_subject_code(code)
                    can_name, can_cred = get_canonical_subject_info(can_code, str(er.get("subject_name") or ""))
                    intl = float(er.get("internal_marks") or 0.0)
                    extl = float(er.get("external_marks") or 0.0)
                    clean_subs.append(
                        {
                            "subject_code": can_code,
                            "subject_name": str(er.get("subject_name") or "").strip() or can_name,
                            "internal_marks": intl,
                            "external_marks": extl,
                            "total_marks": round(intl + extl, 2),
                            "credits": float(er.get("credits") if er.get("credits") is not None else can_cred),
                            "result": str(er.get("result") or "P"),
                        }
                    )
                if not clean_subs:
                    st.error("Please enter at least one valid Subject Code.")
                else:
                    res_info = save_student_semester_result(
                        student_id=target_id,
                        sem_num=sem_num,
                        subjects=clean_subs,
                        official_sgpa=off_sgpa_inp if off_sgpa_inp > 0 else None,
                        marksheet_path=st.session_state.get(ind_path_key, ""),
                        verification_status="Verified",
                        actor_type="staff",
                        actor_id=user_id,
                        notify_faculty=False,
                        notify_student=True,
                    )
                    st.success(
                        f"🎉 Saved Sem {sem_num} VTU result for **{s_row['usn']}** | "
                        f"SGPA: **{res_info['sgpa']:.2f}** | Earned Credits: **{res_info['credits_earned']:g}** | "
                        f"Backlogs Synced: +{res_info['backlogs_activated']} / Cleared: {res_info['backlogs_cleared']}"
                    )
                    st.rerun()

        with sub_t2:
            st.subheader("Individual Current Semester IA Tests & Assignment Marks Entry")
            st.caption("Enter Internal Assessment test scores (IA1, IA2, IA3) and Assignment marks (A1, A2). Total CIE out of 50 and VTU Exam eligibility (>= 20/50) compute automatically.")

            db = db_getter()
            ms_rows = db.execute(
                "SELECT subject_code, subject_name, credits, semester_number FROM master_subjects ORDER BY semester_number ASC, subject_code ASC"
            ).fetchall()
            cur_ia_rows = db.execute(
                """
                SELECT semester_number, subject_code, subject_name, credits, ia1, ia2,
                       COALESCE(ia3, 0) AS ia3,
                       COALESCE(assignment_1, 0) AS assignment_1,
                       COALESCE(assignment_2, 0) AS assignment_2,
                       COALESCE(assignment_marks, 0) AS assignment_marks,
                       COALESCE(cie_total, internal_performance) AS cie_total,
                       internal_performance
                FROM current_semester_records
                WHERE student_id = ?
                ORDER BY semester_number DESC, subject_code ASC
                """,
                (target_id,),
            ).fetchall()
            db.close()

            subject_choices = {f"{r['subject_code']} — {r['subject_name']}": (r["subject_code"], r["subject_name"], float(r["credits"] or 3.0)) for r in ms_rows}
            for k_code, (v_name, v_cred) in list(VTU_KNOWN_SUBJECTS.items())[:25]:
                lbl = f"{k_code} — {v_name}"
                if lbl not in subject_choices:
                    subject_choices[lbl] = (k_code, v_name, float(v_cred))

            with st.form("ind_ia_entry_form"):
                ic1, ic2, ic3 = st.columns(3)
                ia_sem = ic1.number_input("Semester", min_value=1, max_value=8, value=int(s_row.get("current_semester", 1) or 1))
                sel_sub_opt = ic2.selectbox("Subject Preset (optional)", ["Custom Entry"] + list(subject_choices.keys()))
                default_code, default_name, default_cred = subject_choices.get(sel_sub_opt, ("", "", 3.0))
                ia_code = ic3.text_input("Subject Code *", value=default_code)

                ic4, ic5 = st.columns([3, 1])
                ia_name = ic4.text_input("Subject Name *", value=default_name)
                ia_credits = ic5.number_input("Credits", min_value=0.0, max_value=10.0, value=default_cred, step=0.5)

                st.markdown("##### 📝 Internal Assessment (IA) Tests")
                t_col1, t_col2, t_col3 = st.columns(3)
                ia1_v = t_col1.number_input("IA 1 Test Marks", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
                ia2_v = t_col2.number_input("IA 2 Test Marks", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
                ia3_v = t_col3.number_input("IA 3 Test Marks (Optional / Best 2)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

                st.markdown("##### 📚 Assignment & Continuous Assessment Components")
                a_col1, a_col2, a_col3 = st.columns(3)
                a1_v = a_col1.number_input("Assignment 1 Marks (out of 10)", min_value=0.0, max_value=25.0, value=8.5, step=0.5)
                a2_v = a_col2.number_input("Assignment 2 Marks (out of 10)", min_value=0.0, max_value=25.0, value=8.5, step=0.5)
                override_cie = a_col3.number_input("CIE Total Override (0 to auto-calculate)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

                if st.form_submit_button("💾 Save Student IA & Assignment Marks", type="primary"):
                    if not ia_code.strip():
                        st.error("Subject Code is required.")
                    else:
                        ia_res = save_student_ia_subject_mark(
                            student_id=target_id,
                            semester_number=int(ia_sem),
                            subject_code=ia_code.strip().upper(),
                            subject_name=ia_name.strip() or ia_code.strip().upper(),
                            credits=ia_credits,
                            ia1=ia1_v,
                            ia2=ia2_v,
                            ia3=ia3_v,
                            assignment_1=a1_v,
                            assignment_2=a2_v,
                            cie_total=override_cie if override_cie > 0 else None,
                            actor_id=user_id,
                        )
                        st.success(
                            f"✅ Saved IA & Assignments for {s_row['usn']} — {ia_code.upper()} | "
                            f"Total CIE: **{ia_res['cie_total']:.1f}/50** "
                            f"({'✅ Exam Eligible' if ia_res['cie_eligible'] else '⚠️ CIE Shortage'})"
                        )
                        st.rerun()

            if cur_ia_rows:
                st.markdown("##### 📋 Recorded IA & Assignment Marks for Student")
                st.dataframe(pd.DataFrame([dict(r) for r in cur_ia_rows]), use_container_width=True, hide_index=True)

        with sub_t3:
            st.subheader("Individual Attendance Summary & Academic Flags")
            with st.form("ind_att_profile_form"):
                ac1, ac2, ac3, ac4 = st.columns(4)
                upd_sem = ac1.number_input("Current Semester", min_value=1, max_value=8, value=int(s_row.get("current_semester", 1) or 1))
                upd_work = ac2.number_input("Working Days / Held Classes", min_value=0, max_value=500, value=int(s_row.get("working_days", 90) or 90))
                upd_abs = ac3.number_input("Absent Days / Missed Classes", min_value=0, max_value=500, value=int(s_row.get("absent_days", 0) or 0))
                upd_lat = ac4.checkbox("Lateral Entry Student", value=bool(s_row.get("is_lateral_entry", 0)))
                if st.form_submit_button("💾 Save Student Attendance & Semester Profile", type="primary"):
                    update_student_attendance_summary(
                        student_id=target_id,
                        working_days=int(upd_work),
                        absent_days=int(upd_abs),
                        current_semester=int(upd_sem),
                        is_lateral_entry=1 if upd_lat else 0,
                        actor_id=user_id,
                    )
                    st.success(f"✅ Updated attendance & profile for {s_row['usn']}!")
                    st.rerun()

        with sub_t4:
            st.subheader(f"Mentoring & Counseling History — {s_row['student_name']} ({s_row['usn']})")
            with st.form("ind_mentoring_form"):
                mf1, mf2 = st.columns(2)
                m_cat = mf1.selectbox("Intervention Category", ["Attendance / Shortage", "Academic Backlogs", "SGPA Improvement", "Placement Counseling", "Personal / Parent Call"])
                m_act = mf2.text_input("Action Taken", placeholder="Assigned remedial classes / Spoken to parent")
                m_par = st.checkbox("Parent Notified", value=False)
                m_note = st.text_area("Counseling Notes *", placeholder="Record mentoring conversation and action plan...")
                if st.form_submit_button("💾 Record Mentoring Entry", type="primary"):
                    try:
                        add_mentoring_log(
                            student_id=target_id,
                            faculty_id=user_id,
                            faculty_name=username,
                            category=m_cat,
                            notes=m_note,
                            action_taken=m_act,
                            parent_notified=m_par,
                            notify_student=True,
                        )
                        st.success("✅ Mentoring entry logged and student notified!")
                        st.rerun()
                    except ValueError as ve:
                        st.error(str(ve))

            stu_logs = get_mentoring_logs(target_id)
            if stu_logs.empty:
                st.info("No mentoring logs recorded for this student yet.")
            else:
                st.dataframe(
                    stu_logs[["created_at", "faculty_name", "category", "action_taken", "notes", "parent_notified"]],
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# 3. PLACEMENT FACULTY & INSTITUTIONAL PLACEMENT MANAGEMENT PAGE
# ============================================================

def render_placement_management_page(
    upload_dir: str,
    staff_classes: List[Any],
    user_id: str,
    username: str,
    staff_role: str = "placement",
) -> None:
    """Complete Placement Hub: individual entry (with Offer OCR), bulk Excel import, Placed vs Not Placed lists."""
    st.title("💼 Placement Management & Career Records")
    st.caption(
        "Enter, verify, and publish individual or bulk student placement records. "
        "All placement lists are visible to Faculty & Admin across dashboards, and each student sees their personal offers in their Student Portal."
    )

    ana_df = get_comprehensive_student_analytics(class_ids=None)
    all_pl_df = get_all_placements(class_ids=None)
    seg = segregate_students(ana_df)

    placed_cnt = seg["summary"]["placed_count"]
    not_placed_cnt = seg["summary"]["not_placed_count"]
    total_cnt = seg["summary"]["total_students"]
    place_rate = round((placed_cnt / total_cnt) * 100.0, 1) if total_cnt > 0 else 0.0
    max_pkg = round(float(ana_df["highest_package_lpa"].max()), 2) if not ana_df.empty else 0.0
    placed_pkgs = ana_df[ana_df["highest_package_lpa"] > 0]["highest_package_lpa"] if not ana_df.empty else pd.Series(dtype=float)
    avg_pkg = round(float(placed_pkgs.mean()), 2) if not placed_pkgs.empty else 0.0

    # Salary tier segmentation
    super_dream = int((ana_df["highest_package_lpa"] >= 10.0).sum()) if not ana_df.empty else 0
    dream = int(((ana_df["highest_package_lpa"] >= 6.0) & (ana_df["highest_package_lpa"] < 10.0)).sum()) if not ana_df.empty else 0
    regular = int(((ana_df["highest_package_lpa"] > 0.0) & (ana_df["highest_package_lpa"] < 6.0)).sum()) if not ana_df.empty else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Cohort", total_cnt)
    k2.metric("Placed Students", placed_cnt, f"{place_rate}% Placed")
    k3.metric("Not Placed Students", not_placed_cnt)
    k4.metric("Highest Package", f"{max_pkg:.2f} LPA")
    k5.metric("Average Package", f"{avg_pkg:.2f} LPA")

    # Salary Tiers Banner Strip
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; display: flex; gap: 20px; font-size: 13px;">
        <span>🚀 <b>Super Dream (&ge;10 LPA):</b> <span style="color: #7C3AED; font-weight: 700;">{super_dream}</span></span>
        <span>⭐ <b>Dream (6-10 LPA):</b> <span style="color: #2563EB; font-weight: 700;">{dream}</span></span>
        <span>💼 <b>Standard (&lt;6 LPA):</b> <span style="color: #059669; font-weight: 700;">{regular}</span></span>
    </div>
    """, unsafe_allow_html=True)

    t_entry, t_matcher, t_ats, t_lists, t_readiness, t_bulk, t_analytics = st.tabs([
        "➕ Enter / Verify Placements",
        "🎯 Drive Matcher & Shortlist",
        "🚀 Drive ATS & Pipeline",
        "📋 Placed vs Not Placed Lists",
        "🤖 AI Placement Readiness",
        "📥 Automated Bulk Import",
        "📊 Company & Batch Analytics",
    ])

    with t_entry:
        if ana_df.empty:
            st.warning("No students enrolled in database yet.")
        else:
            st.subheader("Enter Individual Student Placement Record")
            stu_map = {
                f"{r['usn']} — {r['student_name']} ({r['class_name']})": r["student_id"]
                for _, r in ana_df.iterrows()
            }
            sel_stu_lbl = st.selectbox("Select Student *", list(stu_map.keys()), key="pl_sel_student")
            sel_stu_id = stu_map[sel_stu_lbl]
            stu_info = ana_df[ana_df["student_id"] == sel_stu_id].iloc[0]

            st.info(
                f"👨‍🎓 **{stu_info['student_name']}** (`{stu_info['usn']}`) | "
                f"CGPA: **{stu_info['cgpa']:.2f}** | Earned Credits: **{stu_info['credits_earned']:g}** | "
                f"Active Backlogs: **{int(stu_info['active_backlogs'])}** | Attendance: **{stu_info['attendance_pct']:.1f}%** | "
                f"Next-Sem: **{stu_info['eligibility_status']}**"
            )

            offer_file = st.file_uploader(
                "Optional: Upload Offer Letter (PDF / Image) to Auto-Extract Company, Role & Package",
                type=["pdf", "png", "jpg", "jpeg"],
                key=f"pl_offer_upload_{sel_stu_id}",
            )
            extracted_defaults: Dict[str, Any] = {}
            offer_saved_path = ""
            if offer_file is not None:
                pl_dir = os.path.join(upload_dir, "placements", sel_stu_id)
                os.makedirs(pl_dir, exist_ok=True)
                offer_saved_path = os.path.join(pl_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{offer_file.name}")
                with open(offer_saved_path, "wb") as f:
                    f.write(offer_file.getbuffer())
                extracted_defaults = extract_offer_letter_details(offer_saved_path)
                st.success(
                    f"📄 Offer Letter scanned! Detected Company: **{extracted_defaults.get('company_name') or 'N/A'}** | "
                    f"Role: **{extracted_defaults.get('job_role') or 'N/A'}** | Package: **{extracted_defaults.get('package_lpa', 0.0):.2f} LPA**"
                )

            with st.form("individual_placement_form"):
                pc1, pc2, pc3 = st.columns(3)
                comp_in = pc1.text_input("Company Name *", value=extracted_defaults.get("company_name", ""), placeholder="e.g. Infosys / TCS / Accenture")
                role_in = pc2.text_input("Job Role / Designation", value=extracted_defaults.get("job_role", ""), placeholder="e.g. Software Engineer")
                pkg_in = pc3.number_input("Package (CTC in LPA)", min_value=0.0, max_value=200.0, value=float(extracted_defaults.get("package_lpa") or 0.0), step=0.25)

                pc4, pc5, pc6 = st.columns(3)
                status_in = pc4.selectbox("Offer / Placement Status *", list(ALL_PLACEMENT_STATUSES), index=0)
                date_in = pc5.text_input("Drive / Offer Date", value=extracted_defaults.get("drive_date", datetime.now().strftime("%Y-%m-%d")))
                loc_in = pc6.text_input("Work Location", value=extracted_defaults.get("location", ""), placeholder="e.g. Bengaluru")
                rem_in = st.text_area("Placement Officer Remarks (optional)", placeholder="Drive round notes, bond details, internship stipend, etc.")

                if st.form_submit_button("💾 Save Placement Record & Notify Student", type="primary", use_container_width=True):
                    try:
                        upsert_placement_record(
                            student_id=sel_stu_id,
                            company_name=comp_in,
                            job_role=role_in,
                            package_lpa=pkg_in,
                            offer_status=status_in,
                            drive_date=date_in,
                            location=loc_in,
                            offer_letter_path=offer_saved_path,
                            remarks=rem_in,
                            entered_by=user_id,
                            entered_role=staff_role or "placement",
                            notify_student=True,
                        )
                        st.success(f"🎉 Placement record saved for **{stu_info['usn']}** ({comp_in or status_in})!")
                        st.rerun()
                    except ValueError as ve:
                        st.error(str(ve))

            existing_stu_pl = get_student_placements(sel_stu_id)
            if existing_stu_pl:
                st.markdown("##### Existing Placement Records for Selected Student")
                for pl in existing_stu_pl:
                    rc1, rc2 = st.columns([4, 1])
                    rc1.write(
                        f"**{pl['company_name']}** — {pl.get('job_role') or 'Role N/A'} | "
                        f"**{float(pl.get('package_lpa') or 0):.2f} LPA** | Status: `{pl['offer_status']}` | "
                        f"Entered by `{pl.get('entered_by', '')}`"
                    )
                    if rc2.button("🗑️ Remove", key=f"del_pl_{pl['id']}", use_container_width=True):
                        delete_placement_record(pl["id"], actor_id=user_id)
                        st.rerun()

            if not all_pl_df.empty and "verification_status" in all_pl_df.columns:
                pend_df = all_pl_df[all_pl_df["verification_status"] == "Pending Verification"]
                if not pend_df.empty:
                    st.divider()
                    st.subheader(f"⏳ Student Off-Campus Offer Proofs Awaiting Verification ({len(pend_df)})")
                    for _, pr in pend_df.iterrows():
                        pc_a, pc_b, pc_c = st.columns([3.5, 1, 1])
                        pc_a.write(
                            f"**{pr['student_name']}** (`{pr['usn']}`) — **{pr['company_name']}** | "
                            f"{pr.get('job_role') or 'Role N/A'} | **{float(pr.get('package_lpa') or 0):.2f} LPA** (`{pr['offer_status']}`)"
                        )
                        if pc_b.button("✅ Approve", key=f"app_pl_{pr['id']}", type="primary", use_container_width=True):
                            review_placement_offer(pr["id"], "Verified", reviewer_id=user_id)
                            st.rerun()
                        if pc_c.button("❌ Reject", key=f"rej_pl_{pr['id']}", use_container_width=True):
                            review_placement_offer(pr["id"], "Rejected", reviewer_id=user_id)
                            st.rerun()

    with t_matcher:
        st.subheader("🎯 Campus Drive Eligibility Matcher & 1-Click Shortlist")
        st.caption("Define company hiring criteria to automatically filter eligible candidates, download the shortlist, and notify students in 1 click.")
        dm1, dm2, dm3, dm4 = st.columns(4)
        d_comp = dm1.text_input("Drive Company Name *", value="TCS Digital", key="dm_comp")
        d_role = dm2.text_input("Role", value="Systems Engineer", key="dm_role")
        d_pkg = dm3.number_input("CTC (LPA)", min_value=0.0, max_value=200.0, value=7.0, step=0.5, key="dm_pkg")
        d_date = dm4.text_input("Drive Date", value=datetime.now().strftime("%Y-%m-%d"), key="dm_date")

        dm5, dm6, dm7, dm8 = st.columns(4)
        d_min_cgpa = dm5.number_input("Min CGPA Required", min_value=0.0, max_value=10.0, value=6.5, step=0.25, key="dm_cgpa")
        d_max_bk = dm6.number_input("Max Active Backlogs", min_value=0, max_value=10, value=0, step=1, key="dm_bk")
        d_min_att = dm7.number_input("Min Attendance %", min_value=0.0, max_value=95.0, value=75.0, step=5.0, key="dm_att")
        d_skills = dm8.text_input("Required Skills (comma-separated)", placeholder="Python, SQL, React", key="dm_skills")

        dc_opt1, dc_opt2 = st.columns(2)
        d_excl_placed = dc_opt1.checkbox("Exclude Already Placed Students", value=False, key="dm_excl_placed")
        d_req_sem = dc_opt2.checkbox("Require Next-Semester Eligibility", value=True, key="dm_req_sem")

        shortlist_df = match_students_for_drive(
            min_cgpa=d_min_cgpa,
            max_active_backlogs=int(d_max_bk),
            min_attendance_pct=d_min_att,
            required_skills_csv=d_skills,
            exclude_already_placed=d_excl_placed,
            require_next_sem_eligible=d_req_sem,
        )
        st.info(f"🎯 **{len(shortlist_df)} eligible candidate(s)** matched criteria for **{d_comp or 'this drive'}**.")
        if not shortlist_df.empty:
            sl_cols = [
                "usn", "student_name", "class_name", "cgpa", "active_backlogs",
                "attendance_pct", "skills_list", "placement_readiness_score",
                "placement_status", "email", "phone",
            ]
            st.dataframe(shortlist_df[sl_cols], use_container_width=True, hide_index=True)
            b_sl1, b_sl2 = st.columns(2)
            b_sl1.download_button(
                "⬇️ Download Recruiter Shortlist (CSV)",
                data=_df_csv_bytes(shortlist_df[sl_cols]),
                file_name=f"{(d_comp or 'drive').replace(' ', '_')}_shortlist.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_drive_shortlist_csv",
            )
            if b_sl2.button("💾 Save Drive & 📩 Send Invite to Shortlisted Students", type="primary", use_container_width=True, key="btn_save_notify_drive"):
                if not d_comp.strip():
                    st.error("Drive Company Name is required.")
                else:
                    save_placement_drive(
                        company_name=d_comp,
                        job_role=d_role,
                        package_lpa=d_pkg,
                        min_cgpa=d_min_cgpa,
                        max_active_backlogs=int(d_max_bk),
                        min_attendance_pct=d_min_att,
                        required_skills=d_skills,
                        exclude_already_placed=d_excl_placed,
                        require_next_sem_eligible=d_req_sem,
                        drive_date=d_date,
                        created_by=user_id,
                    )
                    sent_cnt = notify_shortlisted_students(
                        shortlist_df["student_id"].tolist(),
                        company_name=d_comp,
                        job_role=d_role,
                        drive_date=d_date,
                        actor_id=user_id,
                    )
                    st.success(f"🎉 Saved campus drive **{d_comp}** and sent invite notifications to **{sent_cnt}** student(s)!")

        saved_drives = get_placement_drives()
        if not saved_drives.empty:
            with st.expander(f"📅 Scheduled & Saved Campus Drives ({len(saved_drives)})", expanded=False):
                st.dataframe(
                    saved_drives[["company_name", "job_role", "package_lpa", "min_cgpa", "max_active_backlogs", "required_skills", "drive_date"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with t_ats:
        st.subheader("Campus Recruitment Pipeline & Applicant Tracking (ATS)")
        st.caption("Manage student applicants across recruitment rounds (Applied -> Shortlisted -> Assessment -> Technical -> HR -> Selected/Rejected).")
        drives_list = get_placement_drives()
        if drives_list.empty:
            st.info("No placement drives created yet. Create a drive in the '🎯 Drive Matcher & Shortlist' tab first.")
        else:
            drive_map = {
                f"{r['company_name']} ({r.get('job_role', 'Drive')}) — {float(r.get('package_lpa', 0.0)):.2f} LPA": r["id"]
                for _, r in drives_list.iterrows()
            }
            sel_drv_lbl = st.selectbox("Select Recruitment Drive", list(drive_map.keys()), key="ats_sel_drive")
            sel_drv_id = drive_map[sel_drv_lbl]

            apps_df = get_drive_applications(drive_id=sel_drv_id)
            if apps_df.empty:
                st.info("No students have applied for this campus drive yet.")
            else:
                c_st1, c_st2, c_st3, c_st4 = st.columns(4)
                tot_app = len(apps_df)
                c_st1.metric("Total Applicants", tot_app)
                c_st2.metric("In Progress", len(apps_df[apps_df["application_stage"].isin(["Applied", "Shortlisted", "Assessment", "Technical Round", "HR Round"])]))
                c_st3.metric("Selected / Offered", len(apps_df[apps_df["application_stage"] == "Selected"]))
                c_st4.metric("Rejected", len(apps_df[apps_df["application_stage"] == "Rejected"]))

                st.markdown("#### Candidate Pool")
                st.dataframe(
                    apps_df[["usn", "student_name", "class_name", "application_stage", "resume_headline", "stage_feedback", "applied_at", "updated_at"]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### ⚡ Advance Candidate Stage")
                cand_map = {f"{r['usn']} — {r['student_name']} (Current: {r['application_stage']})": r["id"] for _, r in apps_df.iterrows()}
                ats_c1, ats_c2 = st.columns(2)
                sel_cand_lbl = ats_c1.selectbox("Select Candidate", list(cand_map.keys()), key="ats_sel_cand")
                sel_app_id = cand_map[sel_cand_lbl]
                next_stage = ats_c2.selectbox(
                    "Move To Stage",
                    ["Shortlisted", "Assessment", "Technical Round", "HR Round", "Selected", "Rejected"],
                    key="ats_next_stage",
                )

                ats_f1, ats_f2 = st.columns([2, 1])
                feedback_note = ats_f1.text_input("Round Notes / Interview Feedback", placeholder="e.g., Cleared technical interview with strong DSA skills.", key="ats_feedback")
                cur_drv_row = drives_list[drives_list["id"] == sel_drv_id].iloc[0]
                final_lpa = ats_f2.number_input("Offered Package LPA (if Selected)", min_value=0.0, max_value=150.0, value=float(cur_drv_row.get("package_lpa", 0.0) or 0.0), step=0.5, key="ats_pkg")

                if st.button("🚀 Update Candidate Stage", type="primary", key="btn_update_ats_stage"):
                    succ = update_drive_application_stage(
                        app_id=sel_app_id,
                        new_stage=next_stage,
                        feedback=feedback_note,
                        package_lpa=final_lpa if next_stage == "Selected" else None,
                        reviewer_id=user_id,
                    )
                    if succ:
                        if next_stage == "Selected":
                            st.success(f"🎉 Candidate marked as SELECTED! Official placement record created and student notified.")
                        else:
                            st.success(f"Candidate stage updated to **{next_stage}**!")
                        st.rerun()

    with t_lists:
        col_pl, col_npl = st.columns(2)
        with col_pl:
            st.subheader(f"✅ Placed Students ({placed_cnt})")
            if seg["placed_df"].empty:
                st.info("No students marked Placed yet.")
            else:
                p_cols = ["usn", "student_name", "class_name", "cgpa", "placement_companies", "highest_package_lpa", "offers_count"]
                st.dataframe(seg["placed_df"][p_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Placed List (CSV)",
                    data=_df_csv_bytes(seg["placed_df"][p_cols]),
                    file_name="placed_students.csv",
                    mime="text/csv",
                    key="pl_tab_dl_placed",
                )
        with col_npl:
            st.subheader(f"📋 Not Placed Students ({not_placed_cnt})")
            if seg["not_placed_df"].empty:
                st.success("All students in cohort are placed!")
            else:
                np_cols = ["usn", "student_name", "class_name", "cgpa", "credits_earned", "active_backlogs", "attendance_pct", "eligibility_status"]
                st.dataframe(seg["not_placed_df"][np_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Not Placed List (CSV)",
                    data=_df_csv_bytes(seg["not_placed_df"][np_cols]),
                    file_name="not_placed_students.csv",
                    mime="text/csv",
                    key="pl_tab_dl_not_placed",
                )

        if not all_pl_df.empty:
            st.divider()
            st.subheader("📑 Master Offers Ledger (Visible to All Faculty & Admin)")
            show_cols = [c for c in ["usn", "student_name", "class_name", "company_name", "job_role", "package_lpa", "offer_status", "verification_status", "drive_date", "location", "entered_by"] if c in all_pl_df.columns]
            st.dataframe(all_pl_df[show_cols], use_container_width=True, hide_index=True)

    with t_readiness:
        st.subheader("🤖 AI Placement Readiness Score & Skill-Gap Analyzer")
        st.caption("Scores students (0-100) using CGPA, zero active backlogs, attendance compliance, technical skills, and verified certifications.")
        if ana_df.empty:
            st.info("No students found.")
        else:
            r_df = ana_df.sort_values(by=["placement_readiness_score", "cgpa"], ascending=[False, False])
            tier_counts = r_df["placement_readiness_tier"].value_counts().reset_index()
            tier_counts.columns = ["Readiness Tier", "Students"]
            fig_r = px.bar(tier_counts, x="Readiness Tier", y="Students", color="Readiness Tier", text="Students", title="Cohort Placement Readiness Distribution")
            fig_r.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_r, use_container_width=True)

            r_cols = [
                "usn", "student_name", "class_name", "cgpa", "active_backlogs", "attendance_pct",
                "skills_count", "verified_certs_count", "placement_readiness_score",
                "placement_readiness_tier", "placement_gap_advice",
            ]
            st.dataframe(r_df[r_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Placement Readiness & Skill-Gap Report (CSV)",
                data=_df_csv_bytes(r_df[r_cols]),
                file_name="placement_readiness_skill_gaps.csv",
                mime="text/csv",
                key="dl_readiness_csv",
            )

    with t_bulk:
        st.subheader("Automated Bulk Placement Import (Excel / CSV)")
        template_df = pd.DataFrame([
            {
                "USN": "1TJ24DS028",
                "Company": "Infosys",
                "Job Role": "System Engineer",
                "Package LPA": 5.0,
                "Status": "Placed",
                "Drive Date": datetime.now().strftime("%Y-%m-%d"),
                "Location": "Bengaluru",
                "Remarks": "Campus Drive",
            }
        ])
        st.download_button(
            "📄 Download Placement Bulk Import Template (.csv)",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="placement_import_template.csv",
            mime="text/csv",
            key="dl_pl_template_csv",
        )
        pl_file = st.file_uploader("Upload Placement Sheet (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"], key="pl_bulk_up")
        if pl_file is not None:
            raw_df = pd.read_csv(pl_file) if pl_file.name.lower().endswith(".csv") else pd.read_excel(pl_file)
            val = validate_placement_excel(raw_df)
            if val.get("error"):
                st.error(val["error"])
            else:
                if val["invalid_rows"]:
                    st.warning(f"Skipped {len(val['invalid_rows'])} invalid row(s): {val['invalid_rows'][:5]}")
                if val["valid_rows"]:
                    st.success(f"✅ Found **{len(val['valid_rows'])} valid placement row(s)** ready to import.")
                    st.dataframe(pd.DataFrame(val["valid_rows"]).head(10), use_container_width=True, hide_index=True)
                    if st.button("🚀 Import All Valid Placement Records", type="primary", key="btn_exec_pl_bulk"):
                        cnt, _ = execute_placement_bulk_import(val["valid_rows"], actor_id=user_id, actor_role=staff_role or "placement")
                        st.success(f"🎉 Imported **{cnt}** placement record(s) automatically!")
                        st.rerun()

    with t_analytics:
        if all_pl_df.empty:
            st.info("Placement analytics charts will render as soon as placement records are added.")
        else:
            ac1, ac2 = st.columns(2)
            with ac1:
                comp_stats = all_pl_df.groupby("company_name").agg(
                    Offers=("student_id", "count"),
                    Avg_LPA=("package_lpa", "mean"),
                ).reset_index()
                fig_c = px.bar(comp_stats, x="company_name", y="Offers", color="Avg_LPA", title="Company-Wise Offers & Average Package (LPA)")
                st.plotly_chart(fig_c, use_container_width=True)
            with ac2:
                st_counts = all_pl_df["offer_status"].value_counts().reset_index()
                st_counts.columns = ["Status", "Count"]
                fig_s = px.pie(st_counts, names="Status", values="Count", title="Offer Status Distribution")
                st.plotly_chart(fig_s, use_container_width=True)

            tier_data = pd.DataFrame([
                {"Salary Tier": "Super Dream (>= 10 LPA)", "Offers": int((ana_df["highest_package_lpa"] >= 10.0).sum())},
                {"Salary Tier": "Dream (6 - 10 LPA)", "Offers": int(((ana_df["highest_package_lpa"] >= 6.0) & (ana_df["highest_package_lpa"] < 10.0)).sum())},
                {"Salary Tier": "Standard (< 6 LPA)", "Offers": int(((ana_df["highest_package_lpa"] > 0.0) & (ana_df["highest_package_lpa"] < 6.0)).sum())},
            ])
            fig_tier = px.pie(tier_data, names="Salary Tier", values="Offers", title="Compensation Band Breakdown", color_discrete_sequence=["#7C3AED", "#2563EB", "#059669"])
            st.plotly_chart(fig_tier, use_container_width=True)
        if not ana_df.empty and "admission_year" in ana_df.columns:
            st.divider()
            batch_df = ana_df.groupby("admission_year").agg(
                Total=("student_id", "count"),
                Placed=("is_placed", "sum"),
                Avg_Max_LPA=("highest_package_lpa", "max"),
            ).reset_index()
            batch_df["Placement_Pct"] = ((batch_df["Placed"] / batch_df["Total"]) * 100.0).round(1)
            fig_b = px.bar(batch_df, x="admission_year", y=["Total", "Placed"], barmode="group", title="Batch / Admission-Year Placement Comparison")
            st.plotly_chart(fig_b, use_container_width=True)


# ============================================================
# 4. AUTOMATED BULK DATA ENTRY PANELS (MARKS & ATTENDANCE)
# ============================================================

def render_automated_bulk_marks_panel(user_id: str) -> None:
    """Reusable UI for Automated Bulk Final VTU & IA Marks Excel/CSV Import."""
    st.subheader("📊 Automated Bulk VTU Final & IA Marks Import (Excel / CSV)")
    mode_lbl = st.radio("Marks Import Mode", ["Final VTU Semester Marks (auto SGPA & Backlogs)", "Current Semester IA1 / IA2 Marks"], horizontal=True, key="bulk_marks_mode")
    mode = "final_vtu" if "Final VTU" in mode_lbl else "ia"

    if mode == "final_vtu":
        sample = pd.DataFrame([
            {"USN": "1TJ24DS028", "Semester": 5, "Subject Code": "BCS501", "Subject Name": "Software Engineering", "Internal": 45, "External": 42, "Credits": 4.0, "Result": "P", "SGPA": 8.75}
        ])
    else:
        sample = pd.DataFrame([
            {"USN": "1TJ24DS028", "Semester": 5, "Subject Code": "BCS501", "Subject Name": "Software Engineering", "Credits": 4.0, "IA1": 24, "IA2": 26}
        ])
    st.download_button(
        "📄 Download Sample Marks Template (.csv)",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name=f"edupredict_{mode}_marks_template.csv",
        mime="text/csv",
        key="dl_marks_tpl_btn",
    )

    m_file = st.file_uploader("Upload Marks Sheet (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"], key="bulk_marks_file_up")
    if m_file is not None:
        raw_df = pd.read_csv(m_file) if m_file.name.lower().endswith(".csv") else pd.read_excel(m_file)
        val = validate_marks_excel(raw_df, mode=mode)
        if val.get("error"):
            st.error(val["error"])
        else:
            if val["invalid_rows"]:
                st.warning(f"Skipped {len(val['invalid_rows'])} row(s): {val['invalid_rows'][:5]}")
            if val["valid_rows"]:
                st.success(f"✅ Validated **{len(val['valid_rows'])} subject mark rows** ready to auto-import.")
                st.dataframe(pd.DataFrame(val["valid_rows"]).head(10), use_container_width=True, hide_index=True)
                if st.button("🚀 Confirm Automated Bulk Marks Import", type="primary", key="btn_exec_bulk_marks"):
                    stats = execute_marks_bulk_import(val["valid_rows"], mode=mode, actor_id=user_id)
                    st.success(f"🎉 Auto-imported **{stats['subjects_saved']}** subject rows ({stats['semesters_saved']} VTU semester records synced)!")
                    st.rerun()


def render_automated_bulk_attendance_panel(user_id: str) -> None:
    """Reusable UI for Automated Bulk Attendance Excel/CSV Import."""
    st.subheader("📥 Automated Bulk Attendance Import (Excel / CSV)")
    sample = pd.DataFrame([
        {"USN": "1TJ24DS028", "Subject": "Database Management Systems", "Date": datetime.now().strftime("%Y-%m-%d"), "Hour": 1, "Status": "Present"}
    ])
    st.download_button(
        "📄 Download Attendance Template (.csv)",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="edupredict_attendance_template.csv",
        mime="text/csv",
        key="dl_att_tpl_btn",
    )
    a_file = st.file_uploader("Upload Attendance Sheet (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"], key="bulk_att_file_up")
    if a_file is not None:
        raw_df = pd.read_csv(a_file) if a_file.name.lower().endswith(".csv") else pd.read_excel(a_file)
        val = validate_attendance_excel(raw_df)
        if val.get("error"):
            st.error(val["error"])
        else:
            if val["invalid_rows"]:
                st.warning(f"Skipped {len(val['invalid_rows'])} row(s): {val['invalid_rows'][:5]}")
            if val["valid_rows"]:
                st.success(f"✅ Validated **{len(val['valid_rows'])}** attendance records (Mode: `{val['mode']}`).")
                st.dataframe(pd.DataFrame(val["valid_rows"]).head(10), use_container_width=True, hide_index=True)
                if st.button("🚀 Confirm Automated Bulk Attendance Import", type="primary", key="btn_exec_bulk_att"):
                    cnt = execute_attendance_bulk_import(val["valid_rows"], mode=val["mode"], actor_id=user_id)
                    st.success(f"🎉 Imported **{cnt}** attendance record(s) automatically!")
                    st.rerun()
