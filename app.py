"""
EduPredict AI — Master Application Entry Point.
A complete, reliable, and beautiful academic management and AI-based performance
prediction platform for VTU engineering students and faculty.

Features:
- Role-based Authentication (Student & Staff) with PBKDF2 HMAC SHA-256
- 100% Reliable, Permanent SQLite Database Persistence
- PyMuPDF & Tesseract OCR for semester marksheets with review/confirmation table
- Scikit-Learn Predictive ML Engine (Random Forest, Gradient Boosting, Ridge)
- Consolidated 4-Sheet Excel Import & Export
- Interactive Plotly Analytics & Gauge Visualizations
- Student 360-degree Portfolio & Document Verification Center
- Real-time Hourly Attendance & Academic Risk Warnings
"""

import os
import sys
import sqlite3
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="EduPredict AI — VTU Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core Application Directory Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "edupredict.db")

# Import Application Modules
from db_migrations import run_migrations
from auth import (
    init_session_state,
    logout_user,
    authenticate_student,
    authenticate_staff,
    student_first_time_setup,
    create_first_staff,
    provision_existing_students,
    count_staff_users
)
from portal_student import render_student_portal
from portal_staff import render_staff_portal

# Safe Database Migration and Initialization on Startup (Cached once)
@st.cache_resource(show_spinner=False)
def initialize_system():
    run_migrations(DB_FILE)
    provision_existing_students(default_password="Student@123")
    return True

initialize_system()

# Initialize Session State
init_session_state(st)

# ============================================================
# MODERN DESIGN SYSTEM & SPLIT-SCREEN SAAS THEME (EDUSPHERE AESTHETIC)
# ============================================================
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Soft Luminous Canvas Background with Subtle Organic Curves */
    .stApp {
        background: 
            radial-gradient(circle at 92% 8%, rgba(224, 231, 255, 0.7) 0%, transparent 45%),
            radial-gradient(circle at 8% 92%, rgba(219, 234, 254, 0.6) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(241, 245, 249, 0.5) 0%, transparent 60%),
            #F4F7FC !important;
        background-attachment: fixed !important;
        color: #0F172A !important;
    }
    
    /* Left Hero Card */
    .hero-split-left {
        background: linear-gradient(150deg, #090E24 0%, #0F1A3A 45%, #172B5E 100%) !important;
        border-radius: 28px !important;
        padding: 44px 38px !important;
        color: #FFFFFF !important;
        height: 100% !important;
        min-height: 640px !important;
        box-shadow: 0 25px 60px -15px rgba(15, 26, 58, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .hero-split-left::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -30%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    
    /* Feature Highlight Items */
    .feature-item {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
    }
    .feature-icon-bubble {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Right Floating Login Card */
    .floating-login-card {
        background: #FFFFFF !important;
        border-radius: 28px !important;
        padding: 38px 40px !important;
        box-shadow: 0 20px 50px -10px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(226, 232, 240, 0.8) !important;
    }
    
    /* Top Navigation Bar */
    .top-nav-bar {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 14px 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.04) !important;
    }
    
    /* Modern Streamlit Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        padding: 18px 22px !important;
        border-radius: 16px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, #4F46E5, #8B5CF6);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.12) !important;
        border-color: #CBD5E1 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #1E293B !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #64748B !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }
    
    /* Status Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .badge-verified { background: #ECFDF5; color: #047857; border: 1.5px solid #A7F3D0; }
    .badge-pending { background: #FFFBEB; color: #B45309; border: 1.5px solid #FDE68A; }
    .badge-rejected { background: #FEF2F2; color: #B91C1C; border: 1.5px solid #FECACA; }
    
    /* Modern Card Architecture */
    .edu-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 22px 26px !important;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05) !important;
        margin-bottom: 20px !important;
    }

    /* Primary Pill Buttons & Interactive Feedback */
    .stButton>button, .stDownloadButton>button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 10px 22px !important;
        transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        letter-spacing: 0.2px !important;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #4338CA 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.45) !important;
        background: linear-gradient(135deg, #1D4ED8 0%, #3730A3 100%) !important;
    }
    .stButton>button[kind="primary"]:active {
        transform: scale(0.97) !important;
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.35) !important;
    }
    .stButton>button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #CBD5E1 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    }
    .stButton>button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #64748B !important;
        color: #0F172A !important;
        transform: translateY(-1px) !important;
    }
    .stButton>button[kind="secondary"]:active {
        transform: scale(0.97) !important;
        background: #E2E8F0 !important;
    }
    
    /* Download Buttons */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.4) !important;
    }
    .stDownloadButton>button:active {
        transform: scale(0.97) !important;
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.35) !important;
    }

    /* Expanders with high-clarity borders & shadows */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03) !important;
        margin-bottom: 14px !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 700 !important;
        font-size: 15px !important;
        color: #0F172A !important;
        padding: 14px 18px !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #2563EB !important;
        background: #F8FAFC !important;
    }
    
    /* Clean Modern High-Contrast Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #F1F5F9 !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        gap: 6px !important;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs button[role="tab"],
    .stTabs button {
        opacity: 1 !important;
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .stTabs [data-baseweb="tab"] *,
    .stTabs button[role="tab"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs button[role="tab"]:hover {
        background: #EFF6FF !important;
        border-color: #3B82F6 !important;
    }
    .stTabs [data-baseweb="tab"]:hover *,
    .stTabs button[role="tab"]:hover * {
        color: #1D4ED8 !important;
        -webkit-text-fill-color: #1D4ED8 !important;
    }
    .stTabs [aria-selected="true"],
    .stTabs button[aria-selected="true"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: 1.5px solid #1E40AF !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
    .stTabs [aria-selected="true"] *,
    .stTabs button[aria-selected="true"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    /* Clean Form Inputs & Floating Form Containers */
    [data-testid="stForm"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 20px !important;
        padding: 24px 28px !important;
        box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05) !important;
    }
    
    /* High Contrast Input Labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label,
    .stTextInput label,
    .stSelectbox label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        margin-bottom: 6px !important;
    }
    
    .stTextInput input, .stSelectbox select, .stNumberInput input, .stTextArea textarea {
        background: #F8FAFC !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput input:focus, .stSelectbox select:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #2563EB !important;
        background: #FFFFFF !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    /* Remove browser-native duplicate password reveal/clear icons (Edge & Chrome on Windows) */
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    input[type="password"]::-webkit-contacts-auto-fill-button,
    input[type="password"]::-webkit-credentials-auto-fill-button {
        visibility: hidden !important;
        display: none !important;
        pointer-events: none !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
    }

    /* Clean Streamlit form input instructions to prevent text overlap */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    [data-testid="stTextInput"] input[type="password"] {
        padding-right: 48px !important;
    }
    
    /* Modern Light Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    /* Book Stack Graphic */
    .book-stack-container {
        display: flex;
        align-items: flex-end;
        gap: 14px;
        margin-top: 24px;
    }
    .book-stack {
        display: flex;
        flex-direction: column-reverse;
        gap: 4px;
    }
    .book-spine {
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTHENTICATION ROUTER & PORTAL
# ============================================================

def render_login_view():
    """Render the split-screen modern authentication portal (matching EduSphere design)."""
    
    # Top bar quick switch / sign up info
    top_c1, top_c2 = st.columns([2, 1])
    with top_c2:
        st.markdown("""<div style="text-align: right; font-size: 14px; color: #475569; margin-bottom: 12px; font-weight: 500;">Don't have an account? <span style="color: #2563EB; font-weight: 700;">Select Account Setup below</span></div>""", unsafe_allow_html=True)

    # Master Split-Screen Layout (Left Hero, Right Floating Card)
    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    # ========================================================
    # LEFT HERO CARD (Deep Indigo Aesthetic)
    # ========================================================
    with col_left:
        left_hero_html = """<div class="hero-split-left">
<div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
<div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #38BDF8 0%, #6366F1 100%); display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 8px 16px rgba(99, 102, 241, 0.35);">🎓</div>
<div>
<div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 24px; color: #FFFFFF; letter-spacing: -0.5px;">EduPredict <span style="color: #38BDF8;">AI</span></div>
<div style="font-size: 11px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;">Learn &bull; Predict &bull; Achieve</div>
</div>
</div>
<h1 style="font-family: 'Outfit', sans-serif; font-size: 38px; font-weight: 900; color: #FFFFFF; line-height: 1.15; margin: 0 0 16px 0; letter-spacing: -1px;">Your Future<br>Starts <span style="background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Here</span></h1>
<p style="color: #CBD5E1; font-size: 15px; line-height: 1.6; margin-bottom: 32px;">Access your VTU engineering courses, track credit progression, forecast grades with machine learning, and achieve academic excellence — all in one centralized platform.</p>
<div class="feature-item">
<div class="feature-icon-bubble">🎯</div>
<div>
<div style="font-weight: 700; font-size: 15px; color: #FFFFFF;">AI Grade Predictions</div>
<div style="font-size: 13px; color: #94A3B8;">Multi-algorithm regression (R² = 0.91) for final marks</div>
</div>
</div>
<div class="feature-item">
<div class="feature-icon-bubble">📈</div>
<div>
<div style="font-weight: 700; font-size: 15px; color: #FFFFFF;">Track Your Progress</div>
<div style="font-size: 13px; color: #94A3B8;">Real-time SGPA, CGPA & hourly attendance logs</div>
</div>
</div>
<div class="feature-item">
<div class="feature-icon-bubble">⚡</div>
<div>
<div style="font-weight: 700; font-size: 15px; color: #FFFFFF;">360° Academic Intelligence</div>
<div style="font-size: 13px; color: #94A3B8;">Marksheet OCR, portfolio & faculty verification hub</div>
</div>
</div>
</div>
<div class="book-stack-container">
<div class="book-stack">
<div class="book-spine" style="background: #10B981;">Future 📗</div>
<div class="book-spine" style="background: #F59E0B;">Brighter 📙</div>
<div class="book-spine" style="background: #EC4899;">Skills 📕</div>
<div class="book-spine" style="background: #3B82F6;">Better 📘</div>
</div>
<div style="background: rgba(255, 255, 255, 0.08); padding: 12px 18px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.12); flex-grow: 1;">
<div style="font-size: 12px; font-weight: 700; color: #38BDF8; text-transform: uppercase;">VTU Intelligence</div>
<div style="font-size: 13px; color: #E2E8F0; font-weight: 500;">Visvesvaraya Technological University</div>
</div>
</div>
</div>"""
        st.markdown(left_hero_html, unsafe_allow_html=True)

    # ========================================================
    # RIGHT FLOATING LOGIN CARD
    # ========================================================
    with col_right:
        right_header_html = """<div style="text-align: center; margin-bottom: 20px;">
<div style="display: inline-flex; align-items: center; justify-content: center; width: 52px; height: 52px; border-radius: 16px; background: #EFF6FF; font-size: 26px; margin-bottom: 10px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15); border: 1px solid #DBEAFE;">👋</div>
<h2 style="font-family: 'Outfit', sans-serif; font-size: 28px; font-weight: 800; color: #0F172A; margin: 0 0 4px 0; letter-spacing: -0.5px;">Welcome Back!</h2>
<p style="color: #475569; font-size: 14px; margin: 0; font-weight: 500;">Log in to continue your academic learning journey</p>
</div>"""
        st.markdown(right_header_html, unsafe_allow_html=True)

        login_tab1, login_tab2, login_tab3, login_tab4 = st.tabs([
            "👨‍🎓 Student",
            "🏫 Faculty",
            "🔑 Student Setup",
            "📝 Faculty Setup"
        ])

        # 1. STUDENT LOGIN
        with login_tab1:
            with st.form("student_login_form"):
                s_usn = st.text_input("USN / University Seat Number", placeholder="1TJ")
                s_pwd = st.text_input("Password", type="password", placeholder="Enter password")
                
                c_rem, c_fgt = st.columns([1, 1])
                with c_rem:
                    st.checkbox("Remember me", value=True, key="stu_rem_me")

                s_submit = st.form_submit_button("Log In as Student  ➔", type="primary", use_container_width=True)

                if s_submit:
                    success, user_dict, profile_dict, msg = authenticate_student(s_usn, s_pwd)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user_role"] = "student"
                        st.session_state["user_id"] = user_dict["id"]
                        st.session_state["student_id"] = user_dict["student_id"]
                        st.session_state["username"] = profile_dict["name"]
                        st.session_state["usn"] = profile_dict["usn"]
                        st.success(f"Welcome back, {profile_dict['name']}!")
                        st.rerun()
                    else:
                        st.error(msg)
            st.caption("💡 **Tip:** New students can log in with their **USN** and initial password, then update their profile.")

        # 2. FACULTY LOGIN
        with login_tab2:
            with st.form("staff_login_form"):
                st_id = st.text_input("Staff / TJIT ID", placeholder="TJIT")
                st_pwd = st.text_input("Password", type="password", placeholder="Enter your password")
                
                c_rem2, c_fgt2 = st.columns([1, 1])
                with c_rem2:
                    st.checkbox("Remember me", value=True, key="st_rem_me")
                with c_fgt2:
                    st.markdown("<div style='text-align: right; font-size: 13px; color: #4F46E5; margin-top: 5px;'>Faculty Access</div>", unsafe_allow_html=True)

                st_submit = st.form_submit_button("Log In as Faculty  ➔", type="primary", use_container_width=True)

                if st_submit:
                    success, user_dict, msg = authenticate_staff(st_id, st_pwd)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user_role"] = "staff"
                        st.session_state["user_id"] = user_dict["id"]
                        st.session_state["student_id"] = None
                        st.session_state["username"] = user_dict["name"]
                        st.session_state["tjit_number"] = user_dict["tjit_number"]
                        st.session_state["staff_role"] = user_dict.get("role", "staff")
                        st.success(f"Welcome back, {user_dict['name']}!")
                        st.rerun()
                    else:
                        st.error(msg)

        # 3. FIRST-TIME STUDENT SETUP
        with login_tab3:
            with st.form("student_setup_form"):
                st.caption("If your Class Teacher enrolled your USN, configure your password here to activate your account.")
                stu_usn = st.text_input("Registered USN *", placeholder="1TJ")
                stu_npwd = st.text_input("Create New Password (min 6 chars) *", type="password")
                stu_cpwd = st.text_input("Confirm New Password *", type="password")
                stu_sub = st.form_submit_button("Activate Account & Save Password", type="primary", use_container_width=True)

                if stu_sub:
                    succ, msg = student_first_time_setup(stu_usn, stu_npwd, stu_cpwd)
                    if succ:
                        st.success(msg)
                    else:
                        st.error(msg)

        # 4. FACULTY ACCOUNT REGISTRATION
        with login_tab4:
            with st.form("staff_register_form"):
                st.caption("Register an authorized faculty account or set credentials.")
                f_tjit = st.text_input("Staff / TJIT Number (e.g. TJIT) *", placeholder="TJIT")
                f_name = st.text_input("Full Name & Title (e.g. Prof. Ramesh K) *")
                f_email = st.text_input("College Email")
                f_pwd = st.text_input("Password (min 6 chars) *", type="password")
                f_sub = st.form_submit_button("Register Faculty Account", type="primary", use_container_width=True)

                if f_sub:
                    succ, msg = create_first_staff(f_tjit, f_name, f_pwd, email=f_email)
                    if succ:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown("""
            <div style="display: flex; align-items: center; margin: 24px 0 16px 0;">
                <div style="flex: 1; height: 1px; background: #E2E8F0;"></div>
                <div style="padding: 0 14px; color: #94A3B8; font-size: 12px; font-weight: 600;">OR</div>
                <div style="flex: 1; height: 1px; background: #E2E8F0;"></div>
            </div>
            
            <div style="text-align: center; margin-top: 16px; font-size: 12px; color: #64748B; display: flex; align-items: center; justify-content: center; gap: 6px;">
                <span>🛡️</span> <span>Your academic data is protected with PBKDF2 SHA-256 encryption</span>
            </div>
        """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION ROUTING
# ============================================================

def main():
    if not st.session_state.get("authenticated"):
        render_login_view()
    else:
        # Render Modern Top Navigation Header Bar
        u_role = st.session_state.get("user_role")
        u_name = st.session_state.get("username")
        u_id = st.session_state.get("usn") if u_role == "student" else st.session_state.get("tjit_number")
        badge_class = "badge-verified" if u_role == "student" else "badge-pending"
        badge_text = "👨‍🎓 Student Portal" if u_role == "student" else "🏫 Faculty Portal"

        top_col1, top_col2 = st.columns([5, 1])
        with top_col1:
            st.markdown(f"""
            <div class="top-nav-bar" style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-weight: 800; font-size: 17px; color: #0F172A;">🎓 EduPredict AI</span>
                    <span style="color: #94A3B8; margin: 0 8px;">|</span>
                    <span style="font-weight: 600; color: #1E293B;">{u_name}</span>
                    <span style="font-size: 13px; color: #64748B;">({u_id})</span>
                    <span style="margin-left: 10px;" class="badge-pill {badge_class}">{badge_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with top_col2:
            if st.button("🔒 Logout", key="top_logout_btn", use_container_width=True):
                logout_user(st)

        # Render Active Role Portal
        if u_role == "student":
            render_student_portal()
        elif u_role == "staff":
            render_staff_portal()
        else:
            st.error("Invalid session role. Please log in again.")
            logout_user(st)


if __name__ == "__main__":
    main()
