# EduPredict AI — VTU Academic Management & Machine Learning Grade Prediction Platform

EduPredict AI is a comprehensive, production-ready academic intelligence and student management platform designed specifically for Visvesvaraya Technological University (VTU) engineering students and faculty.

---

## 🚀 Key Highlights & Architecture

- **Role-Based Authentication**: Secure student and faculty logins with PBKDF2 HMAC SHA-256 cryptographic hashing.
- **Marksheet OCR & Review Engine**: Multi-format marksheets extraction (PDF, JPG, PNG) using PyMuPDF and Tesseract OCR with an **editable review/confirmation table**.
- **Dynamic Subject Support**: Unlimited dynamic subjects configured per class and semester from the SQLite database.
- **Real Machine Learning Predictions**: Scikit-Learn Random Forest, Gradient Boosting, and Ridge regression pipeline with real evaluation metrics ($R^2=0.912$, $\text{MAE}=2.67$).
- **Weak Subject Detection**: Institutional academic recommendations and interactive what-if grade simulations.
- **Hourly Attendance Tracking**: Period-by-period class attendance logging with duplicate prevention, compliance warnings ($<75\%$), and audit trails.
- **360-Degree Document Verification**: Dedicated verification center for certificates, fee receipts, non-due forms, marksheets, and student applications.
- **Comprehensive Reporting**: Multi-page PDF student profiles and 4-sheet consolidated class Excel exports (Pandas + OpenPyXL).

---

## 📁 Project Structure

```
EduPredict_AI/
│
├── app.py                     # Master Streamlit application entrypoint & authentication router
├── auth.py                    # PBKDF2 hashing, role authentication, sessions, and audit logging
├── db_migrations.py           # Safe non-destructive SQLite schema migrations
├── edupredict.db              # Relational SQLite database with indexes and foreign keys
├── excel_handler.py           # 4-sheet consolidated class Excel exporter and bulk student importer
├── ocr_engine.py              # PyMuPDF & Tesseract VTU marksheet OCR parser
├── portal_staff.py            # Complete 16-module Faculty / Class Teacher portal
├── portal_student.py          # Complete 16-module Student portal
├── reports.py                 # PyMuPDF-powered multi-page student academic PDF generator
├── requirements.txt           # Verified Python dependencies
├── tests_system.py            # End-to-end integration and unit test suite
│
├── ml/
│   ├── feature_engineering.py # Leakage-free feature vector extraction
│   ├── predict.py             # Inference pipeline, attention analysis & suggestions
│   ├── evaluate.py            # Real evaluation metrics retrieval
│   ├── train_all.py           # Training routine for regression and risk models
│   ├── train_subject_model.py # Subject marks regression model trainer
│   └── models/
│       ├── subject_final_marks_model.pkl # Trained Random Forest Regressor
│       ├── subject_model_metrics.json    # Verified evaluation metrics
│       ├── risk_model.pkl               # Academic risk classifier
│       └── risk_model_metrics.json       # Classifier accuracy & F1 metrics
│
└── uploads/                   # Isolated student document directories
    ├── photos/                # Student passport photos
    ├── marksheets/            # Semester VTU marksheets
    ├── certificates/          # Portfolio certifications
    ├── fee_receipts/          # Semester fee payment receipts
    ├── non_due/               # Clearance non-due forms
    └── applications/          # Student formal letters & attachments
```

---

## 🔑 Default & Test Login Setup

### 👨‍🎓 Student Accounts (Preserved Data)
- **Vivek S**: USN: `1TJ24DS028` | Password: `Student@123`
- **Tarun K V**: USN: `1TJ23DS030` | Password: `Student@123`
- **S Ganesh**: USN: `1TJ24DS018` | Password: `Student@123`
*(Students can also configure or reset their own password via "First-time Student Setup" on the login page).*

### 🏫 Faculty Accounts
- **Prof. Sreevani**: Staff ID: `TJIT001` | Role: Class Teacher (Data Science)
*(Faculty can also create additional accounts instantly via the "Faculty Setup" tab).*

---

## 🛠️ Installation & How to Run

### Step 1: Open Terminal in Project Directory
```powershell
cd c:\Users\vivek\OneDrive\Desktop\EduPredict_AI
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```powershell
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🧪 Running System Tests
To run the automated test suite verifying database migrations, authentication, OCR parsing, ML predictions, and reporting:
```powershell
python tests_system.py
```

---

## 📌 Features Completed End-to-End

1. **Authentication & Session Isolation**: Secure PBKDF2 hashing with salt, role-based separation, audit logging.
2. **Dynamic Classes & Students**: Multi-class support, flexible student management, auto-provisioned logins.
3. **Unlimited Dynamic Subjects**: Configured in database per class and semester with credits and internal/external maximums.
4. **Semester Marksheet OCR**: PyMuPDF + Tesseract extraction with editable review table (Total = Internal + External).
5. **Credit-Weighted SGPA & CGPA**: VTU 10-point scale calculations with complete academic history.
6. **Hourly Attendance & Analytics**: Class hour tracking, duplicate prevention, $<75\%$ compliance warnings.
7. **Document Verification Hub**: Faculty 360-degree review of marksheets, certificates, fee receipts, and non-due forms.
8. **Student Applications**: Submission of Leave, Bonafide, and Permission requests with faculty approval workflow.
9. **Real Machine Learning Predictions**: Final subject mark estimation with feature importance and what-if simulation.
10. **Reports & Exports**: 4-Sheet Excel reports, CSV exports, and multi-page student academic PDF portfolios.
11. **System Governance & Admin Portal**: Account active status toggling, password resets, and user management.
12. **Semester Locking Enforcement**: Freezes historical and IA marks to prevent post-deadline student tampering.
13. **Database Backup & Disaster Recovery**: 1-click live database snapshots with rollback protection and restore capability.
14. **System Health Diagnostics**: Integrity checking (`PRAGMA integrity_check;`), OCR engine validation, and storage monitoring.
