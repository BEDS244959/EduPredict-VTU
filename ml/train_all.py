import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

APP_DIR = r'c:\Users\vivek\OneDrive\Desktop\EduPredict_AI'
MODEL_DIR = os.path.join(APP_DIR, 'ml', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Subject Final Marks Regression Model (VTU 0-100 scale)
# Let's generate a well-calibrated synthetic-grounded dataset if db is small, combined with real patterns:
# VTU: IA (0-50), External Exam (0-50 or 0-100 scaled to 50), Total = IA + External
np.random.seed(42)
N = 500

ia1 = np.random.uniform(15, 50, N)
ia2 = np.random.uniform(15, 50, N)
internal_perf = (ia1 + ia2) / 2.0
ia_trend = ia2 - ia1
credits = np.random.choice([1.0, 2.0, 3.0, 4.0], size=N, p=[0.1, 0.1, 0.6, 0.2])
attendance_pct = np.clip(np.random.normal(82, 12, N), 45, 100)
cgpa = np.clip(np.random.normal(7.5, 1.2, N), 4.0, 9.8)
prev_sgpa = np.clip(cgpa + np.random.normal(0, 0.5, N), 4.0, 10.0)
backlog_count = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.7, 0.15, 0.08, 0.05, 0.02])
current_sem = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=N)
has_prev = np.random.choice([0.0, 1.0], size=N, p=[0.6, 0.4])
prev_sub_marks = np.where(has_prev == 1.0, np.clip(cgpa * 9.5 + np.random.normal(0, 5, N), 35, 95), 0.0)

# Realistic VTU final total marks formula with noise:
# Final total ≈ (Internal * 1.0) + (External based on internal + attendance + historical CGPA)
ext_pred = np.clip((internal_perf * 0.9) + (attendance_pct * 0.1) + (cgpa * 1.5) - (backlog_count * 2.0) + np.random.normal(0, 4.0, N), 18, 50)
target_total = np.clip(internal_perf + ext_pred, 30, 100)

df_reg = pd.DataFrame({
    'ia1': ia1,
    'ia2': ia2,
    'internal_performance': internal_perf,
    'ia_trend': ia_trend,
    'credits': credits,
    'attendance_percentage': attendance_pct,
    'cgpa': cgpa,
    'prev_sgpa': prev_sgpa,
    'backlog_count': backlog_count,
    'current_semester': current_sem,
    'has_prev_subject': has_prev,
    'prev_subject_marks': prev_sub_marks,
    'target_total_marks': target_total
})

feature_cols = [
    'ia1', 'ia2', 'internal_performance', 'ia_trend', 'credits',
    'attendance_percentage', 'cgpa', 'prev_sgpa', 'backlog_count',
    'current_semester', 'has_prev_subject', 'prev_subject_marks'
]

X = df_reg[feature_cols]
y = df_reg['target_total_marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Compare with Gradient Boosting & Ridge
gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Save Regressor Model
rf_path = os.path.join(MODEL_DIR, 'subject_final_marks_model.pkl')
joblib.dump({
    'model': rf,
    'feature_columns': feature_cols,
    'version': 'vtu_subject_rf_v2'
}, rf_path)

# Save comparison and metrics
metrics = {
    'version': 'vtu_subject_rf_v2',
    'sample_count': len(df_reg),
    'mae': round(float(mae), 3),
    'rmse': round(float(rmse), 3),
    'r2': round(float(r2), 3),
    'feature_importances': {col: round(float(imp), 4) for col, imp in zip(feature_cols, rf.feature_importances_)},
    'comparison': {
        'Random Forest': {'MAE': round(float(mae), 3), 'RMSE': round(float(rmse), 3), 'R2': round(float(r2), 3)},
        'Gradient Boosting': {'MAE': round(float(mean_absolute_error(y_test, y_pred_gb)), 3), 'RMSE': round(float(np.sqrt(mean_squared_error(y_test, y_pred_gb))), 3), 'R2': round(float(r2_score(y_test, y_pred_gb)), 3)},
        'Ridge Regression': {'MAE': round(float(mean_absolute_error(y_test, y_pred_ridge)), 3), 'RMSE': round(float(np.sqrt(mean_squared_error(y_test, y_pred_ridge))), 3), 'R2': round(float(r2_score(y_test, y_pred_ridge)), 3)}
    }
}

with open(os.path.join(MODEL_DIR, 'subject_model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("Trained Subject Final Marks Regression Model:")
print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# 2. Student Risk Classification Model based on student_data.csv if available
data_csv = os.path.join(APP_DIR, 'student_data.csv')
if os.path.exists(data_csv):
    df_cls = pd.read_csv(data_csv)
    # Preprocessing
    df_cls['absence_numeric'] = df_cls['student_absence_days']
    df_cls['parent_survey_num'] = (df_cls['parent_answering_survey'] == 'Yes').astype(int)
    df_cls['parent_satisfaction_num'] = (df_cls['parent_school_satisfaction'] == 'Good').astype(int)
    
    cls_features = ['raised_hands', 'visited_resources', 'announcements_view', 'discussion', 'absence_numeric', 'parent_survey_num', 'parent_satisfaction_num']
    Xc = df_cls[cls_features]
    yc = df_cls['performance'] # Low, Medium, High
    
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(Xc_train, yc_train)
    
    yc_pred = clf.predict(Xc_test)
    acc = accuracy_score(yc_test, yc_pred)
    prec = precision_score(yc_test, yc_pred, average='weighted')
    rec = recall_score(yc_test, yc_pred, average='weighted')
    f1 = f1_score(yc_test, yc_pred, average='weighted')
    
    risk_metrics = {
        'version': 'risk_rf_v1',
        'sample_count': len(df_cls),
        'accuracy': round(float(acc), 3),
        'precision': round(float(prec), 3),
        'recall': round(float(rec), 3),
        'f1': round(float(f1), 3),
        'feature_importances': {col: round(float(imp), 4) for col, imp in zip(cls_features, clf.feature_importances_)}
    }
    
    joblib.dump({
        'model': clf,
        'feature_columns': cls_features,
        'version': 'risk_rf_v1'
    }, os.path.join(MODEL_DIR, 'risk_model.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'risk_model_metrics.json'), 'w') as f:
        json.dump(risk_metrics, f, indent=2)
        
    print("Trained Risk Classification Model:")
    print(f"  Accuracy: {acc:.3f}, F1: {f1:.3f}")
