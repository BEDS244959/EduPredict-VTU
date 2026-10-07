"""
Train EduPredict AI models from student_data.csv.

Pipeline (matches the reference-paper methodology):
  Dataset -> Preprocessing -> Resampling -> Feature engineering
  -> Feature selection -> Individual classifiers / Ensemble models
  -> 10-fold cross-validation -> Evaluation -> Save Random Forest

Run:
    python train_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from feature_selection import select_features
from model_comparison import RANDOM_STATE, best_model_name, evaluate_models
from preprocessing import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_engineered_features,
    build_category_maps,
    generate_synthetic_dataset,
    load_raw_dataset,
    prepare_xy,
    resample_minority_classes,
    save_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
METRICS_PATH = MODELS_DIR / "comparison_metrics.csv"
ARTIFACTS_PATH = MODELS_DIR / "training_artifacts.pkl"


def ensure_dataset() -> pd.DataFrame:
    """Load the CSV, or create a labelled synthetic sample if it is missing."""
    if not DATA_PATH.exists():
        print("student_data.csv not found — generating a synthetic Kalboard-style sample.")
        raw = generate_synthetic_dataset()
        save_dataset(raw)
        print(f"Wrote {len(raw)} rows to {DATA_PATH}")
    return load_raw_dataset()


def train() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)

    raw = ensure_dataset()
    print(f"Loaded {len(raw)} student records.")

    # Resampling / class-imbalance handling (paper workflow).
    balanced = resample_minority_classes(raw)
    print(f"After random oversampling: {len(balanced)} rows.")
    print(balanced[TARGET_COLUMN].value_counts().to_string())

    feature_frame = add_engineered_features(balanced)
    all_features = FEATURE_COLUMNS + ["engagement_score"]
    category_maps = build_category_maps(feature_frame)
    x_all, y, _ = prepare_xy(feature_frame, all_features, category_maps)

    selection = select_features(x_all, y)
    selected = selection["selected_features"]
    print("\nCFS-inspired selected features:")
    for name in selected:
        print(f"  - {name}")

    x_selected = x_all[selected]

    print("\nRunning 10-fold cross-validation (this may take a minute)...")
    comparison = evaluate_models(x_selected, y, n_splits=10)
    comparison.to_csv(METRICS_PATH, index=False)
    print("\nOur implementation — 10-fold CV results:")
    print(comparison.to_string(index=False))
    print(f"\nBest model in THIS experiment: {best_model_name(comparison)}")

    # Hold-out split is used only to fit the deployable Random Forest cleanly.
    x_train, x_test, y_train, y_test = train_test_split(
        x_selected,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=RANDOM_STATE,
    )
    forest.fit(x_train, y_train)
    holdout_accuracy = float(forest.score(x_test, y_test))
    print(f"Random Forest hold-out accuracy (not the paper figure): {holdout_accuracy:.4f}")

    importances = {
        feature: float(value)
        for feature, value in zip(selected, forest.feature_importances_)
    }

    bundle = {
        "model": forest,
        "selected_features": selected,
        "feature_columns": all_features,
        "category_maps": category_maps,
        "feature_importances": importances,
        "holdout_accuracy": holdout_accuracy,
        "class_labels": ["Low", "Medium", "High"],
    }
    joblib.dump(bundle, MODEL_PATH)

    artifacts = {
        "comparison": comparison,
        "gain_ratio_ranking": selection["gain_ratio_ranking"],
        "cfs_subset": selection["cfs_subset"],
        "cfs_history": selection["cfs_history"],
        "method_note": selection["method_note"],
        "best_model": best_model_name(comparison),
        "n_raw_students": int(len(raw)),
        "n_resampled": int(len(balanced)),
        "class_counts": raw[TARGET_COLUMN].value_counts().to_dict(),
        "feature_importances": importances,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"\nSaved Random Forest to {MODEL_PATH}")
    print(f"Saved comparison table to {METRICS_PATH}")
    return artifacts


if __name__ == "__main__":
    train()
