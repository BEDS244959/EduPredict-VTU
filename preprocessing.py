"""
Data preprocessing for EduPredict AI.

This module follows the educational-data workflow used in the reference paper:
load LMS-style student records, clean them, encode categorical attributes,
handle missing values, and prepare X/y for classification.

The original Kalboard 360 / xAPI-Edu-Data study stores absence as a
categorical band (Under-7 / Above-7). This mini-project uses an integer
absence count so the demonstration form can accept exact day values
(0, 1, 3, 7, 15, ...), while keeping the same behavioural meaning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "student_data.csv"

# Display names used throughout the app (easy to explain in a viva).
FEATURE_COLUMNS = [
    "gender",
    "nationality",
    "relation",
    "raised_hands",
    "visited_resources",
    "announcements_view",
    "discussion",
    "parent_answering_survey",
    "parent_school_satisfaction",
    "student_absence_days",
]
TARGET_COLUMN = "performance"
ID_COLUMN = "student_id"

CATEGORICAL_COLUMNS = [
    "gender",
    "nationality",
    "relation",
    "parent_answering_survey",
    "parent_school_satisfaction",
]
NUMERIC_COLUMNS = [
    "raised_hands",
    "visited_resources",
    "announcements_view",
    "discussion",
    "student_absence_days",
]

# Performance labels used in the xAPI-Edu-Data family of papers: L / M / H.
CLASS_ORDER = ["Low", "Medium", "High"]

NATIONALITIES = [
    "Indian",
    "Jordan",
    "Kuwait",
    "Palestine",
    "Iraq",
    "Saudi Arabia",
    "Egypt",
    "USA",
    "Lebanon",
    "Tunisia",
    "Other",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and standardise column names to snake_case."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.lower()
    )
    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining missing values with simple, explainable rules."""
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    for col in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        if col in out.columns:
            mode_value = out[col].mode(dropna=True)
            fill = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            out[col] = out[col].fillna(fill)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering step from the paper workflow.

    Engagement is the mean of the four LMS activity counts discussed in the
    Kalboard 360 / xAPI studies (raised hands, visited resources,
    announcements view, discussion).
    """
    out = df.copy()
    out["engagement_score"] = out[
        ["raised_hands", "visited_resources", "announcements_view", "discussion"]
    ].mean(axis=1)
    return out


def load_raw_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the local educational CSV used by this mini-project."""
    path = Path(csv_path) if csv_path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run train_model.py to generate it."
        )
    df = pd.read_csv(path)
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = add_engineered_features(df)
    return df


def resample_minority_classes(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Simple random oversampling of minority performance classes.

    The reference paper includes a resampling / class-imbalance step.
    This implementation does NOT use SMOTE or imbalanced-learn; it copies
    existing minority rows until class counts match the majority class.
    That keeps the dependency list small and the code easy to explain.
    """
    rng = np.random.default_rng(random_state)
    counts = df[TARGET_COLUMN].value_counts()
    majority_size = int(counts.max())
    parts = []
    for label, group in df.groupby(TARGET_COLUMN, sort=False):
        parts.append(group)
        shortage = majority_size - len(group)
        if shortage > 0:
            extra_idx = rng.choice(group.index.to_numpy(), size=shortage, replace=True)
            parts.append(df.loc[extra_idx])
    resampled = pd.concat(parts, ignore_index=True)
    return resampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_category_maps(df: pd.DataFrame) -> dict:
    """Map each categorical value to an integer code (stable, sorted)."""
    maps = {}
    for col in CATEGORICAL_COLUMNS:
        values = sorted(df[col].astype(str).unique().tolist())
        maps[col] = {value: index for index, value in enumerate(values)}
    return maps


def encode_frame(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Replace categorical labels with integer codes using the training maps."""
    encoded = df.copy()
    for col, mapping in category_maps.items():
        unknown_code = mapping.get("Other", 0)
        encoded[col] = (
            encoded[col]
            .astype(str)
            .map(mapping)
            .fillna(unknown_code)
            .astype(int)
        )
    return encoded


def encode_target(series: pd.Series) -> tuple[np.ndarray, dict]:
    """Encode Low/Medium/High as 0/1/2 so MAE can be computed on class ranks."""
    mapping = {label: index for index, label in enumerate(CLASS_ORDER)}
    encoded = series.astype(str).map(mapping).to_numpy()
    return encoded, mapping


def prepare_xy(
    df: pd.DataFrame,
    feature_names: list[str],
    category_maps: dict,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Separate features and target, then encode categoricals."""
    work = encode_frame(df, category_maps)
    x = work[feature_names].copy()
    y, target_map = encode_target(work[TARGET_COLUMN])
    return x, y, target_map


def generate_synthetic_dataset(n_students: int = 480, random_state: int = 42) -> pd.DataFrame:
    """
    Create a labelled synthetic sample inspired by xAPI-Edu-Data attributes.

    The original public file is not stored in this project folder, so this
    function builds a non-identifying sample with the same kinds of
    demographic, academic-behaviour and parental attributes described in
    the paper. No real student identities are used.
    """
    rng = np.random.default_rng(random_state)

    nationalities = rng.choice(
        NATIONALITIES,
        size=n_students,
        p=[0.28, 0.14, 0.12, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.05],
    )
    genders = rng.choice(["Male", "Female"], size=n_students, p=[0.55, 0.45])
    relations = rng.choice(["Father", "Mum", "Other"], size=n_students, p=[0.52, 0.40, 0.08])
    parent_survey = rng.choice(["Yes", "No"], size=n_students, p=[0.58, 0.42])
    parent_sat = rng.choice(["Good", "Bad"], size=n_students, p=[0.62, 0.38])

    # Latent study-engagement drives the four LMS activity counts.
    latent = rng.beta(2.2, 2.0, n_students)
    noise = rng.normal(0, 8, n_students)
    raised = np.clip(latent * 95 + noise, 0, 100).round().astype(int)
    visited = np.clip(latent * 100 + rng.normal(0, 7, n_students), 0, 100).round().astype(int)
    announces = np.clip(latent * 85 + rng.normal(0, 9, n_students), 0, 100).round().astype(int)
    discussion = np.clip(latent * 80 + rng.normal(0, 10, n_students), 0, 100).round().astype(int)

    # Absence is a count of days (integer), not the original Under-7/Above-7 band.
    absence = np.clip(rng.poisson(lam=4.5 + (1 - latent) * 6, size=n_students), 0, 28)

    parent_bonus = np.where(parent_survey == "Yes", 8, -4) + np.where(parent_sat == "Good", 6, -3)
    relation_bonus = np.where(relations == "Mum", 4, np.where(relations == "Father", 1, -2))

    score = (
        0.22 * raised
        + 0.24 * visited
        + 0.16 * announces
        + 0.10 * discussion
        - 3.2 * absence
        + parent_bonus
        + relation_bonus
        + rng.normal(0, 6, n_students)
    )

    # Three-class target used in the educational ensemble papers: Low / Medium / High.
    low_cut, high_cut = np.quantile(score, [0.30, 0.68])
    performance = np.where(score < low_cut, "Low", np.where(score < high_cut, "Medium", "High"))

    df = pd.DataFrame(
        {
            "student_id": [f"STU{i:03d}" for i in range(1, n_students + 1)],
            "gender": genders,
            "nationality": nationalities,
            "relation": relations,
            "raised_hands": raised,
            "visited_resources": visited,
            "announcements_view": announces,
            "discussion": discussion,
            "parent_answering_survey": parent_survey,
            "parent_school_satisfaction": parent_sat,
            "student_absence_days": absence,
            "performance": performance,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, csv_path: Path | None = None) -> Path:
    path = Path(csv_path) if csv_path else DATA_PATH
    df.to_csv(path, index=False)
    return path
