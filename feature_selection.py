"""
Feature selection inspired by the reference paper.

The educational ensemble paper evaluates attributes with:
  1. Gain-Ratio Feature Evaluation
  2. Correlation-based Feature Selection (CFS Subset Evaluation)

scikit-learn does not ship WEKA's CFS or GainRatioAttributeEval.
This file therefore:
  - implements Gain Ratio ranking from information theory
  - implements a CFS-inspired greedy subset search using the Hall merit formula
  - labels both methods honestly (they are practical approximations, not WEKA)

Selected features are expected to include the behavioural and parental
attributes emphasised in the paper (absence, LMS activity, parent survey).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy of a discrete array."""
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _discretize_column(series: pd.Series, max_bins: int = 5) -> np.ndarray:
    """
    Gain Ratio in WEKA is defined on discrete attributes.
    Numeric LMS counts are therefore binned before ranking.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.nunique() > max_bins:
        try:
            return pd.qcut(numeric, q=max_bins, duplicates="drop").astype(str).to_numpy()
        except ValueError:
            return numeric.round().astype(str).to_numpy()
    return series.astype(str).to_numpy()


def gain_ratio_scores(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Rank features by Gain Ratio = InformationGain(y, x) / SplitInformation(x).

    This is the same idea as WEKA's GainRatioAttributeEval, implemented here
    in plain NumPy/pandas so the calculation is visible in source code.
    """
    y = np.asarray(y)
    class_entropy = _entropy(y)
    rows = []

    for column in x.columns:
        discrete = _discretize_column(x[column])
        split_info = _entropy(discrete)
        # Conditional entropy H(Y | X)
        residual = 0.0
        for value in np.unique(discrete):
            mask = discrete == value
            weight = mask.mean()
            residual += weight * _entropy(y[mask])
        information_gain = class_entropy - residual
        gain_ratio = information_gain / split_info if split_info > 1e-12 else 0.0
        rows.append(
            {
                "feature": column,
                "information_gain": round(float(information_gain), 4),
                "split_information": round(float(split_info), 4),
                "gain_ratio": round(float(gain_ratio), 4),
            }
        )

    ranking = pd.DataFrame(rows).sort_values("gain_ratio", ascending=False)
    ranking["gain_ratio_rank"] = range(1, len(ranking) + 1)
    return ranking.reset_index(drop=True)


def _symmetric_uncertainty(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlation proxy used in many CFS implementations:
    SU = 2 * IG(a, b) / (H(a) + H(b)).
    """
    ha = _entropy(a)
    hb = _entropy(b)
    residual = 0.0
    for value in np.unique(a):
        mask = a == value
        residual += mask.mean() * _entropy(b[mask])
    ig = hb - residual
    denom = ha + hb
    return float(2 * ig / denom) if denom > 1e-12 else 0.0


def cfs_merit(feature_matrix: pd.DataFrame, y: np.ndarray, subset: list[str]) -> float:
    """
    Hall's CFS merit:

        merit = k * r_cf / sqrt(k + k*(k-1)*r_ff)

    where r_cf is mean feature-class correlation and r_ff is mean
    feature-feature correlation. Correlations are symmetric uncertainty
    on discretised columns — a CFS-inspired measure, not WEKA CFS.
    """
    if not subset:
        return 0.0
    y_disc = y.astype(str)
    columns = {col: _discretize_column(feature_matrix[col]) for col in subset}
    class_corrs = [_symmetric_uncertainty(columns[col], y_disc) for col in subset]
    r_cf = float(np.mean(class_corrs))

    pair_corrs = []
    for i, col_i in enumerate(subset):
        for col_j in subset[i + 1 :]:
            pair_corrs.append(_symmetric_uncertainty(columns[col_i], columns[col_j]))
    r_ff = float(np.mean(pair_corrs)) if pair_corrs else 0.0
    k = len(subset)
    return float((k * r_cf) / np.sqrt(k + k * (k - 1) * r_ff))


def cfs_inspired_subset(
    x: pd.DataFrame,
    y: np.ndarray,
    max_features: int = 8,
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy forward search that maximises CFS merit.

    This is a teaching-oriented approximation of CFS Subset Evaluation.
    It is NOT sklearn.feature_selection.SelectKBest, and it is not WEKA's
    CfsSubsetEval. The formula and the subset idea are kept visible so they
    can be explained against the paper.
    """
    remaining = list(x.columns)
    chosen: list[str] = []
    history = []
    best_merit = -1.0

    while remaining and len(chosen) < max_features:
        trial_scores = []
        for candidate in remaining:
            trial = chosen + [candidate]
            trial_scores.append((cfs_merit(x, y, trial), candidate))
        trial_scores.sort(reverse=True)
        merit, best_candidate = trial_scores[0]
        if merit <= best_merit + 1e-6 and chosen:
            break
        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        best_merit = merit
        history.append(
            {
                "step": len(chosen),
                "added_feature": best_candidate,
                "cfs_merit": round(float(merit), 4),
                "subset": ", ".join(chosen),
            }
        )

    return chosen, pd.DataFrame(history)


def select_features(x: pd.DataFrame, y: np.ndarray) -> dict:
    """Run both paper-inspired selectors and return a combined view."""
    gain_ranking = gain_ratio_scores(x, y)
    cfs_features, cfs_history = cfs_inspired_subset(x, y)

    # Prefer the CFS-inspired subset when it is non-empty; otherwise top Gain Ratio.
    selected = cfs_features if cfs_features else gain_ranking["feature"].head(8).tolist()
    return {
        "gain_ratio_ranking": gain_ranking,
        "cfs_subset": selected,
        "cfs_history": cfs_history,
        "selected_features": selected,
        "method_note": (
            "Gain Ratio is implemented from information theory. "
            "CFS uses Hall's merit formula with a greedy forward search. "
            "Neither is WEKA's original evaluator."
        ),
    }
