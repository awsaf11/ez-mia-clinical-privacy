from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def _validate_binary_inputs(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if labels.size != scores.size:
        raise ValueError("labels and scores must have the same length.")
    if labels.size == 0:
        raise ValueError("labels and scores cannot be empty.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinite values.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both classes: 0 and 1.")

    return labels, scores


def tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> float:
    """Interpolate TPR at a requested false-positive rate."""
    labels, scores = _validate_binary_inputs(labels, scores)

    fpr_values, tpr_values, _ = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))

    return float(
        np.interp(
            target_fpr,
            np.clip(fpr_values, 0.0, 1.0),
            np.clip(tpr_values, 0.0, 1.0),
        )
    )


def classification_metrics_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = 0.001,
) -> Dict[str, float | int]:
    """Compute thresholded metrics at the best ROC point not exceeding target FPR.

    A larger score is assumed to mean stronger evidence of membership.
    """
    labels, scores = _validate_binary_inputs(labels, scores)

    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    fpr_values, tpr_values, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    eligible = np.flatnonzero(fpr_values <= target_fpr)
    if eligible.size == 0:
        selected_index = 0
    else:
        eligible_tpr = tpr_values[eligible]
        best_tpr = np.max(eligible_tpr)
        candidates = eligible[eligible_tpr == best_tpr]
        candidate_fprs = fpr_values[candidates]
        selected_index = int(candidates[np.argmin(candidate_fprs)])

    threshold = float(thresholds[selected_index])
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "actual_fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }