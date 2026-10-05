"""Cost-sensitive threshold and the scores we compare across experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rainguard import config


def encode_labels(values: np.ndarray | list) -> np.ndarray:
    text = np.asarray(values).astype(str)
    if not set(np.unique(text)).issubset({config.POSITIVE_LABEL, "No"}):
        raise ValueError("labels must be Yes or No")
    return (text == config.POSITIVE_LABEL).astype(int)


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, min_recall: float) -> float:
    """Pick the cheapest threshold that still catches enough rain on validation."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    best_cost: float | None = None
    best_t = 0.5
    fallback_t = 0.05
    fallback_recall = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = proba >= threshold
        recall = float(recall_score(y_true, predicted.astype(int), zero_division=0))
        if recall > fallback_recall:
            fallback_recall = recall
            fallback_t = float(threshold)
        if recall < min_recall:
            continue
        false_pos = int(np.sum(predicted & (y_true == 0)))
        false_neg = int(np.sum(~predicted & (y_true == 1)))
        cost = config.FP_COST * false_pos + config.FN_COST * false_neg
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_t = float(threshold)
    if best_cost is None:
        return fallback_t
    return best_t


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    false_pos = int(np.sum((predicted == 1) & (y_true == 0)))
    false_neg = int(np.sum((predicted == 0) & (y_true == 1)))
    precision = float(precision_score(y_true, predicted, zero_division=0))
    recall = float(recall_score(y_true, predicted, zero_division=0))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "recall": recall,
        "precision": precision,
        "f1": float(f1),
        "threshold": float(threshold),
        "cost": float(config.FP_COST * false_pos + config.FN_COST * false_neg),
        "fp": false_pos,
        "fn": false_neg,
        "predicted_positive_rate": float(predicted.mean()),
    }


def decision_cost(y_true: np.ndarray, predicted: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    predicted = np.asarray(predicted).astype(int)
    false_pos = int(np.sum((predicted == 1) & (y_true == 0)))
    false_neg = int(np.sum((predicted == 0) & (y_true == 1)))
    return float(config.FP_COST * false_pos + config.FN_COST * false_neg)
