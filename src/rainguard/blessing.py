"""Rules that decide whether a candidate may replace the production model."""

from __future__ import annotations


def is_blessed(
    roc_auc: float,
    pos_recall: float,
    baseline_auc: float | None,
    model_mb: float,
    min_roc_auc: float,
    min_improvement: float,
    min_pos_recall: float,
    max_model_mb: float,
) -> tuple[bool, dict]:
    """Return the gate decision and each check, so a skip is explainable."""
    floor_ok = roc_auc >= min_roc_auc
    recall_ok = pos_recall >= min_pos_recall
    size_ok = model_mb <= max_model_mb
    if baseline_auc is None:
        improvement = None
        improve_ok = True
    else:
        improvement = roc_auc - baseline_auc
        improve_ok = improvement >= min_improvement
    blessed = floor_ok and recall_ok and size_ok and improve_ok
    return blessed, {
        "floor_ok": floor_ok,
        "recall_ok": recall_ok,
        "size_ok": size_ok,
        "improve_ok": improve_ok,
        "improvement": improvement,
        "roc_auc": roc_auc,
        "pos_recall": pos_recall,
        "model_mb": model_mb,
        "min_roc_auc": min_roc_auc,
        "min_pos_recall": min_pos_recall,
        "max_model_mb": max_model_mb,
    }
