"""Data drift, concept drift, and the rule that starts a retrain."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rainguard import config
from rainguard.metrics import binary_metrics, encode_labels


def shift_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Move sensor distributions while keeping every value inside the contract."""
    shifted = frame.copy()

    def _num(column: str) -> pd.Series:
        return pd.to_numeric(shifted[column], errors="coerce")

    shifted["Humidity9am"] = (_num("Humidity9am") * 0.35 + 62).clip(0, 100)
    shifted["Humidity3pm"] = (_num("Humidity3pm") * 0.35 + 62).clip(0, 100)
    shifted["Pressure9am"] = (_num("Pressure9am") - 8).clip(970, 1050)
    shifted["Pressure3pm"] = (_num("Pressure3pm") - 8).clip(970, 1050)
    shifted["Rainfall"] = (_num("Rainfall") + 8).clip(0, 500)
    shifted["MinTemp"] = (_num("MinTemp") + 5).clip(-20, 50)
    shifted["MaxTemp"] = (_num("MaxTemp") + 5).clip(-15, 55)
    shifted["Temp9am"] = (_num("Temp9am") + 6).clip(-20, 55)
    shifted["Temp3pm"] = (_num("Temp3pm") + 6).clip(-20, 55)
    shifted["WindSpeed9am"] = (_num("WindSpeed9am") + 15).clip(0, 200)
    shifted["WindSpeed3pm"] = (_num("WindSpeed3pm") + 15).clip(0, 200)
    return shifted


def flip_labels(frame: pd.DataFrame, rate: float = 0.45, seed: int = 7) -> pd.DataFrame:
    """Break the feature-label relationship without moving the features."""
    flipped = frame.copy()
    generator = np.random.default_rng(seed)
    mask = generator.random(len(flipped)) < rate
    swapped = flipped.loc[mask, config.TARGET].map({"Yes": "No", "No": "Yes"})
    flipped.loc[mask, config.TARGET] = swapped
    return flipped


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < bins or len(current) < bins:
        return 0.0
    cuts = reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float)
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return 0.0
    ref_counts = np.histogram(reference, bins=cuts)[0] / len(reference)
    cur_counts = np.histogram(current, bins=cuts)[0] / len(current)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def _total_variation(reference: pd.Series, current: pd.Series) -> float:
    ref_share = reference.astype(str).value_counts(normalize=True)
    cur_share = current.astype(str).value_counts(normalize=True)
    keys = set(ref_share.index) | set(cur_share.index)
    return float(0.5 * sum(abs(float(ref_share.get(key, 0)) - float(cur_share.get(key, 0))) for key in keys))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Share of columns whose distribution moved past a fixed PSI or TV limit."""
    drifted: list[dict] = []
    columns = [column for column in config.RAW_FEATURES if column in reference and column in current]
    for column in columns:
        if column in config.NUMERIC:
            score = _psi(reference[column], current[column])
            limit = 0.2
        else:
            score = _total_variation(reference[column], current[column])
            limit = 0.2
        if score >= limit:
            drifted.append({"column": column, "score": round(score, 4), "limit": limit})
    share = len(drifted) / max(len(columns), 1)
    return {
        "psi_share": share,
        "drifted_features": [item["column"] for item in drifted],
        "details": drifted,
        "n_columns": len(columns),
    }


def _find_drift_share(payload: Any) -> float | None:
    found: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = str(node.get("metric_id") or node.get("metric_name") or "")
            value = node.get("value")
            if "DriftedColumns" in name and isinstance(value, dict) and "share" in value:
                found.append(float(value["share"]))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(payload)
    return found[-1] if found else None


def evidently_share(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run Evidently's data-drift preset. Returns an error string instead of raising."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        return {"evidently_share": None, "error": str(exc)}
    columns = [column for column in config.RAW_FEATURES if column in reference and column in current]
    numeric = [column for column in columns if column in config.NUMERIC]
    categorical = [column for column in columns if column in config.CATEGORICAL]
    prepared_ref = _prepare_monitor_frame(reference, numeric, categorical)
    prepared_cur = _prepare_monitor_frame(current, numeric, categorical)
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    try:
        report = Report([DataDriftPreset(drift_share=config.DRIFT_SHARE_LIMIT)])
        snapshot = report.run(
            Dataset.from_pandas(prepared_cur, data_definition=definition),
            Dataset.from_pandas(prepared_ref, data_definition=definition),
        )
    except Exception as exc:
        return {"evidently_share": None, "error": str(exc), "snapshot": None}
    payload: dict[str, Any]
    if hasattr(snapshot, "dict") and callable(snapshot.dict):
        raw = snapshot.dict()
        payload = raw if isinstance(raw, dict) else {"value": raw}
    else:
        payload = {"repr": str(snapshot)}
    share = _find_drift_share(payload)
    return {"evidently_share": share, "error": None, "snapshot": snapshot}


def _prepare_monitor_frame(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    prepared = frame[numeric + categorical].copy()
    for column in numeric:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in categorical:
        prepared[column] = prepared[column].astype("string").fillna("Unknown").astype(str)
    return prepared


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    use_evidently: bool = True,
) -> dict:
    psi = psi_report(reference, current)
    evidently = {"evidently_share": None, "error": None, "snapshot": None}
    if use_evidently:
        evidently = evidently_share(reference, current)
    shares = [float(psi["psi_share"])]
    if evidently.get("evidently_share") is not None:
        shares.append(float(evidently["evidently_share"]))
    return {
        "drift_share": max(shares),
        "psi_share": psi["psi_share"],
        "evidently_share": evidently.get("evidently_share"),
        "drifted_features": psi["drifted_features"],
        "method": "evidently+psi" if evidently.get("evidently_share") is not None else "psi",
        "evidently_error": evidently.get("error"),
        "snapshot": evidently.get("snapshot"),
    }


def score_labeled(bundle: dict, frame: pd.DataFrame) -> dict:
    labels = encode_labels(frame[config.TARGET])
    matrix = bundle["features"].transform(frame)
    proba = np.asarray(bundle["model"].predict_proba(matrix))[:, 1]
    scores = binary_metrics(labels, proba, float(bundle["threshold"]))
    return scores


def classify_window(drift_share: float, realized_recall: float) -> str:
    """Input shift is data drift. A recall drop on stable inputs is concept drift."""
    if drift_share >= config.DRIFT_SHARE_LIMIT:
        return "data_drift"
    if realized_recall < config.RECALL_RETRAIN:
        return "concept_drift"
    return "stable"


def monitor_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    bundle: dict,
    *,
    use_evidently: bool = True,
    window: str = "week",
) -> dict:
    drift = drift_report(reference, current, use_evidently=use_evidently)
    scores = score_labeled(bundle, current)
    kind = classify_window(drift["drift_share"], scores["recall"])
    record = {
        "window": window,
        "drift_share": round(float(drift["drift_share"]), 4),
        "psi_share": round(float(drift["psi_share"]), 4),
        "evidently_share": drift["evidently_share"],
        "drift_method": drift["method"],
        "drifted_features": drift["drifted_features"],
        "realized_recall": round(float(scores["recall"]), 4),
        "realized_roc_auc": round(float(scores["roc_auc"]), 4),
        "kind": kind,
        "data_drift": kind == "data_drift",
        "concept_drift": kind == "concept_drift",
    }
    snapshot = drift.get("snapshot")
    record["_snapshot"] = snapshot
    return record


def retrain_decision(history: list[dict], *, scheduled: bool = False) -> dict:
    """Weekly schedule, a drifted batch, or two weak recall windows start a retrain."""
    reasons: list[str] = []
    if scheduled:
        reasons.append("scheduled_weekly")
    if history:
        last = history[-1]
        if float(last.get("drift_share", 0)) >= config.DRIFT_SHARE_LIMIT:
            reasons.append("data_drift")
        window = history[-config.RETRAIN_WINDOWS :]
        recalls = [row.get("realized_recall") for row in window]
        if len(window) >= config.RETRAIN_WINDOWS and all(
            recall is not None and float(recall) < config.RECALL_RETRAIN for recall in recalls
        ):
            reasons.append("concept_drift_two_windows")
    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "policy": (
            "Retrain every Monday 06:00, or as soon as drift_share >= "
            f"{config.DRIFT_SHARE_LIMIT:.2f}, or when realized rain recall stays under "
            f"{config.RECALL_RETRAIN:.2f} for {config.RETRAIN_WINDOWS} windows."
        ),
    }
