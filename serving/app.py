"""Synchronous rain-risk API.

A request is rejected by the schema gate before the model sees it. That is the
cascade: cheap contract check, then the production model.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from rainguard import config
from rainguard.storage import load_json
from rainguard.validate import DataValidationError, validate_frame

app = FastAPI(title="RainGuard", version="0.1.0")
_lock = threading.Lock()
_state: dict[str, Any] = {"bundle": None, "mtime": None, "metadata": {}}

REQUESTS = Counter(
    "rainguard_requests_total",
    "HTTP requests",
    ["endpoint", "status"],
)
LATENCY = Histogram(
    "rainguard_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0),
)
RAIN = Counter("rainguard_predicted_rain_total", "Responses that warn of rain tomorrow")
DRY = Counter("rainguard_predicted_dry_total", "Responses that expect a dry tomorrow")
REJECTED = Counter("rainguard_schema_rejected_total", "Requests stopped by the schema gate")
DRIFT = Gauge("rainguard_drift_share", "Latest monitored drift share")
RECALL = Gauge("rainguard_realized_recall", "Latest realized rain recall")
AUC = Gauge("rainguard_model_roc_auc", "ROC AUC of the production model")


def _bundle() -> dict | None:
    path = config.SERVING_DIR / "model.joblib"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    with _lock:
        if _state["bundle"] is None or _state["mtime"] != mtime:
            _state["bundle"] = joblib.load(path)
            _state["mtime"] = mtime
            meta_path = config.SERVING_DIR / "metadata.json"
            _state["metadata"] = load_json(meta_path) if meta_path.exists() else {}
        return _state["bundle"]


def _refresh_gauges() -> None:
    meta = _state.get("metadata") or {}
    metrics = meta.get("metrics") or {}
    if "roc_auc" in metrics:
        AUC.set(float(metrics["roc_auc"]))
    history_path = config.REPORT_DIR / "monitoring_history.json"
    if not history_path.exists():
        return
    history = load_json(history_path)
    if not history:
        return
    last = history[-1]
    DRIFT.set(float(last.get("drift_share", 0)))
    RECALL.set(float(last.get("realized_recall", 0)))


def _frame_from_payload(payload: dict) -> pd.DataFrame:
    row = {column: payload.get(column) for column in [config.DATE_COL, "Location", *config.RAW_FEATURES]}
    return pd.DataFrame([row])


def _action(rain: bool) -> str:
    if rain:
        return "delay_spray_and_cover_harvest"
    return "proceed_field_work"


@app.get("/health")
def health():
    started = time.perf_counter()
    bundle = _bundle()
    if bundle is None:
        REQUESTS.labels(endpoint="health", status="not_ready").inc()
        return JSONResponse({"status": "not_ready"}, status_code=503)
    _refresh_gauges()
    meta = _state["metadata"]
    body = {
        "status": "ok",
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("version") or meta.get("version"),
        "threshold": bundle.get("threshold"),
        "slo": {
            "p50_ms": config.P50_SLO_MS,
            "p95_ms": config.P95_SLO_MS,
            "min_throughput_rps": config.MIN_THROUGHPUT_RPS,
        },
    }
    LATENCY.labels(endpoint="health").observe(time.perf_counter() - started)
    REQUESTS.labels(endpoint="health", status="ok").inc()
    return body


@app.get("/metrics")
def metrics():
    _bundle()
    _refresh_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: dict):
    started = time.perf_counter()
    bundle = _bundle()
    if bundle is None:
        REQUESTS.labels(endpoint="predict", status="not_ready").inc()
        return JSONResponse({"status": "not_ready"}, status_code=503)
    frame = _frame_from_payload(payload)
    try:
        checked = validate_frame(frame, require_target=False)
    except DataValidationError as exc:
        REJECTED.inc()
        REQUESTS.labels(endpoint="predict", status="rejected").inc()
        LATENCY.labels(endpoint="predict").observe(time.perf_counter() - started)
        return JSONResponse(
            {"decision": "rejected_by_schema_gate", "errors": str(exc)},
            status_code=422,
        )
    matrix = bundle["features"].transform(checked)
    probability = float(bundle["model"].predict_proba(matrix)[0, 1])
    rain = probability >= float(bundle["threshold"])
    if rain:
        RAIN.inc()
    else:
        DRY.inc()
    REQUESTS.labels(endpoint="predict", status="ok").inc()
    LATENCY.labels(endpoint="predict").observe(time.perf_counter() - started)
    return {
        "decision": "model",
        "rain_tomorrow": rain,
        "probability": round(probability, 6),
        "threshold": float(bundle["threshold"]),
        "action": _action(rain),
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("version"),
    }
