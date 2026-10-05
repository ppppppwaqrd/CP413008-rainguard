from __future__ import annotations

import joblib
from fastapi.testclient import TestClient
from serving.app import app

from rainguard import config
from rainguard.split import time_split
from rainguard.synthetic import synthetic_weather
from rainguard.train import train_candidates


def _install_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLRUNS_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(config, "SERVING_DIR", tmp_path / "serving")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "reports")
    train, val, test, _info = time_split(synthetic_weather(360, seed=3))
    records = train_candidates(
        train,
        val,
        test,
        tmp_path / "run",
        {"sha256": "synthetic", "file": "synthetic.csv", "bytes": 1},
        only=["logreg"],
    )
    winner = next(row for row in records if row["selected"])
    bundle = joblib.load(winner["bundle_path"])
    bundle["version"] = "1"
    config.SERVING_DIR.mkdir(parents=True)
    joblib.dump(bundle, config.SERVING_DIR / "model.joblib")
    return test.iloc[0]


def test_health_predict_and_schema_gate(tmp_path, monkeypatch):
    row = _install_model(tmp_path, monkeypatch)
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["slo"]["p95_ms"] == 150

    payload = row.drop(labels=["RainTomorrow"]).to_dict()
    payload["Date"] = str(payload["Date"])[:10]
    for key, value in list(payload.items()):
        if value != value:
            payload[key] = None
    predicted = client.post("/predict", json=payload)
    assert predicted.status_code == 200
    body = predicted.json()
    assert body["decision"] == "model"
    assert body["action"] in {"delay_spray_and_cover_harvest", "proceed_field_work"}
    assert body["model_version"] == "1"

    payload["Rainfall"] = -5
    rejected = client.post("/predict", json=payload)
    assert rejected.status_code == 422
    assert rejected.json()["decision"] == "rejected_by_schema_gate"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert b"rainguard_requests_total" in metrics.content


def test_dag_declares_the_full_path():
    text = (config.ROOT / "dags" / "rain_train.py").read_text(encoding="utf-8")
    for name in (
        "example_gen",
        "example_validator",
        "transform",
        "trainer",
        "blessing_gate",
        "pusher",
        "skip_push",
        "0 6 * * 1",
    ):
        assert name in text
