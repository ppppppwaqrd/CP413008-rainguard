from __future__ import annotations

from pathlib import Path

import numpy as np

from rainguard import config
from rainguard.blessing import is_blessed
from rainguard.metrics import binary_metrics, choose_threshold, decision_cost
from rainguard.split import time_split
from rainguard.synthetic import synthetic_weather
from rainguard.train import train_candidates


def test_blessing_rejects_a_weak_or_oversized_model():
    ok, gates = is_blessed(0.90, 0.80, 0.85, 2.0, 0.75, 0.0, 0.70, 25.0)
    assert ok
    assert gates["improve_ok"]
    weak, weak_gates = is_blessed(0.90, 0.40, 0.85, 2.0, 0.75, 0.0, 0.70, 25.0)
    assert not weak
    assert not weak_gates["recall_ok"]
    huge, huge_gates = is_blessed(0.90, 0.80, 0.85, 40.0, 0.75, 0.0, 0.70, 25.0)
    assert not huge
    assert not huge_gates["size_ok"]


def test_cost_threshold_catches_more_rain_than_a_half_cutoff():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.4, 0.35, 0.3, 0.2, 0.1, 0.05])
    threshold = choose_threshold(y_true, proba, min_recall=0.72)
    chosen = (proba >= threshold).astype(int)
    half = (proba >= 0.5).astype(int)
    assert decision_cost(y_true, chosen) <= decision_cost(y_true, half)
    assert binary_metrics(y_true, proba, threshold)["recall"] >= 0.72


def test_three_models_clear_the_quality_gate(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "MLRUNS_DIR", tmp_path / "mlruns")
    train, val, test, _info = time_split(synthetic_weather(480, seed=2))
    records = train_candidates(
        train,
        val,
        test,
        tmp_path,
        {"sha256": "synthetic", "file": "synthetic.csv", "bytes": 1},
    )
    assert {row["name"] for row in records} == {"logreg", "hist_gb", "random_forest"}
    winner = next(row for row in records if row["selected"])
    assert winner["metrics"]["roc_auc"] >= config.MIN_ROC_AUC
    assert winner["metrics"]["recall"] >= config.MIN_POS_RECALL
    assert winner["model_mb"] <= config.MAX_MODEL_MB
    logged = {"code_version", "data_version", "environment", "params", "metrics", "run_id"}
    assert logged <= set(winner)
