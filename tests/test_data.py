from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rainguard import config
from rainguard.features import FeatureBuilder
from rainguard.pipeline import do_example_gen, execute_run
from rainguard.split import time_split
from rainguard.synthetic import synthetic_weather
from rainguard.validate import DataValidationError, validate_frame, validate_or_alert


def test_time_split_keeps_the_future_out_of_training():
    train, val, test, info = time_split(synthetic_weather(400))
    assert train[config.DATE_COL].max() < val[config.DATE_COL].min()
    assert val[config.DATE_COL].max() < test[config.DATE_COL].min()
    assert info["train_rows"] > info["test_rows"]


def test_contract_accepts_a_normal_day():
    checked = validate_frame(synthetic_weather(12), require_target=True)
    assert len(checked) == 12


def test_negative_rainfall_stops_and_writes_an_alert(tmp_path: Path):
    frame = synthetic_weather(20)
    frame.loc[0, "Rainfall"] = -5
    alert = tmp_path / "validation_error.txt"
    with pytest.raises(DataValidationError, match="Rainfall"):
        validate_or_alert(frame, alert, require_target=True)
    text = alert.read_text(encoding="utf-8")
    assert "validation failed" in text
    assert "Rainfall" in text


def test_unknown_label_is_rejected():
    frame = synthetic_weather(8)
    frame.loc[1, "RainTomorrow"] = "Maybe"
    with pytest.raises(DataValidationError):
        validate_frame(frame, require_target=True)


def test_single_row_transform_matches_the_batch():
    frame = synthetic_weather(80)
    builder = FeatureBuilder().fit(frame)
    batch = builder.transform(frame.iloc[:6])
    alone = builder.transform(frame.iloc[[4]])
    assert batch.shape[1] == alone.shape[1]
    assert abs(float(batch[4].sum() - alone[0].sum())) < 1e-6


def test_leak_column_is_removed_before_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path / "artifacts")
    frame = synthetic_weather(15)
    frame["RISK_MM"] = 99
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    run = tmp_path / "run"
    run.mkdir()
    do_example_gen(run, path)
    stored = pd.read_csv(run / "examples.csv")
    assert "RISK_MM" not in stored.columns


def test_pipeline_stops_before_training_when_data_is_bad(tmp_path, monkeypatch):
    for name in ("ARTIFACT_DIR", "SERVING_DIR", "MLRUNS_DIR", "REPORT_DIR", "ALERT_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    frame = synthetic_weather(30)
    frame.loc[0, "Humidity3pm"] = 140
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        execute_run(path, run_id="bad-batch")
    assert not (tmp_path / "serving_dir" / "model.joblib").exists()
    assert (tmp_path / "alert_dir" / "validation_error.txt").exists()
