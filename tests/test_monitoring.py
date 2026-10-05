from __future__ import annotations

from rainguard.monitoring import (
    classify_window,
    flip_labels,
    psi_report,
    retrain_decision,
    shift_features,
)
from rainguard.synthetic import synthetic_weather


def test_shifted_features_cross_the_drift_limit():
    frame = synthetic_weather(500, seed=4)
    report = psi_report(frame, shift_features(frame))
    assert report["psi_share"] >= 0.30
    stable = psi_report(frame.iloc[:250], frame.iloc[250:])
    assert stable["psi_share"] < 0.30


def test_label_flip_is_concept_drift_not_data_drift():
    frame = synthetic_weather(300, seed=5)
    flipped = flip_labels(frame, rate=0.5, seed=9)
    share = psi_report(frame, flipped)["psi_share"]
    assert share < 0.30
    assert classify_window(share, realized_recall=0.42) == "concept_drift"
    assert classify_window(0.55, realized_recall=0.42) == "data_drift"
    assert classify_window(0.05, realized_recall=0.80) == "stable"


def test_retrain_policy_matches_the_two_triggers():
    calm = retrain_decision(
        [{"drift_share": 0.02, "realized_recall": 0.81}, {"drift_share": 0.03, "realized_recall": 0.79}]
    )
    assert not calm["retrain"]
    drifted = retrain_decision([{"drift_share": 0.44, "realized_recall": 0.70}])
    assert drifted["retrain"]
    assert "data_drift" in drifted["reasons"]
    one_bad_week = retrain_decision(
        [
            {"drift_share": 0.05, "realized_recall": 0.40},
            {"drift_share": 0.04, "realized_recall": 0.82},
        ]
    )
    assert "concept_drift_two_windows" not in one_bad_week["reasons"]
    two_bad_weeks = retrain_decision(
        [
            {"drift_share": 0.05, "realized_recall": 0.40},
            {"drift_share": 0.04, "realized_recall": 0.45},
        ]
    )
    assert two_bad_weeks["retrain"]
    assert "concept_drift_two_windows" in two_bad_weeks["reasons"]
    weekly = retrain_decision([], scheduled=True)
    assert weekly["retrain"]
    assert "scheduled_weekly" in weekly["reasons"]
