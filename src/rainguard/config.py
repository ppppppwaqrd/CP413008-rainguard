"""Paths, columns, and the gates the rest of the system shares."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "weatherAUS.csv"
ARTIFACT_DIR = ROOT / "artifacts"
SERVING_DIR = ARTIFACT_DIR / "serving"
MLRUNS_DIR = ROOT / "mlruns"
REPORT_DIR = ROOT / "reports"
ALERT_DIR = ROOT / "alerts"

TARGET = "RainTomorrow"
POSITIVE_LABEL = "Yes"
DATE_COL = "Date"
REGISTRY_NAME = "rainguard"
EXPERIMENT_NAME = "rainguard"

# RISK_MM is a same-day rainfall amount that leaks RainTomorrow in some releases.
LEAK_COLUMNS = ["RISK_MM"]

NUMERIC = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
]
# Missingness itself is a signal for these sparsely observed sensors.
MISSING_INDICATORS = ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
CATEGORICAL = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday",
]
RAW_FEATURES = NUMERIC + CATEGORICAL

# A missed rain (false negative) wastes a spray and can damage a harvest.
# A false alarm only delays field work, so it is cheaper.
FN_COST = 5.0
FP_COST = 1.0

MIN_POS_RECALL = 0.70
MIN_ROC_AUC = 0.75
MIN_IMPROVEMENT = 0.0
MAX_MODEL_MB = 25.0
# Threshold search aims a little above the test gate so the holdout can still clear it.
VAL_RECALL_FLOOR = 0.72

P50_SLO_MS = 100.0
P95_SLO_MS = 150.0
MIN_THROUGHPUT_RPS = 10.0

DRIFT_SHARE_LIMIT = 0.30
RECALL_RETRAIN = 0.60
RETRAIN_WINDOWS = 2

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

DOWNLOAD_URLS = [
    "https://rattle.togaware.com/weatherAUS.csv",
    "https://cdn.jsdelivr.net/gh/sharmaroshan/Weather-Data@master/weatherAUS.csv",
    "https://raw.githubusercontent.com/sharmaroshan/Weather-Data/master/weatherAUS.csv",
    "https://raw.githubusercontent.com/Anjan50/ML-Model-for-Weather-dataset/master/weatherAUS.csv",
]
