"""Physical data contract for weather rows.

Bounds are physical limits, not the min and max of one sample. A later season
may be wetter without being corrupt. Corrupt values fall outside these bounds
and are rejected before training or scoring.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

from rainguard import config

# Inclusive numeric limits. Nulls are allowed because several sensors are
# missing on a large share of days; the shared transformer imputes them.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "MinTemp": (-20.0, 50.0),
    "MaxTemp": (-15.0, 55.0),
    "Rainfall": (0.0, 500.0),
    "Evaporation": (0.0, 200.0),
    "Sunshine": (0.0, 24.0),
    "WindGustSpeed": (0.0, 200.0),
    "WindSpeed9am": (0.0, 200.0),
    "WindSpeed3pm": (0.0, 200.0),
    "Humidity9am": (0.0, 100.0),
    "Humidity3pm": (0.0, 100.0),
    "Pressure9am": (970.0, 1050.0),
    "Pressure3pm": (970.0, 1050.0),
    "Cloud9am": (0.0, 9.0),
    "Cloud3pm": (0.0, 9.0),
    "Temp9am": (-20.0, 55.0),
    "Temp3pm": (-20.0, 55.0),
}

YES_NO = ["Yes", "No"]


class WeatherFeatures(pa.DataFrameModel):
    """Raw fields accepted by both training and the live API."""

    Date: Series[pa.DateTime] = pa.Field(nullable=False)
    Location: Series[str] = pa.Field(nullable=False)
    MinTemp: Series[float] = pa.Field(ge=-20, le=50, nullable=True)
    MaxTemp: Series[float] = pa.Field(ge=-15, le=55, nullable=True)
    Rainfall: Series[float] = pa.Field(ge=0, le=500, nullable=True)
    Evaporation: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    Sunshine: Series[float] = pa.Field(ge=0, le=24, nullable=True)
    WindGustDir: Series[str] = pa.Field(nullable=True)
    WindGustSpeed: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    WindDir9am: Series[str] = pa.Field(nullable=True)
    WindDir3pm: Series[str] = pa.Field(nullable=True)
    WindSpeed9am: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    WindSpeed3pm: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    Humidity9am: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    Humidity3pm: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    Pressure9am: Series[float] = pa.Field(ge=970, le=1050, nullable=True)
    Pressure3pm: Series[float] = pa.Field(ge=970, le=1050, nullable=True)
    Cloud9am: Series[float] = pa.Field(ge=0, le=9, nullable=True)
    Cloud3pm: Series[float] = pa.Field(ge=0, le=9, nullable=True)
    Temp9am: Series[float] = pa.Field(ge=-20, le=55, nullable=True)
    Temp3pm: Series[float] = pa.Field(ge=-20, le=55, nullable=True)
    RainToday: Series[str] = pa.Field(isin=YES_NO, nullable=True)

    class Config:
        strict = False
        coerce = True


class WeatherTraining(WeatherFeatures):
    """Training rows may omit the label; those rows are dropped after the check."""

    RainTomorrow: Series[str] = pa.Field(isin=YES_NO, nullable=True)


def contract_document() -> dict:
    """Serializable copy of the contract, written next to every run."""
    return {
        "version": "weather-contract-v1",
        "target": config.TARGET,
        "positive_label": config.POSITIVE_LABEL,
        "numeric_bounds": {
            name: {"min": low, "max": high} for name, (low, high) in NUMERIC_BOUNDS.items()
        },
        "yes_no_columns": ["RainToday", config.TARGET],
        "required": ["Date", "Location"],
        "leak_columns_dropped": list(config.LEAK_COLUMNS),
        "notes": (
            "Null sensor readings are allowed and imputed by FeatureBuilder. "
            "Values outside physical bounds are rejected."
        ),
    }
