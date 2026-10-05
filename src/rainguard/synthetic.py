"""Synthetic weather rows for tests. The label follows humidity and rainfall."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_weather(n: int = 420, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    humidity = rng.uniform(20, 100, n)
    rainfall = rng.uniform(0, 30, n)
    rain = (humidity > 72) | (rainfall > 12)
    rain = rain ^ (rng.random(n) < 0.04)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2012-01-01", periods=n, freq="D"),
            "Location": rng.choice(["Albury", "Sydney", "Darwin", "Perth"], n),
            "MinTemp": rng.normal(12, 3, n),
            "MaxTemp": rng.normal(24, 4, n),
            "Rainfall": rainfall,
            "Evaporation": rng.choice([np.nan, 4.0, 6.0, 8.0], n),
            "Sunshine": rng.choice([np.nan, 2.0, 8.0, 11.0], n),
            "WindGustDir": rng.choice(["N", "SE", "W", "NW"], n),
            "WindGustSpeed": rng.uniform(10, 60, n),
            "WindDir9am": rng.choice(["N", "SE", "W"], n),
            "WindDir3pm": rng.choice(["N", "S", "E"], n),
            "WindSpeed9am": rng.uniform(0, 30, n),
            "WindSpeed3pm": rng.uniform(0, 40, n),
            "Humidity9am": np.clip(humidity - 5, 0, 100),
            "Humidity3pm": humidity,
            "Pressure9am": rng.normal(1015, 4, n),
            "Pressure3pm": rng.normal(1013, 4, n),
            "Cloud9am": rng.choice([np.nan, 1, 4, 7], n),
            "Cloud3pm": rng.choice([np.nan, 2, 5, 8], n),
            "Temp9am": rng.normal(16, 3, n),
            "Temp3pm": rng.normal(22, 3, n),
            "RainToday": np.where(rainfall > 1, "Yes", "No"),
            "RainTomorrow": np.where(rain, "Yes", "No"),
        }
    )
