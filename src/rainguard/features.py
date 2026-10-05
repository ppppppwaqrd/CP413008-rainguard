"""The only feature transformer. Training and serving both call this object."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rainguard import config


class FeatureBuilder:
    """Fit medians, missing flags, calendar fields, scaling, and one-hot on train only."""

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, frame: pd.DataFrame) -> FeatureBuilder:
        numeric = self._numeric_block(frame, fit=True)
        categorical = self._categorical_block(frame)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(categorical)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(numeric)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.encoder_ is None or self.scaler_ is None or self.medians_ is None:
            raise RuntimeError("FeatureBuilder.transform called before fit")
        numeric = self.scaler_.transform(self._numeric_block(frame, fit=False))
        encoded = self.encoder_.transform(self._categorical_block(frame))
        return np.hstack([numeric, encoded])

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _numeric_block(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        raw = frame.reindex(columns=config.NUMERIC).apply(pd.to_numeric, errors="coerce")
        raw = raw.reset_index(drop=True)
        if fit:
            self.medians_ = raw.median(numeric_only=True).fillna(0.0)
        if self.medians_ is None:
            raise RuntimeError("medians are missing")
        filled = raw.fillna(self.medians_)
        flags = raw[config.MISSING_INDICATORS].isna().astype(float)
        dates = pd.to_datetime(frame[config.DATE_COL]).reset_index(drop=True)
        calendar = pd.DataFrame(
            {
                "month": dates.dt.month.astype(float),
                "day_of_year": dates.dt.dayofyear.astype(float),
            }
        )
        block = pd.concat([filled, flags, calendar], axis=1)
        return block.to_numpy(dtype=float)

    def _categorical_block(self, frame: pd.DataFrame) -> pd.DataFrame:
        categorical = frame.reindex(columns=config.CATEGORICAL).copy()
        for column in config.CATEGORICAL:
            categorical[column] = (
                categorical[column].astype("string").fillna("Unknown").replace({"<NA>": "Unknown"})
            )
            categorical[column] = categorical[column].astype(str)
        return categorical
