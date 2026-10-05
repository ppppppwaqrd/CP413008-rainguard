"""Time-ordered splits so the test set is the future, not a random draw."""

from __future__ import annotations

import pandas as pd

from rainguard import config


def labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that have a Yes/No label and sort them by date."""
    labeled = frame.dropna(subset=[config.TARGET]).copy()
    labeled[config.DATE_COL] = pd.to_datetime(labeled[config.DATE_COL])
    labeled = labeled.sort_values([config.DATE_COL, "Location"]).reset_index(drop=True)
    return labeled


def time_split(
    frame: pd.DataFrame,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split on unique dates so the same day never sits in two sets."""
    labeled = labeled_frame(frame)
    if labeled.empty:
        raise ValueError("no labeled rows to split")
    dates = labeled[config.DATE_COL].drop_duplicates().sort_values().to_list()
    if len(dates) < 3:
        raise ValueError("need at least three distinct dates for a time split")
    train_cut = max(int(len(dates) * train_fraction), 1)
    val_cut = max(int(len(dates) * (train_fraction + val_fraction)), train_cut + 1)
    val_cut = min(val_cut, len(dates) - 1)
    train_end = dates[train_cut - 1]
    val_end = dates[val_cut - 1]
    train = labeled[labeled[config.DATE_COL] <= train_end].reset_index(drop=True)
    val = labeled[
        (labeled[config.DATE_COL] > train_end) & (labeled[config.DATE_COL] <= val_end)
    ].reset_index(drop=True)
    test = labeled[labeled[config.DATE_COL] > val_end].reset_index(drop=True)
    if train.empty or val.empty or test.empty:
        raise ValueError("time split produced an empty partition")
    info = {
        "split": "time",
        "reason": "RainTomorrow is a forecast, so later dates stay out of training.",
        "train_end": str(pd.Timestamp(train_end).date()),
        "val_end": str(pd.Timestamp(val_end).date()),
        "test_end": str(pd.Timestamp(test[config.DATE_COL].max()).date()),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_positive_rate": float((train[config.TARGET] == config.POSITIVE_LABEL).mean()),
        "dropped_unlabeled": int(len(frame) - len(labeled)),
    }
    return train, val, test, info
