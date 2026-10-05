"""Turn a raw table into a frame that either passes the contract or stops the run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from rainguard import config
from rainguard.schema import WeatherFeatures, WeatherTraining
from rainguard.storage import dump_json


class DataValidationError(RuntimeError):
    """Raised when a batch breaks the contract. Callers must not train or score it."""


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop leak columns and coerce types before the contract runs."""
    prepared = frame.copy()
    leaks = [column for column in config.LEAK_COLUMNS if column in prepared.columns]
    if leaks:
        prepared = prepared.drop(columns=leaks)
    if config.DATE_COL in prepared.columns:
        prepared[config.DATE_COL] = pd.to_datetime(prepared[config.DATE_COL], errors="coerce")
    for column in config.NUMERIC:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in config.CATEGORICAL + [config.TARGET]:
        if column not in prepared.columns:
            continue
        text = prepared[column].astype("string").str.strip()
        text = text.replace({"": pd.NA, "NA": pd.NA, "nan": pd.NA, "None": pd.NA})
        prepared[column] = text
    return prepared


def format_schema_errors(exc: SchemaErrors) -> str:
    lines = ["validation failed"]
    failure = exc.failure_cases
    if failure is None or failure.empty:
        lines.append(str(exc))
        return "\n".join(lines)
    preview = failure.head(20)
    for row in preview.itertuples(index=False):
        column = getattr(row, "column", "?")
        check = getattr(row, "check", "?")
        index = getattr(row, "index", "?")
        case = getattr(row, "failure_case", "?")
        lines.append(f"- column={column} check={check} row={index} value={case}")
    lines.append(f"total_failures={len(failure)}")
    return "\n".join(lines)


def validate_frame(frame: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Return a coerced frame, or raise DataValidationError."""
    prepared = prepare_frame(frame)
    model: type[pa.DataFrameModel] = WeatherTraining if require_target else WeatherFeatures
    try:
        validated = model.validate(prepared, lazy=True)
    except SchemaErrors as exc:
        raise DataValidationError(format_schema_errors(exc)) from exc
    return validated


def validate_or_alert(
    frame: pd.DataFrame,
    alert_path: Path,
    *,
    require_target: bool,
) -> pd.DataFrame:
    """Validate and, on failure, write the alert file the operator reads."""
    try:
        validated = validate_frame(frame, require_target=require_target)
    except DataValidationError as exc:
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        alert_path.write_text(str(exc), encoding="utf-8")
        dump_json(
            alert_path.with_suffix(".json"),
            {"status": "fail", "message": str(exc)},
        )
        raise
    if alert_path.exists():
        alert_path.unlink()
    sidecar = alert_path.with_suffix(".json")
    if sidecar.exists():
        sidecar.unlink()
    return validated
