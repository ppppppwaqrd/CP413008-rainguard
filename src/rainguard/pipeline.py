"""File-based steps shared by the one-shot command and the Airflow DAG."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from rainguard import config
from rainguard.features import FeatureBuilder
from rainguard.monitoring import flip_labels, monitor_window, retrain_decision, shift_features
from rainguard.registry import list_versions, production_version, register_bundle, set_production
from rainguard.schema import contract_document
from rainguard.split import labeled_frame, time_split
from rainguard.storage import dump_json, load_json
from rainguard.train import train_candidates
from rainguard.validate import DataValidationError, validate_or_alert


def prepare_run(run_id: str | None = None) -> Path:
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = run_id.replace(":", "_").replace("/", "_")
    path = config.ARTIFACT_DIR / "runs" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def do_example_gen(run: Path, data_file: Path) -> None:
    if not data_file.exists():
        raise FileNotFoundError(f"missing {data_file}. Run python scripts/download_data.py")
    frame = pd.read_csv(data_file)
    leaked = [column for column in config.LEAK_COLUMNS if column in frame.columns]
    if leaked:
        frame = frame.drop(columns=leaked)
    before = int(len(frame))
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame.to_csv(run / "examples.csv", index=False)
    dump_json(
        run / "example_gen.json",
        {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "data_file": data_file.name,
            "sha256": _sha256(data_file),
            "bytes": int(data_file.stat().st_size),
            "dropped_leak_columns": leaked,
            "dropped_duplicate_rows": before - int(len(frame)),
        },
    )


def do_statistics(run: Path) -> None:
    frame = pd.read_csv(run / "examples.csv")
    numeric_stats = {}
    for column in config.NUMERIC:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        numeric_stats[column] = {
            "missing_ratio": float(series.isna().mean()),
            "min": None if series.dropna().empty else float(series.min()),
            "max": None if series.dropna().empty else float(series.max()),
            "mean": None if series.dropna().empty else float(series.mean()),
            "std": None if series.dropna().empty else float(series.std()),
        }
    categorical_stats = {}
    for column in config.CATEGORICAL + [config.TARGET]:
        if column not in frame.columns:
            continue
        counts = frame[column].astype("string").fillna("Missing").value_counts().head(60)
        categorical_stats[column] = {str(key): int(value) for key, value in counts.items()}
    dump_json(
        run / "statistics.json",
        {"rows": int(len(frame)), "numeric": numeric_stats, "categorical": categorical_stats},
    )


def do_schema(run: Path) -> None:
    document = contract_document()
    dump_json(run / "schema.json", document)
    dump_json(config.ARTIFACT_DIR / "schema.json", document)


def do_validate(run: Path) -> None:
    frame = pd.read_csv(run / "examples.csv")
    alert = config.ALERT_DIR / "validation_error.txt"
    try:
        validated = validate_or_alert(frame, alert, require_target=True)
    except DataValidationError:
        (run / "validation_error.txt").write_text(alert.read_text(encoding="utf-8"), encoding="utf-8")
        raise
    validated.to_csv(run / "validated.csv", index=False)
    dump_json(run / "validation.json", {"status": "pass", "rows": int(len(validated))})


def do_transform(run: Path) -> dict:
    frame = pd.read_csv(run / "validated.csv")
    train, val, test, info = time_split(frame)
    train.to_csv(run / "train.csv", index=False)
    val.to_csv(run / "val.csv", index=False)
    test.to_csv(run / "test.csv", index=False)
    builder = FeatureBuilder().fit(train)
    joblib.dump(builder, run / "transformer.joblib")
    width = int(builder.transform(train.head(2)).shape[1])
    info["n_features"] = width
    dump_json(run / "split.json", info)
    dump_json(run / "transform.json", {"n_features": width, "transformer": "FeatureBuilder"})
    return info


def _data_version(run: Path) -> dict:
    example = load_json(run / "example_gen.json")
    split = load_json(run / "split.json")
    return {
        "file": example["data_file"],
        "sha256": example["sha256"],
        "bytes": example["bytes"],
        "rows": {
            "raw": example["rows"],
            "train": split["train_rows"],
            "val": split["val_rows"],
            "test": split["test_rows"],
        },
        "train_end": split["train_end"],
        "val_end": split["val_end"],
        "test_end": split["test_end"],
    }


def do_train(run: Path, *, only: list[str] | None = None, baseline_auc: float | None = None) -> list[dict]:
    train = pd.read_csv(run / "train.csv")
    val = pd.read_csv(run / "val.csv")
    test = pd.read_csv(run / "test.csv")
    builder = joblib.load(run / "transformer.joblib")
    return train_candidates(
        train,
        val,
        test,
        run,
        _data_version(run),
        only=only,
        builder=builder,
        baseline_auc=baseline_auc,
    )


def do_evaluate(run: Path) -> dict:
    records = load_json(run / "experiments.json")
    payload = {
        "models": [
            {
                "name": row["name"],
                "roc_auc": row["metrics"]["roc_auc"],
                "recall": row["metrics"]["recall"],
                "precision": row["metrics"]["precision"],
                "cost": row["metrics"]["cost"],
                "threshold": row["threshold"],
                "model_mb": row["model_mb"],
                "blessed": row["blessed"],
                "selected": row["selected"],
            }
            for row in records
        ]
    }
    slice_path = run / "slice_by_month.json"
    if slice_path.exists():
        payload["slice_by_month"] = load_json(slice_path)
    dump_json(run / "metrics.json", payload)
    return payload


def do_bless(run: Path) -> str:
    records = load_json(run / "experiments.json")
    winner = next((row for row in records if row.get("selected")), None)
    route = "pusher" if winner else "skip_push"
    dump_json(
        run / "blessing.json",
        {
            "blessed": winner is not None,
            "next": route,
            "winner": None if winner is None else winner["name"],
            "candidates": [
                {
                    "name": row["name"],
                    "blessed": row["blessed"],
                    "gates": row["gates"],
                    "roc_auc": row["metrics"]["roc_auc"],
                    "recall": row["metrics"]["recall"],
                    "model_mb": row["model_mb"],
                }
                for row in records
            ],
        },
    )
    return route


def _remember_production(version: str, model_name: str) -> None:
    path = config.REPORT_DIR / "production_history.json"
    history = load_json(path) if path.exists() else []
    if history and str(history[-1].get("version")) == str(version):
        return
    history.append({"version": str(version), "model_name": model_name})
    dump_json(path, history)


def _write_serving(record: dict) -> None:
    bundle = joblib.load(record["bundle_path"])
    bundle["version"] = record["registry_version"]
    config.SERVING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.SERVING_DIR / "model.joblib")
    dump_json(
        config.SERVING_DIR / "metadata.json",
        {
            "model_name": record["name"],
            "registry_name": config.REGISTRY_NAME,
            "version": record["registry_version"],
            "threshold": record["threshold"],
            "metrics": record["metrics"],
            "run_id": record["run_id"],
            "code_version": record["code_version"],
            "data_sha256": record["data_version"].get("sha256"),
        },
    )


def publish(run: Path, *, promote: bool) -> dict:
    records = load_json(run / "experiments.json")
    for record in records:
        record["registry_version"] = register_bundle(record["run_id"])
    winner = next((row for row in records if row.get("selected")), None)
    promoted = False
    if winner is not None and promote:
        set_production(winner["registry_version"])
        _write_serving(winner)
        _remember_production(winner["registry_version"], winner["name"])
        promoted = True
    dump_json(run / "experiments.json", records)
    payload = {
        "promoted": promoted,
        "production_version": production_version(),
        "versions": list_versions(),
        "winner": None if winner is None else winner["name"],
    }
    dump_json(run / "push.json", payload)
    return payload


def do_push(run: Path) -> dict:
    return publish(run, promote=True)


def do_skip(run: Path) -> dict:
    payload = publish(run, promote=False)
    payload["reason"] = "no candidate cleared the blessing gates; production alias was left unchanged"
    dump_json(run / "skip.json", payload)
    return payload


def write_summary(run: Path) -> dict:
    blessing = load_json(run / "blessing.json")
    push = load_json(run / "push.json") if (run / "push.json").exists() else {}
    summary = {
        "run": str(run),
        "blessed": blessing["blessed"],
        "winner": blessing["winner"],
        "candidates": blessing["candidates"],
        "production_version": push.get("production_version"),
        "promoted": push.get("promoted"),
        "split": load_json(run / "split.json") if (run / "split.json").exists() else {},
    }
    dump_json(run / "summary.json", summary)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(config.REPORT_DIR / "latest_run.json", summary)
    if (run / "experiments.json").exists():
        dump_json(config.REPORT_DIR / "experiments.json", load_json(run / "experiments.json"))
    if (run / "slice_by_month.json").exists():
        dump_json(config.REPORT_DIR / "slice_by_month.json", load_json(run / "slice_by_month.json"))
    return summary


def execute_run(data_file: Path, run_id: str | None = None) -> dict:
    """Raw file to a serving bundle, or a hard stop when the contract fails."""
    run = prepare_run(run_id)
    do_example_gen(run, data_file)
    do_statistics(run)
    do_schema(run)
    do_validate(run)
    do_transform(run)
    do_train(run)
    do_evaluate(run)
    route = do_bless(run)
    if route == "pusher":
        do_push(run)
    else:
        do_skip(run)
    return write_summary(run)


def current_production_auc() -> float | None:
    meta = config.SERVING_DIR / "metadata.json"
    if not meta.exists():
        return None
    metrics = load_json(meta).get("metrics") or {}
    if "roc_auc" not in metrics:
        return None
    return float(metrics["roc_auc"])


def execute_retrain_recent(data_file: Path, run_id: str | None = None) -> dict:
    """Fit the selected family on the latest dates and promote it only if it still clears the gates."""
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    frame = pd.read_csv(data_file)
    leaked = [column for column in config.LEAK_COLUMNS if column in frame.columns]
    if leaked:
        frame = frame.drop(columns=leaked)
    validated = validate_or_alert(
        frame, config.ALERT_DIR / "retrain_validation_error.txt", require_target=True
    )
    labeled = labeled_frame(validated)
    dates = labeled[config.DATE_COL].drop_duplicates().sort_values().to_list()
    start = dates[int(len(dates) * 0.60)]
    recent = labeled[labeled[config.DATE_COL] >= start].reset_index(drop=True)
    train, val, test, info = time_split(recent)
    run = prepare_run(run_id or "retrain-recent")
    train.to_csv(run / "train.csv", index=False)
    val.to_csv(run / "val.csv", index=False)
    test.to_csv(run / "test.csv", index=False)
    dump_json(run / "split.json", info)
    dump_json(
        run / "example_gen.json",
        {
            "rows": int(len(recent)),
            "columns": list(recent.columns),
            "data_file": data_file.name,
            "sha256": _sha256(data_file),
            "bytes": int(data_file.stat().st_size),
            "dropped_leak_columns": leaked,
            "window": "latest_40_percent_of_dates",
        },
    )
    builder = FeatureBuilder().fit(train)
    joblib.dump(builder, run / "transformer.joblib")
    do_train(run, only=["hist_gb"], baseline_auc=current_production_auc())
    do_evaluate(run)
    route = do_bless(run)
    if route == "pusher":
        do_push(run)
    else:
        do_skip(run)
    summary = write_summary(run)
    summary["retrain_window"] = info
    dump_json(config.REPORT_DIR / "retrain_run.json", summary)
    return summary


def run_monitoring_demo(data_file: Path | None = None) -> dict:
    """Score a clean week, a shifted week, and two relabeled weeks."""
    data_file = data_file or config.RAW_CSV
    bundle_path = config.SERVING_DIR / "model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("serving model is missing; run the training pipeline first")
    bundle = joblib.load(bundle_path)
    frame = pd.read_csv(data_file)
    validated = validate_or_alert(frame, config.ALERT_DIR / "monitor_validation_error.txt", require_target=True)
    labeled = labeled_frame(validated)
    _train, _val, test, _info = time_split(labeled)
    reference = test.sample(n=min(4000, len(test)), random_state=1)
    clean = test.sample(n=min(4000, len(test)), random_state=2)
    drifted = shift_features(test).sample(n=min(4000, len(test)), random_state=3)
    concept_a = flip_labels(test.sample(n=min(4000, len(test)), random_state=4), seed=11)
    concept_b = flip_labels(test.sample(n=min(4000, len(test)), random_state=5), seed=12)

    history = []
    windows = [
        ("clean", clean, False),
        ("data_drift", drifted, True),
        ("concept_1", concept_a, False),
        ("concept_2", concept_b, False),
    ]
    html_written = False
    for name, current, expect_shift in windows:
        record = monitor_window(reference, current, bundle, use_evidently=True, window=name)
        snapshot = record.pop("_snapshot", None)
        record["injected"] = name
        record["expect_feature_shift"] = expect_shift
        if snapshot is not None and not html_written and hasattr(snapshot, "save_html"):
            config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(str(config.REPORT_DIR / f"evidently_{name}.html"))
            html_written = True
        history.append(record)
        print(
            f"{name}: kind={record['kind']} share={record['drift_share']:.3f} "
            f"recall={record['realized_recall']:.3f}"
        )
    decision_after_drift = retrain_decision(history[:2])
    decision_after_concept = retrain_decision(history)
    payload = {
        "windows": history,
        "decision_after_data_drift": decision_after_drift,
        "decision_after_concept_windows": decision_after_concept,
    }
    dump_json(config.REPORT_DIR / "monitoring.json", payload)
    dump_json(config.REPORT_DIR / "monitoring_history.json", history)
    return payload
