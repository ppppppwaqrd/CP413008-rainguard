"""Train the three candidates and log the six things every experiment must keep."""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from rainguard import config
from rainguard.blessing import is_blessed
from rainguard.features import FeatureBuilder
from rainguard.metrics import binary_metrics, choose_threshold, encode_labels
from rainguard.registry import configure
from rainguard.storage import dump_json


def code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config.ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def environment_fingerprint() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }


def class_sample_weight(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    return np.where(y_true == 1, n_neg / n_pos, 1.0).astype(float)


def candidate_estimators() -> list[tuple[str, object, dict, str]]:
    """Name, estimator, params, and how class imbalance is handled."""
    return [
        (
            "logreg",
            LogisticRegression(max_iter=400, solver="lbfgs", class_weight="balanced"),
            {"max_iter": 400, "solver": "lbfgs", "class_weight": "balanced"},
            "class_weight",
        ),
        (
            "hist_gb",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=42,
            ),
            {
                "max_iter": 150,
                "learning_rate": 0.08,
                "max_leaf_nodes": 31,
                "random_state": 42,
            },
            "sample_weight",
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=40,
                max_depth=10,
                min_samples_leaf=8,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced_subsample",
            ),
            {
                "n_estimators": 40,
                "max_depth": 10,
                "min_samples_leaf": 8,
                "class_weight": "balanced_subsample",
                "random_state": 42,
            },
            "class_weight",
        ),
    ]


def _fit(estimator: object, matrix: np.ndarray, y_true: np.ndarray, weight_mode: str) -> None:
    if weight_mode == "sample_weight":
        estimator.fit(matrix, y_true, sample_weight=class_sample_weight(y_true))
    else:
        estimator.fit(matrix, y_true)


def _positive_proba(estimator: object, matrix: np.ndarray) -> np.ndarray:
    proba = estimator.predict_proba(matrix)
    return np.asarray(proba)[:, 1]


def slice_by_month(
    test: pd.DataFrame,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict:
    frame = test.copy()
    frame["_y"] = y_true
    frame["_p"] = proba
    frame["_month"] = pd.to_datetime(frame[config.DATE_COL]).dt.month
    slices = {}
    for month, part in frame.groupby("_month"):
        if part["_y"].nunique() < 2 or len(part) < 30:
            continue
        scores = binary_metrics(part["_y"].to_numpy(), part["_p"].to_numpy(), threshold)
        slices[str(int(month))] = {
            "rows": int(len(part)),
            "roc_auc": round(scores["roc_auc"], 4),
            "recall": round(scores["recall"], 4),
        }
    return slices


def train_candidates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    run_dir: Path,
    data_version: dict,
    *,
    only: list[str] | None = None,
    builder: FeatureBuilder | None = None,
    baseline_auc: float | None = None,
) -> list[dict]:
    """Fit each candidate with the shared FeatureBuilder and log it to MLflow."""
    run_dir.mkdir(parents=True, exist_ok=True)
    configure()
    y_train = encode_labels(train[config.TARGET])
    y_val = encode_labels(val[config.TARGET])
    y_test = encode_labels(test[config.TARGET])
    if builder is None:
        builder = FeatureBuilder().fit(train)
    x_train = builder.transform(train)
    x_val = builder.transform(val)
    x_test = builder.transform(test)
    joblib.dump(builder, run_dir / "transformer.joblib")

    env = environment_fingerprint()
    code_sha = code_version()
    dump_json(run_dir / "environment.json", env)
    dump_json(run_dir / "data_version.json", data_version)
    dump_json(run_dir / "code_version.json", {"git_sha": code_sha})

    records: list[dict] = []
    for name, estimator, params, weight_mode in candidate_estimators():
        if only is not None and name not in only:
            continue
        started = time.perf_counter()
        _fit(estimator, x_train, y_train, weight_mode)
        train_seconds = round(time.perf_counter() - started, 3)
        val_proba = _positive_proba(estimator, x_val)
        threshold = choose_threshold(y_val, val_proba, config.VAL_RECALL_FLOOR)
        test_proba = _positive_proba(estimator, x_test)
        scores = binary_metrics(y_test, test_proba, threshold)
        bundle_path = run_dir / f"{name}.joblib"
        bundle = {
            "model": estimator,
            "features": builder,
            "threshold": threshold,
            "model_name": name,
            "version": None,
        }
        joblib.dump(bundle, bundle_path)
        serving_bundle = run_dir / "serving_bundle.joblib"
        joblib.dump(bundle, serving_bundle)
        model_mb = round(bundle_path.stat().st_size / 1_000_000, 4)
        with mlflow.start_run(run_name=name) as active:
            mlflow.log_params(params)
            mlflow.log_param("weight_mode", weight_mode)
            mlflow.log_param("threshold", round(threshold, 4))
            mlflow.log_metrics(
                {
                    "roc_auc": scores["roc_auc"],
                    "average_precision": scores["average_precision"],
                    "recall": scores["recall"],
                    "precision": scores["precision"],
                    "f1": scores["f1"],
                    "cost": scores["cost"],
                    "train_seconds": train_seconds,
                    "model_mb": model_mb,
                }
            )
            mlflow.set_tag("code_version", code_sha)
            mlflow.set_tag("data_sha256", str(data_version.get("sha256", "unknown")))
            mlflow.log_dict(env, "environment.json")
            mlflow.log_dict(data_version, "data_version.json")
            mlflow.log_artifact(str(bundle_path))
            mlflow.log_artifact(str(serving_bundle))
            mlflow.sklearn.log_model(estimator, "model")
            run_id = active.info.run_id
        record = {
            "name": name,
            "params": params,
            "weight_mode": weight_mode,
            "threshold": threshold,
            "train_seconds": train_seconds,
            "model_mb": model_mb,
            "bundle_path": str(bundle_path),
            "run_id": run_id,
            "metrics": scores,
            "code_version": code_sha,
            "data_version": data_version,
            "environment": env,
        }
        records.append(record)
        print(
            f"{name}: auc={scores['roc_auc']:.4f} recall={scores['recall']:.4f} "
            f"cost={scores['cost']:.0f} size={model_mb:.2f}MB t={threshold:.2f}"
        )

    if not records:
        raise RuntimeError("no candidates were trained")
    winner_name = _select_winner(records, forced_baseline=baseline_auc)
    for record in records:
        record["selected"] = record["name"] == winner_name
    if winner_name is not None:
        winner = next(record for record in records if record["name"] == winner_name)
        slices = slice_by_month(test, y_test, _proba_from_bundle(winner["bundle_path"], test), winner["threshold"])
        dump_json(run_dir / "slice_by_month.json", slices)
    dump_json(run_dir / "experiments.json", records)
    return records


def _proba_from_bundle(path: str, frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(path)
    matrix = bundle["features"].transform(frame)
    return np.asarray(bundle["model"].predict_proba(matrix))[:, 1]


def _select_winner(records: list[dict], forced_baseline: float | None = None) -> str | None:
    logreg_auc = next((row["metrics"]["roc_auc"] for row in records if row["name"] == "logreg"), None)
    for record in records:
        if forced_baseline is not None:
            base = forced_baseline
        elif record["name"] == "logreg":
            base = None
        else:
            base = logreg_auc
        ok, gates = is_blessed(
            roc_auc=record["metrics"]["roc_auc"],
            pos_recall=record["metrics"]["recall"],
            baseline_auc=base,
            model_mb=record["model_mb"],
            min_roc_auc=config.MIN_ROC_AUC,
            min_improvement=config.MIN_IMPROVEMENT,
            min_pos_recall=config.MIN_POS_RECALL,
            max_model_mb=config.MAX_MODEL_MB,
        )
        record["blessed"] = ok
        record["gates"] = gates
    blessed = [row for row in records if row["blessed"]]
    if not blessed:
        return None
    blessed.sort(key=lambda row: (-row["metrics"]["roc_auc"], row["model_mb"]))
    return blessed[0]["name"]
