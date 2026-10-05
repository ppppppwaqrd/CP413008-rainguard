"""MLflow tracking and the model registry used for promote and rollback."""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from rainguard import config


def tracking_uri() -> str:
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MLRUNS_DIR.resolve().as_uri()


def client() -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri())


def configure() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def register_bundle(run_id: str, name: str = config.REGISTRY_NAME) -> str:
    """Register the sklearn flavor logged under ``model`` and return the version."""
    configure()
    result = mlflow.register_model(f"runs:/{run_id}/model", name)
    return str(result.version)


def set_production(version: str, name: str = config.REGISTRY_NAME) -> None:
    client().set_registered_model_alias(name, "production", version)


def production_version(name: str = config.REGISTRY_NAME) -> str | None:
    try:
        found = client().get_model_version_by_alias(name, "production")
    except Exception:
        return None
    return str(found.version)


def list_versions(name: str = config.REGISTRY_NAME) -> list[dict]:
    found = client().search_model_versions(f"name='{name}'")
    rows = []
    for item in found:
        rows.append(
            {
                "version": str(item.version),
                "run_id": item.run_id,
                "current_stage": item.current_stage,
                "aliases": list(item.aliases or []),
            }
        )
    rows.sort(key=lambda row: int(row["version"]))
    return rows


def download_serving_bundle(run_id: str, destination: Path) -> Path:
    """Copy the joblib bundle logged beside the registered model."""
    destination.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="serving_bundle.joblib",
        dst_path=str(destination),
        tracking_uri=tracking_uri(),
    )
    return Path(local)
