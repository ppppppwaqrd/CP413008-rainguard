"""Point the production alias at the previous registry version and reload the bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainguard import config
from rainguard.registry import (
    download_serving_bundle,
    list_versions,
    production_version,
    set_production,
)
from rainguard.storage import dump_json, load_json


def main() -> None:
    current = production_version()
    versions = list_versions()
    if current is None or len(versions) < 2:
        raise SystemExit("need at least two registered versions before rollback")
    history_path = config.REPORT_DIR / "production_history.json"
    history = load_json(history_path) if history_path.exists() else []
    older_production = [row for row in history if str(row["version"]) != str(current)]
    if older_production:
        target = str(older_production[-1]["version"])
    else:
        older = [int(row["version"]) for row in versions if int(row["version"]) < int(current)]
        if not older:
            raise SystemExit("production is already the oldest version")
        target = str(max(older))
    chosen = next(row for row in versions if row["version"] == target)
    downloaded = download_serving_bundle(chosen["run_id"], config.ARTIFACT_DIR / "rollback")
    bundle_path = downloaded if downloaded.is_file() else downloaded / "serving_bundle.joblib"
    bundle = joblib.load(bundle_path)
    bundle["version"] = target
    config.SERVING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.SERVING_DIR / "model.joblib")
    set_production(target)
    meta_path = config.SERVING_DIR / "metadata.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    meta.update(
        {
            "model_name": bundle.get("model_name"),
            "version": target,
            "threshold": bundle.get("threshold"),
            "rolled_back_from": current,
        }
    )
    dump_json(meta_path, meta)
    trimmed = [row for row in history if str(row["version"]) != str(current)]
    if not any(str(row["version"]) == target for row in trimmed):
        trimmed.append({"version": target, "model_name": bundle.get("model_name")})
    dump_json(history_path, trimmed)
    payload = {"from_version": current, "to_version": target, "model_name": bundle.get("model_name")}
    dump_json(config.REPORT_DIR / "rollback.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
