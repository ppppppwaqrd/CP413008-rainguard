"""Detect injected drift, then train a new model on the latest dates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainguard import config
from rainguard.pipeline import execute_retrain_recent, run_monitoring_demo
from rainguard.storage import dump_json


def main() -> None:
    monitoring = run_monitoring_demo(config.RAW_CSV)
    if not monitoring["decision_after_data_drift"]["retrain"]:
        raise SystemExit("data-drift window did not request a retrain")
    if not monitoring["decision_after_concept_windows"]["retrain"]:
        raise SystemExit("concept-drift windows did not request a retrain")
    retrain = execute_retrain_recent(config.RAW_CSV, run_id="retrain-after-alert")
    payload = {
        "monitoring_decision_data_drift": monitoring["decision_after_data_drift"],
        "monitoring_decision_concept": monitoring["decision_after_concept_windows"],
        "windows": [
            {
                "window": row["window"],
                "kind": row["kind"],
                "drift_share": row["drift_share"],
                "realized_recall": row["realized_recall"],
            }
            for row in monitoring["windows"]
        ],
        "retrain": retrain,
    }
    dump_json(config.REPORT_DIR / "retrain_cycle.json", payload)
    print(payload["retrain"].get("winner"), payload["retrain"].get("promoted"))


if __name__ == "__main__":
    main()
