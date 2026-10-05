"""Build the drifted and relabeled extracts used in the monitoring demo."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainguard import config
from rainguard.monitoring import flip_labels, shift_features
from rainguard.split import time_split
from rainguard.validate import validate_frame


def main() -> None:
    if not config.RAW_CSV.exists():
        raise SystemExit("missing weatherAUS.csv")
    frame = validate_frame(pd.read_csv(config.RAW_CSV), require_target=True)
    _train, _val, test, _info = time_split(frame)
    sample = test.sample(n=min(5000, len(test)), random_state=7)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    drift_path = config.DATA_DIR / "drift_data.csv"
    concept_path = config.DATA_DIR / "concept_drift.csv"
    shift_features(sample).to_csv(drift_path, index=False)
    flip_labels(sample, rate=0.45, seed=7).to_csv(concept_path, index=False)
    print(drift_path)
    print(concept_path)


if __name__ == "__main__":
    main()
