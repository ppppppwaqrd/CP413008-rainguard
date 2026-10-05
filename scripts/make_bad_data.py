"""Write a small corrupt extract so the failure path can be demonstrated."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainguard import config
from rainguard.synthetic import synthetic_weather


def main() -> None:
    frame = synthetic_weather(40, seed=8)
    frame.loc[0, "Rainfall"] = -5
    frame.loc[1, "Humidity3pm"] = 150
    frame.loc[2, "RainTomorrow"] = "Maybe"
    frame.loc[3, "Location"] = None
    destination = config.DATA_DIR / "bad_weather.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
