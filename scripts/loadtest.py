"""Measure p50, p95, and throughput of the live API and compare them with the SLO."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainguard import config
from rainguard.storage import dump_json, load_json


def _wait_until_ready(base: str, process) -> None:
    for _ in range(40):
        if process.poll() is not None:
            raise RuntimeError("API process exited before it was ready")
        try:
            response = requests.get(f"{base}/health", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError("API did not become ready")


def _payload() -> dict:
    summary = load_json(config.REPORT_DIR / "latest_run.json")
    frame = pd.read_csv(Path(summary["run"]) / "test.csv")
    row = frame.iloc[0].drop(labels=[config.TARGET]).to_dict()
    cleaned = {}
    for key, value in row.items():
        if pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    cleaned["Date"] = str(cleaned["Date"])[:10]
    return cleaned


def main() -> None:
    import subprocess

    base = "http://127.0.0.1:8000"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        cwd=config.ROOT,
    )
    try:
        _wait_until_ready(base, process)
        payload = _payload()
        rejected = dict(payload)
        rejected["Rainfall"] = -5
        bad = requests.post(f"{base}/predict", json=rejected, timeout=5)
        if bad.status_code != 422:
            raise RuntimeError(f"schema gate returned {bad.status_code}")

        latencies = []
        for _ in range(200):
            started = time.perf_counter()
            response = requests.post(f"{base}/predict", json=payload, timeout=5)
            latencies.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()
        values = np.asarray(latencies)

        def one_call(_: int) -> int:
            return requests.post(f"{base}/predict", json=payload, timeout=5).status_code

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            codes = list(pool.map(one_call, range(160)))
        elapsed = time.perf_counter() - started
        if any(code != 200 for code in codes):
            raise RuntimeError("throughput run returned a non-200 status")
        health = requests.get(f"{base}/health", timeout=5).json()
        report = {
            "p50_ms": round(float(np.percentile(values, 50)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3),
            "throughput_rps": round(160 / elapsed, 2),
            "requests": 200,
            "slo": {
                "p50_ms": config.P50_SLO_MS,
                "p95_ms": config.P95_SLO_MS,
                "min_throughput_rps": config.MIN_THROUGHPUT_RPS,
            },
            "model_name": health.get("model_name"),
            "model_version": health.get("model_version"),
        }
        report["p50_pass"] = report["p50_ms"] <= config.P50_SLO_MS
        report["p95_pass"] = report["p95_ms"] <= config.P95_SLO_MS
        report["throughput_pass"] = report["throughput_rps"] >= config.MIN_THROUGHPUT_RPS
        dump_json(config.REPORT_DIR / "slo.json", report)
        print(report)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
