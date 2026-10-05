"""Run the three CI gates locally and keep both a pass log and a rejection log."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainguard import config


def _python() -> str:
    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def expect_reject() -> int:
    """Exit 0 only when the contract rejects the corrupt file (pipeline exit 1)."""
    py = _python()
    bad = config.DATA_DIR / "bad_weather.csv"
    if not bad.exists():
        subprocess.run([py, "scripts/make_bad_data.py"], cwd=ROOT, check=True)
    completed = _run([py, "-m", "rainguard.cli", "validate", "--data", str(bad)])
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log = (
        f"command={py} -m rainguard.cli validate --data {bad}\n"
        f"exit_code={completed.returncode}\n"
        f"{completed.stdout}\n{completed.stderr}\n"
    )
    (config.REPORT_DIR / "ci_fail.txt").write_text(log, encoding="utf-8")
    if completed.returncode == 0:
        print("corrupt file was accepted")
        return 1
    print(f"corrupt file rejected with exit code {completed.returncode}")
    return 0


def full_gate() -> int:
    py = _python()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        [py, "-m", "ruff", "check", "src", "tests", "serving", "scripts", "dags"],
        [py, "-m", "pytest", "tests", "-q"],
    ]
    chunks = []
    for command in steps:
        completed = _run(command)
        chunks.append(
            f"$ {' '.join(command)}\nexit_code={completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}\n"
        )
        if completed.returncode != 0:
            print(chunks[-1])
            return completed.returncode
    reject = expect_reject()
    chunks.append(f"corrupt-file gate exit_code={reject}\n")
    (config.REPORT_DIR / "ci_pass.txt").write_text("".join(chunks), encoding="utf-8")
    return reject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-reject", action="store_true")
    args = parser.parse_args()
    raise SystemExit(expect_reject() if args.expect_reject else full_gate())


if __name__ == "__main__":
    main()
