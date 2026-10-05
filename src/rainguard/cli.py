"""Command line for the one-shot path and the failure demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rainguard import config
from rainguard.pipeline import execute_retrain_recent, execute_run, run_monitoring_demo
from rainguard.validate import DataValidationError, validate_or_alert


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rainguard.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="validate, train, bless, and publish in one command")
    run_parser.add_argument("--data", type=Path, default=config.RAW_CSV)

    validate_parser = sub.add_parser("validate", help="stop and alert when a file breaks the contract")
    validate_parser.add_argument("--data", type=Path, required=True)

    sub.add_parser("monitor", help="score clean, drifted, and relabeled weeks")
    retrain_parser = sub.add_parser("retrain", help="retrain on the latest dates and gate the result")
    retrain_parser.add_argument("--data", type=Path, default=config.RAW_CSV)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            summary = execute_run(args.data)
            _print(summary)
            return 0
        if args.command == "validate":
            import pandas as pd

            frame = pd.read_csv(args.data)
            validated = validate_or_alert(
                frame, config.ALERT_DIR / "validation_error.txt", require_target=True
            )
            print(f"validation passed rows={len(validated)}")
            return 0
        if args.command == "monitor":
            _print(run_monitoring_demo())
            return 0
        if args.command == "retrain":
            _print(execute_retrain_recent(args.data))
            return 0
    except DataValidationError as exc:
        print(exc)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
