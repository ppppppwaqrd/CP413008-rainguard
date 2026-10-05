"""Weekly training DAG. The task bodies are the same functions the CLI calls."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.utils.trigger_rule import TriggerRule

try:
    from airflow.sdk import get_current_context
except ImportError:
    from airflow.operators.python import get_current_context

from rainguard import config
from rainguard.pipeline import (
    do_bless,
    do_evaluate,
    do_example_gen,
    do_push,
    do_schema,
    do_skip,
    do_statistics,
    do_train,
    do_transform,
    do_validate,
    prepare_run,
    write_summary,
)
from rainguard.validate import DataValidationError


def _data_file() -> Path:
    context = get_current_context()
    dag_run = context.get("dag_run")
    conf = dict(dag_run.conf) if dag_run is not None and dag_run.conf else {}
    raw = conf.get("data_file")
    return Path(raw) if raw else config.RAW_CSV


def _evaluated_run() -> Path:
    context = get_current_context()
    pulled = context["ti"].xcom_pull(task_ids="evaluator")
    return Path(pulled)


@dag(
    dag_id="rainguard_train",
    schedule="0 6 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rainguard"],
)
def rainguard_train():
    @task
    def example_gen() -> str:
        context = get_current_context()
        run = prepare_run(context["run_id"])
        do_example_gen(run, _data_file())
        return str(run)

    @task
    def statistics_gen(run: str) -> str:
        do_statistics(Path(run))
        return run

    @task
    def schema_gen(run: str) -> str:
        do_schema(Path(run))
        return run

    @task
    def example_validator(run: str) -> str:
        try:
            do_validate(Path(run))
        except DataValidationError as exc:
            raise AirflowFailException(str(exc)) from exc
        return run

    @task
    def transform(run: str) -> str:
        do_transform(Path(run))
        return run

    @task
    def trainer(run: str) -> str:
        do_train(Path(run))
        return run

    @task
    def evaluator(run: str) -> str:
        do_evaluate(Path(run))
        return run

    @task.branch
    def blessing_gate(run: str) -> str:
        route = do_bless(Path(run))
        return "pusher" if route == "pusher" else "skip_push"

    @task
    def pusher() -> str:
        run = _evaluated_run()
        do_push(run)
        return str(run)

    @task
    def skip_push() -> str:
        run = _evaluated_run()
        do_skip(run)
        return str(run)

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def report() -> str:
        summary = write_summary(_evaluated_run())
        return summary["winner"] or "none"

    generated = example_gen()
    stats = statistics_gen(generated)
    schema = schema_gen(stats)
    validated = example_validator(schema)
    transformed = transform(validated)
    trained = trainer(transformed)
    evaluated = evaluator(trained)
    gate = blessing_gate(evaluated)
    pushed = pusher()
    skipped = skip_push()
    finished = report()

    gate >> [pushed, skipped]
    [pushed, skipped] >> finished


rainguard_train()
