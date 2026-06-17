from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architectural_diploma.src.training.trainer import run_experiment
from architectural_diploma.src.utils.config import load_experiment_config, load_json


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для пакетного запуска экспериментов."""

    parser = argparse.ArgumentParser(description="Run all unfinished architectural experiments.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("architectural_diploma/configs/base_config.json"),
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("architectural_diploma/configs/experiments.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    return parser.parse_args()


def is_optional(experiment_id: str, experiment: dict) -> bool:
    """Определить, относится ли эксперимент к дополнительным."""

    return experiment_id == "exp15_head_160" or experiment.get("series", "").endswith("_optional")


def main() -> None:
    """Последовательно запустить все незавершенные архитектурные эксперименты.

    Для каждого архитектурного варианта выполняются все значения из `epoch_values`.
    Уже завершенные run-id пропускаются, если не указан `--overwrite`.
    """

    args = parse_args()
    base_config = load_json(args.base_config)
    experiments = load_json(args.experiments)
    results_dir = Path(base_config["results_dir"])

    # Внешний цикл идет по архитектурам, внутренний - по числу эпох.
    for experiment_id, experiment in experiments.items():
        if is_optional(experiment_id, experiment) and not args.include_optional:
            print(f"Skipping optional experiment: {experiment_id}")
            continue

        for epoch_value in base_config["epoch_values"]:
            run_id = f"{experiment_id}_ep{int(epoch_value):03d}"
            experiment_dir = results_dir / run_id
            # Завершенный запуск определяется по наличию final_metrics.json.
            if not args.overwrite and (experiment_dir / "final_metrics.json").exists():
                print(f"Skipping completed run: {run_id}")
                continue

            try:
                config = load_experiment_config(
                    experiment_id,
                    args.base_config,
                    args.experiments,
                    int(epoch_value),
                )
                output_dir = run_experiment(config, overwrite=args.overwrite)
                print(f"Completed {run_id}: {output_dir}")
            except Exception as error:
                # При ошибке сохраняем текст в папку конкретного run-id.
                experiment_dir.mkdir(parents=True, exist_ok=True)
                (experiment_dir / "error.txt").write_text(str(error), encoding="utf-8")
                print(json.dumps({"experiment_id": run_id, "status": "failed", "error": str(error)}))
                raise


if __name__ == "__main__":
    main()
