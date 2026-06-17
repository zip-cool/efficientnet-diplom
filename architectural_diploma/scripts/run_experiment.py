from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architectural_diploma.src.training.trainer import run_experiment
from architectural_diploma.src.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для запуска одного эксперимента."""

    parser = argparse.ArgumentParser(description="Run one architectural EfficientNet experiment.")
    parser.add_argument("--experiment", required=True)
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all-epoch-values", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Запустить один архитектурный эксперимент.

    Можно указать конкретное число эпох через `--epochs` или прогнать этот же
    эксперимент по всей сетке эпох из `base_config.json`.
    """

    args = parse_args()
    base_config_path = args.base_config
    experiments_path = args.experiments

    # При --all-epoch-values берется список 5/10/20/40/60/80/100 из конфига.
    if args.all_epoch_values:
        from architectural_diploma.src.utils.config import load_json

        epoch_values = load_json(base_config_path)["epoch_values"]
    else:
        epoch_values = [args.epochs]

    # Каждый epoch_value превращается в отдельный run-id и отдельную папку.
    for epoch_value in epoch_values:
        config = load_experiment_config(args.experiment, base_config_path, experiments_path, epoch_value)
        output_dir = Path(config["results_dir"]) / config["experiment_id"]
        if output_dir.exists() and (output_dir / "final_metrics.json").exists() and not args.overwrite:
            print(f"Skipping completed experiment: {output_dir}")
            continue
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists but is incomplete. "
                "Run again with --overwrite to restart this run."
            )
        output_dir = run_experiment(config, overwrite=args.overwrite)
        print(f"Experiment completed: {output_dir}")


if __name__ == "__main__":
    main()
