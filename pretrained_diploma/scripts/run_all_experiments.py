from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.training.trainer import run_experiment
from pretrained_diploma.src.utils.config import load_experiment_config, load_json


def experiment_number(experiment_id: str) -> int | None:
    """Получить числовой номер из имени вида exp30_name."""

    match = re.match(r"exp(\d+)", experiment_id)
    return int(match.group(1)) if match else None


def parse_args() -> argparse.Namespace:
    """Описать параметры пакетного запуска."""

    parser = argparse.ArgumentParser(description="Run all pretrained EfficientNetB0 experiments.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset-size", dest="dataset_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--all-dataset-sizes", action="store_true")
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--experiment-start", type=int, default=None)
    parser.add_argument("--experiment-end", type=int, default=None)
    parser.add_argument("--base-config", type=Path, default=Path("pretrained_diploma/configs/base_config.json"))
    parser.add_argument("--experiments", type=Path, default=Path("pretrained_diploma/configs/experiments.json"))
    return parser.parse_args()


def main() -> None:
    """Запустить все эксперименты для всех значений эпох."""

    args = parse_args()
    base_config = load_json(args.base_config)
    experiments = load_json(args.experiments)
    for experiment_id in experiments:
        if args.experiment_prefix and not experiment_id.startswith(args.experiment_prefix):
            continue
        number = experiment_number(experiment_id)
        if args.experiment_start is not None and (number is None or number < args.experiment_start):
            continue
        if args.experiment_end is not None and (number is None or number > args.experiment_end):
            continue
        experiment = experiments[experiment_id]
        if args.epochs is not None:
            epoch_values = [args.epochs]
        elif "fixed_epochs" in experiment:
            epoch_values = [experiment["fixed_epochs"]]
        else:
            epoch_values = base_config["epoch_values"]
        if args.all_dataset_sizes:
            dataset_sizes = base_config["dataset_size_values"]
        elif args.dataset_sizes is not None:
            dataset_sizes = args.dataset_sizes
        else:
            dataset_sizes = [None]
        for dataset_size in dataset_sizes:
            for epoch_value in epoch_values:
                config = load_experiment_config(
                    experiment_id,
                    args.base_config,
                    args.experiments,
                    int(epoch_value),
                    dataset_size=dataset_size,
                )
                output_dir = Path(config["results_dir"]) / config["experiment_id"]
                if output_dir.exists() and (output_dir / "final_metrics.json").exists() and not args.overwrite:
                    print(f"Skipping completed experiment: {output_dir}")
                    continue
                print(f"Running {config['experiment_id']}")
                run_experiment(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
