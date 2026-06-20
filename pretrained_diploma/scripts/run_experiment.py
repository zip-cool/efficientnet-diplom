from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.training.trainer import run_experiment
from pretrained_diploma.src.utils.config import load_experiment_config, load_json


def result_matches_current_config(output_dir: Path, config: dict) -> bool:
    """Проверить, что сохраненный результат относится к текущему датасету и числу эпох."""

    config_path = output_dir / "config.json"
    if not config_path.exists():
        return False
    saved_config = load_json(config_path)
    saved_dataset_size = saved_config.get("dataset_size", config.get("dataset_size"))
    return (
        saved_config.get("dataset_dir") == config.get("dataset_dir")
        and saved_config.get("max_epochs") == config.get("max_epochs")
        and saved_dataset_size == config.get("dataset_size")
    )


def parse_args() -> argparse.Namespace:
    """Описать параметры запуска одного эксперимента."""

    parser = argparse.ArgumentParser(description="Run one pretrained EfficientNetB0 experiment.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all-epoch-values", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--all-dataset-sizes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-config", type=Path, default=Path("pretrained_diploma/configs/base_config.json"))
    parser.add_argument("--experiments", type=Path, default=Path("pretrained_diploma/configs/experiments.json"))
    return parser.parse_args()


def main() -> None:
    """Запустить один эксперимент для одного или всех чисел эпох."""

    args = parse_args()
    base_config = load_json(args.base_config)
    epoch_values = base_config["epoch_values"] if args.all_epoch_values else [args.epochs]
    dataset_sizes = base_config["dataset_size_values"] if args.all_dataset_sizes else [args.dataset_size]
    for dataset_size in dataset_sizes:
        for epoch_value in epoch_values:
            config = load_experiment_config(
                args.experiment,
                args.base_config,
                args.experiments,
                epoch_value,
                dataset_size=dataset_size,
            )
            output_dir = Path(config["results_dir"]) / config["experiment_id"]
            if output_dir.exists() and (output_dir / "final_metrics.json").exists() and not args.overwrite:
                if result_matches_current_config(output_dir, config):
                    print(f"Skipping completed experiment: {output_dir}")
                    continue
                raise FileExistsError(
                    f"{output_dir} already contains results for another dataset or epoch count. "
                    "Use --overwrite to restart it with the current config."
                )
            if output_dir.exists() and not args.overwrite:
                raise FileExistsError(f"{output_dir} exists but is incomplete. Use --overwrite to restart it.")
            print(f"Running {config['experiment_id']}")
            print(f"Result directory: {run_experiment(config, overwrite=args.overwrite)}")


if __name__ == "__main__":
    main()
