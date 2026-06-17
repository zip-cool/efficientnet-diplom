from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.training.trainer import run_experiment
from pretrained_diploma.src.utils.config import load_experiment_config, load_json


def parse_args() -> argparse.Namespace:
    """Описать параметры пакетного запуска."""

    parser = argparse.ArgumentParser(description="Run all pretrained EfficientNetB0 experiments.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-config", type=Path, default=Path("pretrained_diploma/configs/base_config.json"))
    parser.add_argument("--experiments", type=Path, default=Path("pretrained_diploma/configs/experiments.json"))
    return parser.parse_args()


def main() -> None:
    """Запустить все эксперименты для всех значений эпох."""

    args = parse_args()
    base_config = load_json(args.base_config)
    experiments = load_json(args.experiments)
    for experiment_id in experiments:
        for epoch_value in base_config["epoch_values"]:
            config = load_experiment_config(experiment_id, args.base_config, args.experiments, int(epoch_value))
            output_dir = Path(config["results_dir"]) / config["experiment_id"]
            if output_dir.exists() and (output_dir / "final_metrics.json").exists() and not args.overwrite:
                print(f"Skipping completed experiment: {output_dir}")
                continue
            print(f"Running {config['experiment_id']}")
            run_experiment(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
