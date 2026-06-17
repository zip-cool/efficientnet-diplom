from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pretrained_diploma.src.utils.result_saver import SUMMARY_FIELDS


def parse_args() -> argparse.Namespace:
    """Описать параметры восстановления summary."""

    parser = argparse.ArgumentParser(description="Rebuild pretrained experiments summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    parser.add_argument("--output", type=Path, default=Path("pretrained_diploma/results/tables/experiments_summary.csv"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Прочитать JSON-файл."""

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Пересобрать summary CSV из папок завершенных экспериментов."""

    args = parse_args()
    rows = []
    for experiment_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        if not (experiment_dir / "config.json").exists() or not (experiment_dir / "final_metrics.json").exists():
            continue
        config = load_json(experiment_dir / "config.json")
        metrics = load_json(experiment_dir / "final_metrics.json")
        rows.append(
            {
                "experiment_id": config["experiment_id"],
                "base_experiment_id": config["base_experiment_id"],
                "experiment_name": config["experiment_name"],
                "series": config["series"],
                "resolution": config["input_size"],
                "cut_point": config["cut_point"],
                "cut_layer": config["cut_layer"],
                "tail_type": config["tail_type"],
                "test_accuracy": metrics["test_accuracy"],
                "test_loss": metrics["test_loss"],
                "head_training_time_seconds": metrics["head_training_time_seconds"],
                "finetune_training_time_seconds": metrics["finetune_training_time_seconds"],
                "total_training_time_seconds": metrics["total_training_time_seconds"],
                "completed_epochs": metrics["completed_epochs"],
                "result_directory": str(experiment_dir),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary rebuilt: {args.output}")


if __name__ == "__main__":
    main()
