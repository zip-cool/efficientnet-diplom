from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architectural_diploma.src.utils.result_saver import SUMMARY_FIELDS


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для восстановления summary CSV."""

    parser = argparse.ArgumentParser(description="Rebuild experiments_summary.csv from experiment folders.")
    parser.add_argument("--results-dir", type=Path, default=Path("architectural_diploma/results/experiments"))
    parser.add_argument("--output", type=Path, default=Path("architectural_diploma/results/tables/experiments_summary.csv"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Прочитать JSON-файл в словарь."""

    return json.loads(path.read_text(encoding="utf-8"))


def build_row(experiment_dir: Path) -> dict:
    """Собрать одну строку summary по папке отдельного запуска."""

    config = load_json(experiment_dir / "config.json")
    final_metrics = load_json(experiment_dir / "final_metrics.json")
    test_results = load_json(experiment_dir / "test_results.json")
    return {
        "experiment_id": config["experiment_id"],
        "base_experiment_id": config["base_experiment_id"],
        "experiment_name": config["experiment_name"],
        "series": config["experiment_series"],
        "max_epochs": config["max_epochs"],
        "width_coefficient": config["width_coefficient"],
        "depth_coefficient": config["depth_coefficient"],
        "expansion_ratio": config["expansion_ratio"],
        "use_se": config["use_se"],
        "se_ratio": config["se_ratio"],
        "kernel_mode": config["kernel_mode"],
        "head_channels": config["head_channels"],
        "input_size": config["input_size"],
        "completed_epochs": final_metrics["completed_epochs"],
        "best_val_accuracy": final_metrics["best_val_accuracy"],
        "best_val_loss": final_metrics["best_val_loss"],
        "test_accuracy": test_results["test_accuracy"],
        "test_loss": test_results["test_loss"],
        "total_training_time_seconds": final_metrics["total_training_time_seconds"],
        "average_epoch_time_seconds": final_metrics["average_epoch_time_seconds"],
        "status": final_metrics["experiment_status"],
        "result_directory": str(experiment_dir),
    }


def main() -> None:
    """Пересобрать общую таблицу из уже сохраненных папок экспериментов."""

    args = parse_args()
    rows = []
    # В summary попадают только полностью завершенные эксперименты.
    for experiment_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        required = ["config.json", "final_metrics.json", "test_results.json"]
        if all((experiment_dir / name).exists() for name in required):
            rows.append(build_row(experiment_dir))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary rebuilt: {args.output}")


if __name__ == "__main__":
    main()
