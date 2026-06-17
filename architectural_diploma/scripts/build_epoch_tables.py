from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для построения таблиц по эпохам."""

    parser = argparse.ArgumentParser(description="Build CSV tables grouped by training epoch count.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("architectural_diploma/results/tables/experiments_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("architectural_diploma/results/tables/by_epochs"),
    )
    return parser.parse_args()


def save_epoch_slices(data: pd.DataFrame, output_dir: Path) -> None:
    """Сохранить отдельные CSV для каждого числа эпох.

    Например, все запуски на 20 эпох попадут в `epochs_020.csv`.
    """

    columns = [
        "base_experiment_id",
        "experiment_id",
        "series",
        "max_epochs",
        "test_accuracy",
        "test_loss",
        "best_val_accuracy",
        "best_val_loss",
        "total_training_time_seconds",
        "average_epoch_time_seconds",
        "completed_epochs",
    ]
    for epoch_count, epoch_data in data.groupby("max_epochs"):
        epoch_path = output_dir / f"epochs_{int(epoch_count):03d}.csv"
        epoch_data.sort_values(["series", "base_experiment_id"])[columns].to_csv(
            epoch_path,
            index=False,
        )


def save_pivot_tables(data: pd.DataFrame, output_dir: Path) -> None:
    """Сохранить pivot-таблицы вида `архитектура x число эпох`."""

    for metric in [
        "test_accuracy",
        "test_loss",
        "best_val_accuracy",
        "best_val_loss",
        "total_training_time_seconds",
        "average_epoch_time_seconds",
    ]:
        pivot = data.pivot_table(
            index="base_experiment_id",
            columns="max_epochs",
            values=metric,
            aggfunc="first",
        )
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        pivot.to_csv(output_dir / f"pivot_{metric}.csv")


def main() -> None:
    """Построить все таблицы по эпохам на основе experiments_summary.csv."""

    args = parse_args()
    data = pd.read_csv(args.summary)
    data = data[data["status"] == "completed"].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_epoch_slices(data, args.output_dir)
    save_pivot_tables(data, args.output_dir)
    print(f"Epoch tables saved: {args.output_dir}")


if __name__ == "__main__":
    main()
