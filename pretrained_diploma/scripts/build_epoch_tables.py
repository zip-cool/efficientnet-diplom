from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Описать параметры построения таблиц по числу эпох."""

    parser = argparse.ArgumentParser(description="Build epoch tables for pretrained experiments.")
    parser.add_argument("--summary", type=Path, default=Path("pretrained_diploma/results/tables/experiments_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained_diploma/results/tables/by_epochs"))
    return parser.parse_args()


def main() -> None:
    """Создать таблицы `epochs_XXX.csv` и pivot-таблицы по метрикам."""

    args = parse_args()
    data = pd.read_csv(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch_count, epoch_data in data.groupby("completed_epochs"):
        epoch_data.to_csv(args.output_dir / f"epochs_{int(epoch_count):03d}.csv", index=False)
    for metric in ["test_accuracy", "total_training_time_seconds", "head_training_time_seconds", "finetune_training_time_seconds"]:
        pivot = data.pivot_table(index="base_experiment_id", columns="completed_epochs", values=metric, aggfunc="first")
        pivot.to_csv(args.output_dir / f"pivot_{metric}.csv")
    print(f"Epoch tables saved: {args.output_dir}")


if __name__ == "__main__":
    main()
