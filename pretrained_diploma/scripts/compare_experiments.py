from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Описать параметры построения сравнительных графиков."""

    parser = argparse.ArgumentParser(description="Build comparison plots for pretrained experiments.")
    parser.add_argument("--summary", type=Path, default=Path("pretrained_diploma/results/tables/experiments_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained_diploma/results/comparison_plots"))
    return parser.parse_args()


def save_bar_plot(data: pd.DataFrame, column: str, ylabel: str, output: Path) -> None:
    """Построить столбчатый график по всем завершенным запускам."""

    plt.figure(figsize=(12, 6))
    plt.bar(data["experiment_id"], data[column])
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_accuracy_time_scatter(data: pd.DataFrame, output: Path) -> None:
    """Построить зависимость test accuracy от общего времени обучения."""

    plt.figure(figsize=(8, 6))
    plt.scatter(data["total_training_time_seconds"], data["test_accuracy"])
    for _, row in data.iterrows():
        plt.annotate(row["experiment_id"], (row["total_training_time_seconds"], row["test_accuracy"]), fontsize=8)
    plt.xlabel("Total training time, seconds")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    """Построить базовый набор сравнительных графиков по summary CSV."""

    args = parse_args()
    data = pd.read_csv(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_bar_plot(data, "test_accuracy", "Test accuracy", args.output_dir / "test_accuracy.png")
    save_bar_plot(data, "total_training_time_seconds", "Total training time, seconds", args.output_dir / "training_time.png")
    save_accuracy_time_scatter(data, args.output_dir / "accuracy_vs_time.png")
    print(f"Comparison plots saved: {args.output_dir}")


if __name__ == "__main__":
    main()
