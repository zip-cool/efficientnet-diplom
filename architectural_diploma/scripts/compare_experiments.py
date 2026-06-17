from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для сравнительных графиков."""

    parser = argparse.ArgumentParser(description="Build comparison plots for architectural experiments.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("architectural_diploma/results/tables/experiments_summary.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("architectural_diploma/results/experiments"),
    )
    parser.add_argument("--experiments", nargs="*", default=[])
    return parser.parse_args()


def save_bar_plot(data: pd.DataFrame, column: str, ylabel: str, output: Path) -> None:
    """Сохранить столбчатый график одной метрики по экспериментам."""

    plt.figure(figsize=(12, 6))
    plt.bar(data["experiment_id"], data[column])
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_scatter(data: pd.DataFrame, output: Path) -> None:
    """Сохранить график зависимости test accuracy от времени обучения."""

    plt.figure(figsize=(8, 6))
    plt.scatter(data["total_training_time_seconds"], data["test_accuracy"])
    for _, row in data.iterrows():
        plt.annotate(
            row["experiment_id"],
            (row["total_training_time_seconds"], row["test_accuracy"]),
            fontsize=8,
        )
    plt.xlabel("Total training time, seconds")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_history_comparison(
    results_dir: Path,
    experiment_ids: list[str],
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    """Сравнить validation-кривые нескольких запусков по сохраненным history.csv."""

    plt.figure(figsize=(10, 6))
    for experiment_id in experiment_ids:
        history_csv = results_dir / experiment_id / "history.csv"
        if not history_csv.exists():
            continue
        history = pd.read_csv(history_csv)
        plt.plot(history["epoch"], history[metric], marker="o", label=experiment_id)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    """Построить набор сравнительных графиков по общей summary-таблице."""

    args = parse_args()
    data = pd.read_csv(args.summary)
    data = data[data["status"] == "completed"].copy()
    output_dir = args.results_dir / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_bar_plot(data, "test_accuracy", "Test accuracy", output_dir / "test_accuracy.png")
    save_bar_plot(
        data,
        "total_training_time_seconds",
        "Total training time, seconds",
        output_dir / "training_time.png",
    )
    save_scatter(data, output_dir / "accuracy_vs_time.png")

    selected = args.experiments or data["experiment_id"].head(6).tolist()
    save_history_comparison(
        args.results_dir,
        selected,
        "val_accuracy",
        "Validation accuracy",
        output_dir / "val_accuracy_curves.png",
    )
    save_history_comparison(
        args.results_dir,
        selected,
        "val_loss",
        "Validation loss",
        output_dir / "val_loss_curves.png",
    )
    print(f"Comparison plots saved: {output_dir}")


if __name__ == "__main__":
    main()
