from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Описать параметры построения графиков одного эксперимента по числу эпох."""

    parser = argparse.ArgumentParser(description="Plot train/test metrics by epoch count.")
    parser.add_argument("--experiment", default="exp00_full_model")
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained_diploma/results/epoch_series_plots"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл результата эксперимента."""

    return json.loads(path.read_text(encoding="utf-8"))


def extract_epoch_count(experiment_dir: Path, history: pd.DataFrame, metrics: dict[str, Any]) -> int:
    """Определить число завершенных эпох из отчета, истории или имени папки."""

    if "completed_epochs" in metrics:
        return int(metrics["completed_epochs"])
    if not history.empty and "epoch" in history:
        return int(history["epoch"].max())
    return int(experiment_dir.name.rsplit("_ep", maxsplit=1)[-1])


def collect_epoch_rows(results_dir: Path, experiment: str) -> pd.DataFrame:
    """Собрать таблицу метрик для всех запусков одного эксперимента по эпохам."""

    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(results_dir.glob(f"{experiment}_ep*")):
        history_path = experiment_dir / "history.csv"
        metrics_path = experiment_dir / "final_metrics.json"
        if not history_path.exists() or not metrics_path.exists():
            print(f"Skipping incomplete run: {experiment_dir}")
            continue

        history = pd.read_csv(history_path)
        metrics = load_json(metrics_path)
        last_epoch = history.iloc[-1]
        epoch_time_sum = float(history["epoch_time_seconds"].sum()) if "epoch_time_seconds" in history else None
        epoch_count = extract_epoch_count(experiment_dir, history, metrics)

        rows.append(
            {
                "experiment_id": experiment_dir.name,
                "completed_epochs": epoch_count,
                "train_accuracy": float(last_epoch["accuracy"]),
                "val_accuracy": float(last_epoch["val_accuracy"]),
                "test_accuracy": float(metrics["test_accuracy"]),
                "train_loss": float(last_epoch["loss"]),
                "val_loss": float(last_epoch["val_loss"]),
                "test_loss": float(metrics["test_loss"]),
                "total_training_time_seconds": float(
                    metrics.get("total_training_time_seconds", metrics.get("training_time_seconds", epoch_time_sum))
                ),
                "average_epoch_time_seconds": float(epoch_time_sum / epoch_count) if epoch_time_sum else None,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No completed runs found for {experiment} in {results_dir}")
    return pd.DataFrame(rows).sort_values("completed_epochs")


def save_accuracy_plot(data: pd.DataFrame, output_path: Path) -> None:
    """Построить график train/validation/test accuracy по числу эпох."""

    plt.figure(figsize=(9, 5))
    plt.plot(data["completed_epochs"], data["train_accuracy"], marker="o", label="train accuracy")
    plt.plot(data["completed_epochs"], data["val_accuracy"], marker="o", label="validation accuracy")
    plt.plot(data["completed_epochs"], data["test_accuracy"], marker="o", label="test accuracy")
    plt.xlabel("Количество эпох")
    plt.ylabel("Accuracy")
    plt.title("Точность по числу эпох")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_training_time_plot(data: pd.DataFrame, output_path: Path) -> None:
    """Построить график общего времени обучения по числу эпох."""

    plt.figure(figsize=(9, 5))
    plt.plot(data["completed_epochs"], data["total_training_time_seconds"], marker="o")
    plt.xlabel("Количество эпох")
    plt.ylabel("Общее время обучения, секунды")
    plt.title("Время обучения по числу эпох")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Построить CSV и графики для серии запусков одного эксперимента."""

    args = parse_args()
    output_dir = args.output_dir / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    data = collect_epoch_rows(args.results_dir, args.experiment)
    data.to_csv(output_dir / "epoch_series_metrics.csv", index=False)
    save_accuracy_plot(data, output_dir / "accuracy_by_epochs.png")
    save_training_time_plot(data, output_dir / "training_time_by_epochs.png")
    print(f"Epoch series plots saved: {output_dir}")


if __name__ == "__main__":
    main()
