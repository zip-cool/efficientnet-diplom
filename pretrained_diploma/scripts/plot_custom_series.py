from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ACCURACY_METRICS = {
    "train": ("train_accuracy", "train accuracy"),
    "val": ("val_accuracy", "validation accuracy"),
    "test": ("test_accuracy", "test accuracy"),
}


def parse_args() -> argparse.Namespace:
    """Описать настройки графиков, которые пользователь задает из командной строки."""

    parser = argparse.ArgumentParser(description="Build configurable plots by experiments and epoch counts.")
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained_diploma/results/custom_plots"))

    parser.add_argument("--accuracy-experiments", nargs="+", required=True)
    parser.add_argument("--accuracy-epochs", nargs="+", type=int, default=None)
    parser.add_argument("--accuracy-metrics", nargs="+", choices=sorted(ACCURACY_METRICS), default=["train", "test"])
    parser.add_argument("--include-val-accuracy", action="store_true")

    parser.add_argument("--time-experiments", nargs="+", default=None)
    parser.add_argument("--time-epochs", nargs="+", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-отчет завершенного запуска."""

    return json.loads(path.read_text(encoding="utf-8"))


def experiment_run_dir(results_dir: Path, experiment: str, epoch_count: int) -> Path:
    """Собрать путь к папке запуска по имени эксперимента и числу эпох."""

    return results_dir / f"{experiment}_ep{epoch_count:03d}"


def available_epochs(results_dir: Path, experiment: str) -> list[int]:
    """Найти все числа эпох, для которых есть папки запусков эксперимента."""

    epochs: list[int] = []
    for run_dir in sorted(results_dir.glob(f"{experiment}_ep*")):
        try:
            epochs.append(int(run_dir.name.rsplit("_ep", maxsplit=1)[-1]))
        except ValueError:
            continue
    return sorted(set(epochs))


def resolve_epochs(results_dir: Path, experiment: str, requested_epochs: list[int] | None) -> list[int]:
    """Вернуть заданные пользователем эпохи или все найденные эпохи эксперимента."""

    return requested_epochs if requested_epochs is not None else available_epochs(results_dir, experiment)


def load_run_metrics(results_dir: Path, experiment: str, epoch_count: int) -> dict[str, Any] | None:
    """Загрузить метрики одного завершенного запуска или пропустить неполный запуск."""

    run_dir = experiment_run_dir(results_dir, experiment, epoch_count)
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "final_metrics.json"
    if not history_path.exists() or not metrics_path.exists():
        print(f"Skipping missing or incomplete run: {run_dir}")
        return None

    history = pd.read_csv(history_path)
    metrics = load_json(metrics_path)
    if history.empty:
        print(f"Skipping empty history: {history_path}")
        return None

    last_epoch = history.iloc[-1]
    return {
        "experiment": experiment,
        "experiment_id": run_dir.name,
        "completed_epochs": epoch_count,
        "train_accuracy": float(last_epoch["accuracy"]),
        "val_accuracy": float(last_epoch["val_accuracy"]),
        "test_accuracy": float(metrics["test_accuracy"]),
        "train_loss": float(last_epoch["loss"]),
        "val_loss": float(last_epoch["val_loss"]),
        "test_loss": float(metrics["test_loss"]),
        "total_training_time_seconds": float(
            metrics.get("total_training_time_seconds", metrics.get("training_time_seconds", 0.0))
        ),
    }


def collect_rows(results_dir: Path, experiments: list[str], requested_epochs: list[int] | None) -> pd.DataFrame:
    """Собрать таблицу метрик для выбранных экспериментов и эпох."""

    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for epoch_count in resolve_epochs(results_dir, experiment, requested_epochs):
            row = load_run_metrics(results_dir, experiment, epoch_count)
            if row is not None:
                rows.append(row)

    if not rows:
        raise FileNotFoundError("No completed runs found for selected experiments and epochs.")
    return pd.DataFrame(rows).sort_values(["experiment", "completed_epochs"])


def selected_accuracy_metrics(metric_names: list[str], include_val_accuracy: bool) -> list[str]:
    """Собрать итоговый список accuracy-метрик с опциональным добавлением validation."""

    metrics = list(dict.fromkeys(metric_names))
    if include_val_accuracy and "val" not in metrics:
        metrics.append("val")
    return metrics


def save_accuracy_plot(data: pd.DataFrame, metric_names: list[str], output_path: Path) -> None:
    """Построить график выбранных accuracy-метрик для выбранных экспериментов."""

    plt.figure(figsize=(10, 6))
    for experiment, experiment_data in data.groupby("experiment"):
        for metric_name in metric_names:
            column, label = ACCURACY_METRICS[metric_name]
            plt.plot(
                experiment_data["completed_epochs"],
                experiment_data[column],
                marker="o",
                label=f"{experiment}: {label}",
            )

    plt.xlabel("Количество эпох")
    plt.ylabel("Accuracy")
    plt.title("Точность по выбранным экспериментам и эпохам")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_training_time_plot(data: pd.DataFrame, output_path: Path) -> None:
    """Построить график времени обучения для выбранных экспериментов и эпох."""

    plt.figure(figsize=(10, 6))
    for experiment, experiment_data in data.groupby("experiment"):
        plt.plot(
            experiment_data["completed_epochs"],
            experiment_data["total_training_time_seconds"],
            marker="o",
            label=experiment,
        )

    plt.xlabel("Количество эпох")
    plt.ylabel("Общее время обучения, секунды")
    plt.title("Время обучения по выбранным экспериментам и эпохам")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Построить настраиваемые графики accuracy и времени обучения."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_metrics = selected_accuracy_metrics(args.accuracy_metrics, args.include_val_accuracy)
    accuracy_data = collect_rows(args.results_dir, args.accuracy_experiments, args.accuracy_epochs)
    accuracy_data.to_csv(args.output_dir / "accuracy_selected_data.csv", index=False)
    save_accuracy_plot(accuracy_data, accuracy_metrics, args.output_dir / "accuracy_selected.png")

    time_experiments = args.time_experiments or args.accuracy_experiments
    time_data = collect_rows(args.results_dir, time_experiments, args.time_epochs)
    time_data.to_csv(args.output_dir / "training_time_selected_data.csv", index=False)
    save_training_time_plot(time_data, args.output_dir / "training_time_selected.png")

    print(f"Custom plots saved: {args.output_dir}")


if __name__ == "__main__":
    main()
