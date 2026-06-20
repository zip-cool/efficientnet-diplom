from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "test_accuracy": "test accuracy",
    "train_accuracy": "train accuracy",
    "val_accuracy": "validation accuracy",
    "train_eval_accuracy": "train eval accuracy",
    "val_eval_accuracy": "validation eval accuracy",
}


def parse_args() -> argparse.Namespace:
    """Описать параметры построения графиков точности по размеру датасета."""

    parser = argparse.ArgumentParser(
        description="Plot accuracy dependence on total dataset size for dataset-size experiments."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained_diploma/results/dataset_size_plots"),
    )
    parser.add_argument("--experiment-start", type=int, default=30)
    parser.add_argument("--experiment-end", type=int, default=43)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--metric", choices=sorted(METRIC_LABELS), default="test_accuracy")
    parser.add_argument("--make-combined-plot", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл результата эксперимента."""

    return json.loads(path.read_text(encoding="utf-8"))


def experiment_number(experiment_name: str) -> int | None:
    """Получить номер эксперимента из имени вида exp30_name."""

    match = re.match(r"exp(\d+)", experiment_name)
    return int(match.group(1)) if match else None


def is_selected_experiment(
    base_experiment_id: str,
    selected_experiments: set[str] | None,
    experiment_start: int,
    experiment_end: int,
) -> bool:
    """Проверить, входит ли эксперимент в выбранный список или диапазон номеров."""

    if selected_experiments is not None:
        return base_experiment_id in selected_experiments

    number = experiment_number(base_experiment_id)
    return number is not None and experiment_start <= number <= experiment_end


def total_dataset_size(metrics: dict[str, Any], config: dict[str, Any]) -> int:
    """Вернуть суммарный размер датасета по всем классам, а не размер на класс."""

    if metrics.get("total_size") is not None:
        return int(metrics["total_size"])

    dataset_size = metrics.get("dataset_size", config.get("dataset_size"))
    num_classes = config.get("num_classes")
    if dataset_size is not None and num_classes is not None:
        return int(dataset_size) * int(num_classes)

    split_counts = metrics.get("split_counts", {})
    if split_counts:
        return sum(sum(int(value) for value in split.values()) for split in split_counts.values())

    raise KeyError(f"Cannot determine total dataset size for {config.get('experiment_id')}")


def collect_rows(
    results_dir: Path,
    metric: str,
    selected_experiments: set[str] | None,
    experiment_start: int,
    experiment_end: int,
    dataset_sizes: set[int] | None,
) -> pd.DataFrame:
    """Собрать таблицу точности для всех завершённых запусков серии размеров датасета."""

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "final_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue

        config = load_json(config_path)
        metrics = load_json(metrics_path)
        base_experiment_id = str(config.get("base_experiment_id", ""))
        if not is_selected_experiment(base_experiment_id, selected_experiments, experiment_start, experiment_end):
            continue

        dataset_size = metrics.get("dataset_size", config.get("dataset_size"))
        if dataset_size is None:
            continue
        dataset_size = int(dataset_size)
        if dataset_sizes is not None and dataset_size not in dataset_sizes:
            continue

        metric_value = metrics.get(metric)
        if metric_value is None:
            print(f"Skipping {run_dir.name}: metric {metric} is missing.")
            continue

        rows.append(
            {
                "experiment_id": config["experiment_id"],
                "base_experiment_id": base_experiment_id,
                "source_experiment_id": config.get("source_experiment_id"),
                "dataset_size_per_class": dataset_size,
                "total_dataset_size": total_dataset_size(metrics, config),
                "metric": metric,
                "accuracy": float(metric_value),
                "completed_epochs": int(metrics.get("completed_epochs", config.get("max_epochs", 0))),
                "result_directory": str(run_dir),
            }
        )

    if not rows:
        raise FileNotFoundError("No completed dataset-size experiment runs found.")

    return pd.DataFrame(rows).sort_values(["base_experiment_id", "total_dataset_size"])


def save_experiment_plot(experiment_data: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Построить отдельный график точности от суммарного размера выборки для одного эксперимента."""

    experiment_name = str(experiment_data["base_experiment_id"].iloc[0])
    plt.figure(figsize=(8, 5))
    plt.plot(
        experiment_data["total_dataset_size"],
        experiment_data["accuracy"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Суммарный размер выборки, изображений")
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(f"{experiment_name}: точность от размера выборки")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_combined_plot(data: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Построить общий график для сравнения всех выбранных экспериментов."""

    plt.figure(figsize=(12, 7))
    for experiment_name, experiment_data in data.groupby("base_experiment_id"):
        plt.plot(
            experiment_data["total_dataset_size"],
            experiment_data["accuracy"],
            marker="o",
            linewidth=1.8,
            label=experiment_name,
        )

    plt.xlabel("Суммарный размер выборки, изображений")
    plt.ylabel(METRIC_LABELS[metric])
    plt.title("Точность от суммарного размера выборки")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Сохранить CSV и графики зависимости точности от размера датасета."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_experiments = set(args.experiments) if args.experiments else None
    selected_dataset_sizes = set(args.dataset_sizes) if args.dataset_sizes else None
    data = collect_rows(
        results_dir=args.results_dir,
        metric=args.metric,
        selected_experiments=selected_experiments,
        experiment_start=args.experiment_start,
        experiment_end=args.experiment_end,
        dataset_sizes=selected_dataset_sizes,
    )

    data_path = args.output_dir / f"{args.metric}_by_dataset_size_data.csv"
    data.to_csv(data_path, index=False)

    for experiment_name, experiment_data in data.groupby("base_experiment_id"):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(experiment_name))
        save_experiment_plot(
            experiment_data,
            args.metric,
            args.output_dir / f"{safe_name}_{args.metric}_by_dataset_size.png",
        )

    if args.make_combined_plot:
        save_combined_plot(data, args.metric, args.output_dir / f"combined_{args.metric}_by_dataset_size.png")

    print(f"Dataset-size accuracy plots saved: {args.output_dir}")
    print(f"Plot data saved: {data_path}")


if __name__ == "__main__":
    main()
