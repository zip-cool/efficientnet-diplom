from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.training.trainer import evaluate_model, load_split_dataset
from pretrained_diploma.src.utils.config import save_json


def parse_args() -> argparse.Namespace:
    """Описать параметры пересчета train/validation метрик для готовых запусков."""

    parser = argparse.ArgumentParser(description="Backfill train/validation evaluation metrics for completed runs.")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл отчета эксперимента."""

    return json.loads(path.read_text(encoding="utf-8"))


def available_epochs(results_dir: Path, experiment: str) -> list[int]:
    """Найти завершенные эпохи для выбранного эксперимента."""

    epochs: list[int] = []
    for run_dir in sorted(results_dir.glob(f"{experiment}_ep*")):
        if (run_dir / "final_metrics.json").exists():
            epochs.append(int(run_dir.name.rsplit("_ep", maxsplit=1)[-1]))
    return epochs


def metric_prefix(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Преобразовать результат evaluate_model в плоские поля JSON."""

    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_loss": metrics["loss"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_f1_score": metrics["f1_score"],
    }


def update_report(report_path: Path, updates: dict[str, float]) -> None:
    """Дополнить JSON-отчет новыми метриками без удаления старых полей."""

    report = load_json(report_path)
    report.update(updates)
    save_json(report_path, report)


def update_test_results(test_results_path: Path, updates: dict[str, float]) -> None:
    """Дополнить test_results краткими train/validation eval accuracy."""

    if not test_results_path.exists():
        return
    report = load_json(test_results_path)
    for key in ["train_eval_accuracy", "val_eval_accuracy"]:
        if key in updates:
            report[key] = updates[key]
    save_json(test_results_path, report)


def backfill_run(run_dir: Path) -> None:
    """Пересчитать train/validation метрики для одной папки результата."""

    final_metrics_path = run_dir / "final_metrics.json"
    config_path = run_dir / "config.json"
    if not final_metrics_path.exists() or not config_path.exists():
        print(f"Skipping incomplete run: {run_dir}")
        return

    final_metrics = load_json(final_metrics_path)
    config = load_json(config_path)
    model_path = Path(final_metrics.get("model_path", run_dir / "best_model.keras"))
    if not model_path.exists():
        model_path = run_dir / "best_model.keras"
    if not model_path.exists():
        print(f"Skipping run without model: {run_dir}")
        return

    dataset_dir = Path(config["dataset_dir"])
    image_size = int(config["input_size"])
    batch_size = int(config["batch_size"])
    seed = int(config["random_seed"])
    class_names = list(final_metrics.get("class_names", []))

    train_ds = load_split_dataset(dataset_dir / "train", image_size, batch_size, False, seed).prefetch(tf.data.AUTOTUNE)
    val_ds = load_split_dataset(dataset_dir / "val", image_size, batch_size, False, seed).prefetch(tf.data.AUTOTUNE)
    if not class_names:
        class_names = list(train_ds.class_names)

    model = tf.keras.models.load_model(model_path)
    train_metrics = evaluate_model(model, train_ds, class_names)
    val_metrics = evaluate_model(model, val_ds, class_names)
    updates = {
        **metric_prefix("train_eval", train_metrics),
        **metric_prefix("val_eval", val_metrics),
    }
    update_report(final_metrics_path, updates)
    metrics_report_path = run_dir / "metrics_report.json"
    if metrics_report_path.exists():
        update_report(metrics_report_path, updates)
    update_test_results(run_dir / "test_results.json", updates)
    print(f"Updated eval metrics: {run_dir}")


def main() -> None:
    """Пересчитать train/validation eval метрики для выбранных экспериментов."""

    args = parse_args()
    for experiment in args.experiments:
        epochs = args.epochs if args.epochs is not None else available_epochs(args.results_dir, experiment)
        for epoch_count in epochs:
            backfill_run(args.results_dir / f"{experiment}_ep{epoch_count:03d}")


if __name__ == "__main__":
    main()
