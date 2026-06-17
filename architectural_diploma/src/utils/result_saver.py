from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from architectural_diploma.src.utils.config import save_json
from architectural_diploma.src.utils.plotting import plot_history_from_csv


# Единый порядок колонок для общей таблицы экспериментов.
SUMMARY_FIELDS = [
    "experiment_id",
    "base_experiment_id",
    "experiment_name",
    "series",
    "max_epochs",
    "width_coefficient",
    "depth_coefficient",
    "expansion_ratio",
    "use_se",
    "se_ratio",
    "kernel_mode",
    "head_channels",
    "input_size",
    "completed_epochs",
    "best_val_accuracy",
    "best_val_loss",
    "test_accuracy",
    "test_loss",
    "total_training_time_seconds",
    "average_epoch_time_seconds",
    "status",
    "result_directory",
]


def prepare_experiment_dir(path: Path, overwrite: bool) -> None:
    """Подготовить папку результата и защитить старые данные от перезаписи."""

    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_model_summary(model: tf.keras.Model, path: Path) -> None:
    """Сохранить текстовый `model.summary()` в файл."""

    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    path.write_text(buffer.getvalue(), encoding="utf-8")


def save_history(history: dict[str, list[float]], epoch_times: list[dict[str, Any]], output_dir: Path) -> Path:
    """Сохранить историю обучения в CSV и JSON.

    В `history.csv` добавляется номер эпохи и, если есть данные callback,
    длительность каждой эпохи. Отдельный `epoch_times.csv` хранит только время.
    """

    epochs = list(range(1, len(history.get("loss", [])) + 1))
    rows = {"epoch": epochs, **history}
    if epoch_times:
        rows["epoch_time_seconds"] = [row["epoch_time_seconds"] for row in epoch_times]

    history_csv = output_dir / "history.csv"
    pd.DataFrame(rows).to_csv(history_csv, index=False)
    save_json(output_dir / "history.json", {"history": rows})
    pd.DataFrame(epoch_times).to_csv(output_dir / "epoch_times.csv", index=False)
    return history_csv


def append_summary_row(summary_csv: Path, row: dict[str, Any]) -> None:
    """Добавить или заменить строку эксперимента в общей summary-таблице.

    Если run-id уже есть в CSV, старая строка удаляется. Это нужно, чтобы
    повторный запуск с `--overwrite` не создавал дубликаты.
    """

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if summary_csv.exists():
        with summary_csv.open("r", newline="", encoding="utf-8") as file:
            rows = [
                existing_row
                for existing_row in csv.DictReader(file)
                if existing_row.get("experiment_id") != row.get("experiment_id")
            ]
    rows.append({field: row.get(field) for field in SUMMARY_FIELDS})

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_experiment_artifacts(
    config: dict[str, Any],
    model: tf.keras.Model,
    model_structure: dict[str, Any],
    history: dict[str, list[float]],
    epoch_times: list[dict[str, Any]],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    class_indices: dict[str, int],
    output_dir: Path,
    summary_csv: Path,
) -> None:
    """Сохранить все обязательные артефакты одного запуска.

    Функция отвечает за конфиг, структуру модели, историю обучения, времена
    эпох, итоговые метрики, тестовые результаты, графики и строку общей таблицы.
    """

    # Конфигурация и фактическая структура модели сохраняются первыми.
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "model_structure.json", model_structure)
    save_json(output_dir / "class_indices.json", class_indices)
    save_model_summary(model, output_dir / "model_summary.txt")

    # PNG-графики строятся из history.csv, а не из объекта History в памяти.
    history_csv = save_history(history, epoch_times, output_dir)
    plot_history_from_csv(history_csv, output_dir / "plots")

    best_val_accuracy = max(history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.get("val_loss", [0.0]))
    best_epoch_by_val_accuracy = history.get("val_accuracy", [0.0]).index(best_val_accuracy) + 1
    best_epoch_by_val_loss = history.get("val_loss", [0.0]).index(best_val_loss) + 1
    completed_epochs = len(history.get("loss", []))
    total_time = float(epoch_times[-1]["cumulative_time_seconds"]) if epoch_times else 0.0
    average_epoch_time = total_time / completed_epochs if completed_epochs else 0.0

    # final_metrics разделяет последние метрики и лучшие validation-значения.
    final_metrics = {
        "final_train_accuracy": train_metrics["accuracy"],
        "final_val_accuracy": val_metrics["accuracy"],
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics["loss"],
        "best_epoch_by_val_accuracy": best_epoch_by_val_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch_by_val_loss": best_epoch_by_val_loss,
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": total_time,
        "average_epoch_time_seconds": average_epoch_time,
        "completed_epochs": completed_epochs,
        "stopped_early": completed_epochs < int(config["max_epochs"]),
        "experiment_status": "completed",
    }
    # test_results хранит итоговую оценку лучшей модели на test split.
    test_results = {
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "evaluated_model": "best_model.keras",
        "best_checkpoint_path": str(output_dir / "best_model.keras"),
        "test_samples": test_metrics.get("samples"),
    }
    save_json(output_dir / "final_metrics.json", final_metrics)
    save_json(output_dir / "test_results.json", test_results)
    # Общая таблица нужна для сравнения всех запусков и построения pivot-таблиц.
    append_summary_row(
        summary_csv,
        {
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
            "completed_epochs": completed_epochs,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "total_training_time_seconds": total_time,
            "average_epoch_time_seconds": average_epoch_time,
            "status": "completed",
            "result_directory": str(output_dir),
        },
    )
