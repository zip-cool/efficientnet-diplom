from __future__ import annotations

import csv

import io

import time

import shutil

from pathlib import Path

from typing import Any

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import (  # type: ignore

    convert_variables_to_constants_v2_as_graph,

)

from pretrained_diploma.src.utils.config import save_json

from pretrained_diploma.src.utils.plotting import plot_history_from_csv

SUMMARY_FIELDS = [

    "experiment_id",

    "base_experiment_id",

    "experiment_name",

    "series",

    "resolution",

    "cut_point",

    "cut_layer",

    "tail_type",

    "test_accuracy",

    "test_loss",

    "head_training_time_seconds",

    "finetune_training_time_seconds",

    "total_training_time_seconds",

    "completed_epochs",

    "result_directory",

]

def prepare_experiment_dir(path: Path, overwrite: bool) -> None:

    """Создать папку запуска и защитить существующие результаты."""

    if path.exists() and any(path.iterdir()) and not overwrite:

        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    if path.exists() and overwrite:

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

def save_model_summary(model: tf.keras.Model, path: Path) -> None:

    """Сохранить `model.summary()` в текстовый файл."""

    buffer = io.StringIO()

    model.summary(print_fn=lambda line: buffer.write(line + "\n"))

    path.write_text(buffer.getvalue(), encoding="utf-8")

def save_history(history_rows: list[dict[str, Any]], output_dir: Path) -> Path:

    """Сохранить историю обучения в формате, пригодном для восстановления графиков."""

    history_csv = output_dir / "history.csv"

    pd.DataFrame(history_rows).to_csv(history_csv, index=False)

    save_json(output_dir / "history.json", {"history": history_rows})

    return history_csv

def upsert_summary(summary_csv: Path, row: dict[str, Any]) -> None:

    """Добавить или заменить строку run-id в общей summary-таблице."""

    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if summary_csv.exists():

        with summary_csv.open("r", newline="", encoding="utf-8") as file:

            rows = [

                old_row

                for old_row in csv.DictReader(file)

                if old_row.get("experiment_id") != row["experiment_id"]

            ]

    rows.append({field: row.get(field) for field in SUMMARY_FIELDS})

    with summary_csv.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)

        writer.writeheader()

        writer.writerows(rows)

def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_dir: Path) -> tuple[Path, Path]:

    """????????? confusion matrix ? CSV ? PNG, ??? ? ?????????? ??????."""

    csv_path = output_dir / "confusion_matrix.csv"

    png_path = output_dir / "confusion_matrix.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["true/pred", *class_names])

        for class_name, row in zip(class_names, matrix):

            writer.writerow([class_name, *row.tolist()])

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix, interpolation="nearest", cmap="Blues")

    plt.title("Confusion matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names, rotation=30, ha="right")

    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted")

    plt.ylabel("True")

    threshold = matrix.max() / 2 if matrix.size else 0

    for row_index in range(matrix.shape[0]):

        for col_index in range(matrix.shape[1]):

            value = matrix[row_index, col_index]

            color = "white" if value > threshold else "black"

            plt.text(col_index, row_index, value, ha="center", va="center", color=color)

    plt.tight_layout()

    plt.savefig(png_path, dpi=160)

    plt.close()

    return csv_path, png_path

def model_file_metrics(model_path: Path) -> dict[str, int | float]:

    """????????? ?????? ??????????? ?????? ?? ?????."""

    size_bytes = model_path.stat().st_size

    return {

        "model_file_size_bytes": size_bytes,

        "model_file_size_mb": size_bytes / (1024 * 1024),

    }

def parameter_metrics(model: tf.keras.Model) -> dict[str, int]:

    """????????? ?????, ????????? ? ???????????? ?????????? ??????????."""

    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))

    non_trainable = int(sum(np.prod(variable.shape) for variable in model.non_trainable_weights))

    return {

        "parameters_total": int(model.count_params()),

        "parameters_trainable": trainable,

        "parameters_non_trainable": non_trainable,

    }

def calculate_flops(model: tf.keras.Model, image_size: int) -> int | None:

    """??????? FLOPs ?? ???? forward pass ????? TensorFlow profiler."""

    try:

        input_spec = tf.TensorSpec([1, image_size, image_size, 3], tf.float32)

        concrete = tf.function(lambda inputs: model(inputs, training=False)).get_concrete_function(input_spec)

        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

        with tf.Graph().as_default() as graph:

            tf.graph_util.import_graph_def(graph_def, name="")

            options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            options["output"] = "none"
            profile = tf.compat.v1.profiler.profile(graph=graph, cmd="op", options=options)

        return int(profile.total_float_ops) if profile is not None else None

    except Exception as error:

        print(f"Could not calculate FLOPs for {model.name}: {error}")

        return None

def benchmark_inference(

    model: tf.keras.Model,

    image_size: int,

    batch_size: int = 1,

    warmup_runs: int = 2,

    benchmark_runs: int = 5,

) -> dict[str, float]:

    """???????? ???????? ????????? ?? ????????? batch ??? ??????? ??????."""

    inputs = tf.random.uniform([batch_size, image_size, image_size, 3], dtype=tf.float32)

    for _ in range(warmup_runs):

        _ = model(inputs, training=False)

    start = time.perf_counter()

    for _ in range(benchmark_runs):

        _ = model(inputs, training=False)

    elapsed = time.perf_counter() - start

    images_processed = batch_size * benchmark_runs

    images_per_second = images_processed / elapsed

    seconds_per_image = elapsed / images_processed

    return {

        "inference_benchmark_batch_size": batch_size,

        "inference_images_per_second": images_per_second,

        "inference_ms_per_image": seconds_per_image * 1000,

    }

def build_full_metrics_report(

    config: dict[str, Any],

    history_rows: list[dict[str, Any]],

    stage_times: dict[str, float],

    test_metrics: dict[str, Any],

    dataset_info: dict[str, Any],

    model: tf.keras.Model,

    model_path: Path,

    plot_path: Path,

    confusion_csv_path: Path,

    confusion_png_path: Path,

) -> dict[str, Any]:

    """??????? ?????? JSON-????? ? ???? ?? ?????????, ??? ???? ? ?????? ??????."""

    last_epoch = history_rows[-1] if history_rows else {}

    finetune_enabled = bool(config.get("finetune_enabled", False))

    report: dict[str, Any] = {

        "experiment_name": config["experiment_id"],

        "model_name": config.get("model_name", "pretrained_efficientnet"),

        "dataset_dir": config["dataset_dir"],

        "train_size": dataset_info["train_size"],

        "val_size": dataset_info["val_size"],

        "test_size": dataset_info["test_size"],

        "total_size": dataset_info["total_size"],

        "images_per_class": dataset_info["images_per_class"],

        "split_counts": dataset_info["split_counts"],

        "training_strategy": "head_only" if not finetune_enabled else "head_then_finetune",

        "augmentation": bool(config.get("augmentation", {}).get("enabled", False)),

        "weights": "imagenet",

        "seed": config["random_seed"],

        "image_size": config["input_size"],

        "batch_size": config["batch_size"],

        "epochs_head": len(history_rows) if not finetune_enabled else config.get("head_epochs", len(history_rows)),

        "epochs_finetune": 0 if not finetune_enabled else config.get("finetune_epochs", 0),

        "fine_tune_layers": 0 if not finetune_enabled else config.get("fine_tune_layers", 0),

        "learning_rate_head": config["head_learning_rate"],

        "learning_rate_finetune": 0 if not finetune_enabled else config["finetune_learning_rate"],

        "train_accuracy": last_epoch.get("accuracy"),

        "val_accuracy": last_epoch.get("val_accuracy"),

        "train_loss": last_epoch.get("loss"),

        "val_loss": last_epoch.get("val_loss"),

        "test_accuracy": test_metrics["accuracy"],

        "accuracy": test_metrics["sklearn_accuracy"],

        "precision": test_metrics["precision"],

        "recall": test_metrics["recall"],

        "f1_score": test_metrics["f1_score"],

        "test_loss": test_metrics["loss"],

        "training_time_seconds": stage_times["total"],

        "head_training_time_seconds": stage_times["head"],

        "finetune_training_time_seconds": stage_times["finetune"],

        "total_training_time_seconds": stage_times["total"],

        "completed_epochs": len(history_rows),

        "model_path": str(model_path),

        "plot_path": str(plot_path),

        "confusion_matrix_csv_path": str(confusion_csv_path),

        "confusion_matrix_png_path": str(confusion_png_path),

        "class_names": dataset_info["class_names"],

    }

    report.update(model_file_metrics(model_path))

    report.update(parameter_metrics(model))

    flops = calculate_flops(model, int(config["input_size"]))

    if flops is not None:

        report["flops_per_inference"] = flops

        report["gflops_per_inference"] = flops / 1_000_000_000

    report.update(benchmark_inference(model, int(config["input_size"])))

    if flops is not None:

        report["estimated_gflops_per_second"] = (

            report["gflops_per_inference"] * report["inference_images_per_second"]

        )

    return report

def save_artifacts(

    config: dict[str, Any],

    model: tf.keras.Model,

    structure: dict[str, Any],

    history_rows: list[dict[str, Any]],

    stage_times: dict[str, float],

    test_metrics: dict[str, Any],

    dataset_info: dict[str, Any],

    output_dir: Path,

    summary_csv: Path,

) -> None:

    """????????? ??? ???????????? ????? ?????? pretrained-????????????."""

    save_json(output_dir / "config.json", config)

    save_json(output_dir / "model_structure.json", structure)

    save_model_summary(model, output_dir / "model_summary.txt")

    history_csv = save_history(history_rows, output_dir)

    plot_history_from_csv(history_csv, output_dir / "plots")

    model_path = output_dir / "best_model.keras"

    if not model_path.exists():

        model_path = output_dir / "final_model.keras"

    plot_path = output_dir / "plots" / "accuracy.png"

    confusion_csv_path, confusion_png_path = save_confusion_matrix(

        test_metrics["confusion_matrix"],

        dataset_info["class_names"],

        output_dir,

    )

    final_metrics = build_full_metrics_report(

        config=config,

        history_rows=history_rows,

        stage_times=stage_times,

        test_metrics=test_metrics,

        dataset_info=dataset_info,

        model=model,

        model_path=model_path,

        plot_path=plot_path,

        confusion_csv_path=confusion_csv_path,

        confusion_png_path=confusion_png_path,

    )

    save_json(output_dir / "final_metrics.json", final_metrics)

    save_json(output_dir / "metrics_report.json", final_metrics)

    save_json(

        output_dir / "test_results.json",

        {

            "test_accuracy": test_metrics["accuracy"],

            "accuracy": test_metrics["sklearn_accuracy"],

            "precision": test_metrics["precision"],

            "recall": test_metrics["recall"],

            "f1_score": test_metrics["f1_score"],

            "test_loss": test_metrics["loss"],

            "evaluated_model": model_path.name,

            "test_samples": test_metrics["samples"],

        },

    )

    upsert_summary(

        summary_csv,

        {

            "experiment_id": config["experiment_id"],

            "base_experiment_id": config["base_experiment_id"],

            "experiment_name": config["experiment_name"],

            "series": config["series"],

            "resolution": config["input_size"],

            "cut_point": config["cut_point"],

            "cut_layer": config["cut_layer"],

            "tail_type": config["tail_type"],

            "test_accuracy": test_metrics["accuracy"],

            "test_loss": test_metrics["loss"],

            "head_training_time_seconds": stage_times["head"],

            "finetune_training_time_seconds": stage_times["finetune"],

            "total_training_time_seconds": stage_times["total"],

            "completed_epochs": len(history_rows),

            "result_directory": str(output_dir),

        },

    )

