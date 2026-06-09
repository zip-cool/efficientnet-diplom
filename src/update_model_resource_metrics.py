"""Добавить ресурсные метрики модели в отчеты экспериментов.

Скрипт обновляет уже существующие JSON-отчеты и общий CSV:

- размер сохраненного файла модели;
- количество параметров;
- FLOPs на один forward pass;
- скорость инференса на текущей машине;
- оценку GFLOPs/s как FLOPs на изображение * изображений в секунду.

Важно: FLOPs на изображение описывает вычислительную сложность модели, а
GFLOPs/s зависит от железа и текущих условий запуска.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (  # type: ignore
    convert_variables_to_constants_v2_as_graph,
)


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки."""

    parser = argparse.ArgumentParser(
        description="Update experiment reports with model size, FLOPs and inference speed."
    )
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--experiments-csv", type=Path, default=Path("results/metrics/experiments.csv"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-отчет эксперимента."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Сохранить JSON-отчет без потери русских символов."""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def model_file_metrics(model_path: Path) -> dict[str, int | float]:
    """Посчитать размер файла модели на диске."""

    size_bytes = model_path.stat().st_size
    return {
        "model_file_size_bytes": size_bytes,
        "model_file_size_mb": size_bytes / (1024 * 1024),
    }


def parameter_metrics(model: tf.keras.Model) -> dict[str, int]:
    """Посчитать общее, обучаемое и замороженное количество параметров."""

    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
    non_trainable = int(sum(np.prod(variable.shape) for variable in model.non_trainable_weights))
    return {
        "parameters_total": int(model.count_params()),
        "parameters_trainable": trainable,
        "parameters_non_trainable": non_trainable,
    }


def calculate_flops(model: tf.keras.Model, image_size: int) -> int | None:
    """Оценить количество float operations на один forward pass.

    TensorFlow profiler считает операции для графа модели. Если для какой-то
    версии TensorFlow/Keras расчет недоступен, возвращаем None и не ломаем
    обновление остальных метрик.
    """

    try:
        input_spec = tf.TensorSpec([1, image_size, image_size, 3], tf.float32)
        concrete = tf.function(lambda inputs: model(inputs, training=False)).get_concrete_function(input_spec)
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            profile = tf.compat.v1.profiler.profile(graph=graph, cmd="op", options=options)

        return int(profile.total_float_ops) if profile is not None else None
    except Exception as error:
        print(f"Could not calculate FLOPs: {error}")
        return None


def benchmark_inference(
    model: tf.keras.Model,
    image_size: int,
    batch_size: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, float]:
    """Измерить скорость инференса на случайном batch.

    Это не универсальная характеристика модели: результат зависит от CPU/GPU,
    загруженности системы, версии TensorFlow и batch size. Но для сравнения
    одинаковых моделей в рамках одного компьютера метрика полезна.
    """

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
        "inference_benchmark_batch_size": float(batch_size),
        "inference_images_per_second": images_per_second,
        "inference_ms_per_image": seconds_per_image * 1000,
    }


def update_csv(csv_path: Path, reports: list[dict[str, Any]]) -> None:
    """Пересобрать общий CSV на основе JSON-отчетов."""

    if not reports:
        return

    excluded = {"class_names", "split_counts"}
    fieldnames: list[str] = []
    for report in reports:
        for key in report:
            if key not in excluded and key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    csv_path = args.experiments_csv.resolve()

    json_paths = sorted(
        path
        for path in metrics_dir.glob("*.json")
        if path.name != "summary.json"
    )
    if not json_paths:
        raise FileNotFoundError(f"No JSON reports found in {metrics_dir}")

    updated_reports: list[dict[str, Any]] = []

    for json_path in json_paths:
        report = load_json(json_path)
        model_path_value = report.get("model_path")
        if not model_path_value:
            print(f"Skipping {json_path.name}: no model_path")
            updated_reports.append(report)
            continue

        model_path = Path(model_path_value)
        if not model_path.exists():
            print(f"Skipping {json_path.name}: model not found: {model_path}")
            updated_reports.append(report)
            continue

        print(f"Updating {json_path.name}...")
        model = tf.keras.models.load_model(model_path, compile=False)

        report.update(model_file_metrics(model_path))
        report.update(parameter_metrics(model))

        flops = calculate_flops(model, args.image_size)
        if flops is not None:
            report["flops_per_inference"] = flops
            report["gflops_per_inference"] = flops / 1_000_000_000

        benchmark = benchmark_inference(
            model=model,
            image_size=args.image_size,
            batch_size=args.benchmark_batch_size,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        report.update(benchmark)

        if flops is not None:
            report["estimated_gflops_per_second"] = (
                report["gflops_per_inference"] * report["inference_images_per_second"]
            )

        write_json(json_path, report)
        updated_reports.append(report)

    update_csv(csv_path, updated_reports)
    print(f"Updated {len(updated_reports)} reports and {csv_path}")


if __name__ == "__main__":
    main()
