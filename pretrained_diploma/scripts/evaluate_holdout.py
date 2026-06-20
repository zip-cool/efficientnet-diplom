from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.training.trainer import evaluate_model, load_split_dataset
from pretrained_diploma.src.utils.result_saver import SUMMARY_FIELDS


def parse_args() -> argparse.Namespace:
    """Описать параметры проверки сохранённых моделей на независимой holdout-выборке."""

    parser = argparse.ArgumentParser(description="Evaluate saved experiment models on an external holdout dataset.")
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path("pretrained_diploma/data/holdout_not_train_val_300_per_class/test"),
    )
    parser.add_argument("--summary-csv", type=Path, default=Path("pretrained_diploma/results/tables/experiments_summary.csv"))
    parser.add_argument("--run-ids", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--experiment-start", type=int, default=None)
    parser.add_argument("--experiment-end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл."""

    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Сохранить словарь в JSON с нормальной кириллицей."""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def experiment_number(experiment_id: str) -> int | None:
    """Получить числовой номер из имени эксперимента вида exp30_name."""

    match = re.match(r"exp(\d+)", experiment_id)
    return int(match.group(1)) if match else None


def should_evaluate(run_dir: Path, config: dict[str, Any], args: argparse.Namespace) -> bool:
    """Проверить, входит ли запуск в выбранный пользователем набор."""

    if args.run_ids is not None:
        return run_dir.name in set(args.run_ids)

    experiment_name = str(config.get("base_experiment_id", run_dir.name.rsplit("_ep", maxsplit=1)[0]))
    if args.experiments is not None and experiment_name not in set(args.experiments):
        return False

    number = experiment_number(experiment_name)
    if args.experiment_start is not None and (number is None or number < args.experiment_start):
        return False
    if args.experiment_end is not None and (number is None or number > args.experiment_end):
        return False

    return True


def find_model_path(run_dir: Path, final_metrics: dict[str, Any]) -> Path:
    """Найти лучшую сохранённую модель для оценки."""

    candidates = []
    if final_metrics.get("model_path"):
        candidates.append(Path(final_metrics["model_path"]))
    candidates.extend([run_dir / "best_model.keras", run_dir / "final_model.keras"])
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved model found in {run_dir}")


def save_holdout_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_dir: Path) -> tuple[Path, Path]:
    """Сохранить confusion matrix holdout-оценки в CSV и PNG."""

    csv_path = output_dir / "holdout_confusion_matrix.csv"
    png_path = output_dir / "holdout_confusion_matrix.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Holdout confusion matrix")
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
            plt.text(col_index, row_index, int(value), ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    return csv_path, png_path


def holdout_updates(
    metrics: dict[str, Any],
    holdout_dir: Path,
    model_path: Path,
    confusion_csv_path: Path,
    confusion_png_path: Path,
) -> dict[str, Any]:
    """Подготовить плоские поля holdout-метрик для final_metrics.json."""

    return {
        "holdout_dataset_dir": str(holdout_dir),
        "holdout_evaluated_model": model_path.name,
        "holdout_accuracy": metrics["accuracy"],
        "holdout_sklearn_accuracy": metrics["sklearn_accuracy"],
        "holdout_loss": metrics["loss"],
        "holdout_precision": metrics["precision"],
        "holdout_recall": metrics["recall"],
        "holdout_f1_score": metrics["f1_score"],
        "holdout_samples": metrics["samples"],
        "holdout_confusion_matrix_csv_path": str(confusion_csv_path),
        "holdout_confusion_matrix_png_path": str(confusion_png_path),
    }


def update_json_report(path: Path, updates: dict[str, Any]) -> None:
    """Дописать holdout-поля в JSON-отчёт эксперимента."""

    if not path.exists():
        return
    data = load_json(path)
    data.update(updates)
    save_json(path, data)


def save_holdout_results(path: Path, metrics: dict[str, Any], updates: dict[str, Any], class_names: list[str]) -> None:
    """Сохранить отдельный подробный holdout-отчёт."""

    report = {
        **updates,
        "class_names": class_names,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    save_json(path, report)


def update_summary(summary_csv: Path, experiment_id: str, updates: dict[str, Any]) -> None:
    """Обновить holdout-поля в общей summary-таблице."""

    if not summary_csv.exists():
        return

    with summary_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    fields = list(SUMMARY_FIELDS)
    for field in ("holdout_accuracy", "holdout_loss", "holdout_samples"):
        if field not in fields:
            fields.append(field)

    for row in rows:
        if row.get("experiment_id") == experiment_id:
            row["holdout_accuracy"] = updates["holdout_accuracy"]
            row["holdout_loss"] = updates["holdout_loss"]
            row["holdout_samples"] = updates["holdout_samples"]

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def evaluate_run(run_dir: Path, holdout_dir: Path, summary_csv: Path, overwrite: bool) -> bool:
    """Оценить один завершённый эксперимент на holdout-выборке."""

    config = load_json(run_dir / "config.json")
    final_metrics_path = run_dir / "final_metrics.json"
    final_metrics = load_json(final_metrics_path)

    if final_metrics.get("holdout_accuracy") is not None and not overwrite:
        print(f"Skipping holdout already evaluated: {run_dir.name}")
        return False

    image_size = int(config["input_size"])
    batch_size = int(config["batch_size"])
    seed = int(config["random_seed"])

    holdout_ds = load_split_dataset(holdout_dir, image_size, batch_size, False, seed)
    class_names = list(holdout_ds.class_names)
    holdout_ds = holdout_ds.prefetch(tf.data.AUTOTUNE)

    model_path = find_model_path(run_dir, final_metrics)
    model = tf.keras.models.load_model(model_path)
    metrics = evaluate_model(model, holdout_ds, class_names)

    confusion_csv_path, confusion_png_path = save_holdout_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        run_dir,
    )
    updates = holdout_updates(metrics, holdout_dir, model_path, confusion_csv_path, confusion_png_path)

    update_json_report(final_metrics_path, updates)
    update_json_report(run_dir / "metrics_report.json", updates)
    save_holdout_results(run_dir / "holdout_results.json", metrics, updates, class_names)
    update_summary(summary_csv, config["experiment_id"], updates)

    print(f"{run_dir.name}: holdout_accuracy={updates['holdout_accuracy']:.6f}")
    return True


def main() -> None:
    """Запустить holdout-оценку для выбранных сохранённых экспериментов."""

    args = parse_args()
    if not args.holdout_dir.exists():
        raise FileNotFoundError(f"Holdout directory not found: {args.holdout_dir}")

    evaluated = 0
    for run_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        final_metrics_path = run_dir / "final_metrics.json"
        if not config_path.exists() or not final_metrics_path.exists():
            continue

        config = load_json(config_path)
        if not should_evaluate(run_dir, config, args):
            continue

        if evaluate_run(run_dir, args.holdout_dir, args.summary_csv, args.overwrite):
            evaluated += 1

    print(f"Holdout evaluation finished. Newly evaluated runs: {evaluated}")


if __name__ == "__main__":
    main()
