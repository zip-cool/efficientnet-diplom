"""Запуск экспериментов EfficientNetB0 для классификации болезней картофеля.

Скрипт сделан как единая точка запуска для нескольких экспериментов из
дипломной работы:

1. ``head_only`` - базовая часть EfficientNetB0 заморожена, обучается только
   новая классификационная голова.
2. ``fine_tune`` - сначала обучается голова, затем размораживается часть
   последних слоев EfficientNetB0 и выполняется дообучение с малым learning
   rate.

Для честного сравнения скрипт сохраняет не только итоговую accuracy, но и
precision, recall, F1-score, loss, confusion matrix, график обучения, модель и
время обучения.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для воспроизводимого эксперимента."""

    parser = argparse.ArgumentParser(description="Run an EfficientNetB0 training experiment.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/experiments/balanced_1000_per_class_split"),
    )
    parser.add_argument("--experiment-name", default="exp01_head_only_1000_per_class")
    parser.add_argument(
        "--training-strategy",
        choices=("head_only", "fine_tune"),
        default="head_only",
    )
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=5)

    # Для transfer learning обычно используют более высокий learning rate для
    # новой головы и значительно меньший learning rate для fine-tuning, чтобы не
    # разрушить предобученные признаки EfficientNet.
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-5)
    parser.add_argument("--fine-tune-layers", type=int, default=20)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use 'none' only for offline smoke tests; thesis experiments should use imagenet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    parser.add_argument(
        "--confusion-dir",
        type=Path,
        default=Path("results/confusion_matrices"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models/final"))
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    """Зафиксировать генераторы случайных чисел для повторяемости запусков."""

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_split_dataset(
    split_dir: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Загрузить один split датасета из папки с подпапками классов.

    ``image_dataset_from_directory`` сам назначает метки по названиям папок.
    Используем ``categorical`` labels, потому что модель обучается с
    ``categorical_crossentropy`` и softmax-выходом на 3 класса.
    """

    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def count_images_by_class(split_dir: Path) -> dict[str, int]:
    """Посчитать количество изображений в split по каждому классу."""

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: dict[str, int] = {}

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for image_path in class_dir.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in image_extensions
        )

    return counts


def total_count(counts: dict[str, int]) -> int:
    """Вернуть общее количество изображений по словарю class -> count."""

    return sum(counts.values())


def infer_images_per_class(split_counts: dict[str, dict[str, int]]) -> int | None:
    """Определить размер исходной сбалансированной выборки на класс.

    Для текущих экспериментов после split 700/150/150 по каждому классу
    складывается обратно в 1000. Если классы окажутся несбалансированными,
    возвращаем None, чтобы не вводить в заблуждение.
    """

    class_totals: dict[str, int] = {}
    for counts in split_counts.values():
        for class_name, count in counts.items():
            class_totals[class_name] = class_totals.get(class_name, 0) + count

    unique_totals = set(class_totals.values())
    if len(unique_totals) != 1:
        return None

    return unique_totals.pop() if unique_totals else None


def build_model(
    image_size: int,
    class_names: list[str],
    augmentation: bool,
    weights: str,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Собрать модель EfficientNetB0 с новой головой классификации."""

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = inputs

    if augmentation:
        # Аугментация встроена в модель, поэтому она применяется только во время
        # обучения. На validation/test TensorFlow автоматически переводит эти
        # слои в режим без случайных преобразований.
        x = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomTranslation(0.05, 0.05),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )(x)

    # include_top=False убирает исходную ImageNet-классификационную голову.
    # В Keras EfficientNet уже содержит нужную preprocessing-логику внутри
    # модели, поэтому отдельно tf.keras.applications.efficientnet.preprocess_input
    # здесь не вызывается.
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None if weights == "none" else weights,
        input_shape=(image_size, image_size, 3),
    )

    # На первом этапе transfer learning замораживаем EfficientNetB0 и обучаем
    # только новую классификационную часть. Это эксперимент "head_only".
    base_model.trainable = False

    # training=False важно для замороженной базы: BatchNorm-слои EfficientNet не
    # обновляют свои статистики на новом датасете во время обучения головы.
    x = base_model(x, training=False)

    # GlobalAveragePooling превращает карту признаков EfficientNet в компактный
    # вектор признаков, а Dropout немного снижает риск переобучения.
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)

    # Softmax возвращает вероятности трех классов: Early_Blight, Healthy,
    # Late_Blight. Порядок классов берется из train-папки.
    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
        name="predictions",
    )(x)

    return tf.keras.Model(inputs, outputs), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Скомпилировать модель перед очередным этапом обучения."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_last_layers(base_model: tf.keras.Model, fine_tune_layers: int) -> None:
    """Разморозить последние слои EfficientNetB0 для fine-tuning."""

    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - fine_tune_layers)

    # Все ранние слои оставляем замороженными: они извлекают более общие
    # признаки вроде границ, текстур и простых форм.
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # BatchNormalization часто оставляют замороженным при fine-tuning маленьких
    # или средних датасетов, чтобы не испортить накопленные статистики.
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Собрать истинные и предсказанные классы для тестовой выборки."""

    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true_batches.append(np.argmax(labels.numpy(), axis=1))
        y_pred_batches.append(np.argmax(predictions, axis=1))

    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)


def plot_history(history: dict[str, list[float]], path: Path) -> None:
    """Сохранить графики accuracy и loss по эпохам."""

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], label="Точность на обучающей выборке")
    plt.plot(epochs, history["val_accuracy"], label="Точность на тестовой выборке")
    plt.xlabel("Количество эпох")
    plt.ylabel("Точность")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], label="Функция потерь на обучающей выборке")
    plt.plot(epochs, history["val_loss"], label="Функция потерь на тестовой выборке")
    plt.xlabel("Количество эпох")
    plt.ylabel("Функция потерь")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    csv_path: Path,
    png_path: Path,
) -> None:
    """Сохранить confusion matrix в CSV и PNG.

    CSV удобен для таблиц в дипломе, PNG - для визуального анализа ошибок между
    классами.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

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


def merge_histories(histories: list[tf.keras.callbacks.History]) -> dict[str, list[float]]:
    """Объединить историю обучения головы и fine-tuning в один график."""

    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def append_metrics_row(csv_path: Path, row: dict[str, object]) -> None:
    """Добавить строку эксперимента в общий CSV с результатами."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)

    # Ожидается классическая структура:
    # dataset_dir/train/<class_name>/*.jpg
    # dataset_dir/val/<class_name>/*.jpg
    # dataset_dir/test/<class_name>/*.jpg
    dataset_dir = args.dataset_dir.resolve()
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"
    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    # train перемешивается, validation и test не перемешиваются. Для оценки это
    # не влияет на метрики, но делает порядок предсказаний более прозрачным.
    train_ds = load_split_dataset(train_dir, args.image_size, args.batch_size, True, args.seed)
    val_ds = load_split_dataset(val_dir, args.image_size, args.batch_size, False, args.seed)
    test_ds = load_split_dataset(test_dir, args.image_size, args.batch_size, False, args.seed)
    class_names = train_ds.class_names

    # Размеры split считаем по файловой структуре, а не по confusion matrix.
    # Confusion matrix отражает только test split, поэтому поле dataset_size
    # раньше могло сбивать с толку.
    split_counts = {
        "train": count_images_by_class(train_dir),
        "val": count_images_by_class(val_dir),
        "test": count_images_by_class(test_dir),
    }
    train_size = total_count(split_counts["train"])
    val_size = total_count(split_counts["val"])
    test_size = total_count(split_counts["test"])
    total_size = train_size + val_size + test_size
    images_per_class = infer_images_per_class(split_counts)

    # Prefetch подготавливает следующий batch параллельно с обучением текущего,
    # что обычно ускоряет обучение без изменения результатов эксперимента.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model, base_model = build_model(
        image_size=args.image_size,
        class_names=class_names,
        augmentation=args.augmentation,
        weights=args.weights,
    )

    histories: list[tf.keras.callbacks.History] = []
    start_time = time.perf_counter()

    # Этап 1: обучаем новую классификационную голову при замороженной
    # EfficientNetB0. Даже для fine_tune-эксперимента этот этап выполняется
    # первым, чтобы голова сначала адаптировалась к трем классам.
    compile_model(model, args.learning_rate_head)
    histories.append(
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_head,
            verbose=1,
        )
    )

    if args.training_strategy == "fine_tune":
        # Этап 2: размораживаем часть последних слоев и дообучаем модель с
        # маленьким learning rate. Это проверяет, дает ли адаптация признаков
        # EfficientNet прирост качества на целевом датасете.
        unfreeze_last_layers(base_model, args.fine_tune_layers)
        compile_model(model, args.learning_rate_finetune)
        histories.append(
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs_finetune,
                verbose=1,
            )
        )

    # Время обучения фиксируем отдельно, потому что в дипломе оптимизация
    # EfficientNet оценивается не только по качеству, но и по затратам времени.
    training_time = time.perf_counter() - start_time

    # Итоговая оценка проводится только на test split, который не участвовал в
    # обучении и подборе параметров.
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_true, y_pred = collect_predictions(model, test_ds)

    # Macro-average считает метрики одинаково важными для каждого класса. Это
    # удобно для многоклассовой классификации, особенно если в будущих выборках
    # баланс классов изменится.
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Пути артефактов строятся от experiment_name, чтобы результаты разных
    # запусков не перезаписывали друг друга случайно.
    history = merge_histories(histories)
    metrics_dir = args.metrics_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    confusion_dir = args.confusion_dir.resolve()
    models_dir = args.models_dir.resolve()

    plot_path = plots_dir / f"{args.experiment_name}_history.png"
    confusion_csv_path = confusion_dir / f"{args.experiment_name}_confusion_matrix.csv"
    confusion_png_path = confusion_dir / f"{args.experiment_name}_confusion_matrix.png"
    model_path = models_dir / f"{args.experiment_name}.keras"
    metrics_json_path = metrics_dir / f"{args.experiment_name}.json"
    metrics_csv_path = metrics_dir / "experiments.csv"

    # Сохраняем все артефакты, которые нужны для анализа и текста диплома.
    plot_history(history, plot_path)
    save_confusion_matrix(cm, class_names, confusion_csv_path, confusion_png_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    # final_metrics содержит как параметры запуска, так и итоговые метрики.
    # Такой json удобно использовать для проверки воспроизводимости эксперимента.
    final_metrics = {
        "experiment_name": args.experiment_name,
        "model_name": "EfficientNetB0",
        "dataset_dir": str(dataset_dir),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_size": total_size,
        "images_per_class": images_per_class,
        "split_counts": split_counts,
        "training_strategy": args.training_strategy,
        "augmentation": args.augmentation,
        "weights": args.weights,
        "seed": args.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune
        if args.training_strategy == "fine_tune"
        else 0,
        "fine_tune_layers": args.fine_tune_layers
        if args.training_strategy == "fine_tune"
        else 0,
        "learning_rate_head": args.learning_rate_head,
        "learning_rate_finetune": args.learning_rate_finetune
        if args.training_strategy == "fine_tune"
        else 0,
        "train_accuracy": history["accuracy"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "train_loss": history["loss"][-1],
        "val_loss": history["val_loss"][-1],
        "test_accuracy": float(test_accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "test_loss": float(test_loss),
        "training_time_seconds": float(training_time),
        "model_path": str(model_path),
        "plot_path": str(plot_path),
        "confusion_matrix_csv_path": str(confusion_csv_path),
        "confusion_matrix_png_path": str(confusion_png_path),
        "class_names": class_names,
    }

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Общий CSV нужен для сравнения нескольких экспериментов в одной таблице:
    # head_only, fine_tune, augmentation и разные объемы данных.
    append_metrics_row(
        metrics_csv_path,
        {
            key: value
            for key, value in final_metrics.items()
            if key not in {"class_names", "split_counts"}
        },
    )

    # Печатаем тот же словарь в консоль, чтобы после запуска сразу видеть
    # главные результаты без открытия файлов.
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
