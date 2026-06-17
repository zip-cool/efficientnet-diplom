from __future__ import annotations

import time

from pathlib import Path

from typing import Any

import numpy as np

import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from pretrained_diploma.src.model.pretrained_efficientnet import (

    build_pretrained_model,

    freeze_pretrained_base,

    unfreeze_last_remaining_stage,

)

from pretrained_diploma.src.training.callbacks import TimeAndLearningRateCallback

from pretrained_diploma.src.utils.reproducibility import set_reproducibility

from pretrained_diploma.src.utils.result_saver import prepare_experiment_dir, save_artifacts

def load_split_dataset(split_dir: Path, image_size: int, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:

    """Загрузить train/val/test split из папки с подпапками классов."""

    return tf.keras.utils.image_dataset_from_directory(

        split_dir,

        labels="inferred",

        label_mode="categorical",

        image_size=(image_size, image_size),

        batch_size=batch_size,

        shuffle=shuffle,

        seed=seed,

    )

def add_augmentation(dataset: tf.data.Dataset, config: dict[str, Any]) -> tf.data.Dataset:

    """Добавить одинаковую аугментацию только к train dataset."""

    augmentation_config = config["augmentation"]

    if not augmentation_config.get("enabled", False):

        return dataset

    augmentation = tf.keras.Sequential(

        [

            tf.keras.layers.RandomFlip(augmentation_config["random_flip"]),

            tf.keras.layers.RandomRotation(float(augmentation_config["random_rotation"])),

            tf.keras.layers.RandomZoom(float(augmentation_config["random_zoom"])),

            tf.keras.layers.RandomTranslation(

                float(augmentation_config["random_translation"]),

                float(augmentation_config["random_translation"]),

            ),

            tf.keras.layers.RandomContrast(float(augmentation_config["random_contrast"])),

        ],

        name="train_augmentation",

    )

    return dataset.map(lambda images, labels: (augmentation(images, training=True), labels), num_parallel_calls=tf.data.AUTOTUNE)

def compile_model(model: tf.keras.Model, learning_rate: float, loss: str, metric: str) -> None:

    """Скомпилировать модель с Adam и заданным learning rate."""

    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=loss,

        metrics=[metric],

    )

def split_epochs(total_epochs: int, head_fraction: float) -> tuple[int, int]:

    """Разделить общее число эпох на обучение головы и fine-tuning.

    В актуальной серии fine-tuning отключен, поэтому эта функция используется

    только как запасной механизм, если его снова явно включат в конфиге.

    """

    head_epochs = max(1, int(round(total_epochs * head_fraction)))

    finetune_epochs = max(1, total_epochs - head_epochs)

    if total_epochs == 1:

        return 1, 0

    if head_epochs >= total_epochs:

        head_epochs = total_epochs - 1

        finetune_epochs = 1

    return head_epochs, finetune_epochs

def history_to_rows(history: tf.keras.callbacks.History, start_epoch: int) -> list[dict[str, Any]]:

    """Преобразовать Keras History в строки общего `history.csv`."""

    rows = []

    history_data = history.history

    epoch_count = len(history_data.get("loss", []))

    for index in range(epoch_count):

        rows.append(

            {

                "epoch": start_epoch + index,

                "accuracy": float(history_data["accuracy"][index]),

                "val_accuracy": float(history_data["val_accuracy"][index]),

                "loss": float(history_data["loss"][index]),

                "val_loss": float(history_data["val_loss"][index]),

                "epoch_time_seconds": float(history_data.get("epoch_time_seconds", [0.0] * epoch_count)[index]),

                "learning_rate": float(history_data.get("learning_rate", [0.0] * epoch_count)[index]),

            }

        )

    return rows

def count_images_by_class(split_dir: Path) -> dict[str, int]:

    """????????? ?????????? ??????????? ? split ?? ??????? ??????."""

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

    """??????? ?????????? ??????????? ?? ???? ??????? ?????? split."""

    return sum(counts.values())

def infer_images_per_class(split_counts: dict[str, dict[str, int]]) -> int | None:

    """?????????? ????? ?????? ???????????????? ??????? ?? ?????."""

    class_totals: dict[str, int] = {}

    for counts in split_counts.values():

        for class_name, count in counts.items():

            class_totals[class_name] = class_totals.get(class_name, 0) + count

    unique_totals = set(class_totals.values())

    if len(unique_totals) != 1:

        return None

    return unique_totals.pop() if unique_totals else None

def count_samples(dataset: tf.data.Dataset) -> int:

    """????????? ?????????? ??????????? ? tf.data.Dataset."""

    return sum(int(labels.shape[0]) for _, labels in dataset)

def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:

    """??????? ???????? ? ????????????? ?????? ??? ??????? ?????? ????????."""

    y_true_batches: list[np.ndarray] = []

    y_pred_batches: list[np.ndarray] = []

    for images, labels in dataset:

        predictions = model.predict(images, verbose=0)

        y_true_batches.append(np.argmax(labels.numpy(), axis=1))

        y_pred_batches.append(np.argmax(predictions, axis=1))

    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)

def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict[str, Any]:

    """??????? ?????? ?? test split ? ????????? ??????? ??? ? ?????????? ??????."""

    loss, accuracy = model.evaluate(dataset, verbose=0)

    y_true, y_pred = collect_predictions(model, dataset)

    precision, recall, f1_score, _ = precision_recall_fscore_support(

        y_true,

        y_pred,

        average="macro",

        zero_division=0,

    )

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {

        "loss": float(loss),

        "accuracy": float(accuracy),

        "sklearn_accuracy": float(accuracy_score(y_true, y_pred)),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1_score),

        "confusion_matrix": matrix,

        "samples": count_samples(dataset),

    }

def run_experiment(config: dict[str, Any], overwrite: bool = False) -> Path:

    """Выполнить один pretrained-эксперимент от загрузки данных до сохранения файлов."""

    set_reproducibility(int(config["random_seed"]))

    output_dir = Path(config["results_dir"]) / config["experiment_id"]

    prepare_experiment_dir(output_dir, overwrite)

    dataset_dir = Path(config["dataset_dir"])

    image_size = int(config["input_size"])

    batch_size = int(config["batch_size"])

    seed = int(config["random_seed"])

    train_ds = load_split_dataset(dataset_dir / "train", image_size, batch_size, True, seed)
    train_eval_ds = load_split_dataset(dataset_dir / "train", image_size, batch_size, False, seed)
    val_ds = load_split_dataset(dataset_dir / "val", image_size, batch_size, False, seed)
    test_ds = load_split_dataset(dataset_dir / "test", image_size, batch_size, False, seed)
    class_names = list(train_ds.class_names)

    split_counts = {

        "train": count_images_by_class(dataset_dir / "train"),

        "val": count_images_by_class(dataset_dir / "val"),

        "test": count_images_by_class(dataset_dir / "test"),

    }

    dataset_info = {

        "train_size": total_count(split_counts["train"]),

        "val_size": total_count(split_counts["val"]),

        "test_size": total_count(split_counts["test"]),

        "total_size": sum(total_count(counts) for counts in split_counts.values()),

        "images_per_class": infer_images_per_class(split_counts),

        "split_counts": split_counts,

        "class_names": class_names,

    }

    train_ds = add_augmentation(train_ds, config).prefetch(tf.data.AUTOTUNE)
    train_eval_ds = train_eval_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model, base_model, structure = build_pretrained_model(config)

    if bool(config.get("finetune_enabled", False)):

        head_epochs, finetune_epochs = split_epochs(int(config["max_epochs"]), float(config["head_epoch_fraction"]))

    else:

        head_epochs = int(config["max_epochs"])

        finetune_epochs = 0

    history_rows: list[dict[str, Any]] = []

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

        output_dir / "best_model.keras",

        monitor="val_loss",

        save_best_only=True,

    )

    # В актуальной методике вся EfficientNetB0 остается замороженной.

    # Обучаются только новые слои классификационной головы или хвоста.

    freeze_pretrained_base(base_model)

    compile_model(model, float(config["head_learning_rate"]), config["loss"], config["metric"])

    head_timer = TimeAndLearningRateCallback()

    head_start = time.perf_counter()

    head_history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=head_epochs,

        callbacks=[head_timer, checkpoint_callback],

        verbose=1,

    )

    head_time = time.perf_counter() - head_start

    history_rows.extend(history_to_rows(head_history, start_epoch=1))

    finetune_time = 0.0

    if finetune_epochs > 0:

        unfreeze_last_remaining_stage(base_model, config["cut_layer"])

        compile_model(model, float(config["finetune_learning_rate"]), config["loss"], config["metric"])

        finetune_timer = TimeAndLearningRateCallback()

        finetune_start = time.perf_counter()

        finetune_history = model.fit(

            train_ds,

            validation_data=val_ds,

            epochs=head_epochs + finetune_epochs,

            initial_epoch=head_epochs,

            callbacks=[finetune_timer, checkpoint_callback],

            verbose=1,

        )

        finetune_time = time.perf_counter() - finetune_start

        history_rows.extend(history_to_rows(finetune_history, start_epoch=head_epochs + 1))

    model.save(output_dir / "final_model.keras")

    if (output_dir / "best_model.keras").exists():
        model = tf.keras.models.load_model(output_dir / "best_model.keras")

    train_eval_metrics = evaluate_model(model, train_eval_ds, class_names)
    val_eval_metrics = evaluate_model(model, val_ds, class_names)
    test_metrics = evaluate_model(model, test_ds, class_names)

    save_artifacts(

        config=config,

        model=model,

        structure=structure,

        history_rows=history_rows,

        stage_times={

            "head": float(head_time),

            "finetune": float(finetune_time),

            "total": float(head_time + finetune_time),

        },

        test_metrics=test_metrics,
        train_eval_metrics=train_eval_metrics,
        val_eval_metrics=val_eval_metrics,
        dataset_info=dataset_info,

        output_dir=output_dir,

        summary_csv=Path(config["summary_csv"]),

    )

    return output_dir

