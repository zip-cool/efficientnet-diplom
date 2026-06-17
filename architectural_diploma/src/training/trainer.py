from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from architectural_diploma.src.model.efficientnet_modified import build_modified_efficientnet
from architectural_diploma.src.training.callbacks import EpochTimeCallback
from architectural_diploma.src.utils.reproducibility import set_reproducibility
from architectural_diploma.src.utils.result_saver import prepare_experiment_dir, save_experiment_artifacts


def load_split_dataset(
    split_dir: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Загрузить один split датасета из папки с подпапками классов.

    Используется стандартная структура `train/<class_name>`, `val/<class_name>`,
    `test/<class_name>`. Метки загружаются в формате one-hot, потому что модель
    обучается с `categorical_crossentropy`.
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


def count_dataset_samples(dataset: tf.data.Dataset) -> int:
    """Посчитать количество изображений в уже собранном `tf.data.Dataset`."""

    total = 0
    for _, labels in dataset:
        total += int(labels.shape[0])
    return total


def evaluate_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, float]:
    """Оценить модель на датасете и вернуть loss, accuracy и число примеров."""

    loss, accuracy = model.evaluate(dataset, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy), "samples": count_dataset_samples(dataset)}


def build_callbacks(config: dict[str, Any], output_dir: Path, timer: EpochTimeCallback) -> list[tf.keras.callbacks.Callback]:
    """Собрать одинаковый набор callbacks для всех архитектурных экспериментов.

    В список входят измерение времени, CSVLogger, сохранение лучшей модели,
    EarlyStopping и optional ReduceLROnPlateau из общего конфига.
    """

    callbacks: list[tf.keras.callbacks.Callback] = [timer]
    callbacks.append(tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        )
    )

    early = config["early_stopping"]
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=early["monitor"],
            patience=int(early["patience"]),
            restore_best_weights=bool(early["restore_best_weights"]),
        )
    )

    reduce_lr = config["reduce_lr_on_plateau"]
    if reduce_lr.get("enabled", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_lr["monitor"],
                factor=float(reduce_lr["factor"]),
                patience=int(reduce_lr["patience"]),
                min_lr=float(reduce_lr["min_lr"]),
            )
        )
    return callbacks


def run_experiment(config: dict[str, Any], overwrite: bool = False) -> Path:
    """Выполнить полный цикл одного запуска архитектурного эксперимента.

    Функция выставляет seed, загружает train/val/test, строит модель из конфига,
    обучает ее, оценивает лучшую сохраненную модель и сохраняет все артефакты
    запуска в отдельную папку.
    """

    set_reproducibility(int(config["random_seed"]))

    # Датасет берется из уже подготовленного фиксированного разбиения.
    dataset_dir = Path(config["dataset_dir"])
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"
    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    # Каждый run-id имеет собственную папку, например exp00_baseline_ep020.
    output_dir = Path(config["results_dir"]) / config["experiment_id"]
    prepare_experiment_dir(output_dir, overwrite=overwrite)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    image_size = int(config["input_size"])
    batch_size = int(config["batch_size"])
    seed = int(config["random_seed"])
    # Train перемешивается, validation и test остаются в стабильном порядке.
    train_ds = load_split_dataset(train_dir, image_size, batch_size, True, seed)
    val_ds = load_split_dataset(val_dir, image_size, batch_size, False, seed)
    test_ds = load_split_dataset(test_dir, image_size, batch_size, False, seed)
    class_indices = {name: index for index, name in enumerate(train_ds.class_names)}

    # Prefetch ускоряет подачу батчей без изменения состава выборок.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # Архитектура строится строго из параметров config.
    model, structure = build_modified_efficientnet(
        input_shape=(image_size, image_size, 3),
        num_classes=int(config["num_classes"]),
        width_coefficient=float(config["width_coefficient"]),
        depth_coefficient=float(config["depth_coefficient"]),
        expansion_ratio=config["expansion_ratio"],
        se_ratio=float(config["se_ratio"]),
        use_se=bool(config["use_se"]),
        kernel_mode=config["kernel_mode"],
        head_channels=int(config["head_channels"]),
        dropout_rate=float(config["dropout_rate"]),
        model_name=config["model_name"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["learning_rate"])),
        loss=config["loss"],
        metrics=[config["metric"]],
    )

    # Callback времени передается отдельно, чтобы затем сохранить epoch_times.csv.
    timer = EpochTimeCallback()
    callbacks = build_callbacks(config, output_dir, timer)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["max_epochs"]),
        callbacks=callbacks,
        verbose=1,
    ).history
    model.save(output_dir / "final_model.keras")

    # Для итоговой оценки загружается лучшая модель по val_loss.
    best_model_path = output_dir / "best_model.keras"
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    train_metrics = evaluate_dataset(model, train_ds)
    val_metrics = evaluate_dataset(model, val_ds)
    test_metrics = evaluate_dataset(model, test_ds)

    # Вся история, метрики, модели и графики сохраняются в едином формате.
    save_experiment_artifacts(
        config=config,
        model=model,
        model_structure=structure,
        history={key: [float(value) for value in values] for key, values in history.items()},
        epoch_times=timer.epoch_times,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        class_indices=class_indices,
        output_dir=output_dir,
        summary_csv=Path(config["summary_csv"]),
    )
    return output_dir
