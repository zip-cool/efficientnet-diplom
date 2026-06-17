from __future__ import annotations

import time

import tensorflow as tf


class EpochTimeCallback(tf.keras.callbacks.Callback):
    """Callback для измерения времени обучения и времени каждой эпохи.

    Он добавляет `epoch_time_seconds` в logs Keras и хранит отдельный список
    времен эпох, который затем сохраняется в `epoch_times.csv`.
    """

    def __init__(self) -> None:
        """Инициализировать счетчики времени перед началом обучения."""

        super().__init__()
        self.training_start = 0.0
        self.epoch_start = 0.0
        self.epoch_times: list[dict[str, float | int]] = []
        self.total_training_time_seconds = 0.0

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Запомнить время начала всего обучения."""

        self.training_start = time.perf_counter()

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Запомнить время начала текущей эпохи."""

        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Посчитать длительность эпохи и накопленное время обучения."""

        epoch_time = time.perf_counter() - self.epoch_start
        cumulative_time = time.perf_counter() - self.training_start
        if logs is not None:
            logs["epoch_time_seconds"] = epoch_time
        self.epoch_times.append(
            {
                "epoch": epoch + 1,
                "epoch_time_seconds": float(epoch_time),
                "cumulative_time_seconds": float(cumulative_time),
            }
        )

    def on_train_end(self, logs: dict | None = None) -> None:
        """Посчитать итоговое время после завершения обучения."""

        self.total_training_time_seconds = time.perf_counter() - self.training_start
