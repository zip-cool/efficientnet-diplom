from __future__ import annotations

import time

import tensorflow as tf


class TimeAndLearningRateCallback(tf.keras.callbacks.Callback):
    """Callback для сохранения времени эпохи и текущего learning rate."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_start = 0.0
        self.epoch_times: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Запомнить время начала эпохи."""

        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Добавить в logs длительность эпохи и learning rate."""

        logs = logs or {}
        epoch_time = time.perf_counter() - self.epoch_start
        self.epoch_times.append(float(epoch_time))
        logs["epoch_time_seconds"] = float(epoch_time)
        learning_rate = self.model.optimizer.learning_rate
        logs["learning_rate"] = float(tf.keras.backend.get_value(learning_rate))
