from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Зафиксировать генераторы случайных чисел перед каждым экспериментом.

    Очищается предыдущая TensorFlow-сессия, выставляются seed для Python, NumPy
    и TensorFlow. По возможности включаются deterministic operations.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
