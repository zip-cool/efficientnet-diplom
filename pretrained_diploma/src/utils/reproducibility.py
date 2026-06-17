from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_reproducibility(seed: int) -> None:
    """Зафиксировать seed и очистить предыдущую TensorFlow-сессию."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
