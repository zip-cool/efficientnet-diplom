from __future__ import annotations

import math


def round_filters(filters: int, width_coefficient: float, divisor: int = 8) -> int:
    """Масштабировать число каналов и округлить его до кратности `divisor`.

    EfficientNet использует похожее округление, чтобы после изменения ширины
    сети количество каналов оставалось удобным для вычислений и не становилось
    слишком маленьким относительно заданного коэффициента.
    """

    scaled = filters * width_coefficient
    new_filters = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * scaled:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Масштабировать количество повторений блоков в стадии.

    Значение округляется вверх, а минимум равен одному блоку, чтобы ни одна
    стадия EfficientNet-подобной архитектуры полностью не исчезала.
    """

    return max(1, int(math.ceil(depth_coefficient * repeats)))
