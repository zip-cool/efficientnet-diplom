from __future__ import annotations

from dataclasses import asdict, dataclass

import tensorflow as tf

from architectural_diploma.src.model.mbconv import mbconv_block
from architectural_diploma.src.model.scaling import round_filters, round_repeats


@dataclass(frozen=True)
class StageConfig:
    """Описание одной стадии базовой EfficientNetB0-подобной архитектуры."""

    kernel_size: int
    repeats: int
    input_filters: int
    output_filters: int
    expansion_ratio: int
    stride: int
    se_ratio: float


BASE_STAGES = [
    StageConfig(3, 1, 32, 16, 1, 1, 0.25),
    StageConfig(3, 2, 16, 24, 6, 2, 0.25),
    StageConfig(5, 2, 24, 40, 6, 2, 0.25),
    StageConfig(3, 3, 40, 80, 6, 2, 0.25),
    StageConfig(5, 3, 80, 112, 6, 1, 0.25),
    StageConfig(5, 4, 112, 192, 6, 2, 0.25),
    StageConfig(3, 1, 192, 320, 6, 1, 0.25),
]


def resolve_stages(
    width_coefficient: float,
    depth_coefficient: float,
    expansion_ratio: int | None,
    se_ratio: float,
    kernel_mode: str,
) -> list[dict[str, int | float]]:
    """Подготовить фактические параметры стадий для конкретного эксперимента.

    Здесь применяются коэффициенты ширины и глубины, замена expansion ratio,
    изменение SE ratio и режим ядер `standard` или `all_3`.
    """

    stages = []
    for stage in BASE_STAGES:
        resolved_expansion = stage.expansion_ratio
        if expansion_ratio is not None and stage.expansion_ratio != 1:
            resolved_expansion = expansion_ratio

        stages.append(
            {
                **asdict(stage),
                "kernel_size": 3 if kernel_mode == "all_3" else stage.kernel_size,
                "repeats": round_repeats(stage.repeats, depth_coefficient),
                "input_filters": round_filters(stage.input_filters, width_coefficient),
                "output_filters": round_filters(stage.output_filters, width_coefficient),
                "expansion_ratio": resolved_expansion,
                "se_ratio": se_ratio,
            }
        )
    return stages


def build_modified_efficientnet(
    input_shape: tuple[int, int, int],
    num_classes: int,
    width_coefficient: float = 1.0,
    depth_coefficient: float = 1.0,
    expansion_ratio: int | None = None,
    se_ratio: float = 0.25,
    use_se: bool = True,
    kernel_mode: str = "standard",
    head_channels: int = 1280,
    dropout_rate: float = 0.2,
    model_name: str = "modified_efficientnet",
) -> tuple[tf.keras.Model, dict[str, object]]:
    """Собрать параметризуемую EfficientNetB0-подобную модель.

    Функция возвращает саму Keras-модель и словарь `structure`, где сохраняются
    фактические параметры архитектуры после масштабирования. Этот словарь нужен
    для воспроизводимости и последующего описания эксперимента в дипломе.
    """

    if kernel_mode not in {"standard", "all_3"}:
        raise ValueError("kernel_mode must be 'standard' or 'all_3'.")

    resolved_stages = resolve_stages(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        expansion_ratio=expansion_ratio,
        se_ratio=se_ratio,
        kernel_mode=kernel_mode,
    )

    # Входные изображения приводятся к диапазону [0, 1] прямо внутри модели.
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Stem-блок уменьшает spatial-размер и создает начальное число каналов.
    stem_filters = round_filters(32, width_coefficient)
    x = tf.keras.layers.Conv2D(
        stem_filters,
        3,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("swish", name="stem_activation")(x)

    # Основная часть сети состоит из семи стадий MBConv-блоков.
    current_filters = stem_filters
    for stage_index, stage in enumerate(resolved_stages, start=1):
        for repeat_index in range(stage["repeats"]):
            stride = int(stage["stride"]) if repeat_index == 0 else 1
            block_input_filters = current_filters
            block_output_filters = int(stage["output_filters"])
            x = mbconv_block(
                inputs=x,
                input_filters=block_input_filters,
                output_filters=block_output_filters,
                kernel_size=int(stage["kernel_size"]),
                stride=stride,
                expansion_ratio=int(stage["expansion_ratio"]),
                se_ratio=float(stage["se_ratio"]),
                use_se=use_se,
                prefix=f"block{stage_index}{chr(97 + repeat_index)}",
            )
            current_filters = block_output_filters

    # Head-блок собирает признаки перед финальной классификацией.
    x = tf.keras.layers.Conv2D(
        head_channels,
        1,
        padding="same",
        use_bias=False,
        name="top_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="top_bn")(x)
    x = tf.keras.layers.Activation("swish", name="top_activation")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=model_name)
    # Структура сохраняется отдельно, чтобы видеть реальные каналы и повторы.
    structure = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "stem_filters": stem_filters,
        "head_channels": head_channels,
        "width_coefficient": width_coefficient,
        "depth_coefficient": depth_coefficient,
        "expansion_ratio_override": expansion_ratio,
        "use_se": use_se,
        "se_ratio": se_ratio,
        "kernel_mode": kernel_mode,
        "stages": resolved_stages,
        "parameter_count": int(model.count_params()),
    }
    return model, structure
