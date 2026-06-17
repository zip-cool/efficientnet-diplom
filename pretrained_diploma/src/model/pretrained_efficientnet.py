from __future__ import annotations

from typing import Any

import tensorflow as tf


STAGE_CUT_LAYERS = {
    "full_model": "top_activation",
    "without_stage7": "block6d_add",
    "without_stages6_7": "block5c_add",
    "without_stages5_7": "block4c_add",
}

FINETUNE_PREFIXES = {
    "top_activation": ("block7", "top_"),
    "block6d_add": ("block6",),
    "block5c_add": ("block5",),
    "block4c_add": ("block4",),
}


def simple_mbconv_tail(inputs: tf.Tensor, expansion_ratio: int = 2, prefix: str = "tail_mbconv") -> tf.Tensor:
    """Добавить упрощенный MBConv-хвост без Squeeze-and-Excitation.

    Хвост обучается с нуля и не меняет spatial-размер карты признаков.
    Residual connection используется только если форма входа и выхода совпадает.
    """

    input_channels = int(inputs.shape[-1])
    expanded_channels = input_channels * expansion_ratio

    x = tf.keras.layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False, name=f"{prefix}_expand_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}_expand_bn")(x)
    x = tf.keras.layers.Activation("swish", name=f"{prefix}_expand_swish")(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{prefix}_dwconv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}_dw_bn")(x)
    x = tf.keras.layers.Activation("swish", name=f"{prefix}_dw_swish")(x)
    x = tf.keras.layers.Conv2D(input_channels, 1, padding="same", use_bias=False, name=f"{prefix}_project_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}_project_bn")(x)

    if inputs.shape == x.shape:
        x = tf.keras.layers.Add(name=f"{prefix}_add")([inputs, x])
    return x


def add_tail(
    features: tf.Tensor,
    tail_type: str,
    num_classes: int,
    dropout_rate: float,
) -> tf.Tensor:
    """Добавить классификационный хвост выбранного типа."""

    x = features
    if tail_type == "gap":
        pass
    elif tail_type == "conv1x1":
        x = tf.keras.layers.Conv2D(256, 1, padding="same", use_bias=False, name="tail_conv1x1")(x)
        x = tf.keras.layers.BatchNormalization(name="tail_conv1x1_bn")(x)
        x = tf.keras.layers.Activation("swish", name="tail_conv1x1_swish")(x)
    elif tail_type == "separable_conv":
        x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False, name="tail_sepconv")(x)
        x = tf.keras.layers.BatchNormalization(name="tail_sepconv_bn")(x)
        x = tf.keras.layers.Activation("swish", name="tail_sepconv_swish")(x)
    elif tail_type == "simple_mbconv":
        x = simple_mbconv_tail(x)
    else:
        raise ValueError(f"Unknown tail_type: {tail_type}")

    x = tf.keras.layers.GlobalAveragePooling2D(name="tail_gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="tail_dropout")(x)
    return tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_pretrained_model(config: dict[str, Any]) -> tuple[tf.keras.Model, tf.keras.Model, dict[str, Any]]:
    """Собрать модель на основе `EfficientNetB0(weights='imagenet')`.

    Все слои EfficientNetB0 получают предобученные веса. После выбранной точки
    обрезки добавляется новый хвост, который обучается с нуля.
    """

    input_size = int(config["input_size"])
    cut_layer = config["cut_layer"]
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size, input_size, 3),
    )
    if cut_layer not in {layer.name for layer in base_model.layers}:
        raise ValueError(f"Layer '{cut_layer}' not found in EfficientNetB0.")

    cut_output = base_model.get_layer(cut_layer).output
    outputs = add_tail(
        features=cut_output,
        tail_type=config["tail_type"],
        num_classes=int(config["num_classes"]),
        dropout_rate=float(config["dropout_rate"]),
    )
    model = tf.keras.Model(base_model.input, outputs, name=config["model_name"])
    structure = {
        "model_name": config["model_name"],
        "weights": "imagenet",
        "input_size": input_size,
        "cut_layer": cut_layer,
        "cut_point": config["cut_point"],
        "tail_type": config["tail_type"],
        "base_layers": len(base_model.layers),
        "model_layers": len(model.layers),
        "parameter_count": int(model.count_params()),
        "cut_output_shape": [dim if dim is None else int(dim) for dim in cut_output.shape],
        "finetune_prefixes": list(FINETUNE_PREFIXES.get(cut_layer, ())),
    }
    return model, base_model, structure


def freeze_pretrained_base(base_model: tf.keras.Model) -> None:
    """Заморозить всю предобученную часть EfficientNetB0."""

    base_model.trainable = False


def unfreeze_last_remaining_stage(base_model: tf.keras.Model, cut_layer: str) -> None:
    """Разморозить только последнюю оставшуюся стадию EfficientNetB0.

    BatchNormalization оставляется замороженной, чтобы fine-tuning был
    стабильнее на небольшом датасете.
    """

    prefixes = FINETUNE_PREFIXES.get(cut_layer, ())
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = any(layer.name.startswith(prefix) for prefix in prefixes)
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
