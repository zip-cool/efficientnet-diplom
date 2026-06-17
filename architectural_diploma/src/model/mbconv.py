from __future__ import annotations

import tensorflow as tf


def squeeze_excite(
    inputs: tf.Tensor,
    expanded_filters: int,
    se_ratio: float,
    prefix: str,
) -> tf.Tensor:
    """Построить Squeeze-and-Excitation часть MBConv-блока.

    Слой сначала сжимает пространственные признаки глобальным усреднением,
    затем учит веса важности каналов и умножает исходные признаки на эти веса.
    """

    reduced_filters = max(1, int(expanded_filters * se_ratio))
    x = tf.keras.layers.GlobalAveragePooling2D(name=f"{prefix}_se_squeeze")(inputs)
    x = tf.keras.layers.Reshape((1, 1, expanded_filters), name=f"{prefix}_se_reshape")(x)
    x = tf.keras.layers.Conv2D(
        reduced_filters,
        1,
        activation="swish",
        padding="same",
        name=f"{prefix}_se_reduce",
    )(x)
    x = tf.keras.layers.Conv2D(
        expanded_filters,
        1,
        activation="sigmoid",
        padding="same",
        name=f"{prefix}_se_expand",
    )(x)
    return tf.keras.layers.Multiply(name=f"{prefix}_se_excite")([inputs, x])


def mbconv_block(
    inputs: tf.Tensor,
    input_filters: int,
    output_filters: int,
    kernel_size: int,
    stride: int,
    expansion_ratio: int,
    se_ratio: float,
    use_se: bool,
    prefix: str,
) -> tf.Tensor:
    """Построить один MBConv-блок EfficientNet.

    Блок состоит из optional expand-свертки 1x1, depthwise-свертки, optional
    Squeeze-and-Excitation, project-свертки 1x1 и residual connection, если
    размеры входа и выхода совпадают.
    """

    x = inputs
    expanded_filters = input_filters * expansion_ratio

    # Expand-свертка увеличивает число каналов перед depthwise-сверткой.
    if expansion_ratio != 1:
        x = tf.keras.layers.Conv2D(
            expanded_filters,
            1,
            padding="same",
            use_bias=False,
            name=f"{prefix}_expand_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{prefix}_expand_bn")(x)
        x = tf.keras.layers.Activation("swish", name=f"{prefix}_expand_activation")(x)

    # Depthwise-свертка обрабатывает каждый канал отдельно и задает spatial часть блока.
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{prefix}_dwconv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn")(x)
    x = tf.keras.layers.Activation("swish", name=f"{prefix}_activation")(x)

    # SE можно отключать целиком для отдельной серии экспериментов.
    if use_se and se_ratio > 0:
        x = squeeze_excite(x, expanded_filters, se_ratio, prefix)

    # Project-свертка возвращает блок к нужному числу выходных каналов.
    x = tf.keras.layers.Conv2D(
        output_filters,
        1,
        padding="same",
        use_bias=False,
        name=f"{prefix}_project_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}_project_bn")(x)

    # Residual connection используется только когда форма тензоров совместима.
    if stride == 1 and input_filters == output_filters:
        x = tf.keras.layers.Add(name=f"{prefix}_add")([inputs, x])

    return x
