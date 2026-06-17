from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """Описать параметры вывода слоев EfficientNetB0."""

    parser = argparse.ArgumentParser(description="Print EfficientNetB0 layer names and output shapes.")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("pretrained_diploma/docs/efficientnetb0_layers.txt"))
    return parser.parse_args()


def main() -> None:
    """Сохранить список слоев стандартной EfficientNetB0 для выбора границ стадий."""

    args = parse_args()
    model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(args.input_size, args.input_size, 3),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{index:03d} {layer.name}: {layer.output.shape}" for index, layer in enumerate(model.layers)]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Layer list saved: {args.output}")


if __name__ == "__main__":
    main()
