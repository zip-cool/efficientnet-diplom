from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.model.pretrained_efficientnet import build_pretrained_model
from pretrained_diploma.src.utils.config import load_experiment_config, save_json


def parse_args() -> argparse.Namespace:
    """Описать параметры проверки полной реконструкции EfficientNetB0."""

    parser = argparse.ArgumentParser(description="Validate repeated-block EfficientNetB0 reconstruction.")
    parser.add_argument("--experiment", default="exp21_repeat_full")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--base-config", type=Path, default=Path("pretrained_diploma/configs/base_config.json"))
    parser.add_argument("--experiments", type=Path, default=Path("pretrained_diploma/configs/experiments.json"))
    parser.add_argument("--output", type=Path, default=Path("pretrained_diploma/results/validation/architecture_validation.json"))
    return parser.parse_args()


def main() -> None:
    """Проверить совпадение reconstructed full backbone со стандартной EfficientNetB0."""

    args = parse_args()
    config = load_experiment_config(args.experiment, args.base_config, args.experiments, args.epochs)
    model, backbone, structure = build_pretrained_model(config)
    standard = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(int(config["input_size"]), int(config["input_size"]), 3),
    )
    test_input = tf.random.uniform(
        [1, int(config["input_size"]), int(config["input_size"]), 3],
        seed=int(config["random_seed"]),
    )
    reconstructed_output = backbone(test_input, training=False)
    standard_output = standard(test_input, training=False)
    max_abs_difference = float(tf.reduce_max(tf.abs(reconstructed_output - standard_output)).numpy())

    report = {
        "experiment": args.experiment,
        "stage_repeats": config.get("stage_repeats"),
        "standard_stage_repeats": [1, 2, 2, 3, 3, 4, 1],
        "reconstructed_model_name": model.name,
        "backbone_layer_count": len(backbone.layers),
        "standard_layer_count": len(standard.layers),
        "backbone_parameter_count": int(backbone.count_params()),
        "standard_parameter_count": int(standard.count_params()),
        "output_shape": [dim if dim is None else int(dim) for dim in reconstructed_output.shape],
        "max_abs_difference": max_abs_difference,
        "is_equivalent": max_abs_difference < 1e-6,
        "weight_transfer_report": structure.get("weight_transfer_report"),
    }
    save_json(args.output, report)
    print(f"Architecture validation saved: {args.output}")
    print(f"max_abs_difference={max_abs_difference}")


if __name__ == "__main__":
    main()
