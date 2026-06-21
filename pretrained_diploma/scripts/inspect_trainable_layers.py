from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.model.pretrained_efficientnet import build_pretrained_model, freeze_pretrained_base
from pretrained_diploma.src.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    """Описать параметры проверки обучаемых слоёв модели."""

    parser = argparse.ArgumentParser(description="Inspect trainable layers after ImageNet transfer and freezing.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base-config", type=Path, default=Path("pretrained_diploma/configs/base_config.json"))
    parser.add_argument("--experiments", type=Path, default=Path("pretrained_diploma/configs/experiments.json"))
    return parser.parse_args()


def main() -> None:
    """Показать, какие слои backbone обучаются после заморозки перенесённых весов."""

    args = parse_args()
    config = load_experiment_config(args.experiment, args.base_config, args.experiments, args.epochs)
    model, backbone, structure = build_pretrained_model(config)
    freeze_pretrained_base(backbone)

    report = structure.get("weight_transfer_report", {})
    skipped_layers = [item["layer"] for item in report.get("skipped_layers", [])]
    trainable_backbone_layers = [layer.name for layer in backbone.layers if layer.trainable]

    result = {
        "experiment_id": config["experiment_id"],
        "dropped_block_numbers": structure.get("dropped_block_numbers"),
        "removed_blocks": structure.get("removed_blocks"),
        "transferred_layer_count": report.get("transferred_layer_count"),
        "skipped_layer_count": report.get("skipped_layer_count"),
        "skipped_layers": skipped_layers,
        "trainable_backbone_layers": trainable_backbone_layers,
        "trainable_skipped_layers": [
            layer_name for layer_name in skipped_layers if layer_name in trainable_backbone_layers
        ],
        "trainable_weight_names": [weight.name for weight in model.trainable_weights],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
