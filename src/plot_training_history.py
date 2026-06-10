"""Построение графиков обучения EfficientNet в стиле для диплома.

Скрипт строит один PNG с двумя графиками:
- Loss by Epoch;
- Accuracy by Epoch.

Подписи эпох делятся на этапы ``head`` и ``finetune``, а граница между этапами
показывается пунктирной линией. Такой формат удобнее для анализа двухэтапного
обучения EfficientNet: сначала обучение классификационной головы, затем
fine-tuning последних слоев.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EfficientNet training history.")
    parser.add_argument("--history-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--training-strategy", choices=("head_only", "fine_tune"), default="fine_tune")
    return parser.parse_args()


def build_epoch_labels(total_epochs: int, epochs_head: int) -> list[str]:
    labels: list[str] = []
    for epoch_index in range(total_epochs):
        if epoch_index < epochs_head:
            labels.append(f"head-{epoch_index + 1}")
        else:
            labels.append(f"finetune-{epoch_index - epochs_head + 1}")
    return labels


def plot_training_history(
    history: dict[str, list[float]],
    output_path: Path,
    epochs_head: int,
    training_strategy: str,
) -> None:
    """Сохранить графики loss и accuracy в стиле примера."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_epochs = len(history["loss"])
    x_values = np.arange(total_epochs)
    epoch_labels = build_epoch_labels(total_epochs, epochs_head)
    has_finetune_stage = training_strategy == "fine_tune" and total_epochs > epochs_head

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    axes[0].plot(x_values, history["loss"], marker="o", label="Train loss")
    axes[0].plot(x_values, history["val_loss"], marker="o", label="Val loss")
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Global epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_values, history["accuracy"], marker="o", label="Train accuracy")
    axes[1].plot(x_values, history["val_accuracy"], marker="o", label="Val accuracy")
    axes[1].set_title("Accuracy by Epoch")
    axes[1].set_xlabel("Global epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for axis in axes:
        axis.set_xticks(x_values)
        axis.set_xticklabels(epoch_labels, rotation=45, ha="right")
        if has_finetune_stage:
            axis.axvline(epochs_head - 0.5, linestyle="--", color="gray", alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_history(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "history" in data:
        return data["history"]
    return data


def main() -> None:
    args = parse_args()
    history = load_history(args.history_json)
    plot_training_history(
        history=history,
        output_path=args.output,
        epochs_head=args.epochs_head,
        training_strategy=args.training_strategy,
    )


if __name__ == "__main__":
    main()
