from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history_from_csv(history_csv: Path, plots_dir: Path) -> None:
    """Построить графики accuracy и loss из сохраненного `history.csv`."""

    history = pd.read_csv(history_csv)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, train_label, test_label, output_name in [
        (
            "accuracy",
            "Точность",
            "Точность на обучающей выборке",
            "Точность на тестовой выборке",
            "accuracy.png",
        ),
        (
            "loss",
            "Функция потерь",
            "Функция потерь на обучающей выборке",
            "Функция потерь на тестовой выборке",
            "loss.png",
        ),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history["epoch"], history[metric], marker="o", label=train_label)
        plt.plot(history["epoch"], history[f"val_{metric}"], marker="o", label=test_label)
        plt.xlabel("Количество эпох")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / output_name, dpi=180)
        plt.close()
