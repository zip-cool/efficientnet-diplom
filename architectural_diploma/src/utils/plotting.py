from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history_from_csv(history_csv: Path, plots_dir: Path) -> None:
    """Построить графики accuracy и loss из сохраненного `history.csv`.

    Эта функция специально не зависит от объекта Keras History, чтобы графики
    можно было пересоздавать после обучения только по табличным данным.
    """

    history = pd.read_csv(history_csv)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Один и тот же код строит пару графиков: accuracy и loss.
    for metric, ylabel, output_name in [
        ("accuracy", "Accuracy", "accuracy.png"),
        ("loss", "Loss", "loss.png"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history["epoch"], history[metric], marker="o", label=f"train {metric}")
        plt.plot(history["epoch"], history[f"val_{metric}"], marker="o", label=f"val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / output_name, dpi=180)
        plt.close()
