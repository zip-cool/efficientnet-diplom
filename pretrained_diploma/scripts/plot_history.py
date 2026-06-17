from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrained_diploma.src.utils.plotting import plot_history_from_csv


def parse_args() -> argparse.Namespace:
    """Описать параметры восстановления графиков."""

    parser = argparse.ArgumentParser(description="Rebuild plots from history.csv.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("pretrained_diploma/results/experiments"))
    return parser.parse_args()


def main() -> None:
    """Построить графики accuracy/loss без повторного обучения."""

    args = parse_args()
    experiment_dir = args.results_dir / args.experiment
    plot_history_from_csv(experiment_dir / "history.csv", experiment_dir / "plots")
    print(f"Plots rebuilt: {experiment_dir / 'plots'}")


if __name__ == "__main__":
    main()
