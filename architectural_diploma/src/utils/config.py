from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл в словарь Python."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_config(
    experiment_id: str,
    base_config_path: Path,
    experiments_path: Path,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Объединить общий конфиг и конфиг конкретного архитектурного эксперимента.

    Если передано `max_epochs`, создается отдельный run-id вида
    `exp00_baseline_ep020`. Это позволяет хранить запуски одной архитектуры для
    разных чисел эпох в разных папках.
    """

    base_config = load_json(base_config_path)
    experiments = load_json(experiments_path)
    if experiment_id not in experiments:
        known = ", ".join(sorted(experiments))
        raise KeyError(f"Unknown experiment '{experiment_id}'. Available: {known}")

    experiment = experiments[experiment_id]
    config = {**base_config, **experiment}
    # Эпохи могут прийти из CLI или из общего поля max_epochs.
    resolved_epochs = int(max_epochs if max_epochs is not None else config["max_epochs"])
    config["base_experiment_id"] = experiment_id
    config["experiment_id"] = f"{experiment_id}_ep{resolved_epochs:03d}"
    config["experiment_name"] = experiment_id
    config["experiment_series"] = experiment["series"]
    config["max_epochs"] = resolved_epochs
    config["model_name"] = f"modified_efficientnet_{experiment_id}_ep{resolved_epochs:03d}"
    return config


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Сохранить словарь в JSON с UTF-8 и читаемым форматированием."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
