"""Создание сбалансированной случайной подвыборки.

Скрипт используется перед первой серией экспериментов. Он берет очищенный
датасет ``data/processed`` и создает отдельную копию с одинаковым количеством
изображений в каждом классе. Исходный датасет не изменяется.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки."""

    parser = argparse.ArgumentParser(
        description="Create a balanced random subset with the same number of images per class."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/balanced_1000_per_class"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/dataset_subsets/balanced_1000_per_class"),
    )
    parser.add_argument("--images-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output and report directories before creating the subset.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    """Найти все изображения внутри папки одного класса.

    Сортировка нужна для стабильного исходного порядка файлов. После этого
    случайный выбор с фиксированным seed будет воспроизводимым.
    """

    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Подготовить папку для результата или отчета.

    Если папка уже заполнена, скрипт останавливается, чтобы случайно не
    перезаписать предыдущую выборку. Пересоздание разрешено только явно через
    флаг ``--overwrite``.
    """

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is not empty. Use --overwrite to recreate the subset."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_selected_images(selected: dict[str, list[Path]], source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Скопировать выбранные изображения и подготовить строки manifest.

    Manifest связывает исходный путь изображения с путем в новой подвыборке.
    Это полезно для проверки состава данных и для описания эксперимента в
    дипломе.
    """

    manifest_rows: list[dict[str, str]] = []

    for class_name, image_paths in selected.items():
        target_class_dir = output_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for source_path in image_paths:
            relative_source = source_path.relative_to(source_dir)
            target_path = target_class_dir / source_path.name

            # copy2 сохраняет метаданные файла и не изменяет исходное
            # изображение в data/processed.
            shutil.copy2(source_path, target_path)
            manifest_rows.append(
                {
                    "class_name": class_name,
                    "source_path": str(source_path),
                    "subset_path": str(target_path),
                    "relative_source_path": str(relative_source),
                }
            )

    return manifest_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Сохранить CSV со списком всех изображений подвыборки."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "source_path",
                "subset_path",
                "relative_source_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    # Количество изображений на класс должно быть положительным: нулевая или
    # отрицательная выборка не имеет смысла для обучения.
    if args.images_per_class <= 0:
        raise ValueError("--images-per-class must be greater than zero.")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    # random.Random создает отдельный генератор случайных чисел. При одинаковом
    # seed он каждый раз выберет один и тот же набор файлов.
    rng = random.Random(args.seed)
    selected: dict[str, list[Path]] = {}
    available_counts: dict[str, int] = {}

    # Проходим по классам отдельно, чтобы в итоговой выборке было одинаковое
    # число изображений каждого класса.
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        image_paths = list_images(class_dir)
        available_counts[class_dir.name] = len(image_paths)

        # Если в каком-то классе изображений меньше, чем требуется, лучше
        # остановиться сразу, чем получить несбалансированную выборку.
        if len(image_paths) < args.images_per_class:
            raise ValueError(
                f"Class {class_dir.name} has only {len(image_paths)} images, "
                f"but {args.images_per_class} were requested."
            )

        # rng.sample выбирает без повторений. sorted после выбора делает порядок
        # копирования стабильным и удобным для чтения manifest.
        selected[class_dir.name] = sorted(rng.sample(image_paths, args.images_per_class))

    if not selected:
        raise ValueError(f"No class directories found in {source_dir}")

    # Подготовка выходных папок выполняется после всех проверок, чтобы не
    # удалять старые результаты при ошибочных параметрах запуска.
    prepare_output_dir(output_dir, args.overwrite)
    prepare_output_dir(report_dir, args.overwrite)

    manifest_rows = copy_selected_images(selected, source_dir, output_dir)
    write_manifest(report_dir / "manifest.csv", manifest_rows)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "images_per_class": args.images_per_class,
        "seed": args.seed,
        "available_counts": available_counts,
        "subset_counts": {class_name: len(paths) for class_name, paths in selected.items()},
        "total_images": len(manifest_rows),
    }

    # summary.json содержит компактную сводку: сколько было доступно
    # изображений, сколько выбрано и с каким seed.
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Печатаем summary в консоль, чтобы сразу после запуска видеть результат.
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
