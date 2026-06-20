from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    """Описать параметры командной строки для создания датасетов разных размеров."""

    parser = argparse.ArgumentParser(
        description="Create nested pretrained dataset variants with fixed train/val/test ratios."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("pretrained_diploma/data/balanced_1000_per_class_split"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pretrained_diploma/data/dataset_size_variants"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("pretrained_diploma/results/dataset_splits/dataset_size_variants"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 600, 800])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    """Вернуть изображения класса в стабильном порядке."""

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_dir(path: Path, overwrite: bool) -> None:
    """Создать папку или пересоздать её при явном флаге overwrite."""

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty. Use --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_counts_for_size(images_per_class: int) -> dict[str, int]:
    """Посчитать число изображений train/val/test для заданного общего размера класса."""

    train_count = int(round(images_per_class * SPLIT_RATIOS["train"]))
    val_count = int(round(images_per_class * SPLIT_RATIOS["val"]))
    test_count = images_per_class - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def copy_images(source_paths: list[Path], target_dir: Path) -> list[Path]:
    """Скопировать выбранные изображения в целевую папку и вернуть новые пути."""

    target_dir.mkdir(parents=True, exist_ok=True)
    target_paths: list[Path] = []
    for source_path in source_paths:
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        target_paths.append(target_path)
    return target_paths


def add_manifest_rows(
    rows: list[dict[str, object]],
    dataset_name: str,
    images_per_class: int,
    split_name: str,
    class_name: str,
    source_paths: list[Path],
    target_paths: list[Path],
) -> None:
    """Добавить строки manifest для одного split и класса."""

    for source_path, target_path in zip(source_paths, target_paths):
        rows.append(
            {
                "dataset": dataset_name,
                "images_per_class": images_per_class,
                "split": split_name,
                "class_name": class_name,
                "source_path": str(source_path),
                "target_path": str(target_path),
            }
        )


def main() -> None:
    """Создать датасеты 100/200/400/600/800 изображений на класс."""

    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    report_dir = args.report_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source split directory not found: {source_dir}")
    if any(size <= 0 for size in args.sizes):
        raise ValueError("All dataset sizes must be greater than zero.")

    prepare_dir(output_root, args.overwrite)
    prepare_dir(report_dir, args.overwrite)

    rng = random.Random(args.seed)
    class_names = sorted(path.name for path in (source_dir / "train").iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class directories found in {source_dir / 'train'}")

    selected_by_split: dict[str, dict[str, list[Path]]] = {}
    available_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for split_name in ("train", "val", "test"):
        selected_by_split[split_name] = {}
        for class_name in class_names:
            image_paths = list_images(source_dir / split_name / class_name)
            shuffled_paths = image_paths[:]
            rng.shuffle(shuffled_paths)
            selected_by_split[split_name][class_name] = shuffled_paths
            available_counts[split_name][class_name] = len(image_paths)

    manifest_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "report_dir": str(report_dir),
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "available_counts": available_counts,
        "datasets": {},
    }

    for images_per_class in sorted(args.sizes):
        split_counts = split_counts_for_size(images_per_class)
        dataset_name = f"balanced_{images_per_class}_per_class_split"
        dataset_dir = output_root / dataset_name
        dataset_split_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

        for split_name, split_count in split_counts.items():
            for class_name in class_names:
                available = selected_by_split[split_name][class_name]
                if len(available) < split_count:
                    raise ValueError(
                        f"Class {class_name} split {split_name} has {len(available)} images, "
                        f"but {split_count} are required for dataset size {images_per_class}."
                    )
                source_paths = sorted(available[:split_count])
                target_paths = copy_images(source_paths, dataset_dir / split_name / class_name)
                add_manifest_rows(
                    manifest_rows,
                    dataset_name,
                    images_per_class,
                    split_name,
                    class_name,
                    source_paths,
                    target_paths,
                )
                dataset_split_counts[split_name][class_name] = len(source_paths)

        summary["datasets"][dataset_name] = {
            "dataset_dir": str(dataset_dir),
            "images_per_class": images_per_class,
            "train_per_class": split_counts["train"],
            "val_per_class": split_counts["val"],
            "test_per_class": split_counts["test"],
            "train_size": split_counts["train"] * len(class_names),
            "val_size": split_counts["val"] * len(class_names),
            "test_size": split_counts["test"] * len(class_names),
            "total_size": images_per_class * len(class_names),
            "split_counts": dataset_split_counts,
        }

    manifest_path = report_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "images_per_class",
                "split",
                "class_name",
                "source_path",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
