"""Create a reproducible stratified train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a class-folder image dataset into train, val and test folders."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/experiments/balanced_1000_per_class"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/balanced_1000_per_class_split"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/dataset_splits/balanced_1000_per_class"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output and report directories before creating the split.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty. Use --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_class_images(
    image_paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    shuffled = image_paths[:]
    rng.shuffle(shuffled)

    train_count = int(round(len(shuffled) * train_ratio))
    val_count = int(round(len(shuffled) * val_ratio))
    test_count = len(shuffled) - train_count - val_count

    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("Each split must contain at least one image per class.")

    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count :]),
    }


def copy_split(
    split_by_class: dict[str, dict[str, list[Path]]],
    source_dir: Path,
    output_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for class_name, split_images in split_by_class.items():
        for split_name, image_paths in split_images.items():
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for source_path in image_paths:
                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                rows.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "source_path": str(source_path),
                        "split_path": str(target_path),
                        "relative_source_path": str(source_path.relative_to(source_dir)),
                    }
                )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "class_name",
                "source_path",
                "split_path",
                "relative_source_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_split(split_by_class: dict[str, dict[str, list[Path]]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for class_name, split_images in split_by_class.items():
        for split_name, image_paths in split_images.items():
            counts[split_name][class_name] = len(image_paths)
    return counts


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    rng = random.Random(args.seed)
    split_by_class: dict[str, dict[str, list[Path]]] = {}

    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        image_paths = list_images(class_dir)
        if not image_paths:
            raise ValueError(f"No images found in class directory: {class_dir}")
        split_by_class[class_dir.name] = split_class_images(
            image_paths,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )

    if not split_by_class:
        raise ValueError(f"No class directories found in {source_dir}")

    prepare_dir(output_dir, args.overwrite)
    prepare_dir(report_dir, args.overwrite)

    manifest_rows = copy_split(split_by_class, source_dir, output_dir)
    write_manifest(report_dir / "manifest.csv", manifest_rows)

    split_counts = count_split(split_by_class)
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "split_counts": split_counts,
        "total_images": len(manifest_rows),
    }

    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
