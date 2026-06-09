"""Create dataset splits for the second experiment series.

The second series studies how the amount of training data affects EfficientNetB0.
Only the train/val size changes between experiments. Test images are selected
once and then reused in every split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POPCOUNT_BY_BYTE = [bin(value).count("1") for value in range(256)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pHash-diverse fixed test and size-series train/val splits."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/size_series"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/dataset_splits/size_series"),
    )
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000, 2500, 3000],
        help="Train+val images per class, excluding the fixed test set.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output and report directories before creating the splits.",
    )
    return parser.parse_args()


def build_dct_matrix(size: int) -> np.ndarray:
    """Build a DCT matrix used by pHash."""

    matrix = np.zeros((size, size), dtype=np.float64)
    factor = np.pi / (2 * size)

    for row in range(size):
        scale = np.sqrt(1 / size) if row == 0 else np.sqrt(2 / size)
        for col in range(size):
            matrix[row, col] = scale * np.cos((2 * col + 1) * row * factor)

    return matrix


DCT_MATRIX_32 = build_dct_matrix(32)


def calculate_phash(image_path: Path) -> int:
    """Calculate a 64-bit perceptual hash for visual-diversity sampling."""

    with Image.open(image_path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L").resize((32, 32))
        pixels = np.asarray(grayscale, dtype=np.float64)
        dct = np.einsum("ij,jk,lk->il", DCT_MATRIX_32, pixels, DCT_MATRIX_32, optimize=True)
        low_frequencies = dct[:8, :8]
        coefficients = low_frequencies.flatten()[1:]
        median = np.median(coefficients)

    hash_value = 0
    for coefficient in coefficients:
        hash_value = (hash_value << 1) | int(coefficient > median)

    return hash_value


def hamming_distance(left_hash: int, right_hash: int) -> int:
    """Return Hamming distance between two 64-bit hashes."""

    difference = left_hash ^ right_hash
    distance = 0
    while difference:
        distance += POPCOUNT_BY_BYTE[difference & 0xFF]
        difference >>= 8
    return distance


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


def select_diverse_test(
    image_paths: list[Path],
    hashes: dict[Path, int],
    test_per_class: int,
    rng: random.Random,
) -> list[Path]:
    """Select visually diverse images with farthest-first sampling.

    The first image is random but reproducible. Every next image maximizes its
    minimal pHash distance to already selected test images.
    """

    if len(image_paths) < test_per_class:
        raise ValueError(
            f"Cannot select {test_per_class} test images from only {len(image_paths)} images."
        )

    remaining = image_paths[:]
    first_index = rng.randrange(len(remaining))
    selected = [remaining.pop(first_index)]

    min_distances = {
        image_path: hamming_distance(hashes[image_path], hashes[selected[0]])
        for image_path in remaining
    }

    while len(selected) < test_per_class:
        next_image = max(
            remaining,
            key=lambda image_path: (min_distances[image_path], str(image_path)),
        )
        remaining.remove(next_image)
        selected.append(next_image)

        next_hash = hashes[next_image]
        for image_path in remaining:
            distance = hamming_distance(hashes[image_path], next_hash)
            if distance < min_distances[image_path]:
                min_distances[image_path] = distance

    return sorted(selected)


def copy_images(image_paths: list[Path], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []

    for source_path in image_paths:
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)

    return copied_paths


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "class_name",
        "source_path",
        "target_path",
        "sample_size_per_class",
        "selection_method",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_manifest_rows(
    rows: list[dict[str, object]],
    dataset_name: str,
    split_name: str,
    class_name: str,
    source_paths: list[Path],
    target_paths: list[Path],
    sample_size_per_class: int | None,
    selection_method: str,
) -> None:
    for source_path, target_path in zip(source_paths, target_paths):
        rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "class_name": class_name,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "sample_size_per_class": sample_size_per_class,
                "selection_method": selection_method,
            }
        )


def main() -> None:
    args = parse_args()
    if args.test_per_class <= 0:
        raise ValueError("--test-per-class must be greater than zero.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if any(size <= 0 for size in args.sample_sizes):
        raise ValueError("All sample sizes must be greater than zero.")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    prepare_dir(output_dir, args.overwrite)
    prepare_dir(report_dir, args.overwrite)

    rng = random.Random(args.seed)
    class_images = {
        class_dir.name: list_images(class_dir)
        for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir())
    }
    if not class_images:
        raise ValueError(f"No class directories found in {source_dir}")

    max_sample_size = max(args.sample_sizes)
    available_counts = {class_name: len(paths) for class_name, paths in class_images.items()}
    for class_name, image_paths in class_images.items():
        required = args.test_per_class + max_sample_size
        if len(image_paths) < required:
            raise ValueError(
                f"Class {class_name} has {len(image_paths)} images, "
                f"but {required} are required for test + max sample size."
            )

    print("Calculating pHash values...")
    hashes = {
        image_path: calculate_phash(image_path)
        for image_paths in class_images.values()
        for image_path in image_paths
    }

    fixed_test: dict[str, list[Path]] = {}
    train_val_pool: dict[str, list[Path]] = {}
    manifest_rows: list[dict[str, object]] = []

    print("Selecting fixed pHash-diverse test images...")
    for class_name, image_paths in class_images.items():
        test_images = select_diverse_test(
            image_paths=image_paths,
            hashes=hashes,
            test_per_class=args.test_per_class,
            rng=rng,
        )
        fixed_test[class_name] = test_images
        test_set = set(test_images)
        train_val_pool[class_name] = [path for path in image_paths if path not in test_set]

    fixed_test_dir = output_dir / "fixed_test"
    for class_name, test_images in fixed_test.items():
        target_paths = copy_images(test_images, fixed_test_dir / class_name)
        add_manifest_rows(
            manifest_rows,
            dataset_name="fixed_test",
            split_name="test",
            class_name=class_name,
            source_paths=test_images,
            target_paths=target_paths,
            sample_size_per_class=None,
            selection_method="phash_farthest_first",
        )

    split_summaries: dict[str, dict[str, object]] = {}

    for sample_size in sorted(args.sample_sizes):
        dataset_name = f"size_{sample_size}_per_class"
        dataset_dir = output_dir / dataset_name
        train_per_class = int(round(sample_size * args.train_ratio))
        val_per_class = sample_size - train_per_class

        split_counts = {"train": {}, "val": {}, "test": {}}

        for class_name, pool_images in train_val_pool.items():
            shuffled_pool = pool_images[:]
            rng.shuffle(shuffled_pool)
            selected_train_val = sorted(shuffled_pool[:sample_size])
            train_images = selected_train_val[:train_per_class]
            val_images = selected_train_val[train_per_class:]

            train_targets = copy_images(train_images, dataset_dir / "train" / class_name)
            val_targets = copy_images(val_images, dataset_dir / "val" / class_name)
            test_targets = copy_images(fixed_test[class_name], dataset_dir / "test" / class_name)

            add_manifest_rows(
                manifest_rows,
                dataset_name,
                "train",
                class_name,
                train_images,
                train_targets,
                sample_size,
                "random_from_non_test_pool",
            )
            add_manifest_rows(
                manifest_rows,
                dataset_name,
                "val",
                class_name,
                val_images,
                val_targets,
                sample_size,
                "random_from_non_test_pool",
            )
            add_manifest_rows(
                manifest_rows,
                dataset_name,
                "test",
                class_name,
                fixed_test[class_name],
                test_targets,
                sample_size,
                "fixed_phash_farthest_first",
            )

            split_counts["train"][class_name] = len(train_images)
            split_counts["val"][class_name] = len(val_images)
            split_counts["test"][class_name] = len(fixed_test[class_name])

        split_summaries[dataset_name] = {
            "dataset_dir": str(dataset_dir),
            "sample_size_per_class": sample_size,
            "train_per_class": train_per_class,
            "val_per_class": val_per_class,
            "test_per_class": args.test_per_class,
            "train_size": train_per_class * len(class_images),
            "val_size": val_per_class * len(class_images),
            "test_size": args.test_per_class * len(class_images),
            "total_train_val_size": sample_size * len(class_images),
            "split_counts": split_counts,
        }

    write_manifest(report_dir / "manifest.csv", manifest_rows)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "seed": args.seed,
        "test_selection_method": "phash_farthest_first",
        "train_val_selection_method": "random_from_non_test_pool",
        "test_per_class": args.test_per_class,
        "sample_sizes": sorted(args.sample_sizes),
        "train_ratio": args.train_ratio,
        "val_ratio": 1 - args.train_ratio,
        "available_counts": available_counts,
        "fixed_test_counts": {
            class_name: len(paths) for class_name, paths in fixed_test.items()
        },
        "splits": split_summaries,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
