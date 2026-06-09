"""Create a balanced random subset from a class-folder image dataset."""

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
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is not empty. Use --overwrite to recreate the subset."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_selected_images(selected: dict[str, list[Path]], source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []

    for class_name, image_paths in selected.items():
        target_class_dir = output_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for source_path in image_paths:
            relative_source = source_path.relative_to(source_dir)
            target_path = target_class_dir / source_path.name
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
    if args.images_per_class <= 0:
        raise ValueError("--images-per-class must be greater than zero.")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    rng = random.Random(args.seed)
    selected: dict[str, list[Path]] = {}
    available_counts: dict[str, int] = {}

    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        image_paths = list_images(class_dir)
        available_counts[class_dir.name] = len(image_paths)

        if len(image_paths) < args.images_per_class:
            raise ValueError(
                f"Class {class_dir.name} has only {len(image_paths)} images, "
                f"but {args.images_per_class} were requested."
            )

        selected[class_dir.name] = sorted(rng.sample(image_paths, args.images_per_class))

    if not selected:
        raise ValueError(f"No class directories found in {source_dir}")

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

    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
