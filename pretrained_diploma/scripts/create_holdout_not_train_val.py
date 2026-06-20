from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_VAL_SPLITS = {"train", "val", "validation"}


def parse_args() -> argparse.Namespace:
    """Описать параметры создания независимой holdout-выборки."""

    parser = argparse.ArgumentParser(
        description="Create holdout images that were never used in train or validation manifests."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained_diploma/data/holdout_not_train_val_300_per_class"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("pretrained_diploma/results/dataset_splits/holdout_not_train_val_300_per_class"),
    )
    parser.add_argument("--manifests-root", type=Path, default=Path("."))
    parser.add_argument("--images-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    """Вернуть все изображения класса в стабильном порядке."""

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_dir(path: Path, overwrite: bool) -> None:
    """Создать пустую папку или пересоздать её при флаге overwrite."""

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty. Use --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_identity(class_name: str | None, image_path: Path) -> tuple[str, str] | None:
    """Получить устойчивый идентификатор изображения: класс и имя файла."""

    if not class_name or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return class_name, image_path.name


def collect_train_val_identities(manifests_root: Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Собрать все изображения, которые встречались в train или validation manifest."""

    excluded: set[tuple[str, str]] = set()
    used_manifests: list[str] = []

    for manifest_path in sorted(manifests_root.rglob("manifest.csv")):
        with manifest_path.open(encoding="utf-8") as file:
            reader = csv.DictReader(file)
            manifest_used = False
            for row in reader:
                split_name = (row.get("split") or "").lower()
                if split_name not in TRAIN_VAL_SPLITS:
                    continue

                class_name = row.get("class_name")
                for field_name in ("source_path", "target_path", "split_path", "subset_path", "processed_path"):
                    field_value = row.get(field_name)
                    if not field_value:
                        continue
                    identity = image_identity(class_name, Path(field_value))
                    if identity is not None:
                        excluded.add(identity)
                        manifest_used = True

            if manifest_used:
                used_manifests.append(str(manifest_path))

    return excluded, used_manifests


def copy_holdout_images(
    selected: dict[str, list[Path]],
    output_dir: Path,
) -> list[dict[str, str]]:
    """Скопировать выбранные изображения в папку test/<class> и вернуть строки manifest."""

    rows: list[dict[str, str]] = []
    for class_name, image_paths in selected.items():
        target_class_dir = output_dir / "test" / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for source_path in image_paths:
            target_path = target_class_dir / source_path.name
            shutil.copy2(source_path, target_path)
            rows.append(
                {
                    "split": "test",
                    "class_name": class_name,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "selection_rule": "not_in_any_train_or_val_manifest",
                }
            )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Сохранить CSV manifest для новой holdout-выборки."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class_name", "source_path", "target_path", "selection_rule"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Создать holdout из изображений, не входивших ни в один train/validation split."""

    args = parse_args()
    if args.images_per_class <= 0:
        raise ValueError("--images-per-class must be greater than zero.")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    excluded, used_manifests = collect_train_val_identities(args.manifests_root.resolve())
    rng = random.Random(args.seed)

    class_candidates: dict[str, list[Path]] = {}
    available_counts: dict[str, int] = {}
    selected: dict[str, list[Path]] = {}

    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        candidates = [
            image_path
            for image_path in list_images(class_dir)
            if (class_dir.name, image_path.name) not in excluded
        ]
        available_counts[class_dir.name] = len(candidates)
        if len(candidates) < args.images_per_class:
            raise ValueError(
                f"Class {class_dir.name} has only {len(candidates)} eligible images, "
                f"but {args.images_per_class} were requested."
            )
        class_candidates[class_dir.name] = candidates
        selected[class_dir.name] = sorted(rng.sample(candidates, args.images_per_class))

    if not selected:
        raise ValueError(f"No class directories found in {source_dir}")

    prepare_dir(output_dir, args.overwrite)
    prepare_dir(report_dir, args.overwrite)

    manifest_rows = copy_holdout_images(selected, output_dir)
    write_manifest(report_dir / "manifest.csv", manifest_rows)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "images_per_class": args.images_per_class,
        "total_images": len(manifest_rows),
        "seed": args.seed,
        "selection_rule": "exclude every class+filename found in any train or validation manifest",
        "used_manifests": used_manifests,
        "excluded_train_val_identities": len(excluded),
        "eligible_counts": available_counts,
        "selected_counts": {class_name: len(paths) for class_name, paths in selected.items()},
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
