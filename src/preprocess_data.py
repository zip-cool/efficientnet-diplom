"""Предварительная обработка датасета листьев картофеля.

Скрипт не изменяет ``data/raw``. Он проверяет изображения, исключает точные
дубликаты из обработанной копии, находит похожие изображения и сохраняет
отчеты CSV/JSON, которые можно использовать в дипломной работе.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POPCOUNT_BY_BYTE = [bin(value).count("1") for value in range(256)]

# Названия папок из распакованного архива -> названия классов в проекте.
CLASS_DIR_MAP = {
    "HEALTHY": "Healthy",
    "EARLY LEAF SPOT": "Early_Blight",
    "LATE LEAF SPOT": "Late_Blight",
}


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    source_class: str
    class_name: str
    md5: str


@dataclass(frozen=True)
class HashRecord:
    image: ImageRecord
    dhash: int
    phash: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверить изображения, убрать точные дубликаты и найти похожие."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/preprocessing"))
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Необязательный размер квадратного изображения, например 224.",
    )
    parser.add_argument(
        "--skip-similar",
        action="store_true",
        help="Не искать похожие изображения, а выполнить только проверку и дубликаты.",
    )
    parser.add_argument(
        "--similar-threshold",
        type=int,
        default=None,
        help="Общий порог для dHash и pHash. Оставлен для совместимости.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=5,
        help="Максимальное расстояние dHash для записи пары в отчет.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=8,
        help="Максимальное расстояние pHash для записи пары в отчет.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Удалить существующую папку processed перед новой обработкой.",
    )
    return parser.parse_args()


def iter_source_images(raw_dir: Path) -> tuple[list[tuple[Path, str, str]], list[str]]:
    images: list[tuple[Path, str, str]] = []
    warnings: list[str] = []

    if not raw_dir.exists():
        raise FileNotFoundError(f"Папка с исходным датасетом не найдена: {raw_dir}")

    for class_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        class_name = CLASS_DIR_MAP.get(class_dir.name)
        if class_name is None:
            warnings.append(f"Пропущена неизвестная папка: {class_dir}")
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((image_path, class_dir.name, class_name))

    return images, warnings


def file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_images(
    candidates: list[tuple[Path, str, str]]
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    valid_records: list[ImageRecord] = []
    bad_files: list[dict[str, str]] = []

    for image_path, source_class, class_name in candidates:
        try:
            with Image.open(image_path) as image:
                image.verify()
            valid_records.append(
                ImageRecord(
                    source_path=image_path,
                    source_class=source_class,
                    class_name=class_name,
                    md5=file_md5(image_path),
                )
            )
        except Exception as error:  # PIL может вернуть разные ошибки чтения файла.
            bad_files.append({"path": str(image_path), "error": str(error)})

    return valid_records, bad_files


def split_exact_duplicates(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.md5].append(record)

    unique_records: list[ImageRecord] = []
    duplicate_rows: list[dict[str, str]] = []

    for md5_hash, group in sorted(by_hash.items()):
        ordered_group = sorted(group, key=lambda item: str(item.source_path))
        keeper = ordered_group[0]
        unique_records.append(keeper)

        for duplicate in ordered_group[1:]:
            duplicate_rows.append(
                {
                    "md5": md5_hash,
                    "kept_path": str(keeper.source_path),
                    "duplicate_path": str(duplicate.source_path),
                    "kept_class": keeper.class_name,
                    "duplicate_class": duplicate.class_name,
                }
            )

    return sorted(unique_records, key=lambda item: str(item.source_path)), duplicate_rows


def copy_or_resize_image(source: Path, destination: Path, image_size: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if image_size is None:
        shutil.copy2(source, destination)
        return

    destination = destination.with_suffix(".jpg")
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=95, optimize=True)


def write_processed_dataset(
    records: list[ImageRecord],
    processed_dir: Path,
    image_size: int | None,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []

    for record in records:
        destination_name = record.source_path.name
        if image_size is not None:
            destination_name = f"{record.source_path.stem}.jpg"
        destination = processed_dir / record.class_name / destination_name
        copy_or_resize_image(record.source_path, destination, image_size)
        manifest.append(
            {
                "class_name": record.class_name,
                "source_class": record.source_class,
                "source_path": str(record.source_path),
                "processed_path": str(destination),
                "md5": record.md5,
            }
        )

    return manifest


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_by_class(records: list[ImageRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.class_name] += 1
    return dict(sorted(counts.items()))


def calculate_dhash(image: Image.Image) -> int:
    """Посчитать простой perceptual hash без внешних зависимостей.

    Difference hash сравнивает яркость соседних пикселей на уменьшенной
    черно-белой версии изображения. Он не заменяет ручную проверку, но хорошо
    подходит для первичного поиска почти одинаковых фотографий.
    """

    grayscale = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
    pixels = list(grayscale.getdata())

    hash_value = 0
    for row in range(8):
        row_start = row * 9
        for col in range(8):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            hash_value = (hash_value << 1) | int(left > right)

    return hash_value


def build_dct_matrix(size: int) -> np.ndarray:
    """Построить матрицу дискретного косинусного преобразования."""

    matrix = np.zeros((size, size), dtype=np.float64)
    factor = np.pi / (2 * size)

    for row in range(size):
        scale = np.sqrt(1 / size) if row == 0 else np.sqrt(2 / size)
        for col in range(size):
            matrix[row, col] = scale * np.cos((2 * col + 1) * row * factor)

    return matrix


DCT_MATRIX_32 = build_dct_matrix(32)


def calculate_phash(image: Image.Image) -> int:
    """Посчитать pHash через частотные признаки изображения.

    pHash уменьшает изображение, применяет дискретное косинусное
    преобразование и сравнивает низкочастотные коэффициенты. Такой хеш обычно
    устойчивее к изменению размера, сжатию и небольшим изменениям яркости.
    """

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
    """Посчитать расстояние Хэмминга между двумя 64-битными хешами."""

    difference = left_hash ^ right_hash
    distance = 0
    while difference:
        distance += POPCOUNT_BY_BYTE[difference & 0xFF]
        difference >>= 8
    return distance


def iter_chunk_variants(chunk_value: int, max_distance: int) -> list[int]:
    """Вернуть варианты блока хеша в пределах заданного битового расстояния."""

    variants = [chunk_value]
    bit_positions = range(16)

    for distance in range(1, max_distance + 1):
        for changed_bits in combinations(bit_positions, distance):
            variant = chunk_value
            for bit_index in changed_bits:
                variant ^= 1 << bit_index
            variants.append(variant)

    return variants


def add_hash_to_index(
    hash_index: dict[tuple[int, int], list[int]], hash_value: int, image_index: int
) -> None:
    """Добавить 64-битный хеш в индекс по четырем 16-битным блокам."""

    for chunk_number in range(4):
        shift = chunk_number * 16
        chunk_value = (hash_value >> shift) & 0xFFFF
        hash_index[(chunk_number, chunk_value)].append(image_index)


def collect_candidate_indexes(
    hash_index: dict[tuple[int, int], list[int]],
    hash_value: int,
    threshold: int,
) -> set[int]:
    """Найти кандидатов, которые могут попасть в заданное расстояние Хэмминга."""

    candidate_indexes: set[int] = set()
    chunk_distance = threshold // 4

    for chunk_number in range(4):
        shift = chunk_number * 16
        chunk_value = (hash_value >> shift) & 0xFFFF
        for variant in iter_chunk_variants(chunk_value, chunk_distance):
            candidate_indexes.update(hash_index.get((chunk_number, variant), []))

    return candidate_indexes


def find_similar_pairs(
    records: list[ImageRecord], dhash_threshold: int, phash_threshold: int
) -> list[dict[str, str | int]]:
    hashes: list[HashRecord] = []
    for record in records:
        try:
            with Image.open(record.source_path) as image:
                hashes.append(
                    HashRecord(
                        image=record,
                        dhash=calculate_dhash(image),
                        phash=calculate_phash(image),
                    )
                )
        except Exception:
            continue

    pairs: list[dict[str, str | int]] = []
    dhash_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    phash_index: dict[tuple[int, int], list[int]] = defaultdict(list)

    for current_index, current_hashes in enumerate(hashes):
        # Индексы по блокам хеша позволяют не сравнивать каждое изображение
        # с каждым. В кандидаты попадают только хеши, которые теоретически
        # могут пройти заданные пороги dHash или pHash.
        candidate_indexes = collect_candidate_indexes(
            dhash_index, current_hashes.dhash, dhash_threshold
        )
        candidate_indexes.update(
            collect_candidate_indexes(phash_index, current_hashes.phash, phash_threshold)
        )

        for candidate_index in sorted(candidate_indexes):
            candidate_hashes = hashes[candidate_index]
            dhash_distance = hamming_distance(
                current_hashes.dhash, candidate_hashes.dhash
            )
            phash_distance = hamming_distance(
                current_hashes.phash, candidate_hashes.phash
            )
            matched_by = []
            if dhash_distance <= dhash_threshold:
                matched_by.append("dhash")
            if phash_distance <= phash_threshold:
                matched_by.append("phash")

            if matched_by:
                pairs.append(
                    {
                        "path_1": str(candidate_hashes.image.source_path),
                        "class_1": candidate_hashes.image.class_name,
                        "path_2": str(current_hashes.image.source_path),
                        "class_2": current_hashes.image.class_name,
                        "dhash_distance": dhash_distance,
                        "phash_distance": phash_distance,
                        "matched_by": "+".join(matched_by),
                    }
                )

        add_hash_to_index(dhash_index, current_hashes.dhash, current_index)
        add_hash_to_index(phash_index, current_hashes.phash, current_index)

    return pairs


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    report_dir = args.report_dir.resolve()

    if processed_dir.exists() and any(processed_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{processed_dir} не пустая. Используйте --overwrite для пересоздания."
            )
        shutil.rmtree(processed_dir)

    # 1. Собираем пути к изображениям только из известных папок классов.
    candidates, warnings = iter_source_images(raw_dir)

    # 2. Проверяем, что файлы открываются как изображения, и сразу считаем MD5.
    valid_records, bad_files = validate_images(candidates)

    # 3. Оставляем по одному экземпляру каждого точного дубликата.
    unique_records, duplicate_rows = split_exact_duplicates(valid_records)

    # 4. Копируем очищенный датасет без разделения на train/val/test.
    manifest = write_processed_dataset(unique_records, processed_dir, args.image_size)

    write_csv(
        report_dir / "manifest.csv",
        manifest,
        ["class_name", "source_class", "source_path", "processed_path", "md5"],
    )
    write_csv(report_dir / "bad_files.csv", bad_files, ["path", "error"])
    write_csv(
        report_dir / "exact_duplicates.csv",
        duplicate_rows,
        ["md5", "kept_path", "duplicate_path", "kept_class", "duplicate_class"],
    )

    similar_count = None
    dhash_threshold = args.similar_threshold or args.dhash_threshold
    phash_threshold = args.similar_threshold or args.phash_threshold

    if not args.skip_similar:
        # 5. Похожие изображения не удаляем автоматически, а сохраняем в отчет.
        similar_pairs = find_similar_pairs(
            unique_records,
            dhash_threshold=dhash_threshold,
            phash_threshold=phash_threshold,
        )
        similar_count = len(similar_pairs)
        write_csv(
            report_dir / "similar_pairs.csv",
            similar_pairs,
            [
                "path_1",
                "class_1",
                "path_2",
                "class_2",
                "dhash_distance",
                "phash_distance",
                "matched_by",
            ],
        )

    summary = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "report_dir": str(report_dir),
        "class_dir_map": CLASS_DIR_MAP,
        "warnings": warnings,
        "total_candidates": len(candidates),
        "valid_images": len(valid_records),
        "bad_files": len(bad_files),
        "exact_duplicate_files_removed_from_processed": len(duplicate_rows),
        "unique_images": len(unique_records),
        "processed_counts": count_by_class(unique_records),
        "image_size": args.image_size,
        "dhash_threshold": None if args.skip_similar else dhash_threshold,
        "phash_threshold": None if args.skip_similar else phash_threshold,
        "similar_pairs": similar_count,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
