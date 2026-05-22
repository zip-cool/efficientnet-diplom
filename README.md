# EfficientNet Diplom Project

Дипломная работа: "Оптимизация использования сверточной нейронной сети EfficientNet".

## Тема

Классификация заболеваний листьев картофеля с использованием сверточной нейронной сети EfficientNet.

## Классы

- Healthy
- Early Blight
- Late Blight

## Структура проекта

- `src/` — основной Python-код
- `notebooks/` — эксперименты и анализ
- `configs/` — конфигурации обучения
- `reports/` — материалы для диплома
- `data/` — локальная папка с датасетом, не загружается в GitHub
- `models/` — сохраненные модели, не загружаются в GitHub
- `results/` — результаты экспериментов, графики и метрики

## Примечание

Датасет не хранится в репозитории. Его нужно скачать отдельно и поместить в папку `data/`.

## Предварительная обработка данных

Перед обучением модели необходимо подготовить изображения: проверить структуру папок, удалить поврежденные файлы, найти дубликаты и очень похожие изображения. Это важно, потому что одинаковые или почти одинаковые фотографии могут попасть одновременно в обучающую и тестовую выборки. В таком случае качество модели будет казаться выше, чем оно есть на самом деле.

Рекомендуемая исходная структура датасета:

```text
data/raw/
├── Healthy/
├── Early_Blight/
└── Late_Blight/
```

После очистки данные можно сохранить в отдельную папку:

```text
data/processed/
├── Healthy/
├── Early_Blight/
└── Late_Blight/
```

### 1. Проверка изображений

Сначала нужно убедиться, что все файлы действительно открываются как изображения. Поврежденные файлы лучше удалить или перенести в отдельную папку для ручной проверки.

```python
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

bad_files = []

for image_path in DATA_DIR.rglob("*"):
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        bad_files.append(image_path)

print(f"Поврежденных файлов: {len(bad_files)}")
for path in bad_files:
    print(path)
```

### 2. Поиск точных дубликатов

Точные дубликаты — это полностью одинаковые файлы. Для их поиска можно использовать MD5-хеш. Если у нескольких файлов одинаковый хеш, значит их содержимое совпадает.

```python
from pathlib import Path
import hashlib
from collections import defaultdict

DATA_DIR = Path("data/raw")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hashes = defaultdict(list)

for image_path in DATA_DIR.rglob("*"):
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
    hashes[file_hash].append(image_path)

duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}

print(f"Групп точных дубликатов: {len(duplicates)}")
for paths in duplicates.values():
    print("\nДубликаты:")
    for path in paths:
        print(path)
```

Удалять дубликаты автоматически лучше осторожно. Обычно оставляют первый файл из группы, а остальные переносят в отдельную папку, например `data/duplicates/`, чтобы можно было проверить результат вручную.

### 3. Поиск похожих изображений

Похожие изображения могут отличаться размером, яркостью, сжатием или небольшим поворотом, но по смыслу быть почти одинаковыми. Для их поиска можно использовать perceptual hash. В проекте для этого используется библиотека `imagehash`.

Пример поиска похожих изображений через `phash`:

```python
from pathlib import Path
from PIL import Image
import imagehash

DATA_DIR = Path("data/raw")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hashes = []

for image_path in DATA_DIR.rglob("*"):
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:
        with Image.open(image_path) as img:
            img_hash = imagehash.phash(img)
            hashes.append((image_path, img_hash))
    except Exception:
        continue

threshold = 5
similar_pairs = []

for i in range(len(hashes)):
    path_1, hash_1 = hashes[i]
    for j in range(i + 1, len(hashes)):
        path_2, hash_2 = hashes[j]
        distance = hash_1 - hash_2

        if distance <= threshold:
            similar_pairs.append((path_1, path_2, distance))

print(f"Найдено похожих пар: {len(similar_pairs)}")
for path_1, path_2, distance in similar_pairs:
    print(f"{path_1} <-> {path_2}, distance={distance}")
```

Чем меньше расстояние между perceptual hash, тем более похожи изображения. Значение `threshold = 5` можно использовать как начальный вариант. Если находится слишком много разных изображений, порог нужно уменьшить. Если похожие изображения не находятся, порог можно немного увеличить.

### 4. Что удалять, а что оставлять

При очистке датасета рекомендуется придерживаться следующих правил:

- точные дубликаты можно удалять, оставляя один экземпляр изображения;
- похожие изображения лучше сначала просматривать вручную;
- если два изображения почти одинаковые, оставляется более четкое и информативное;
- если похожие изображения относятся к разным классам, их нельзя удалять автоматически — такие случаи нужно проверить отдельно;
- после очистки нужно заново посчитать количество изображений в каждом классе.

### 5. Разделение на обучающую, валидационную и тестовую выборки

После удаления дубликатов датасет разделяется на три части:

- `train` — для обучения модели;
- `val` — для подбора параметров и контроля переобучения;
- `test` — для финальной проверки качества модели.

Пример структуры после разбиения:

```text
data/processed/
├── train/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
├── val/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
└── test/
    ├── Healthy/
    ├── Early_Blight/
    └── Late_Blight/
```

Важно выполнять разбиение после удаления дубликатов и похожих изображений. Это уменьшает риск утечки данных, когда почти одинаковые фотографии оказываются в разных выборках.

### 6. Подготовка изображений для EfficientNet

На этапе обучения изображения приводятся к единому размеру, соответствующему выбранной версии EfficientNet. Также выполняется нормализация значений пикселей. Дополнительно могут использоваться аугментации: случайные повороты, горизонтальное отражение, изменение яркости и контрастности, небольшое масштабирование и сдвиги.

Эти действия помогают модели лучше работать с изображениями, снятыми в разных условиях: при разном освещении, фоне, расстоянии до листа и угле съемки.
