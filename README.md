# EfficientNet Diplom Project

Дипломная работа: "Оптимизация использования сверточной нейронной сети EfficientNet".

## Тема

Классификация заболеваний листьев картофеля с использованием сверточной 
нейронной сети EfficientNet.

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

Датасет не хранится в репозитории. Его нужно скачать отдельно и 
поместить в папку `data/`.

## Предварительная обработка данных

Перед обучением модели необходимо подготовить изображения: проверить 
структуру папок, удалить поврежденные файлы, найти дубликаты и 
очень похожие изображения. Это важно, потому что одинаковые или 
почти одинаковые фотографии могут попасть одновременно в 
обучающую и тестовую выборки. В таком случае качество модели 
будет казаться выше, чем оно есть на самом деле.

Рекомендуемая исходная структура датасета:

```text
data/raw/
├── Healthy/
├── Early_Blight/
└── Late_Blight/
```

В распакованном архиве Irish Potato Imagery Dataset папки могут называться 
иначе:

```text
data/raw/
├── HEALTHY/
├── EARLY LEAF SPOT/
└── LATE LEAF SPOT/
```

Для подготовки данных в проекте есть скрипт:

```bash
python3 src/preprocess_data.py
```

Он сопоставляет реальные названия папок с классами проекта:

- `HEALTHY` → `Healthy`
- `EARLY LEAF SPOT` → `Early_Blight`
- `LATE LEAF SPOT` → `Late_Blight`

Скрипт проверяет изображения, исключает точные дубликаты из обработанной 
копии датасета, ищет похожие изображения и сохраняет отчеты в 
`results/preprocessing/`. На этом этапе данные не делятся на `train`, `val` 
и `test`, потому что в дальнейших экспериментах может использоваться разное 
количество изображений.

Если нужно пересоздать уже заполненную папку `data/processed`, используется 
команда:

```bash
python3 src/preprocess_data.py --overwrite
```

Похожие изображения скрипт только записывает в отчет 
`results/preprocessing/similar_pairs.csv`, потому что такие случаи лучше 
проверять вручную. Для каждой пары в отчете сохраняются расстояния по 
`dHash` и `pHash`, а также поле `matched_by`, где указано, каким методом 
пара была найдена.

Если нужно выполнить только проверку изображений и поиск точных дубликатов, 
без поиска похожих изображений, можно запустить:

```bash
python3 src/preprocess_data.py --overwrite --skip-similar
```

Поиск похожих изображений можно запускать раздельно:

```bash
python3 src/preprocess_data.py --overwrite --similar-method dhash
python3 src/preprocess_data.py --overwrite --similar-method phash
python3 src/preprocess_data.py --overwrite --similar-method both
```

Пороги задаются отдельно:

```bash
python3 src/preprocess_data.py --overwrite --similar-method dhash --dhash-threshold 5
python3 src/preprocess_data.py --overwrite --similar-method phash --phash-threshold 5
python3 src/preprocess_data.py --overwrite --similar-method both --dhash-threshold 5 --phash-threshold 5
```

Если нужно не только найти похожие изображения, но и исключить их из 
`data/processed`, используется флаг `--remove-similar`:

```bash
# Исключить из data/processed изображения, похожие по dHash
python3 src/preprocess_data.py --overwrite --similar-method dhash --dhash-threshold 5 --remove-similar

# Исключить из data/processed изображения, похожие по pHash
python3 src/preprocess_data.py --overwrite --similar-method phash --phash-threshold 5 --remove-similar

# Исключить из data/processed изображения, похожие по dHash или pHash
python3 src/preprocess_data.py --overwrite --similar-method both --dhash-threshold 5 --phash-threshold 5 --remove-similar
```

Эта команда не удаляет файлы из `data/raw`. Она пересоздает `data/processed` 
и не копирует туда изображения, которые попали в отчет похожих пар. Список 
исключенных файлов сохраняется в `results/preprocessing/removed_similar.csv`.

Если нужно искать только визуальные дубликаты с полностью одинаковым 
perceptual hash, можно поставить порог `0`:

```bash
python3 src/preprocess_data.py --overwrite --similar-method dhash --dhash-threshold 0
python3 src/preprocess_data.py --overwrite --similar-method phash --phash-threshold 0
python3 src/preprocess_data.py --overwrite --similar-method both --similar-threshold 0
```

После очистки данные можно сохранить в отдельную папку:

```text
data/processed/
├── Healthy/
├── Early_Blight/
└── Late_Blight/
```

## Отдельная выборка для первых экспериментов

Очищенный датасет `data/processed` используется как основной подготовленный
набор данных и не изменяется при создании экспериментальных выборок.

Для первых экспериментов EfficientNet создана отдельная сбалансированная
выборка:

```text
data/experiments/balanced_1000_per_class/
├── Early_Blight/  # 1000 изображений
├── Healthy/       # 1000 изображений
└── Late_Blight/   # 1000 изображений
```

Изображения выбираются случайно из `data/processed`, но с фиксированным
`seed = 42`, поэтому выборку можно воспроизвести. Сводка и список выбранных
файлов сохраняются в:

```text
results/dataset_subsets/balanced_1000_per_class/
├── manifest.csv
└── summary.json
```

Команда для воспроизведения этой выборки:

```bash
python src/create_balanced_subset.py --source-dir data/processed --output-dir data/experiments/balanced_1000_per_class --report-dir results/dataset_subsets/balanced_1000_per_class --images-per-class 1000 --seed 42
```

Если нужно пересоздать уже существующую выборку, используется флаг
`--overwrite`:

```bash
python src/create_balanced_subset.py --source-dir data/processed --output-dir data/experiments/balanced_1000_per_class --report-dir results/dataset_subsets/balanced_1000_per_class --images-per-class 1000 --seed 42 --overwrite
```

## Разбиение выборки для экспериментов

Для первых экспериментов выборка `balanced_1000_per_class` разделена на
`train`, `val` и `test` в соотношении `70/15/15` с тем же `seed = 42`.
В каждом классе получается:

```text
train: 700 изображений
val:   150 изображений
test:  150 изображений
```

Разбиение сохраняется отдельно и не изменяет исходную выборку:

```text
data/experiments/balanced_1000_per_class_split/
├── train/
├── val/
└── test/
```

Команда для воспроизведения разбиения:

```bash
python src/split_dataset.py --source-dir data/experiments/balanced_1000_per_class --output-dir data/experiments/balanced_1000_per_class_split --report-dir results/dataset_splits/balanced_1000_per_class --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Если нужно пересоздать уже существующее разбиение, используется флаг
`--overwrite`.

## Эксперимент 1: EfficientNetB0, обучение головы

Первый эксперимент соответствует режиму `head_only`: базовая часть
EfficientNetB0 замораживается, обучается только новая классификационная
голова для трех классов.

Команда запуска:

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/balanced_1000_per_class_split --experiment-name exp01_head_only_1000_per_class --training-strategy head_only --image-size 224 --batch-size 32 --epochs-head 10 --learning-rate-head 0.001 --weights imagenet --seed 42
```

Скрипт сохраняет:

- метрики эксперимента в `results/metrics/`;
- общий csv с экспериментами в `results/metrics/experiments.csv`;
- график accuracy/loss в `results/plots/`;
- confusion matrix в `results/confusion_matrices/`;
- обученную модель в `models/final/`;
- время обучения в json/csv с метриками.

При первом запуске с `--weights imagenet` TensorFlow может скачать
предобученные веса EfficientNetB0, если их еще нет в локальном кэше.

## Эксперимент 2: EfficientNetB0, fine-tuning последних слоев

Второй эксперимент соответствует режиму `fine_tune`. Сначала выполняется тот
же этап, что и в первом эксперименте: обучается новая классификационная голова
при замороженной EfficientNetB0. Затем размораживаются последние слои базовой
модели и выполняется дообучение с малым learning rate.

Условия остаются такими же, как в первом эксперименте:

- тот же датасет `balanced_1000_per_class_split`;
- то же разбиение `train/val/test`;
- тот же `seed = 42`;
- тот же размер изображения `224`;
- тот же `batch_size = 32`.

Команда запуска:

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/balanced_1000_per_class_split --experiment-name exp02_fine_tune_1000_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

Этот эксперимент нужен, чтобы проверить, дает ли дообучение последних слоев
EfficientNetB0 прирост качества по сравнению с обучением только
классификационной головы.

## Эксперимент 3: EfficientNetB0, fine-tuning с аугментацией

Третий эксперимент проверяет влияние аугментации данных. Используется тот же
режим обучения, что во втором эксперименте (`fine_tune`), но дополнительно
включаются случайные преобразования изображений во время обучения:

- горизонтальное отражение;
- небольшой поворот;
- масштабирование;
- небольшой сдвиг;
- изменение контрастности.

Validation и test изображения не аугментируются. Это важно, потому что модель
должна оцениваться на реальных неизмененных изображениях.

Команда запуска:

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/balanced_1000_per_class_split --experiment-name exp03_fine_tune_aug_1000_per_class --training-strategy fine_tune --augmentation --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

Этот эксперимент нужен, чтобы проверить, помогает ли аугментация повысить
устойчивость модели к разным условиям съемки и уменьшить путаницу между
классами болезней.

### 1. Проверка изображений

Сначала нужно убедиться, что все файлы действительно открываются как 
изображения. Поврежденные файлы лучше удалить или перенести в отдельную 
папку для ручной проверки.

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

Точные дубликаты — это полностью одинаковые файлы. Для их поиска можно 
использовать MD5-хеш. Если у нескольких файлов одинаковый хеш, значит их 
содержимое совпадает.

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

Удалять дубликаты автоматически лучше осторожно. Обычно оставляют первый 
файл из группы, а остальные переносят в отдельную папку, 
например `data/duplicates/`, чтобы можно было проверить результат вручную.

### 3. Поиск похожих изображений

Похожие изображения могут отличаться размером, яркостью, сжатием или 
небольшим поворотом, но по смыслу быть почти одинаковыми. Для их поиска 
можно использовать perceptual hash. В скрипте `src/preprocess_data.py` 
используются два варианта:

- `dHash` — сравнивает яркость соседних пикселей;
- `pHash` — сравнивает низкочастотные признаки изображения после 
дискретного косинусного преобразования.

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

Чем меньше расстояние между perceptual hash, тем более похожи изображения. 
Значение `threshold = 5` можно использовать как начальный вариант. Если 
находится слишком много разных изображений, порог нужно уменьшить. Если 
похожие изображения не находятся, порог можно немного увеличить.

### 4. Что удалять, а что оставлять

При очистке датасета рекомендуется придерживаться следующих правил:

- точные дубликаты можно удалять, оставляя один экземпляр изображения;
- похожие изображения лучше сначала просматривать вручную;
- если два изображения почти одинаковые, оставляется более четкое и информативное;
- если похожие изображения относятся к разным классам, их нельзя 
удалять автоматически — такие случаи нужно проверить отдельно;
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

Важно выполнять разбиение после удаления дубликатов и похожих изображений. 
Это уменьшает риск утечки данных, когда почти одинаковые фотографии 
оказываются в разных выборках.

### 6. Подготовка изображений для EfficientNet

На этапе обучения изображения приводятся к единому размеру, 
соответствующему выбранной версии EfficientNet. Также выполняется нормализация 
значений пикселей. Дополнительно могут использоваться аугментации: случайные 
повороты, горизонтальное отражение, изменение яркости и контрастности, 
небольшое масштабирование и сдвиги.

Эти действия помогают модели лучше работать с изображениями, снятыми в 
разных условиях: при разном освещении, фоне, расстоянии до листа и 
угле съемки.
