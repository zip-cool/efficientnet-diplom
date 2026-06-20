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

## Эксперимент 4: EfficientNetB0, на всякий случай (head_only с аугментацией)


```bash
python src/train_efficientnet.py --dataset-dir data/experiments/balanced_1000_per_class_split --experiment-name exp03_1_head_only_aug_1000_per_class --training-strategy head_only --augmentation --image-size 224 --batch-size 32 --epochs-head 10 --learning-rate-head 0.001 --weights imagenet --seed 42
```



## Вторая серия: влияние объема обучающей выборки

Во второй серии экспериментов меняется только объем данных. Режим обучения
фиксируется по результатам первой серии:

```text
training_strategy: fine_tune
augmentation: false
fine_tune_layers: 20
learning_rate_head: 0.001
learning_rate_finetune: 0.00001
epochs_head: 10
epochs_finetune: 5
batch_size: 32
seed: 42
```

Перед обучением создана фиксированная тестовая выборка с визуально
разнообразными изображениями:

```text
test_per_class: 500
test_size: 1500
selection_method: pHash + farthest-first sampling
```

Эта test-выборка одинаковая для всех экспериментов второй серии. Train и val
формируются из оставшихся изображений.

Команда подготовки splits второй серии:

```bash
python src/create_size_series_splits.py --source-dir data/processed --output-dir data/experiments/size_series --report-dir results/dataset_splits/size_series --test-per-class 500 --sample-sizes 500 1000 1500 2000 2500 3000 --train-ratio 0.8 --seed 42
```

Если нужно пересоздать папки второй серии, добавляется `--overwrite`.

Команды запуска экспериментов:

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/size_series/size_500_per_class --experiment-name exp04_fine_tune_size500_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/size_series/size_1000_per_class --experiment-name exp05_fine_tune_size1000_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/size_series/size_1500_per_class --experiment-name exp06_fine_tune_size1500_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/size_series/size_2000_per_class --experiment-name exp07_fine_tune_size2000_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/size_series/size_2500_per_class --experiment-name exp08_fine_tune_size2500_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

```bash
python src/train_efficientnet.py --dataset-dir data/experiments/size_series/size_3000_per_class --experiment-name exp09_fine_tune_size3000_per_class --training-strategy fine_tune --image-size 224 --batch-size 32 --epochs-head 10 --epochs-finetune 5 --learning-rate-head 0.001 --learning-rate-finetune 0.00001 --fine-tune-layers 20 --weights imagenet --seed 42
```

## Ресурсные метрики моделей

После обучения экспериментов можно дополнительно записать в JSON и общий CSV
ресурсные характеристики сохраненных моделей:

- размер файла модели;
- количество параметров;
- FLOPs на один forward pass;
- скорость инференса на текущей машине;
- оценку GFLOPs/s.

Команда:

```bash
python src/update_model_resource_metrics.py --metrics-dir results/metrics --experiments-csv results/metrics/experiments.csv --image-size 224 --benchmark-batch-size 1 --warmup-runs 3 --benchmark-runs 10
```

FLOPs на одно изображение показывает вычислительную сложность модели.
GFLOPs/s зависит от компьютера, версии TensorFlow, CPU/GPU и batch size, поэтому
эту метрику нужно указывать как измеренную на текущей машине.

## Новая часть диплома: архитектурные эксперименты EfficientNet

> Важно: этот раздел относится к ранней scratch-версии, где EfficientNet-подобная
> модель обучалась с нуля. После пересмотра плана актуальной является серия
> `pretrained_diploma/` ниже. Для новых запусков используйте команды из раздела
> "Новая актуальная серия: предобученная EfficientNetB0".

Для новой версии дипломной работы создана отдельная папка:

```text
architectural_diploma/
```

В ней находится новая система экспериментов, отделенная от старого pipeline.
Старые скрипты `src/train_efficientnet.py`, старые модели в `models/` и старые
результаты в `results/` не удалялись и не переносились.

Новая часть предназначена не для сравнения режимов transfer learning, а для
исследования упрощения самой архитектуры EfficientNetB0-подобной сети,
обучаемой с нуля.

### Датасет для новой части

Для всех новых архитектурных экспериментов используется фиксированное
разбиение:

```text
architectural_diploma/data/balanced_1500_per_class_split/
```

Оно создано из очищенного датасета `data/processed` с `seed = 42`.
В каждом классе используется по 1500 изображений:

- `Early_Blight` - 1500;
- `Healthy` - 1500;
- `Late_Blight` - 1500.

Разбиение `train / validation / test` выполнено в пропорции `70 / 15 / 15`.
В каждом классе получается:

- train - 1050 изображений;
- validation - 225 изображений;
- test - 225 изображений.

### Что добавлено

```text
architectural_diploma/
├── configs/
│   ├── base_config.json
│   └── experiments.json
├── docs/
│   └── architectural_experiments.md
├── scripts/
│   ├── run_experiment.py
│   ├── run_all_experiments.py
│   ├── plot_history.py
│   ├── rebuild_summary.py
│   ├── compare_experiments.py
│   └── build_epoch_tables.py
└── src/
    ├── model/
    │   ├── efficientnet_modified.py
    │   ├── mbconv.py
    │   └── scaling.py
    ├── training/
    │   ├── callbacks.py
    │   └── trainer.py
    └── utils/
        ├── config.py
        ├── plotting.py
        ├── reproducibility.py
        └── result_saver.py
```

### Новая модель

В `architectural_diploma/src/model/` реализована собственная параметризуемая
EfficientNetB0-подобная архитектура:

- MBConv-блоки;
- Squeeze-and-Excitation;
- residual connection при совпадении размеров и `stride = 1`;
- масштабирование ширины сети;
- масштабирование глубины сети;
- изменение `expansion_ratio`;
- включение и отключение SE;
- изменение depthwise-ядер;
- изменение числа каналов в head;
- изменение входного разрешения.

Базовая функция построения модели находится в:

```text
architectural_diploma/src/model/efficientnet_modified.py
```

### Конфигурации экспериментов

Общие параметры обучения находятся в:

```text
architectural_diploma/configs/base_config.json
```

Там задаются путь к датасету, путь для новых результатов, batch size, learning
rate, optimizer, loss, seed, dropout, callbacks и список чисел эпох.

Список архитектурных экспериментов находится в:

```text
architectural_diploma/configs/experiments.json
```

Добавлены эксперименты:

- `exp00_baseline`;
- `exp01_width_075`;
- `exp02_width_050`;
- `exp03_width_035`;
- `exp04_depth_080`;
- `exp05_depth_060`;
- `exp06_depth_040`;
- `exp07_expansion_4`;
- `exp08_expansion_3`;
- `exp09_expansion_2`;
- `exp10_se_0125`;
- `exp11_no_se`;
- `exp12_all_kernels_3`;
- `exp13_head_640`;
- `exp14_head_320`;
- `exp15_head_160`;
- `exp16_resolution_192`;
- `exp17_resolution_160`;
- `exp18_resolution_128`.

`exp15_head_160` считается дополнительным и по умолчанию не запускается в общем
запуске всех экспериментов.

### Эксперименты по числу эпох

Для каждого архитектурного варианта предусмотрены запуски на:

```text
5, 10, 20, 40, 60, 80, 100 эпох
```

Поэтому один базовый эксперимент получает отдельные идентификаторы запусков:

```text
exp00_baseline_ep005
exp00_baseline_ep010
exp00_baseline_ep020
exp00_baseline_ep040
exp00_baseline_ep060
exp00_baseline_ep080
exp00_baseline_ep100
```

Это нужно для того, чтобы потом строить таблицы и сравнивать качество модели не
только по архитектуре, но и по количеству эпох обучения.

### Запуск одного эксперимента

Запуск одного эксперимента на конкретное число эпох:

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

Запуск одного эксперимента сразу для всех чисел эпох:

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

Если папка результата уже существует, повторный запуск без флага `--overwrite`
остановится, чтобы случайно не перезаписать старые данные.

Принудительная перезапись:

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

### Запуск всех экспериментов

Запуск всех обязательных архитектурных экспериментов для всех чисел эпох:

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

Запуск вместе с дополнительным `exp15_head_160`:

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

### Что сохраняется для каждого запуска

Каждый запуск сохраняется в отдельную папку:

```text
architectural_diploma/results/experiments/<experiment_id>/
```

Например:

```text
architectural_diploma/results/experiments/exp00_baseline_ep020/
```

Внутри сохраняются:

- `config.json`;
- `model_structure.json`;
- `class_indices.json`;
- `model_summary.txt`;
- `history.csv`;
- `history.json`;
- `epoch_times.csv`;
- `training_log.csv`;
- `final_metrics.json`;
- `test_results.json`;
- `best_model.keras`;
- `final_model.keras`;
- графики в папке `plots/`.

Главная идея: графики можно восстановить из `history.csv` без повторного
обучения модели.

### Повторное построение графиков

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

Скрипт читает:

```text
architectural_diploma/results/experiments/exp00_baseline_ep020/history.csv
```

и заново строит графики accuracy и loss.

### Восстановление общей сводной таблицы

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

Скрипт заново собирает:

```text
architectural_diploma/results/tables/experiments_summary.csv
```

по папкам отдельных экспериментов.

### Таблицы по количеству эпох

Для построения таблиц по числу эпох используется:

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

Результаты сохраняются в:

```text
architectural_diploma/results/tables/by_epochs/
```

Создаются таблицы:

- `epochs_005.csv`;
- `epochs_010.csv`;
- `epochs_020.csv`;
- `epochs_040.csv`;
- `epochs_060.csv`;
- `epochs_080.csv`;
- `epochs_100.csv`.

Также создаются pivot-таблицы:

- `pivot_test_accuracy.csv`;
- `pivot_test_loss.csv`;
- `pivot_best_val_accuracy.csv`;
- `pivot_best_val_loss.csv`;
- `pivot_total_training_time_seconds.csv`;
- `pivot_average_epoch_time_seconds.csv`.

Эти таблицы позволяют сравнивать один и тот же архитектурный эксперимент при
разном количестве эпох обучения.

### Сравнительные графики

```bash
# Архивная scratch-версия. Для актуальных экспериментов используйте pretrained_diploma ниже.
```

## Новая актуальная серия: предобученная EfficientNetB0

После пересмотра плана добавлена отдельная папка:

```text
pretrained_diploma/
```

Она предназначена для экспериментов на основе
`EfficientNetB0(weights="imagenet")`. В этой серии модель не обучается с нуля:
оставшиеся слои EfficientNetB0 используют ImageNet-веса, а с нуля обучаются
только новая классификационная головка и добавленные хвосты. Fine-tuning
предобученных стадий отключен, потому что предыдущие эксперименты показали его
неэффективность для этой задачи. Аугментация в актуальной серии также
отключена.

Основная документация:

```text
pretrained_diploma/docs/pretrained_experiments.md
```

Используемый датасет:

```text
pretrained_diploma/data/balanced_1000_per_class_split/
```

Для актуальной pretrained-серии используется по `1000` изображений на класс:

- train - 700 изображений;
- validation - 150 изображений;
- test - 150 изображений.

Сетка эпох для каждого эксперимента:

```text
1, 2, 4, 6, 8, 10
```

Команда для проверки слоев EfficientNetB0 и границ стадий:

```bash
python pretrained_diploma/scripts/inspect_efficientnet_layers.py
```

Запуск базового эксперимента:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp00_full_model --epochs 1
```

Запуск базового эксперимента для всех чисел эпох:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp00_full_model --all-epoch-values
```

Запуск всех pretrained-экспериментов:

```bash
python pretrained_diploma/scripts/run_all_experiments.py
```

Новая серия `exp30`-`exp46` проверяет влияние размера датасета. Для неё созданы датасеты по `100`, `200`, `400`, `600`, `800`, `1200` и `1400` изображений на класс, а точка `1000` изображений на класс добавлена копиями уже готовых запусков на основном датасете. Датасеты `1200` и `1400` на класс собраны без изображений из holdout:

```text
pretrained_diploma/data/dataset_size_variants/
```

Запуск всей серии на 10 эпохах:

```bash
python pretrained_diploma/scripts/run_all_experiments.py --experiment-start 30 --experiment-end 46 --epochs 10 --all-dataset-sizes
```

Запуск серии только на выбранных размерах:

```bash
python pretrained_diploma/scripts/run_all_experiments.py --experiment-start 30 --experiment-end 46 --epochs 10 --dataset-size 1200 1400
```

Графики точности от суммарного размера выборки:

```bash
python pretrained_diploma/scripts/plot_accuracy_by_dataset_size.py --make-combined-plot
```

Независимая holdout-выборка для повторной проверки test accuracy:

```text
pretrained_diploma/data/holdout_not_train_val_300_per_class/test/
```

Она содержит `300` изображений на класс и не пересекается с train/validation по найденным manifest-файлам.

Проверка сохранённых моделей на holdout-выборке:

```bash
python pretrained_diploma/scripts/evaluate_holdout.py --experiment-start 30 --experiment-end 46
```

Восстановление графиков без обучения:

```bash
python pretrained_diploma/scripts/plot_history.py --experiment exp01_resolution_224_ep001
```

Восстановление общей таблицы:

```bash
python pretrained_diploma/scripts/rebuild_summary.py
```

Построение таблиц по числу эпох:

```bash
python pretrained_diploma/scripts/build_epoch_tables.py
```

Построение сравнительных графиков:

```bash
python pretrained_diploma/scripts/compare_experiments.py
```

Скрипт строит:

- график `test_accuracy` по экспериментам;
- график общего времени обучения;
- график зависимости `test_accuracy` от времени обучения;
- сравнение кривых `val_accuracy`;
- сравнение кривых `val_loss`.

Результаты сохраняются в:

```text
pretrained_diploma/results/comparison_plots/
```

### Краткая последовательность работы

1. Проверить один короткий pretrained-эксперимент:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp00_full_model --epochs 1
```

2. Запустить этот же эксперимент для всех чисел эпох:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp00_full_model --all-epoch-values
```

3. Запустить все pretrained-эксперименты:

```bash
python pretrained_diploma/scripts/run_all_experiments.py
```

4. Восстановить сводную таблицу:

```bash
python pretrained_diploma/scripts/rebuild_summary.py
```

5. Построить таблицы по числу эпох:

```bash
python pretrained_diploma/scripts/build_epoch_tables.py
```

6. Построить сравнительные графики:

```bash
python pretrained_diploma/scripts/compare_experiments.py
```
