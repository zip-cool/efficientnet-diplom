# Эксперименты с предобученной EfficientNetB0

Эта папка содержит новую систему экспериментов на основе
`EfficientNetB0(weights="imagenet")`. Модель не обучается с нуля: все
оставшиеся слои EfficientNetB0 используют ImageNet-веса, а с нуля обучаются
только новая классификационная головка и добавленные хвосты.

## Датасет

Используется фиксированное разбиение:

```text
pretrained_diploma/data/balanced_1000_per_class_split/
```

В каждом классе по 1000 изображений:

- train - 700;
- validation - 150;
- test - 150.

## Стратегия обучения

Для каждого запуска:

1. замораживается вся предобученная EfficientNetB0;
2. обучается только новая классификационная головка или добавленный хвост;
3. fine-tuning предобученных стадий не выполняется.

Аугментация отключена: train, validation и test используются без случайных
геометрических или цветовых преобразований.

Fine-tuning отключен намеренно: предыдущие эксперименты показали, что он не
дает полезного выигрыша для этой задачи, но заметно увеличивает время обучения.
В таблицах поле времени fine-tuning сохраняется для совместимости и равно `0`.

Общее число эпох задается сеткой:

```text
1, 2, 4, 6, 8, 10
```

## Начальный baseline

Первым запускается базовый эксперимент:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp00_full_model --all-epoch-values
```

`exp00_full_model` используется как основной baseline: это полная
предобученная `EfficientNetB0(weights="imagenet")` с входным разрешением `224`,
без обрезки стадий и с обычным хвостом
`GlobalAveragePooling2D → Dropout → Dense(3)`.

С ним сравниваются все последующие варианты: изменение разрешения, обрезка
поздних стадий и замена хвоста.

## Серии экспериментов

Серия разрешений:

- `exp01_resolution_224`;
- `exp02_resolution_192`;
- `exp03_resolution_160`;
- `exp04_resolution_128`.

Серия обрезки модели:

- `exp06_without_stage7`;
- `exp07_without_stages6_7`;
- `exp08_without_stages5_7`.

Серия компактных хвостов после stage 6 (`block6d_add`):

- `exp09_tail_gap`;
- `exp10_tail_conv1x1`;
- `exp11_tail_separable_conv`;
- `exp12_tail_simple_mbconv`.

Дополнительная серия компактных хвостов после stage 5 (`block5c_add`):

- `exp13_stage5_tail_gap`;
- `exp14_stage5_tail_conv1x1`;
- `exp15_stage5_tail_separable_conv`;
- `exp16_stage5_tail_simple_mbconv`.

Дополнительная серия компактных хвостов после stage 4 (`block4c_add`):

- `exp17_stage4_tail_gap`;
- `exp18_stage4_tail_conv1x1`;
- `exp19_stage4_tail_separable_conv`;
- `exp20_stage4_tail_simple_mbconv`.

Эти дополнительные серии нужны, чтобы проверить, может ли более выразительный
хвост компенсировать падение точности после более сильной обрезки модели.

## Инспекция слоев EfficientNetB0

```bash
python pretrained_diploma/scripts/inspect_efficientnet_layers.py
```

Список слоев сохраняется в:

```text
pretrained_diploma/docs/efficientnetb0_layers.txt
```

Используемые границы:

- full model - `top_activation`;
- without stage 7 - `block6d_add`;
- without stages 6-7 - `block5c_add`;
- without stages 5-7 - `block4c_add`.

## Запуск

Один эксперимент на 5 эпох:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp01_resolution_224 --epochs 1
```

Один эксперимент на все числа эпох:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp01_resolution_224 --all-epoch-values
```

Все эксперименты:

```bash
python pretrained_diploma/scripts/run_all_experiments.py
```

## Ручной последовательный запуск каждого эксперимента

Если нужно запускать эксперименты вручную по одному, но каждый сразу для всех
чисел эпох `1, 2, 4, 6, 8, 10`, используйте следующие команды.

Начальный baseline:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp00_full_model --all-epoch-values
```

Серия 1. Изменение входного разрешения:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp01_resolution_224 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp02_resolution_192 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp03_resolution_160 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp04_resolution_128 --all-epoch-values
```

Серия 2. Уменьшение глубины через обрезку модели:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp06_without_stage7 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp07_without_stages6_7 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp08_without_stages5_7 --all-epoch-values
```

Серия 3. Компактная замена хвоста:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp09_tail_gap --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp10_tail_conv1x1 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp11_tail_separable_conv --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp12_tail_simple_mbconv --all-epoch-values
```

Серия 4. Компактная замена хвоста после stage 5:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp13_stage5_tail_gap --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp14_stage5_tail_conv1x1 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp15_stage5_tail_separable_conv --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp16_stage5_tail_simple_mbconv --all-epoch-values
```

Серия 5. Компактная замена хвоста после stage 4:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp17_stage4_tail_gap --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp18_stage4_tail_conv1x1 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp19_stage4_tail_separable_conv --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp20_stage4_tail_simple_mbconv --all-epoch-values
```

## Результаты

Каждый запуск сохраняется в:

```text
pretrained_diploma/results/experiments/<experiment_id>/
```

Сохраняются:

- `config.json`;
- `model_structure.json`;
- `model_summary.txt`;
- `history.csv`;
- `history.json`;
- `final_metrics.json`;
- `metrics_report.json`;
- `test_results.json`;
- `confusion_matrix.csv`;
- `confusion_matrix.png`;
- `best_model.keras`;
- `final_model.keras`;
- графики `accuracy.png` и `loss.png`.

`final_metrics.json` и `metrics_report.json` содержат тот же набор основных
метрик, что и отчеты предыдущих серий: размеры train/validation/test, число
изображений на класс, параметры обучения, train/validation/test accuracy и loss,
macro precision, recall, F1-score, confusion matrix paths, размер файла модели,
число параметров, FLOPs на инференс, скорость инференса и оценку GFLOPs/s.

Графики строятся из `history.csv`, поэтому их можно восстановить без повторного
обучения:

```bash
python pretrained_diploma/scripts/plot_history.py --experiment exp01_resolution_224_ep001
```

Графики одного эксперимента по числу эпох строятся отдельной командой:

```bash
python pretrained_diploma/scripts/plot_epoch_series.py --experiment exp00_full_model
```

Она сохраняет `accuracy_by_epochs.png`, `training_time_by_epochs.png` и
таблицу `epoch_series_metrics.csv`.

Если нужно вручную выбрать эксперименты, эпохи и метрики, используется
настраиваемый скрипт:

```bash
python pretrained_diploma/scripts/plot_custom_series.py --accuracy-experiments exp00_full_model --accuracy-epochs 1 2 4 6 8 10 --accuracy-metrics train test --time-experiments exp00_full_model --time-epochs 1 2 4 6 8 10
```

Добавить validation accuracy на график точности можно флагом:

```bash
python pretrained_diploma/scripts/plot_custom_series.py --accuracy-experiments exp00_full_model exp02_resolution_192 --accuracy-epochs 1 2 4 6 8 10 --accuracy-metrics train test --include-val-accuracy --time-experiments exp00_full_model exp02_resolution_192 --time-epochs 1 2 4 6 8 10
```

Для честного сравнения train/test accuracy лучше использовать `train_eval`:
это train accuracy, посчитанная после обучения в inference-режиме, то есть с
выключенным Dropout и так же, как считается test accuracy.

Для уже готовых запусков такие метрики можно досчитать без повторного обучения:

```bash
python pretrained_diploma/scripts/backfill_eval_metrics.py --experiments exp00_full_model --epochs 1 2 4 6 8 10
```

После этого график строится так:

```bash
python pretrained_diploma/scripts/plot_custom_series.py --accuracy-experiments exp00_full_model --accuracy-epochs 1 2 4 6 8 10 --accuracy-metrics train_eval test --time-experiments exp00_full_model --time-epochs 1 2 4 6 8 10
```

По умолчанию результаты сохраняются в:

```text
pretrained_diploma/results/custom_plots/
```

Восстановить summary:

```bash
python pretrained_diploma/scripts/rebuild_summary.py
```

Построить таблицы по числу эпох:

```bash
python pretrained_diploma/scripts/build_epoch_tables.py
```

Построить сравнительные графики:

```bash
python pretrained_diploma/scripts/compare_experiments.py
```
