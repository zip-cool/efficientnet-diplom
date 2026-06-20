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

Серия сокращения повторяющихся MBConv-блоков внутри стадий:

- `exp21_repeat_full` - реконструкция полной EfficientNetB0 с повторами `[1, 2, 2, 3, 3, 4, 1]`;
- `exp22_repeat_light` - мягкое сокращение `[1, 2, 2, 2, 2, 3, 1]`;
- `exp23_repeat_medium` - среднее сокращение `[1, 1, 1, 2, 2, 2, 1]`;
- `exp24_repeat_minimal` - по одному MBConv-блоку на стадию `[1, 1, 1, 1, 1, 1, 1]`.

Серия компактных хвостов для варианта `exp23_repeat_medium`:

- `exp25_repeat_medium_tail_conv1x1`;
- `exp26_repeat_medium_tail_separable_conv`;
- `exp27_repeat_medium_tail_simple_mbconv`.

В этой серии не удаляются целые поздние стадии. Все семь стадий остаются, но
сокращается число повторных блоков внутри стадий. Первый блок каждой стадии
сохраняется, потому что он может менять spatial-размер и число каналов. Удаляются
только последние повторные блоки, которые сохраняют форму тензора.

Перед запуском сокращенных вариантов проверяется `exp21_repeat_full`: после
переноса ImageNet-весов она должна совпадать со стандартной `EfficientNetB0`.
Результат проверки сохраняется в `pretrained_diploma/results/validation/architecture_validation.json`.

Серия `exp25`-`exp27` использует ту же сокращённую по повторам основу, что и `exp23_repeat_medium`: сохраняются все семь стадий EfficientNetB0, но часть повторных MBConv-блоков удалена. После выхода `top_activation` добавляется один из компактных хвостов, аналогичных экспериментам `exp14`-`exp16`. Слои перенесённой EfficientNetB0 остаются предобученными на ImageNet и замороженными, а новые слои хвоста обучаются с нуля.

### Подробно о слоях в сериях 4 и 5

Серии 4 и 5 повторяют идею серии 3, но используют более ранние точки обрезки
EfficientNetB0. Во всех этих экспериментах ранняя часть EfficientNetB0 остается
предобученной на ImageNet и замороженной, а новые слои хвоста обучаются с нуля.

Серия 4 использует выход слоя:

```text
block5c_add
```

Это соответствует модели, обрезанной после stage 5. Все слои EfficientNetB0,
которые идут после `block5c_add`, не используются:

```text
block6*
block7*
top_conv
top_bn
top_activation
```

После `block5c_add` добавляются новые хвосты:

- `exp13_stage5_tail_gap`

```text
GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

- `exp14_stage5_tail_conv1x1`

```text
Conv2D(256, kernel_size=1)
→ BatchNormalization
→ Swish
→ GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

- `exp15_stage5_tail_separable_conv`

```text
SeparableConv2D(256, kernel_size=3, padding="same")
→ BatchNormalization
→ Swish
→ GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

- `exp16_stage5_tail_simple_mbconv`

```text
Conv2D(filters_in * 2, kernel_size=1)
→ BatchNormalization
→ Swish
→ DepthwiseConv2D(kernel_size=3, padding="same")
→ BatchNormalization
→ Swish
→ Conv2D(filters_in, kernel_size=1)
→ BatchNormalization
→ residual connection, если формы совпадают
→ GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

Серия 5 использует выход слоя:

```text
block4c_add
```

Это соответствует модели, обрезанной после stage 4. Все слои EfficientNetB0,
которые идут после `block4c_add`, не используются:

```text
block5*
block6*
block7*
top_conv
top_bn
top_activation
```

После `block4c_add` добавляются такие же четыре типа хвоста:

- `exp17_stage4_tail_gap`

```text
GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

- `exp18_stage4_tail_conv1x1`

```text
Conv2D(256, kernel_size=1)
→ BatchNormalization
→ Swish
→ GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

- `exp19_stage4_tail_separable_conv`

```text
SeparableConv2D(256, kernel_size=3, padding="same")
→ BatchNormalization
→ Swish
→ GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

- `exp20_stage4_tail_simple_mbconv`

```text
Conv2D(filters_in * 2, kernel_size=1)
→ BatchNormalization
→ Swish
→ DepthwiseConv2D(kernel_size=3, padding="same")
→ BatchNormalization
→ Swish
→ Conv2D(filters_in, kernel_size=1)
→ BatchNormalization
→ residual connection, если формы совпадают
→ GlobalAveragePooling2D
→ Dropout
→ Dense(3, activation="softmax")
```

Таким образом, серии 4 и 5 проверяют, можно ли компенсировать удаление более
глубоких стадий EfficientNetB0 за счет небольшого обучаемого хвоста. Если
точность заметно восстанавливается при небольшом росте времени, такой хвост
может быть полезным компромиссом между качеством и вычислительной сложностью.

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

Серия 6. Сокращение повторяющихся MBConv-блоков:

Сначала проверить полную реконструкцию EfficientNetB0:

```bash
python pretrained_diploma/scripts/validate_repeat_reconstruction.py
```

После успешной проверки можно запускать серию:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp21_repeat_full --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp22_repeat_light --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp23_repeat_medium --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp24_repeat_minimal --all-epoch-values
```

Серия 7. Компактные хвосты для `exp23_repeat_medium`:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp25_repeat_medium_tail_conv1x1 --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp26_repeat_medium_tail_separable_conv --all-epoch-values
```

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp27_repeat_medium_tail_simple_mbconv --all-epoch-values
```

Серия 8. Влияние размера датасета при фиксированных 10 эпохах:

Созданы датасеты по `100`, `200`, `400`, `600` и `800` изображений на класс. Также в эту же серию добавляется контрольная точка `1000` изображений на класс: для неё используются уже готовые результаты соответствующих экспериментов на основном датасете `balanced_1000_per_class_split`. Каждый датасет сохраняет пропорцию текущего разбиения `70/15/15`:

- `100/class`: train `70`, validation `15`, test `15`;
- `200/class`: train `140`, validation `30`, test `30`;
- `400/class`: train `280`, validation `60`, test `60`;
- `600/class`: train `420`, validation `90`, test `90`;
- `800/class`: train `560`, validation `120`, test `120`.
- `1000/class`: train `700`, validation `150`, test `150`.

Папки датасетов:

```text
pretrained_diploma/data/dataset_size_variants/
```

Отчёт о составе датасетов:

```text
pretrained_diploma/results/dataset_splits/dataset_size_variants/summary.json
pretrained_diploma/results/dataset_splits/dataset_size_variants/manifest.csv
```

Эксперименты серии начинаются с `exp30`. Один эксперимент запускается на всех размерах датасета через параметр `--all-dataset-sizes`, а в имени результата появляется суффикс вида `data0100`, `data0200`, `data0400`, `data0600`, `data0800`, `data1000`.

Соответствие новым экспериментам:

- `exp30_full_model_dataset_size` повторяет `exp00_full_model`;
- `exp31_without_stage7_dataset_size` повторяет `exp06_without_stage7`;
- `exp32_without_stages6_7_dataset_size` повторяет `exp07_without_stages6_7`;
- `exp33_without_stages5_7_dataset_size` повторяет `exp08_without_stages5_7`;
- `exp34_stage5_tail_conv1x1_dataset_size` повторяет `exp14_stage5_tail_conv1x1`;
- `exp35_stage5_tail_separable_conv_dataset_size` повторяет `exp15_stage5_tail_separable_conv`;
- `exp36_stage5_tail_simple_mbconv_dataset_size` повторяет `exp16_stage5_tail_simple_mbconv`;
- `exp37_stage4_tail_conv1x1_dataset_size` повторяет `exp18_stage4_tail_conv1x1`;
- `exp38_stage4_tail_separable_conv_dataset_size` повторяет `exp19_stage4_tail_separable_conv`;
- `exp39_stage4_tail_simple_mbconv_dataset_size` повторяет `exp20_stage4_tail_simple_mbconv`;
- `exp40_repeat_full_dataset_size` повторяет `exp21_repeat_full`;
- `exp41_repeat_light_dataset_size` повторяет `exp22_repeat_light`;
- `exp42_repeat_medium_dataset_size` повторяет `exp23_repeat_medium`;
- `exp43_repeat_minimal_dataset_size` повторяет `exp24_repeat_minimal`.

Создать датасеты заново:

```bash
python pretrained_diploma/scripts/create_dataset_size_variants.py --overwrite
```

Запустить один эксперимент на всех размерах датасета:

```bash
python pretrained_diploma/scripts/run_experiment.py --experiment exp30_full_model_dataset_size --epochs 10 --all-dataset-sizes
```

Запустить всю серию `exp30`-`exp43` на всех размерах датасета:

```bash
python pretrained_diploma/scripts/run_all_experiments.py --experiment-start 30 --experiment-end 43 --epochs 10 --all-dataset-sizes
```

Построить для каждого эксперимента график зависимости точности от суммарного размера выборки:

```bash
python pretrained_diploma/scripts/plot_accuracy_by_dataset_size.py --make-combined-plot
```

В этих графиках ось X показывает не размер на класс, а общее число изображений во всех классах: `300`, `600`, `1200`, `1800`, `2400`, `3000`.

## Результаты

Каждый запуск сохраняется в:

```text
pretrained_diploma/results/experiments/<experiment_id>/
```

Сохраняются:

- `config.json`;
- `model_structure.json`;
- `weight_transfer_report.txt` для серии сокращения повторов;
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
