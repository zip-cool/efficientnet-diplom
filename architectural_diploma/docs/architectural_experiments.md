# Архитектурные эксперименты EfficientNet

В папке `architectural_diploma` находится отдельный pipeline для новой части
диплома: экспериментов с архитектурой EfficientNet-подобной модели, обучаемой с
нуля. Старые скрипты transfer learning и их результаты остаются отдельно и не
изменяются.

## Датасет

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

## Один эксперимент

```bash
python architectural_diploma/scripts/run_experiment.py --experiment exp00_baseline --epochs 20
```

Запуск одного и того же эксперимента для всех заданных чисел эпох:

```bash
python architectural_diploma/scripts/run_experiment.py --experiment exp00_baseline --all-epoch-values
```

Флаг `--overwrite` используйте только тогда, когда существующую папку
эксперимента можно заменить.

## Все эксперименты

```bash
python architectural_diploma/scripts/run_all_experiments.py
```

Команда запускает все обязательные архитектурные эксперименты для `5, 10, 20,
40, 60, 80, 100` эпох обучения.

Дополнительный эксперимент `exp15_head_160` по умолчанию пропускается. Чтобы
включить его, используйте:

```bash
python architectural_diploma/scripts/run_all_experiments.py --include-optional
```

## Повторное построение графиков

```bash
python architectural_diploma/scripts/plot_history.py --experiment exp00_baseline_ep020
```

Графики строятся заново из файла `history.csv`, поэтому PNG-файлы не являются
единственным источником данных.

## Восстановление сводной таблицы

```bash
python architectural_diploma/scripts/rebuild_summary.py
```

Команда заново создает
`architectural_diploma/results/tables/experiments_summary.csv` по сохраненным
папкам отдельных экспериментов.

## Сравнение экспериментов

```bash
python architectural_diploma/scripts/compare_experiments.py
```

Сравнительные графики сохраняются в
`architectural_diploma/results/experiments/comparison_plots/`.

## Таблицы по числу эпох

```bash
python architectural_diploma/scripts/build_epoch_tables.py
```

Команда создает отдельные таблицы для каждого числа эпох, например
`epochs_005.csv`, а также сводные pivot-таблицы, например
`pivot_test_accuracy.csv`.
