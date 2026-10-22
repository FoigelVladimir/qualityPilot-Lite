# Sprint 2 — PatchCore, evaluation и сравнение

Дата: 21 октября 2026. Статус: **проверен реальным запуском**.

Run: `bottle-patchcore-20261021T103450Z-5eddf169`.
Полные локальные артефакты: папка запуска (`runs/bottle-patchcore-20261021T103450Z-5eddf169/`, только локально; исключено из Git).
Исходные изображения, веса и отчёты с изображениями исключены из Git.

## Проверка Sprint 1

[Preflight](sprint2_preflight.md) создан до реализации. Заново проверены структура
и декодирование данных, маски, дубликаты, точное совпадение manifest, загрузка
baseline и повторение его scores. Исходные 40 тестов и Ruff прошли.
Рабочая архитектура baseline сохранена; добавлен отказ от перезаписи его артефактов.

## Данные и протокол

Только MVTec AD bottle. Источник: сторонняя копия
[foersben/mvtec-ad](https://huggingface.co/datasets/foersben/mvtec-ad), commit
`c75b39616f84db43677bcc8228caaafaf5096d7f`, CC BY-NC-SA 4.0.
Авторы и copyright MVTec сохранены. Хеши копии проверены; идентичность
недоступному официальному архиву независимо не подтверждена.

Train: 167 нормальных изображений. Calibration: 42 нормальных изображения,
отложенных из official train/good. Official test не изменён: 83 изображения,
20 нормальных и 63 дефектных (20 broken_large, 22 broken_small, 21 contamination).

Manifest SHA256:
`82dfcf4d1a2907ddb1a6d112c5b1add85be1e931f2b7a679c779f7e6b431dce1`.
Обе модели проверены на одинаковых test paths/labels в одном порядке.
Пересечений и точных дубликатов RGB-пикселей не обнаружено; near-duplicates не проверены.

Конфигурация PatchCore заранее записана в preflight и config.yaml.
Test не использовался для выбора backbone, разрешения, coreset и порогов.
Результаты baseline уже были известны: это описательное **post-hoc сравнение**
фиксированных методов, а не новый слепой benchmark и не подбор лучшего варианта по test.
После просмотра результатов параметры не менялись и повторного поиска моделей не было.

## Методы

Baseline: 60 глобальных RGB/текстурных признаков, Isolation Forest, 200 деревьев,
resize 128×128, seed 42. Модель Sprint 1 загружена без переобучения;
scores совпали с историческим CSV с допуском 1e-12.

PatchCore: нативный `anomalib.models.image.patchcore.torch_model.PatchcoreModel`.
Использованы предобученные ImageNet-признаки ResNet18 (`timm/resnet18.a1_in1k`),
слои layer2/layer3. Backbone не обучается с нуля и не оптимизируется.
Anomalib выполняет coreset, поиск соседей, image score и anomaly map;
собственной реализации PatchCore нет. Engine и стандартный MVTecAD datamodule
не используются, чтобы не подключать test-derived validation/postprocessor.

| Настройка | Значение |
|---|---|
| Размер входа | 224×224, без crop и поворота |
| Преобразования | RGB, float32 / 255, bilinear antialiased resize, ImageNet normalize |
| Backbone / layers | ResNet18 / layer2, layer3 |
| Coreset ratio | 0.01 |
| Число соседей | 9 |
| Training batch size | 4 |
| Device | CUDA, RTX 3050 |
| Seed | 42 |
| CPU threads | 4 |
| Патчей до coreset | 130 928 |
| Memory bank | 1 309 × 384 float32, 2 010 624 байт |
| Подготовка модели | 16.11 s, включая инициализацию/получение весов, признаки и coreset |
| Артефакт model.pt | 13 247 119 байт |

ResNet18 выбран как компактный backbone для 8 GB VRAM. Успешный запуск подтвердил
пригодность этой конфигурации на текущей машине; пиковая VRAM отдельно не измерялась.
Детерминированные алгоритмы включены, TF32 и cudnn benchmark выключены.
Битовая идентичность обучения на другом GPU/ОС не обещается.

## Калибровка до test

Image thresholds: эмпирические normal-only квантили 0.95/0.99, метод `higher`:

- Baseline: accept 0.5005465295666041, reject 0.5561937682120061.
- PatchCore: accept 8.626294136047363, reject 9.292860984802246.

`score <= accept` → ACCEPT; `accept < score <= reject` → REVIEW;
`score > reject` → REJECT. Один порог: `score > accept` означает подозрительное изделие.
Scores не являются вероятностями, их шкалы между моделями несопоставимы.

Pixel threshold PatchCore: **8.860112190246582**, квантиль 0.99 максимумов native
anomaly maps по 42 нормальным calibration-изображениям, метод `higher`.
Test masks для порога не использовались. Это эмпирическая operating policy,
а не оптимальный Dice/IoU threshold. Нет гарантии FPR, recall или частоты
ложных пиксельных срабатываний на будущих данных.

## Реальные результаты

| Metric | Baseline CPU | PatchCore CUDA |
|---|---:|---:|
| Image AUROC | 0.624603 | 1.000000 |
| Average Precision | 0.841982 | 1.000000 |
| Precision, фиксированный accept | 0.818182 | 0.940299 |
| Recall, фиксированный accept | 0.571429 | 1.000000 |
| F1, фиксированный accept | 0.672897 | 0.969231 |
| False positive rate | 0.400000 | 0.200000 |
| Median latency, ms | 22.0084 | 28.6125 |
| p95 latency, ms | 22.9425 | 31.1595 |
| Pixel AUROC, 224×224 | N/A | 0.976049 |
| Micro Dice, frozen pixel threshold | N/A | 0.560382 |
| Micro IoU, frozen pixel threshold | N/A | 0.389257 |

Pixel AUROC рассчитан точно по 4 164 608 пикселям native сетки 224×224 всех 83
test-изображений, включая normal. Ground truth уменьшена nearest-neighbour;
Dice/IoU агрегированы micro по всему набору. Это метрики на сетке модели,
не оценка точности границы на исходной сетке 900×900.
Baseline не поддерживает локализацию, поэтому соответствующие значения N/A.

Матрицы ошибок (строки true, столбцы predicted: normal, defect):

```text
Baseline:  [[12, 8], [27, 36]]
PatchCore: [[16, 4], [ 0, 63]]
```

| Стратегия | Baseline | PatchCore |
|---|---:|---:|
| Один порог: пропущенные дефекты | 27 | 0 |
| Один порог: ложные подозрения на normal | 8 | 4 |
| Два порога: ложные ACCEPT дефектов | 27 | 0 |
| Два порога: ложные REJECT normal | 3 | 0 |
| Два порога: REVIEW | 26 / 83 (31.33%) | 4 / 83 (4.82%) |

Четыре PatchCore REVIEW — нормальные изделия. Их исходы при ручной проверке
неизвестны, поэтому нельзя заявлять безошибочную систему из нулевых автоматических ошибок.

## Визуализация и ошибки

В каждом примере: original, heatmap, overlay, ground truth, predicted mask.
Карты bilinear возвращаются в исходные координаты 900×900; crop/rotation нет.
GT и predicted mask в PNG имеют ту же геометрию, что исходник.
Численные pixel metrics выше используют native 224×224. Цветовая шкала фиксирована
от 0 до 1.5 × calibration pixel threshold, не подстраивается отдельно под test-картинки.

- TP: broken_large/000.png (`runs/bottle-patchcore-20261021T103450Z-5eddf169/examples/tp.png`, только локально; исключено из Git)
- FP при одном пороге, REVIEW при двух (`runs/bottle-patchcore-20261021T103450Z-5eddf169/examples/fp.png`, только локально; исключено из Git)
- TN: normal good/000.png (`runs/bottle-patchcore-20261021T103450Z-5eddf169/examples/tn.png`, только локально; исключено из Git)
- FN: **0 случаев**; пример отсутствует, фиктивное изображение не создавалось.

Выбран первый пример каждого типа в фиксированном порядке, без ручного отбора.
Все три PNG просмотрены: карты согласованы с исходниками. На TP карта покрывает
реальный дефект, но захватывает существенно более широкую область — это согласуется
с умеренным Dice/IoU. На первом FP карта реагирует на небольшой участок внутреннего
края нормального изделия; причинный вывод по одному примеру сделать нельзя.

## Скорость

Batch 1, 83 измерения, три calibration-image warmup перед каждой серией.
CUDA синхронизируется до начала и после preprocessing/model inference.
Total включает чтение файла, RGB/resize/normalize/features, передачу, модель и
решение; map в этой серии возвращается в native сетке. Запись PNG и расчёт метрик
не входят. Загруженная модель остаётся в памяти; cold start отдельно не измерен.

PatchCore CUDA, средние времена стадий:
preprocessing **21.2664 ms**, inference с CPU-передачей результата **7.2137 ms**,
postprocessing **0.0471 ms**. Суммарное среднее **28.5272 ms**.
Названия `latency_components_ms` обозначают стадии; вложенный общий `scope`
в сохранённом JSON описывает весь pipeline, не каждую стадию отдельно.

Дополнительно реальный PatchCore CPU benchmark: median **48.3133 ms**,
p95 **55.3131 ms**, mean **48.8129 ms**, 83 измерения после трёх warmup.
CPU fallback реально выполнен, а не только заявлен. В этой реализации CUDA
быстрее CPU PatchCore, но полный PatchCore CUDA pipeline медленнее baseline CPU.
Качество повысилось; ускорения относительно baseline эксперимент не показал.

## Hardware / software и артефакты

Windows build 26200, 16 GB RAM, NVIDIA RTX 3050 8 GB, драйвер 610.88,
Python 3.11.14, torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA runtime 12.8,
cuDNN 91002, Anomalib 2.2.0, timm 1.0.20, Lightning 2.5.5,
pytorch-lightning 2.6.6, torchmetrics 1.8.2, NumPy 2.2.6, sklearn 1.7.2,
OpenCV 4.12.0.88, Pillow 11.3.0, huggingface-hub 0.36.0, requests 2.32.5.
Полное разрешённое дерево в uv.lock; `uv pip check` прошёл.

В папке успешного запуска: config.yaml, environment.json, manifest.csv,
dataset_summary.json, calibration.csv, experiment.json, model.pt,
model_checksum.json, reload_check.json, comparison.json/markdown,
baseline/metrics.json и results.csv, patchcore/metrics.json и results.csv,
confusion matrices, examples и source_and_measurement_notes.json.

Модель сохранена до первого test inference. Повторная загрузка проверена на
calibration: максимальная разница score и карты **0.0**, форма **224×224**.
Повторно используется весь state_dict, включая backbone и memory bank,
`pre_trained=False` при восстановлении; повторная загрузка весов из сети не нужна.
Checkpoint читается с `weights_only=True`. SHA256 файла дополнительно перепроверен.
Loader восстанавливает также seed, CPU threads и deterministic/TF32 policy:
это устранило обнаруженный небольшой дрейф при загрузке из нового процесса
с иными PyTorch defaults. Независимая проверка всех 83 test scores в новом процессе
на CUDA прошла: максимальная разница с сохранённым CSV **3.55e-15**.
Результат в fresh_process_check.json; SHA256 всех 357 исходных файлов, включая маски,
повторно проверены. Сохранённые результаты эксперимента не перезаписывались.

Первый запуск `bottle-patchcore-20261021T103240Z-8cd95dd9` завершился FAILED до
обучения: timm 1.0.20 не распознал API huggingface-hub 2.2.0. Установлен pin
0.36.0, затем выполнен один успешный запуск с исходной ML-конфигурацией.
Также явно добавлен requests: импорт Anomalib обращается к нему через video-модуль.
Неудачный запуск сохранён вместе с конфигурацией и диагностикой, не скрыт.
Anomalib выдаёт upstream DeprecationWarning про embeddings даже при вызове
subsample_embedding без этого аргумента; это предупреждение, не ошибка тестов.

## Проверки и ограничения

- Реальный CPU/CUDA inference, reload, карты, сравнение и CPU benchmark выполнены.
- Тесты без датасета/GPU проверяют normal-only gate, metrics, latency, формы,
  сериализацию, отсутствие файлов, повреждённый checkpoint и сохранность артефактов.
- Дополнительный smoke использует реальную обученную модель, не случайные веса.
- Итог: **62 passed**, без пропусков при явно включённом real smoke;
  тесты прошли с HF_HUB_OFFLINE=1. Ruff check/format и uv lock/pip check прошли.
- Linux и удалённый CI ещё не запускались; CI baseline остаётся без загрузки ML/данных.
- 42 normal calibration — маленькая выборка; anomalous validation отсутствует.
- AUROC 1.0 относится только к этой категории и этим 83 изображениям.
  Он не означает идеальный фиксированный порог или надёжность на других изделиях.
- Проверены только точные дубликаты; source — сторонняя копия; нет теста на дрейф,
  иной свет/камеру или промышленные cost-sensitive ограничения.

Спринт 2 реализован. Следующий этап — Streamlit поверх общего Inspector,
без повторной реализации preprocessing, inference, порогов и метрик.
