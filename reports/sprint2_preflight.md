# Sprint 2 preflight — 21 октября 2026

Аудит выполнен до реализации PatchCore. Изучены README, progress, pyproject,
обе конфигурации, исходники data/baseline/decision/inference/evaluation/CLI,
тесты, docs, отчёт baseline, Git-статус и артефакты. AGENTS.md в проекте не найден.
Git содержит исходный LICENSE; файлы Sprint 1 пока не закоммичены. Их не удаляем.

## Фактически проверено

- Windows, Python 3.11.14, uv 0.9.5, RTX 3050 8192 MiB, драйвер 610.88.
- Torch и Anomalib до спринта не установлены. Наличие GPU пока подтверждено
  только nvidia-smi, не вычислением PyTorch.
- 40 тестов прошли, `ruff check .` и `ruff format --check .` прошли.
- Заново выполнен prepare без записи поверх прошлых результатов: все изображения
  и маски декодируются, структура корректна, точных pixel-дубликатов нет.
- Новый manifest побайтно по значениям совпадает с сохранённым manifest.csv.
- Manifest SHA256: `82dfcf4d1a2907ddb1a6d112c5b1add85be1e931f2b7a679c779f7e6b431dce1`.
- Только bottle: train 167 normal, calibration 42 normal,
  test 20 normal + 20 broken_large + 22 broken_small + 21 contamination.
- Baseline загружен из joblib и заново предсказал те же 83 test-файла:
  scores совпали с CSV с абсолютным допуском 1e-14, метрики воспроизвелись.
- AUROC 0.624603, AP 0.841982, F1 0.672897; локальные JSON/CSV/матрица/примеры есть.
- Данные, веса, отчёты с изображениями исключены из Git.

## ML-протокол и риски

Calibration выделена только из official train/good с seed 42. Official test
не используется для калибровки и выбора параметров. Независимой anomalous validation
нет. Используются нормальные эмпирические квантили без гарантии recall/FPR.

Результаты baseline уже известны. Сравнение с заранее выбранной конфигурацией
PatchCore — описательное post-hoc сравнение методов, а не новый слепой benchmark.
Не подбирать разрешение, backbone, coreset или пороги по test. После получения
результатов не перебирать параметры ради лучшей test-метрики.

Данные получены из сторонней HF-копии `foersben/mvtec-ad` с фиксированным commit,
авторством/лицензией MVTec и проверенными хешами копии. Независимого сравнения
с официальным архивом нет: официальная ссылка вернула HTTP 404.
Near-duplicates не проверяются.

Anomalib MVTecAD по умолчанию может строить validation из test. Для этого проекта
повторно используем manifest и normal-only calibration, без этого datamodule
и без автоматического postprocessor, обучающегося на test.

Старый baseline CLI допускает перезапись одного output_dir; исходные артефакты
Sprint 1 сохраняем. Для новых запусков нужны уникальные run-директории и отказ
от перезаписи. Нужна строгая проверка совпадения manifest/test paths при сравнении.

## Повторное использование и изменения

Повторно используем data.prepare, manifest, baseline, Thresholds/calibrate,
image metrics и функции отчётов. Рабочий baseline не переписываем.
Расширяем общий inference-контракт для карт и метаданных модели, pixel metrics,
синхронизированных latency summaries и визуализации с совпадающей геометрией.

Фиксированный первый PatchCore: ResNet18 (компактнее WideResNet), layer2/layer3,
pretrained, resize 224×224 без crop, coreset 1%, batch 4, seed 42,
auto CUDA/CPU. Пиксельный порог: квантиль 0.99 максимумов anomaly maps по
нормальным calibration-изображениям, до test (консервативная image-family policy,
без статистической гарантии). Image-пороги как у baseline: 0.95/0.99.

Блокирующих ошибок первого спринта не выявлено. Следующие шаги: добавить ML extra
в существующее окружение, проверить реальные API/CUDA, реализовать и запустить
PatchCore, проверить reload и сравнить на неизменённом test. Streamlit вне спринта.
