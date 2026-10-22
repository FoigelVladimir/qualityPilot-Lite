# Progress — 21 октября 2026

## Спринт 1: код и реальный baseline проверены

Сделано:

- Клонирован FoigelVladimir/qualityPilot-Lite; существующий LICENSE сохранён.
- Созданы структура src, настройки, packaging, CLI, pytest и Ruff, Windows/Linux CI.
- Установлен изолированный Python 3.11.14; baseline установлен в `.venv`.
- Проверены документация Anomalib 2.2.0 и официальная пара torch/torchvision.
- Сформирован uv.lock для воспроизводимости, включая optional ML/UI зависимости.
- Валидация изображений/масок/структуры, manifest, распределение, seed, хеши.
- Отдельные train/calibration/test: calibration только из нормального official train.
- Проверка пересечения путей и точных дубликатов декодированных изображений.
- 60 признаков + Isolation Forest; сохранение/загрузка; общий Inspector.
- Калибровка 2 квантилей, ACCEPT/REVIEW/REJECT, отчёт по обеим стратегиям.
- Image-level метрики, N/A с причинами, CPU latency, CSV/JSON/матрица, TP/FP/FN.
- Проверка официального доступа, безопасный импорт локального tar-архива.
- Инструкции получения данных и продолжения спринтов 2–4.

Успешно выполнено локально (Windows, Python 3.11.14):

```text
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
uv lock
ruff check .
ruff format --check .
python -m pytest -q                         # 40 passed после добавления mirror-тестов
python -m qualitypilot.cli --help
python -m qualitypilot.cli access           # обнаружена форма MVTec
```

В тестах выполнялись обучение, калибровка, сохранение/загрузка baseline,
инференс и полный экспорт отчётов на маленьких синтетических данных.
Тестовые фикстуры не используются для утверждений о качестве MVTec AD.

После запроса пользователя продолжить без обращения в поддержку:

- Проверена публичная сторонняя копия `foersben/mvtec-ad` на Hugging Face.
  Сохранены copyright/лицензия и атрибуция; оригинальная структура присутствует.
- Явная команда download-mirror закрепляет commit, проверяет LFS SHA256/Git blob
  хеши, скачивает только категорию и сохраняет provenance в данные, модель и отчёт.
- Успешно: `python -m qualitypilot.cli download-mirror` — 357 файлов, 156.6 MB.
- Валидация структуры, всех изображений, масок, точных дубликатов прошла.
- Успешно: `python -m qualitypilot.cli baseline` — обучение 167 normal,
  calibration 42 normal, test 83 (20 normal + 63 defect).
- Фактически измерено: AUROC 0.624603, AP 0.841982, precision 0.818182,
  recall 0.571429, F1 0.672897, FPR 0.4, средняя CPU latency 21.13 ms.
- При одном пороге FN=27, FP=8. При двух: false ACCEPT=27,
  false REJECT=3, REVIEW=26/83 (31.33%); исходы ручной проверки неизвестны.
- Полные результаты: reports/runs/bottle-baseline/; краткий отчёт без изображений:
  reports/bottle-baseline.md. После просмотра test настройки не изменялись.

Блокеры и ограничения:

- Пользователь заполнил официальную форму MVTec. Ссылка Bottle с полученной
  страницы downloads проверена через GET и HEAD: HTTP 404, HTML MyDrive
  «Seite nicht gefunden». Официальный архив не получен. Копия позволила продолжить,
  но её идентичность официальному архиву независимо не подтверждена.
- Этот блокер больше не мешает baseline; provenance сторонней копии обязателен
  при интерпретации результатов. Инструкции и ограничения: docs/data.md.
- ML extra разрешён resolver-ом, но не установлен и не проверен запуском.
  Наличие RTX 3050 подтверждено nvidia-smi; CUDA inference ещё не выполнялся.
- PatchCore, heatmaps, сравнение моделей и Streamlit пока не реализованы.
- Тесты на Linux и удалённый CI ещё не выполнялись.
- Проверяются точные дубликаты; near-duplicates не выявляются.
- Нет аномальной validation; normal-only calibration не гарантирует defect recall.
- Публикация изменений в GitHub пока не выполнялась.

Дальше: перейти к PatchCore и задачам docs/next-sprints.md, сохранить текущий test
для честного сравнения при заранее фиксированной конфигурации и калибровке.
Весь проект завершённым не считается.

## Спринт 2 — PatchCore и сравнение, 21 октября 2026

История Sprint 1 выше сохранена. Статус: реализация, реальный эксперимент и
окончательные проверки выполнены: **62 passed**, Ruff check/format прошли.

Аудит:

- Создан reports/sprint2_preflight.md до реализации PatchCore.
- Реально повторены data validation, сравнение manifest, загрузка baseline и
  inference на всех 83 test-файлах; scores совпали с историческим CSV.
- Исходные 40 тестов и Ruff прошли; критических блокеров данных/протокола нет.
- Исправлен риск перезаписи: baseline и prepare сохраняют существующие результаты,
  каждый PatchCore run получает уникальный идентификатор и свою папку.

Окружение:

- Существующее Python 3.11.14 окружение расширено через uv, без полного пересоздания.
- Torch 2.8.0+cu128 / torchvision 0.23.0+cu128 из официального cu128-индекса.
- CUDA runtime 12.8, RTX 3050 8 GB; матричные вычисления, обучение memory bank и
  inference реально выполнены на CUDA. CPU fallback проверен отдельной серией.
- Anomalib 2.2.0, timm 1.0.20; добавлен requests 2.32.5 для импортов Anomalib.
- huggingface-hub закреплён на 0.36.0 после обнаруженной несовместимости 2.2.0
  с timm. uv.lock актуален, `uv pip check` прошёл.

Реализация:

- Использована нативная реализация Anomalib PatchcoreModel, без переписывания алгоритма.
- ResNet18 pretrained ImageNet, layer2/layer3, 224×224 без crop,
  coreset 1%, training batch 4, seed 42; настройки зафиксированы до test.
- Normal-only gate разрешает обучение/калибровку только на official train/good.
  Стандартный Anomalib datamodule/postprocessor с test-derived validation не используется.
- Повторно используются исходный manifest, baseline, image metrics и Thresholds.
- Pixel threshold вычисляется по 0.99 квантилю максимумов native calibration maps,
  не по test masks; image thresholds — нормальные эмпирические квантили 0.95/0.99.
- Единый Inspector принимает путь/PIL.Image, baseline .joblib/PatchCore .pt,
  возвращает score, decision, category/device, timings и float32 anomaly map.
- По умолчанию карта возвращается в исходные координаты; native map доступна
  через map_resolution="model". Проверяются повреждённые файлы/форматы/checkpoints.
- Exact pixel AUROC и micro Dice/IoU на сетке 224×224, N/A с причинами, latency summaries.
- Новые запуски проверяют SHA256 provenance всех исходных файлов, включая маски;
  одного image manifest для проверки изменения masks недостаточно.
- Модель, конфигурация, environment, provenance, пороги, контрольные суммы,
  reload check, CSV/JSON, матрицы и примеры сохраняются в run-директории.

Реальные запуски:

1. `bottle-patchcore-20261021T103240Z-8cd95dd9`: FAILED до обучения из-за
   несовместимого API huggingface-hub/timm. Конфигурация и причина сохранены.
2. `bottle-patchcore-20261021T103450Z-5eddf169`: COMPLETE, CUDA.
   167 train / 42 calibration / 83 test, тот же manifest, что у baseline.
   130 928 патчей → memory bank 1 309×384, 2 010 624 байт.
   Подготовка 16.11 s, model.pt 13 247 119 байт.

Измеренные метрики:

| Metric | Baseline CPU | PatchCore CUDA |
|---|---:|---:|
| Image AUROC | 0.624603 | 1.000000 |
| AP | 0.841982 | 1.000000 |
| Precision | 0.818182 | 0.940299 |
| Recall | 0.571429 | 1.000000 |
| F1 | 0.672897 | 0.969231 |
| Median latency ms | 22.0084 | 28.6125 |
| p95 latency ms | 22.9425 | 31.1595 |
| Pixel AUROC | N/A | 0.976049 |
| Dice | N/A | 0.560382 |
| IoU | N/A | 0.389257 |

PatchCore CPU median 48.3133 ms, p95 55.3131 ms. Измерения: batch 1,
83 изображения, 3 calibration-image warmup, CUDA synchronize в точках замера.
PatchCore повысил качество, но оказался медленнее baseline в полном pipeline.

PatchCore confusion matrix [[16,4],[0,63]]: 0 FN, 4 FP при одном пороге.
При двух порогах: 0 ложных ACCEPT, 0 ложных REJECT, REVIEW 4/83 (4.82%),
все REVIEW нормальные; исходы ручной проверки неизвестны.
Baseline: 27 FN, 8 FP; при двух порогах REVIEW 26/83.

Визуализации TP/FP/TN созданы и просмотрены, FN отсутствуют и не имитировались.
На TP маска шире ground truth, что согласуется с умеренным Dice/IoU.

Повторная загрузка:

- Перед первым test проверена на calibration: score/map max abs diff = 0.0.
- Smoke с реальным model.pt и изображением прошёл на CPU без сетевой загрузки.
- Проверка CLI из нового процесса выявила небольшой TF32-дрейф; исправлен loader:
  восстанавливаются seed/CPU threads/deterministic/TF32-настройки из артефакта.
- Отдельно перепроверены scores всех 83 test-файлов из нового процесса на CUDA:
  максимальная разница с CSV 3.55e-15; SHA256 всех 357 исходных файлов совпали.
  Результат PASSED в fresh_process_check.json, исходные CSV/метрики не менялись.

Команды:

```text
uv sync --locked --extra dev --extra ml
python -m qualitypilot.cli --config configs/patchcore.yaml patchcore
python -m qualitypilot.cli inspect IMAGE --model RUN/model.pt --device cuda
uv lock --check
uv pip check --python .venv/Scripts/python.exe
ruff check .
ruff format --check .
pytest -q  # 62 passed, 1 upstream warning; real smoke enabled, HF_HUB_OFFLINE=1
```

Отчёты: reports/sprint2_report.md и полный локальный run. README и зависимости обновлены.
Streamlit не реализовывался согласно scope Sprint 2.

Ограничения:

- Post-hoc сравнение фиксированной конфигурации, baseline был известен заранее;
  test не использовался для тюнинга. Параметры после просмотра результатов не менялись.
- Только bottle и 83 test-изображения; AUROC 1.0 не означает идеальный порог
  или промышленную надёжность. FPR одного image-порога 20%.
- 42 normal calibration, нет независимой anomalous validation и recall/FPR гарантий.
- Pixel metrics на 224×224; визуализации растянуты обратно до 900×900.
- Сторонний источник данных, проверка только точных дубликатов.
- Один upstream DeprecationWarning Anomalib про embeddings; проверки проходят.
- Linux/удалённый CI ещё не выполнялись. GitHub-публикация не выполнялась.

Дальше: спринт 3, Streamlit Inspection/Evaluation/About поверх общего Inspector;
не менять текущий исследовательский протокол ради улучшения test-метрик.


## Аудит Sprint 1–2 — 22 октября 2026

Выполнен по отдельному заданию перед Sprint 3. История выше сохранена; подробный
отчёт: [reports/sprint1_2_audit.md](reports/sprint1_2_audit.md).

Исправления:

- Подтверждён P1: `data/` в .gitignore исключало src/qualitypilot/data.
  Правило заменено на `/data/`; код и тесты доступны Git, данные/веса исключены.
- После этого Ruff впервые проверил пакет data: исправлены E501 и форматирование
  трёх файлов. AST всех 32 Python-файлов совпал с первоначальным снимком;
  функциональная ML-логика, гиперпараметры, split и пороги не менялись.
- Зависимости не обновлялись, основное окружение не пересоздавалось.

Проверка поставки:

- Временная копия только Git-доступных файлов, включая untracked; данные, модели
  и окружения не копировались. На момент снимка 50 файлов, все 25 модулей и 7
  файлов тестов. Финальная сборка wheel установлена не editable в отдельный
  Python 3.11.14 venv с baseline/dev без torch.
- Импорт qualitypilot.data.pipeline из site-packages и CLI --help работают.
  Хеши установленных модулей совпали с копией: 25/25.
- Чистый wheel: 58 passed, 4 skipped (ML extra намеренно отсутствует), 0 warnings.

Проверка моделей без переобучения, новым процессом:

- Исторические baseline bottle-baseline и PatchCore
  bottle-patchcore-20261021T103450Z-5eddf169 загружены без fit.
- Manifest совпал в обоих runs, train/calibration/test = 167/42/83, пересечений
  и точных RGB-дубликатов нет. SHA256 всех 357 файлов provenance, включая masks,
  совпали. Paths, порядок и labels 83 test совпали у обеих моделей.
- Baseline score max abs diff 5.551115123125783e-17 (rtol=0, atol=1e-12).
- PatchCore CUDA score max abs diff 3.552713678800501e-15
  (rtol=1e-6, atol=1e-6). CUDA реально доступна, использована RTX 3050.
- Decisions и binary predictions: 0 несовпадений. Независимо пересчитанные image
  metrics, pixel AUROC/micro Dice/IoU 224×224: разница 0.0, допуск 1e-6.
- Модели, параметры и thresholds не менялись. Новые CSV/maps/JSON сохранены
  отдельно; исторические CSV и метрики не переписывались.

Проверки:

- Основной ML pytest: 61 passed, 1 skipped, 1 warning. Пропущен только real smoke
  без переменных окружения; затем он выполнен отдельно с локальными файлами и
  HF_HUB_OFFLINE=1: 1 passed, 0 skipped, 0 warnings.
- Ruff check/format: PASS, 32 файла; uv lock --check и uv pip check: PASS,
  76 совместимых пакетов. Состав/версии основного окружения не изменились.
- 405 исторических файлов данных/артефактов/runs и 4 исторических Markdown-отчёта
  проверены по SHA256 после аудита: изменений нет.
- DeprecationWarning Anomalib 2.2.0 изучен: apply_defaults декоратора добавляет
  embeddings=None и затем предупреждает о наличии имени. Безопасной необходимой
  правки вызывающего кода нет; предупреждение не подавлено, алгоритм не менялся.

Все новые evidence-артефакты:
`reports/runs/sprint1-2-audit-20261021T215712Z-3db46b83/`.
Внутри audit_driver.py, recheck_delivery.py, delivery/source-final, временный venv,
финальный wheel, логи/JUnit, source_hashes, reproducibility и preservation JSON,
новые baseline/patchcore CSV и карты. Имя папки использует UTC; локальная дата — 22 октября.

Вердикт: подтверждена готовность к разработке Sprint 3 поверх существующего
Inspector. Sprint 3 не начинался; коммит/push/публикация не выполнялись.
Остаются P2 upstream warning и оговорка scope latency metadata, а также прежние
ограничения: сторонняя копия, только bottle/83 test, нет Linux/удалённого CI,
near-duplicates и производственной валидации. AUROC 1.0 не отменяет 4 FP и Dice 0.5604.


## Спринт 3 — Streamlit, 22 октября 2026

Статус: локальный результат реализован и проверен, готов к независимому ревью.
Отчёт: [reports/sprint3_report.md](reports/sprint3_report.md).

Сделано:

- Тонкий app.py и пакет qualitypilot.ui: read-only artifacts, in-memory upload,
  session state и виджеты отдельно. Русские разделы: Проверка изображения,
  Результаты эксперимента, О проекте.
- Использован исходный Inspector/Prediction/Thresholds. Только доверенные локальные
  веса; UI ничего не обучает/калибрует/скачивает и не принимает checkpoints.
- Установлен существующий extra ui Streamlit 1.50.0 через uv с dev/ml/ui;
  torch 2.8.0+cu128 и Anomalib 2.2.0 сохранены, lock и pyproject не менялись.
- configs/ui.yaml: пути относительно проекта, default COMPLETE PatchCore run
  bottle-patchcore-20261021T103450Z-5eddf169. FAILED/incomplete/audit не выбираются.
- Один JPG/PNG upload до 10 MiB/20 млн pixels с фактическим decode/RGB, или
  отсортированный известный test-пример с явной маркировкой. Обработка в памяти.
- Ленивая загрузка одной модели в сессию; отдельное время загрузки и predict.
  Смена model/run/device освобождает старый объект и сбрасывает prediction;
  смена изображения сбрасывает старый результат.
- PatchCore heatmap/overlay/mask в исходных координатах, фиксированная inferno
  шкала и saved pixel threshold; GT только доступного локального примера.
  Baseline честно сообщает об отсутствии локализации.
- Experimental Thresholds — отдельный объект, finite accept≤reject, допускается
  пустая REVIEW-зона. Original/experimental решения и reset; никаких записей
  в model/файлы. Alpha/temporary thresholds не запускают inference.
- Saved reports читаются без модели/ML extra: проверяются manifest SHA, test paths,
  порядок/labels CSV и согласованность comparison/metrics. Есть confusion matrices,
  single/two thresholds, REVIEW, N/A reasons, TP/FP/FN/TN и готовые иллюстрации.
- Обработаны missing model/config/reports/dataset, corrupt upload, ML extra/CUDA;
  available страницы остаются работоспособны. Explicit CUDA не меняется на CPU молча.

Проверки:

- Полный pytest: 74 passed, 1 skipped, 1 warning (Anomalib upstream).
  Opt-in artifact smoke пропущен обычным запуском и отдельно успешно выполнен:
  1 passed, 0 skipped, 0 warnings, HF_HUB_OFFLINE=1.
- UI unit/AppTest Streamlit 1.50.0: 13 passed, включая reset/state/caching/errors.
  AppTest file_uploader не подменялся несуществующим API; decode протестирован
  отдельно, настоящий JPG/PNG upload проверен в браузере.
- Отдельное Python 3.11 окружение с lock-export dev/ui без torch: те же 13 UI tests
  прошли. Новый процесс импорта qualitypilot.ui.app не импортирует torch.
- Реальный offline AppTest для frozen Baseline/PatchCore на good/000 и
  broken_large/000: score/decision/map совпали с прямым Inspector, max abs diff=0.0.
  Только 2 UI loads/4 predictions. Alpha, temporary thresholds, reset и переход
  к reports не увеличили счётчики, сохранённые пороги и карты не изменились.
- Реальный браузер: normal example ACCEPT, PNG upload PatchCore REJECT
  (19.767408), JPEG upload REJECT (19.731306), corrupt PNG — русская ошибка без
  traceback/старого prediction; просмотрены overlay/mask/legend и saved reports.
- Ruff check/format: PASS, 40 Python-файлов. uv lock --check: PASS;
  uv pip check: 96 совместимых пакетов. Старый CLI --help и тесты работают.
- SHA256 52 исторических файлов моделей/results/reports: без изменений.
  SHA256 всех 357 файлов provenance совпали. 25 модулей ядра, ML configs,
  pyproject и uv.lock совпали с проверенным снимком Sprint 1–2.

Evidence: `reports/runs/sprint3-ui-20261021T230241Z-111dbe48/` — real_ui_smoke JSON/script, логи/JUnit,
no-ml venv/export, screenshots в ignored папке, preservation.json. Снимки с данными
и JPEG для теста формата не коммитятся. README содержит установку и запуск:
`python -m streamlit run app.py`, localhost 127.0.0.1:8501.

Ограничения: локальный single-image режим, только bottle/83 test, normal-only
calibration без recall/FPR гарантий, умеренная локализация, сторонняя копия данных.
Стандартные browser upload подписи Streamlit остаются английскими. Linux/remote CI
и кросс-браузерная полнота не проверены. Сессии исчезают при перезапуске сервера;
после редактирования кода сервер нужно перезапустить. Промышленная сертификация
не заявляется. Нет БД/auth/cloud/batch. Коммит/push/публикация не выполнялись.

### 22 октября 2026 — замечания ревью Sprint 3 закрыты

Валидация CSV и структуры отображаемых JSON теперь выполняется до рендеринга.
Синтаксические ошибки baseline YAML преобразуются в ValueError с файлом и позицией.
Добавлены 40 регрессионных AppTest-сценариев повреждённых файлов; проверяются
понятная ошибка без traceback и доступность остальных разделов. Отдельный CI job
dev/ui для Ubuntu/Windows требует установленный Streamlit, отсутствие torch и
нулевое число пропущенных UI-тестов по JUnit.

Проверки: полный pytest 114 passed/1 skipped (opt-in smoke)/1 прежний warning;
UI 53 passed, отдельно без torch 53 passed/0 skipped; Ruff check/format PASS.
Настоящие сохранённые отчёты на 83 test-строках открыты AppTest без загрузки весов.
SHA256 49 файлов моделей, ML-конфигураций и исторических результатов совпали.
Обновлены reports/sprint3_review.md и sprint3_report.md; evidence в ignored
reports/runs/sprint3-review-fixes-20261022/. Нового обучения, изменения порогов,
коммита/push или удалённого запуска Actions не было.


### 22 октября 2026 — Sprint 4: воспроизводимость и поставка

Перестроен README, добавлены docs/reproduce.md, demo.md и interview.md,
обновлён docs/next-sprints.md. Удалены Markdown-ссылки на непубличные ignored
вложения из старых отчётов, численные результаты сохранены. Production-код,
тесты, конфиги ML, зависимости и CI workflow не менялись.

Создана уникальная ignored папка reports/runs/sprint4-20261022T093225Z-a48ee46e/.
В чистой копии будущих Git-файлов установлены отдельные Python 3.11 среды
baseline/dev/ui (без torch) и dev/ml/ui по существующему lock. Данные bottle
скачаны заново, 357 файлов совпали с прежней provenance. Полный новый baseline
и PatchCore pipeline завершён COMPLETE: bottle-patchcore-20261022T093802Z-aef5e02e.
Manifest, calibration/test scores, пороги, decisions и quality metrics совпали;
все 92 state_dict тензора PatchCore идентичны. Новые времена приведены отдельно
в reports/sprint4_report.md. Это повтор известного test, не независимый benchmark.

Финальные проверки: полный pytest 114 passed/1 opt-in skip/1 прежний warning;
реальный offline smoke отдельно 1 passed; UI без ML 53 passed/0 skipped;
локальный baseline CI dev-only 66 passed/5 ожидаемых skips. Ruff check/format,
lock и dependency check, CLI/API, wheel, документированные команды и браузер
normal/defective/PNG/JPG/corrupt/temporary/saved reports — PASS. Offline
Inspector/AppTest работали с пустым HF-кэшем и запрещёнными сокетами: 0 попыток сети.
Состав будущей поставки и ссылки проверены; SHA256 52 исторических файлов совпали.

Linux/WSL отсутствует, удалённый Actions не запускался. Исторические артефакты,
ML-конфиги, lock, LICENSE и основная среда сохранены. Подробности, диагностика
первого cold AppTest timeout и ограничения — reports/sprint4_report.md.
Готово к финальному независимому ревью; коммит/push/публикация не выполнялись.

### 22 октября 2026 — подготовка финальной публикации

Независимое ревью четырёх спринтов завершено; пользователь разрешил обычный
push в существующий origin/main. Перед работой fetch подтвердил совпадение
локальной main и origin/main, исторические SHA256 сохранены. README дополнен
точным git clone, ссылкой/badge checks.yml и актуальным статусом разработки;
добавлен docs/resume.md, обновлены текущие следующие шаги.

Перед первым push выполняются итоговые pytest, Ruff check/format, lock/dependency
check и проверка полного staged-состава. Удалённый CI пока не объявлен прошедшим:
нужно проверить четыре job-варианта для отправленного SHA и свежий checkout.
Точные итоговые SHA/URL и фактические результаты будут приведены в сообщении
о завершении. Публикация данных/весов и облачный deploy не входят в задачу.

Итоговая локальная проверка перед публикацией: pytest **114 passed, 1 skipped,
1 warning**. Skip — opt-in smoke без переменных; warning — прежний upstream
Anomalib deprecation. Ruff check/format (41 файл), uv lock --check и совместимость
96 установленных пакетов — PASS. ML-код не менялся, обучение не повторялось.
