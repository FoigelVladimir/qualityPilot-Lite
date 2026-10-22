# Sprint 3 — локальный Streamlit UI

Дата: 22 октября 2026. **Реализован и проверен локально; готов к независимому ревью.**

Evidence: sprint3-ui-20261021T230241Z-111dbe48 (`runs/sprint3-ui-20261021T230241Z-111dbe48/`, только локально; исключено из Git).
Timestamp имени — UTC. Датасетные изображения, screenshots, тестовые загрузки и
временные окружения находятся только в ignored run-папке.

## Что реализовано

Тонкий корневой `app.py` вызывает `qualitypilot.ui.app.main`. Пакет UI разделён:

- artifacts.py — конфигурация, фильтрация COMPLETE runs, проверка manifest/CSV/metrics;
- inspection.py — декодирование upload, лимиты, heatmap/overlay/mask и сообщения ошибок;
- state.py — одна ленивая модель на session state, очистка, cache и временные Thresholds;
- app.py — виджеты трёх русских разделов.

Используются существующие Inspector, Prediction и Thresholds; публичные форматы и
API моделей/CLI не менялись. Models/decision/inference/evaluation/experiments сохранены.
Нет БД, авторизации, batch processing, облачного deploy, автоматического download,
обучения/калибровки или загрузки checkpoints через браузер.

Установлен существующий extra ui: **Streamlit 1.50.0**, через uv с сохранением
`dev,ml,ui`. Torch остался **2.8.0+cu128**, Anomalib **2.2.0**.
Зависимости ML и uv.lock не обновлялись произвольно; `uv lock --check` проходит.

## Запуск

Из корня репозитория, в Python 3.11 окружении:

```text
uv sync --locked --extra dev --extra ml --extra ui
python -m streamlit run app.py
```

Без активации venv в PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Linux: `.venv/bin/python -m streamlit run app.py`.
Приложение локально: http://127.0.0.1:8501; address 127.0.0.1,
maxUploadSize 10 MB, gatherUsageStats=false. После изменения исходников сервер
нужно перезапустить: fileWatcherType=none. Linux фактически не проверялся.

configs/ui.yaml содержит baseline config, runs_root и выбранный run
`bottle-patchcore-20261021T103450Z-5eddf169`. Пути относительно проекта.
Допускаются только COMPLETE PatchCore runs с ожидаемым набором отчётов;
FAILED, incomplete и audit-папки не выбираются. model.pt намеренно не является
условием доступности saved reports: отчёты можно читать без весов.

## Проверка изображения

Выбор baseline/PatchCore и устройства auto/cpu/cuda; показано фактическое устройство.
Explicit unavailable CUDA показывает ошибку и предложение cpu/auto, без silent fallback.
Источник — одиночный JPG/PNG upload либо отсортированный test-пример выбранного run.
Примеры помечены как известный test, не новая независимая оценка.

Проверяются фактический формат, декодирование, 10 MiB и 20 000 000 pixels до загрузки
полного изображения. RGB сохраняет геометрию; EXIF rotation/crop не выполняются,
как и в принятом preprocessing Inspector. Upload обрабатывается в памяти.

Inference — только кнопкой. Загрузка модели измеряется отдельно и не входит в
показанную latency Inspector.predict. Это одиночное измерение, оно может включать
первый прогрев CUDA; оно не подменяет исторические warmup benchmark timings.
Одна активная модель на сессию; при смене model/device/run старый объект удаляется,
при CUDA освобождается доступный cache. Изображение/model/device/run сбрасывают старый prediction.

PatchCore показывает карту и overlay в исходном H×W, binary mask по сохранённому
pixel threshold, легенду inferno с общей шкалой 0..max(1.5×threshold,1e-6).
Alpha меняет только overlay. Ground truth показывается только для локального
примера с доступной разметкой; upload явно не имеет GT. Baseline честно сообщает
отсутствие локализации.

Экспериментальные пороги конечны и упорядочены accept ≤ reject. Новый объект
Thresholds пересчитывает рекомендацию готового score; cached Inspector, saved pixel
threshold, карты, checkpoint и исторические отчёты не изменяются. Равенство порогов
допустимо, REVIEW пуст. Есть исходная/экспериментальная рекомендация, предупреждение
о настройке по test и reset. При смене model/run временные значения сбрасываются.

## Сохранённые результаты и About

До сравнительной таблицы проверяются:

- контрольная сумма manifest по сохранённым строкам Sample;
- допустимые splits/labels, отсутствие дубликатов manifest;
- одинаковые path/порядок/labels обеих таблиц относительно test manifest;
- согласованность manifest SHA в summary/experiment/comparison;
- совпадение comparison с отдельными metrics.json.

При несоответствии показывается ошибка вместо таблицы. Модель не загружается и
полный experiment не выполняется. Метрики image AUROC/AP, precision/recall/F1/FPR,
latency и pixel metrics показаны как сохранены, N/A сопровождается причиной.
Показаны single/two-threshold ошибки, confusion matrices, REVIEW fraction,
условия latency/устройство и дополнительный CPU benchmark PatchCore.

Есть строки TP/FP/FN/TN, локальные оригиналы и доступные готовые иллюстрации.
Отсутствующие случаи (у PatchCore FN=0) не заменяются фиктивными изображениями.
REVIEW явно обозначен как нагрузка на ручную проверку, не исправленная ошибка.

About описывает baseline, pretrained ResNet18/memory bank, split 167/42/83,
нормальную калибровку и ACCEPT/REVIEW/REJECT. Указаны CC BY-NC-SA 4.0,
сторонняя копия/commit, ограничение bottle/83 test, moderate Dice и отсутствие
промышленной сертификации. Score — не вероятность.

## Проверки

### Модульные и AppTest

Использована фактически установленная Streamlit **1.50.0**, её локальные сигнатуры
и `streamlit.testing.v1.AppTest`. File uploader не эмулируется через недоступный
AppTest API: decode_upload тестируется отдельно, реальный upload — в браузере.

13 UI-проверок покрывают run discovery/фильтрацию, отсутствие весов, mismatch
results/manifest, JPG/PNG/RGB, повреждённый/неподдерживаемый файл и оба лимита,
finite/equal/invalid временные пороги, reset, отсутствие мутации карты/Thresholds,
navigation, baseline без карты, cached model, image/model/device changes,
ошибку CUDA и работу других разделов после ошибки.

| Серия | Passed | Skipped | Warnings |
|---|---:|---:|---:|
| Полный pytest, dev/ml/ui | 74 | 1 | 1 |
| Отдельный реальный integration smoke | 1 | 0 | 0 |
| UI-тесты в отдельном окружении без torch | 13 | 0 | 0 |

Пропущен в обычном полном pytest только opt-in real artifact smoke без ENV.
Он отдельно выполнен с QUALITYPILOT_SMOKE_MODEL/IMAGE и HF_HUB_OFFLINE=1.
Один warning — известный upstream DeprecationWarning Anomalib embeddings,
не скрывался и не исправлялся изменением алгоритма.
Повторные строки не суммируются как уникальные тесты.

В отдельном Python 3.11 venv установлен lock-export только dev/ui, torch отсутствует.
Все 13 UI-тестов прошли; страницы About/reports не требуют ML extra. Отдельный
новый процесс импорта UI также подтвердил отсутствие torch в sys.modules.

Ruff check/format проходят (40 Python-файлов). uv lock --check и uv pip check
проходят; установленные 96 пакетов совместимы. CLI и его старые тесты проходят.

### Реальный offline UI smoke

real_ui_smoke.py запускал AppTest с реальными frozen моделями, без обучения:

| Модель | Пример | Score | Решение | Diff с прямым Inspector |
|---|---|---:|---|---:|
| Baseline CPU | good/000.png | 0.43224340106043047 | ACCEPT | 0.0 |
| Baseline CPU | broken_large/000.png | 0.4913530368406759 | ACCEPT | 0.0 |
| PatchCore CUDA | good/000.png | 8.04456615447998 | ACCEPT | 0.0 |
| PatchCore CUDA | broken_large/000.png | 19.76740837097168 | REJECT | 0.0 |

Авторитетный источник — `real_ui_smoke.json`: точные значения и decisions всех
четырёх случаев. PatchCore map 900×900 совпала с прямым Inspector с max abs diff=0.0.
Инструментированный счётчик UI: **2 model loads и 4 predictions**. После alpha,
temporary accept/reject и reset счётчики не увеличились; score/map/saved thresholds
остались прежними. Переход к reports также не загрузил модель и не вызвал predict.

### Отдельный browser smoke

Приложение реально открыто в in-app браузере по localhost. Это отдельная проверка,
не подмена браузерного smoke AppTest:

- локальный normal пример: PatchCore CUDA, score 8.044566, ACCEPT;
- настоящий PNG upload broken_large/000.png: score 19.767408, REJECT;
- JPEG upload, созданный из того же примера для теста формата: score 19.731306,
  REJECT. JPEG сжат с потерями, поэтому от него не требуется совпадение с PNG;
- при смене файла старый результат исчезает до новой проверки;
- corrupt.png: читаемая русская ошибка, без traceback/старого prediction;
- просмотрены карта, overlay, predicted mask, фиксированная легенда и отсутствие GT у upload;
- открыта таблица исторических результатов и confusion matrices.

Screenshots в `screenshots/`: png-inspection.png, overlay.png, jpg-inspection.png,
corrupt-upload.png, reports.png и reports-final.png. Снимки с датасетом не попадают в Git.
Системные подписи стандартного file uploader («Browse files», drag-and-drop)
остаются английскими; прикладные разделы, инструкции и ошибки — русские.

## Сохранность и ограничения

Перед началом рассчитаны SHA256 исторических моделей, run-результатов и Markdown
отчётов в `historical_before.json`. Итоговая сверка preservation.json: **52 файла
без изменений**, также совпали все **357 файлов provenance** датасета.
Все **25 существовавших модулей ядра**, ML-конфигурации, pyproject.toml и uv.lock
совпадают с проверенным снимком Sprint 1–2. Новые изменения ограничены UI,
его конфигурацией, тестовыми фикстурами и документацией.
Модели и исторические эксперименты не перезаписывались; manual thresholds нигде
не сохраняются. Никакие реальные ML-параметры не подбирались по test.

Остаются ограничения предыдущих спринтов: только bottle/83 test, сторонняя копия,
42 normal calibration без гарантий recall/FPR, near-duplicates не проверены,
умеренная локализация (Dice≈0.5604), нет промышленной сертификации или validation
других камер/категорий. Нет облачной публикации, БД, авторизации и batch flow.
Сессии не сохраняются после перезапуска сервера. Нет фактической проверки Linux/CI
и exhaustive кросс-браузерной/мобильной проверки. Windows и текущий браузер проверены.

README обновлён, progress сохраняет историю. Коммит/push/публикация не выполнялись.
Локальный результат готов для независимого ревью Sprint 3.

## Исправления после независимого ревью, 22 октября 2026

Три замечания из `sprint3_review.md` закрыты: CSV/отображаемые JSON проверяются
до рендеринга, синтаксическая ошибка baseline YAML становится понятной ошибкой
с именем файла, UI-тесты включены в отдельный CI job Ubuntu/Windows без ML extra.
Job проверяет наличие Streamlit и отсутствие пропущенных тестов по JUnit.
Добавлены 40 регрессионных AppTest-сценариев повреждённых файлов с проверкой
отсутствия traceback и доступности остальных разделов. Исправлена только граница
чтения baseline YAML; ML-код, параметры моделей и сохранённые пороги не менялись.

Повторно: UI **53 passed**; отдельное окружение без torch **53 passed, 0 skipped**;
весь набор **114 passed, 1 skipped** (opt-in ML smoke), один прежний warning Anomalib;
Ruff check/format прошли. Настоящие сохранённые отчёты открыты AppTest без модели.
SHA256 49 файлов моделей/ML-конфигураций/исторических результатов совпали со снимком.
Подробности и границы проверки — в `sprint3_review.md`; evidence — ignored каталог
`reports/runs/sprint3-review-fixes-20261022/`. GitHub Actions удалённо ещё не запускался.
