# Sprint 4 — воспроизводимость и подготовка к финальному ревью

Дата: 22 октября 2026, Asia/Omsk. **Выполнен локально на Windows; готов к
финальному независимому ревью. Коммит, push и публикация не выполнялись.**

Evidence — уникальная ignored папка
`reports/runs/sprint4-20261022T093225Z-a48ee46e/` (далее `E`). Все временные
окружения, скачанные данные, модели, изображения, screenshots и подробные логи
находятся только там. Старые run-папки не изменялись.

## Объём изменений

README перестроен для читателя GitHub: возможности, таблица сопоставимых измерений
Sprint 2, быстрый запуск без артефактов, явное получение данных и полный pipeline,
архитектура Mermaid/Python API, тесты и ограничения. Историческое время Sprint 1
с одним прогревом отделено от matched benchmark Sprint 2 с тремя прогревами.
Добавлены [протокол воспроизведения](../docs/reproduce.md),
[демонстрация на 3–5 минут](../docs/demo.md) и
[объяснение для собеседования](../docs/interview.md).
Обновлены progress и [следующие шаги](../docs/next-sprints.md).

В docs/data.md команды используют uv без необходимости активации окружения;
в docs/dependencies.md полный sync сохраняет dev/ml/ui. В трёх старых Markdown
отчётах ссылки на ignored evidence/PNG заменены текстовыми локальными путями:
на GitHub они не должны выглядеть как доступные публичные вложения. Численные
исторические результаты не редактировались. Production-код, тесты, CI workflow,
ML-конфигурации, pyproject.toml и uv.lock в Sprint 4 не менялись.

## Исходное состояние и комплектность

`E/initial_state.json`: исходный Git HEAD, status и SHA256 всех 64 исходных файлов
будущей поставки, включая untracked. Git содержит исходный LICENSE-коммит;
реализация пока преимущественно untracked. `git diff` сам по себе её не охватывает.
`E/historical_before.json`: SHA256 52 исторических файлов — модели, run-артефакты,
ML-конфигурации, pyproject.toml, uv.lock и LICENSE. Основной .venv не пересоздавался.

Чистая копия `E/delivery/` создана из
`git ls-files --cached --others --exclude-standard -z`, без данных/моделей/окружений.
Исходники импортировались именно из этой копии. После проверок документация в
копии синхронизирована с итоговой поставкой; SHA256 файлов поставки сверены.
Список, проверка ссылок/секретов/типов файлов и состав wheel сохранены в
`E/delivery-audit.json` и `E/delivery-manifest.json`.

В итоговой поставке **68 файлов**: исходники, все тесты и конфиги, app.py, .streamlit, .github, документация,
LICENSE и lock. Датасет, производные изображения, веса, окружения, кэши, evidence
и персональные абсолютные пути исключены. Существующий LICENSE сохранён побайтно:
код MIT; MVTec AD и изображения имеют отдельные ограничения CC BY-NC-SA 4.0.
Wheel успешно собран из чистой копии; проверена комплектность всех **31 Python-модуля**.
Это проверка checkout-поставки: app.py/configs/docs поставляются в репозитории,
не объявляются отдельным самодостаточным GUI wheel.

## Чистые установки

Python **3.11.14**, uv **0.9.5**, существующий lock, без обновления зависимостей.
Каждая среда указана через отдельный UV_PROJECT_ENVIRONMENT; основная среда проекта
не использовалась для выполнения pipeline чистой копии.

| Среда | Установка и фактическая проверка |
|---|---|
| `E/env-baseline` | dev/ui, 55 пакетов, torch отсутствует; 111 passed / 4 ожидаемых ML skips |
| `E/env-ml` | dev/ml/ui, 96 пакетов; Torch 2.8.0+cu128, Anomalib 2.2.0, CUDA RTX 3050 8 GB |
| `E/env-ci-baseline` | дополнительный локальный эквивалент baseline job, только dev: 66 passed / 5 ожидаемых skips (4 ML и модуль AppTest без UI extra) |

Установка брала wheel из доступного package cache; это чистые окружения, а не
проверка скачивания каждого wheel с нуля. Установка uv==0.9.5 через pip отдельно
проверена в временной bootstrap-среде. Python 3.11 уже был доступен. Команда
`uv python install 3.11` вернула предупреждение о существующем неуправляемом
python3.11 alias; alias не заменялся, sync/все запуски использовали 3.11.14.

В отдельной пустой копии `E/empty-ui/` AppTest подтвердил предупреждение об
отсутствии baseline-весов, инструкции вместо пустых результатов и доступный About.
После создания run та же среда без Torch прочитала настоящие saved reports,
выполнила реальный baseline inference и показала понятное сообщение о необходимости
ML extra при попытке PatchCore. Evidence: `clean-baseline-ui.json/log`,
`real-no-ml-reports.json/log`, `install-baseline.log`, `install-ml.log`.

Первый вспомогательный cold AppTest не уложился в стандартные 3 секунды. Диагностика
сохранена в `clean-baseline-ui-initial-timeout.log`; повтор на пустой копии с
явным timeout 30 s прошёл. Production-код и timeout штатных тестов не менялись;
весь стандартный AppTest-набор затем прошёл. Это не скрытый пропуск проверки.

## Новый полный эксперимент с нуля

В `E/delivery/` выполнены документированные команды `download-mirror`, `prepare`,
`baseline`, `patchcore`. Копии YAML находятся в `E/delivery/configs/`, их относительные
пути указывают только на новые артефакты чистой копии:

- baseline: `E/delivery/artifacts/bottle-baseline.joblib`,
  reference-run `E/delivery/reports/runs/bottle-baseline/`;
- PatchCore: `E/delivery/reports/runs/bottle-patchcore-20261022T093802Z-aef5e02e/`;
- PatchCore baseline_config/reference_run разрешены к этому **новому** baseline,
  а не к историческому оригиналу.

Данные скачаны заново существующим явным mirror-механизмом, только bottle:
357 файлов, 156 570 806 байт, fixed commit
`c75b39616f84db43677bcc8228caaafaf5096d7f`. Состав и хеши совпали с исторической
provenance; маски/структура/дубликаты проверены pipeline. Совпадение с официальным
архивом не заявляется. Новый pretrained HF-кэш был пустым; ImageNet ResNet18
получен по сети в `E/pretrained-cache/`, старый пользовательский кэш не требовался.

Baseline обучен заново. PatchCore заново извлёк train-признаки и подготовил coreset:
167 train normal, 130 928 патчей, memory bank **1309×384**, 2 010 624 байт.
Calibration — 42 normal, official test — прежние 83 изображения. Прежние seed,
признаки, гиперпараметры и квантили сохранены; вариантов настроек не перебирали.
Модели сохранены и загружены вновь. PatchCore reload проверен на calibration до
первого test: score/map max abs diff **0**, статус PASSED. Полная оценка обеих
моделей, pixel metrics, confusion matrices, comparison и примеры завершены:
статус **COMPLETE**, не только успешный inference. Скрипт `E/pipeline.py` и
`prepare.log`, `baseline.log`, `patchcore.log` фиксируют последовательность.

### Сравнение с историческим Sprint 2

`E/reproduction-comparison.json` и `E/compare.py` фиксируют результаты:

- manifest совпал побайтно, SHA256
  `82dfcf4d1a2907ddb1a6d112c5b1add85be1e931f2b7a679c779f7e6b431dce1`;
- calibration paths/scores обеих моделей совпали, max abs score diff **0**;
- test paths/labels, все decisions и binary predictions совпали; max abs score
  diff обеих моделей **0**;
- image/pixel quality metrics, single/two-threshold показатели совпали;
- все **92 тензора state_dict** PatchCore совпали по значениям (`torch.equal`);
- checkpoint SHA256 различается: metadata содержит новые пути, время подготовки,
  provenance скачивания и окружение. Равенство checkpoint-байтов не требовалось.

| Показатель нового повторяющего запуска | Baseline CPU | PatchCore CUDA |
|---|---:|---:|
| accept / reject | 0.5005465296 / 0.5561937682 | 8.626294136 / 9.292860985 |
| Image AUROC | 0.624603 | 1.000000 |
| F1 | 0.672897 | 0.969231 |
| Confusion matrix | `[[12,8],[27,36]]` | `[[16,4],[0,63]]` |
| Median / p95 latency, ms | 22.1553 / 23.2007 | 27.6271 / 30.5481 |
| Pixel AUROC / Dice / IoU | N/A | 0.976049 / 0.560382 / 0.389257 |

Новый pixel threshold **8.860112190246582**, совпал. Три прогрева, batch 1,
83 измерения, CUDA synchronized; дополнительно PatchCore CPU median/p95
**49.1735 / 55.5768 ms**. Эти времена относятся только к повтору Sprint 4.
Историческая таблица README оставлена на сопоставимых измерениях Sprint 2.

Подготовка PatchCore: **45.80 s**, исторически **16.11 s**. В этот раз в scope
вошли получение pretrained-весов в пустой кэш и cold initialization. Latency также
варьируется с состоянием системы и I/O; отдельного контролируемого исследования
причин временных различий не проводилось. Это не новое улучшение алгоритма.
Test уже известен: повтор проверяет воспроизводимость, **не независимый benchmark**.

## Offline inference и UI

`E/real_ui.py` выполнялся в чистой ML-среде с `HF_HUB_OFFLINE=1`, отдельным пустым
HF-кэшем и запрещёнными socket connect/create_connection. Сетевых попыток **0**.
Новые baseline/PatchCore загружены из сохранённых артефактов; реальные normal и
defective предсказания через Inspector и AppTest совпали, map difference **0**.
Изменение alpha/временного reject и просмотр reports/About не запускали новый
inference. Evidence: `real-ui-offline.json/log`. Отдельный opt-in smoke на CPU
с новым model.pt и HF_HUB_OFFLINE=1: **1 passed, 0 skipped, 0 warnings**.

Браузерная проверка нового приложения из чистой ML-копии, localhost 8501:

- PatchCore good/000.png → **8.044566, ACCEPT**, broken_large/000.png →
  **19.767408, REJECT**, фактическое устройство CUDA;
- baseline на том же дефекте → **0.491353, ACCEPT**: известный FN сохранён;
- настоящий PNG upload → **19.767408, REJECT**; JPG quality 95 →
  **19.731306, REJECT**. JPEG меняет пиксели, поэтому небольшой сдвиг score ожидаем;
- corrupt.png → русская ошибка декодирования, без traceback и прежнего prediction;
- временный reject=30 → REVIEW при прежних score/исходном REJECT; сброс выполнен;
- просмотрены heatmap/overlay/predicted mask, GT у local-примера, saved reports
  с обеими confusion matrices и About. При смене изображения старый score исчез.

`E/browser-checks.json`, `E/screenshots/`, `E/browser-inputs/` и `browser-server.log`
содержат локальные evidence. Ни одно изображение не включено в Git-поставку.
Все test-примеры — известные демонстрационные данные; upload не даёт неизвестную GT.
Одиночные времена UI не использованы в таблице benchmark.

## Финальные проверки

| Проверка | Результат / evidence внутри E |
|---|---|
| Полный dev/ml/ui pytest | **114 passed, 1 skipped, 1 warning**, final-full-tests.xml/log |
| Отдельный реальный offline smoke | **1 passed**, integration.xml/log |
| UI без ML | **53 passed, 0 skipped**, final-ui-no-ml.xml/log |
| Локальный baseline CI без ML/UI | **66 passed, 5 ожидаемых skips**, ci-baseline-tests.xml/log |
| Ruff check / format | PASS, 41 Python-файл, ruff.log / format.log |
| uv lock --check | PASS, 113 resolved, lock.log |
| Dependency check | PASS, 96 ML и 55 baseline/UI пакетов, dependency-*.log |
| CLI --help / baseline inspect / PatchCore inspect | PASS, cli-*.log |
| README Python API и PowerShell smoke | PASS, documented-commands.json и соответствующие логи |
| Сборка wheel / состав checkout | PASS, wheel.log / delivery-audit.json |
| Браузер | PASS в текущем встроенном браузере, browser-checks.json |
| SHA256 исторических файлов | **52 / 52 совпали**, preservation.json |

Skip полного набора — opt-in ML smoke без переменных; он выполнен отдельно.
Единственный pytest warning — прежний upstream Anomalib embeddings deprecation.
Во время получения pretrained-весов HF сообщил об отсутствии optional hf_xet
(успешный HTTP fallback) и недоступности Windows symlink cache; это не ошибки
обучения. Никакие зависимости ради предупреждений не обновлялись.

## Платформы, границы и готовность

Фактически проверены Windows/PowerShell, Python 3.11.14, CPU и CUDA RTX 3050.
WSL/Linux не установлен: Linux-команды и удалённый GitHub Actions **NOT RUN**.
CI job уже настроен для Ubuntu/Windows, baseline/UI на синтетических фикстурах;
локальные эквиваленты не скачивали датасет/pretrained-веса. Его pip-установка
закрепляет прямые версии, но не идентична lock-установке; проверка удалённого
runner/транзитивного разрешения остаётся после разрешённого push.
Внешних блокеров полного Windows pipeline нет. Глобальную сетевую изоляцию
браузерного сервера не заявляем; строгий запрет сокетов проверен отдельным скриптом.

Нет новой категории/модели, настройки по test, API/Docker/новой инфраструктуры,
промышленного внедрения или гарантий для других камер. Ограничения данных,
калибровки, AUROC и локализации сохранены. Модели/пороги/исторические результаты
и LICENSE не перезаписаны. Исходники между финальными проверками не менялись.

Для независимого ревью открыть [README](../README.md), этот отчёт,
[воспроизведение](../docs/reproduce.md), [demo](../docs/demo.md) и
[interview](../docs/interview.md); при необходимости изучить локальные evidence.
Далее — независимое ревью, отдельное разрешение на коммит/push и фактическая
проверка Actions. Проект ещё не опубликован и не заявлен промышленно готовым.
