# Воспроизведение от исходников до COMPLETE run

Основной проверенный маршрут PowerShell находится в [README](../README.md).
Нужны исходники, tests, configs, app.py, .streamlit, pyproject.toml и uv.lock;
checkout одного LICENSE недостаточен. Локальные evidence, данные и веса не нужны
для установки или синтетических тестов; для полного pipeline они создаются явно.
Используйте Python 3.11 и uv 0.9.5, `uv sync --locked`, без обновления lock.

## Чистая установка

1. Откройте новую копию всех будущих Git-файлов, без data/artifacts/reports/runs/.venv.
2. Выполните `uv sync --locked --python 3.11 --extra dev --extra ui`.
3. Запустите `uv run --no-sync pytest -q` и приложение из README. Без ML extra
   ML-тесты явно пропускаются; About и сообщения об отсутствии артефактов работают.
   Для проверки только UI: `uv run --no-sync pytest tests/test_ui_data.py tests/test_ui_app.py -q`;
   здесь пропусков быть не должно.
4. Для полного pipeline добавьте `--extra ml`, сохранив `dev,ui`.
   `uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   проверит фактическую доступность CUDA; CPU также поддерживается.

В Sprint 4 использованы две отдельные временные среды с переменной
`UV_PROJECT_ENVIRONMENT`, указывающей на каждую из них, и одной чистой копией
исходников. Это изолирует проверку от основного .venv. В обычном checkout uv
создаёт `.venv`, переменная не нужна. Python/uv и wheel-cache могут уже существовать;
установка пакетов из кэша не означает наличие готовой модели проекта.

## Данные, обучение, калибровка, оценка

Последовательно выполните команды раздела README: `download-mirror` **либо**
`import-data`, `prepare`, `baseline`, `patchcore`. Mirror — явный сторонний источник,
не автоматический fallback. Фиксированный commit и хеши записываются в provenance.
Первый запуск PatchCore получает ImageNet-веса; доступ к сети и свободное место
для wheel/cache необходимы. UI ничего не скачивает и не обучает.

При стандартных конфигах baseline создаёт `reports/runs/bottle-baseline/` и
`artifacts/bottle-baseline.joblib`. PatchCore проверяет этот manifest и сохранённый
baseline, строит memory bank только из normal train, калибрует image/pixel thresholds
на normal calibration, сохраняет model.pt и проверяет reload **до** первого test.
Далее оценивает обе модели на одинаковых 83 test paths/labels, сохраняет comparison,
confusion matrices и иллюстрации. COMPLETE означает завершение всего этого pipeline;
FAILED сохраняет диагностику и не предлагается в UI.

Результат PatchCore: `reports/runs/bottle-patchcore-<UTC>-<id>/`:

- status/config/environment/experiment, manifest, dataset_summary;
- calibration.csv, model.pt, model_checksum.json, reload_check.json;
- baseline/ и patchcore/: results.csv, metrics.json и confusion_matrix.png;
- comparison.json/md; examples/ с иллюстрациями только реально существующих случаев.

Скопируйте имя созданной папки из вывода CLI. В боковой панели Streamlit выберите
именно этот COMPLETE run. UI baseline_config должен соответствовать baseline,
использованному этим экспериментом; текущий стандарт — configs/bottle.yaml.
При отсутствии весов saved reports доступны; без данных недоступен preview,
но таблицы сохраняются. Повреждённые файлы дают сообщение с файлом/полем без traceback.

## Повтор, сохранность и сравнение

Baseline откажется перезаписать существующие model_path/metrics.json. Самый простой
повтор с неизменными конфигами — новая копия исходников без артефактов, как в Sprint 4.
При переиспользовании данных сделайте отдельные копии конфигов и измените только пути:

| Поле | Требование |
|---|---|
| baseline data_root | существующая проверенная bottle |
| baseline output_dir | новая пустая папка |
| baseline model_path | новый отсутствующий .joblib |
| PatchCore baseline_config | путь к этому новому baseline YAML |
| PatchCore reference_run | тот же новый baseline output_dir |
| PatchCore output_root | папка для новых уникальных PatchCore runs |
| UI baseline_config | новый YAML внутри корня проекта |
| UI runs_root / selected_run | папка и имя полученного COMPLETE run |

Не меняйте seed, split fraction, признаки, размеры, backbone/layers, coreset,
neighbours, quantiles или warmup ради совпадения результатов. Все пути YAML
разрешаются от **папки файла конфигурации**, кроме UI-путей от корня проекта.
Сверяйте manifest SHA256, paths/labels, calibration scores/thresholds, test scores,
decisions и image/pixel metrics. Время зависит от нагрузки/кэша; checkpoint хранит
metadata с путями и окружением, поэтому его байтовый SHA256 между запусками
не является критерием равенства модели. Старые checkpoint проверяются по SHA256
только для доказательства, что они не изменены.

Test уже известен: повтор проверяет воспроизводимость подготовки, а не создаёт
независимый benchmark. При отклонениях сохраните оба набора результатов и опишите
их; не подбирайте параметры. Если доступны только inference и reload, нельзя
объявлять подтверждённым обучение memory bank с нуля.

## Offline и проверки

После подготовки используйте свой `model.pt` через Inspector или CLI `inspect --model`.
В README есть opt-in smoke с `HF_HUB_OFFLINE=1`. Не удаляйте model.pt: он содержит
все backbone weights и memory bank. Установка зависимостей не становится offline
автоматически; этот режим относится к inference уже сохранённой модели.

Проверки перед ревью: pytest, отдельный offline smoke, UI без ML, Ruff, lock/dependency
check, CLI и браузерные normal/defective/upload/corrupt/temporary-threshold/saved-report
сценарии. CI baseline/UI использует синтетические данные и не получает pretrained-веса.
Remote Actions и Linux требуют отдельного фактического запуска, локальный PASS
Windows их не заменяет. [Фактический отчёт Sprint 4](../reports/sprint4_report.md).
