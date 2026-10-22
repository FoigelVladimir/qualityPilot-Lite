# Зависимости и проверенные источники

Проверено 21 октября 2026. Выбрана фиксированная версия Anomalib 2.2.0;
она не заявлена как самая новая. Цель — воспроизводимая совместимая конфигурация.

- API Patchcore / Engine / preprocessing:
  https://anomalib.readthedocs.io/en/v2.2.0/markdown/guides/reference/models/image/patchcore.html
- Требования Anomalib 2.2.0: Python >=3.10, torch >=2.4.0, torchvision >=0.19.0:
  https://github.com/open-edge-platform/anomalib/blob/v2.2.0/pyproject.toml
- Официальные пары версий PyTorch и команды CUDA:
  https://pytorch.org/get-started/previous-versions/

Выбранная пара: torch 2.8.0 / torchvision 0.23.0, Anomalib 2.2.0,
Lightning 2.5.5, torchmetrics 1.8.2, timm 1.0.20, OpenCV 4.12.0.88.
Версии baseline, dev и UI зафиксированы в pyproject.toml.
`uv.lock` фиксирует разрешённое дерево, включая optional extras.
Импорты, CUDA-вычисления, реальный PatchCore и CPU fallback проверены в спринте 2.
Для API timm закреплён huggingface-hub 0.36.0: версия 2.2.0 оказалась несовместима.
Добавлен requests 2.32.5, необходимый при импорте video-модулей Anomalib.

## CUDA

Рекомендуемый путь через существующий менеджер и lock:

```text
uv sync --locked --python 3.11 --extra dev --extra ml --extra ui
```

В pyproject.toml torch/torchvision ML extra привязаны к официальному индексу cu128.
Для разового запуска используйте `uv run --no-sync ...`, чтобы не менять набор extras.
Альтернативный pip-путь в активированном Python 3.11 (прямые версии закреплены,
но транзитивное дерево pip не равно гарантированно uv.lock):

```text
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[ml]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Для отдельных CPU-only колёс индекс `https://download.pytorch.org/whl/cpu`.
Запуск на CPU поддерживается и в CUDA-сборке: настройка `device: cpu`.
Версия CUDA в nvidia-smi — возможность драйвера, а не версия установленного torch.
На текущей машине RTX 3050 8 ГБ и драйвер 610.88; PyTorch 2.8.0+cu128,
CUDA runtime 12.8 фактически выполнили обучение memory bank и inference.
Установка CUDA Toolkit отдельно не является
обязательным шагом для официальных готовых колёс PyTorch.

## Защита независимого test

Официальные примеры используют `MVTecAD()`, `Patchcore(...)`, `Engine.fit(...)`.
Их значения разбиения по умолчанию нельзя слепо переносить в этот проект:
validation может строиться из test. Реализован адаптер нативного PatchcoreModel
без Engine/datamodule/postprocessor; читается manifest спринта 1, train и calibration
проходят normal-only gate. Pixel threshold — квантиль максимумов normal calibration
maps, зафиксированный до test. Подробности и ограничения в reports/sprint2_report.md.
