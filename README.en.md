# QualityPilot Lite

[![checks](https://github.com/FoigelVladimir/qualityPilot-Lite/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/FoigelVladimir/qualityPilot-Lite/actions/workflows/checks.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-147d73)](LICENSE)

[Русский README](README.md) · [Setup and reproduction (Russian)](docs/reproduce.md) · [Demo guide (Russian)](docs/demo.md)

**Visual anomaly detection with a reproducible experiment and a local inspection app.**

Educational ML/CV project comparing global features + Isolation Forest with
PatchCore on MVTec AD bottle. A Russian-language Streamlit app provides anomaly
scores, suspicious regions and ACCEPT / REVIEW / REJECT recommendations.
An anomaly score is not a defect probability.

![Actual app showing saved evaluation results; no dataset photographs](docs/assets/results-ui.png)

## Results and limits

One shared test set: **83 images, 20 normal and 63 defective**.
167 normal training images and 42 separate normal calibration images;
thresholds were fixed before test evaluation without using test masks.

| Metric | Isolation Forest | PatchCore |
|---|---:|---:|
| Image AUROC | 0.624603 | 1.000000 |
| Recall, single accept threshold | 0.571429 | 1.000000 |
| F1, single accept threshold | 0.672897 | 0.969231 |
| Micro Dice, 224×224 | N/A | 0.560382 |
| Micro IoU, 224×224 | N/A | 0.389257 |

![Image AUROC on the known bottle test set](docs/assets/image-auroc.png)

Source: [Sprint 2 experiment](reports/sprint2_report.md),
`bottle-patchcore-20261021T103450Z-5eddf169`. AUROC measures image ranking,
not defect boundary accuracy. This chart is not a new experiment.
PatchCore still has **4 false positives out of 20 normal images** at the single
threshold; those images go to REVIEW under the two-threshold policy.
Localization is moderate; the baseline has no anomaly map.

This is a post-hoc comparison: baseline test results were known when PatchCore
was integrated. A [clean-environment rerun](reports/sprint4_report.md) verifies
reproducibility on the same test, not independent generalization.
One category, known test data, no evaluation on other cameras, drift or near-duplicates.
Human review outcomes are unknown. No industrial performance guarantee is claimed.

## Run locally

Verified setup: Windows / PowerShell, Python 3.11, uv 0.9.5.

```powershell
git clone https://github.com/FoigelVladimir/qualityPilot-Lite.git
Set-Location qualityPilot-Lite
py -m pip install uv==0.9.5
uv python install 3.11
uv sync --locked --python 3.11 --extra dev --extra ui
uv run --no-sync python -m streamlit run app.py
```

Open [localhost:8501](http://127.0.0.1:8501). This lightweight installation has
**no Torch, data or trained models**. Navigation and missing-artifact guidance work;
the pictured results and inference require the [full preparation workflow](docs/reproduce.md).
The app does not download data or train models automatically.

## Implementation and verification

Project code covers data validation/provenance, split/manifest, baseline features,
model integration, normal-only calibration, a shared Inspector, metrics/benchmark,
checkpoint reload, presentation, tests and CI. Isolation Forest is provided by
scikit-learn; PatchCore, coreset and nearest-neighbour scoring by Anomalib;
the frozen ResNet18 backbone uses pretrained ImageNet weights.

Development used substantial **OpenAI Codex** assistance for code generation and
technical checks. The author directs the project, participates in demo validation
and studies the design; this is not presented as an unaided algorithm implementation.

CI checks baseline and UI on Windows/Ubuntu, with UI tests running without Torch
and without skips. CUDA and full PatchCore training are not covered by CI.
See [reports](reports/README.md), [dependencies](docs/dependencies.md) and the
[interview notes](docs/interview.md) for details (Russian).

Code: [MIT](LICENSE). MVTec AD: **CC BY-NC-SA 4.0**, non-commercial use;
© MVTec Software GmbH, Bergmann et al. Experiments used a pinned third-party copy;
equivalence to the official archive has not been independently established.
See [data provenance](docs/data.md). Dataset photographs, derived maps, model
weights and local evidence are excluded from this repository.
