# Состояние этапов и следующие действия

- **Sprint 1:** data pipeline, baseline, normal-only calibration и первый bottle run.
  [Результаты](../reports/bottle-baseline.md).
- **Sprint 2:** PatchCore/Anomalib, карты, pixel metrics и сопоставимое сравнение.
  [Отчёт](../reports/sprint2_report.md), [аудит Sprint 1–2](../reports/sprint1_2_audit.md).
- **Sprint 3:** локальный Streamlit UI, state/cache, saved reports.
  [Отчёт](../reports/sprint3_report.md), [ревью и устранённые замечания](../reports/sprint3_review.md).
- **Sprint 4:** чистые установки, полный повтор фиксированного эксперимента,
  README, demo/interview и проверка будущей поставки. Фактические проверки,
  ограничения и evidence: [отчёт Sprint 4](../reports/sprint4_report.md).

## Публикация и проверка поставки

Независимое ревью четырёх спринтов завершено. Пользователь разрешил коммит и
обычный push в существующий origin/main; история и LICENSE сохраняются.

1. Проверить точный staged-состав и итоговые локальные тесты, затем опубликовать
   реализацию обычным push. Данные, веса, dataset screenshots, локальные окружения
   и evidence в Git не включать.
2. Для отправленного HEAD дождаться всех четырёх вариантов
   [checks](https://github.com/FoigelVladimir/qualityPilot-Lite/actions/workflows/checks.yml):
   baseline и UI на Ubuntu/Windows. UI должен выполнить тесты без Torch и skips.
   До завершения конкретного запуска CI не считается подтверждённым.
3. Проверить свежий checkout опубликованного HEAD: dev/ui-установку, импорт data
   pipeline, CLI, UI без артефактов, GitHub README/Mermaid/badge и внутренние ссылки.
4. После успешной публикации использовать [demo](demo.md), [interview](interview.md)
   и [формулировку для резюме](resume.md). Точные финальные SHA и Actions URL
   фиксируются в сообщении о завершении; самоссылочные коммиты не нужны.

Новые модели, категории, API, Docker и инфраструктура в этот этап не входят.
Промышленное внедрение потребовало бы независимых данных других камер/партий,
оценки стоимости ошибок и проверки дрейфа; это не реализованные возможности.
