# Получение MVTec AD

Официальная страница:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad#c3391

1. Прокрутите до **DOWNLOAD DATASET → Fill in the form**.
2. Введите имя (First Name), фамилию (Last Name) и Email. Position/Function
   необязательное; для самостоятельного учебного проекта допустимо указать
   Independent learner.
3. Подтвердите некоммерческое использование: **I am aware that the datasets are
   for non-commercial use only**. Подписка Campus Newsletter необязательна.
4. Нажмите **SUBMIT AND CONTINUE TO DOWNLOAD PAGE**.
5. На странице Download the MVTec AD dataset выберите **Bottle (148 MB)**,
   а не весь архив (около 4.9 GB), AD2, LOCO AD или 3D-AD.
6. Сохраните `bottle.tar.xz` в локальную папку и импортируйте через CLI.

Если форма пустая, дождитесь загрузки или временно отключите блокировщик рекламы
для официальной страницы. Не нужны платные API или подписки.

## Обнаруженная проблема доступа — 21 октября 2026

После заполнения формы пользователем проверена ссылка Bottle непосредственно
с официальной страницы
https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads

Она ведёт на MyDrive:
https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz

При GET и HEAD сервер вернул **HTTP 404**, HTML на немецком с заголовком
**Seite nicht gefunden**. Это не архив. Регистрация в MyDrive и установка его
клиента не являются решением обнаруженной ошибки ссылки.

Получить актуальную официальную ссылку можно у MVTec через **Get in touch**.
Альтернативный публичный источник для продолжения без ожидания описан ниже.
Не используйте угаданные адреса или непроверенные зеркала.
Код `uv run --no-sync qualitypilot access` проверяет доступность официальной страницы и наличие
формы; он не утверждает, что архив доступен, и не обходит форму.

Текст для обращения (самостоятельно, проект ничего не отправляет):

> Hello, I am using MVTec AD for a non-commercial educational project.
> After submitting the official access form, the Bottle download link on the
> downloads page returns HTTP 404 ("Seite nicht gefunden") from MyDrive.
> Could you provide an updated official download link for bottle.tar.xz?

## Публичная копия без обращения в поддержку

После запроса пользователя продолжить без ожидания проверена сторонняя копия
https://huggingface.co/datasets/foersben/mvtec-ad

Карточка указывает оригинальный источник MVTec, авторов и CC BY-NC-SA 4.0.
В категории bottle сохранены исходная структура train/test/ground_truth и тексты
license.txt/readme.txt с copyright MVTec Software GmbH. Это не официальный
аккаунт MVTec; утверждать официальное подтверждение подлинности нельзя.

Явная команда для этого источника:

```text
uv run --no-sync qualitypilot --config configs/bottle.yaml download-mirror
```

Загружаются только bottle (около 157 MB) и текстовые файлы лицензии, без исполнения
кода из удалённого репозитория. Источник зафиксирован на commit
`c75b39616f84db43677bcc8228caaafaf5096d7f`, а не плавающей ветке main.
Проверяются размер и LFS SHA256 каждого файла (Git blob hash для небольших файлов).
Это подтверждает целостность передачи относительно копии, но не идентичность
с недоступным официальным архивом. Точное сравнение с оригиналом остаётся ограничением.

Сохраняется `data/mvtec_ad/provenance-bottle.json` с источником, commit, временем,
лицензией и хешами. Эта информация также включается в experiment.json и модель.
После загрузки CLI проверяет структуру, декодирование, маски и отсутствие утечек.
Прерванную загрузку можно повторить: существующие файлы повторно проверяются,
изменённые файлы не перезаписываются молча.

## Локальная структура

```text
data/mvtec_ad/bottle/
  train/good/*.png
  test/good/*.png
  test/broken_large/*.png
  test/broken_small/*.png
  test/contamination/*.png
  ground_truth/broken_large/*_mask.png
  ground_truth/broken_small/*_mask.png
  ground_truth/contamination/*_mask.png
```

Имена типов дефектов читаются автоматически. Маска должна иметь размер исходного
изображения, один канал, значения 0/1 или 0/255 и ненулевую область дефекта.
Дубликаты по декодированным RGB-пикселям запрещены, в том числе при разных
расширениях. Near-duplicates автоматически не определяются.
Нормальные изображения test не переносятся в calibration.

Импорт принимает tar/tar.gz/tar.xz с корневой папкой `bottle/`, проверяет пути,
ссылки, специальные файлы, дубликаты, лимит распаковки 5 GB и не перезаписывает
существующую категорию. Если получен полный официальный архив, извлекается только
выбранная категория, но скачивать полный архив специально не требуется.
SHA256 импортируемого файла выводится для фиксации происхождения. Сам по себе
локальный SHA256 не является проверкой подлинности без опубликованного хеша MVTec.

Данные: **CC BY-NC-SA 4.0**, исключительно некоммерческое использование.
Официальные условия: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
Изображения, маски и производные примеры нельзя случайно публиковать вместе с кодом:
`data/`, `artifacts/`, `reports/runs/` и архивы исключены в `.gitignore`.
