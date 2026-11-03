# Source PDF downloads: полная цепочка

Первый этап - хранение PDF и download по выбранному логу. Search, embeddings и
RAG в этот этап не входят. Настоящие PDF статей пока не загружены.

## 1. Где будут файлы

На рабочем компьютере создаём каталог `articles/` внутри рабочей копии WebApp.
Туда кладём PDF и JSON с подтверждённой привязкой к логам. Этот каталог исключён
из Git: PDF отправляются в S3, а код upload script остаётся в GitHub.

В существующем бакете `gds-viewer-data-745399722372-eu-north-1`:

| S3 key | Содержание |
| --- | --- |
| `publications/files/<sha256>.pdf` | Один PDF; content-addressed key предотвращает перезапись другим файлом |
| `publications/catalog.json` | Название, download filename, SHA256, размер и привязки к log_id |

Каталог статей независим от `releases/v002/`. Добавление PDF не изменяет
`current.json`, LAS, Parquet или существующие каталоги кривых. Один PDF можно
связать с несколькими кривыми нескольких скважин.

## 2. Сначала получить сами статьи

Для первой загрузки нужен один реальный PDF и указание, какие логи взяты из него.
Сверяем статью с существующими Reference и названиями LAS. Если соответствие
неочевидно, подтверждаем его по статье. Название файла само по себе не доказывает
происхождение лога. Прикрепляем только явно указанные кривые.

## 3. Обновить код приложения

Heroku: Deploy → GitHub → `gds-viewer-iteration-2` → Deploy Branch.
Обновлённый `/readyz` содержит `source_pdf_downloads: true`.

## 4. Разрешить Heroku читать новый префикс S3

Текущая read policy могла разрешать только `current.json` и `releases/*`.
К тому же IAM user/role, чьи AWS credentials использует Heroku, добавляем
отдельную inline policy `GDSReadSourcePDFs`; существующую policy сохраняем:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "ReadGDSSourcePDFs",
    "Effect": "Allow",
    "Action": ["s3:GetObject"],
    "Resource": ["arn:aws:s3:::gds-viewer-data-745399722372-eu-north-1/publications/*"]
  }]
}
```

Этот JSON можно получить без AWS-запросов:

```text
python tools/upload_publication.py --print-read-policy --bucket gds-viewer-data-745399722372-eu-north-1
```

Uploader использует отдельную AWS-сессию с правами чтения текущих метаданных,
загрузки в `publications/*` и проверок существующего S3-хранилища. В CloudShell
можно использовать текущую авторизованную сессию; на компьютере - настроенный
AWS profile. Heroku остаётся reader.

## 5. Подготовить binding JSON

Рядом с PDF создаём `articles/source.json`. Формат есть в
`examples/publication_input.json`. Нужно заменить все значения примера:

```json
{
  "pdf_file": "YOUR_PAPER.pdf",
  "title": "Full publication title",
  "links": [{
    "las_filename": "EXACT_LAS_FILENAME.las",
    "curves": ["GR", "DT"]
  }]
}
```

`pdf_file` задаётся относительно JSON. `las_filename` должен точно совпадать с
именем в активном наборе. `curves` - конкретные существующие мнемоники этого LAS.
Для нескольких скважин добавляем элементы в `links`. Upload script сам получает
record_id/log_id из проверенного каталога S3 и сохраняет SHA256 исходного LAS.

## 6. Запустить uploader из VS Code

Обновить рабочую копию ветки `gds-viewer-iteration-2`. Команды ниже - для
PowerShell в корне WebApp. Устанавливаются только boto3 и его зависимости:

```powershell
py -m venv .venv-upload
.\.venv-upload\Scripts\python.exe -m pip install --no-cache-dir -r requirements-deploy.txt
```

Если локальный AWS profile уже настроен, сначала выполняем dry run:

```powershell
.\.venv-upload\Scripts\python.exe tools/upload_publication.py --profile YOUR_AWS_PROFILE --bucket gds-viewer-data-745399722372-eu-north-1 --region eu-north-1 --input articles/source.json
```

Dry run только читает активные метаданные и выводит план: PDF, размер, bucket,
скважины и кривые. После проверки плана выполняем ту же команду с `--apply`:

```powershell
.\.venv-upload\Scripts\python.exe tools/upload_publication.py --profile YOUR_AWS_PROFILE --bucket gds-viewer-data-745399722372-eu-north-1 --region eu-north-1 --input articles/source.json --apply
```

В CloudShell используется тот же Python script с зависимостями
`requirements-deploy.txt`; `--profile` не нужен. Не требуется устанавливать
Dash, Plotly, NumPy, PyArrow, lasio или запускать rebuild данных.

Сначала загружается PDF. Затем catalog.json обновляется с условием по ETag,
чтобы параллельные загрузки не затёрли изменения друг друга. При повторной
загрузке того же PDF существующие привязки сохраняются и объединяются с новыми.

## 7. Что увидит пользователь

В Wells выбираем скважину и кривые → Logs → Build logs. В разделе Publications
отображаются название статьи, привязанные скважины/кривые и **Download source PDF**.

Кнопка вызывает `/source-pdf/<publication_id>`. Heroku проверяет каталог и
выдаёт временную S3 download URL. Сам PDF скачивается из S3 в браузер.
Если PDF для выбранного лога ещё не прикреплён, кнопки нет: остаётся citation
и сообщение `Source PDF has not been uploaded for the selected logs.`

## 8. Что проверить

Download должен вернуть именно загруженный PDF. Ссылка не должна появляться у
несвязанных кривых. Если LAS позже заменят другим файлом с тем же именем,
прежняя привязка автоматически не переносится: проверяется SHA256 исходного LAS.
При ошибке чтения PDF-каталога графики продолжают работать, а ошибка пишется в log.

Для начала следующего шага передайте первый PDF и имя скважины/лога либо
существующий Reference. На этой основе подготавливается реальный binding JSON.
