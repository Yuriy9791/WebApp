# PDF по существующему Reference

Связь логов с публикациями уже хранится в старом CSV и используется приложением.
Новая связь по скважинам, log_id или мнемоникам не создаётся.

## Пакетная загрузка

Для подготовленного архива с несколькими статьями достаточно одной команды:

```bash
bash GDS_Publications_Batch/RUN_UPLOAD.sh
```

Пакет содержит все исходные документы и уже заполненный `batch.json`.
Он сопоставляет файл с существующим Reference, а не создаёт связи с логами.
Uploader сначала проверяет весь пакет: пути, PDF/PPTX, SHA-256, существование
Reference в активном CSV и конфликты с уже загруженными файлами.
Затем загружает документы. Повторный запуск пропускает идентичные объекты;
после сетевого сбоя можно выполнить ту же команду ещё раз.

Из рабочей копии проекта тот же режим доступен как:

```bash
python tools/upload_publication.py --batch /path/to/batch.json --bucket BUCKET --region eu-north-1
```

Минимальная запись batch.json:

```json
{"schema_version": 1, "files": [{"file": "articles/paper.pdf", "sha256": "SHA256_OF_FILE", "reference": "EXISTING CSV REFERENCE"}]}
```

Пустой reference сохраняет оригинал в `publications/originals/` без назначения
его источником какого-либо лога. Для PowerPoint допускается только этот режим.
Файлы со ссылкой сохраняются в `publications/<reference-hash>.pdf`, без дублирования
оригинала. LAS, Parquet, metadata CSV и current.json не меняются.
Скрипт запуска работает в локальной AWS-сессии или CloudShell и устанавливает
только boto3 во временное окружение /tmp; полный requirements.txt не нужен.
Дополнительный аргумент `--dry-run` проверяет пакет без записи.

## Загрузка одного PDF

Из рабочей копии WebApp с установленным requirements-deploy.txt и настроенной
AWS-сессией выполните одну команду:

```text
python tools/upload_publication.py --pdf "Liu_2017.pdf" --reference "Liu et al 2017 N Iraq (Kurdistan)"
```

Имя PDF может быть любым. Значение --reference копируется из существующего
раздела Publications / CSV. Доступные значения можно вывести командой:

```text
python tools/upload_publication.py --list-references
```

При локальном запуске при необходимости добавляются --profile, --bucket и --region.
В CloudShell используется текущая авторизованная AWS-сессия. Для preview можно
добавить --dry-run; обычная команда сразу загружает PDF.

Скрипт проверяет, что Reference существует в текущем CSV, и сохраняет PDF в
publications/ существующего бакета. Имя объекта получается автоматически из
Reference. JSON-каталог, title и ручные связи с логами не нужны. Повторная
загрузка для того же Reference обновляет PDF; S3 Versioning сохраняет версии.
LAS, Parquet и current.json остаются прежними.

## Приложение

В разделе Publications сохраняется та же строка, например:
Liu et al 2017 N Iraq (Kurdistan).

Когда PDF загружен, сама эта строка становится ссылкой на скачивание. У всех
логов с тем же Reference ссылка появляется автоматически. Если PDF ещё нет,
название остаётся обычным текстом. Дополнительные карточки и кнопки не создаются.

Для применения кода: Heroku → Deploy → gds-viewer-iteration-2 → Deploy Branch.
После обновления страницы выберите логи и нажмите Build logs.

## Read access для Heroku

К существующей IAM policy reader, используемого Heroku, добавляется отдельное
правило чтения нового префикса; существующие разрешения сохраняются:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject"],
    "Resource": ["arn:aws:s3:::gds-viewer-data-745399722372-eu-north-1/publications/*"]
  }]
}
```

PDF скачивается напрямую из S3 по presigned URL, выданной приложением.
Upload script требует только boto3, без Dash, Plotly, NumPy или PyArrow.
Подготовка пакета локально не означает загрузку в AWS: запись выполняется
после запуска RUN_UPLOAD.sh в вашей AWS-сессии.
