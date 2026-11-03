# PDF по существующему Reference

Связь логов с публикациями уже хранится в старом CSV и используется приложением.
Новая связь по скважинам, log_id или мнемоникам не создаётся.

## Загрузка

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
Настоящие PDF в этой сессии пока не предоставлены и не загружены.
