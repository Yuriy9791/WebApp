#!/usr/bin/env bash
set -Eeuo pipefail

GDS_APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GDS_TEMP_ROOT="$(mktemp -d /tmp/gds-v002-deploy.XXXXXXXX)"
trap 'rm -rf -- "$GDS_TEMP_ROOT"' EXIT

echo 'Preparing a temporary upload environment in /tmp (AWS dependencies only).'
python3 -m venv --system-site-packages "$GDS_TEMP_ROOT/venv"
"$GDS_TEMP_ROOT/venv/bin/python" -m pip install \
  --disable-pip-version-check --no-cache-dir --only-binary=:all: \
  -r "$GDS_APP_ROOT/requirements-deploy.txt"
"$GDS_TEMP_ROOT/venv/bin/python" -u "$GDS_APP_ROOT/tools/deploy_release.py" "$@"
