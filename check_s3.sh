#!/usr/bin/env bash
set -Eeuo pipefail
GDS_APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$GDS_APP_DIR"
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt
.venv/bin/python -u -m gds_viewer.cloudshell_check "$@"
