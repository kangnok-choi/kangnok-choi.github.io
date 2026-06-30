#!/usr/bin/env bash
# Serve the MkDocs blog locally with live reload.
set -euo pipefail

cd "$(dirname "$0")"

# Use uv if available (project is managed with uv), otherwise fall back to mkdocs directly.
if command -v uv >/dev/null 2>&1; then
    exec uv run mkdocs serve "$@"
else
    exec mkdocs serve "$@"
fi
