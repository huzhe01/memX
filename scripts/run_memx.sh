#!/usr/bin/env bash
set -euo pipefail

MEMX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MEMX_REPOSITORY_ROOT="$(cd -- "${MEMX_SCRIPT_DIR}/.." && pwd -P)"

if [[ -n "${MEMX_HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="${MEMX_HF_ENDPOINT}"
fi

if [[ "${MEMX_USE_VENDOR_ENV:-0}" == "1" ]]; then
  export PYTHONPATH="${MEMX_REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec python3 -m ratemem.experiment.cli "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required outside the vendor PPU container; install uv 0.8.14" >&2
  exit 2
fi
exec uv run --frozen memx "$@"
