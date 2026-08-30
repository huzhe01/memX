#!/usr/bin/env bash
set -euo pipefail

MEMX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MEMX_REPOSITORY_ROOT="$(cd -- "${MEMX_SCRIPT_DIR}/.." && pwd -P)"
cd "${MEMX_REPOSITORY_ROOT}"

if [[ "${MEMX_USE_VENDOR_ENV:-0}" == "1" ]]; then
  python3 - <<'PY'
import importlib.metadata
import sys

if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit("memX vendor execution requires Python 3.11 or 3.12")
for distribution in (
    "datasets",
    "diffusers",
    "numpy",
    "pydantic",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
):
    print(f"{distribution}={importlib.metadata.version(distribution)}")
PY
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv 0.8.14 is required; see https://docs.astral.sh/uv/" >&2
    exit 2
  fi
  uv sync --all-extras --frozen
fi

"${MEMX_SCRIPT_DIR}/run_memx.sh" --help >/dev/null
"${MEMX_SCRIPT_DIR}/run_memx.sh" runtime preflight --device "${DEVICE:-cpu}"
