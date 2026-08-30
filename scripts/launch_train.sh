#!/usr/bin/env bash
set -euo pipefail

MEMX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MEMX_REPOSITORY_ROOT="$(cd -- "${MEMX_SCRIPT_DIR}/.." && pwd -P)"
export PYTHONPATH="${MEMX_REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MEMX_MODE="${MEMX_MODE:-train}"
MEMX_NNODES="${NNODES:-1}"
MEMX_WORLD_SIZE="${WORLD_SIZE:-1}"
MEMX_LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${MEMX_WORLD_SIZE}}"
MEMX_NODE_RANK="${NODE_RANK:-0}"
MEMX_MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MEMX_MASTER_PORT="${MASTER_PORT:-29500}"
MEMX_DEVICE="${DEVICE:-ppu}"
MEMX_CONFIG="${CONFIG:-configs/experiments/smoke.yaml}"
MEMX_DATA_ROOT="${DATA_ROOT:-${MEMX_REPOSITORY_ROOT}/data/memx}"
MEMX_MODEL_ROOT="${MODEL_ROOT:-${MEMX_REPOSITORY_ROOT}/.cache/memx/models}"
MEMX_RUN_ROOT="${RUN_ROOT:-${MEMX_REPOSITORY_ROOT}/artifacts/company/smoke}"
MEMX_RESUME="${RESUME:-never}"

if [[ "${MEMX_MODE}" != "train" && "${MEMX_MODE}" != "evaluate" ]]; then
  echo "MEMX_MODE must be train or evaluate" >&2
  exit 2
fi
for MEMX_INTEGER in \
  "${MEMX_NNODES}" \
  "${MEMX_WORLD_SIZE}" \
  "${MEMX_LOCAL_WORLD_SIZE}" \
  "${MEMX_NODE_RANK}" \
  "${MEMX_MASTER_PORT}"; do
  if [[ ! "${MEMX_INTEGER}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "distributed launch integers must be canonical nonnegative decimals" >&2
    exit 2
  fi
done
if (( MEMX_NNODES < 1 || MEMX_WORLD_SIZE < 1 || MEMX_LOCAL_WORLD_SIZE < 1 )); then
  echo "node and world sizes must be positive" >&2
  exit 2
fi
if (( MEMX_NNODES * MEMX_LOCAL_WORLD_SIZE != MEMX_WORLD_SIZE )); then
  echo "WORLD_SIZE must equal NNODES multiplied by LOCAL_WORLD_SIZE" >&2
  exit 2
fi
if (( MEMX_NODE_RANK >= MEMX_NNODES )); then
  echo "NODE_RANK must be smaller than NNODES" >&2
  exit 2
fi

MEMX_EXTRA_ARGS=()
if [[ "${MEMX_MODE}" == "train" ]]; then
  MEMX_EXTRA_ARGS=(--resume "${MEMX_RESUME}")
fi

if [[ "${MEMX_USE_VENDOR_ENV:-0}" == "1" ]]; then
  MEMX_TORCHRUN=(python3 -m torch.distributed.run)
else
  MEMX_TORCHRUN=(uv run --frozen torchrun)
fi

cd "${MEMX_REPOSITORY_ROOT}"
exec "${MEMX_TORCHRUN[@]}" \
  --nnodes="${MEMX_NNODES}" \
  --nproc-per-node="${MEMX_LOCAL_WORLD_SIZE}" \
  --node-rank="${MEMX_NODE_RANK}" \
  --master-addr="${MEMX_MASTER_ADDR}" \
  --master-port="${MEMX_MASTER_PORT}" \
  -m ratemem.experiment.cli \
  "${MEMX_MODE}" \
  --config "${MEMX_CONFIG}" \
  --data-root "${MEMX_DATA_ROOT}" \
  --model-root "${MEMX_MODEL_ROOT}" \
  --run-root "${MEMX_RUN_ROOT}" \
  --device "${MEMX_DEVICE}" \
  "${MEMX_EXTRA_ARGS[@]}" \
  "$@"
