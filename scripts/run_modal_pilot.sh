#!/usr/bin/env bash
set -euo pipefail
umask 077

run_guarded_uv() {
  /usr/bin/env -i \
    HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
    LANG=C.UTF-8 \
    MODAL_CONFIG_PATH=/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml \
    MODAL_PROFILE=ratemem-pilot \
    MODAL_ENVIRONMENT=main \
    /home/ubuntu/.local/bin/uv run --extra modal "$@"
}

run_guarded_uv ratemem-pilot validate-modal-config configured
run_guarded_uv ratemem-pilot preflight
run_guarded_uv ratemem-pilot provision-volumes
RATEMEM_ATTEMPT_ID="$(run_guarded_uv ratemem-pilot permit-field attempt_id)"
RATEMEM_DOWNLOAD_ROOT="artifacts/pilot"
RATEMEM_DESTINATION="artifacts/pilot/${RATEMEM_ATTEMPT_ID}"
if [[ ! -d "${RATEMEM_DOWNLOAD_ROOT}" ]]; then
  echo "artifact download root is not a real directory; refusing launch" >&2
  exit 5
fi
if [[ -L "${RATEMEM_DOWNLOAD_ROOT}" ]]; then
  echo "artifact download root is not a real directory; refusing launch" >&2
  exit 5
fi
if [[ -e "${RATEMEM_DESTINATION}" ]]; then
  echo "artifact destination already exists; refusing overwrite" >&2
  exit 5
fi
if [[ -L "${RATEMEM_DESTINATION}" ]]; then
  echo "artifact destination is a symlink; refusing overwrite" >&2
  exit 5
fi
run_guarded_uv modal run -m ratemem.pilot.modal_app
run_guarded_uv modal volume get --env main ratemem-pilot-artifacts "attempts/${RATEMEM_ATTEMPT_ID}" "${RATEMEM_DOWNLOAD_ROOT}"
run_guarded_uv ratemem-pilot security-scan "${RATEMEM_DESTINATION}"
run_guarded_uv ratemem-pilot validate-artifact "${RATEMEM_DESTINATION}/attempt.pending.json"
