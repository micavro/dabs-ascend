#!/usr/bin/env bash
# Run on the Linux host; only a temporary CPU controller is created.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen2.5-1.5B-Instruct}"
DEVICE_IDS="${DEVICE_IDS:-0,1,2,3,4,5,6,7}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-wushibo-qwen-small-npu}"
CONTROLLER_PYTHON="${CONTROLLER_PYTHON:-/usr/local/python3.12.13/bin/python}"
MODE="${1:-smoke}"
if (( $# > 0 )); then shift; fi
case "$MODE" in smoke|benchmark|check|test|plan) ;; *) echo 'Use smoke/benchmark/check/test/plan' >&2; exit 1;; esac
[[ "$DEVICE_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid DEVICE_IDS' >&2; exit 1; }
[[ "$CONTAINER_PREFIX" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid CONTAINER_PREFIX' >&2; exit 1; }
# Host snapshot and controller selection must agree. Use environment overrides.
for arg in "$@"; do
  case "$arg" in
    --devices|--devices=*|--base-port|--base-port=*|--model|--model=*|--container-prefix|--container-prefix=*|--deployment-manifest|--deployment-manifest=*)
      echo 'Use DEVICE_IDS / BASE_PORT / MODEL_PATH / CONTAINER_PREFIX environment variables for deployment settings.' >&2; exit 1;;
  esac
done
FIRST_DEVICE="${DEVICE_IDS%%,*}"
IMAGE_ID="$(sudo docker inspect --format '{{.Image}}' "${CONTAINER_PREFIX}${FIRST_DEVICE}")"
sudo docker image inspect "$IMAGE_ID" >/dev/null
MOUNTS=()
EXTRA=()
if [[ "$MODE" != test && "$MODE" != plan ]]; then
  [[ -f "$MODEL_PATH/config.json" ]] || { echo 'MODEL_PATH/config.json not found' >&2; exit 1; }
  MODEL_PATH="$(cd -- "$MODEL_PATH" && pwd)"
  MANIFEST="$(python3 "$ROOT/tools/inspect_dabs_existing.py" --devices "$DEVICE_IDS" --container-prefix "$CONTAINER_PREFIX")"
  MOUNTS=(--mount "type=bind,src=$MODEL_PATH,dst=/models/qwen,readonly")
  EXTRA=(--deployment-manifest "/project/outputs/$(basename "$MANIFEST")")
fi

# No NPU mapping, Docker socket, model startup, or signals to existing services.
sudo docker run --rm --pull never --network host \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROOT,dst=/project" "${MOUNTS[@]}" \
  -e PYTHONUTF8=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --workdir /project --entrypoint "$CONTROLLER_PYTHON" "$IMAGE_ID" \
  tools/run_dabs_existing.py "$MODE" "${EXTRA[@]}" \
  --devices "$DEVICE_IDS" --container-prefix "$CONTAINER_PREFIX" \
  --base-port "${BASE_PORT:-8001}" --max-inflight "${MAX_INFLIGHT:-2}" \
  --condition-timeout 1800 --request-timeout 300 "$@"
