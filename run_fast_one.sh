#!/usr/bin/env bash
# Additive single-scenario entry; no install, NPU access, service restart or grid.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-plan}"
if (( $# )); then shift; fi
case "$MODE" in
  plan|logs) exec python3 "$ROOT/tools/run_fast_one.py" "$MODE" "$@" ;;
  selftest) cd "$ROOT"; exec python3 -m unittest discover -s tests/ascend -p test_fast_one.py -v ;;
  run) ;;
  *) echo 'Usage: bash run_fast_one.sh plan|selftest|logs|run --input <native_test_dir>' >&2; exit 2 ;;
esac
INPUT=""
ARGS=()
while (( $# )); do
  case "$1" in
    --input) [[ $# -ge 2 ]] || { echo '--input needs a directory' >&2; exit 2; }; INPUT="$2"; shift 2 ;;
    --input=*) INPUT="${1#*=}"; shift ;;
    --base-port|--base-port=*) echo 'Use BASE_PORT environment variable' >&2; exit 2 ;;
    --out|--out=*) echo 'Output is a fresh outputs/native_fast-one_* directory' >&2; exit 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -n "$INPUT" && -d "$INPUT" ]] || { echo '--input must be an existing current native_test directory' >&2; exit 2; }
INPUT="$(cd -- "$INPUT" && pwd)"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-wushibo-qwen-small-npu}"
CONTROLLER_PYTHON="${CONTROLLER_PYTHON:-/usr/local/python3.12.13/bin/python}"
BASE_PORT="${BASE_PORT:-8001}"
[[ "$CONTAINER_PREFIX" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid CONTAINER_PREFIX' >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[0-9]+$ ]] || { echo 'Invalid BASE_PORT' >&2; exit 2; }
# This first version intentionally supports the existing npu0..3 / four-port map.
# Do not silently reinterpret DEVICE_IDS used by a different deployment.
[[ "${DEVICE_IDS:-0,1,2,3}" == '0,1,2,3' ]] || { echo 'fast-one requires DEVICE_IDS=0,1,2,3' >&2; exit 2; }
for R in 0 1 2 3; do
  [[ "$(sudo docker inspect --format '{{.State.Running}}' "${CONTAINER_PREFIX}${R}")" == true ]] || { echo "Service ${R} not running; not restarting it" >&2; exit 1; }
done
IMAGE_ID="$(sudo docker inspect --format '{{.Image}}' "${CONTAINER_PREFIX}0")"
sudo docker image inspect "$IMAGE_ID" >/dev/null
printf -v DABS_CONTROLLER_COMMAND '%q ' bash "$ROOT/run_fast_one.sh" run --input "$INPUT" "${ARGS[@]}"
# Serializes fast-one invocations from DIFFERENT source directories on this host.
# Existing older controllers do not use this host lock; stop them before running.
exec 9>"/tmp/dabs-fast-one-${UID}-${BASE_PORT}.lock"
flock -n 9 || { echo 'Another fast-one controller holds these ports' >&2; exit 1; }
mkdir -p "$ROOT/outputs"
LOG="$ROOT/outputs/fast-one-console-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
echo "Console log: $LOG"
sudo docker run --rm --pull never --network host \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROOT,dst=/project" \
  --mount "type=bind,src=$INPUT,dst=/evidence,readonly" \
  -e PYTHONUTF8=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=1 \
  -e "DABS_HOST_PROJECT=$ROOT" -e "DABS_HOST_COMMAND=$DABS_CONTROLLER_COMMAND" \
  -e "DABS_CONTROLLER_IMAGE=$IMAGE_ID" \
  --workdir /project --entrypoint "$CONTROLLER_PYTHON" "$IMAGE_ID" \
  tools/run_fast_one.py run --input /evidence --base-port "$BASE_PORT" "${ARGS[@]}" 2>&1 | tee "$LOG"
