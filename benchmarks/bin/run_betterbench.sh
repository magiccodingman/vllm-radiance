#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VENV=${VLLM_BENCH_VENV:-/nvme/ediloca-1/venv/vllm-bench-env}
BETTERBENCH_COMMIT=575cc3925bac922d6ad4a39e62502673799979d9
BASE_URL=${BASE_URL:-http://127.0.0.1:11435/v1}
MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
PROFILE=${BETTERBENCH_PROFILE:-qualification}

RUN_DIR=
CONFIG=
MAX_MODEL_LEN=8192

usage() {
  echo "Usage: $0 --run-dir DIR --config NAME [--max-model-len N]"
}

while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $RUN_DIR && -n $CONFIG ]] || { usage >&2; exit 2; }
[[ $MAX_MODEL_LEN -eq 8192 ]] || {
  echo "The cross-mode BetterBench contract is normalized to MAX_MODEL_LEN=8192" >&2
  exit 2
}
[[ -x ${VENV}/bin/betterbench ]] || {
  echo "BetterBench is not installed in ${VENV}" >&2
  exit 1
}
BETTERBENCH_ROOT=${BETTERBENCH_ROOT:-$("${VENV}/bin/python" -c \
  'import pathlib, betterbench; print(pathlib.Path(betterbench.__file__).resolve().parents[1])')}
[[ -d ${BETTERBENCH_ROOT}/.git ]] || {
  echo "Pinned BetterBench checkout is missing: ${BETTERBENCH_ROOT}" >&2
  exit 1
}
actual_commit=$(git -C "$BETTERBENCH_ROOT" rev-parse HEAD)
[[ $actual_commit == "$BETTERBENCH_COMMIT" ]] || {
  echo "BetterBench commit mismatch: expected ${BETTERBENCH_COMMIT}, got ${actual_commit}" >&2
  exit 1
}
profile_path=${BENCH_ROOT}/betterbench/${PROFILE}.json
[[ -f $profile_path ]] || { echo "Unknown BetterBench profile: ${PROFILE}" >&2; exit 2; }

mkdir -p "$RUN_DIR/betterbench" "$RUN_DIR/logs" "$RUN_DIR/telemetry"
cp "$profile_path" "$RUN_DIR/betterbench/profile.json"
printf '%s\n' "$BETTERBENCH_COMMIT" >"$RUN_DIR/betterbench/commit.txt"
"${VENV}/bin/betterbench" --version >"$RUN_DIR/betterbench/version.txt"

telemetry=${RUN_DIR}/telemetry/betterbench.jsonl
"${SCRIPT_DIR}/telemetry.py" "$telemetry" --interval 1 &
telemetry_pid=$!
cleanup() {
  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

"${VENV}/bin/betterbench" run \
  --endpoint "$BASE_URL" \
  --model "$MODEL_NAME" \
  --config "$profile_path" \
  --corpus "$BETTERBENCH_ROOT/corpus/v1" \
  --max-model-len "$MAX_MODEL_LEN" \
  --out "$RUN_DIR/betterbench/results.json" \
  >"$RUN_DIR/logs/betterbench.log" 2>&1

cleanup
trap - EXIT
"${VENV}/bin/betterbench" report "$RUN_DIR/betterbench/results.json" \
  --out "$RUN_DIR/betterbench/report.md"
