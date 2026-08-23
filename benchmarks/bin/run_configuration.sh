#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
COMPOSE_FILE=${BENCH_ROOT}/compose.yaml
IMAGE=${RADIANCE_IMAGE:-vllm-radiance:dev}
MODEL_HOST=${MODEL_HOST:-/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman}
MODEL=/models/$(basename "$MODEL_HOST")
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_HOST")}
WEIGHT_QUANTIZATION=${WEIGHT_QUANTIZATION:-fp8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
ALLOW_DIAGNOSTIC_NON_FP8_KV=${ALLOW_DIAGNOSTIC_NON_FP8_KV:-0}
GPU_MEMORY_UTILIZATION=0.85
READINESS_TIMEOUT=3600
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-R4D}
ADDITIONAL_CONFIG_JSON=${ADDITIONAL_CONFIG_JSON:-}
SPECULATIVE_CONFIG_JSON=${SPECULATIVE_CONFIG_JSON:-}
COMPILATION_CONFIG_JSON=${COMPILATION_CONFIG_JSON:-}
PREFIX_CACHING=${PREFIX_CACHING:-default}
MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-}
WORKLOAD_FILTER=${BENCH_WORKLOADS:-all}
ENFORCE_EAGER=0
DISABLE_CUDAGRAPH=0

RUN_ROOT=
LABEL=
TP=
SPEC=
CPU_OFFLOAD_GB=0
NOTES=
SUITE=quick

usage() {
  echo "Usage: $0 --run-root DIR --label NAME --tp 1|2 --spec on|off [--image REF] [--max-model-len N] [--disable-cudagraph|--enforce-eager] [--cpu-offload-gb N] [--notes TEXT] [--suite smoke|quick|standard|qualification|betterbench]"
}

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --spec) SPEC=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER=1; shift ;;
    --disable-cudagraph) DISABLE_CUDAGRAPH=1; shift ;;
    --cpu-offload-gb) CPU_OFFLOAD_GB=$2; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION=$2; shift 2 ;;
    --notes) NOTES=$2; shift 2 ;;
    --suite) SUITE=$2; shift 2 ;;
    --readiness-timeout) READINESS_TIMEOUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $RUN_ROOT && -n $LABEL && -n $TP && -n $SPEC ]] || { usage >&2; exit 2; }
[[ $TP == 1 || $TP == 2 ]] || { echo "--tp must be 1 or 2" >&2; exit 2; }
[[ $SPEC == on || $SPEC == off ]] || { echo "--spec must be on or off" >&2; exit 2; }
[[ $SUITE == smoke || $SUITE == quick || $SUITE == standard || $SUITE == qualification || $SUITE == betterbench ]] || {
  echo "Invalid suite: $SUITE" >&2
  exit 2
}

for target in /nvme/ediloca-1 /nvme/lexar-1 /nvme/lexar-2; do
  findmnt -T "$target" >/dev/null || { echo "Required mount missing: $target" >&2; exit 1; }
done
export RENDER_GID=${RENDER_GID:-$(getent group render | cut -d: -f3)}
export VIDEO_GID=${VIDEO_GID:-$(getent group video | cut -d: -f3)}
[[ -n $RENDER_GID && -n $VIDEO_GID ]] || {
  echo "Could not resolve render/video group IDs" >&2
  exit 1
}
[[ $(docker info --format '{{.DockerRootDir}}') == /nvme/lexar-1/docker/data ]] || {
  echo "Docker root is not on /nvme/lexar-1" >&2
  exit 1
}
[[ -f ${MODEL_HOST}/config.json ]] || { echo "Model is missing: $MODEL_HOST" >&2; exit 1; }
if [[ $WEIGHT_QUANTIZATION == fp8 ]]; then
  jq -e '.quantization_config.quant_method == "fp8"' "${MODEL_HOST}/config.json" >/dev/null || {
    echo "Model does not declare native FP8 quantization: $MODEL_HOST" >&2
    exit 1
  }
fi
if [[ $KV_CACHE_DTYPE != fp8 && $ALLOW_DIAGNOSTIC_NON_FP8_KV != 1 ]]; then
  echo "Non-FP8 KV is diagnostic-only; set ALLOW_DIAGNOSTIC_NON_FP8_KV=1 explicitly" >&2
  exit 2
fi
case "$PREFIX_CACHING" in
  default|on|off) ;;
  *) echo "PREFIX_CACHING must be default, on, or off" >&2; exit 2 ;;
esac

CONFIG_DIR=${RUN_ROOT}/${LABEL}
mkdir -p "$CONFIG_DIR" "${CONFIG_DIR}/logs"
printf '%s\n' "$NOTES" >"${CONFIG_DIR}/notes.md"

container=radiance-bench-${LABEL//[^a-zA-Z0-9_.-]/-}
container=${container:0:63}
gpu_devices=0,1
((TP == 1)) && gpu_devices=0

server_args=(
  "$MODEL"
  "--served-model-name=${MODEL_NAME}"
  "--kv-cache-dtype=${KV_CACHE_DTYPE}"
  "--tensor-parallel-size=${TP}"
  "--gpu-memory-utilization=${GPU_MEMORY_UTILIZATION}"
  "--max-model-len=${MAX_MODEL_LEN}"
  "--max-num-seqs=${MAX_NUM_SEQS}"
  "--max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS}"
  "--attention-backend=${ATTENTION_BACKEND}"
)
if [[ $WEIGHT_QUANTIZATION != auto ]]; then
  server_args+=("--quantization=${WEIGHT_QUANTIZATION}")
fi
if [[ $SPEC == on ]]; then
  if [[ -z $SPECULATIVE_CONFIG_JSON ]]; then
    SPECULATIVE_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":8,"attention_backend":"R4D","disable_padded_drafter_batch":true}'
  fi
  server_args+=("--speculative-config=${SPECULATIVE_CONFIG_JSON}")
fi
if [[ -n $ADDITIONAL_CONFIG_JSON ]]; then
  server_args+=("--additional-config=${ADDITIONAL_CONFIG_JSON}")
fi
if [[ $PREFIX_CACHING == on ]]; then
  server_args+=(--enable-prefix-caching)
elif [[ $PREFIX_CACHING == off ]]; then
  server_args+=(--no-enable-prefix-caching)
fi
if [[ -n $MAMBA_CACHE_MODE ]]; then
  server_args+=("--mamba-cache-mode=${MAMBA_CACHE_MODE}")
fi
if ((ENFORCE_EAGER)); then
  server_args+=(--enforce-eager)
fi
if ((DISABLE_CUDAGRAPH)); then
  [[ -z $COMPILATION_CONFIG_JSON ]] || {
    echo "DISABLE_CUDAGRAPH and COMPILATION_CONFIG_JSON are mutually exclusive" >&2
    exit 2
  }
  server_args+=('--compilation-config={"cudagraph_mode":"NONE"}')
elif [[ -n $COMPILATION_CONFIG_JSON ]]; then
  server_args+=("--compilation-config=${COMPILATION_CONFIG_JSON}")
fi
if [[ $CPU_OFFLOAD_GB != 0 && $CPU_OFFLOAD_GB != 0.0 ]]; then
  server_args+=("--cpu-offload-gb=${CPU_OFFLOAD_GB}")
fi
server_args+=(
  --no-async-scheduling
  --host=0.0.0.0
  --port=8000
  --enable-tokenizer-info-endpoint
  --enable-auto-tool-choice
  --tool-call-parser=hermes
  --language-model-only
)

printf '%q ' "${server_args[@]}" >"${CONFIG_DIR}/server-command.txt"
printf '\n' >>"${CONFIG_DIR}/server-command.txt"

cleanup_done=0
final_status=failed
cleanup() {
  if ((cleanup_done)); then
    return
  fi
  docker logs "$container" >"${CONFIG_DIR}/logs/server.log" 2>&1 || true
  docker stop --time 30 "$container" >/dev/null 2>&1 || true
  docker rm "$container" >/dev/null 2>&1 || true
  printf '%s\n' "$final_status" >"${CONFIG_DIR}/status.txt"
  cleanup_done=1
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" down --remove-orphans >/dev/null 2>&1 || true
docker rm "$container" >/dev/null 2>&1 || true

echo "[$(date -u +%FT%TZ)] Starting ${LABEL} (${IMAGE})"
container_env=(-e "HIP_VISIBLE_DEVICES=${gpu_devices}")
if [[ -n ${VLLM_USE_V2_MODEL_RUNNER:-} ]]; then
  container_env+=(-e "VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER}")
fi
RADIANCE_IMAGE="$IMAGE" docker compose -f "$COMPOSE_FILE" run --detach --no-deps --service-ports \
  --name "$container" \
  "${container_env[@]}" \
  vllm-radiance "${server_args[@]}" >"${CONFIG_DIR}/container-id.txt"

start_epoch=$(date +%s)
while true; do
  if curl --fail --silent http://127.0.0.1:11435/v1/models >"${CONFIG_DIR}/models.json"; then
    break
  fi
  if [[ $(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null || true) != true ]]; then
    docker logs "$container" >"${CONFIG_DIR}/logs/server.log" 2>&1 || true
    echo "Server exited before becoming ready; see ${CONFIG_DIR}/logs/server.log" >&2
    exit 1
  fi
  now=$(date +%s)
  if ((now - start_epoch >= READINESS_TIMEOUT)); then
    docker logs "$container" >"${CONFIG_DIR}/logs/server.log" 2>&1 || true
    echo "Server readiness timed out after ${READINESS_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
done

ready_epoch=$(date +%s)
printf '%s\n' "$((ready_epoch - start_epoch))" >"${CONFIG_DIR}/startup-seconds.txt"
docker logs "$container" >"${CONFIG_DIR}/logs/server-ready.log" 2>&1 || true

docker inspect --format '{{json .Config.Cmd}}' "$container" >"${CONFIG_DIR}/container-command.json"
jq -e --arg expected "--kv-cache-dtype=${KV_CACHE_DTYPE}" 'index($expected) != null' \
  "${CONFIG_DIR}/container-command.json" >/dev/null || {
  echo "Refusing to benchmark: container command does not contain ${KV_CACHE_DTYPE} KV cache" >&2
  exit 1
}

HIP_VISIBLE_DEVICES=$gpu_devices RADIANCE_IMAGE="$IMAGE" "${SCRIPT_DIR}/capture_manifest.py" \
  --output "${CONFIG_DIR}/manifest.json" \
  --label "$LABEL" --tp "$TP" --spec "$SPEC" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --container "$container" --image "$IMAGE" --model-host "$MODEL_HOST" \
  --suite "$SUITE" --kv-cache-dtype "$KV_CACHE_DTYPE" --max-model-len "$MAX_MODEL_LEN" \
  --weight-quantization "$WEIGHT_QUANTIZATION" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --workload-filter "$WORKLOAD_FILTER" \
  --enforce-eager "$ENFORCE_EAGER" --disable-cudagraph "$DISABLE_CUDAGRAPH" --notes "$NOTES"

if [[ $SUITE == betterbench ]]; then
  MODEL_NAME="$MODEL_NAME" "${SCRIPT_DIR}/run_betterbench.sh" \
    --run-dir "$CONFIG_DIR" --config "$LABEL" --max-model-len "$MAX_MODEL_LEN"
else
  MODEL_HOST="$MODEL_HOST" MODEL_NAME="$MODEL_NAME" BENCH_WORKLOADS="$WORKLOAD_FILTER" "${SCRIPT_DIR}/run_suite.sh" \
    --run-dir "$CONFIG_DIR" --config "$LABEL" --tp "$TP" --spec "$SPEC" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" --max-model-len "$MAX_MODEL_LEN" --suite "$SUITE"
fi

docker logs "$container" >"${CONFIG_DIR}/logs/server.log" 2>&1 || true
curl --fail --silent http://127.0.0.1:11435/metrics >"${CONFIG_DIR}/metrics-final.prom"
if [[ $SUITE != betterbench ]]; then
  "${SCRIPT_DIR}/summarize.py" "$RUN_ROOT"
else
  "${SCRIPT_DIR}/summarize.py" --telemetry-only "$RUN_ROOT"
fi
final_status=completed
cleanup
trap - EXIT
find "$CONFIG_DIR" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${CONFIG_DIR}/SHA256SUMS"
echo "[$(date -u +%FT%TZ)] Completed ${LABEL}"
