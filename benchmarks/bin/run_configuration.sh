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
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-R4D}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-hermes}
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
if [[ ${RADIANCE_MXFP4:-0} == 1 || ${RADIANCE_MXFP4_W4A8:-0} == 1 ]]; then
  [[ $WEIGHT_QUANTIZATION == auto ]] || {
    echo "MXFP4 requires WEIGHT_QUANTIZATION=auto so vLLM reads checkpoint Quark metadata" >&2
    exit 2
  }
  jq -e '.quantization_config.quant_method == "quark"' "${MODEL_HOST}/config.json" >/dev/null || {
    echo "MXFP4 profile requires a Quark checkpoint: $MODEL_HOST" >&2
    exit 1
  }
  [[ ${RADIANCE_MXFP4:-0} == 1 && ${RADIANCE_MXFP4_W4A8:-0} == 1 ]] || {
    echo "Qualified gfx1201 MXFP4 requires both RADIANCE_MXFP4=1 and RADIANCE_MXFP4_W4A8=1" >&2
    exit 2
  }
  [[ ${RADIANCE_MXFP4_W4A8_MIN_M:-0} == 0 ]] || {
    echo "Refusing unsafe MXFP4 benchmark: RADIANCE_MXFP4_W4A8_MIN_M must be 0" >&2
    exit 2
  }
  if [[ ${SPECULATIVE_CONFIG_JSON:-} == *'"method":"mtp"'* ]]; then
    [[ ${RADIANCE_QUARK_BF16_MTP:-0} == 1 ]] || {
      echo "Quark MXFP4 MTP requires RADIANCE_QUARK_BF16_MTP=1 for the qualified AMD checkpoint" >&2
      exit 2
    }
  fi
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
  if [[ ${RADIANCE_NORMQUANT_FUSION:-0} == 1 ]]; then
    export RADIANCE_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
  else
    server_args+=('--compilation-config={"cudagraph_mode":"NONE"}')
  fi
elif [[ -n $COMPILATION_CONFIG_JSON ]]; then
  if [[ ${RADIANCE_NORMQUANT_FUSION:-0} == 1 ]]; then
    # RX4 must merge its pass_config into the same JSON object. Pass the base
    # object through the entrypoint instead of adding an explicit CLI flag,
    # because argparse keeps only one --compilation-config value.
    export RADIANCE_COMPILATION_CONFIG=$COMPILATION_CONFIG_JSON
  else
    server_args+=("--compilation-config=${COMPILATION_CONFIG_JSON}")
  fi
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
  "--tool-call-parser=${TOOL_CALL_PARSER}"
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

manifest_draft_args=()
if [[ $SPEC == on && -n $SPECULATIVE_CONFIG_JSON ]]; then
  draft_container_path=$(jq -er '.model // empty' <<<"$SPECULATIVE_CONFIG_JSON" 2>/dev/null || true)
  if [[ $draft_container_path == /models/* ]]; then
    DRAFT_MODEL_HOST="$(dirname "$MODEL_HOST")/${draft_container_path#/models/}"
    [[ -f ${DRAFT_MODEL_HOST}/config.json ]] || {
      echo "Resolved draft model is missing: ${DRAFT_MODEL_HOST}" >&2
      exit 1
    }
    manifest_draft_args=(--draft-model-host "$DRAFT_MODEL_HOST")
  fi
fi

HIP_VISIBLE_DEVICES=$gpu_devices RADIANCE_IMAGE="$IMAGE" "${SCRIPT_DIR}/capture_manifest.py" \
  --output "${CONFIG_DIR}/manifest.json" \
  --label "$LABEL" --tp "$TP" --spec "$SPEC" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --container "$container" --image "$IMAGE" --model-host "$MODEL_HOST" \
  "${manifest_draft_args[@]}" \
  --suite "$SUITE" --kv-cache-dtype "$KV_CACHE_DTYPE" --max-model-len "$MAX_MODEL_LEN" \
  --weight-quantization "$WEIGHT_QUANTIZATION" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --workload-filter "$WORKLOAD_FILTER" \
  --enforce-eager "$ENFORCE_EAGER" --disable-cudagraph "$DISABLE_CUDAGRAPH" --notes "$NOTES"

if [[ $SUITE == betterbench ]]; then
  MODEL_NAME="$MODEL_NAME" "${SCRIPT_DIR}/run_betterbench.sh" \
    --run-dir "$CONFIG_DIR" --config "$LABEL" --max-model-len "$MAX_MODEL_LEN"
  # Keep the publication-grade performance suite and the strict output gate in
  # the same immutable run directory. Run this after BetterBench so fixed
  # prompts cannot warm or otherwise perturb the measured corpus.
  "${SCRIPT_DIR}/run_correctness.py" \
    --base-url http://127.0.0.1:11435 \
    --model "$MODEL_NAME" \
    --prompts "${BENCH_ROOT}/fixtures/correctness-prompts.json" \
    --output "${CONFIG_DIR}/raw/correctness_fixed.json" \
    --max-tokens 128 --repetitions 1 --logprobs 5
else
  MODEL_HOST="$MODEL_HOST" MODEL_NAME="$MODEL_NAME" BENCH_WORKLOADS="$WORKLOAD_FILTER" "${SCRIPT_DIR}/run_suite.sh" \
    --run-dir "$CONFIG_DIR" --config "$LABEL" --tp "$TP" --spec "$SPEC" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" --max-model-len "$MAX_MODEL_LEN" --suite "$SUITE"
fi

# Optional structured-tool regression gate. It is deliberately explicit so
# ordinary throughput iteration does not acquire 30 extra requests, while a
# milestone run can bind parser correctness to the same live server, manifest,
# image, and immutable configuration directory.
if [[ ${BENCH_TOOL_SCHEMA_ATTEMPTS:-0} =~ ^[1-9][0-9]*$ ]]; then
  BASE_URL=http://127.0.0.1:11435/v1 MODEL_NAME="$MODEL_NAME" \
    ATTEMPTS="$BENCH_TOOL_SCHEMA_ATTEMPTS" \
    RUN_ID="${LABEL}-tool-schema" RUN_DIR="${CONFIG_DIR}/tool-schema-gate" \
    "${SCRIPT_DIR}/run_tool_schema_gate.sh"
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
