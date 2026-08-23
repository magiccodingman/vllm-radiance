#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNS_DIR=${BENCH_RUNS_DIR:-${BENCH_ROOT}/runs}
IMAGE=${RADIANCE_IMAGE:-vllm-radiance:dev}
SUITE=${BENCH_SUITE:-quick}
# The everyday gate is deliberately non-speculative. Add tp2_spec-on explicitly
# at milestone checkpoints; TP1 MTP does not fit this 27B model without offload.
CONFIGS=${BENCH_CONFIGS:-tp2_spec-off,tp1-eager8k_spec-off}
IMAGE_SLUG=$(printf '%s' "$IMAGE" | tr '/:@' '---' | tr -cd 'a-zA-Z0-9_.-')
MODEL_SLUG=$(basename "${MODEL_HOST:-/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman}" | tr -cd 'a-zA-Z0-9_.-')
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)_${IMAGE_SLUG}_${MODEL_SLUG}_${SUITE}
RUN_ROOT=${RUNS_DIR}/${RUN_ID}
NOTES=${BENCH_NOTES:-"Bounded native-weight baseline with mandatory FP8 KV cache; ${SUITE} profile."}

mkdir -p "$RUN_ROOT"
printf '%s\n' "$NOTES" >"${RUN_ROOT}/notes.md"
printf '%s\n' running >"${RUN_ROOT}/status.txt"
printf '%s\n' "$RUN_ROOT"

status=0
IFS=, read -r -a selected <<<"$CONFIGS"
for config in "${selected[@]}"; do
  case "$config" in
    tp2_spec-off)         tp=2; spec=off; util=${TP2_GPU_UTIL:-0.85}; max_len=${TP2_MAX_MODEL_LEN:-16384}; eager=0 ;;
    tp1-eager8k_spec-off) tp=1; spec=off; util=${TP1_GPU_UTIL:-0.95}; max_len=${TP1_MAX_MODEL_LEN:-8192}; eager=1 ;;
    tp2_spec-on)          tp=2; spec=on;  util=${TP2_GPU_UTIL:-0.85}; max_len=${TP2_MAX_MODEL_LEN:-16384}; eager=0 ;;
    tp1-eager8k_spec-on)  tp=1; spec=on;  util=${TP1_GPU_UTIL:-0.95}; max_len=${TP1_MAX_MODEL_LEN:-8192}; eager=1 ;;
    *) echo "Unknown BENCH_CONFIGS entry: $config" >&2; exit 2 ;;
  esac
  graph_args=()
  ((eager)) && graph_args+=(--enforce-eager)
  "${SCRIPT_DIR}/run_configuration.sh" --run-root "$RUN_ROOT" \
    --label "$config" --tp "$tp" --spec "$spec" --image "$IMAGE" \
    --gpu-memory-utilization "$util" --max-model-len "$max_len" \
    "${graph_args[@]}" --suite "$SUITE" --notes "$NOTES" || status=1
done

"${SCRIPT_DIR}/summarize.py" "$RUN_ROOT"
if ((status == 0)); then
  printf '%s\n' completed >"${RUN_ROOT}/status.txt"
else
  printf '%s\n' partial-failure >"${RUN_ROOT}/status.txt"
fi
exit "$status"
