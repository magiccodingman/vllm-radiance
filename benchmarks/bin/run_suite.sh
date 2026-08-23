#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VENV=/nvme/ediloca-1/venv/vllm-bench-env
MODEL_HOST=${MODEL_HOST:-/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman}
MODEL_NAME=${MODEL_NAME:-Qwen3.8-27B-heretic-ara-fp8}
BASE_URL=http://127.0.0.1:11435
SEED=20260822
CASE_TIMEOUT=${BENCH_CASE_TIMEOUT:-900}
PREFILL_INPUT_TOKENS=${BENCH_PREFILL_INPUT_TOKENS:-2048}
QUICK_CONTEXT_TOKENS=${BENCH_QUICK_CONTEXT_TOKENS:-8192}

RUN_DIR=
CONFIG=
TP=
SPEC=
CPU_OFFLOAD_GB=0
SUITE=quick
MAX_MODEL_LEN=16384

usage() {
  echo "Usage: $0 --run-dir DIR --config NAME --tp 1|2 --spec on|off [--max-model-len N] [--cpu-offload-gb N] [--suite smoke|quick|standard|qualification]"
}

while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --spec) SPEC=$2; shift 2 ;;
    --cpu-offload-gb) CPU_OFFLOAD_GB=$2; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
    --suite) SUITE=$2; shift 2 ;;
    --base-url) BASE_URL=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $RUN_DIR && -n $CONFIG && -n $TP && -n $SPEC ]] || { usage >&2; exit 2; }
[[ $TP == 1 || $TP == 2 ]] || { echo "--tp must be 1 or 2" >&2; exit 2; }
[[ $SPEC == on || $SPEC == off ]] || { echo "--spec must be on or off" >&2; exit 2; }
[[ $SUITE == smoke || $SUITE == quick || $SUITE == standard || $SUITE == qualification ]] || {
  echo "--suite must be smoke, quick, standard, or qualification" >&2
  exit 2
}

for target in /nvme/ediloca-1 /nvme/lexar-2; do
  findmnt -T "$target" >/dev/null || { echo "Required mount missing: $target" >&2; exit 1; }
done
[[ -x ${VENV}/bin/python ]] || { echo "Benchmark venv is missing: $VENV" >&2; exit 1; }
[[ -f ${MODEL_HOST}/config.json ]] || { echo "Model is missing: $MODEL_HOST" >&2; exit 1; }
curl --fail --silent --show-error "${BASE_URL}/v1/models" >/dev/null

RAW_DIR=${RUN_DIR}/raw
LOG_DIR=${RUN_DIR}/logs
TELEMETRY_DIR=${RUN_DIR}/telemetry
mkdir -p "$RAW_DIR" "$LOG_DIR" "$TELEMETRY_DIR"

run_one() {
  local workload=$1
  local input_tokens=$2
  local output_tokens=$3
  local concurrency=$4
  local repetition=$5
  local prompts=$6
  local warmups=$7
  local stem
  stem=$(printf '%s_in%s_out%s_c%s_r%s' "$workload" "$input_tokens" "$output_tokens" "$concurrency" "$repetition")
  local result=${RAW_DIR}/${stem}.json
  local log=${LOG_DIR}/${stem}.log
  local warmup_log=${LOG_DIR}/${stem}.shape-warmup.log
  local telemetry=${TELEMETRY_DIR}/${stem}.jsonl
  # Repetitions and shapes get stable but distinct prompts. Reusing one global
  # seed makes prefix caching leak across samples; using the benchmark client's
  # built-in warmup is worse because it reuses the measurement's test prompt.
  local case_seed=$((SEED + input_tokens * 31 + output_tokens * 17 + concurrency * 101 + repetition * 1009))
  local warmup_seed=$((case_seed + 500000))

  local common_args=(
    --backend openai
    --base-url "$BASE_URL"
    --endpoint /v1/completions
    --model "$MODEL_NAME"
    --tokenizer "$MODEL_HOST"
    --dataset-name random
    --input-len "$input_tokens"
    --output-len "$output_tokens"
    --random-range-ratio 0.0
    --max-concurrency "$concurrency"
    --request-rate inf
    --ignore-eos
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,90,95,99
    --disable-tqdm
  )

  if ((warmups > 0)); then
    echo "[$(date -u +%FT%TZ)] ${CONFIG}: shape warmup, in=${input_tokens}, out=${output_tokens}, c=${concurrency}, prompts=${warmups}"
    if ! timeout --signal=TERM --kill-after=30 "${CASE_TIMEOUT}" \
      "${VENV}/bin/python" "${SCRIPT_DIR}/vllm_bench_serve.py" \
      "${common_args[@]}" --num-prompts "$warmups" --num-warmups 0 \
      --seed "$warmup_seed" >"$warmup_log" 2>&1; then
      echo "Shape warmup failed (${stem}); see ${warmup_log}" >&2
      return 1
    fi
  fi

  echo "[$(date -u +%FT%TZ)] ${CONFIG}: ${workload}, in=${input_tokens}, out=${output_tokens}, c=${concurrency}, rep=${repetition}, prompts=${prompts}"
  "${SCRIPT_DIR}/telemetry.py" "$telemetry" --interval 1 &
  local telemetry_pid=$!
  set +e
  timeout --signal=TERM --kill-after=30 "${CASE_TIMEOUT}" \
    "${VENV}/bin/python" "${SCRIPT_DIR}/vllm_bench_serve.py" \
    "${common_args[@]}" \
    --num-prompts "$prompts" \
    --num-warmups 0 \
    --seed "$case_seed" \
    --save-result \
    --save-detailed \
    --result-filename "$result" \
    --metadata "config=${CONFIG}" "workload=${workload}" "input_tokens=${input_tokens}" \
      "output_tokens=${output_tokens}" "repetition=${repetition}" "tp=${TP}" \
      "spec=${SPEC}" "cpu_offload_gb=${CPU_OFFLOAD_GB}" "seed=${case_seed}" \
    >"$log" 2>&1
  local status=$?
  set -e
  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  if ((status != 0)); then
    echo "Benchmark failed (${stem}); see ${log}" >&2
    return "$status"
  fi
  "${SCRIPT_DIR}/summarize.py" "$(dirname -- "$RUN_DIR")"
}

if [[ $SUITE == smoke ]]; then
  run_one smoke 128 64 1 1 2 1
  exit 0
fi

# Warm the model and compile any remaining request-path kernels once. Individual
# measurements deliberately do not repeat warmups.
run_one warmup 128 64 1 1 2 1

# Everyday A/B gate: two waves at each concurrency and two repetitions. At the
# slowest known BF16 baseline this is about four minutes of measured decoding;
# native FP8 is faster. A third repetition belongs in standard/qualification.
decode_concurrencies=(1 2 4 8)
# TP1 is a constrained reference for large models, not a VRAM saturation test.
# Stop at c4 so hybrid-state/KV capacity does not dominate the kernel result.
((TP == 1)) && decode_concurrencies=(1 2 4)
for concurrency in "${decode_concurrencies[@]}"; do
  prompts=$((concurrency * 2))
  ((prompts < 4)) && prompts=4
  for repetition in 1 2; do
    # Warm the exact batch shape before its first measured repetition. Current
    # vLLM reports otherwise-lazy Triton JITs (notably c2 GDN decode) during
    # inference, which can turn a compile pause into a fake performance delta.
    warmups=0
    ((repetition == 1)) && warmups=$concurrency
    run_one decode 256 256 "$concurrency" "$repetition" "$prompts" "$warmups"
  done
done

# A bounded prefill/mixed sweep captures TTFT and prompt throughput without
# turning every iterative comparison into a soak test.
prefill_concurrencies=(1 4 8)
((TP == 1)) && prefill_concurrencies=(1)
for concurrency in "${prefill_concurrencies[@]}"; do
  prompts=$((concurrency * 2))
  ((prompts < 2)) && prompts=2
  # Prefill has a distinct GDN/attention kernel family, so the short decode
  # warmup above cannot make this measurement hot.
  run_one prefill "$PREFILL_INPUT_TOKENS" 64 "$concurrency" 1 "$prompts" "$concurrency"
done

# Long-context cases are capacity/correctness checks, not stability repetitions.
if ((TP == 1)); then
  context_input=$((MAX_MODEL_LEN - 256))
  run_one context_tp1 "$context_input" 64 1 1 1 1
else
  context_input=$QUICK_CONTEXT_TOKENS
  ((context_input + 64 > MAX_MODEL_LEN)) && context_input=$((MAX_MODEL_LEN - 64))
  run_one context_quick "$context_input" 64 1 1 1 1
fi

if [[ $SUITE == quick ]]; then
  exit 0
fi

# Standard adds a third decode sample and another mixed-workload repetition.
for concurrency in "${decode_concurrencies[@]}"; do
  prompts=$((concurrency * 2))
  ((prompts < 4)) && prompts=4
  run_one decode 256 256 "$concurrency" 3 "$prompts" 0
done
for concurrency in "${prefill_concurrencies[@]}"; do
  prompts=$((concurrency * 2))
  ((prompts < 2)) && prompts=2
  run_one prefill "$PREFILL_INPUT_TOKENS" 64 "$concurrency" 2 "$prompts" 0
done

if [[ $SUITE == standard ]]; then
  exit 0
fi

# Qualification is reserved for milestone builds: longer steady decode plus a
# concurrent near-envelope context check. It remains bounded and avoids the old
# 6-8 hour matrix while adapting to an explicitly larger capacity envelope.
for concurrency in "${decode_concurrencies[@]}"; do
  prompts=$((concurrency * 2))
  ((prompts < 4)) && prompts=4
  run_one sustained_decode 256 512 "$concurrency" 1 "$prompts" 0
done
if ((TP == 2)); then
  if ((MAX_MODEL_LEN >= 8192)); then
    context_input=$((MAX_MODEL_LEN - 256))
    run_one context_envelope "$context_input" 64 2 1 2 1
  fi
  if ((MAX_MODEL_LEN >= 32768)); then
    run_one context_32k 31744 64 1 1 1 0
  fi
fi
