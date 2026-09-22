#!/usr/bin/env bash
# Matched publication lanes; delegates all workload/warmup/reporting to BetterBench.
set -Eeuo pipefail
lane=${1:?Usage: run_v030_publication.sh non-spec|mtp-k4|dflash-k5|dflash-k7 RUN_ROOT IMAGE}
root=${2:?RUN_ROOT required}
image=${3:?IMAGE required}
export COMPOSE_PROJECT_NAME=radiance-v030-publication
export MODEL_HOST=/nvme/lexar-2/ai/models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd
export MODEL_NAME=Qwen3.8-27B-Quark-AWQ-MXFP4
export WEIGHT_QUANTIZATION=auto KV_CACHE_DTYPE=fp8
export BETTERBENCH_PROFILE=standard BENCH_TOOL_SCHEMA_ATTEMPTS=30 BENCH_PLATFORM_INSPECT=1
export PREFIX_CACHING=off MAMBA_CACHE_MODE=none VLLM_USE_V2_MODEL_RUNNER=1
export RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0
export RADIANCE_MXFP4_WPERM=1 RADIANCE_MXFP4_DECODE_NT=1
export RADIANCE_MXFP4_DECODE_MAX_M=64 RADIANCE_MXFP4_TN4_MIN_M=2048
export RADIANCE_MXFP4_A_TILED_MIN_M=0 RADIANCE_GDN_NORM_QUANT=0
export RADIANCE_NORMQUANT_FUSION=0 RADIANCE_FP8_STREAM=0
export RADIANCE_DYNAMIC_DRAFT=0 RADIANCE_DYNAMIC_WIDTH=0 RADIANCE_DRAFT_RERANK=64
export RADIANCE_VERIFY_HEAD=1 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=4096
export COMPILATION_CONFIG_JSON='{"cudagraph_mode":"PIECEWISE"}'
export CONTAINER_VLLM_CACHE_ROOT="/cache/v030-publication/$(basename "$root")/$lane/vllm"
export CONTAINER_TORCHINDUCTOR_CACHE_DIR="/cache/v030-publication/$(basename "$root")/$lane/inductor"
spec=on
case "$lane" in
  non-spec)
    spec=off
    export RADIANCE_FAST_DRAFT=0 SPECULATIVE_CONFIG_JSON=''
    ;;
  mtp-k4)
    export RADIANCE_FAST_DRAFT=1 RADIANCE_QUARK_BF16_MTP=1
    export SPECULATIVE_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true}'
    ;;
  dflash-k5|dflash-k7)
    depth=${lane##*-k}
    export RADIANCE_FAST_DRAFT=1
    export SPECULATIVE_CONFIG_JSON="{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-FP8-tcclaviger\",\"num_speculative_tokens\":$depth,\"draft_tensor_parallel_size\":2,\"attention_backend\":\"TRITON_ATTN\",\"max_model_len\":8192,\"disable_padded_drafter_batch\":true}"
    ;;
  *) echo "Unknown publication lane: $lane" >&2; exit 2 ;;
esac
test ! -e "$root/$lane" || { echo "Refusing to overwrite existing lane: $root/$lane" >&2; exit 2; }
exec "$(dirname "$0")/run_configuration.sh" --run-root "$root" --label "$lane" \
  --tp 2 --spec "$spec" --image "$image" --max-model-len 8192 --suite betterbench \
  --notes "v0.30 publication; fixed depth, safe WPERM/NT, cold nonce corpus; no kernel tuning."
