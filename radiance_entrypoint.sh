#!/usr/bin/env bash
# RADIANCE entrypoint: banner + environment preamble, then hand off to `vllm serve`.
# Runs ONCE in the container's main process (not per TP worker). Every check is best-effort:
# nothing here blocks the serve. Replaces the bare ENTRYPOINT ["vllm","serve"]; the container
# `command:` / `docker run` args after the image become "$@" and are passed through verbatim.
set -u

# Fast path: help/version queries skip the GPU preamble entirely.
case "${1:-}" in
  -h|--help|--version|--help-all) exec vllm serve "$@" ;;
esac

# ------------------------------------------------------------------ NUMA binding
# Optional: pin the whole vLLM fleet to the NUMA node(s) local to the visible GPUs. vLLM's
# TP workers inherit the parent's CPU-affinity mask + NUMA memory policy, so wrapping the
# final `vllm serve` in `numactl` binds them all. Enable with `--numa-bind[=SPEC]` (consumed
# here, never forwarded to vLLM) or RADIANCE_NUMA_BIND; the flag wins. Off by default; a no-op
# on single-node hosts. Needs the `numactl` binary and, under Docker's default seccomp,
# `--cap-add SYS_NICE` for the mempolicy syscalls.
#   SPEC: auto (detect GPU-local node) | <nodes> | bind=<nodes> | interleave[=<nodes>]
#         | preferred=<node> | none
_numa_spec=""
_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --numa-bind)   _numa_spec="auto"; shift ;;
    --numa-bind=*) _numa_spec="${1#*=}"; shift ;;
    *)             _args+=("$1"); shift ;;
  esac
done
set -- ${_args[@]+"${_args[@]}"}
[ -z "$_numa_spec" ] && _numa_spec="${RADIANCE_NUMA_BIND:-}"

# Compose-friendly weight-format selection. `auto` deliberately means no CLI
# flag so vLLM reads quantization_config from the checkpoint (required by Quark
# OCP MXFP4). Explicit command-line quantization always wins.
if [ -n "${RADIANCE_WEIGHT_QUANTIZATION:-}" ] \
    && [ "${RADIANCE_WEIGHT_QUANTIZATION,,}" != "auto" ]; then
  _has_weight_quantization=0
  for _arg in "$@"; do
    case "$_arg" in
      --quantization|-q|--quantization=*) _has_weight_quantization=1; break ;;
    esac
  done
  if [ "$_has_weight_quantization" -eq 1 ]; then
    echo "[radiance] WARN RADIANCE_WEIGHT_QUANTIZATION ignored because --quantization was passed explicitly" >&2
  else
    set -- "$@" "--quantization=${RADIANCE_WEIGHT_QUANTIZATION}"
  fi
fi

# Compose-friendly speculative decoding. A raw JSON config in the environment is
# appended as one argv item, so users can opt into MTP/DFlash without editing the
# portable Compose command list. An explicit CLI flag always wins.
if [ -n "${RADIANCE_SPECULATIVE_CONFIG:-}" ]; then
  _has_speculative_config=0
  for _arg in "$@"; do
    case "$_arg" in
      --speculative-config|--speculative-config=*) _has_speculative_config=1; break ;;
    esac
  done
  if [ "$_has_speculative_config" -eq 1 ]; then
    echo "[radiance] WARN RADIANCE_SPECULATIVE_CONFIG ignored because --speculative-config was passed explicitly" >&2
  else
    set -- "$@" "--speculative-config=${RADIANCE_SPECULATIVE_CONFIG}"
  fi
fi

# The DFlash2 profile also needs an explicit graph mode. Keep this optional so
# ordinary Radiance retains vLLM's normal compilation defaults.
if [ -n "${RADIANCE_COMPILATION_CONFIG:-}" ]; then
  _has_compilation_config=0
  for _arg in "$@"; do
    case "$_arg" in
      --compilation-config|--compilation-config=*) _has_compilation_config=1; break ;;
    esac
  done
  if [ "$_has_compilation_config" -eq 1 ]; then
    echo "[radiance] WARN RADIANCE_COMPILATION_CONFIG ignored because --compilation-config was passed explicitly" >&2
  else
    set -- "$@" "--compilation-config=${RADIANCE_COMPILATION_CONFIG}"
  fi
fi

# NUMA node ids local to the *visible* AMD GPUs, PCI-bus-ordered to match HIP enumeration.
_numa_gpu_nodes() {
  local vis n; local -a ordered=() sel=() nodes=()
  while IFS= read -r n; do ordered+=("$n"); done < <(
    for d in /sys/class/drm/renderD*; do
      [ "$(cat "$d/device/vendor" 2>/dev/null)" = "0x1002" ] || continue
      printf '%s\t%s\n' "$(basename "$(readlink -f "$d/device")")" "$d"
    done | sort | cut -f2-)
  [ "${#ordered[@]}" -gt 0 ] || return 0
  vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
  if [[ "$vis" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    local -a idx; IFS=',' read -ra idx <<<"$vis"
    for n in "${idx[@]}"; do [ "$n" -lt "${#ordered[@]}" ] && sel+=("${ordered[$n]}"); done
  else
    sel=("${ordered[@]}")
  fi
  for d in "${sel[@]}"; do
    n=$(cat "$d/device/numa_node" 2>/dev/null); [[ "$n" =~ ^[0-9]+$ ]] && nodes+=("$n")
  done
  [ "${#nodes[@]}" -gt 0 ] && printf '%s\n' "${nodes[@]}" | sort -un
}

NUMACTL=()
if [ -n "$_numa_spec" ]; then
  export RADIANCE_NUMA_BIND="$_numa_spec"      # so the preamble reports it (flag or env)
  case "${_numa_spec,,}" in
    none|off|"") ;;
    *)
      if ! command -v numactl >/dev/null 2>&1; then
        echo "[radiance] WARN --numa-bind='$_numa_spec' but numactl not found; launching unbound" >&2
      else
        case "${_numa_spec,,}" in
          auto)
            mapfile -t _n < <(_numa_gpu_nodes)
            if   [ "${#_n[@]}" -eq 0 ]; then echo "[radiance] --numa-bind=auto: no GPU-local NUMA node (non-NUMA host?); unbound" >&2
            elif [ "${#_n[@]}" -eq 1 ]; then NUMACTL=(--cpunodebind="${_n[0]}" --membind="${_n[0]}")
            else _s=$(IFS=,; echo "${_n[*]}"); NUMACTL=(--cpunodebind="$_s" --localalloc); fi ;;
          all|interleave) NUMACTL=(--interleave=all) ;;
          interleave=*)   NUMACTL=(--interleave="${_numa_spec#*=}") ;;
          preferred=*)    NUMACTL=(--preferred="${_numa_spec#*=}") ;;
          bind=*)         _s="${_numa_spec#*=}"; NUMACTL=(--cpunodebind="$_s" --membind="$_s") ;;
          [0-9]*)         NUMACTL=(--cpunodebind="$_numa_spec" --membind="$_numa_spec") ;;
          *) echo "[radiance] WARN unrecognised --numa-bind='$_numa_spec' (want: auto|<nodes>|bind=<nodes>|interleave[=nodes]|preferred=node|none); unbound" >&2 ;;
        esac
      fi ;;
  esac
fi
[ "${#NUMACTL[@]}" -gt 0 ] && export RADIANCE_NUMA_ACTIVE="numactl ${NUMACTL[*]}"

# Topology + bandwidth sweep, kicked off in the BACKGROUND so it never delays startup.
# rocm-bandwidth-test (no args) prints: device list, inter-device access (P2P) matrix, NUMA
# distance, and uni + bidirectional peak copy bandwidth (GB/s) for every agent pair (d2d, d2h,
# h2d). It surfaces a few seconds into vLLM's startup logs. ON by default (the image ships the
# binary and sets RADIANCE_RUN_BWTEST=1; it costs about a second); set RADIANCE_RUN_BWTEST=0 to
# skip it.
if [ "${RADIANCE_RUN_BWTEST:-0}" = "1" ] && ! command -v rocm-bandwidth-test >/dev/null 2>&1; then
  echo "[radiance] WARN RADIANCE_RUN_BWTEST=1 but rocm-bandwidth-test is not on PATH; skipping the sweep" >&2
fi
if [ "${RADIANCE_RUN_BWTEST:-0}" = "1" ] && command -v rocm-bandwidth-test >/dev/null 2>&1; then
  (
    report=$(timeout 150 rocm-bandwidth-test 2>&1) \
      || report="${report}"$'\n'"(rocm-bandwidth-test exited non-zero / timed out)"
    # same colour opt-out the banner honours, so a log scraper gets plain text everywhere
    _c=$'\033[1;38;5;39m'; _r=$'\033[0m'
    if [ -n "${NO_COLOR:-}" ] || [ "${RADIANCE_BANNER_PLAIN:-0}" = "1" ]; then _c=""; _r=""; fi
    printf '\n%s┌─[ RADIANCE · GPU TOPOLOGY & BANDWIDTH (rocm-bandwidth-test) ]%s%s\n%s\n%s└─[ end bandwidth report ]%s%s\n' \
      "$_c" "$(printf '─%.0s' {1..2})" "$_r" "$report" "$_c" "$(printf '─%.0s' {1..30})" "$_r"
  ) &
fi

# Synchronous banner + arch / P2P / optimizations / versions checks. The serve's own args are
# passed through so the banner can report the attention backend that was actually selected.
python /opt/radiance_preamble.py "$@" || true

# Hand off. exec so vLLM becomes PID 1 and receives signals directly; the backgrounded
# bandwidth job keeps its inherited stdout and prints its report when it finishes. When
# --numa-bind resolved a binding, wrap the exec in numactl (TP workers inherit it).
if [ "${#NUMACTL[@]}" -gt 0 ]; then
  exec numactl "${NUMACTL[@]}" vllm serve "$@"
fi
exec vllm serve "$@"
