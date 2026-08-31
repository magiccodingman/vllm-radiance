#!/usr/bin/env python3
"""Capture stable software, hardware, model, and command metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_metadata(model: Path, filename: str) -> tuple[str | None, str | None]:
    """Return the pinned HF revision and content OID without rereading huge weights."""
    metadata = model / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    if not metadata.is_file():
        return None, None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    revision = lines[0] if lines else None
    oid = lines[1] if len(lines) > 1 else None
    return revision, oid


def model_record(model: Path) -> dict[str, object]:
    weights = []
    revisions: set[str] = set()
    for path in sorted(model.glob("*.safetensors")):
        revision, oid = hf_metadata(model, path.name)
        if revision:
            revisions.add(revision)
        weights.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "hf_content_oid": oid,
            }
        )
    config_revision, config_oid = hf_metadata(model, "config.json")
    if config_revision:
        revisions.add(config_revision)
    return {
        "path": str(model),
        "hf_revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "config_sha256": sha256(model / "config.json"),
        "config_hf_oid": config_oid,
        "index_sha256": sha256(model / "model.safetensors.index.json"),
        "weights": weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--spec", choices=("on", "off"), required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-host", type=Path, required=True)
    parser.add_argument("--draft-model-host", type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--enforce-eager", type=int, choices=(0, 1), required=True)
    parser.add_argument("--disable-cudagraph", type=int, choices=(0, 1), required=True)
    parser.add_argument("--weight-quantization", required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--workload-filter", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    model = args.model_host
    project = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "notes": args.notes,
        "runtime": {
            "tensor_parallel_size": args.tp,
            "speculative_decode": args.spec,
            "cpu_offload_gb": args.cpu_offload_gb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "suite": args.suite,
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_model_len": args.max_model_len,
            "enforce_eager": bool(args.enforce_eager),
            "disable_cudagraph": bool(args.disable_cudagraph),
            "weight_quantization": args.weight_quantization,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "workload_filter": args.workload_filter,
            "container": args.container,
            "image": args.image,
            "image_inspect": command("docker", "image", "inspect", args.image),
            "container_inspect": command("docker", "inspect", args.container),
        },
        "model": model_record(model),
        "draft_model": (
            model_record(args.draft_model_host) if args.draft_model_host else None
        ),
        "project": {
            "path": str(project),
            "compose_sha256": sha256(project / "compose.yaml"),
            "resolved_compose": command(
                "docker", "compose", "-f", str(project / "compose.yaml"), "config"
            ),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "cpu": command("lscpu"),
            "memory": command("free", "-b"),
            "mounts": command(
                "findmnt", "-T", "/nvme/ediloca-1", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"
            )
            + "\n"
            + command(
                "findmnt", "-T", "/nvme/lexar-1", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"
            )
            + "\n"
            + command(
                "findmnt", "-T", "/nvme/lexar-2", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"
            ),
            "gpu_products": command("rocm-smi", "--showproductname"),
            "gpu_vbios": command("rocm-smi", "--showvbios"),
            "gpu_firmware": command("rocm-smi", "--showfwinfo"),
            "gpu_topology": command("rocm-smi", "--showtopo"),
            "docker_version": command("docker", "version"),
            "docker_root": command("docker", "info", "--format", "{{.DockerRootDir}}"),
            "rocm_version": command("sh", "-c", "test -f /opt/rocm/.info/version && sed -n '1p' /opt/rocm/.info/version"),
        },
        "environment": {
            key: os.environ[key]
            for key in (
                "HIP_VISIBLE_DEVICES",
                "HIP_FORCE_DEV_KERNARG",
                "TORCH_BLAS_PREFER_HIPBLASLT",
                "VLLM_ROCM_USE_AITER",
                "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION",
                "VLLM_ROCM_USE_AITER_LINEAR",
                "RADIANCE_PRESHUFFLE",
                "RADIANCE_ATTN_TUNE",
                "RADIANCE_USE_R4D",
                "RADIANCE_USE_R4D_GDN",
                "RADIANCE_R4D_REPORT",
                "RADIANCE_USE_R4D_AR",
                "RADIANCE_USE_R4D_AR_QUANT",
                "RADIANCE_SKINNY_GEMM",
                "RADIANCE_GDN_META",
                "RADIANCE_GDN_MERGE_INPROJ",
                "RADIANCE_GDN_FUSED_UPDATE",
                "RADIANCE_GDN_SHARED_BUILD",
                "RADIANCE_TOPK_TRITON_MIN_ROWS",
                "RADIANCE_TOPK_COMPOSITE",
                "RADIANCE_TOPK_COMPOSITE_KCAP",
                "RADIANCE_FAST_DRAFT",
                "RADIANCE_DRAFT_RERANK",
                "RADIANCE_DFLASH_SELECTOR_TOPK",
                "RADIANCE_VERIFY_HEAD",
                "RADIANCE_DYNAMIC_WIDTH",
                "RADIANCE_DYNW_MIN_BATCH",
                "RADIANCE_DYNW_ALPHA",
                "RADIANCE_DYNW_LOW",
                "RADIANCE_DYNW_HIGH",
                "RADIANCE_MXFP4",
                "RADIANCE_MXFP4_W4A8",
                "RADIANCE_MXFP4_W4A8_MIN_M",
                "RADIANCE_MXFP4_DECODE_MAX_M",
                "RADIANCE_MXFP4_TN4_MIN_M",
                "RADIANCE_MXFP4_EPIFAST",
                "RADIANCE_MXFP4_WPERM",
                "RADIANCE_NORMQUANT_FUSION",
                "RADIANCE_MXFP4_HOIST_QUANT",
                "RADIANCE_MXFP4_TRACED_QUANT",
                "RADIANCE_FP8_STREAM",
                "RADIANCE_QUARK_BF16_MTP",
                "RADIANCE_MXFP4_SANITIZE",
                "RADIANCE_MXFP4_DEBUG",
                "RADIANCE_MXFP4_SYNC",
                "RADIANCE_MXFP4_CHECKX",
                "RADIANCE_MXFP4_CHECKALL",
                "RADIANCE_MXFP4_KERNEL_N",
                "RADIANCE_MXFP4_KERNEL_NK",
                "RADIANCE_MXFP4_PERBLOCK_NK",
                "WEIGHT_QUANTIZATION",
                "RADIANCE_SPECULATIVE_CONFIG",
                "RADIANCE_COMPILATION_CONFIG",
                "RADIANCE_DYNAMIC_DRAFT",
                "RADIANCE_DRAFT_SCHEDULE",
                "RADIANCE_DRAFT_TAU",
                "RADIANCE_FAST_REDUCE",
                "RADIANCE_AR_MAX_KB",
                "RADIANCE_AR_QUANT",
                "RADIANCE_AR_QUANT_MIN_KB",
                "RADIANCE_AR_QNT",
                "RADIANCE_AR_QNB",
                "RADIANCE_KV_GROUP_OPT",
                "RADIANCE_KV_GROUP_SIZE",
                "RADIANCE_KV_GROUP_MAX_GROUPS",
                "R4D_ATTN_FP8",
                "RADIANCE_GDN_WMMA",
                "KV_CACHE_DTYPE",
                "ALLOW_DIAGNOSTIC_NON_FP8_KV",
                "PREFIX_CACHING",
                "PREFIX_CACHING_FLAG",
                "MAMBA_CACHE_MODE",
                "VLLM_USE_V2_MODEL_RUNNER",
                "BENCH_CORRECTNESS_MAX_TOKENS",
                "BENCH_CORRECTNESS_REPETITIONS",
                "BENCH_CORRECTNESS_LOGPROBS",
                "BENCH_CORRECTNESS_PROMPTS",
                "BENCH_TOOL_SCHEMA_ATTEMPTS",
                "TOOL_CALL_PARSER",
                "BETTERBENCH_PROFILE",
                "BETTERBENCH_ROOT",
            )
            if key in os.environ
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
