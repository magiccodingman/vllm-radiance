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
    weights = [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in sorted(model.glob("*.safetensors"))
    ]
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
        "model": {
            "path": str(model),
            "config_sha256": sha256(model / "config.json"),
            "index_sha256": sha256(model / "model.safetensors.index.json"),
            "weights": weights,
        },
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
            )
            if key in os.environ
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
