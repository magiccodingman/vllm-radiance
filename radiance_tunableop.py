"""Persisted PyTorch TunableOp workflow for Radiance.

Collection happens with runtime tuning disabled.  The resulting untuned GEMM
signatures are tuned offline during a maintenance window, producing a hashed
manifest.  Serving consumes only that immutable, validated result set.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _torch_runtime() -> tuple[str | None, list[str]]:
    try:
        import torch
    except ImportError:
        return None, []
    return (
        getattr(torch.version, "hip", None),
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    )


def result_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.glob("results*.csv") if path.is_file() and path.stat().st_size
    )


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def write_manifest(root: Path, *, input_pattern: str, num_gpus: int) -> Path:
    target = manifest_path(root)
    if target.exists():
        raise FileExistsError(f"immutable TunableOp manifest already exists: {target}")
    if type(num_gpus) is not int or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    files = result_files(root)
    if not files:
        raise RuntimeError(f"TunableOp produced no non-empty results*.csv beneath {root}")
    names = {path.name for path in files}
    for ordinal in range(num_gpus):
        if f"results{ordinal}.csv" not in names:
            raise ValueError(f"TunableOp produced no non-empty results{ordinal}.csv")
    rocm, devices = _torch_runtime()
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_pattern": input_pattern,
        "num_gpus": num_gpus,
        "tuning": {
            name: os.environ.get(name)
            for name in (
                "PYTORCH_TUNABLEOP_NUMERICAL_CHECK",
                "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS",
                "PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS",
                "PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE",
            )
        },
        "results": [
            {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
        "software": {
            "radiance": os.environ.get("RADIANCE_VERSION"),
            "torch": _version("torch"),
            "rocm": rocm,
            "python": platform.python_version(),
        },
        "device_names": devices,
    }
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return target


def verify(root: Path) -> dict[str, Any]:
    manifest = manifest_path(root)
    if not manifest.is_file():
        raise ValueError(f"missing TunableOp manifest: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported TunableOp manifest schema: {data.get('schema_version')}")
    listed = data.get("results")
    if not isinstance(listed, list) or not listed:
        raise ValueError("TunableOp manifest has no result files")
    expected_devices = data.get("num_gpus")
    if type(expected_devices) is not int or expected_devices < 1:
        raise ValueError("TunableOp manifest num_gpus must be a positive integer")
    names = {record["filename"] for record in listed}
    for ordinal in range(expected_devices):
        if f"results{ordinal}.csv" not in names:
            raise ValueError(f"TunableOp manifest lacks results{ordinal}.csv")
    for record in listed:
        path = root / record["filename"]
        if not path.is_file():
            raise ValueError(f"missing TunableOp result: {path}")
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"TunableOp result checksum mismatch: {path}")
    built_for = data.get("software", {}).get("radiance")
    running = os.environ.get("RADIANCE_VERSION")
    if built_for and running and built_for != running:
        raise ValueError(
            f"TunableOp results belong to Radiance {built_for}, running {running}"
        )
    built_torch = data.get("software", {}).get("torch")
    running_torch = _version("torch")
    if built_torch and running_torch and built_torch != running_torch:
        raise ValueError(
            f"TunableOp results belong to torch {built_torch}, running {running_torch}"
        )
    return data


def tune(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if type(args.num_gpus) is not int or args.num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if manifest_path(root).exists() or any(root.glob("results*.csv")):
        raise FileExistsError(
            f"TunableOp output already exists beneath {root}; preserve it and select a fresh namespace"
        )
    pattern = str(root / args.input_pattern)
    inputs = sorted(Path(path) for path in glob.glob(pattern))
    if not inputs:
        raise FileNotFoundError(f"no TunableOp collection files match {pattern}")

    # These must be present before importing torch. Tuning is explicit in this
    # one-shot command and never occurs in the live serving process.
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "0"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(root / "results.csv")
    os.environ["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = str(root / "untuned.csv")
    os.environ["PYTORCH_TUNABLEOP_NUMERICAL_CHECK"] = args.numerical_check
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = str(args.max_duration_ms)
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS"] = str(args.max_iterations)
    os.environ["PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE"] = str(
        args.rotating_buffer_mib
    )

    import torch
    import torch.cuda.tunable as tunable

    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(
            f"requested {args.num_gpus} GPUs, only {torch.cuda.device_count()} are visible"
        )
    tunable.mgpu_tune_gemm_in_file(pattern, args.num_gpus)
    manifest = write_manifest(root, input_pattern=args.input_pattern, num_gpus=args.num_gpus)
    print(f"[radiance] TunableOp results and manifest ready beneath {root}: {manifest}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--root", required=True)
    run = sub.add_parser("tune")
    run.add_argument("--root", required=True)
    run.add_argument("--input-pattern", default="untuned?.csv")
    run.add_argument("--num-gpus", type=int, default=2)
    run.add_argument("--numerical-check", default="1e-3_1e-3")
    run.add_argument("--max-duration-ms", type=int, default=30)
    run.add_argument("--max-iterations", type=int, default=100)
    run.add_argument("--rotating-buffer-mib", type=int, default=-1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify":
        data = verify(Path(args.root).resolve())
        print(json.dumps({"status": "ok", **data}, indent=2, sort_keys=True))
    else:
        tune(args)


if __name__ == "__main__":
    main()
