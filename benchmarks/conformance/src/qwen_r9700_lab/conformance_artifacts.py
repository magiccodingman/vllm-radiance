"""Content-free runtime identity, with unobserved device artifacts explicit.

Reading process mappings and library files does not execute GPU code. A mapped
file's hash is not proof of the actual device instructions dispatched from it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal

FLAGS = (
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_ROCM_USE_AITER",
    "RADIANCE_MXFP4",
    "RADIANCE_MXFP4_W4A8",
    "RADIANCE_MXFP4_WPERM",
    "RADIANCE_MXFP4_DECODE_NT",
    "RADIANCE_MXFP4_DECODE_MAX_M",
    "RADIANCE_MXFP4_W4A8_MIN_M",
    "RADIANCE_MXFP4_TN4_MIN_M",
    "RADIANCE_VERIFY_HEAD",
    "RADIANCE_VERIFY_HEAD_TOPK",
    "RADIANCE_DYNAMIC_SPEC_WIDTH",
    "RADIANCE_NORMQUANT_FUSION",
    "RADIANCE_FP8_STREAM",
    "RADIANCE_TILED_PREFILL",
    "RADIANCE_GDN_NORMQUANT_FUSION",
    "RADIANCE_PREFILL_FP8",
    "GPU_MAX_HW_QUEUES",
    "HSA_ENABLE_MWAITX",
    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
)


def file_identity(path):
    """Hash bytes from one stable file descriptor; never substitute a guessed hash."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, f) != getattr(after, f) for f in fields):
        raise DiagnosticError("runtime artifact changed during hashing")
    return {"sha256": h.hexdigest(), "bytes": after.st_size}


def mapped_libraries(text):
    paths = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5]
            if ".so" in Path(path).name or path.endswith((".hsaco", ".co")):
                paths.add(path)
    return sorted(paths)


def compiler_settings(modules=None):
    """Observe an already-loaded compiler; never import/initialize Torch here."""
    modules = sys.modules if modules is None else modules
    config = modules.get("torch._inductor.config")
    value = getattr(config, "emulate_precision_casts", None) if config is not None else None
    return {"emulate_precision_casts": value if type(value) is bool else None}


def capture_runtime(*, maps_path=Path("/proc/self/maps"), environ=None):
    """Observe library files and an explicit nonsecret environment allowlist."""
    env = os.environ if environ is None else environ
    libraries, unavailable = {}, {}
    try:
        mapped = mapped_libraries(maps_path.read_text())
    except OSError as exc:
        mapped = []
        unavailable["process_mappings"] = type(exc).__name__
    for name in mapped:
        try:
            libraries[name] = file_identity(Path(name))
        except (OSError, DiagnosticError) as exc:
            unavailable[name] = type(exc).__name__
    versions = {}
    for name in ("numpy", "vllm", "torch", "triton", "aiter", "libr4d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return seal(
        {
            "schema": "urn:qwen:conformance-runtime-artifacts:v1",
            "python": sys.version,
            "kernel": platform.release(),
            "machine": platform.machine(),
            "packages": versions,
            "flags": {name: env.get(name) for name in FLAGS},
            "compiler_settings": compiler_settings(),
            "mapped_file_bytes": libraries,
            "unavailable_files": unavailable,
            "unproved": {
                "device_code_objects": "Mapped library hashes do not attest dispatched GPU ISA.",
                "compiler_flags_and_generated_ir": "Requires compiler/dispatch instrumentation.",
                "firmware_and_driver_execution": "Not independently verified by this collector.",
            },
            "artifact_inventory_complete": False,
            "exact_device_binary_attested": False,
        }
    )


@lru_cache(maxsize=128)
def _cached_file_identity(path, stat_key):
    # The key contains inode, timestamps and size. Replacements invalidate it.
    return file_identity(Path(path))


def reference_runtime_identity():
    """Bind the CPU oracle's executable dependencies, not just package versions.

    No GPU imports/queries. Loaded-file identity and CPU dispatch declarations
    are evidence under a stable-process assumption, not a compiler proof.
    """
    import math

    import numpy as np

    core = sys.modules["numpy._core._multiarray_umath"]
    paths = {Path(sys.executable).resolve(), Path(core.__file__).resolve()}
    if getattr(math, "__file__", None):
        paths.add(Path(math.__file__).resolve())
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise DiagnosticError("CPU reference library mapping is unavailable")
    for path in mapped_libraries(maps.read_text()):
        name = Path(path).name
        if name.startswith(("libm.so", "libc.so", "libpython", "ld-linux")):
            paths.add(Path(path).resolve())
    libraries = {}
    for path in sorted(paths):
        info = path.stat()
        key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        libraries[str(path)] = dict(_cached_file_identity(str(path), key))
    return seal(
        {
            "schema": "urn:qwen:reference-cpu-runtime:v1",
            "python": sys.version,
            "numpy": np.__version__,
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "files": libraries,
            "numpy_cpu_features": dict(core.__cpu_features__),
            "assumptions": [
                ("loaded memory matches recorded files"),
                ("NumPy/libm implement the declared arithmetic"),
                ("no concurrent runtime patching"),
            ],
            "compiler_correctness": "UNPROVED",
            "gpu_used": False,
        }
    )
