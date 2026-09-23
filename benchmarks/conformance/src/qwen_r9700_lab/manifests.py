"""Capture content-addressed, create-once evidence manifests."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_r9700_lab import __version__
from qwen_r9700_lab.config import (
    ConfigurationError,
    find_project_root,
    load_json_object,
    validate_config,
    validate_instance,
    validate_profile_semantics,
)

SCHEMA_IDS = {
    "environment": "urn:qwen-r9700-lab:schema:environment-manifest:v1",
    "model": "urn:qwen-r9700-lab:schema:model-manifest:v1",
    "run": "urn:qwen-r9700-lab:schema:run-manifest:v1",
}

RELEVANT_ENVIRONMENT_VARIABLES = (
    "AMD_LOG_LEVEL",
    "GGML_VK_ALLOW_GRAPHICS_QUEUE",
    "GPU_DEVICE_ORDINAL",
    "HIP_LAUNCH_BLOCKING",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VULKAN_SDK",
)


def utc_now() -> str:
    """Return a compact UTC timestamp suitable for evidence records."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON canonically for stable content hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_with_id(document: dict[str, Any]) -> dict[str, Any]:
    if "manifest_id" in document:
        raise ConfigurationError("manifest payload already contains manifest_id")
    result = dict(document)
    result["manifest_id"] = f"sha256:{hashlib.sha256(canonical_bytes(document)).hexdigest()}"
    return result


def write_immutable_manifest(
    output: Path,
    document: dict[str, Any],
    schema_name: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically link a read-only manifest without replacing a path."""

    root = project_root or find_project_root(output.parent)
    completed = _manifest_with_id(document)
    validate_instance(completed, root / "schemas" / schema_name, str(output))

    # Keep the final path itself unresolved so a pre-existing symlink is rejected by the
    # no-replace hard link instead of being followed to its target.
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(completed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".qwen-r9700-manifest-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o444)
        try:
            os.link(temporary_path, output)
        except FileExistsError as error:
            raise ConfigurationError(f"refusing to replace existing manifest: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return completed


def _read_os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor() or None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_total_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _probe(argv: list[str], timeout_seconds: float = 10.0, limit: int = 65_536) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"argv": argv, "status": "unavailable"}
    try:
        process = subprocess.run(
            [executable, *argv[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr
        )
        return {
            "argv": argv,
            "status": "timeout",
            "stdout": (stdout or "")[:limit],
            "stderr": (stderr or "")[:limit],
        }

    stdout = process.stdout[:limit]
    stderr = process.stderr[:limit]
    return {
        "argv": argv,
        "status": "completed",
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": len(process.stdout) > limit,
        "stderr_truncated": len(process.stderr) > limit,
    }


def _read_sysfs_text(path: Path) -> str | None:
    """Read one small sysfs attribute, preserving absence as null evidence."""

    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return value or None


def _read_sysfs_int(path: Path) -> int | None:
    value = _read_sysfs_text(path)
    if value is None:
        return None
    try:
        return int(value, 10)
    except ValueError:
        return None


def _resource0_evidence(device_path: Path) -> dict[str, Any]:
    byte_size: int | None = None
    with suppress(OSError):
        byte_size = (device_path / "resource0").stat().st_size

    table_entry = None
    resource_table = _read_sysfs_text(device_path / "resource")
    if resource_table is not None:
        table_entry = resource_table.splitlines()[0]

    range_start = None
    range_end = None
    flags = None
    range_size_bytes: int | None = None
    if table_entry is not None:
        fields = table_entry.split()
        if len(fields) >= 3:
            try:
                start = int(fields[0], 16)
                end = int(fields[1], 16)
            except ValueError:
                pass
            else:
                range_start = f"0x{start:016x}"
                range_end = f"0x{end:016x}"
                flags = fields[2]
                if start == 0 and end == 0:
                    range_size_bytes = 0
                elif end >= start:
                    range_size_bytes = end - start + 1

    sizes_match = None
    if byte_size is not None and range_size_bytes is not None:
        sizes_match = byte_size == range_size_bytes
    return {
        "byte_size": byte_size,
        "resource_table_entry": table_entry,
        "range_start": range_start,
        "range_end": range_end,
        "range_flags": flags,
        "range_size_bytes": range_size_bytes,
        "sizes_match": sizes_match,
    }


def _drm_node_type(name: str) -> str | None:
    for prefix, node_type in (
        ("card", "primary"),
        ("renderD", "render"),
        ("controlD", "control"),
    ):
        suffix = name.removeprefix(prefix)
        if suffix != name and suffix.isdigit():
            return node_type
    return None


def _drm_nodes(device_path: Path, dev_dri_root: Path) -> list[dict[str, Any]]:
    drm_path = device_path / "drm"
    try:
        candidates = sorted(drm_path.iterdir(), key=lambda path: path.name)
    except OSError:
        return []

    nodes: list[dict[str, Any]] = []
    for candidate in candidates:
        node_type = _drm_node_type(candidate.name)
        if node_type is None:
            continue
        device_node = dev_dri_root / candidate.name
        nodes.append(
            {
                "name": candidate.name,
                "node_type": node_type,
                "major_minor": _read_sysfs_text(candidate / "dev"),
                "device_path": str(device_node),
                "device_path_exists": device_node.exists(),
            }
        )
    return nodes


def _driver_name(device_path: Path) -> str | None:
    try:
        return (device_path / "driver").resolve(strict=True).name
    except (OSError, RuntimeError):
        return None


def _lspci_region_0(pci_bdf: str) -> tuple[str | None, dict[str, Any]]:
    probe = _probe(["lspci", "-s", pci_bdf, "-vv"])
    region_0 = None
    if probe.get("status") == "completed":
        for line in probe.get("stdout", "").splitlines():
            stripped = line.strip()
            if stripped.startswith("Region 0:"):
                region_0 = stripped
                break
    return region_0, probe


def _amd_gpu_evidence(
    device_path: Path,
    pci_bdf: str,
    class_code: str,
    vendor_id: str,
    device_id: str | None,
    dev_dri_root: Path,
) -> dict[str, Any]:
    region_0, lspci_probe = _lspci_region_0(pci_bdf)
    return {
        "pci_bdf": pci_bdf,
        "class_code": class_code,
        "vendor_id": vendor_id,
        "device_id": device_id,
        "driver": _driver_name(device_path),
        "lspci_region_0": region_0,
        "lspci_probe": lspci_probe,
        "resource0": _resource0_evidence(device_path),
        "vram": {
            "total_bytes": _read_sysfs_int(device_path / "mem_info_vram_total"),
            "used_bytes": _read_sysfs_int(device_path / "mem_info_vram_used"),
        },
        "pcie_link": {
            "current_speed": _read_sysfs_text(device_path / "current_link_speed"),
            "current_width": _read_sysfs_text(device_path / "current_link_width"),
            "maximum_speed": _read_sysfs_text(device_path / "max_link_speed"),
            "maximum_width": _read_sysfs_text(device_path / "max_link_width"),
        },
        "power": {
            "control": _read_sysfs_text(device_path / "power" / "control"),
            "runtime_status": _read_sysfs_text(device_path / "power" / "runtime_status"),
        },
        "drm_nodes": _drm_nodes(device_path, dev_dri_root),
    }


def collect_pci_display_inventory(
    pci_devices_root: Path = Path("/sys/bus/pci/devices"),
    dev_dri_root: Path = Path("/dev/dri"),
) -> dict[str, Any]:
    """Enumerate display-class PCI devices and collect per-BDF AMD GPU evidence."""

    try:
        candidates = sorted(pci_devices_root.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return {
            "status": "unavailable",
            "sysfs_pci_devices_path": str(pci_devices_root),
            "display_device_count": 0,
            "amd_gpu_count": 0,
            "display_devices": [],
            "amd_gpus": [],
        }
    except OSError as error:
        return {
            "status": "error",
            "error": str(error),
            "sysfs_pci_devices_path": str(pci_devices_root),
            "display_device_count": 0,
            "amd_gpu_count": 0,
            "display_devices": [],
            "amd_gpus": [],
        }

    display_devices: list[dict[str, Any]] = []
    amd_gpus: list[dict[str, Any]] = []
    for device_path in candidates:
        class_code = _read_sysfs_text(device_path / "class")
        if class_code is None:
            continue
        try:
            is_display = int(class_code, 16) >> 16 == 0x03
        except ValueError:
            continue
        if not is_display:
            continue

        class_code = class_code.lower()
        vendor_id_value = _read_sysfs_text(device_path / "vendor")
        device_id_value = _read_sysfs_text(device_path / "device")
        vendor_id = vendor_id_value.lower() if vendor_id_value is not None else None
        device_id = device_id_value.lower() if device_id_value is not None else None
        display_devices.append(
            {
                "pci_bdf": device_path.name,
                "class_code": class_code,
                "vendor_id": vendor_id,
                "device_id": device_id,
                "driver": _driver_name(device_path),
            }
        )
        if vendor_id is not None and vendor_id.lower() == "0x1002":
            amd_gpus.append(
                _amd_gpu_evidence(
                    device_path,
                    device_path.name,
                    class_code,
                    vendor_id,
                    device_id,
                    dev_dri_root,
                )
            )

    return {
        "status": "captured",
        "sysfs_pci_devices_path": str(pci_devices_root),
        "display_device_count": len(display_devices),
        "amd_gpu_count": len(amd_gpus),
        "display_devices": display_devices,
        "amd_gpus": amd_gpus,
    }


def capture_environment(
    hardware_config: Path,
    output: Path,
    project_root: Path | None = None,
    *,
    pci_devices_root: Path = Path("/sys/bus/pci/devices"),
    dev_dri_root: Path = Path("/dev/dri"),
) -> dict[str, Any]:
    """Capture host identity, runtime versions, and bounded GPU-stack probes."""

    root = project_root or find_project_root(hardware_config.parent)
    hardware = validate_config(hardware_config, root)
    if hardware["kind"] != "hardware":
        raise ConfigurationError(f"{hardware_config}: expected kind 'hardware'")

    document: dict[str, Any] = {
        "$schema": SCHEMA_IDS["environment"],
        "schema_version": 1,
        "manifest_type": "environment",
        "captured_at": utc_now(),
        "collector": {"name": "qwen-r9700-lab", "version": __version__},
        "expected_hardware": {
            "config_path": str(hardware_config.expanduser().resolve()),
            "config_sha256": sha256_file(hardware_config),
            "config": hardware,
        },
        "host": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "kernel_release": platform.release(),
            "kernel_version": platform.version(),
            "operating_system": platform.system(),
            "os_release": _read_os_release(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_bytes": _memory_total_bytes(),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "relevant_environment": {
                name: os.environ.get(name) for name in RELEVANT_ENVIRONMENT_VARIABLES
            },
        },
        "pci_display_inventory": collect_pci_display_inventory(pci_devices_root, dev_dri_root),
        "probes": {
            "uv": _probe(["uv", "--version"]),
            "pci": _probe(["lspci", "-Dnnk"]),
            "rocm_info": _probe(["rocminfo"]),
            "rocm_smi": _probe(
                ["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram"]
            ),
            "amd_smi": _probe(["amd-smi", "version"]),
            "vulkan": _probe(["vulkaninfo", "--summary"]),
            "glslc": _probe(["glslc", "--version"]),
            "hipcc": _probe(["hipcc", "--version"]),
        },
    }
    return write_immutable_manifest(
        output,
        document,
        "environment-manifest.schema.json",
        root,
    )


def _is_excluded(relative_path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns)


def _model_files(model_dir: Path, exclude: list[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*")):
        relative = path.relative_to(model_dir).as_posix()
        if _is_excluded(relative, exclude):
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ConfigurationError(f"model tree contains a non-file entry: {path}")
        record: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.is_symlink():
            record["symlink_target"] = str(path.readlink())
        files.append(record)
    return files


def capture_model(
    model_config: Path,
    model_dir: Path,
    output: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Hash a local model tree and bind it to a checked-in model declaration."""

    root = project_root or find_project_root(model_config.parent)
    configuration = validate_config(model_config, root)
    if configuration["kind"] != "model":
        raise ConfigurationError(f"{model_config}: expected kind 'model'")

    try:
        resolved_model_dir = model_dir.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ConfigurationError(f"model directory does not exist: {model_dir}") from error
    if not resolved_model_dir.is_dir():
        raise ConfigurationError(f"model path is not a directory: {resolved_model_dir}")

    files = _model_files(resolved_model_dir, configuration["manifest_exclude"])
    if not files:
        raise ConfigurationError(
            f"model directory contains no included files: {resolved_model_dir}"
        )
    tree_sha256 = hashlib.sha256(canonical_bytes(files)).hexdigest()

    document: dict[str, Any] = {
        "$schema": SCHEMA_IDS["model"],
        "schema_version": 1,
        "manifest_type": "model",
        "captured_at": utc_now(),
        "collector": {"name": "qwen-r9700-lab", "version": __version__},
        "configuration": {
            "config_path": str(model_config.expanduser().resolve()),
            "config_sha256": sha256_file(model_config),
            "config": configuration,
        },
        "model_tree": {
            "directory": str(resolved_model_dir),
            "file_count": len(files),
            "total_size_bytes": sum(record["size_bytes"] for record in files),
            "tree_sha256": tree_sha256,
            "files": files,
        },
    }
    return write_immutable_manifest(output, document, "model-manifest.schema.json", root)


def load_manifest(path: Path, schema_name: str, project_root: Path) -> dict[str, Any]:
    """Load a manifest and verify both schema and embedded content address."""

    manifest = load_json_object(path)
    validate_instance(manifest, project_root / "schemas" / schema_name, str(path))
    expected_id = manifest["manifest_id"]
    payload = dict(manifest)
    del payload["manifest_id"]
    actual_id = f"sha256:{hashlib.sha256(canonical_bytes(payload)).hexdigest()}"
    if expected_id != actual_id:
        raise ConfigurationError(
            f"{path}: manifest_id mismatch (expected {actual_id}, found {expected_id})"
        )
    return manifest


def prepare_run(
    environment_manifest: Path,
    model_manifest: Path,
    engine_config: Path,
    profile_config: Path,
    reasoning_effort: str,
    preserve_thinking: bool,
    output: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Bind immutable evidence and orthogonal policy choices into a run declaration."""

    root = project_root or find_project_root(engine_config.parent)
    environment = load_manifest(
        environment_manifest,
        "environment-manifest.schema.json",
        root,
    )
    model = load_manifest(model_manifest, "model-manifest.schema.json", root)
    engine = validate_config(engine_config, root)
    profile = validate_config(profile_config, root)
    if engine["kind"] != "engine":
        raise ConfigurationError(f"{engine_config}: expected kind 'engine'")
    if profile["kind"] != "profile":
        raise ConfigurationError(f"{profile_config}: expected kind 'profile'")
    validate_profile_semantics(profile, profile_config)

    model_format = model["configuration"]["config"]["format"]
    if model_format not in engine["supported_model_formats"]:
        raise ConfigurationError(
            f"engine {engine['id']!r} does not declare support for model format {model_format!r}"
        )

    thinking_enabled = reasoning_effort != "off"
    document: dict[str, Any] = {
        "$schema": SCHEMA_IDS["run"],
        "schema_version": 1,
        "manifest_type": "run",
        "created_at": utc_now(),
        "collector": {"name": "qwen-r9700-lab", "version": __version__},
        "inputs": {
            "environment": {
                "path": str(environment_manifest.expanduser().resolve()),
                "file_sha256": sha256_file(environment_manifest),
                "manifest_id": environment["manifest_id"],
            },
            "model": {
                "path": str(model_manifest.expanduser().resolve()),
                "file_sha256": sha256_file(model_manifest),
                "manifest_id": model["manifest_id"],
            },
        },
        "configuration": {
            "engine": {
                "path": str(engine_config.expanduser().resolve()),
                "file_sha256": sha256_file(engine_config),
                "config": engine,
            },
            "profile": {
                "path": str(profile_config.expanduser().resolve()),
                "file_sha256": sha256_file(profile_config),
                "config": profile,
            },
            "reasoning": {
                "effort": reasoning_effort,
                "enable_thinking": thinking_enabled,
                "preserve_thinking": preserve_thinking if thinking_enabled else False,
            },
        },
        "execution": {"state": "prepared", "backend_invoked": False},
    }
    return write_immutable_manifest(output, document, "run-manifest.schema.json", root)
