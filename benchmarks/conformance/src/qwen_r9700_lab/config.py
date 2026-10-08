"""Configuration discovery and validation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_FILENAMES = {
    "engine": "engine.config.schema.json",
    "hardware": "hardware.config.schema.json",
    "model": "model.config.schema.json",
    "profile": "profile.config.schema.json",
}


class LabError(RuntimeError):
    """Base class for expected, user-facing errors."""


class ConfigurationError(LabError):
    """A configuration or manifest did not satisfy its contract."""


def find_project_root(start: Path | None = None) -> Path:
    """Find a source checkout containing the configuration and schemas."""

    configured = os.environ.get("QWEN_R9700_LAB_ROOT")
    roots: list[Path] = []
    if configured:
        roots.append(Path(configured).expanduser())

    for origin in (start or Path.cwd(), Path(__file__).resolve()):
        candidate = origin if origin.is_dir() else origin.parent
        roots.extend((candidate, *candidate.parents))

    for root in roots:
        if (root / "schemas").is_dir() and (root / "configs").is_dir():
            return root.resolve()
    raise ConfigurationError(
        "could not locate the project root; run from the checkout or set QWEN_R9700_LAB_ROOT"
    )


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object with concise diagnostics."""

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise ConfigurationError(f"file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"invalid JSON in {path} at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigurationError(f"could not read {path}: {error}") from error

    if not isinstance(value, dict):
        raise ConfigurationError(f"expected a JSON object in {path}")
    return value


def validate_instance(instance: dict[str, Any], schema_path: Path, label: str) -> None:
    """Validate an instance and report all schema violations together."""

    schema = load_json_object(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:  # pragma: no cover - indicates a repository defect
        raise ConfigurationError(f"invalid repository schema {schema_path}: {error}") from error

    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return

    lines = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        lines.append(f"{location}: {error.message}")
    raise ConfigurationError(f"{label} failed validation:\n  " + "\n  ".join(lines))


def validate_config(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Validate one typed repository configuration file."""

    project_root = root or find_project_root(path.parent)
    instance = load_json_object(path)
    kind = instance.get("kind")
    if kind not in SCHEMA_FILENAMES:
        expected = ", ".join(sorted(SCHEMA_FILENAMES))
        raise ConfigurationError(
            f"{path}: unknown config kind {kind!r}; expected one of {expected}"
        )

    validate_instance(instance, project_root / "schemas" / SCHEMA_FILENAMES[kind], str(path))
    if kind == "profile":
        validate_profile_semantics(instance, path)
    return instance


def validate_profile_semantics(profile: dict[str, Any], path: Path) -> None:
    """Enforce context/RoPE invariants that are clearer in code than JSON Schema."""

    maximum = profile["max_context_tokens"]
    native = profile["native_context_tokens"]
    qualification = profile["backend_qualification"]
    status = qualification["status"]
    rope = profile["rope"]
    kv_cache = profile["kv_cache"]
    speculation = profile["speculation"]

    if maximum <= native:
        if rope["type"] != "native":
            raise ConfigurationError(f"{path}: native context profiles must not enable YaRN")
        if "factor" in rope:
            raise ConfigurationError(f"{path}: native RoPE must not declare a scaling factor")
    else:
        if rope["type"] != "yarn":
            raise ConfigurationError(f"{path}: extended context profiles must use YaRN")
        minimum_factor = maximum / native
        if rope["factor"] + 1e-9 < minimum_factor:
            raise ConfigurationError(
                f"{path}: YaRN factor {rope['factor']} is too small for {maximum} tokens "
                f"(minimum {minimum_factor:.3f})"
            )
        if rope["original_max_position_embeddings"] != native:
            raise ConfigurationError(
                f"{path}: YaRN original_max_position_embeddings must equal "
                f"native_context_tokens ({native})"
            )

    if qualification["target_only"] and (
        speculation["preferred_drafter"] != "none"
        or speculation["maximum_draft_tokens"] != 0
        or speculation["adaptive"]
    ):
        raise ConfigurationError(
            f"{path}: target-only backend qualifications must disable speculation"
        )

    quality_gates = qualification["quality_gates"]
    if status == "unavailable":
        if kv_cache["representation"] != "unavailable":
            raise ConfigurationError(
                f"{path}: unavailable profiles must not claim an executable KV representation"
            )
        if not quality_gates:
            raise ConfigurationError(
                f"{path}: unavailable profiles must state the research gates that block execution"
            )
        return

    if kv_cache["representation"] == "unavailable":
        raise ConfigurationError(
            f"{path}: executable profiles must declare an implemented KV representation"
        )
    if kv_cache["prefix_caching"] != (kv_cache["mamba_cache_mode"] == "align"):
        raise ConfigurationError(
            f"{path}: prefix caching requires mamba-cache-mode align; disabled prefix caching "
            "requires mode none"
        )

    if status == "production_ready":
        if maximum > 131072:
            raise ConfigurationError(
                f"{path}: stock-vLLM production qualification stops at 131072 tokens"
            )
        if kv_cache["representation"] != "bfloat16" or kv_cache["cli_dtype"] != "auto":
            raise ConfigurationError(
                f"{path}: production-ready profiles must use BF16 storage via KV dtype auto"
            )
        if quality_gates:
            raise ConfigurationError(
                f"{path}: production-ready profiles cannot retain unresolved quality gates"
            )
        if maximum >= 65536 and not kv_cache["prefix_caching"]:
            raise ConfigurationError(
                f"{path}: 64K and 128K production profiles require aligned prefix caching"
            )
    elif status == "experimental_quality_gate":
        if maximum != native:
            raise ConfigurationError(
                f"{path}: the current experimental qualification is only for native 262144 context"
            )
        if kv_cache["representation"] != "fp8_e4m3" or kv_cache["cli_dtype"] != "fp8_e4m3":
            raise ConfigurationError(
                f"{path}: the 262144-token experiment must declare FP8 E4M3 explicitly"
            )
        if not quality_gates:
            raise ConfigurationError(
                f"{path}: experimental profiles must state their unresolved quality gates"
            )


def validate_profiles(paths: Iterable[Path], root: Path | None = None) -> list[dict[str, Any]]:
    """Validate profiles individually and enforce uniqueness across a set."""

    path_list = list(paths)
    if not path_list:
        raise ConfigurationError("no profile configuration files were selected")

    profiles: list[dict[str, Any]] = []
    names: dict[str, Path] = {}
    for path in path_list:
        profile = validate_config(path, root)
        if profile["kind"] != "profile":
            raise ConfigurationError(f"{path}: expected kind 'profile'")
        name = profile["name"]
        if name in names:
            raise ConfigurationError(f"duplicate profile name {name!r}: {names[name]} and {path}")
        names[name] = path
        profiles.append(profile)
    return profiles


def all_config_paths(root: Path) -> list[Path]:
    """Return every checked-in configuration in stable order."""

    return sorted((root / "configs").glob("*/*.json"))
