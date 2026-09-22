#!/usr/bin/env python3
"""Validate Radiance's maintenance-control-plane registries against the repository."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required for control-plane validation. Install with: pip install PyYAML"
    ) from exc


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"{path}: expected a mapping at the document root")
    return data


def unique(records: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for item in records:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(f"{label}: every record needs non-empty {field!r}")
        if value in values:
            raise AssertionError(f"{label}: duplicate {field} {value!r}")
        values[value] = item
    return values


def docker_args(source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in re.finditer(r"(?m)^ARG\s+([A-Z0-9_]+)=([^\s#]+)", source):
        result[match.group(1)] = match.group(2)
    return result


def docker_patch_loop(source: str) -> set[str]:
    match = re.search(r"for\s+p\s+in\s+(.*?)\s*;\s*do", source, re.S)
    if not match:
        raise AssertionError("Dockerfile: could not find guarded patch application loop")
    body = match.group(1).replace("\\\n", " ").replace("\\", " ")
    return {token for token in body.split() if token}


def check_stack(repo: Path, stack: dict[str, Any], errors: list[str]) -> None:
    source = (repo / "Dockerfile").read_text()
    args = docker_args(source)
    components = stack.get("components")
    if not isinstance(components, list):
        errors.append("stack.yaml: components must be a list")
        return
    try:
        unique(components, "id", "stack components")
    except AssertionError as exc:
        errors.append(str(exc))

    for component in components:
        declared = component.get("docker_args", {})
        if declared:
            if not isinstance(declared, dict):
                errors.append(f"stack component {component.get('id')}: docker_args must be a mapping")
                continue
            for name, expected in declared.items():
                actual = args.get(name)
                if actual != str(expected):
                    errors.append(
                        f"stack component {component.get('id')}: Dockerfile ARG {name}={actual!r}, "
                        f"manifest expects {expected!r}"
                    )
        one_arg = component.get("docker_arg")
        contains = component.get("docker_value_contains")
        if one_arg and contains:
            actual = args.get(one_arg, "")
            if str(contains) not in actual:
                errors.append(
                    f"stack component {component.get('id')}: Dockerfile ARG {one_arg} does not "
                    f"contain {contains!r}"
                )


def check_tests(repo: Path, tests_doc: dict[str, Any], errors: list[str]) -> set[str]:
    records = tests_doc.get("tests")
    if not isinstance(records, list):
        errors.append("tests.yaml: tests must be a list")
        return set()
    try:
        by_id = unique(records, "id", "tests")
    except AssertionError as exc:
        errors.append(str(exc))
        return set()

    hardware = set(tests_doc.get("hardware_levels", []))
    for test_id, item in by_id.items():
        if item.get("hardware") not in hardware:
            errors.append(f"test {test_id}: unknown hardware level {item.get('hardware')!r}")
        if not isinstance(item.get("command"), str) or not item["command"].strip():
            errors.append(f"test {test_id}: command must be non-empty")
        areas = item.get("areas")
        if not isinstance(areas, list) or not areas:
            errors.append(f"test {test_id}: areas must be a non-empty list")
        if item.get("ci") and not item.get("automatic"):
            errors.append(f"test {test_id}: CI tests must also be automatic")

    for profile, ids in (tests_doc.get("profiles") or {}).items():
        if not isinstance(ids, list):
            errors.append(f"test profile {profile}: expected a list")
            continue
        for test_id in ids:
            if test_id not in by_id:
                errors.append(f"test profile {profile}: unknown test {test_id!r}")

    for index, mapping in enumerate(tests_doc.get("change_map") or []):
        pattern = mapping.get("pattern")
        try:
            re.compile(pattern)
        except (TypeError, re.error) as exc:
            errors.append(f"change_map[{index}]: invalid regex {pattern!r}: {exc}")
        if not mapping.get("areas"):
            errors.append(f"change_map[{index}]: areas must be non-empty")

    return set(by_id)


def check_patches(
    repo: Path, patches_doc: dict[str, Any], test_ids: set[str], errors: list[str]
) -> None:
    raw_records = patches_doc.get("patches")
    if not isinstance(raw_records, dict):
        errors.append("patches.yaml: patches must be a mapping keyed by patch stem")
        return
    defaults = patches_doc.get("defaults") or {}
    if not isinstance(defaults, dict):
        errors.append("patches.yaml: defaults must be a mapping")
        return
    area_tests = patches_doc.get("area_tests") or {}
    if not isinstance(area_tests, dict):
        errors.append("patches.yaml: area_tests must be a mapping")
        return

    by_stem: dict[str, dict[str, Any]] = {}
    for stem, raw in raw_records.items():
        if not isinstance(stem, str) or not stem:
            errors.append("patches.yaml: patch stem keys must be non-empty strings")
            continue
        if not isinstance(raw, dict):
            errors.append(f"patch {stem}: entry must be a mapping")
            continue
        item = {**defaults, **raw}
        item["stem"] = stem
        item["file"] = f"{stem}.py"
        by_stem[stem] = item

    known_activations = set(patches_doc.get("activation_values") or [])
    known_runners = set(patches_doc.get("runner_values") or [])
    for stem, item in by_stem.items():
        path = repo / item["file"]
        removed = item.get("removed_at")
        if removed and (item.get("active") or item.get("v030_disposition") != "UPSTREAM_OWNED"):
            errors.append(f"patch {stem}: removed overlay must be inactive and upstream-owned")
        if not path.is_file() and not removed:
            errors.append(f"patch {stem}: registered file does not exist: {item['file']!r}")
        if item.get("activation") not in known_activations:
            errors.append(f"patch {stem}: unknown activation {item.get('activation')!r}")
        if item.get("runner") not in known_runners:
            errors.append(f"patch {stem}: unknown runner {item.get('runner')!r}")
        if not item.get("reason"):
            errors.append(f"patch {stem}: reason is required")
        tests = list(item.get("baseline_tests") or [])
        for area in item.get("areas") or []:
            tests.extend(area_tests.get(area, []))
        tests.extend(item.get("tests_add") or [])
        for test_id in dict.fromkeys(tests):
            if test_id not in test_ids:
                errors.append(f"patch {stem}: references unknown test {test_id!r}")

    for area, ids in area_tests.items():
        if not isinstance(ids, list):
            errors.append(f"patch area_tests {area}: expected a list")
            continue
        for test_id in ids:
            if test_id not in test_ids:
                errors.append(f"patch area_tests {area}: unknown test {test_id!r}")

    root_patch_stems = {path.stem for path in repo.glob("patch_*.py")}
    registered_patch_stems = {stem for stem, item in by_stem.items()
                             if stem.startswith("patch_") and not item.get("removed_at")}
    missing = sorted(root_patch_stems - registered_patch_stems)
    extra = sorted(registered_patch_stems - root_patch_stems)
    if missing:
        errors.append("patches.yaml: unregistered root patches: " + ", ".join(missing))
    if extra:
        errors.append("patches.yaml: registered patch files missing from root: " + ", ".join(extra))

    direct_manifest = {
        stem for stem, item in by_stem.items()
        if item.get("active") and item.get("activation") == "dockerfile-direct"
    }
    direct_docker = docker_patch_loop((repo / "Dockerfile").read_text())
    if direct_manifest != direct_docker:
        missing_registry = sorted(direct_docker - direct_manifest)
        not_direct = sorted(direct_manifest - direct_docker)
        if missing_registry:
            errors.append(
                "patches.yaml: Dockerfile-applied overlays not marked dockerfile-direct: "
                + ", ".join(missing_registry)
            )
        if not_direct:
            errors.append(
                "patches.yaml: overlays marked dockerfile-direct but not in Dockerfile loop: "
                + ", ".join(not_direct)
            )


def check_upstreams(upstreams_doc: dict[str, Any], errors: list[str]) -> None:
    records = upstreams_doc.get("upstreams")
    if not isinstance(records, list):
        errors.append("upstreams.yaml: upstreams must be a list")
        return
    try:
        unique(records, "id", "upstreams")
    except AssertionError as exc:
        errors.append(str(exc))
        return
    for item in records:
        if not item.get("repository"):
            errors.append(f"upstream {item.get('id')}: repository is required")
        if not item.get("relationship"):
            errors.append(f"upstream {item.get('id')}: relationship must be non-empty")
        if not item.get("last_audited_ref"):
            errors.append(f"upstream {item.get('id')}: last_audited_ref is required")


def check_policy_files(repo: Path, errors: list[str]) -> None:
    required = [
        "AGENTS.md",
        "CONTRIBUTING.md",
        "docs/maintenance/ARCHITECTURE.md",
        "docs/maintenance/UPGRADE_POLICY.md",
        "docs/maintenance/IMPORT_POLICY.md",
        "docs/maintenance/TESTING_POLICY.md",
        "docs/maintenance/RELEASE_POLICY.md",
    ]
    for rel in required:
        if not (repo / rel).is_file():
            errors.append(f"missing policy/front-door file: {rel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)"
    )
    args = parser.parse_args()
    repo = args.root.resolve()

    manifests = repo / ".radiance"
    stack = load_yaml(manifests / "stack.yaml")
    patches = load_yaml(manifests / "patches.yaml")
    upstreams = load_yaml(manifests / "upstreams.yaml")
    tests = load_yaml(manifests / "tests.yaml")

    errors: list[str] = []
    check_policy_files(repo, errors)
    check_stack(repo, stack, errors)
    test_ids = check_tests(repo, tests, errors)
    check_patches(repo, patches, test_ids, errors)
    check_upstreams(upstreams, errors)

    if errors:
        print("Radiance control-plane validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Radiance control-plane validation: PASS "
        f"({len(patches.get('patches', {}))} overlays, "
        f"{len(tests.get('tests', []))} tests, "
        f"{len(upstreams.get('upstreams', []))} tracked upstreams)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
