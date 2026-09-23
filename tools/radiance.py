#!/usr/bin/env python3
"""Small front door for Radiance maintenance test/qualification selection."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required by the Radiance maintenance CLI. Install with: pip install PyYAML"
    ) from exc


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / ".radiance/tests.yaml"


def load() -> dict[str, Any]:
    data = yaml.safe_load(REGISTRY.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{REGISTRY}: invalid registry root")
    return data


def by_id(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in doc["tests"]}


def changed_files(base: str | None) -> list[str]:
    if base:
        command = ["git", "diff", "--name-only", f"{base}...HEAD"]
    else:
        env_base = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "origin/main"],
            cwd=REPO, text=True, capture_output=True
        ).stdout.strip()
        command = ["git", "diff", "--name-only", f"{env_base}...HEAD"] if env_base else [
            "git", "diff", "--name-only", "HEAD~1", "HEAD"
        ]
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    if proc.returncode:
        raise SystemExit(proc.stderr.strip() or f"git diff failed: {' '.join(command)}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def areas_for_files(doc: dict[str, Any], files: list[str]) -> set[str]:
    areas: set[str] = set()
    for filename in files:
        for mapping in doc.get("change_map", []):
            if re.search(mapping["pattern"], filename):
                areas.update(mapping["areas"])
    return areas


def select_area(doc: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [test for test in doc["tests"] if area in test.get("areas", [])]


def select_profile(doc: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    index = by_id(doc)
    return [index[test_id] for test_id in doc["profiles"][profile]]


def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item["id"] not in seen:
            seen.add(item["id"])
            result.append(item)
    return result


def show_plan(items: list[dict[str, Any]]) -> None:
    for item in items:
        auto = "auto" if item.get("automatic") else "manual"
        print(f"- {item['id']} [{item['class']} / {item['hardware']} / {auto}]")
        print(f"    {item['command']}")


def run_auto(items: list[dict[str, Any]], dry_run: bool) -> int:
    for item in items:
        if not item.get("automatic"):
            print(f"SKIP {item['id']}: requires {item['hardware']}")
            continue
        print(f"{'PLAN' if dry_run else 'RUN '} {item['id']}: {item['command']}")
        if dry_run:
            continue
        result = subprocess.run(item["command"], cwd=REPO, shell=True)
        if result.returncode:
            print(f"FAIL {item['id']} ({result.returncode})")
            return result.returncode or 1
        print(f"PASS {item['id']}")
    return 0


def test_command(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    baseline = select_profile(doc, "pr")
    if args.selector == "pr":
        items = baseline
    elif args.selector == "changed":
        files = changed_files(args.base)
        areas = areas_for_files(doc, files)
        print("Changed files:")
        for path in files:
            print(f"  {path}")
        print("Selected areas:", ", ".join(sorted(areas)) if areas else "(baseline only)")
        items = baseline[:]
        for area in sorted(areas):
            items.extend(select_area(doc, area))
    else:
        items = baseline + select_area(doc, args.selector)

    items = dedupe(items)
    print("Radiance test plan:")
    show_plan(items)
    return run_auto(items, args.dry_run)


def qualify_command(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    items = select_profile(doc, args.profile)
    print(f"Radiance {args.profile} qualification plan:")
    show_plan(items)
    if not args.run_auto:
        print("\nPlan only. Pass --run-auto to execute the automatic CPU subset.")
        return 0
    return run_auto(items, args.dry_run)


def list_command(doc: dict[str, Any]) -> int:
    show_plan(doc["tests"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="radiance")
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("test", help="run/select automatic tests")
    p_test.add_argument("selector", help="pr, changed, or an area tag")
    p_test.add_argument("--base", help="base ref for changed-file selection")
    p_test.add_argument("--dry-run", action="store_true")

    p_qual = sub.add_parser("qualify", help="print an upgrade/release qualification plan")
    p_qual.add_argument("profile", choices=["upgrade", "release"])
    p_qual.add_argument("--run-auto", action="store_true", help="execute only registry entries marked automatic")
    p_qual.add_argument("--dry-run", action="store_true")

    sub.add_parser("list", help="list all registered tests")

    args = parser.parse_args()
    doc = load()
    if args.command == "test":
        return test_command(args, doc)
    if args.command == "qualify":
        return qualify_command(args, doc)
    return list_command(doc)


if __name__ == "__main__":
    raise SystemExit(main())
