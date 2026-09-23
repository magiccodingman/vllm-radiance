#!/usr/bin/env python3
"""Apply the release overlay sequence to an isolated source/dependency fixture.

This is source compatibility evidence, never an installed/runtime qualification.
Failures are retained and the audit continues so an upgrade sees every drift.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=False)
    site = args.out / "site"
    site.mkdir()
    shutil.copytree(args.source / "vllm", site / "vllm")
    installed = Path(sysconfig.get_paths()["purelib"])
    for name in ("aiter", "triton", "transformers"):
        shutil.copytree(installed / name, site / name)
    (site / "torch" / "_dynamo").mkdir(parents=True)
    shutil.copy2(installed / "torch/_dynamo/utils.py", site / "torch/_dynamo/utils.py")
    for src in repo.glob("radiance_*.py"):
        shutil.copy2(src, site / src.name)
    docker = (repo / "Dockerfile").read_text()
    loop = re.search(r"for p in (.*?); do", docker, re.S).group(1)
    patches = loop.replace("\\", " ").split()
    results = []
    for patch in patches:
        code = (
            "import sysconfig,runpy,sys; "
            "original=sysconfig.get_paths; "
            f"sysconfig.get_paths=lambda *a,**k:dict(original(*a,**k),purelib={str(site)!r}); "
            f"sys.path.insert(0,{str(repo)!r}); "
            f"runpy.run_path({str(repo / (patch + '.py'))!r},run_name='__main__')"
        )
        proc = subprocess.run([sys.executable, "-c", code], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.out / (patch + ".log")).write_text(proc.stdout)
        results.append({"patch": patch, "exit_code": proc.returncode,
                        "output": proc.stdout})
        if proc.returncode == 0:
            repeat = subprocess.run([sys.executable, "-c", code], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (args.out / (patch + ".repeat.log")).write_text(repeat.stdout)
            results.append({"patch": patch + ":repeat", "exit_code": repeat.returncode,
                            "output": repeat.stdout})
        print(patch, proc.returncode, proc.stdout.strip()[-400:], flush=True)
    syntax_errors = []
    for path in (site / "vllm").rglob("*.py"):
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            syntax_errors.append({"path": str(path.relative_to(site)), "error": str(exc)})
    (args.out / "result.json").write_text(json.dumps({
        "scope": "SOURCE_APPLICATION_ONLY", "patches": results,
        "syntax_errors": syntax_errors}, indent=2) + "\n")
    return int(bool(syntax_errors or any(row["exit_code"] for row in results)))


if __name__ == "__main__":
    raise SystemExit(main())
