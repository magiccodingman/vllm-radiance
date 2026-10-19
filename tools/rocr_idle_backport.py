#!/usr/bin/env python3
"""Build an explicit ROCr 7.14 idle-spin candidate; never install it on the host."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

REPO = Path(__file__).resolve().parents[1]
SERIES = REPO / "backports/rocr-idle"


def manifest() -> dict:
    return json.loads((SERIES / "manifest.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_digest(path: Path) -> str:
    # The upstream tree contains both CRLF and LF files; compiler input is identical.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, encoding="utf-8")


def verify_series() -> dict:
    spec = manifest()
    for patch in spec["patches"]:
        if digest(SERIES / patch["file"]) != patch["sha256"]:
            raise ValueError(f"Backport checksum mismatch: {patch['file']}")
    return spec


def verify_source(source: Path) -> None:
    spec = verify_series()
    if run("git", "rev-parse", "HEAD", cwd=source).strip() != spec["source_commit"]:
        raise ValueError("ROCr source is not the pinned 7.14.0 revision")
    for name, expected in spec["patched_files"].items():
        if source_digest(source / name) != expected:
            raise ValueError(f"Patched source mismatch: {name}")
    changed = set(run("git", "diff", "--name-only", "HEAD", cwd=source).splitlines())
    unknown = set(run("git", "ls-files", "--others", "--exclude-standard", cwd=source).splitlines())
    if (changed | unknown) != set(spec["patched_files"]):
        raise ValueError("Unexpected or missing changes in ROCr source checkout")


def prepare(source: Path) -> None:
    spec = verify_series()
    # Refuse to reuse a checkout: interrupted/previous attempts stay as evidence.
    source.mkdir(parents=True, exist_ok=False)
    run("git", "init", str(source))
    run("git", "remote", "add", "origin", spec["repository"], cwd=source)
    run("git", "config", "core.autocrlf", "false", cwd=source)
    run("git", "config", "core.longpaths", "true", cwd=source)
    run("git", "fetch", "--depth=1", "--filter=blob:none", "origin", spec["source_commit"], cwd=source)
    run("git", "sparse-checkout", "set", "projects/rocr-runtime", cwd=source)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
    for patch in spec["patches"]:
        path = str(SERIES / patch["file"])
        # Match upstream's mixed line endings; exact normalized output hashes below
        # still reject all unexpected context/code changes.
        run("git", "apply", "--ignore-space-change", "--check", path, cwd=source)
        run("git", "apply", "--ignore-space-change", path, cwd=source)
    verify_source(source)
    print(f"Prepared exact ROCr source and {len(spec['patches'])} backports: {source}")


def fetch_sdk(destination: Path) -> None:
    """Extract checksum-pinned AMD build packages into a fresh staging root."""
    spec = manifest()
    destination.mkdir(parents=True, exist_ok=False)
    cache = destination / "packages"
    cache.mkdir()
    for package in spec["packages"]:
        path = cache / Path(package["path"]).name
        with urllib.request.urlopen(spec["packages_base"] + package["path"], timeout=120) as response:
            with path.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if digest(path) != package["sha256"]:
            raise ValueError(f"AMD package checksum mismatch: {package['name']}")
        subprocess.run(["dpkg-deb", "--extract", str(path), str(destination)], check=True)
        print(f"Verified and extracted {package['name']}={package['version']}", flush=True)


def check_runtime(path: Path) -> None:
    expected = manifest()["original_runtime"]
    if digest(path) != expected["sha256"]:
        raise ValueError("Original libhsa differs from the audited 7.14.0 binary; re-audit before replacement")


def exported_symbols(path: Path) -> set[str]:
    exports = set()
    for line in run("readelf", "--wide", "--dyn-syms", str(path)).splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[4] in ("GLOBAL", "WEAK") and fields[6] != "UND":
            exports.add(fields[7])
    return exports


def needed_libraries(path: Path) -> set[str]:
    return set(re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", run("readelf", "--wide", "--dynamic", str(path))))


def build(source: Path, directory: Path, output: Path, sdk: Path, jobs: int) -> None:
    verify_source(source)
    directory.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    llvm = sdk / "lib/llvm"
    if not llvm.exists():
        llvm = sdk / "llvm"
    compiler = llvm / "bin/clang++"
    if not compiler.is_file():
        raise ValueError(f"Pinned ROCm compiler not found: {compiler}")
    compiler_version = run(str(compiler), "--version")
    compiler_identity = manifest()["identity_evidence"]["compiler"].split("; ", 1)[1]
    if compiler_identity not in compiler_version:
        raise ValueError("Compiler differs from the audited ROCm 7.14.0 SDK")
    sysdeps = sdk / "lib/rocm_sysdeps"
    environment = dict(os.environ)
    environment["PKG_CONFIG_PATH"] = str(sysdeps / "lib/pkgconfig") + os.pathsep + environment.get("PKG_CONFIG_PATH", "")
    subprocess.run([
        "cmake", "-S", str(source / "projects/rocr-runtime"), "-B", str(directory), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        f"-DCMAKE_C_COMPILER={llvm / 'bin/clang'}", f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DCMAKE_PREFIX_PATH={sysdeps};{sdk};{llvm}", "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/rocm_sysdeps/lib",
        "-DCMAKE_VERBOSE_MAKEFILE=OFF", "-DENABLE_LDCONFIG=OFF",
    ], check=True, env=environment)
    cache = (directory / "CMakeCache.txt").read_text()
    if "HSA_DEP_ROCPROFILER_REGISTER:INTERNAL=ON" not in cache:
        raise ValueError("Build lost rocprofiler-register support; refusing a profiling-disabled candidate")
    subprocess.run(["cmake", "--build", str(directory), "--target", "hsa-runtime64", "--parallel", str(jobs)], check=True)
    library = directory / "rocr/lib/libhsa-runtime64.so.1.21.0"
    target = output / library.name
    shutil.copyfile(library, target)
    original = sdk / "lib" / library.name
    check_runtime(original)
    missing_exports = exported_symbols(original) - exported_symbols(target)
    if missing_exports:
        raise ValueError(f"Candidate lost versioned exports: {sorted(missing_exports)}")
    original_deps, candidate_deps = needed_libraries(original), needed_libraries(target)
    preserved = {name for name in original_deps if name.startswith(("librocm_sysdeps_", "librocprofiler-register"))}
    if preserved - candidate_deps or {"libelf.so.1", "libdrm.so.2", "libdrm_amdgpu.so.1", "libnuma.so.1"} & candidate_deps:
        raise ValueError(f"Candidate changed bundled ROCm dependencies: original={sorted(original_deps)}, candidate={sorted(candidate_deps)}")
    if "Library soname: [libhsa-runtime64.so.1]" not in run("readelf", "--wide", "--dynamic", str(target)):
        raise ValueError("Candidate changed the runtime SONAME")
    # Record exported ABI/dependencies for comparison with the original package.
    for flag, name in [("--dyn-syms", "symbols.txt"), ("--dynamic", "dynamic.txt")]:
        (output / name).write_text(run("readelf", "--wide", flag, str(target)), encoding="utf-8")
    report = dict(manifest(), candidate_sha256=digest(target), compiler=compiler_version,
                  original_needed=sorted(original_deps), candidate_needed=sorted(candidate_deps),
                  hardware_qualification="NOT RUN; R9700 TP2 required")
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(SERIES / "LICENSE.md", output / "LICENSE.md")
    shutil.copyfile(SERIES / "LICENSE-MIT", output / "LICENSE-MIT")
    print(f"Built candidate (not installed): {target}")


def load_candidate(path: Path) -> None:
    ctypes.CDLL(str(path.resolve()), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    print("Candidate dynamic load passed; hsa_init/GPU execution NOT RUN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-series")
    for name in ("prepare", "verify-source", "fetch-sdk", "check-runtime", "load-candidate"):
        sub.add_parser(name).add_argument("path", type=Path)
    command = sub.add_parser("build")
    for name in ("source", "directory", "output"):
        command.add_argument(name, type=Path)
    command.add_argument("--sdk", type=Path, default=Path("/opt/rocm/core-7.14"))
    command.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.command == "verify-series":
        verify_series()
    elif args.command == "build":
        build(args.source.resolve(), args.directory.resolve(), args.output.resolve(), args.sdk.resolve(), args.jobs)
    else:
        {"prepare": prepare, "verify-source": verify_source, "fetch-sdk": fetch_sdk,
         "check-runtime": check_runtime, "load-candidate": load_candidate}[args.command](args.path.resolve())


if __name__ == "__main__":
    main()
