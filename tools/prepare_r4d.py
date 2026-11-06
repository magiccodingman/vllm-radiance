#!/usr/bin/env python3
"""Adapt maintained libr4d kernel families to Radiance's qualified Python ABI.

The caller checks out both immutable commits before invoking this tool. Keep
the existing bindings/collectives and private extensions, while compiling the
current attention and GDN scan from StillDeadcode/radiance. Engine plugin
entrypoints are excluded; their operand headers are compile-time dependencies.
No standalone engine is installed or invoked.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"libr4d compatibility anchor drift: {old[:90]!r}")
    return source.replace(old, new, 1)


def prepare(legacy: Path, upstream: Path, version: str, upstream_commit: str):
    if re.fullmatch(r"[0-9a-f]{40}", upstream_commit) is None:
        raise ValueError("An immutable maintained-source commit is required")
    source = upstream / "libr4d"
    destination = legacy / "maintained"
    if destination.exists():
        raise RuntimeError("Use a fresh libr4d source checkout for preparation")
    destination.mkdir()
    receipt = {}
    for path in source.iterdir():
        if path.suffix in (".h", ".hip"):
            shutil.copyfile(path, destination / path.name)
            receipt[f"libr4d/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ("abi", "plugin"):
        shutil.copytree(upstream / directory, legacy / f"maintained-{directory}")
    for name in ("LICENSE", "NOTICE"):
        if (upstream / name).exists():
            shutil.copyfile(upstream / name, destination / name)

    # These families retain raw-pointer C entrypoints as well as engine/plugin
    # entrypoints. The latter reference unrelated geometries and are not part
    # of the Python ABI. A missing boundary is fatal, never a fuzzy truncation.
    marker = "// ============================================================== the plugin ABI entry points"
    units = ("r4d_attn_paged_h256_gqa6", "r4d_gdn_chunk_scan_k128_v128_c64_bf16")
    for unit in units:
        path = destination / f"{unit}.hip"
        text = path.read_text()
        if text.count(marker) != 1:
            raise RuntimeError(f"Maintained kernel/plugin boundary drift: {unit}")
        path.write_text(text.split(marker)[0])

    # Both sides must agree on the complete struct, including zero-valued
    # fields not used by the Python consumer. Copying only an old prefix would
    # let the new kernels read beyond a smaller stack object.
    current_header = (source / "r4d.h").read_text()
    legacy_header = (legacy / "r4d.h").read_text()
    pattern = r"struct R4DArgs \{.*?\n\};"
    current_struct = re.findall(pattern, current_header, flags=re.S)
    old_struct = re.findall(pattern, legacy_header, flags=re.S)
    if len(current_struct) != 1 or len(old_struct) != 1:
        raise RuntimeError("Unexpected R4DArgs declaration")
    legacy_header = legacy_header.replace(old_struct[0], current_struct[0])
    legacy_header = replace_once(
        legacy_header, '#define R4D_VERSION "0.5.0"', f'#define R4D_VERSION "{version}"'
    )
    # The maintained scan additionally supports fp16 state. Our existing ABI
    # is fp32-state only: pass an explicit zero rather than reinterpret storage.
    start = legacy_header.index("int r4d_gdn_chunk_scan_k128_v128_c64_bf16(")
    end = legacy_header.index(";", start) + 1
    declaration = legacy_header[start:end]
    legacy_header = legacy_header[:start] + replace_once(
        declaration, "float scale, void* stream", "float scale, int state_fp16, void* stream"
    ) + legacy_header[end:]
    (legacy / "r4d.h").write_text(legacy_header)

    module = (legacy / "r4d_module.hip").read_text()
    module = replace_once(module, "  R4DArgs a{};", "  R4DArgs a{};\n  a.causal = 1;")
    module = replace_once(
        module, "head_k, head_v, chunk, (float)scale, (void*)stream);",
        "head_k, head_v, chunk, (float)scale, 0, (void*)stream);",
    )
    module = replace_once(
        module, 'm.attr("__version__") = R4D_VERSION;',
        'm.attr("__version__") = R4D_VERSION;\n'
        f'  m.attr("__maintained_commit__") = "{upstream_commit}";',
    )
    (legacy / "r4d_module.hip").write_text(module)
    build = (legacy / "build.sh").read_text()
    for unit in units:
        build = replace_once(build, f'"{unit}:', f'"maintained/{unit}:')
    build = replace_once(
        build, 'BASE="-O3 ', 'BASE="-Imaintained-abi -Imaintained-plugin -O3 '
    )
    (legacy / "build.sh").write_text(build)
    (legacy / "maintained-source-hashes.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Prepared maintained libr4d attention/GDN scan with Python ABI {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()
    prepare(args.legacy, args.upstream, args.version, args.upstream_commit)
