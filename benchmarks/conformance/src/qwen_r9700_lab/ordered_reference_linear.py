"""Optional CPU implementation of the canonical ordered linear operation.

This module is not enabled by importing it. A caller must build and bind the
artifact, then explicitly supply its callable to the reference. Compiler and
hardware semantics remain trusted assumptions, not a formal certificate.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from qwen_r9700_lab.conformance_artifacts import file_identity
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, seal, write_private

SOURCE = Path(__file__).with_suffix(".c")
FLAGS = (
    "-std=c11",
    "-O3",
    "-shared",
    "-fPIC",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-frounding-math",
    "-fno-finite-math-only",
)
SCHEMA = "urn:qwen:ordered-reference-linear:v1"
ARITHMETIC = "increasing-k binary32 multiply then binary32 add; no FMA or reassociation"


def build(root: Path) -> dict:
    """Compile in a new private artifact directory; never import a GPU library."""
    compiler_name = shutil.which("cc")
    if not compiler_name:
        raise DiagnosticError("ordered reference linear requires a C compiler")
    compiler = Path(compiler_name).resolve()
    root = root.resolve()
    root.mkdir(mode=0o700)
    source = root / "ordered-linear.c"
    source.write_bytes(SOURCE.read_bytes())
    source.chmod(0o600)
    library = root / "ordered-linear.so"
    identities = {"source": file_identity(source), "compiler": file_identity(compiler)}
    command = [str(compiler), *FLAGS, str(source), "-lm", "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    log = root / "compiler.log"
    log.write_text(result.stdout + result.stderr)
    log.chmod(0o600)
    if result.returncode:
        raise DiagnosticError("ordered reference linear compilation failed; compiler log retained")
    if (
        file_identity(source) != identities["source"]
        or file_identity(compiler) != identities["compiler"]
    ):
        raise DiagnosticError("ordered reference compiler or source changed during compilation")
    library.chmod(0o600)
    binding = seal(
        {
            "schema": SCHEMA,
            "arithmetic": ARITHMETIC,
            "flags": list(FLAGS),
            "source": {"path": str(source), **identities["source"]},
            "compiler": {"path": str(compiler), **identities["compiler"]},
            "library": {"path": str(library), **file_identity(library)},
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "compiler_and_hardware": "ASSUMED",
            "qualification": "BUILT_UNTESTED",
        }
    )
    write_private(root / "binding.json", binding)
    return binding


def validate_binding(binding: dict) -> None:
    """Validate exact bytes before loading executable code."""
    authenticate(binding)
    if (
        binding.get("schema") != SCHEMA
        or binding.get("arithmetic") != ARITHMETIC
        or binding.get("flags") != list(FLAGS)
        or binding.get("adapter_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or binding.get("source", {}).get("sha256") != file_identity(SOURCE)["sha256"]
    ):
        raise DiagnosticError("ordered reference implementation binding differs")
    for key in ("source", "compiler", "library"):
        row = binding[key]
        if file_identity(Path(row["path"])) != {"sha256": row["sha256"], "bytes": row["bytes"]}:
            raise DiagnosticError(f"ordered reference {key} artifact changed")


class OrderedLinear:
    """A callable replacement; it never modifies the reference module globally."""

    def __init__(self, reference, binding: dict):
        validate_binding(binding)
        self.reference = reference
        self.original = reference.linear
        self.binding = binding
        self.library = ctypes.CDLL(binding["library"]["path"])
        # Check the artifact again after loading; the generated library has no
        # user initializers. Artifact immutability during execution is assumed.
        validate_binding(binding)
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.function = self.library.qwen_ordered_linear
        self.function.argtypes = [pointer, pointer, pointer, *([ctypes.c_size_t] * 3)]
        self.function.restype = ctypes.c_int
        self.calls = 0
        self.fallbacks = 0

    def __call__(self, value, weight, *, quantize_activation=False, output_bf16=True):
        def fallback():
            self.fallbacks += 1
            return self.original(
                value, weight, quantize_activation=quantize_activation, output_bf16=output_bf16
            )

        x, w = np.asarray(value, np.float32), np.asarray(weight, np.float32)
        if x.ndim < 1 or w.ndim != 2 or x.shape[-1] != w.shape[-1]:
            return fallback()
        if quantize_activation:
            code, scale = self.reference.activation_quantize(x)
            x = np.multiply(self.reference.fp8_decode(code), scale, dtype=np.float32)
        x, w = np.ascontiguousarray(x), np.ascontiguousarray(w)
        if not np.isfinite(x).all() or not np.isfinite(w).all():
            return fallback()
        output = np.empty((*x.shape[:-1], w.shape[0]), np.float32)
        status = self.function(x, w, output, math.prod(x.shape[:-1]), w.shape[0], w.shape[-1])
        self.calls += 1
        if status:
            raise DiagnosticError("ordered reference linear requires round-to-nearest arithmetic")
        if not np.isfinite(output).all():
            return fallback()
        return self.reference.bf16(output) if output_bf16 else output
