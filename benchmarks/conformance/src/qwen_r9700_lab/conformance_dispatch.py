"""Opt-in, library-bound entrypoint tracing, without device reads or GPU imports.

Python aliases of reviewed native exports are replaced too. This observes calls
through those entrypoints, not launches hidden in C++, graphs, or a captured
closure. A library hash is deliberately never called an ISA attestation.
"""

from __future__ import annotations

import functools
import threading
import time
from pathlib import Path

from qwen_r9700_lab.conformance_artifacts import file_identity
from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    require_name,
    require_sha,
    seal,
    write_private,
)


def replace_aliases(value, replacements, ancestors=frozenset()):
    """Copy alias containers; do not mutate a list retained by another owner."""
    replacement = replacements.get(id(value))
    if replacement is not None:
        return replacement, True
    if isinstance(value, (list, tuple, dict)):
        if id(value) in ancestors:
            raise DiagnosticError("cyclic native alias container is unsupported")
        ancestors = ancestors | {id(value)}
    if isinstance(value, (list, tuple)):
        children = [replace_aliases(child, replacements, ancestors) for child in value]
        if any(changed for _, changed in children):
            return type(value)(child for child, _ in children), True
    elif isinstance(value, dict):
        children = {
            key: replace_aliases(child, replacements, ancestors) for key, child in value.items()
        }
        if any(changed for _, changed in children.values()):
            return {key: child for key, (child, _) in children.items()}, True
    return value, False


class DispatchRecorder:
    def __init__(self, root: Path, *, execution: str):
        self.execution = require_sha(execution)
        root.mkdir(mode=0o700)
        self.root, self.hooks = root, HookSet()
        self.bindings, self.rows, self.required, self.aliases = {}, [], set(), []
        self.lock, self.closed = threading.Lock(), False

    def bind(self, module, binding, *, aliases=()):
        """Bind a reviewed, complete public callable export set before replay.

        binding: sealed {schema, module, library_sha256, exports, required}.
        exports maps names to kernel/metadata; required names must execute.
        No metadata functions (including registry helpers) are invoked here.
        """
        authenticate(binding)
        if set(binding) != {"schema", "module", "library_sha256", "exports", "required", "sha256"}:
            raise DiagnosticError("incomplete native entrypoint binding")
        if binding["schema"] != "urn:qwen:native-entrypoint-binding:v1":
            raise DiagnosticError("unsupported native entrypoint binding")
        name = require_name(binding["module"])
        if module.__name__ != name or name in self.bindings:
            raise DiagnosticError("native entrypoint module identity mismatch")
        exports = binding["exports"]
        observed = {
            key
            for key, value in vars(module).items()
            if not key.startswith("_") and callable(value)
        }
        if (
            not exports
            or set(exports) != observed
            or any(kind not in {"kernel", "metadata"} for kind in exports.values())
        ):
            raise DiagnosticError("native export inventory changed or is incomplete")
        required = binding["required"]
        if (
            not isinstance(required, list)
            or not required
            or len(set(required)) != len(required)
            or any(exports.get(key) != "kernel" for key in required)
        ):
            raise DiagnosticError("native binding needs an explicit nonempty exercised domain")
        if file_identity(Path(module.__file__))["sha256"] != require_sha(binding["library_sha256"]):
            raise DiagnosticError("native library changed before entrypoint binding")
        replacements = {}
        for symbol, kind in exports.items():
            require_name(symbol)
            if kind != "kernel":
                continue
            original = getattr(module, symbol)
            if id(original) in replacements:
                raise DiagnosticError("native exports alias each other; mapping is ambiguous")
            site = name + "." + symbol

            @functools.wraps(original)
            def call(*args, _site=site, _fn=original, **kwargs):
                with self.lock:
                    if self.closed:
                        raise DiagnosticError("native dispatch recorder already finalized")
                    row = {
                        "index": len(self.rows),
                        "site": _site,
                        "started_ns": time.monotonic_ns(),
                        "completed": False,
                        "argument_count": len(args),
                        "keyword_count": len(kwargs),
                    }
                    self.rows.append(row)
                path = self.root / f"entry-{row['index']:09d}"
                write_private(path.with_suffix(".started.json"), seal(row))
                try:
                    result = _fn(*args, **kwargs)
                    row["completed"] = True
                    return result
                except BaseException as exc:
                    row["exception_type"] = type(exc).__qualname__
                    raise
                finally:
                    row["ended_ns"] = time.monotonic_ns()
                    write_private(path.with_suffix(".finished.json"), seal(row))

            replacements[id(original)] = call
        # Validate every export before replacing any attribute.
        changes = []
        for owner in (module, *aliases):
            for attribute, value in list(vars(owner).items()):
                # Skip arbitrary objects/closures and cyclic runtime internals.
                if attribute.startswith("__"):
                    continue
                replacement, changed = replace_aliases(value, replacements)
                if changed:
                    changes.append((owner, attribute, replacement))
        for owner, attribute, replacement in changes:
            self.hooks.replace(owner, attribute, replacement)
            self.aliases.append({"owner": owner.__name__, "attribute": attribute})
        self.bindings[name] = {"binding": binding, "library": str(module.__file__)}
        self.required.update(name + "." + symbol for symbol in required)

    def finish(self):
        try:
            with self.lock:
                self.closed = True
                if not self.rows or any(not row["completed"] for row in self.rows):
                    raise DiagnosticError("native entrypoint observation is incomplete")
                if not self.required <= {row["site"] for row in self.rows}:
                    raise DiagnosticError("required native entrypoint was not exercised")
            for entry in self.bindings.values():
                if (
                    file_identity(Path(entry["library"]))["sha256"]
                    != entry["binding"]["library_sha256"]
                ):
                    raise DiagnosticError("native library changed during capture")
            result = seal(
                {
                    "schema": "urn:qwen:native-entrypoint-capture:v1",
                    "execution": self.execution,
                    "bindings": self.bindings,
                    "aliases": self.aliases,
                    "calls": self.rows,
                    "scope": "observed library entrypoints and explicit Python aliases only",
                    "device_completion": "UNPROVED; return does not mean device completion",
                    "argument_and_state_equivalence": (
                        "UNPROVED; pointer values are not tensor evidence"
                    ),
                    "hidden_dispatch_and_graphs": "UNPROVED",
                    "exact_device_binary_attested": False,
                }
            )
            write_private(self.root / "dispatch.json", result)
            return result
        finally:
            self.hooks.close()

    def close(self):
        self.hooks.close()
