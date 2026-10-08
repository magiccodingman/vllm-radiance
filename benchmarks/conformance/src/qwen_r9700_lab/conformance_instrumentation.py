"""Portable, source-bound call instrumentation and exact tensor capture.

No backend is imported here. Native adapters supply tensor export and logical
position mapping explicitly. Tensor mode synchronizes through that exporter;
metadata mode never exports a tensor and cannot claim numerical equivalence.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import marshal
import threading
import time
from pathlib import Path

import numpy as np

from qwen_r9700_lab.conformance_state import FrameWriter, compare_frames, read_frame
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    private_json,
    require_name,
    require_sha,
    seal,
    write_private,
)

CALL_SCHEMA = "urn:qwen:semantic-calls:v1"


def _is_custom_op(function):
    cls = type(function)
    return cls.__module__ == "torch._library.custom_ops" and cls.__qualname__ == "CustomOpDef"


def _custom_op_registration(function):
    items = function._backend_fns.items()
    if any(key is not None and not isinstance(key, str) for key in function._backend_fns):
        raise DiagnosticError("unsupported custom operator backend registry")
    return (
        function._init_fn,
        tuple(sorted(items, key=lambda item: (item[0] is not None, item[0] or ""))),
        frozenset(function._disabled_kernel),
        function._schema,
    )


def callable_identity(function):
    if _is_custom_op(function):
        initial, backends, disabled, schema = _custom_op_registration(function)
        registered = []
        for device, wrapper in backends:
            # CustomOpDef wraps each registered implementation in a dispatcher
            # closure. Binding just __call__ or the default body would miss a
            # different CUDA implementation under the same operator name.
            implementation = inspect.getclosurevars(inspect.unwrap(wrapper)).nonlocals.get("fn")
            if implementation is None:
                raise DiagnosticError("custom operator has an uninspectable registered kernel")
            registered.append(
                {
                    "device": device,
                    "wrapper": callable_identity(wrapper),
                    "implementation": callable_identity(implementation),
                }
            )
        return digest(
            {
                "kind": "torch.library.custom_op",
                "name": function._qualname,
                "schema": schema,
                "dispatcher": callable_identity(type(function).__call__),
                "initial": callable_identity(initial),
                "registered": registered,
                "disabled": sorted(disabled),
            }
        )
    target = inspect.unwrap(function)
    try:
        source = inspect.getsource(target).encode()
    except (TypeError, OSError) as exc:
        raise DiagnosticError("call site requires an explicit inspectable source binding") from exc
    dispatched = getattr(function, "__func__", function)
    code = getattr(dispatched, "__code__", None)
    if code is None:
        raise DiagnosticError("call site requires an explicit executable code binding")
    # getsource/unwrap alone would name the original function even if a wrapper
    # with different executable bytecode actually receives the invocation.
    return digest(
        {
            "source": hashlib.sha256(source).hexdigest(),
            "python_code": hashlib.sha256(marshal.dumps(code)).hexdigest(),
        }
    )


class HookSet:
    """Restore the exact previous attribute, including class-bound methods.

    This is diagnostic instrumentation, not a process-wide singleton. An alias
    cached elsewhere is a separate binding and cannot be claimed as observed.
    """

    def __init__(self):
        self.entries = []

    def replace(self, owner, name, replacement):
        if any(obj is owner and key == name for obj, key, *_ in self.entries):
            raise DiagnosticError("call site already instrumented")
        original = getattr(owner, name)
        own = name in vars(owner)
        setattr(owner, name, replacement)
        self.entries.append((owner, name, original, own, replacement))

    def close(self):
        changed = []
        for owner, name, original, own, replacement in reversed(self.entries):
            if getattr(owner, name, None) is not replacement:
                changed.append(name)
                continue  # do not overwrite somebody else's later instrumentation
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.entries.clear()
        if changed:
            raise DiagnosticError("instrumented call site changed before cleanup")


class CallRecorder:
    """Capture immutable inputs *before* a call, mutated arguments and outputs after.

    Bindings identify actual Python functions called, not the selected HSACO.
    The latter needs a device-dispatch receipt. Unknown argument objects are
    explicitly inventoried, not incorrectly labelled complete tensor state.
    """

    def __init__(
        self,
        root: Path,
        *,
        contract,
        execution,
        adapter,
        tensor_export,
        is_tensor,
        mode="tensor",
        required_sites=(),
    ):
        for value in (contract, execution, adapter):
            require_sha(value)
        if mode not in {"tensor", "metadata"}:
            raise DiagnosticError("unsupported observation mode")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.root, self.mode = root, mode
        self.identities = {"contract": contract, "execution": execution, "adapter": adapter}
        self.tensor_export, self.is_tensor = tensor_export, is_tensor
        self.lock, self.next_id, self.closed = threading.RLock(), 0, False
        self.next_observation = 0
        self.parent = contextvars.ContextVar(f"qwen-conformance-parent-{id(self)}", default=None)
        self.rows, self.bindings, self.required = {}, {}, set(required_sites)
        for site in self.required:
            require_name(site)

    def bind(self, owner, name, *, site, hooks: HookSet, source_sha256=None, context=None):
        require_name(site)
        if site in self.bindings:
            raise DiagnosticError("duplicate semantic call identity")
        original = getattr(owner, name)
        identity = callable_identity(original)
        if source_sha256 is not None and identity != require_sha(source_sha256):
            raise DiagnosticError("semantic call source changed")
        self.bindings[site] = identity
        registration = _custom_op_registration(original) if _is_custom_op(original) else None

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if registration is not None and _custom_op_registration(original) != registration:
                raise DiagnosticError("custom operator registration changed after source binding")
            logical = context() if context is not None else {}
            if logical is None:  # explicitly unselected positions, no tensors read
                return original(*args, **kwargs)
            return self.invoke(site, original, args, kwargs, logical)

        hooks.replace(owner, name, observed)

    def _capture(self, root, values, logical):
        with self.lock:
            ordinal, self.next_observation = self.next_observation, self.next_observation + 1
        arrays, descriptors, unsupported = {}, {}, []

        def visit(path, value):
            if self.is_tensor(value):
                # Metadata mode must not call tensor_export, .cpu(), .item(),
                # synchronize(), or dereference a device pointer.
                descriptors[path] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if self.mode == "tensor":
                    arrays[path] = np.array(self.tensor_export(value), copy=True, order="C")
            elif value is None or type(value) in (bool, int, float):
                descriptors[path] = {"scalar": value}
            elif isinstance(value, (tuple, list)):
                descriptors[path] = {"container": type(value).__name__, "length": len(value)}
                for i, child in enumerate(value):
                    visit(path + f".{i}", child)
            elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
                descriptors[path] = {"keys": sorted(value)}
                for key, child in sorted(value.items()):
                    require_name(key)
                    visit(path + "." + key, child)
            else:
                # Never repr() objects or strings: they can contain request text.
                descriptors[path] = {"unexported_type": type(value).__qualname__}
                unsupported.append(path)

        for path, value in values.items():
            visit(path, value)
        frame = None
        if arrays:
            writer = FrameWriter(
                root,
                **self.identities,
                input_digest=digest(logical),
                phase="operator",
                consumed=logical.get("consumed", 0),
                pending=None,
                expected=list(arrays),
                logical={"capture": "call-tensors"},
            )
            for name, value in arrays.items():
                native_dtype = descriptors[name]["dtype"]
                encoding = {
                    "torch.bfloat16": "bf16",
                    "torch.float8_e4m3fn": "fp8_e4m3fn",
                    "torch.float8_e4m3fnuz": "fp8_e4m3fnuz",
                }.get(native_dtype)
                if encoding is not None:
                    writer.add(name, value.tobytes(), dtype=encoding, shape=value.shape)
                else:
                    writer.array(name, value)
            frame = writer.finish()["sha256"]
        return {
            "frame": frame,
            "descriptors": descriptors,
            "unexported": unsupported,
            "ordinal": ordinal,
        }

    def invoke(self, site, function, args, kwargs, logical):
        with self.lock:
            if self.closed:
                raise DiagnosticError("semantic recorder already finalized")
            index, self.next_id = self.next_id, self.next_id + 1
            row = {
                "index": index,
                "site": site,
                "parent": self.parent.get(),
                "logical": logical,
                "thread": threading.get_ident(),
                "started_ns": time.monotonic_ns(),
            }
            self.rows[index] = row
        path = self.root / f"call-{index:09d}"
        path.mkdir(mode=0o700)
        row["before"] = self._capture(path / "before", {"args": args, "kwargs": kwargs}, logical)
        write_private(path / "started.json", seal(row))
        token = self.parent.set(index)
        try:
            result = function(*args, **kwargs)
            row["after"] = self._capture(
                path / "after", {"args": args, "kwargs": kwargs, "result": result}, logical
            )
            row["completed"] = True
            return result
        except BaseException as exc:
            row["completed"], row["exception_type"] = False, type(exc).__qualname__
            raise
        finally:
            self.parent.reset(token)
            row["ended_ns"] = time.monotonic_ns()
            write_private(path / "finished.json", seal(row))

    def finish(self):
        with self.lock:
            if self.closed:
                raise DiagnosticError("semantic recorder already finalized")
            self.closed = True
            if not self.rows or any(not r.get("completed") for r in self.rows.values()):
                raise DiagnosticError("incomplete semantic call capture")
            seen = {r["site"] for r in self.rows.values()}
            if not self.required <= seen:
                raise DiagnosticError("required semantic call was never observed")
            rows = [seal(self.rows[i]) for i in range(self.next_id)]
            doc = seal(
                {
                    "schema": CALL_SCHEMA,
                    **self.identities,
                    "mode": self.mode,
                    "required_sites": sorted(self.required),
                    "bindings": self.bindings,
                    "calls": rows,
                    "device_binary_identity": "UNPROVED",
                    "non_tensor_state": "UNPROVED: explicit native state adapter required",
                }
            )
            write_private(self.root / "calls.json", doc)
            return doc


def compare_calls(reference: Path, candidate: Path, output: Path):
    docs = [private_json(p / "calls.json") for p in (reference, candidate)]
    for doc in docs:
        authenticate(doc)
        for name in ("contract", "execution", "adapter"):
            require_sha(doc[name])
        if doc.get("schema") != CALL_SCHEMA or doc.get("mode") != "tensor" or not doc.get("calls"):
            raise DiagnosticError("exact call comparison requires complete tensor evidence")
        if not set(doc["required_sites"]) <= {r["site"] for r in doc["calls"]}:
            raise DiagnosticError("call trace omitted required sites")
        if not isinstance(doc.get("bindings"), dict) or not doc["bindings"]:
            raise DiagnosticError("semantic call source bindings are missing")
        for name, identity in doc["bindings"].items():
            require_name(name)
            require_sha(identity)
    if docs[0]["contract"] != docs[1]["contract"] or len(docs[0]["calls"]) != len(docs[1]["calls"]):
        raise DiagnosticError("different semantic contracts or call schedules")
    first, tensors, observations = None, 0, []
    for i, (a, b) in enumerate(zip(docs[0]["calls"], docs[1]["calls"], strict=True)):
        for root, row, doc in ((reference, a, docs[0]), (candidate, b, docs[1])):
            authenticate(row)
            if row["index"] != i or not row["completed"]:
                raise DiagnosticError("reordered or incomplete semantic call")
            recorded = private_json(root / f"call-{i:09d}" / "finished.json")
            if row != recorded:
                raise DiagnosticError("semantic call receipt changed")
            if row["site"] not in doc["bindings"]:
                raise DiagnosticError("observed call is not bound to an implementation")
            started = private_json(root / f"call-{i:09d}" / "started.json")
            authenticate(started)
            if any(row.get(k) != v for k, v in started.items() if k != "sha256"):
                raise DiagnosticError("call changed its captured input receipt")
        if any(a[k] != b[k] for k in ("site", "parent", "logical")):
            raise DiagnosticError("call boundaries need an explicit logical adapter")
        for phase in ("before", "after"):
            ca, cb = a[phase], b[phase]
            if ca["ordinal"] != cb["ordinal"]:
                raise DiagnosticError("different causal observation order")
            observations.append((ca["ordinal"], i, a["site"], phase, ca, cb))
    if sorted(r[0] for r in observations) != list(range(len(observations))):
        raise DiagnosticError("missing or duplicate call observations")
    for _, i, site, phase, ca, cb in sorted(observations):
        if ca["descriptors"] != cb["descriptors"] and first is None:
            first = {"call": i, "site": site, "phase": phase, "kind": "metadata"}
        if ca["unexported"] or cb["unexported"]:
            raise DiagnosticError("call contains unexported state; add its explicit adapter")
        if bool(ca["frame"]) != bool(cb["frame"]):
            raise DiagnosticError("call tensor coverage differs")
        if ca["frame"]:
            roots = [p / f"call-{i:09d}" / phase for p in (reference, candidate)]
            for root, capture, doc in zip(roots, (ca, cb), docs, strict=True):
                frame = read_frame(root)
                if frame["sha256"] != capture["frame"]:
                    raise DiagnosticError("call tensors changed after capture")
                if any(frame[k] != doc[k] for k in ("contract", "execution", "adapter")):
                    raise DiagnosticError("call tensors belong to a different execution")
            result = compare_frames(*roots)
            tensors += 1
            if not result["equal"] and first is None:
                first = {
                    "call": i,
                    "site": site,
                    "phase": phase,
                    "difference": result["first_difference"],
                }
    if tensors == 0:
        raise DiagnosticError("no numerical observations in call comparison")
    report = seal(
        {
            "schema": "urn:qwen:call-comparison:v1",
            "equal": first is None,
            "first_difference": first,
            "calls": len(docs[0]["calls"]),
            "status": "TESTED",
            "scope": "captured call tensors and metadata",
            "native_equivalence": "UNPROVED",
        }
    )
    write_private(output, report)
    return report
