"""Canonical, private tensor frames and streaming exact comparison (CPU only).

Frames describe logical values; physical block numbers never enter equality.
Raw bytes are compared even when their digests match. Numerical distances are
diagnostics, never an excuse for passing an unequal exact comparison.
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    integer,
    private_json,
    require_name,
    require_sha,
    seal,
    write_private,
)
from qwen_r9700_lab.exact_fp8_metrics import DTYPES as FP8_METRIC_DTYPES
from qwen_r9700_lab.exact_fp8_metrics import MAX_ELEMENTS as FP8_METRIC_MAX
from qwen_r9700_lab.exact_fp8_metrics import chunk_metrics as fp8_chunk_metrics

SCHEMA = "urn:qwen:canonical-state-frame:v1"
DTYPES = {
    name: np.dtype(name)
    for name in ("<f4", "<f8", "<f2", "<i4", "<i8", "<u4", "<u8", "|u1", "|i1", "|b1")
}
DTYPES["bf16"] = np.dtype("<u2")
DTYPES["fp8_e4m3fn"] = np.dtype("u1")
DTYPES["fp8_e4m3fnuz"] = np.dtype("u1")
DTYPES["bytes"] = np.dtype("u1")
PHASES = {"prefill", "step", "commit", "restore", "operator"}


def private_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise DiagnosticError("evidence directory must be private and owned")


def open_blob(root: Path, name: str):
    if len(name) != 68 or not name.endswith(".bin"):
        raise DiagnosticError("invalid canonical blob name")
    require_sha(name[:-4])
    fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise DiagnosticError("tensor evidence must be private and owned")
    return os.fdopen(fd, "rb")


def values(raw: bytes, dtype: str) -> np.ndarray:
    result = np.frombuffer(raw, dtype=DTYPES[dtype])
    if dtype == "bf16":
        return (result.astype("<u4") << 16).view("<f4").astype(np.float64)
    if dtype.startswith("fp8_"):
        code = result.astype(np.uint16)
        exponent, mantissa = (code >> 3) & 15, code & 7
        bias = 8 if dtype.endswith("fnuz") else 7
        decoded = np.ldexp(1.0 + mantissa / 8, exponent.astype(int) - bias)
        decoded = np.where(exponent == 0, np.ldexp(mantissa / 8, 1 - bias), decoded)
        decoded = np.where(code & 128, -decoded, decoded)
        nan = code == 128 if bias == 8 else (exponent == 15) & (mantissa == 7)
        return np.where(nan, np.nan, decoded)
    return result.astype(np.float64)


class FrameWriter:
    """A frame becomes readable only after its final manifest is published."""

    def __init__(
        self,
        root: Path,
        *,
        contract: str,
        execution: str,
        adapter: str,
        input_digest: str,
        phase: str,
        consumed: int,
        pending: int | None,
        expected: Iterable[str],
        logical: Mapping | None = None,
    ):
        for item in (contract, execution, adapter, input_digest):
            require_sha(item)
        if phase not in PHASES:
            raise DiagnosticError("unsupported capture phase")
        integer(consumed)
        if pending is not None:
            integer(pending)
        names = tuple(require_name(v) for v in expected)
        if not names or len(set(names)) != len(names):
            raise DiagnosticError("frame coverage must be nonempty and unique")
        root.mkdir(mode=0o700)
        private_directory(root)
        self.root, self.expected, self.components = root, names, {}
        self.header = {
            "schema": SCHEMA,
            "contract": contract,
            "execution": execution,
            "adapter": adapter,
            "input_digest": input_digest,
            "phase": phase,
            "consumed": consumed,
            "pending": pending,
            "logical": dict(logical or {}),
        }
        self.finished = False

    def add(self, name: str, raw: bytes, *, dtype: str, shape: Iterable[int]) -> None:
        if type(raw) is not bytes:
            raise DiagnosticError("unsupported tensor representation")
        self.add_stream(name, (raw,), dtype=dtype, shape=shape)

    def add_stream(self, name: str, chunks, *, dtype: str, shape: Iterable[int]) -> None:
        """Write bounded chunks; incomplete/extra data cannot publish a frame."""
        if self.finished or name not in self.expected or name in self.components:
            raise DiagnosticError("duplicate, unexpected or late tensor observation")
        shape = list(shape)
        for dimension in shape:
            integer(dimension)
        if dtype not in DTYPES:
            raise DiagnosticError("unsupported tensor representation")
        expected = math.prod(shape) * DTYPES[dtype].itemsize
        name_digest = digest(name)
        filename = name_digest + ".bin"
        fd = os.open(self.root / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        size, checksum = 0, hashlib.sha256()
        with os.fdopen(fd, "wb") as f:
            for chunk in chunks:
                if not isinstance(chunk, (bytes, memoryview)):
                    raise DiagnosticError("unsupported tensor chunk")
                size += len(chunk)
                if size > expected:
                    raise DiagnosticError("tensor shape does not describe its payload")
                f.write(chunk)
                checksum.update(chunk)
            if size != expected:
                raise DiagnosticError("tensor shape does not describe its payload")
            f.flush()
            os.fsync(f.fileno())
        self.components[name] = {
            "file": filename,
            "dtype": dtype,
            "shape": shape,
            "nbytes": size,
            "sha256": checksum.hexdigest(),
        }

    def array(self, name: str, value: np.ndarray) -> None:
        array = np.asarray(value)
        dtype = array.dtype.newbyteorder("<")
        if dtype.str not in DTYPES:
            raise DiagnosticError("unsupported canonical array dtype")
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        raw = memoryview(contiguous.reshape(-1)).cast("B")
        self.add_stream(
            name,
            (raw[start : start + 1024 * 1024] for start in range(0, len(raw), 1024 * 1024)),
            dtype=dtype.str,
            shape=array.shape,
        )

    def finish(self) -> dict:
        if self.finished or set(self.components) != set(self.expected):
            raise DiagnosticError("cannot publish incomplete or already published frame")
        document = seal(
            {**self.header, "coverage": list(self.expected), "components": self.components}
        )
        write_private(self.root / "frame.json", document)
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.finished = True
        return document


def read_frame(root: Path) -> dict:
    private_directory(root)
    frame = private_json(root / "frame.json")
    authenticate(frame)
    if set(frame) != {
        "schema",
        "contract",
        "execution",
        "adapter",
        "input_digest",
        "phase",
        "consumed",
        "pending",
        "logical",
        "coverage",
        "components",
        "sha256",
    }:
        raise DiagnosticError("frame has incomplete metadata")
    if frame["schema"] != SCHEMA or frame["phase"] not in PHASES:
        raise DiagnosticError("unsupported canonical frame")
    for name in ("contract", "execution", "adapter", "input_digest"):
        require_sha(frame[name])
    integer(frame["consumed"])
    if not isinstance(frame["logical"], dict) or not isinstance(frame["components"], dict):
        raise DiagnosticError("invalid logical state metadata")
    if frame["pending"] is not None:
        integer(frame["pending"])
    coverage = frame["coverage"]
    if (
        not isinstance(coverage, list)
        or not coverage
        or any(not isinstance(v, str) for v in coverage)
    ):
        raise DiagnosticError("invalid frame coverage")
    if len(set(coverage)) != len(coverage) or set(coverage) != set(frame["components"]):
        raise DiagnosticError("incomplete frame coverage")
    for name, descriptor in frame["components"].items():
        require_name(name)
        if set(descriptor) != {"file", "dtype", "shape", "nbytes", "sha256"}:
            raise DiagnosticError("incomplete tensor descriptor")
        require_sha(descriptor["sha256"])
        if descriptor["file"] != digest(name) + ".bin" or descriptor["dtype"] not in DTYPES:
            raise DiagnosticError("tensor name or dtype mismatch")
        if not isinstance(descriptor["shape"], list):
            raise DiagnosticError("invalid tensor shape")
        for dimension in descriptor["shape"]:
            integer(dimension)
        if (
            integer(descriptor["nbytes"])
            != math.prod(descriptor["shape"]) * DTYPES[descriptor["dtype"]].itemsize
        ):
            raise DiagnosticError("tensor size mismatch")
    return frame


def compare_frames(left: Path, right: Path, *, chunk_bytes: int = 1024 * 1024) -> dict:
    """Bounded-memory exact comparison, including numerics and first byte offset."""
    a, b = read_frame(left), read_frame(right)
    for key in ("contract", "input_digest", "phase", "coverage"):
        if a[key] != b[key]:
            raise DiagnosticError("different inputs, semantics or observation coverage")
    if chunk_bytes < 8 or chunk_bytes % 8:
        raise DiagnosticError("comparison chunks must align every admitted dtype")
    results, first = [], None
    metadata_equal = all(a[k] == b[k] for k in ("consumed", "pending", "logical"))
    if not metadata_equal:
        first = {"boundary": "logical_state", "kind": "metadata"}
    for name in a["coverage"]:
        da, db = a["components"][name], b["components"][name]
        representation_equal = all(da[k] == db[k] for k in ("dtype", "shape", "nbytes"))
        if not representation_equal:
            raise DiagnosticError("tensor representations require an explicit adapter")
        hashes = [hashlib.sha256(), hashlib.sha256()]
        byte_offset, first_offset, mismatch_count, nonfinite = 0, None, 0, [0, 0]
        max_abs, squared, ref_squared, count = (
            np.longdouble(0),
            np.longdouble(0),
            np.longdouble(0),
            0,
        )
        with open_blob(left, da["file"]) as fa, open_blob(right, db["file"]) as fb:
            identities = [os.fstat(f.fileno()) for f in (fa, fb)]
            while True:
                ra, rb = fa.read(chunk_bytes), fb.read(chunk_bytes)
                if not ra and not rb:
                    break
                if len(ra) != len(rb) or len(ra) % DTYPES[da["dtype"]].itemsize:
                    raise DiagnosticError("tensor payload truncated")
                hashes[0].update(ra)
                hashes[1].update(rb)
                different = np.frombuffer(ra, dtype=np.uint8) != np.frombuffer(rb, dtype=np.uint8)
                if different.any():
                    if first_offset is None:
                        first_offset = byte_offset + int(np.flatnonzero(different)[0])
                    mismatch_count += int(np.count_nonzero(different))
                if (
                    da["dtype"] in FP8_METRIC_DTYPES
                    and len(ra) <= FP8_METRIC_MAX
                    and np.finfo(np.longdouble).nmant >= 63
                ):
                    metrics = fp8_chunk_metrics(ra, rb, da["dtype"])
                    nonfinite[0] += metrics["nonfinite"][0]
                    nonfinite[1] += metrics["nonfinite"][1]
                    max_abs = max(max_abs, metrics["max_abs"])
                    squared += metrics["squared"]
                    ref_squared += metrics["reference_squared"]
                    count += metrics["count"]
                elif da["dtype"] != "bytes":
                    va, vb = values(ra, da["dtype"]), values(rb, db["dtype"])
                    finite = np.isfinite(va) & np.isfinite(vb)
                    nonfinite[0] += int(np.count_nonzero(~np.isfinite(va)))
                    nonfinite[1] += int(np.count_nonzero(~np.isfinite(vb)))
                    delta = va[finite].astype(np.longdouble) - vb[finite].astype(np.longdouble)
                    if delta.size:
                        max_abs = max(max_abs, np.max(np.abs(delta)))
                        squared += np.sum(delta * delta)
                        ref_squared += np.sum(va[finite].astype(np.longdouble) ** 2)
                        count += delta.size
                byte_offset += len(ra)
            for f, before in zip((fa, fb), identities, strict=True):
                after = os.fstat(f.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise DiagnosticError("tensor changed during comparison")
        if byte_offset != da["nbytes"] or any(
            h.hexdigest() != d["sha256"] for h, d in zip(hashes, (da, db), strict=True)
        ):
            raise DiagnosticError("tensor content does not match its sealed descriptor")
        equal = mismatch_count == 0 and nonfinite == [0, 0]
        row = {
            "boundary": name,
            "exact_equal": equal,
            "differing_bytes": mismatch_count,
            "first_byte_offset": first_offset,
            "nonfinite": nonfinite,
            "max_abs": finite_metric(max_abs),
            "rmse": finite_metric(np.sqrt(squared / count)) if count else None,
            "relative_l2": finite_metric(np.sqrt(squared / ref_squared)) if ref_squared else None,
        }
        results.append(row)
        if not equal and first is None:
            first = {"boundary": name, "kind": "tensor", "byte_offset": first_offset}
    return seal(
        {
            "schema": "urn:qwen:canonical-state-comparison:v1",
            "equal": first is None,
            "first_difference": first,
            "components": results,
            "metadata_equal": metadata_equal,
            "reference_frame": a["sha256"],
            "candidate_frame": b["sha256"],
            "scope": "observed_exact_logical_state",
            "formal_backend_equivalence": "UNPROVED",
        }
    )


def finite_metric(value):
    """Overflowing diagnostics never turn an exact mismatch into a pass."""
    return float(value) if abs(value) <= np.finfo(np.float64).max else "overflow"


def load_arrays(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    frame = read_frame(root)
    arrays = {}
    for name, d in frame["components"].items():
        with open_blob(root, d["file"]) as f:
            before = os.fstat(f.fileno())
            raw = f.read()
            after = os.fstat(f.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise DiagnosticError("frame payload changed while reading")
        if len(raw) != d["nbytes"] or hashlib.sha256(raw).hexdigest() != d["sha256"]:
            raise DiagnosticError("frame payload is corrupt")
        arrays[name] = np.frombuffer(raw, dtype=DTYPES[d["dtype"]]).reshape(d["shape"]).copy()
    return frame, arrays


def archive_frame(source: Path, destination: Path, *, reflink: bool = False) -> dict:
    """Create an independent, durable copy and authenticate its actual payload.

    The optional Linux FICLONE path still reads and hashes every destination
    byte. It requires filesystem support: there is no silent full-copy fallback
    that could exhaust a caller's storage budget. Neither path uses hard links.
    """
    f = read_frame(source)
    writer = FrameWriter(
        destination,
        **{
            k: f[k]
            for k in (
                "contract",
                "execution",
                "adapter",
                "input_digest",
                "phase",
                "consumed",
                "pending",
                "logical",
            )
        },
        expected=f["coverage"],
    )
    # Stream instead of allocating a complete 250K frame in RAM.
    for name in f["coverage"]:
        d = f["components"][name]
        h, size = hashlib.sha256(), 0
        with open_blob(source, d["file"]) as src:
            before = os.fstat(src.fileno())
            fd = os.open(destination / d["file"], os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w+b") as dst:
                if reflink:
                    # Linux fs.h: FICLONE = _IOW(0x94, 9, int).
                    fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
                    # Verify the clone itself, not just the source or manifest.
                    while chunk := dst.read(1024 * 1024):
                        h.update(chunk)
                        size += len(chunk)
                else:
                    while chunk := src.read(1024 * 1024):
                        h.update(chunk)
                        size += len(chunk)
                        dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            after = os.fstat(src.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise DiagnosticError("frame changed during archival")
        if h.hexdigest() != d["sha256"] or size != d["nbytes"]:
            raise DiagnosticError("frame payload changed before archival")
        writer.components[name] = dict(d)
    result = writer.finish()
    if result != f:
        raise DiagnosticError("archival changed canonical frame metadata")
    return result
