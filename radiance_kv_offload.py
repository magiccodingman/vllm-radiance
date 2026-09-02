"""Host-registration helpers for mmap-backed KV offload.

The vLLM CPU KV-offload mmap is best-effort pinned with ``hipHostRegister`` on
ROCm.  A failed registration leaves a pending, thread-local HIP runtime error;
if it is not consumed, the next unrelated torch operation can fail with
``hipErrorInvalidValue``.  Tensor-parallel workers also map the same backing
file independently, so registration must resolve to one coherent state across
the model-parallel group.

This module deliberately has no torch/vLLM imports.  The source overlay in
``patch_kv_offload_registration.py`` owns distributed coordination while these
helpers provide deterministic policy parsing, row-aligned chunk planning, and
same-runtime-handle registration/rollback.  Keeping this layer pure makes the
failure paths testable without a GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


PIN_POLICY_ENV = "RADIANCE_KV_OFFLOAD_PIN_POLICY"
REGISTER_CHUNK_GIB_ENV = "RADIANCE_KV_OFFLOAD_REGISTER_CHUNK_GIB"
VALID_PIN_POLICIES = frozenset({"auto", "required", "disabled"})


class HostRuntime(Protocol):
    """Subset of ``CudaRTLibrary`` used by KV host registration."""

    def cudaHostRegister(self, ptr: int, size: int, flags: int = 0) -> int: ...

    def cudaHostUnregister(self, ptr: int) -> int: ...

    def drain_pending_error(self) -> int: ...


@dataclass(frozen=True)
class RollbackResult:
    remaining_chunks: tuple[tuple[int, int], ...]
    error_code: int | None = None
    exception: Exception | None = None
    drained_error_code: int | None = None

    @property
    def ok(self) -> bool:
        return not self.remaining_chunks and self.exception is None


@dataclass(frozen=True)
class RegistrationResult:
    chunks: tuple[tuple[int, int], ...]
    error_code: int | None = None
    exception: Exception | None = None
    drained_error_code: int | None = None
    rollback: RollbackResult | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None and self.exception is None

    @property
    def rollback_ok(self) -> bool:
        return self.rollback is None or self.rollback.ok


def get_pin_policy(environ: dict[str, str] | None = None) -> str:
    """Return the validated host-registration policy.

    ``auto`` is intentionally the default: use pinned DMA only when every
    model-parallel worker succeeds, otherwise fall back coherently to pageable
    DMA.  ``required`` converts an ordinary registration failure into startup
    failure.  ``disabled`` skips registration entirely.
    """

    source = os.environ if environ is None else environ
    policy = source.get(PIN_POLICY_ENV, "auto").strip().lower()
    if policy not in VALID_PIN_POLICIES:
        choices = ", ".join(sorted(VALID_PIN_POLICIES))
        raise ValueError(f"{PIN_POLICY_ENV} must be one of {choices}; got {policy!r}")
    return policy


def get_register_chunk_bytes(environ: dict[str, str] | None = None) -> int:
    """Read the opt-in registration chunk size, returning zero for one call."""

    source = os.environ if environ is None else environ
    raw = source.get(REGISTER_CHUNK_GIB_ENV, "0").strip()
    try:
        gib = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{REGISTER_CHUNK_GIB_ENV} must be a number; got {raw!r}"
        ) from error
    if gib < 0:
        raise ValueError(f"{REGISTER_CHUNK_GIB_ENV} must be >= 0; got {raw!r}")
    return int(gib * 1024**3)


def plan_registration_chunks(
    total_size: int,
    row_stride: int,
    requested_chunk_bytes: int,
) -> tuple[tuple[int, int], ...]:
    """Plan complete, row-aligned registrations over ``total_size`` bytes."""

    if total_size <= 0:
        raise ValueError("total_size must be positive")
    if row_stride <= 0 or total_size % row_stride:
        raise ValueError("row_stride must be positive and divide total_size")
    if requested_chunk_bytes < 0:
        raise ValueError("requested_chunk_bytes must be non-negative")

    if requested_chunk_bytes == 0 or requested_chunk_bytes >= total_size:
        return ((0, total_size),)

    rows_per_chunk = max(requested_chunk_bytes // row_stride, 1)
    chunk_size = rows_per_chunk * row_stride
    chunks: list[tuple[int, int]] = []
    offset = 0
    while offset < total_size:
        size = min(chunk_size, total_size - offset)
        chunks.append((offset, size))
        offset += size
    return tuple(chunks)


def rollback_host_chunks(
    runtime: HostRuntime,
    base_ptr: int,
    chunks: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> RollbackResult:
    """Unregister in reverse order and preserve any chunks not released."""

    remaining = list(chunks)
    first_error: int | None = None
    first_exception: Exception | None = None
    drained: int | None = None

    for offset, _size in reversed(tuple(chunks)):
        try:
            result = runtime.cudaHostUnregister(base_ptr + offset)
        except Exception as error:  # retain ownership for later cleanup
            if first_exception is None:
                first_exception = error
            continue
        if result != 0:
            if first_error is None:
                first_error = result
            try:
                drained = runtime.drain_pending_error()
            except Exception as error:
                if first_exception is None:
                    first_exception = error
            continue
        remaining.remove((offset, _size))

    return RollbackResult(
        remaining_chunks=tuple(remaining),
        error_code=first_error,
        exception=first_exception,
        drained_error_code=drained,
    )


def register_host_chunks(
    runtime: HostRuntime,
    base_ptr: int,
    total_size: int,
    row_stride: int,
    requested_chunk_bytes: int = 0,
) -> RegistrationResult:
    """Register all chunks or roll back every locally successful chunk.

    A failed runtime call is drained through the *same* library handle before
    returning.  This is the key invariant that prevents the subsequent torch
    call from inheriting a stale HIP error.
    """

    planned = plan_registration_chunks(total_size, row_stride, requested_chunk_bytes)
    registered: list[tuple[int, int]] = []
    for offset, size in planned:
        try:
            result = runtime.cudaHostRegister(base_ptr + offset, size, 0)
        except Exception as error:
            rollback = rollback_host_chunks(runtime, base_ptr, registered)
            return RegistrationResult(
                chunks=rollback.remaining_chunks,
                exception=error,
                rollback=rollback,
            )
        if result != 0:
            try:
                drained = runtime.drain_pending_error()
            except Exception as error:
                rollback = rollback_host_chunks(runtime, base_ptr, registered)
                return RegistrationResult(
                    chunks=rollback.remaining_chunks,
                    error_code=result,
                    exception=error,
                    rollback=rollback,
                )
            rollback = rollback_host_chunks(runtime, base_ptr, registered)
            return RegistrationResult(
                chunks=rollback.remaining_chunks,
                error_code=result,
                drained_error_code=drained,
                rollback=rollback,
            )
        registered.append((offset, size))

    return RegistrationResult(chunks=tuple(registered))
