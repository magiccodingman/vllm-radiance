#!/usr/bin/env python3
"""GPU-free regression checks for Radiance KV host registration helpers."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from radiance_kv_offload import (  # noqa: E402
    PIN_POLICY_ENV,
    REGISTER_CHUNK_GIB_ENV,
    get_pin_policy,
    get_register_chunk_bytes,
    plan_registration_chunks,
    register_host_chunks,
    rollback_host_chunks,
)


class FakeRuntime:
    def __init__(
        self,
        register_results: list[int] | None = None,
        unregister_results: list[int] | None = None,
    ) -> None:
        self.register_results = list(register_results or [])
        self.unregister_results = list(unregister_results or [])
        self.events: list[tuple[str, int, int] | tuple[str, int] | tuple[str]] = []

    def cudaHostRegister(self, ptr: int, size: int, flags: int = 0) -> int:
        self.events.append(("register", ptr, size))
        return self.register_results.pop(0) if self.register_results else 0

    def cudaHostUnregister(self, ptr: int) -> int:
        self.events.append(("unregister", ptr))
        return self.unregister_results.pop(0) if self.unregister_results else 0

    def drain_pending_error(self) -> int:
        self.events.append(("drain",))
        return 1


def check_policy() -> None:
    assert get_pin_policy({}) == "auto"
    for value in ("auto", "required", "disabled"):
        assert get_pin_policy({PIN_POLICY_ENV: value.upper()}) == value
    try:
        get_pin_policy({PIN_POLICY_ENV: "maybe"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pin policy was accepted")

    assert get_register_chunk_bytes({}) == 0
    assert get_register_chunk_bytes({REGISTER_CHUNK_GIB_ENV: "0.5"}) == 512 * 1024**2


def check_chunk_planning() -> None:
    assert plan_registration_chunks(100, 10, 0) == ((0, 100),)
    assert plan_registration_chunks(100, 10, 45) == (
        (0, 40),
        (40, 40),
        (80, 20),
    )
    assert plan_registration_chunks(30, 10, 1) == (
        (0, 10),
        (10, 10),
        (20, 10),
    )


def check_success() -> None:
    runtime = FakeRuntime()
    result = register_host_chunks(runtime, 1000, 100, 10, 40)
    assert result.ok and result.chunks == ((0, 40), (40, 40), (80, 20))
    assert runtime.events == [
        ("register", 1000, 40),
        ("register", 1040, 40),
        ("register", 1080, 20),
    ]


def check_failure_drains_and_rolls_back() -> None:
    runtime = FakeRuntime(register_results=[0, 1])
    result = register_host_chunks(runtime, 2000, 100, 10, 40)
    assert not result.ok and result.error_code == 1
    assert result.drained_error_code == 1
    assert result.rollback is not None and result.rollback.ok
    assert result.chunks == ()
    assert runtime.events == [
        ("register", 2000, 40),
        ("register", 2040, 40),
        ("drain",),
        ("unregister", 2000),
    ]


def check_rollback_preserves_failed_ownership() -> None:
    runtime = FakeRuntime(unregister_results=[1, 0])
    result = rollback_host_chunks(runtime, 3000, [(0, 40), (40, 40)])
    assert not result.ok
    assert result.error_code == 1
    assert result.remaining_chunks == ((40, 40),)
    assert runtime.events == [
        ("unregister", 3040),
        ("drain",),
        ("unregister", 3000),
    ]


def check_image_wiring() -> None:
    for name in ("Dockerfile", "Dockerfile.dev", "Dockerfile.patch"):
        source = (REPO / name).read_text()
        assert "radiance_kv_offload.py" in source, name
        assert "patch_kv_offload_registration" in source, name

    entrypoint = (REPO / "radiance_entrypoint.sh").read_text()
    assert '"--kv-offloading-size=${RADIANCE_KV_OFFLOADING_SIZE}"' in entrypoint

    compose = (REPO / "docker-compose.yml").read_text()
    for variable in (
        "RADIANCE_KV_OFFLOADING_SIZE",
        "RADIANCE_KV_OFFLOAD_PIN_POLICY",
        "RADIANCE_KV_OFFLOAD_REGISTER_CHUNK_GIB",
    ):
        assert variable in compose


def main() -> None:
    check_policy()
    check_chunk_planning()
    check_success()
    check_failure_drains_and_rolls_back()
    check_rollback_preserves_failed_ownership()
    check_image_wiring()
    print("KV offload host-registration regression checks: PASS")


if __name__ == "__main__":
    main()
