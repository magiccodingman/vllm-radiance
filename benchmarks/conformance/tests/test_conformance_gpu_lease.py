"""Process-level ownership tests; no GPU imports or device access."""

import multiprocessing
import os
import sys

import pytest

from qwen_r9700_lab.conformance_gpu_lease import block_cleanup, cleanup_block, gpu_lease
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json


def hold(path, evidence, acquired, release):
    os.environ["QWEN_CONFORMANCE_GPU_LOCK"] = str(path)
    with gpu_lease(evidence):
        acquired.set()
        if not release.wait(15):
            raise TimeoutError("test did not release its worker")


def die_holding(path, evidence):
    os.environ["QWEN_CONFORMANCE_GPU_LOCK"] = str(path)
    with gpu_lease(evidence):
        os._exit(9)


def test_two_controllers_never_hold_gpu_lease_together(tmp_path):
    context = multiprocessing.get_context("spawn")
    acquired = [context.Event(), context.Event()]
    release = [context.Event(), context.Event()]
    workers = [
        context.Process(
            target=hold, args=(tmp_path / "lock", tmp_path / f"p{i}", acquired[i], release[i])
        )
        for i in range(2)
    ]
    try:
        workers[0].start()
        assert acquired[0].wait(10)
        workers[1].start()
        assert not acquired[1].wait(0.2)
        release[0].set()
        assert acquired[1].wait(10)
        release[1].set()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        assert private_json(tmp_path / "lock.owner.json")["status"] == "released"
    finally:
        for event in release:
            event.set()
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(10)


def test_unclean_owner_blocks_next_case_until_cleanup_is_reviewed(tmp_path, monkeypatch):
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "lock"
    worker = context.Process(target=die_holding, args=(path, tmp_path / "dead"))
    worker.start()
    worker.join(10)
    if worker.is_alive():
        worker.terminate()
        worker.join(10)
        pytest.fail("fault worker did not exit")
    assert worker.exitcode == 9
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(path))
    with (
        pytest.raises(DiagnosticError, match="did not release cleanly"),
        gpu_lease(tmp_path / "next"),
    ):
        pytest.fail("an unreviewed dead owner must prevent admission")


def test_python_error_releases_after_owned_gpu_scope_unwinds(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(tmp_path / "lock"))
    with pytest.raises(ValueError), gpu_lease(tmp_path / "first"):
        raise ValueError("synthetic failure")
    with gpu_lease(tmp_path / "second"):
        assert private_json(tmp_path / "lock.owner.json")["status"] == "active"


def test_unconfigured_lease_does_not_create_resources(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_CONFORMANCE_GPU_LOCK", raising=False)
    with gpu_lease(tmp_path / "unused"):
        pass
    assert not (tmp_path / "unused").exists()


def test_symlink_lock_is_rejected(tmp_path, monkeypatch):
    target, path = tmp_path / "target", tmp_path / "lock"
    target.touch()
    path.symlink_to(target)
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(path))
    with pytest.raises(OSError), gpu_lease(tmp_path / "evidence"):
        pytest.fail("symlink lock admitted")


def test_surviving_worker_blocks_next_lease_and_repeated_close(tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_transport as transport

    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(tmp_path / "lock"))
    # The real launcher exits; the injected observation models a kernel-stuck
    # descendant that cannot be created safely in an ordinary regression test.
    monkeypatch.setattr(
        transport, "await_group_exit", lambda group: [{"pid": group + 1, "state": "D"}]
    )
    with pytest.raises(DiagnosticError, match="remain alive"), gpu_lease(tmp_path / "first"):
        child = transport.OwnedProcess(
            [sys.executable, "-c", "pass"], tmp_path / "child", env=dict(os.environ), timeout=10
        )
        child.wait()
    assert child.process.poll() is not None
    assert child.log.closed
    assert cleanup_block()["members"][0]["state"] == "D"
    assert private_json(tmp_path / "lock.owner.json")["status"] == "cleanup_incomplete"
    assert not (tmp_path / "first/released.json").exists()
    with pytest.raises(DiagnosticError, match="remain alive") as caught:
        child.close()
    assert caught.value is child.close_error
    with pytest.raises(DiagnosticError, match="admission blocked"), gpu_lease(tmp_path / "next"):
        pytest.fail("a dead launcher does not prove its workers released the GPU")


def test_diagnostic_write_failure_does_not_certify_clean_release(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(tmp_path / "lock"))
    with pytest.raises(FileNotFoundError), gpu_lease(tmp_path / "scope"):
        block_cleanup(
            tmp_path / "missing-directory", process_group=17, reason="fixture", members=[]
        )
    assert cleanup_block() is not None
    assert private_json(tmp_path / "lock.owner.json")["status"] == "cleanup_incomplete"
    assert not (tmp_path / "scope/released.json").exists()


def test_failure_to_write_block_marker_still_preserves_unclean_owner(tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_gpu_lease as lease

    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(tmp_path / "lock"))
    replace = lease.replace_private

    def unavailable(root, name, document):
        if name.endswith(".blocked.json"):
            raise OSError("synthetic marker write failure")
        return replace(root, name, document)

    monkeypatch.setattr(lease, "replace_private", unavailable)
    with pytest.raises(OSError, match="marker write failure"), gpu_lease(tmp_path / "scope"):
        block_cleanup(tmp_path / "scope", process_group=17, reason="fixture", members=[])
    assert not (tmp_path / "lock.blocked.json").exists()
    assert private_json(tmp_path / "lock.owner.json")["status"] == "cleanup_incomplete"
    assert not (tmp_path / "scope/released.json").exists()
    with pytest.raises(DiagnosticError, match="admission blocked"), gpu_lease(tmp_path / "next"):
        pytest.fail("a diagnostic I/O failure must not permit another GPU job")
