"""Durable priority queue for the finite GPU campaign; no GPU imports."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private

STAGES = ("pilot", "focused", "extended")


def replace_private(root, name, document):
    temporary = root / f".{name}.{os.getpid()}.{time.time_ns()}"
    write_private(temporary, document)
    temporary.replace(root / name)
    descriptor = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def space_required(case, reserve):
    # Conservative state-export allowance, not a prediction of disk traffic.
    # Long forced M1/D7 comparisons keep multiple complete state frames.
    frames = 20 if case["family"] == "forced_d7" else 8
    capture = case["context"] * 32768 * frames
    return reserve + max(4 * 1024**3, capture)


def request_pause(root):
    root = Path(root).resolve()
    private_json(root / "campaign.json")
    replace_private(root, "pause-request.json", {"requested_ns": time.time_ns()})
    return {"pause_requested": True, "boundary": "after_current_case"}


def run(
    campaign,
    root,
    *,
    selected,
    keep_going,
    resume,
    through,
    budget_seconds,
    retry_failed,
    min_free_bytes,
):
    from qwen_r9700_lab import conformance_campaign as runner

    if through not in STAGES or (budget_seconds is not None and budget_seconds <= 0):
        raise DiagnosticError("invalid campaign stage or time budget")
    if min_free_bytes < 0:
        raise DiagnosticError("invalid free-space reserve")
    root = Path(root).resolve()
    if resume:
        if not root.is_dir():
            raise DiagnosticError("cannot resume a missing campaign")
    else:
        root.mkdir(mode=0o700)
    lock = os.open(root / "queue.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DiagnosticError("campaign already has an active controller") from error
        if resume:
            results = runner.load_results(campaign, root)
            # An interrupted attempt is evidence, never silently a fresh case.
            for attempt in sorted(root.glob("case-*")):
                if not (attempt / "input.json").is_file() or (attempt / "result.json").exists():
                    continue
                payload = private_json(attempt / "input.json")
                if payload["campaign"] != campaign:
                    raise DiagnosticError("interrupted attempt belongs to another campaign")
                case = payload["case"]
                result = runner.case_result(
                    campaign,
                    case,
                    "ERROR",
                    started=time.monotonic(),
                    error_type="InterruptedAttempt",
                    detail="Controller stopped before a durable result; artifacts retained.",
                )
                write_private(attempt / "result.json", result)
                results.setdefault(case["id"], []).append(result)
            (root / "pause-request.json").unlink(missing_ok=True)
        else:
            write_private(root / "campaign.json", campaign)
            results = {}
        runner.coverage(campaign, results)  # validate all resumed attempts first
        started = time.monotonic()

        def checkpoint(status, current=None, **details):
            report = runner.coverage(campaign, results)
            replace_private(root, "coverage.json", report)
            value = seal(
                {
                    "schema": "urn:qwen:conformance-checkpoint:v1",
                    "campaign": campaign["sha256"],
                    "updated_ns": time.time_ns(),
                    "status": status,
                    "current": current,
                    "through": through,
                    "counts": report["counts"],
                    "elapsed_this_run": time.monotonic() - started,
                    **details,
                }
            )
            replace_private(root, "checkpoint.json", value)
            print(
                json.dumps({"status": status, "case": current, "counts": report["counts"]}),
                flush=True,
            )
            return report

        checkpoint("running")
        for index, case in enumerate(campaign["cases"]):
            if case["id"] not in selected or STAGES.index(case["stage"]) > STAGES.index(through):
                continue
            attempts = results.get(case["id"], [])
            if attempts and attempts[-1]["status"] == "TESTED":
                continue
            if attempts and not retry_failed:
                if not keep_going:
                    return checkpoint("failed_attempt_requires_review", case["id"])
                continue
            if (root / "pause-request.json").exists():
                return checkpoint("paused", case["id"])
            from qwen_r9700_lab.conformance_gpu_lease import cleanup_block

            if cleanup_block() is not None:
                return checkpoint("gpu_cleanup_incomplete", case["id"])
            if budget_seconds is not None and time.monotonic() - started >= budget_seconds:
                return checkpoint("time_budget_reached", case["id"])
            required = space_required(case, min_free_bytes)
            free = shutil.disk_usage(root).free
            if free < required:
                return checkpoint(
                    "insufficient_space", case["id"], free_bytes=free, required_bytes=required
                )
            name = f"case-{index:05d}" + (f"-attempt-{len(attempts):03d}" if attempts else "")
            case_root = root / name
            case_root.mkdir(mode=0o700)
            write_private(case_root / "input.json", {"campaign": campaign, "case": case})
            checkpoint("running_case", case["id"], attempt=name)
            result = runner.execute_case(campaign, case, case_root)
            write_private(case_root / "result.json", result)
            results.setdefault(case["id"], []).append(result)
            report = checkpoint("case_finished", case["id"])
            write_private(root / f"checkpoint-{name}.json", report)
            if result["status"] != "TESTED" and not keep_going:
                return checkpoint("failed", case["id"])
        return checkpoint(
            "complete" if runner.coverage(campaign, results)["complete"] else "stage_finished"
        )
    finally:
        os.close(lock)
