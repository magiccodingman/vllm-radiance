"""Concrete native and production-interface campaign drivers.

All prompts are deterministic synthetic records. No saved Pi transcript, user's
cache directory or pre-existing server endpoint is accepted by these drivers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qwen_r9700_lab.conformance_boundaries import compare_boundaries
from qwen_r9700_lab.conformance_observer import read_events
from qwen_r9700_lab.conformance_runtime import NativeServer, worker_environment
from qwen_r9700_lab.conformance_session import compare_campaign
from qwen_r9700_lab.conformance_state import compare_frames, read_frame
from qwen_r9700_lab.conformance_transport import BackendResponseError, OwnedProcess
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    private_json,
    seal,
    write_private,
)
from qwen_r9700_lab.radiance_cache import ChatStore, cache_salt, request_tail_flush


class UnavailableError(DiagnosticError):
    pass


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def synthetic_text(seed, lines):
    rng = random.Random(seed)
    # Nonrepeating public records, not the repeated prose that can inflate
    # speculative acceptance and conceal repetition failures in a benchmark.
    return "\n".join(
        json.dumps(
            {
                "record": i,
                "amount": rng.randrange(1, 10000000),
                "tag": f"{rng.getrandbits(64):016x}",
            },
            sort_keys=True,
        )
        for i in range(lines)
    )


def tokens(spec, count, seed):
    from tokenizers import Tokenizer

    checkpoint = Path(spec["native_config"]["model"])
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    result = tokenizer.encode(synthetic_text(seed, count // 8 + 32), add_special_tokens=False).ids
    require(len(result) >= count, "synthetic fixture did not fill the declared token window")
    return result[:count]


def plan_for(spec, case, forced, *, accepted=None):
    from qwen_r9700_lab.conformance_cli import make_plan

    prefix = tokens(spec, case["context"], case["seed"])
    # Explicit capture windows retain boundary-adjacent and accepted rows. This
    # limits diagnostic tensor traffic; the omitted positions remain unobserved.
    positions = sorted(
        {0, len(prefix) - 1, *list(range(max(0, len(prefix) - 8), len(prefix) + len(forced) - 1))}
    )
    plan, _, _ = make_plan(
        {
            "checkpoint": spec["native_config"]["model"],
            "checkpoint_files": spec["checkpoint_files"],
            "kv_scales": spec["kv_scales"],
            "reference_profile": spec["reference_profile"],
            "prefix": prefix,
            "forced_tokens": forced,
            "observation_positions": positions,
            **({"accepted_widths": accepted} if accepted is not None else {}),
            **(
                {"reference_linear": spec["reference_linear"]} if "reference_linear" in spec else {}
            ),
        }
    )
    return plan


def native_configuration(spec, *, speculation=True):
    config = dict(spec["native_config"])
    config.update(enforce_eager=True, max_num_seqs=1, async_scheduling=False)
    config.pop("kv_transfer_config", None)
    config.pop("scheduler_cls", None)
    config.pop("additional_config", None)
    config.pop("compilation_config", None)
    if not speculation:
        config.pop("speculative_config", None)
    return config


def native(spec, plan, root, *, speculation=True, experiment=None):
    from qwen_r9700_lab.conformance_cli import run_native

    root.mkdir(mode=0o700)
    config = native_configuration(spec, speculation=speculation)
    write_private(root / "plan.json", plan)
    write_private(root / "config.json", config)
    write_private(root / "binding.json", spec["binding"])
    prior = {
        key: os.environ.get(key)
        for key in (
            "RADIANCE_VERIFY_HEAD",
            "QWEN_CONFORMANCE_NATIVE_EXPERIMENT",
            "QWEN_CONFORMANCE_CALL_MODE",
        )
    }
    os.environ["RADIANCE_VERIFY_HEAD"] = "0"
    os.environ["QWEN_CONFORMANCE_CALL_MODE"] = spec["native_call_mode"]
    os.environ.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
    if experiment is not None:
        write_private(root / "experiment.json", experiment)
        os.environ["QWEN_CONFORMANCE_NATIVE_EXPERIMENT"] = str(root / "experiment.json")
    try:
        run_native(
            root / "plan.json",
            root / "config.json",
            root / "binding.json",
            root / "run",
            allow_gpu=True,
        )
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return root / "run/capture"


def serial_reference_identity(spec, root):
    """Bind all effective settings, hashing environment values instead of logging them."""
    from qwen_r9700_lab.conformance_reference_store import SCHEMA, copied_tree_identity

    payload = private_json(root / "input.json")
    campaign = payload["campaign"]
    authenticate(campaign)
    require(campaign["spec"] == spec, "serial reference spec differs from its case")
    env = dict(os.environ)
    env.update(RADIANCE_VERIFY_HEAD="0", QWEN_CONFORMANCE_CALL_MODE=spec["native_call_mode"])
    env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
    # Only exact per-case scratch paths installed by worker_environment may be
    # normalized. Unknown variables/paths remain bound, causing a safe miss.
    for key, directory in {
        "VLLM_CACHE_ROOT": "vllm",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "XDG_CACHE_HOME": "xdg-cache",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        if env.get(key) == str(root / "runtime" / directory):
            env[key] = "<qualification-case-runtime>/" + directory
    # worker_environment copies this source/JIT tree, unlike empty scratch
    # caches. Bind every copied byte before execution; never normalize away a
    # different kernel tree just because its directory name is familiar.
    aiter_tree = root / "runtime/aiter"
    if env.get("AITER_ROOT_DIR") == str(aiter_tree):
        env["AITER_ROOT_DIR"] = {
            "path": "<qualification-case-runtime>/aiter",
            "copied_tree": copied_tree_identity(aiter_tree),
        }
    return seal(
        {
            "schema": SCHEMA + "/identity",
            "campaign": campaign["sha256"],
            "configuration": native_configuration(spec, speculation=False),
            "environment": {k: digest(v) for k, v in sorted(env.items())},
        }
    )


def serial_reference_for_d7(spec, case, requested, root):
    """Reuse one complete serial eleven-token baseline; each D7 arm stays fresh."""
    from qwen_r9700_lab.conformance_reference_store import SerialReferenceStore

    require(private_json(root / "input.json")["case"] == case, "serial reference case changed")
    complete = plan_for(spec, case, tokens(spec, 11, case["seed"] + 1701))
    identity = serial_reference_identity(spec, root)
    serial = root / "serial"
    serial.mkdir(mode=0o700)
    write_private(serial / "plan.json", requested)
    write_private(serial / "config.json", identity["configuration"])
    write_private(serial / "binding.json", spec["binding"])
    (serial / "run").mkdir(mode=0o700)
    destination = serial / "run/capture"
    with SerialReferenceStore(root.parent / "serial-reference-store") as store:
        projection = store.project(identity, complete, requested, destination)
        hit = projection is not None
        if not hit:
            store.retire()
            baseline = native(spec, complete, root / "serial-population", speculation=False)
            store.publish(identity, complete, baseline, root)
            projection = store.project(identity, complete, requested, destination)
            require(projection is not None, "new serial reference was not published")
    write_private(
        serial / "reuse.json",
        seal(
            {
                "identity": identity["sha256"],
                "hit": hit,
                "projection": projection["sha256"],
                "fresh_d7_required": True,
            }
        ),
    )
    return destination


def aligned_frames(reference, candidate, root, *, require_equal=True):
    schedules = [private_json(p / "schedule.json") for p in (reference, candidate)]
    for schedule in schedules:
        authenticate(schedule)
        require(
            schedule.get("initial_state") == "independent_zero_state",
            "native comparison reused an unvalidated initial state",
        )
    reference_frames = {(f["consumed"], f["pending"]): f for f in schedules[0]["frames"]}
    reports = []
    for frame in schedules[1]["frames"]:
        corresponding = reference_frames.get((frame["consumed"], frame["pending"]))
        require(corresponding is not None, "M1 replay missed a D7 committed position")
        result = compare_frames(reference / corresponding["name"], candidate / frame["name"])
        reports.append(result)
    write_private(root / "aligned-comparison.json", {"comparisons": reports})
    require(bool(reports), "M1/D7 comparison has no aligned frames")
    if require_equal:
        require(
            all(r["equal"] for r in reports),
            "M1/D7 logical state or logits diverged; comparison retained",
        )
    return reports


def forced_reference(spec, case, root):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_replay import run_reference

    forced = tokens(spec, 4, case["seed"] + 1701)
    plan = plan_for(spec, case, forced)
    run_reference(plan, root / "reference")
    with gpu_lease(root / "gpu-lease"):
        candidate = native(spec, plan, root / "candidate", speculation=False)
    state = compare_campaign(root / "reference", candidate, root / "state-comparison")
    boundary = compare_boundaries(
        root / "reference/boundaries", candidate / "boundaries", root / "boundary-comparison"
    )
    require(
        state["equal"] and boundary["equal"],
        "independent reference diverged; first boundaries retained",
    )
    return [
        "independent_initial_prefill",
        "forced_transition_state_and_logits",
        "selected_layer_boundaries",
    ]


def forced_d7(spec, case, root):
    width = case["axes"]["accepted"]
    forced = tokens(spec, width + 4, case["seed"] + 1701)
    plan = plan_for(spec, case, forced, accepted=[width, 0, 0])
    serial_plan = plan_for(spec, case, forced)
    serial = (
        serial_reference_for_d7(spec, case, serial_plan, root)
        if spec.get("reuse_serial_reference", False)
        else native(spec, serial_plan, root / "serial", speculation=False)
    )
    candidate = native(spec, plan, root / "d7")
    state = aligned_frames(serial, candidate, root, require_equal=False)
    boundary = compare_boundaries(
        serial / "boundaries", candidate / "boundaries", root / "boundary-comparison"
    )
    # Complete both diagnoses before failing the case. A state mismatch is a
    # reason to retain the causal boundary report, not a reason to omit it.
    require(
        all(r["equal"] for r in state),
        "M1/D7 logical state or logits diverged; comparison retained",
    )
    require(boundary["equal"], "M1/D7 causal layer boundary divergence")
    return [
        f"accept_width_{width}",
        "same_consumed_pending_and_all_logical_state",
        "selected_causal_rows",
        "initial_prefill",
    ]


def rejected_suffix(spec, case, root):
    width = case["axes"]["accepted"]
    forced = tokens(spec, width + 4, case["seed"] + 1701)
    plan = plan_for(spec, case, forced, accepted=[width, 0, 0])
    a = native(spec, plan, root / "suffix-a", experiment={"rejected_suffix_token": 0})
    b = native(spec, plan, root / "suffix-b", experiment={"rejected_suffix_token": 1})
    report = compare_campaign(a, b, root / "comparison")
    require(report["equal"], "rejected draft suffix contaminated retained state or logits")
    return ["accepted_prefix_fixed", "rejected_suffix_changed", "exact_committed_state_and_logits"]


def native_fault(spec, case, root):
    fault = case["variant"]
    plan = plan_for(spec, case, tokens(spec, 3, 1701))
    clean = native(spec, plan, root / "clean", speculation=False)
    # A no-fault repeated capture first establishes that this negative control
    # is not just counting an unrelated natural numerical divergence as a hit.
    repeat = native(spec, plan, root / "clean-repeat", speculation=False)
    require(
        compare_campaign(clean, repeat, root / "clean-check")["equal"],
        "clean native control is not repeatable",
    )
    step = len(plan["forced_tokens"]) - 1 if fault == "missing_observation" else 0
    failed = False
    try:
        damaged = native(
            spec,
            plan,
            root / "damaged",
            speculation=False,
            experiment={"fault": fault, "step": step},
        )
    except DiagnosticError:
        failed = True
        damaged = root / "damaged/run/capture"
    applied = private_json(damaged / "fault-applied.json")
    require(applied.get("fault") == fault, "native fault was never applied")
    if fault in {"kv", "gdn", "conv"}:
        require(not failed, "native fault failed before the byte comparator observed it")
        report = compare_campaign(clean, damaged, root / "fault-comparison")
        require(not report["equal"], "actual GPU corruption escaped the state comparator")
        # Detect the intended state component, not a later unrelated output.
        comparisons = sorted((root / "fault-comparison").glob("comparison-*.json"))
        first = private_json(comparisons[0])
        require(
            any(
                not row["exact_equal"] and row["boundary"] == applied["component"]
                for row in first["components"]
            ),
            "fault comparison did not identify the corrupted component",
        )
    else:
        require(failed, "invalid native metadata or missing observation was accepted")
        error = private_json(root / "damaged/run/worker-failure.json")
        expected = {
            "pending": "native committed position/prefix",
            "position": "native committed position/prefix",
            "version": "invalid native committed state version",
            "missing_observation": "backend stopped before the replay finished",
        }
        require(
            error.get("type") == "DiagnosticError" and expected[fault] in error.get("message", ""),
            "negative control failed for an unrelated reason",
        )
    return ["clean_native_repeat", "actual_fault_applied", "intended_fault_detected"]


def chat_identity(root, label, generation="initial"):
    return {
        "id": digest([str(root), label]),
        "generation": digest([str(root), generation]),
        "title": "synthetic conformance fixture",
        "cwd": str(root),
    }


def completion_body(spec, chat, prefix, *, stream=False, count=None):
    return {
        "model": spec["server_config"].get("served_model_name", spec["server_config"]["model"]),
        "prompt": prefix,
        "max_tokens": count or spec["output_tokens"],
        "ignore_eos": True,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
        "return_token_ids": True,
        "stream": stream,
        "cache_salt": cache_salt(chat),
        "kv_transfer_params": {
            "qwen_chat": chat,
            "qwen_snapshot_abi": spec["binding"]["live_data_abi"],
        },
    }


def generate(
    server, spec, chat, prefix, name, *, count=None, stream=False, cancel_after=None, timeout=600
):
    result = server.client.completion(
        "/v1/completions",
        completion_body(spec, chat, prefix, stream=stream, count=count),
        evidence=server.root / name,
        allow_length=True,
        cancel_after=cancel_after,
        timeout=timeout,
    )
    if cancel_after is None:
        require(
            len(result["token_ids"]) == (count or spec["output_tokens"]),
            "native server omitted requested raw token IDs",
        )
    return result


def same_output(a, b):
    require(
        a["token_ids"] and a["token_ids"] == b["token_ids"], "native greedy continuation changed"
    )
    require(a["finish_reason"] == b["finish_reason"], "native stop boundary changed")


def observed(server, *names):
    events = read_events(server.root, server.executions)
    for name in names:
        require(
            any(e["event"] == name for e in events), "required native path did not execute: " + name
        )
    return events


def flush(server, chat):
    return request_tail_flush(chat, control_directory=server.root / "control", timeout=120)


def capture_request(server, spec, case, prefix, name):
    plan = plan_for(spec, case, tokens(spec, 2, 1701))
    require(
        plan["prefix"] == prefix, "native capture prompt differs from declared synthetic fixture"
    )
    write_private(server.root / (name + ".plan.json"), plan)
    write_private(server.root / (name + ".binding.json"), spec["binding"])
    marker = server.root / "capture-request.json"
    marker.unlink(missing_ok=True)
    write_private(
        marker,
        {
            "name": "state-" + name,
            "consumed": len(prefix),
            "input_digest": digest(prefix),
            "plan": str(server.root / (name + ".plan.json")),
            "binding": str(server.root / (name + ".binding.json")),
        },
    )


def captured_generation(server, spec, case, chat, prefix, name, *, timeout=600):
    capture_request(server, spec, case, prefix, name)
    try:
        result = generate(server, spec, chat, prefix, name, timeout=timeout)
        frame = read_frame(server.root / ("state-" + name))
        require(
            frame["pending"] == result["token_ids"][0],
            "captured pending token differs from actual published response",
        )
        return result
    finally:
        (server.root / "capture-request.json").unlink(missing_ok=True)


def corrupt_restore_recovery(server, spec, case, chat, prefix, baseline, store, root, damaged_key):
    # A hung loader previously passed this test when the client's 600-second
    # timeout was caught as if it were a deliberate backend rejection.
    try:
        result = captured_generation(
            server, spec, case, chat, prefix, "corrupt-restore", timeout=30
        )
    except BackendResponseError as error:
        require(
            error.http_status is None or 500 <= error.http_status < 600,
            "request/authorization errors do not establish corrupt-cache rejection",
        )
        observed(server, "snapshot.load.error")
        return [
            "verified_native_snapshot",
            "actual_payload_corruption",
            "native_restore_rejected_corruption",
        ]
    # Success is admissible only after detection and correct recomputation,
    # durable repair, and a second restart proving that repair is usable.
    observed(server, "snapshot.load.error", "state.captured")
    same_output(baseline, result)
    comparison = compare_frames(server.root / "state-live", server.root / "state-corrupt-restore")
    write_private(root / "corrupt-recovery-state.json", comparison)
    require(comparison["equal"], "corrupt-cache recomputation changed native logical state")
    flush(server, chat)
    repaired = store.metadata()
    require(damaged_key in repaired.get("head", []), "repair dropped the damaged head block")
    require(
        repaired.get("publication", {}).get("result") == "committed",
        "corrupt-cache replacement was not verified and published",
    )
    store.read(damaged_key, repaired["verified_block_size"])
    events = observed(server, "snapshot.publish.return")
    require(
        any(
            e["execution"] == server.executions[-1] and e["event"] == "snapshot.publish.return"
            for e in events
        ),
        "corrupt-cache recovery did not publish a replacement in this server incarnation",
    )
    write_private(root / "head-repaired.json", repaired)
    # Explicit flush, rather than shutdown's final flush, must have made it durable.
    server.restart(crash=True)
    reloaded = captured_generation(server, spec, case, chat, prefix, "repair-reload", timeout=30)
    same_output(baseline, reloaded)
    comparison = compare_frames(server.root / "state-live", server.root / "state-repair-reload")
    write_private(root / "repaired-reload-state.json", comparison)
    require(comparison["equal"], "repaired snapshot changed native logical state after restart")
    events = read_events(server.root, server.executions)
    current = [e["event"] for e in events if e["execution"] == server.executions[-1]]
    require("snapshot.load.return" in current, "repaired snapshot was not loaded after restart")
    require("snapshot.load.error" not in current, "repaired snapshot failed again after restart")
    return [
        "verified_native_snapshot",
        "actual_payload_corruption",
        "native_corruption_detected",
        "identical_recomputed_state_and_output",
        "verified_repaired_disk_head",
        "identical_durable_reload_state_and_output",
    ]


def eviction_capacity(head, configured_bytes):
    """Fit one measured durable head with transfer slack, but not two heads."""
    blocks = len(head.get("head", []))
    size = head.get("verified_block_size", 0)
    require(blocks >= 2 and size > 0, "eviction fixture has no measured complete head")
    capacity = ((3 * blocks + 1) // 2) * size
    require(capacity < configured_bytes, "configured primary cache is too small for this trial")
    return capacity


def interrupted_write_prefix(spec, case, prefix):
    # An ordinary 256-token continuation need not cross a snapshot-block
    # boundary. Change a full suffix within the same context budget so a real
    # immutable object write is required even at the maximum context length.
    count = min(8192, len(prefix))
    changed = prefix[:-count] + tokens(spec, count, case["seed"] + 9419)
    require(changed != prefix, "interrupted-write fixture did not change any token")
    return changed


def pending_tail(status, chat):
    """Return measured, unpublished state for this exact chat generation."""
    require(
        status.get("schema") == "urn:qwen-r9700:radiance-tail-residency:v1"
        and isinstance(status.get("chats"), list),
        "invalid native pending-tail observation",
    )
    rows = [
        row
        for row in status["chats"]
        if row.get("chat_id") == chat["id"] and row.get("generation") == chat["generation"]
    ]
    require(len(rows) <= 1, "duplicate native pending-tail observation")
    if not rows:
        return None
    row = rows[0]
    require(
        all(
            type(row.get(k)) is int and row[k] >= 0
            for k in ("tokens", "durable_tokens", "blocks", "bytes")
        ),
        "invalid native pending-tail counts",
    )
    if row["tokens"] > row["durable_tokens"] and row["blocks"] > 0 and row["bytes"] > 0:
        return dict(row)
    return None


def wait_for_pending_tail(server, chat, *, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        server.process.check()
        try:
            status = private_json(server.root / "tail.json")
        except FileNotFoundError:
            status = None
        if status is not None and (row := pending_tail(status, chat)) is not None:
            return row
        require(
            time.monotonic() < deadline, "shutdown fixture has no measured pending snapshot tail"
        )
        time.sleep(0.05)


def shutdown_seed_prefix(prefix):
    # An initial 8K+ response is already due for the normal 8K tail flush.
    # Save a predecessor first, so the tested response advances its durable
    # head by only 2K tokens and leaves a real RAM-only tail to flush on exit.
    require(len(prefix) > 2048, "shutdown fixture requires a durable predecessor")
    return prefix[:-2048]


def require_shutdown_flush(events, execution):
    events = [e for e in events if e["execution"] == execution]
    starts = [e for e in events if e["event"] == "snapshot.shutdown.enter"]
    ends = [e for e in events if e["event"] == "snapshot.shutdown.return"]
    require(len(starts) == len(ends) == 1, "shutdown observation is missing or duplicated")
    start, end = starts[0], ends[0]
    require(
        start["pid"] == end["pid"] and start["monotonic_ns"] < end["monotonic_ns"],
        "shutdown observation has invalid process or timing",
    )
    during = {
        e["event"]
        for e in events
        if e["pid"] == start["pid"]
        and start["monotonic_ns"] < e["monotonic_ns"] < end["monotonic_ns"]
    }
    require(
        {"snapshot.flush.return", "snapshot.publish.return"} <= during,
        "pending tail was not flushed and published inside shutdown",
    )


def lifecycle(spec, case, root):
    variant = case["variant"]
    prefix = tokens(spec, case["context"], case["seed"])
    chat = chat_identity(root, "A")
    with NativeServer(spec, root / "server", allow_gpu=True) as server:
        if variant == "shutdown_pending_tail":
            generate(server, spec, chat, shutdown_seed_prefix(prefix), "shutdown-seed")
            flush(server, chat)
            seed_head = ChatStore(server.root / "data", chat).metadata()
            require(
                seed_head.get("head"), "shutdown fixture did not establish a durable predecessor"
            )
            write_private(root / "head-shutdown-seed.json", seed_head)
        baseline = captured_generation(server, spec, case, chat, prefix, "live")
        if variant != "shutdown_pending_tail":
            flush(server, chat)
        store = ChatStore(server.root / "data", chat)
        before = store.metadata()
        write_private(root / "head-before.json", before)
        if variant == "eviction":
            original = spec["server_config"]["kv_transfer_config"]["kv_connector_extra_config"][
                "cpu_bytes_to_use"
            ]
            capacity = eviction_capacity(before, original)
            server.stop()
            server.variant["primary_cache_bytes"] = capacity
            server.start()
            warmed = captured_generation(server, spec, case, chat, prefix, "eviction-warm")
            same_output(baseline, warmed)
            comparison = compare_frames(
                server.root / "state-live", server.root / "state-eviction-warm"
            )
            write_private(root / "eviction-warm-state.json", comparison)
            require(comparison["equal"], "eviction setup changed the baseline native state")
            flush(server, chat)
            write_private(
                root / "eviction-capacity.json",
                {
                    "original_bytes": original,
                    "trial_bytes": capacity,
                    "head_blocks": len(before["head"]),
                },
            )
        if variant in {"ram", "eviction"}:
            for n in range(1 if variant == "ram" else 3):
                other = chat_identity(root, f"other-{n}")
                other_prefix = prefix if variant == "eviction" else prefix[:8192]
                generate(server, spec, other, other_prefix, f"other-{n}")
                flush(server, other)
            observed(server, "bank.activate.return")
            worker = json.loads((server.root / "fair-worker.json").read_text())
            write_private(root / "handover.json", worker)
            if variant == "ram":
                require(
                    any(r.get("chat_id") == chat["id"] for r in worker["residency"]["images"]),
                    "RAM case did not leave the tested chat in RAM",
                )
            else:
                require(
                    not any(r.get("chat_id") == chat["id"] for r in worker["residency"]["images"]),
                    "eviction case failed to evict the tested RAM bank",
                )
        elif variant in {"clean_restart", "crash_restart", "shutdown_pending_tail"}:
            if variant == "shutdown_pending_tail":
                pending = wait_for_pending_tail(server, chat)
                write_private(root / "pending-tail-before-shutdown.json", pending)
                process_root = server.root / f"process-{server.incarnation - 1}"
                server.stop()
                stopped = private_json(process_root / "shutdown.json")
                require(
                    not stopped["grace_expired"] and stopped["cleanup_error"] is None,
                    "clean shutdown exceeded its owned process grace",
                )
                events = observed(server, "snapshot.shutdown.return", "snapshot.flush.return")
                require_shutdown_flush(events, server.executions[-1])
                durable = store.metadata()
                write_private(root / "head-after-shutdown.json", durable)
                require(
                    durable.get("generation") == chat["generation"]
                    and durable.get("tokens", 0) >= pending["tokens"]
                    and durable.get("head")
                    and durable.get("publication", {}).get("result") == "committed",
                    "shutdown did not publish the measured pending tail",
                )
                for key in durable["head"]:
                    store.read(key, durable["verified_block_size"])
                server.start()
            else:
                server.restart(crash=variant == "crash_restart")
        elif variant in {"corrupt_disk", "missing_disk_block"}:
            server.stop()
            require(before.get("head"), "disk mutation case has no verified durable head")
            path = store.path(before["head"][0])
            preserved = server.root / "original-snapshot-block"
            import shutil

            shutil.copyfile(path, preserved)
            preserved.chmod(0o600)
            if variant == "missing_disk_block":
                path.unlink()
            else:
                with path.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    old = stream.read(1)
                    stream.seek(-1, os.SEEK_END)
                    stream.write(bytes([old[0] ^ 1]))
                    stream.flush()
                    os.fsync(stream.fileno())
            write_private(
                root / "disk-fault.json",
                {
                    "fault": variant,
                    "key": path.name,
                    "original_sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
                },
            )
            server.start()
            if variant == "corrupt_disk":
                return corrupt_restore_recovery(
                    server, spec, case, chat, prefix, baseline, store, root, path.name
                )
        elif variant == "interrupted_write":
            (server.root / "interrupt-write.arm").write_text("owned qualification only\n")
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    generate,
                    server,
                    spec,
                    chat,
                    interrupted_write_prefix(spec, case, prefix),
                    "pending-write",
                )
                deadline = time.monotonic() + 120
                reached = False
                while time.monotonic() < deadline:
                    if any(
                        e["event"] == "snapshot.partial_write"
                        for e in read_events(server.root, server.executions)
                    ):
                        reached = True
                        break
                    if future.done():
                        future.result()
                        # A tail can remain in RAM until explicitly flushed.
                        break
                    time.sleep(0.05)
                if not reached:
                    # Flush in another worker because the real writer is about
                    # to be held at the injected partial-write boundary.
                    with ThreadPoolExecutor(max_workers=1) as flush_pool:
                        flush_future = flush_pool.submit(flush, server, chat)
                        while time.monotonic() < deadline and not reached:
                            reached = any(
                                e["event"] == "snapshot.partial_write"
                                for e in read_events(server.root, server.executions)
                            )
                            time.sleep(0.05)
                        server.stop(crash=True)
                        with contextlib.suppress(RuntimeError, TimeoutError):
                            flush_future.result(timeout=125)
                else:
                    server.stop(crash=True)
                with contextlib.suppress(Exception):
                    future.result(timeout=10)
            require(reached, "interrupted-write fault never reached a real native snapshot write")
            (server.root / "interrupt-write.arm").unlink()
            # The previous complete head must still be readable and unchanged.
            require(
                store.metadata().get("head") == before.get("head"),
                "interrupted publication replaced the previous head",
            )
            server.start()
        elif variant == "compaction":
            successor = chat_identity(root, "A", "compacted")
            new_store = ChatStore(server.root / "data", successor)
            new_store.activate()
            short_case = {**case, "context": min(8192, case["context"])}
            short = tokens(spec, short_case["context"], case["seed"])
            changed = captured_generation(server, spec, short_case, successor, short, "compacted")
            flush(server, successor)
            require(
                not (new_store.generations / chat["generation"]).exists(),
                "compaction retained predecessor generation",
            )
            fresh = chat_identity(root, "fresh-compacted")
            fresh_result = captured_generation(
                server, spec, short_case, fresh, short, "fresh-compacted"
            )
            same_output(changed, fresh_result)
            comparison = compare_frames(
                server.root / "state-compacted", server.root / "state-fresh-compacted"
            )
            write_private(root / "compaction-state.json", comparison)
            require(comparison["equal"], "compaction successor differs from fresh prefill")
            observed(server, "snapshot.publish.return", "bank.retire.return")
            return [
                "successful_generation_publication",
                "old_generation_retired",
                "fresh_successor_state_and_output",
            ]
        elif variant == "cancellation":
            interrupted = generate(
                server, spec, chat, prefix, "cancel", stream=True, cancel_after=2
            )
            require(interrupted["cancelled"], "cancellation was not exercised")
        elif variant != "warm":
            raise UnavailableError("unknown cache lifecycle variant")
        restore_started = time.monotonic_ns()
        restored = captured_generation(server, spec, case, chat, prefix, "restored")
        same_output(baseline, restored)
        comparison = compare_frames(server.root / "state-live", server.root / "state-restored")
        write_private(root / "recovery-state.json", comparison)
        require(comparison["equal"], "live/restored native logical state differs")
        if variant in {
            "clean_restart",
            "crash_restart",
            "interrupted_write",
            "eviction",
            "shutdown_pending_tail",
        }:
            events = observed(server, "snapshot.load.return")
            require(
                any(
                    e["event"] == "snapshot.load.return" and e["monotonic_ns"] >= restore_started
                    for e in events
                ),
                "restored request did not load a snapshot; an earlier setup read is insufficient",
            )
        flush(server, chat)
        observed(server, "snapshot.publish.return", "state.captured")
        if variant == "missing_disk_block":
            require(path.exists(), "missing disk block was not repaired after successful prefill")
            repaired = store.metadata()
            require(
                repaired.get("publication", {}).get("result") == "committed",
                "repaired checkpoint was not verified and published",
            )
        return [
            "initial_native_state",
            "actual_" + variant,
            "identical_restored_state",
            "identical_greedy_suffix",
            "verified_durable_head",
            *(
                ["measured_pending_tail", "verified_head_before_restart"]
                if variant == "shutdown_pending_tail"
                else []
            ),
        ]


def natural(spec, case, root):
    prefix = tokens(spec, case["context"], case["seed"])
    chat = chat_identity(root, "natural")
    variant = case["variant"]
    settings = {
        "speculation": ({"speculation": False}, {"speculation": True}),
        "verify_head": ({"head": False}, {"head": True, "head_audit": True}),
        "graphs": ({"graphs": False}, {"graphs": True}),
        "async_experimental": ({"asynchronous": False}, {"asynchronous": True}),
        "dynamic_width": ({"dynamic_width": False}, {"dynamic_width": True}),
        "head_omission_control": (
            {"head": True, "head_audit": True},
            {"head": True, "head_audit": True, "head_fault": True},
        ),
    }[variant]
    responses, evidence = [], []
    for index, options in enumerate(settings):
        with NativeServer(spec, root / f"server-{index}", allow_gpu=True, **options) as server:
            responses.append(generate(server, spec, chat, prefix, "output"))
            events = observed(server, "runner.prepare.return", "runner.commit.return")
            if options.get("graphs"):
                observed(server, "graph.replay.return")
            if options.get("speculation"):
                require(
                    any(e["details"].get("draft_tokens", 0) > 0 for e in events),
                    "no natural speculative verification occurred",
                )
            if options.get("head_audit"):
                audits = [
                    e["details"] for e in events if e["details"].get("head_audit") == "compared"
                ]
                require(audits, "fast verify head was never compared on actual hidden states")
                if options.get("head_fault"):
                    observed(server, "head.fault_applied")
                    require(
                        all(a["omitted_winners"] > 0 and a["different_argmax"] > 0 for a in audits),
                        "omitted-winner negative control escaped the actual head checker",
                    )
                    evidence.append({"variant": server.variant, "execution": server.executions})
                    continue
                require(
                    all(
                        not a["omitted_winners"]
                        and not a["different_retained_logits"]
                        and not a["different_argmax"]
                        for a in audits
                    ),
                    "verify-head discrepancy; actual hidden states/logits preserved",
                )
            evidence.append({"variant": server.variant, "execution": server.executions})
            if options.get("dynamic_width"):
                widths = {
                    e["details"].get("draft_tokens")
                    for e in events
                    if e["event"] == "runner.prepare.return"
                    and e["details"].get("draft_tokens", 0) > 0
                }
                require(
                    any(0 < width < 7 for width in widths),
                    "dynamic verifier never exercised a shorter natural width",
                )
    write_private(root / "natural-variants.json", {"variants": evidence})
    same_output(*responses)
    return [
        "unforced_greedy_tokens",
        "same_prompt_and_seed",
        "actual_" + variant,
        "exact_output_and_stop_equality",
    ]


def operator(spec, case, root):
    if case["variant"] == "native_dispatch":
        if not spec["binding"].get("native_entrypoints"):
            raise UnavailableError("no reviewed native export binding for this binary")
        from qwen_r9700_lab.conformance_reference import reference_precision

        plan = plan_for(spec, case, tokens(spec, 10, 1701), accepted=[7, 0])
        capture = native(spec, plan, root / "dispatch")
        receipt = private_json(capture / "native-receipt.json")
        require(
            isinstance(receipt.get("native_entrypoints"), str)
            and len(receipt["native_entrypoints"]) == 64,
            "native export dispatch evidence absent",
        )
        report = private_json(capture / "dispatch/dispatch.json")
        authenticate(report)
        dtype = "fp8" if reference_precision(spec["reference_profile"])["kv_fp8"] else "bf16"
        require(
            report["sha256"] == receipt["native_entrypoints"], "native dispatch receipt changed"
        )
        for symbol in (
            f"attn_prefill_h256_gqa6_{dtype}kv",
            f"attn_decode_h256_gqa6_{dtype}kv",
            "gdn_chunk_scan_k128_v128_c64_bf16",
        ):
            require_native_dispatch(report, "r4d." + symbol)
        return [
            "exact_reviewed_native_export_inventory",
            "native_library_identity",
            "real_prefill_decode_GDN_dispatch",
        ]
    scripts = Path(__file__).resolve().parents[2] / "experiments/radiance-public"
    names = {
        "mxfp4": ("probe_mxfp4_numerics.py", "mxfp4-probe.json"),
        "gdn": ("probe_gdn_numerics.py", "gdn-probe.json"),
        "norm_rope": ("probe_norm_rope_numerics.py", "norm-rope-probe.json"),
        "attention": ("probe_r4d_attention_numerics.py", "attention-probe.json"),
        "sampling": ("probe_dflash_sampling_rng.py", "sampling-probe.json"),
    }
    script, filename = names[case["variant"]]
    data = root / "probe"
    data.mkdir(mode=0o700)
    write_private(data / "production-profile.json", spec["operator_profile"])
    manifest = {**spec["probe_manifest"], "checkpoint": spec["native_config"]["model"]}
    require(
        not manifest.get("reference_linear") and not manifest.get("gdn_repair_sha256"),
        "campaign probes the deployed repair, not an extra experimental source overlay",
    )
    write_private(data / "manifest.json", manifest)
    args = [spec["python"], str(scripts / script)]
    args += (
        ["--output", str(data / filename)]
        if case["variant"] == "sampling"
        else ["--root", str(data)]
    )
    process = OwnedProcess(
        args,
        root / "probe-process",
        env=worker_environment(spec, root),
        timeout=spec["case_timeout_seconds"],
    )
    code = process.wait()
    report = json.loads((data / filename).read_text())
    require(code == 0, "native operator runner failed; original report/log retained")
    validate_operator_report(case["variant"], report)
    return [
        "real_native_operator",
        "independent_numeric_or_statistical_oracle",
        "fixed_predeclared_tolerance",
        "nonempty_report_domain",
    ]


def validate_operator_report(variant, report):
    """Check the real probe report, including both sides of RNG fault controls."""
    rows = report.get("rows", report.get("cases"))
    require(isinstance(rows, list) and rows, "operator probe produced an empty domain")
    if variant in {"mxfp4", "attention"}:
        require(
            report.get("all_passed") is True
            and all(
                r.get("finite") is True
                and r.get("guard_regions_intact") is True
                and 0 <= r["relative_error_vs_fp32"] < 0.02
                for r in rows
            ),
            "operator numerical or memory guard oracle failed",
        )
    elif variant == "gdn":
        require(
            report.get("all_within_one_percent") is True
            and all(r["within_one_percent"] for r in rows),
            "native GDN differs from independent recurrence beyond declared diagnostic bound",
        )
    elif variant == "norm_rope":
        require(
            report.get("all_finite") is True
            and report["maximum_relative_error"] <= 0.02
            and all(r.get("unrotated_channels_exact", True) for r in rows),
            "norm/RoPE oracle failed",
        )
    else:
        require(
            report.get("greedy_exact") is True and report.get("native_target_rows"),
            "sampler probe omitted native/greedy rows",
        )
        require(
            any(r["independent_proposal_noise"] for r in rows),
            "independent RNG distribution observations were omitted",
        )
        require(
            all(r["max_error"] < 0.004 for r in rows if r["independent_proposal_noise"]),
            "independent RNG distribution regression",
        )
        require(
            any(not r["independent_proposal_noise"] for r in rows),
            "sampling negative control was omitted",
        )
        require(
            all(r["max_error"] > 0.01 for r in rows if not r["independent_proposal_noise"]),
            "shared-RNG negative control was not detected",
        )
        require(
            all(r["max_error"] < 0.004 for r in report["native_target_rows"]),
            "native target distribution regression",
        )


def require_native_dispatch(report, site):
    authenticate(report)
    require(
        any(
            row.get("site") == site and row.get("completed") is True and "exception_type" not in row
            for row in report.get("calls", [])
        ),
        "required native export was not observed: " + site,
    )


TOOL = {
    "type": "function",
    "function": {
        "name": "record",
        "description": "Record the supplied synthetic fixture; no external effects.",
        "parameters": {
            "type": "object",
            "properties": {"number": {"type": "integer"}},
            "required": ["number"],
            "additionalProperties": False,
        },
    },
}


def tool_body(spec, chat, messages=None, *, thinking=False, stream=True):
    return {
        "model": spec["server_config"].get("served_model_name", spec["server_config"]["model"]),
        "messages": messages
        or [
            {
                "role": "user",
                "content": "Call record with number 37. After its result, reply exactly RECORDED.",
            }
        ],
        "tools": [TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
        "max_tokens": spec["output_tokens"] * (8 if thinking else 1),
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": stream,
        "cache_salt": cache_salt(chat),
        "kv_transfer_params": {
            "qwen_chat": chat,
            "qwen_snapshot_abi": spec["binding"]["live_data_abi"],
        },
    }


def record_tool(result):
    require(
        result["finish_reason"] == "tool_calls" and len(result["tools"]) == 1,
        "synthetic model task did not emit the requested tool boundary",
    )
    tool = result["tools"][0]
    require(
        tool["name"] == "record" and tool["parsed_arguments"] == {"number": 37},
        "synthetic tool call has incorrect arguments",
    )
    return tool


def tool_continuation(body, result):
    tool = record_tool(result)
    assistant = {
        "role": "assistant",
        "content": result["content"] or None,
        "tool_calls": [
            {
                "id": tool["id"],
                "type": "function",
                "function": {"name": tool["name"], "arguments": tool["arguments"]},
            }
        ],
    }
    if result["reasoning"]:
        assistant["reasoning_content"] = result["reasoning"]
    return {
        **body,
        "messages": body["messages"]
        + [
            assistant,
            {"role": "tool", "tool_call_id": tool["id"], "content": "Recorded 37 successfully."},
        ],
    }


def fit_tool_context(server, spec, case, body):
    """Measure the real chat template, reserving room for the tool/result turn."""
    from tokenizers import Tokenizer

    target = min(case["context"], spec["max_context"] - 3 * spec["output_tokens"] - 512)
    messages = [{"role": "system", "content": ""}, *body["messages"]]
    request = {
        "model": body["model"],
        "messages": messages,
        "tools": body["tools"],
        "chat_template_kwargs": body["chat_template_kwargs"],
        "add_special_tokens": False,
        "add_generation_prompt": True,
    }
    base = server.client.json("/tokenize", request)
    reserve = len(base["tokens"])
    require(target > reserve + 8, "priority context cannot hold the real tool template")
    tokenizer = Tokenizer.from_file(str(Path(spec["native_config"]["model"]) / "tokenizer.json"))
    padding = tokens(spec, target - reserve + 8, case["seed"])
    count = target - reserve
    for _ in range(8):
        messages[0]["content"] = tokenizer.decode(padding[:count], skip_special_tokens=False)
        measured = len(server.client.json("/tokenize", request)["tokens"])
        if target - 4 <= measured <= target:
            write_private(
                server.root / "priority-context.json",
                {
                    "A_prompt_tokens": measured,
                    "B_prompt_tokens": case["context"],
                    "A_target": target,
                },
            )
            return {**body, "messages": messages}
        count = max(1, min(len(padding), count + target - measured))
    raise DiagnosticError("could not fit synthetic priority prompt to the declared window")


def protocol(spec, case, root):
    variant = case["variant"]
    if variant == "parser_fragments":
        from qwen_r9700_lab.conformance_parser import qualify_parser

        qualify_parser(
            spec["native_config"]["model"],
            spec["observer_sources"]["vllm.parser.qwen3"],
            root / "parser.json",
        )
        qualify_parser(
            spec["native_config"]["model"],
            spec["observer_sources"]["vllm.parser.qwen3"],
            root / "parser-deployed.json",
            parser_config={
                "tool_parser_name": spec["server_config"]["tool_call_parser"],
                "reasoning_parser_name": spec["server_config"]["reasoning_parser"],
                "enable_auto_tools": spec["server_config"]["enable_auto_tool_choice"],
            },
        )
        return [
            "actual_Qwen3Parser",
            "registered_serving_parser",
            "single_token_and_grouped_chunks",
            "thinking_tool_boundary",
            "incomplete_and_colon_are_not_complete_tools",
        ]
    if variant == "pi_provider_fragments":
        if not spec.get("pi_runtime") or not spec.get("pi_runtime_sha256"):
            raise UnavailableError(
                "the installed Pi provider artifact has not been bound in this campaign"
            )
        write_private(
            root / "pi.json", {"provider": spec["pi_runtime"], "sha256": spec["pi_runtime_sha256"]}
        )
        driver = Path(__file__).resolve().parents[2] / "tests/conformance_pi_driver.mjs"
        process = OwnedProcess(
            ["node", str(driver), str(root / "pi.json"), str(root / "pi-result.json")],
            root / "pi-process",
            env=dict(os.environ),
            timeout=120,
        )
        code = process.wait()
        result = private_json(root / "pi-result.json")
        require(
            code == 0 and result["passed"] is True and len(result["rows"]) == 5,
            "real Pi provider lost or duplicated a fragmented tool event",
        )
        return [
            "installed_Pi_provider",
            "UTF8_and_SSE_fragments",
            "exactly_one_tool_event",
            "no_tool_execution",
        ]
    release_variants = {
        "minimal_release": ((True, False), True, True, True),
        "minimal_release_full_head": ((True, False), False, True, True),
        "repeat_release_full_head": ((False, False), False, True, True),
        "minimal_release_target_only": ((True, False), False, False, True),
        "repeat_release_target_only": ((False, False), False, False, True),
        "minimal_release_target_only_eager": ((True, False), False, False, False),
        "repeat_release_target_only_eager": ((False, False), False, False, False),
    }
    if variant in release_variants:
        from qwen_r9700_lab.conformance_protocol import compare_protocol_pair

        observations, head, speculation, graphs = release_variants[variant]
        prefix = tokens(spec, case["context"], case["seed"])
        results = []
        for index, observe in enumerate(observations):
            with NativeServer(
                spec,
                root / f"server-{index}",
                allow_gpu=True,
                observe=observe,
                head=head,
                speculation=speculation,
                graphs=graphs,
            ) as server:
                results.append(
                    generate(server, spec, chat_identity(root, "minimal"), prefix, "response")
                )
                if observe:
                    names = ["runner.commit.return"]
                    if graphs:
                        names.append("graph.replay.return")
                    observed(server, *names)
                else:
                    require(
                        not list(server.root.glob("events-*.jsonl")),
                        "minimal replay unexpectedly loaded diagnostic observers",
                    )
        compare_protocol_pair(
            *results, root / "release-comparison.json", kind="fresh_release_instances"
        )
        return [
            "instrumented_vs_unobserved_release"
            if any(observations)
            else "two_unobserved_fresh_release_instances",
            "fast_head_enabled" if head else "target_verify_head_disabled",
            "speculative_decode" if speculation else "target_only",
            "graph_replay_requested" if graphs else "eager_execution_requested",
            "complete_raw_prompt_and_output_ID_comparison",
            "same_greedy_output",
            "no_tensor_or_dispatch_hooks_in_minimal",
        ]
    with NativeServer(spec, root / "server", allow_gpu=True) as server:
        chat = chat_identity(root, "protocol")
        body = tool_body(spec, chat, thinking=variant == "long_thinking")
        body["return_token_ids"] = True
        if variant == "long_thinking":
            # The preserved 2,048-token attempt ended at its explicit budget.
            # Increase this synthetic fixture's budget without changing its
            # prompt, sampler, stop rules or treatment of an incomplete answer.
            body["max_tokens"] = spec["output_tokens"] * 32
            body["messages"][0]["content"] = (
                "Reason carefully through the sum of squares of integers 1 through 100, "
                "checking it with two independent derivations. Then call record with number 37. "
                "After its result reply exactly RECORDED."
            )
        result = server.client.completion(
            "/v1/chat/completions", body, evidence=server.root / "tool"
        )
        record_tool(result)
        if variant == "long_thinking":
            require(
                len(result["reasoning"]) >= 100, "long-thinking path was not exercised by the model"
            )
        follow = tool_continuation(body, result)
        answer = server.client.completion(
            "/v1/chat/completions", follow, evidence=server.root / "answer"
        )
        require(
            answer["finish_reason"] == "stop" and answer["content"].strip() == "RECORDED",
            "synthetic tool continuation did not finish correctly",
        )
        nonstream = server.client.completion(
            "/v1/chat/completions", {**body, "stream": False}, evidence=server.root / "nonstream"
        )
        record_tool(nonstream)
        from qwen_r9700_lab.conformance_protocol import compare_protocol_pair

        compare_protocol_pair(result, nonstream, server.root / "stream-nonstream-comparison.json")
        observed(server, "parser.tool_stream.return", "parser.tool_nonstream.return")
        if variant == "long_thinking":
            observed(server, "parser.reasoning_stream.return", "parser.reasoning_nonstream.return")
        return [
            "actual_native_tool_parser",
            "tool_result_continuation",
            "stream_nonstream_consistency",
            "complete_raw_prompt_and_output_ID_comparison",
            "honest_finish_reason",
        ]


def priority(spec, case, root):
    from qwen_r9700_lab.conformance_priority import PriorityLease

    variant = case["variant"]
    levels = {
        "equal0": (0, 0),
        "equal1": (1, 1),
        "equal2": (2, 2),
        "priority1_owner": (1, 0),
        "priority1_waiter": (0, 1),
        "priority2_waiter": (0, 2),
    }[variant]
    a, b = chat_identity(root, "priority-A"), chat_identity(root, "priority-B")
    with NativeServer(spec, root / "server", allow_gpu=True) as server:
        body = fit_tool_context(server, spec, case, tool_body(spec, a))
        prefix = tokens(spec, case["context"], case["seed"])
        leases = [
            PriorityLease(server.client, chat, spec["binding"]["live_data_abi"], root / name)
            for chat, name in ((a, "priority-control-A"), (b, "priority-control-B"))
        ]
        la, lb = leases
        try:
            la.start(levels[0]).result()
            lb.start(0).result()
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(
                    server.client.completion,
                    "/v1/chat/completions",
                    body,
                    evidence=server.root / "A",
                )
                deadline = time.monotonic() + spec["case_timeout_seconds"]
                # Wait for actual A generation before B requests ownership.
                while time.monotonic() < deadline:
                    phases = server.root / "fair-phases.json"
                    if phases.exists():
                        status = json.loads(phases.read_text())
                        # Actual phase records are inspected below as well; a
                        # response completing before overlap is a coverage gap.
                        if '"generate"' in json.dumps(status) and a["id"] in json.dumps(status):
                            break
                    require(
                        not fa.done(), "A finished before overlapping generation could be tested"
                    )
                    time.sleep(0.02)
                else:
                    raise TimeoutError("A never reached observed generation")

                def run_b():
                    # Independent Pi processes have independent control queues.
                    # B's admission must not delay consuming A's tool response.
                    try:
                        if levels[1]:
                            lb.change(levels[1]).result()
                        return generate(server, spec, b, prefix, "B")
                    finally:
                        lb.release().result()

                fb = pool.submit(run_b)
                result_a = fa.result(timeout=spec["case_timeout_seconds"])
                a_finished = time.monotonic_ns()
                record_tool(result_a)
                # Real client-side tool duration: no model output is forced.
                if variant == "priority1_owner":
                    until = time.monotonic() + 3
                    while time.monotonic() < until:
                        require(not fb.done(), "lower-priority chat ran during the retained answer")
                        time.sleep(0.02)
                elif variant.startswith("equal"):
                    time.sleep(0.25)  # explicitly test the short-tool grace
                follow = tool_continuation(body, result_a)
                a_continuation_submitted = time.monotonic_ns()
                continuation = server.client.completion(
                    "/v1/chat/completions", follow, evidence=server.root / "A-continuation"
                )
                a_answer_finished = time.monotonic_ns()
                require(continuation["finish_reason"] == "stop", "priority answer failed to finish")
                la.release().result()
                result_b = fb.result(timeout=spec["case_timeout_seconds"])
            events = observed(server, "scheduler.step.return", "bank.activate.return")
            b_steps = [
                e
                for e in events
                if e["event"] == "scheduler.step.return"
                and e["details"].get("active", "").startswith(b["id"] + ":")
                and e["details"].get("scheduled")
            ]
            require(b_steps, "waiting chat never received actual scheduled tokens")
            first_b = b_steps[0]["monotonic_ns"]
            a_finishes = [
                e["monotonic_ns"]
                for e in events
                if e["event"] == "response.finish.return" and e["details"].get("chat_id") == a["id"]
            ]
            require(len(a_finishes) == 2, "native A response completion evidence is incomplete")
            write_private(
                root / "priority-order.json",
                {
                    "first_B_scheduled_ns": first_b,
                    "A_tool_finished_ns": a_finished,
                    "A_continuation_submitted_ns": a_continuation_submitted,
                    "A_answer_finished_ns": a_answer_finished,
                    "A_native_response_finished_ns": a_finishes,
                    "control_failures": [r for lease in leases for r in lease.failures],
                },
            )
            # A short intended sleep is insufficient: verify that the actual
            # native tool boundary to continuation submission was also short.
            if variant.startswith("equal"):
                require(
                    a_continuation_submitted - a_finishes[0] < 2_000_000_000,
                    "short-tool fixture exceeded the two-second grace; inspect timing receipts",
                )
            if variant in {"priority1_waiter", "priority2_waiter"}:
                activated = [r for r in lb.confirmations if r["purpose"] == "change"]
                require(
                    len(activated) == 1 and activated[0]["finished_ns"] < a_finishes[0],
                    "B priority was not confirmed during A generation; overlap was not exercised",
                )
            if variant == "priority2_waiter":
                require(
                    first_b < a_finishes[0],
                    "priority two did not preempt ongoing lower-priority generation",
                )
            elif variant == "priority1_owner":
                require(
                    first_b >= a_answer_finished, "priority one owner lost GPU before answer end"
                )
            else:
                require(
                    first_b >= a_finishes[0],
                    "ordinary/priority-one waiter interrupted ongoing generation",
                )
            if variant.startswith("equal"):
                require(first_b >= a_finishes[-1], "equal priorities lost the short-tool grace")
            baseline = generate(server, spec, chat_identity(root, "B-fresh"), prefix, "B-fresh")
            same_output(result_b, baseline)
            require(
                not any(lease.failures for lease in leases),
                "priority control had unconfirmed updates; inspect timed HTTP receipts",
            )
        finally:
            for lease in leases:
                lease.close()
        return [
            "default_priority_without_control_IPC" if levels == (0, 0) else "real_priority_IPC",
            "timed_independent_control_queues",
            "overlapping_requests",
            "tool_boundary_and_answer_ownership",
            "no_resumed_output_corruption",
        ]


SCENARIOS = {
    "native_fault": native_fault,
    "forced_reference": forced_reference,
    "forced_d7": forced_d7,
    "rejected_suffix": rejected_suffix,
    "operator": operator,
    "natural": natural,
    "lifecycle": lifecycle,
    "priority": priority,
    "protocol": protocol,
}
