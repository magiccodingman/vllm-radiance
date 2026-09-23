"""Offline conformance CLI; GPU use is a separate, explicitly armed command."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from qwen_r9700_lab.conformance_artifacts import reference_runtime_identity
from qwen_r9700_lab.conformance_boundaries import compare_boundaries
from qwen_r9700_lab.conformance_control import audit_control
from qwen_r9700_lab.conformance_instrumentation import compare_calls
from qwen_r9700_lab.conformance_invariants import interval_certificate
from qwen_r9700_lab.conformance_lifecycle import compare_recovery
from qwen_r9700_lab.conformance_obligations import proof_obligations
from qwen_r9700_lab.conformance_proofs import run_obligations
from qwen_r9700_lab.conformance_reference import (
    REFERENCE_PROFILES,
    reference_contract,
    reference_semantics,
)
from qwen_r9700_lab.conformance_replay import (
    PLAN_SCHEMA,
    reference_code_identity,
    replay_operator,
    run_reference,
    validate_plan,
)
from qwen_r9700_lab.conformance_session import (
    CheckedSession,
    SessionMismatchError,
    compare_campaign,
)
from qwen_r9700_lab.conformance_state import compare_frames
from qwen_r9700_lab.diagnostic_contract import (
    ARTIFACT_GROUPS,
    DiagnosticError,
    authenticate,
    digest,
    private_json,
    seal,
    semantic_identity,
    write_private,
)

HELP = """NAME
  qwen-conformance - exact state comparison and independent inference replay

SYNOPSIS
  qwen-conformance inventory
  qwen-conformance proof-obligations [--profile NAME] [--output FILE] [--require-proved]
  qwen-conformance make-plan --spec FILE --output FILE
  qwen-conformance reference --plan FILE --output DIRECTORY
  qwen-conformance native --plan FILE --config FILE --binding FILE --output DIRECTORY --allow-gpu
  qwen-conformance compare --reference DIRECTORY --candidate DIRECTORY --output DIRECTORY
  qwen-conformance boundaries --reference DIRECTORY --candidate DIRECTORY --output DIRECTORY
  qwen-conformance calls --reference DIRECTORY --candidate DIRECTORY --output FILE
  qwen-conformance recovery --plan FILE --reference DIRECTORY --live DIRECTORY
    --restored DIRECTORY --output DIRECTORY
  qwen-conformance control --trace FILE --output FILE
  qwen-conformance certificate --bounds FILE --output FILE
  qwen-conformance operator --capsule DIRECTORY --output DIRECTORY
  qwen-conformance prove --output DIRECTORY
  qwen-conformance suite-plan --spec FILE --output FILE
  qwen-conformance suite-run --campaign FILE --output DIRECTORY --allow-gpu
    [--case ID] [--keep-going] [--resume] [--through pilot|focused|extended]
    [--budget-seconds N] [--retry-failed] [--min-free-gib N]
  qwen-conformance suite-pause --output DIRECTORY
  qwen-conformance suite-status --campaign FILE --results DIRECTORY [--results DIRECTORY]
    [--require-passing]
  qwen-conformance gate --authority DIRECTORY --transition FILE [--create]

DESCRIPTION
  Compares actual logical-state bytes, including independent initial prefill,
  forced-token transitions and original quantized checkpoint operators. Finds
  the first observed discrepancy without decoding or displaying private tokens.
  Finite comparisons establish evidence for those observations, not a universal
  proof. Missing observations and unsupported native layouts fail closed.

OPTIONS
  --allow-gpu       Required for native and suite-run. Planning, comparison and
                    status commands use the CPU and never import GPU libraries.
  --plan FILE       Private, sealed execution plan from make-plan.
  --spec FILE       Private JSON: checkpoint path, checkpoint_files hashes,
                    kv_scales, prefix and forced_tokens; optional accepted_widths
                    reference_profile (default weight-only-bf16), and an explicit
                    observation_positions subset for a bounded tensor capture. Explicit
                    radiance-fp8 keeps the historical additional FP8 quantizers.
  --config FILE     Private native LLM kwargs; one eager TP1 worker, no connector.
  --binding FILE    Reviewed source binding for precisely one Radiance version.
  --output PATH    Private evidence destination; suite-run --resume continues it.
  --create         Create a new checked authority; otherwise resume its revision.
  --trace FILE     Complete sealed control event sequence, including source identity.
  --bounds FILE    Private full-vocabulary lower/upper rational bounds, winner,
                    and bound_origin. Bound soundness remains an explicit assumption.
  --profile NAME   Reference precision profile for the proof-obligations report.
  --require-proved Return 1 while any whole-backend obligation remains unproved.
  --campaign FILE  Source-bound finite GPU campaign from suite-plan.
  --case ID        Select one case (repeatable); unselected cases remain NOT_RUN.
  --keep-going     Retain the first failure and continue independent cases.
  --resume         Continue the same checkpoint; passed cases are not rerun.
  --through STAGE  Run in priority order through pilot, focused or extended.
  --budget-seconds N Stop between cases after this invocation's time budget.
  --retry-failed   Explicitly retry a reviewed failure; retain its earlier result.
  --min-free-gib N Keep this free-space reserve plus estimated capture space (20).
  --results DIR    Existing campaign evidence (repeatable); retains all attempts.
  --require-passing Return 1 for missing, failed, unsupported or unrun cases.

OPERATION
  make-plan -> independent reference/native captures -> compare/boundaries.
  operator replays a captured numerical operation against the CPU reference.
  prove checks actual small helper expressions, retaining SMT obligations.
  proof-obligations lists the mathematical contract, component obligations,
  implementation/test hashes and remaining gaps. It never upgrades tests to proofs.
  recovery compares reference, live and restored logical state at the same prefix.
  control checks completion, state versions, publication, ownership and snapshots.
  calls compares captured module/operator inputs, mutations and outputs in causal order.
  certificate checks the conditional argmax interval inequality for every token.
  gate compares complete tentative state and output before an atomic revision
  can be published. Tool events require separate checked protocol coverage.
  Native replay is separate from the live Pi backend. It changes no live port,
  snapshot, service or conversation. It needs an available GPU and is slow.
  Full boundary capture includes every materialized prefill/accepted token and
  can generate very large private evidence. Native BF16/FP8 extraction paths
  exist but still require isolated GPU qualification before relying on them.
  suite-plan wires fault injection, operator probes, forced and natural decode,
  cache lifecycles, priority, graphs, parser and Pi-provider qualification.
  suite-run launches only owned processes, loopback ports and private caches.
  The complete inventory remains visible even when running a selected subset.

EXAMPLES
  qwen-conformance inventory
  qwen-conformance prove --output /tmp/private-helper-proof
  qwen-conformance compare --reference /tmp/ref --candidate /tmp/candidate --output /tmp/diff

FILES
  plan.json, frame.json and private tensor blobs: replay inputs and evidence.
  schedule.json, boundaries.json: complete causal observation inventories.
  authority.sqlite3: validated output revisions and retained state copies.

PATHS
  configs/profiles/radiance-conformance-v320260914.json: native source binding.
  docs/backend-conformance.md: formats, admitted domain and GPU qualification plan.

SECURITY NOTES
  Evidence contains raw tokens and model state. Keep it on the authorized
  machine, outside git, mode 0700 for directories and 0600 for files.
  Native logs stay private. This is diagnostic process separation, not a
  sandbox against malicious GPU kernels or another process with the same UID.

EXIT STATUS
  0  Requested finite comparison/check completed successfully.
  1  A mismatch, failed helper obligation or undischarged --require-proved claim.
  2  Malformed evidence, missing coverage, unsupported path or worker failure.

AUTHORS
  Terrydaktal and contributors.
"""


def inventory():
    return {
        "schema": "urn:qwen:conformance-readiness:v1",
        "components": {
            "independent_mxfp4_fp8_cpu_operators": "TESTED",
            "tiny_hybrid_model_prefill_decode_restore": "TESTED",
            "full_state_byte_comparator_and_fault_controls": "TESTED",
            "durable_publication_gate": "TESTED",
            "mapped_runtime_file_identity": "TESTED",
            "d7_pending_and_rejected_suffix_helpers": "TESTED",
            "compressed_store_ram_eviction_restart_recovery": "TESTED",
            "three_way_recovery_corruption_localization": "TESTED",
            "control_trace_faults_and_copy_on_write": "TESTED",
            "source_bound_semantic_calls_and_cleanup": "TESTED",
            "finite_native_campaign_wiring_and_missing_case_gate": "TESTED on CPU",
            "owned_process_tree_deadlines_and_cleanup": "TESTED on CPU",
            "native_lifecycle_priority_graph_head_and_protocol_campaign": "NOT_RUN on GPU",
            "conditional_full_vocabulary_interval_checker": "TESTED; bound soundness ASSUMED",
            "small_smt_helpers": "run prove for implementation-bound evidence",
            "native_radiance_state_extractor": "UNPROVED",
            "native_forced_m1_and_d7_replay": "UNPROVED",
            "native_operator_and_boundary_equivalence": "UNPROVED",
            "native_snapshot_restore_and_concurrency": "UNPROVED",
            "sampled_distribution_and_tool_parser": "UNPROVED",
            "compiled_graphs_and_async_races": "UNPROVED",
            "compiler_and_hardware": "ASSUMED",
            "actual_dispatched_device_binary_identity": "UNPROVED",
            "whole_backend_formal_equivalence": "UNPROVED",
        },
        "gpu_use_by_default": False,
        "production_hooks_installed": False,
        "reused_infrastructure": [
            "diagnostic_contract",
            "conformance_gate",
            "conformance_proofs",
            "radiance layer/GDN capsules",
            "verify-head observer",
            "MXFP4/RMSNorm/R4D numerical probes",
            "logical cache ownership validator",
        ],
        "references": {name: reference_contract(name) for name in REFERENCE_PROFILES},
        "default_new_plan_profile": "weight-only-bf16",
        "whole_stack_obligations": proof_obligations(),
    }


def make_plan(spec):
    from qwen_r9700_lab.conformance_model import Checkpoint

    allowed = {
        "checkpoint",
        "checkpoint_files",
        "kv_scales",
        "prefix",
        "forced_tokens",
        "accepted_widths",
        "reference_profile",
        "observation_positions",
        "reference_linear",
    }
    if set(spec) - allowed or allowed - {
        "accepted_widths",
        "reference_profile",
        "observation_positions",
        "reference_linear",
    } - set(spec):
        raise DiagnosticError("replay specification has missing or unsupported fields")
    # Hash the actual checkpoint before naming it in the reference contract.
    weights = Checkpoint(Path(spec["checkpoint"]), spec["checkpoint_files"])
    try:
        config = weights.config.get("text_config", weights.config)
        profile = spec.get("reference_profile", "weight-only-bf16")
        arithmetic = reference_contract(profile)
        semantics = reference_semantics(
            spec["checkpoint_files"], config, spec["kv_scales"], profile
        )
        code = reference_code_identity()
        runtime = reference_runtime_identity()
        execution = seal(
            {
                "schema": "urn:qwen:conformance-reference-execution:v1",
                "reference_code": code,
                "reference_runtime": runtime,
                "reference_linear": spec.get("reference_linear"),
                "semantics": semantic_identity(semantics),
                "python": sys.version,
                "unavailable": {
                    group: "native run must record this artifact group"
                    for group in sorted(ARTIFACT_GROUPS)
                    if group not in {"reference", "model"}
                },
                "complete_attestation": False,
            }
        )
        plan = validate_plan(
            seal(
                {
                    "schema": PLAN_SCHEMA,
                    **spec,
                    "contract": semantic_identity(semantics),
                    "execution": execution["sha256"],
                    "adapter": digest(code),
                    "reference_arithmetic": arithmetic,
                    "reference_profile": profile,
                    "reference_semantics": semantics,
                    "reference_runtime": runtime["sha256"],
                }
            )
        )
        return plan, seal(semantics), execution
    finally:
        weights.close()


@contextlib.contextmanager
def reject_dead_native_rpcs(client):
    """Fail pending diagnostic RPCs when the pinned sync transport loses its worker.

    Its output thread forwards engine death to generation consumers but leaves
    utility Futures pending. This observer changes no live engine work and adds
    no inference deadline: it only wakes waiters after declared engine death.
    """
    from concurrent.futures import InvalidStateError

    resources, pending = client.resources, client.utility_results
    stopped = threading.Event()

    def watch():
        while not stopped.wait(0.1):
            if resources.engine_dead:
                for future in list(pending.values()):
                    with contextlib.suppress(InvalidStateError):
                        future.set_exception(DiagnosticError("native worker exited during RPC"))

    thread = threading.Thread(target=watch, name="conformance-worker-death", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def native_worker(plan_path, config_path, binding_path, root):
    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("GPU worker is not armed")
    plan = validate_plan(private_json(plan_path))
    kwargs = private_json(config_path)
    if (
        kwargs.get("model") != plan["checkpoint"]
        or kwargs.get("enforce_eager") is not True
        or kwargs.get("max_num_seqs") != 1
    ):
        raise DiagnosticError("native config must match the plan and request one eager sequence")
    if kwargs.get("kv_transfer_config") is not None or kwargs.get("async_scheduling", False):
        raise DiagnosticError(
            "native replay cannot use shared snapshots or asynchronous scheduling"
        )
    if kwargs.get("tensor_parallel_size", 1) != 1 or kwargs.get("pipeline_parallel_size", 1) != 1:
        raise DiagnosticError("native replay admits one GPU only")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise DiagnosticError("the source-bound adapter requires the V2 model runner")
    if os.environ.get("RADIANCE_VERIFY_HEAD", "0") != "0":
        raise DiagnosticError(
            "use full-vocabulary target logits for the initial conformance replay"
        )
    import importlib.util

    from qwen_r9700_lab.conformance_model import Checkpoint
    from qwen_r9700_lab.conformance_radiance import verify_sources

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise DiagnosticError("the pinned vLLM package is not installed")
    verify_sources(Path(spec.origin).parent.parent, private_json(binding_path))
    checkpoint = Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"])
    checkpoint.close()
    # Everything above runs before importing a GPU library or constructing LLM.
    from vllm import LLM, SamplingParams

    extension = "qwen_r9700_lab.conformance_radiance.ConformanceWorkerExtension"
    if kwargs.get("worker_extension_cls") not in (None, "", extension):
        raise DiagnosticError("native replay requires the qualified worker extension")
    kwargs["worker_extension_cls"] = extension
    kwargs["disable_log_stats"] = True
    llm = LLM(**kwargs)
    engine = llm.llm_engine
    params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        ignore_eos=True,
        # The engine process can run ahead of the output consumer. Bound it at
        # the final forced token; a client-side abort arrives too late.
        max_tokens=len(plan["forced_tokens"]),
        detokenize=False,
    )
    req = "qwen-isolated-conformance"
    try:
        with reject_dead_native_rpcs(engine.engine_core):
            llm.collective_rpc(
                "qwen_conformance_install",
                args=(str(plan_path), str(root / "capture"), str(binding_path)),
            )
            engine.add_request(req, {"prompt_token_ids": plan["prefix"]}, params)
            reached = False
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.request_id != req:
                        raise DiagnosticError("unexpected request in isolated replay")
                    if output.outputs and len(output.outputs[0].token_ids) >= len(
                        plan["forced_tokens"]
                    ):
                        reached = True
                if reached:
                    break
            if not reached:
                raise DiagnosticError("native backend ended before the forced replay completed")
            receipt = llm.collective_rpc("qwen_conformance_finish")
            write_private(
                root / "receipt.json", {"workers": receipt, "native_equivalence": "UNPROVED"}
            )
    except Exception as error:
        failure_path = root / "capture/worker-error.json"
        if failure_path.exists():
            failure = private_json(failure_path)
            authenticate(failure)
            if (
                failure.get("schema") != "urn:qwen:native-worker-failure:v1"
                or failure.get("plan") != plan["sha256"]
                or failure.get("binding") != private_json(binding_path)["sha256"]
            ):
                raise DiagnosticError("native failure receipt belongs to another replay") from error
            if failure.get("type") == "DiagnosticError":
                raise DiagnosticError(failure["message"]) from error
        raise
    finally:
        engine.abort_request([req])


def run_native(plan, config, binding, root, *, allow_gpu=False):
    if not allow_gpu:
        raise DiagnosticError(
            "GPU use was not authorized; pass --allow-gpu only when the GPU is available"
        )
    validate_plan(private_json(plan))
    root.mkdir(mode=0o700)
    # The public source binding contains no private data; copy it into the private
    # worker input directory so all worker inputs use one strict file policy.
    bound = json.loads(binding.read_text())
    authenticate(bound)
    write_private(root / "binding.json", bound)
    for path, name in ((plan, "plan.json"), (config, "native-config.json")):
        write_private(root / name, private_json(path))
    args = [
        sys.executable,
        "-m",
        "qwen_r9700_lab.conformance_cli",
        "_native-worker",
        "--output",
        str(root.resolve()),
    ]
    fd = os.open(root / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as log:
        env = {**os.environ, "QWEN_CONFORMANCE_GPU": "1"}
        result = subprocess.run(args, stdout=log, stderr=log, env=env, check=False)
    if result.returncode:
        raise DiagnosticError(
            "native replay failed; evidence is retained in the private worker log"
        )
    return {"captured": True, "native_equivalence": "UNPROVED"}


def gate(authority: Path, transition: dict, *, create=False):
    required = {
        "contract",
        "coverage",
        "reference",
        "candidate",
        "base_revision",
        "reference_tokens",
        "candidate_tokens",
        "reference_stop",
        "candidate_stop",
    }
    if set(transition) != required:
        raise DiagnosticError("incomplete publication transition")
    session = CheckedSession(
        authority,
        contract=transition["contract"],
        required_components=transition["coverage"],
        create=create,
    )
    try:
        receipt = session.commit(
            Path(transition["reference"]),
            Path(transition["candidate"]),
            base_revision=transition["base_revision"],
            reference_tokens=tuple(transition["reference_tokens"]),
            candidate_tokens=tuple(transition["candidate_tokens"]),
            reference_stop=transition["reference_stop"],
            candidate_stop=transition["candidate_stop"],
        )
        # Raw tokens are available to an authorized consumer by revision, not
        # printed to terminal logs by this diagnostic command.
        return {
            "revision": receipt["revision"],
            "output_count": len(receipt["tokens"]),
            "receipt": receipt["receipt"],
        }
    finally:
        session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("inventory")
    p = subs.add_parser("proof-obligations")
    p.add_argument("--profile", choices=tuple(REFERENCE_PROFILES), default="weight-only-bf16")
    p.add_argument("--output", type=Path)
    p.add_argument("--require-proved", action="store_true")
    p = subs.add_parser("make-plan")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("suite-plan")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("suite-run")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-gpu", action="store_true")
    p.add_argument("--case", action="append")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--through", choices=("pilot", "focused", "extended"), default="extended")
    p.add_argument("--budget-seconds", type=int)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--min-free-gib", type=int, default=20)
    p = subs.add_parser("suite-pause")
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("suite-status")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--results", type=Path, action="append", required=True)
    p.add_argument("--require-passing", action="store_true")
    for command in ("reference", "native"):
        p = subs.add_parser(command)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if command == "native":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--binding", type=Path, required=True)
            p.add_argument("--allow-gpu", action="store_true")
    for command in ("compare", "boundaries", "frame", "calls"):
        p = subs.add_parser(command)
        p.add_argument("--reference", type=Path, required=True)
        p.add_argument("--candidate", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("recovery")
    for option in ("plan", "reference", "live", "restored", "output"):
        p.add_argument("--" + option, type=Path, required=True)
    for command, option in (("control", "trace"), ("certificate", "bounds")):
        p = subs.add_parser(command)
        p.add_argument("--" + option, type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    for command in ("operator", "prove", "_native-worker"):
        p = subs.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command == "operator":
            p.add_argument("--capsule", type=Path, required=True)
    p = subs.add_parser("gate")
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--transition", type=Path, required=True)
    p.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory()
        elif args.command == "proof-obligations":
            result = proof_obligations(args.profile)
            if args.output is not None:
                write_private(args.output, result)
            if args.require_proved:
                print(json.dumps(result, allow_nan=False))
                return 1 if result["undischarged"] else 0
        elif args.command == "make-plan":
            plan, semantics, execution = make_plan(private_json(args.spec))
            write_private(args.output.with_suffix(".semantics.json"), semantics)
            write_private(args.output.with_suffix(".execution.json"), execution)
            write_private(args.output, plan)
            result = {"plan": plan["sha256"], "contract": plan["contract"]}
        elif args.command == "suite-pause":
            from qwen_r9700_lab.conformance_queue import request_pause

            result = request_pause(args.output)
        elif args.command in {"suite-plan", "suite-run", "suite-status"}:
            from qwen_r9700_lab.conformance_campaign import (
                build_campaign,
                coverage,
                load_results,
                run_campaign,
            )

            if args.command == "suite-plan":
                campaign = build_campaign(private_json(args.spec))
                write_private(args.output, campaign)
                result = {
                    "campaign": campaign["sha256"],
                    "cases": len(campaign["cases"]),
                    "gpu_executed": False,
                }
            else:
                campaign = private_json(args.campaign)
                if args.command == "suite-run":
                    report = run_campaign(
                        campaign,
                        args.output,
                        allow_gpu=args.allow_gpu,
                        selected=args.case,
                        keep_going=args.keep_going,
                        resume=args.resume,
                        through=args.through,
                        budget_seconds=args.budget_seconds,
                        retry_failed=args.retry_failed,
                        min_free_bytes=args.min_free_gib * 1024**3,
                    )
                else:
                    report = coverage(campaign, load_results(campaign, args.results))
                result = {
                    "campaign": report["campaign"],
                    "counts": report["counts"],
                    "complete": report["complete"],
                    "universal_correctness": "UNPROVED",
                }
                if args.command == "suite-status":
                    result["checkpoints"] = []
                    for directory in args.results:
                        path = directory / "checkpoint.json"
                        if path.exists():
                            checkpoint = private_json(path)
                            authenticate(checkpoint)
                            if checkpoint["campaign"] != campaign["sha256"]:
                                raise DiagnosticError("checkpoint belongs to another campaign")
                            result["checkpoints"].append(
                                {
                                    key: checkpoint.get(key)
                                    for key in (
                                        "status",
                                        "current",
                                        "through",
                                        "elapsed_this_run",
                                        "updated_ns",
                                    )
                                }
                            )
                print(json.dumps(result, allow_nan=False))
                return (
                    1
                    if (args.command == "suite-run" or args.require_passing)
                    and not report["complete"]
                    else 0
                )
        elif args.command == "reference":
            r = run_reference(private_json(args.plan), args.output)
            result = {
                "frames": len(r["frames"]),
                "schedule": r["sha256"],
                "native_equivalence": "UNPROVED",
            }
        elif args.command == "native":
            result = run_native(
                args.plan, args.config, args.binding, args.output, allow_gpu=args.allow_gpu
            )
        elif args.command == "_native-worker":
            root = args.output
            try:
                native_worker(
                    root / "plan.json", root / "native-config.json", root / "binding.json", root
                )
            except Exception as error:
                write_private(
                    root / "worker-failure.json",
                    {"type": type(error).__name__, "message": str(error)},
                )
                raise
            return 0
        elif args.command in {"compare", "boundaries"}:
            fn = compare_campaign if args.command == "compare" else compare_boundaries
            r = fn(args.reference, args.candidate, args.output)
            result = {
                "equal": r["equal"],
                "report": r["sha256"],
                "first_difference": r["first_difference"],
            }
        elif args.command == "frame":
            result = compare_frames(args.reference, args.candidate)
            write_private(args.output, result)
        elif args.command == "calls":
            result = compare_calls(args.reference, args.candidate, args.output)
        elif args.command == "control":
            result = audit_control(private_json(args.trace), args.output)
        elif args.command == "certificate":
            report = interval_certificate(**private_json(args.bounds))
            write_private(args.output, report)
            result = {
                "equal": report["certified_under_bounds"],
                "report": report["sha256"],
                "bounds_soundness": report["bounds_soundness"],
            }
        elif args.command == "recovery":
            from qwen_r9700_lab.conformance_model import Checkpoint, state_names
            from qwen_r9700_lab.conformance_state import read_frame

            plan = validate_plan(private_json(args.plan))
            checkpoint = Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"])
            try:
                config = checkpoint.config.get("text_config", checkpoint.config)
                if any(
                    read_frame(p)["contract"] != plan["contract"]
                    for p in (args.reference, args.live, args.restored)
                ):
                    raise DiagnosticError("recovery frames do not implement the supplied contract")
                report = compare_recovery(
                    args.reference,
                    args.live,
                    args.restored,
                    args.output,
                    required=state_names(config),
                )
                result = {
                    "equal": report["equal"],
                    "classification": report["classification"],
                    "report": report["sha256"],
                    "native_equivalence": "UNPROVED",
                }
            finally:
                checkpoint.close()
        elif args.command == "operator":
            result = replay_operator(args.capsule, args.output)
            write_private(args.output / "comparison.json", result)
        elif args.command == "prove":
            result = run_obligations(args.output)
        elif args.command == "gate":
            result = gate(args.authority, private_json(args.transition), create=args.create)
        else:
            raise DiagnosticError("unknown conformance action")
        print(json.dumps(result, allow_nan=False))
        return (
            1 if result.get("equal") is False or result.get("all_expected_results") is False else 0
        )
    except SessionMismatchError as error:
        print(
            json.dumps({"error": str(error), "receipt": error.receipt["sha256"]}), file=sys.stderr
        )
        return 1
    except DiagnosticError as error:
        print("conformance refused: " + str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        # No input data or request text in error logs. Details of a native
        # failure remain in its private worker log, not the user's chat.
        print(
            "conformance refused: invalid/incomplete evidence or unsupported execution; "
            "no output published",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
