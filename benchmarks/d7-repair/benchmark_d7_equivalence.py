"""Private, source-bound M1/M8 top-k replay on the retained Pi benchmark corpus.

The corpus is frozen before inference. Both arms consume the same saved natural
continuations; approximate target heads and snapshots are disabled. Outputs are
never decoded, and no generated tool calls execute. Private IDs/logits stay in
owned tmpfs; public reports contain aggregate measurements and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from qwen_r9700_lab.conformance_topk import (
    ReplaySchedule,
    aggregate,
    choose_continuations,
    compare_measurements,
    compare_rows,
    compare_saved_rows,
    require,
    summarize_logits,
)
from qwen_r9700_lab.diagnostic_contract import (
    authenticate,
    digest,
    private_json,
    seal,
    write_private,
)


def private_root(path):
    require(
        path.is_absolute() and str(path).startswith("/dev/shm/qwen-"),
        "private corpus must remain in tmpfs",
    )
    require(
        path.resolve() == path and not path.is_symlink(), "private root cannot traverse a symlink"
    )
    require(
        path.is_dir() and path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o077,
        "private root ownership/permissions differ",
    )


def target_head_identity(model):
    # Qwen3.5's conditional-generation wrapper retains language_model even in
    # language_model_only mode. The drafter is not part of this target object.
    target = getattr(model, "language_model", model)
    require(hasattr(target, "lm_head"), "missing target vocabulary head")
    head = target.lm_head
    require(str(head.weight.dtype) == "torch.bfloat16", "target vocabulary weights are not BF16")
    return {"dtype": str(head.weight.dtype), "shape": list(head.weight.shape)}


def prepare(args):
    """CPU-only rendering, authenticated against the original /tokenize hashes."""
    from transformers import AutoTokenizer

    private_root(args.saved)
    args.corpus.mkdir(mode=0o700)
    private_root(args.corpus)
    reports = args.saved_report
    require(
        json.loads((reports / "completed.json").read_text())["status"] == "MEASURED",
        "source benchmark is incomplete",
    )
    records = json.loads((reports / "capture-generation.json").read_text())
    selected = choose_continuations(records, args.positions)
    expected = json.loads((reports / "fixtures.json").read_text())
    payloads = sorted(args.saved.glob("payload-*.json"))
    require(len(payloads) == len(expected), "saved prompt inventory changed")
    config = private_json(args.saved / "server/server-0.json")["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    template_path = Path(config["chat_template"])
    template = template_path.read_text()
    prefixes, metadata = {}, []
    for row in selected:
        index = row["fixture"]
        if index not in prefixes:
            payload = private_json(payloads[index])
            # Match vLLM's tool-argument normalization before Jinja rendering.
            for message in payload["messages"]:
                if message.get("role") == "assistant" and message.get("reasoning") is not None:
                    message["reasoning_content"] = message["reasoning"]
                for call in message.get("tool_calls", []):
                    if isinstance(call["function"].get("arguments"), str):
                        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            ids = tokenizer.apply_chat_template(
                payload["messages"],
                tools=payload.get("tools"),
                chat_template=template,
                tokenize=True,
                add_generation_prompt=True,
                **{
                    **config.get("default_chat_template_kwargs", {}),
                    **payload.get("chat_template_kwargs", {}),
                },
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            require(
                isinstance(ids, list) and digest(ids) == expected[index]["token_sha256"],
                "rendered prompt differs from the original Pi fixture",
            )
            require(len(ids) == expected[index]["input_tokens"], "prompt token count changed")
            prefixes[index] = ids
        output = private_json(args.saved / f"output-{row['trial']:02d}.json")["tokens"]
        require(
            len(output) == row["output_tokens"] and digest(output) == row["output_sha256"],
            "saved natural completion changed",
        )
        output = output[: row["evaluate_positions"] + 1]
        item = seal({"prefix": prefixes[index], "output": output})
        name = f"continuation-{len(metadata):03d}.json"
        write_private(args.corpus / name, item)
        metadata.append(
            {
                "name": name,
                "sha256": item["sha256"],
                "fixture": index,
                "source_trial": row["trial"],
                "source_seed": row["seed"],
                "prefix_tokens": len(prefixes[index]),
                "prefix_sha256": digest(prefixes[index]),
                "output_sha256": digest(output),
                "source_output_sha256": row["output_sha256"],
                "source_output_tokens": row["output_tokens"],
                "evaluate_positions": row["evaluate_positions"],
            }
        )
    manifest = seal(
        {
            "schema": "urn:qwen:d7-equivalence-corpus:v1",
            "positions": args.positions,
            "source": "retained natural full-BF16 Pi benchmark continuations",
            "selection": (
                "first chronological reference responses; complete M8 groups only; "
                "initial prefill prediction and final short groups excluded"
            ),
            "source_records_sha256": hashlib.sha256(
                (reports / "capture-generation.json").read_bytes()
            ).hexdigest(),
            "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
            "continuations": metadata,
            "tools_executed": 0,
        }
    )
    require(
        sum(row["evaluate_positions"] for row in metadata) == args.positions,
        "incomplete corpus budget",
    )
    write_private(args.corpus / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "CORPUS_FROZEN",
                "sha256": manifest["sha256"],
                "positions": args.positions,
                "responses": len(metadata),
            }
        ),
        flush=True,
    )


class EquivalenceWorkerExtension:
    def qwen_equivalence_runtime(self):
        from qwen_r9700_lab.conformance_artifacts import capture_runtime

        return capture_runtime()

    def qwen_equivalence_install(self, task_path):
        runner = self.model_runner
        require(not hasattr(runner, "_qwen_equivalence"), "benchmark probe already attached")
        probe = EquivalenceProbe(runner, private_json(Path(task_path)))
        try:
            probe.attach()
        except BaseException:
            probe.close()
            raise
        runner._qwen_equivalence = probe
        return {"installed": True}

    def qwen_equivalence_finish(self):
        runner = self.model_runner
        probe = runner._qwen_equivalence
        try:
            return probe.finish()
        finally:
            probe.close()
            del runner._qwen_equivalence


class EquivalenceProbe:
    def __init__(self, runner, task):
        from qwen_r9700_lab.conformance_instrumentation import HookSet
        from qwen_r9700_lab.conformance_radiance import verify_sources

        authenticate(task)
        corpus = private_json(Path(task["continuation"]))
        authenticate(corpus)
        require(corpus["sha256"] == task["continuation_sha256"], "continuation identity differs")
        package = next(
            p
            for p in Path(inspect.getfile(type(runner))).resolve().parents
            if p.name == "site-packages"
        )
        verify_sources(package, task["binding"])
        require(os.environ.get("RADIANCE_VERIFY_HEAD") == "0", "approximate target head is enabled")
        self.validate_execution(runner)
        require(runner.vllm_config.kv_transfer_config is None, "snapshot reuse is not admitted")
        require(runner.cache_config.mamba_cache_mode == "align", "hybrid state convention changed")
        self.target_head = target_head_identity(runner.model)
        require((runner.speculator is not None) == task["speculation"], "wrong speculation arm")
        self.runner, self.task, self.hooks = runner, task, HookSet()
        self.repairs = None
        self.row_trace = None
        self.schedule = ReplaySchedule(
            corpus["prefix"], corpus["output"], speculation=task["speculation"]
        )
        self.root = Path(task["private_output"])
        self.root.mkdir(mode=0o700)
        self.rows, self.prefill, self.logits = [], None, None
        self.pending_positions, self.input_batches, self.head_shapes = None, 0, {}
        self.started = time.monotonic()
        self.last_progress = self.started
        self.input_hash = hashlib.sha256()
        self.reference = None
        if task.get("reference_rows"):
            self.reference = private_json(Path(task["reference_rows"]))
            authenticate(self.reference)
            require(
                self.reference["sha256"] == task["reference_rows_sha256"]
                and self.reference["continuation"] == task["continuation_sha256"],
                "live M1 reference differs from its admitted receipt",
            )

    @staticmethod
    def validate_execution(runner):
        require(
            runner.model_config.enforce_eager
            and not runner.vllm_config.scheduler_config.async_scheduling,
            "requires serialized eager execution",
        )

    def close(self):
        try:
            if self.row_trace is not None:
                self.row_trace.close()
            self.hooks.close()
        finally:
            if self.repairs is not None:
                self.repairs.close()

    def progress(self, phase, **extra):
        from qwen_r9700_lab.conformance_queue import replace_private

        if self.reference is not None and self.rows:
            extra["agreement"] = aggregate(
                [
                    compare_rows(a["logits"], b["logits"])
                    for a, b in zip(self.reference["rows"], self.rows, strict=False)
                ]
            )
            extra["reference_sha256"] = self.reference["sha256"]
            extra["prefill_logits_exact"] = (
                self.reference["prefill"]["logits_sha256"] == self.prefill["logits_sha256"]
            )
        replace_private(
            Path(self.task["report_root"]),
            "progress.json",
            {
                "arm": self.task["arm"],
                "continuation": self.task["index"],
                "phase": phase,
                "evaluated_positions": len(self.rows),
                "continuation_positions": len(self.schedule.output) - 1,
                "elapsed_seconds": time.monotonic() - self.started,
                **extra,
            },
        )
        self.last_progress = time.monotonic()

    def attach(self):
        if self.task.get("repair_manifest"):
            from stock_gdn_runtime import RuntimeRepairs

            self.repairs = RuntimeRepairs(
                self.task["repair_manifest"], target_model=self.runner.model
            )
        runner = self.runner
        original_prepare, original_logits, original_sample = (
            runner.prepare_inputs,
            runner.model.compute_logits,
            runner.sample,
        )

        def prepare_inputs(*args, **kwargs):
            batch = original_prepare(*args, **kwargs)
            require(batch.num_reqs == 1, "unexpected concurrent request")
            positions = batch.positions[: batch.num_tokens].detach().cpu().tolist()
            inputs = batch.input_ids[: batch.num_tokens].detach().cpu().tolist()
            self.schedule.check_inputs(positions, inputs)
            self.pending_positions = positions
            self.input_hash.update(
                bytes.fromhex(digest({"positions": positions, "inputs": inputs}))
            )
            self.input_batches += 1
            if time.monotonic() - self.last_progress > 5:
                self.progress(
                    "decode" if self.schedule.prefill_done else "prefill",
                    prepared=positions[-1] + 1,
                )
            return batch

        def logits(*args, **kwargs):
            result = original_logits(*args, **kwargs)
            require(
                result.ndim == 2
                and result.shape[1] == runner.model_config.hf_text_config.vocab_size,
                "incomplete full vocabulary head",
            )
            self.logits = result.detach().float().cpu().numpy().copy()
            shape = str(tuple(result.shape))
            self.head_shapes[shape] = self.head_shapes.get(shape, 0) + 1
            return result

        def sample(hidden_states, batch, grammar_output):
            require(grammar_output is None, "grammar transformations are outside this measurement")
            self.logits = None
            result, ns, nr = original_sample(hidden_states, batch, grammar_output)
            if not int(ns[0].item()):
                return result, ns, nr
            drafts = int(batch.num_draft_tokens)
            require(
                self.logits is not None and self.logits.shape[0] == drafts + 1,
                "missing target rows",
            )
            step = self.schedule.commit(drafts)
            if step["prefill"]:
                require(
                    self.pending_positions[-1] == len(self.schedule.prefix) - 1,
                    "incomplete prompt prefill",
                )
                self.prefill = summarize_logits(self.logits[0])
                self.progress("decode")
            else:
                for row in range(step["count"]):
                    position = step["start"] + row
                    self.rows.append(
                        {
                            "position": position,
                            "absolute_position": len(self.schedule.prefix) + position,
                            "row": row,
                            "target_rows": drafts + 1,
                            "logits": summarize_logits(self.logits[row]),
                        }
                    )
            result.sampled_token_ids.fill_(-1)
            for j, token in enumerate(step["tokens"]):
                result.sampled_token_ids[0, j] = token
            ns.fill_(step["count"])
            nr.fill_(step["reject"])
            return result, ns, nr

        self.hooks.replace(runner, "prepare_inputs", prepare_inputs)
        self.hooks.replace(runner.model, "compute_logits", logits)
        self.hooks.replace(runner, "sample", sample)
        if runner.speculator is not None:
            original_propose = runner.speculator.propose

            def propose(*args, **kwargs):
                out = original_propose(*args, **kwargs)
                require(tuple(out.shape) == (1, 7), "drafter width differs from fixed D7")
                for j, token in enumerate(self.schedule.proposals()):
                    out[0, j] = token
                return out

            self.hooks.replace(runner.speculator, "propose", propose)

        if self.task.get("trace_rows"):
            from trace_private_d7_rows import PrivateRowTrace

            self.row_trace = PrivateRowTrace(self)

    def finish(self):
        require(
            self.schedule.done and len(self.rows) == len(self.schedule.output) - 1,
            "incomplete measured decode positions",
        )
        repair_receipt = self.repairs.receipt() if self.repairs is not None else None
        trace_receipt = self.row_trace.finish() if self.row_trace is not None else None
        if repair_receipt is not None:
            require(
                not self.task["speculation"] or repair_receipt["recurrent_calls"] > 0,
                "D7 repair was not actually dispatched",
            )
            require(
                not self.repairs.manifest["prefill"] or repair_receipt["prefill_calls"] > 0,
                "prefill repair was not actually dispatched",
            )
            target = repair_receipt.get("target_arithmetic") or {}
            if self.repairs.manifest["stock_attention"] and self.task["speculation"]:
                require(
                    (repair_receipt.get("target_attention") or {}).get("calls", 0) > 0,
                    "independent-query attention repair was not actually dispatched",
                )
            for name in ("norm", "head"):
                require(
                    not self.repairs.manifest["stock_" + name]
                    or target.get(name + "_calls", 0) > 0,
                    "target " + name + " arithmetic repair was not actually dispatched",
                )
        result = seal(
            {
                "schema": "urn:qwen:d7-equivalence-private-rows:v1",
                "task": self.task["sha256"],
                "continuation": self.task["continuation_sha256"],
                "prefill": self.prefill,
                "rows": self.rows,
                "head_shapes": self.head_shapes,
                "target_head": self.target_head,
                "input_batches": self.input_batches,
                "input_batches_sha256": self.input_hash.hexdigest(),
                "wall_seconds": time.monotonic() - self.started,
                "repair_receipt": repair_receipt,
                "row_trace": trace_receipt,
            }
        )
        write_private(self.root / "rows.json", result)
        self.progress("complete")
        return {
            "sha256": result["sha256"],
            "positions": len(self.rows),
            "head_shapes": self.head_shapes,
            "repair_receipt": repair_receipt,
        }


def run_worker(args):
    from qwen_r9700_lab.conformance_cli import reject_dead_native_rpcs
    from qwen_r9700_lab.conformance_model import Checkpoint
    from qwen_r9700_lab.conformance_radiance import verify_sources

    private_root(args.corpus)
    corpus = private_json(args.corpus / "manifest.json")
    authenticate(corpus)
    spec = private_json(args.spec)
    package = Path(importlib.util.find_spec("vllm").origin).parent.parent
    verify_sources(package, spec["binding"])
    require(os.environ.get("RADIANCE_VERIFY_HEAD") == "0", "full BF16 target head is required")
    require(
        not os.environ.get("QWEN_CONFORMANCE_NATIVE_EXPERIMENT"),
        "baseline cannot use a repair hook",
    )
    checkpoint = Checkpoint(Path(spec["native_config"]["model"]), spec["checkpoint_files"])
    write_private(args.output / f"{args.arm}-checkpoint.json", {"identity": checkpoint.identity})
    checkpoint.close()
    from vllm import LLM, SamplingParams

    # Match the pinned native replay's isolation settings. Keep this small
    # adapter independent of newer campaign-controller helpers.
    config = dict(spec["native_config"])
    config.update(enforce_eager=True, max_num_seqs=1, async_scheduling=False)
    for name in ("kv_transfer_config", "scheduler_cls", "additional_config", "compilation_config"):
        config.pop(name, None)
    if args.arm == "m1":
        config.pop("speculative_config", None)
    config.update(
        # Disabling prefix caching also changes vLLM's hybrid-state convention.
        # Preserve align mode; unique request salts prevent cross-request reuse.
        enable_prefix_caching=True,
        disable_log_stats=True,
        worker_extension_cls="benchmark_d7_equivalence.EquivalenceWorkerExtension",
    )
    write_private(args.output / f"{args.arm}-config.json", seal(config))
    llm = LLM(**config)
    engine = llm.llm_engine
    receipts = []
    with reject_dead_native_rpcs(engine.engine_core):
        write_private(
            args.output / f"{args.arm}-runtime-before.json",
            {"workers": llm.collective_rpc("qwen_equivalence_runtime")},
        )
    for index, row in enumerate(corpus["continuations"]):
        data = private_json(args.corpus / row["name"])
        authenticate(data)
        require(data["sha256"] == row["sha256"], "frozen corpus changed")
        task = seal(
            {
                "continuation": str(args.corpus / row["name"]),
                "continuation_sha256": row["sha256"],
                "private_output": str(args.corpus / args.revision / args.arm / f"{index:03d}"),
                "binding": spec["binding"],
                "speculation": args.arm == "m8",
                "arm": args.arm,
                "index": index,
                "report_root": str(args.output),
                "repair_manifest": str(args.repair_manifest) if args.repair_manifest else None,
                "trace_rows": args.trace_rows,
                "reference_rows": (
                    str(args.corpus / args.revision / "m1" / f"{index:03d}/rows.json")
                    if args.reuse_m1 and args.arm == "m8"
                    else None
                ),
                "reference_rows_sha256": (
                    private_json(args.corpus / args.revision / "m1" / f"{index:03d}/rows.json")[
                        "sha256"
                    ]
                    if args.reuse_m1 and args.arm == "m8"
                    else None
                ),
            }
        )
        task_path = args.output / f"{args.arm}-task-{index:03d}.json"
        write_private(task_path, task)
        params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            ignore_eos=True,
            max_tokens=len(data["output"]),
            detokenize=False,
        )
        request_id = f"qwen-equivalence-{args.arm}-{index}"
        reached = False
        try:
            with reject_dead_native_rpcs(engine.engine_core):
                llm.collective_rpc("qwen_equivalence_install", args=(str(task_path),))
                engine.add_request(
                    request_id,
                    {
                        "prompt_token_ids": data["prefix"],
                        "cache_salt": digest({"corpus": corpus["sha256"], "response": index}),
                    },
                    params,
                )
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        require(output.request_id == request_id, "unexpected output request")
                        if output.outputs and len(output.outputs[0].token_ids) >= len(
                            data["output"]
                        ):
                            require(
                                list(output.outputs[0].token_ids) == data["output"],
                                "emitted forced stream differs from the frozen corpus",
                            )
                            reached = True
                    if reached:
                        break
                require(reached, "replay stopped before its requested position budget")
                receipt = llm.collective_rpc("qwen_equivalence_finish")
                receipts.append(receipt)
                write_private(
                    args.output / f"{args.arm}-receipt-{index:03d}.json", {"workers": receipt}
                )
        finally:
            engine.abort_request([request_id])
    write_private(
        args.output / f"{args.arm}-complete.json",
        seal({"corpus": corpus["sha256"], "receipts": receipts}),
    )
    with reject_dead_native_rpcs(engine.engine_core):
        write_private(
            args.output / f"{args.arm}-runtime-after.json",
            {"workers": llm.collective_rpc("qwen_equivalence_runtime")},
        )


def summarize(args):
    corpus = private_json(args.corpus / "manifest.json")
    authenticate(corpus)
    all_rows, prefill, per_response = [], [], []
    for index, row in enumerate(corpus["continuations"]):
        arms = [
            private_json(args.corpus / args.revision / arm / f"{index:03d}/rows.json")
            for arm in ("m1", "m8")
        ]
        for arm in arms:
            authenticate(arm)
            require(
                arm["continuation"] == row["sha256"]
                and len(arm["rows"]) == row["evaluate_positions"],
                "incomparable or incomplete response",
            )
        compared = []
        for position, (left, right) in enumerate(
            zip(arms[0]["rows"], arms[1]["rows"], strict=True)
        ):
            require(
                left["position"] == right["position"] == position
                and left["absolute_position"] == right["absolute_position"],
                "prediction positions differ",
            )
            require(
                left["target_rows"] == 1 and right["target_rows"] == 8,
                "not a genuine M1/M8 decode comparison",
            )
            compared.append(compare_rows(left["logits"], right["logits"]))
        all_rows.extend(compared)
        prefill.append(compare_rows(arms[0]["prefill"], arms[1]["prefill"]))
        per_response.append(
            {
                "index": index,
                "fixture": row["fixture"],
                "metrics": aggregate(compared),
                "arm_sha256": [arm["sha256"] for arm in arms],
            }
        )
    require(len(all_rows) == corpus["positions"], "not all requested positions were measured")
    result = seal(
        {
            "schema": "urn:qwen:d7-equivalence-summary:v1",
            "status": "MEASURED",
            "revision": args.revision,
            "corpus": corpus["sha256"],
            "metrics": aggregate(all_rows),
            "initial_prefill": aggregate(prefill),
            "responses": per_response,
            "ordering": (
                "score descending, token ID ascending; inclusive boundary ties also reported"
            ),
            "scope": (
                "forced natural Pi histories, eager TP1, no snapshot/prefix reuse, full BF16 "
                "vocabulary head; not universal equivalence or free-running quality"
            ),
            "formal_equivalence": "UNPROVED",
            "post_repair_measurement": "MEASURED" if args.repair_manifest else "PENDING",
            "repair_manifest": private_json(args.repair_manifest) if args.repair_manifest else None,
            "m1_reference_reuse": (
                private_json(args.output / "reference-reuse.json") if args.reuse_m1 else None
            ),
        }
    )
    write_private(args.output / "summary.json", result)
    print(
        json.dumps(
            {"status": result["status"], "sha256": result["sha256"], "metrics": result["metrics"]}
        ),
        flush=True,
    )


def reuse_m1_reference(args, spec):
    """Admit an explicit, source-bound reference reuse decision, not inferred equivalence."""
    certificate = private_json(args.reuse_m1)
    authenticate(certificate)
    corpus = private_json(args.corpus / "manifest.json")
    authenticate(corpus)
    bundle = private_json(args.repair_manifest)
    authenticate(bundle)
    require(
        certificate["schema"] == "urn:qwen:d7-m1-reuse:v1"
        and certificate["corpus"] == corpus["sha256"]
        and certificate["candidate_bundle"] == bundle["sha256"]
        and certificate["native_binding"] == spec["binding"]["sha256"],
        "M1 reuse decision does not cover this candidate, corpus and backend",
    )
    revision = certificate["reference_revision"]
    require(
        revision and all(c.isalnum() or c in "-_" for c in revision), "unsafe reference revision"
    )
    require(revision != args.revision, "reference cannot overwrite itself")
    source = Path(certificate["reference_report"])
    complete, measurement, config = [
        private_json(source / name)
        for name in ("m1-complete.json", "measurement.json", "m1-config.json")
    ]
    for doc in (complete, measurement, config):
        authenticate(doc)
    require(
        complete["sha256"] == certificate["reference_complete"]
        and complete["corpus"] == corpus["sha256"]
        and measurement["revision"] == revision
        and measurement["native_binding"] == certificate["native_binding"]
        and measurement["spec_sha256"] == hashlib.sha256(args.spec.read_bytes()).hexdigest()
        and len(measurement["repair_hooks"]) == 1
        and measurement["repair_hooks"][0]["sha256"] == certificate["reference_bundle"]
        and private_json(source / "m1-process-result.json")["returncode"] == 0,
        "M1 reference was incomplete or used a different backend, fixture or repair",
    )
    require(
        config["max_num_batched_tokens"] == 2048
        and config["enforce_eager"] is True
        and config["max_num_seqs"] == 1
        and config["async_scheduling"] is False
        and "speculative_config" not in config
        and all(r["prefix_tokens"] % 2048 > 8 for r in corpus["continuations"]),
        "M1 prefill may enter the changed small-batch normalization path",
    )
    require(len(complete["receipts"]) == len(corpus["continuations"]), "incomplete M1 receipts")
    docs = []
    for index, expected in enumerate(corpus["continuations"]):
        row = private_json(args.corpus / revision / "m1" / f"{index:03d}/rows.json")
        authenticate(row)
        receipt = complete["receipts"][index]
        require(
            len(receipt) == 1
            and receipt[0]["sha256"] == row["sha256"]
            and row["continuation"] == expected["sha256"]
            and len(row["rows"]) == expected["evaluate_positions"]
            and row["repair_receipt"]["bundle"] == certificate["reference_bundle"]
            # Each one-token target decode executes 129 hidden/residual norms
            # and 32 Q/K norms. Extra admitted norm calls would mean a small
            # prefill batch, for which this M8-only reuse argument is invalid.
            and row["repair_receipt"]["target_arithmetic"]["norm_calls"]
            == row["repair_receipt"]["target_arithmetic"]["norm_rows"]
            == 161 * expected["evaluate_positions"]
            and all(
                r["target_rows"] == 1
                and r["position"] == i
                and r["absolute_position"] == expected["prefix_tokens"] + i
                for i, r in enumerate(row["rows"])
            ),
            "saved M1 rows differ from the completed reference",
        )
        docs.append(row)
    # Copy authenticated evidence inside private tmpfs; original measurements
    # retain their own source identities and are explicitly labelled reused.
    destination = args.corpus / args.revision / "m1"
    destination.mkdir(mode=0o700)
    for index, row in enumerate(docs):
        directory = destination / f"{index:03d}"
        directory.mkdir(mode=0o700)
        write_private(directory / "rows.json", row)
    record = seal(
        {
            "decision": certificate,
            "positions": corpus["positions"],
            "reference_rows": [r["sha256"] for r in docs],
            "scope": "Previously measured M1, explicitly reused; no new M1 execution.",
        }
    )
    write_private(args.output / "reference-reuse.json", record)
    return record


def run(args):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    require(
        args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"),
        "GPU use requires explicit admission and the shared lease",
    )
    if args.repair_manifest:
        from stock_gdn_runtime import validate_manifest

        validate_manifest(args.repair_manifest)
        require(not args.revision.startswith("baseline"), "repairs require a distinct revision")
    private_root(args.corpus)
    args.output.mkdir(mode=0o700)
    revision_root = args.corpus / args.revision
    revision_root.mkdir(mode=0o700)
    spec = private_json(args.spec)
    reference = reuse_m1_reference(args, spec) if args.reuse_m1 else None
    env = worker_environment(spec, args.output)
    env["RADIANCE_VERIFY_HEAD"] = "0"
    env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env["PYTHONPATH"]
    write_private(args.output / "binding.json", spec["binding"])
    write_private(
        args.output / "measurement.json",
        seal(
            {
                "revision": args.revision,
                "corpus": private_json(args.corpus / "manifest.json")["sha256"],
                "native_binding": spec["binding"]["sha256"],
                "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metrics_sha256": hashlib.sha256(
                    Path(inspect.getfile(ReplaySchedule)).read_bytes()
                ).hexdigest(),
                "environment": {k: digest(v) for k, v in sorted(env.items())},
                "repair_hooks": (
                    [private_json(args.repair_manifest)] if args.repair_manifest else []
                ),
                "m1_reference_reuse": reference,
            }
        ),
    )
    with gpu_lease(args.output / "gpu-lease"):
        for arm in ("m8",) if reference else ("m1", "m8"):
            (revision_root / arm).mkdir(mode=0o700)
            argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--arm", arm]
            for key in ("spec", "corpus", "output", "revision"):
                argv += ["--" + key, str(getattr(args, key))]
            if args.repair_manifest:
                argv += ["--repair-manifest", str(args.repair_manifest)]
            if args.trace_rows:
                argv += ["--trace-rows"]
            if args.reuse_m1:
                argv += ["--reuse-m1", str(args.reuse_m1)]
            with OwnedProcess(
                argv, args.corpus / args.revision / f"{arm}-process", env=env, timeout=14400
            ) as process:
                code = process.wait()
            write_private(args.output / f"{arm}-process-result.json", {"returncode": code})
            require(
                code == 0,
                "private equivalence worker failed; log retained without printing chat contents",
            )
        summarize(args)


def compare_revisions(args):
    before, after = private_json(args.before), private_json(args.after)
    result = compare_measurements(before, after)
    if args.corpus is not None:
        for report in (before, after):
            require(
                report["revision"] and all(c.isalnum() or c in "-_" for c in report["revision"]),
                "unsafe saved revision name",
            )
        private_root(args.corpus)
        corpus = private_json(args.corpus / "manifest.json")
        authenticate(corpus)
        require(corpus["sha256"] == result["corpus"], "saved corpus differs from reports")
        drift = {}
        for arm, width in (("m1", 1), ("m8", 8)):
            rows, prefill = [], []
            for index, expected in enumerate(corpus["continuations"]):
                saved = [
                    private_json(args.corpus / report["revision"] / arm / f"{index:03d}/rows.json")
                    for report in (before, after)
                ]
                for report, data in zip((before, after), saved, strict=True):
                    arm_index = 0 if arm == "m1" else 1
                    require(
                        data["sha256"] == report["responses"][index]["arm_sha256"][arm_index]
                        and data["continuation"] == expected["sha256"]
                        and len(data["rows"]) == expected["evaluate_positions"],
                        "saved replay differs from published measurement",
                    )
                compared, initial = compare_saved_rows(*saved, target_rows=width)
                rows.extend(compared)
                prefill.append(initial)
            require(len(rows) == 10000, "incomplete before/after arm comparison")
            drift[arm] = {"metrics": aggregate(rows), "initial_prefill": aggregate(prefill)}
        result = seal({**{k: v for k, v in result.items() if k != "sha256"}, "arm_drift": drift})
    write_private(args.report, result)
    print(json.dumps({"status": "COMPARED", "sha256": result["sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for key in ("saved", "saved-report", "corpus"):
        preparation.add_argument("--" + key, type=Path, required=True)
    preparation.add_argument("--positions", type=int, default=10000)
    comparison = commands.add_parser("compare")
    for key in ("before", "after", "report"):
        comparison.add_argument("--" + key, type=Path, required=True)
    comparison.add_argument("--corpus", type=Path)
    for name in ("run", "worker", "summarize"):
        command = commands.add_parser(name)
        for key in ("spec", "corpus", "output"):
            command.add_argument("--" + key, type=Path, required=True)
        command.add_argument("--revision", default="baseline-unfixed")
        command.add_argument("--repair-manifest", type=Path)
        command.add_argument("--trace-rows", action="store_true")
        command.add_argument("--reuse-m1", type=Path)
        if name == "run":
            command.add_argument("--allow-gpu", action="store_true")
        if name == "worker":
            command.add_argument("--arm", choices=("m1", "m8"), required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "compare":
        compare_revisions(args)
        return
    if hasattr(args, "revision"):
        require(
            args.revision and all(c.isalnum() or c in "-_" for c in args.revision),
            "unsafe revision name",
        )
    {"prepare": prepare, "run": run, "worker": run_worker, "summarize": summarize}[args.command](
        args
    )


if __name__ == "__main__":
    main()
