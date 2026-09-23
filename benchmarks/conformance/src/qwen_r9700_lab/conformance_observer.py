"""Source-bound observers loaded only in an owned qualification subprocess.

Metadata mode does not copy tensors or synchronize the GPU. Head-audit mode
deliberately does both and is reported separately. Neither mode is installed in
the live backend. Host entry/return records do not attest individual device ISA
instructions or prove that an asynchronous launch has completed.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, private_json


class Events:
    def __init__(self, root, execution):
        self.root, self.execution = Path(root), execution
        self.pid, self.sequence, self.previous = os.getpid(), 0, "0" * 64
        self.lock = threading.Lock()
        os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def emit(self, event, **details):
        with self.lock:
            pid = os.getpid()
            if pid != self.pid:
                self.pid, self.sequence, self.previous = pid, 0, "0" * 64
            row = {
                "execution": self.execution,
                "pid": pid,
                "sequence": self.sequence,
                "previous": self.previous,
                "monotonic_ns": time.monotonic_ns(),
                "event": event,
                "details": details,
            }
            row["sha256"] = digest(row)
            fd = os.open(
                self.root / f"events-{self.execution}-{pid}.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            with os.fdopen(fd, "a") as output:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.sequence += 1
            self.previous = row["sha256"]


def read_events(root, execution):
    executions = {execution} if isinstance(execution, str) else set(execution)
    all_rows = []
    files = sorted(Path(root).glob("events-*.jsonl"))
    if not files:
        raise DiagnosticError("no native path observations were produced")
    for path in files:
        previous, sequence = "0" * 64, 0
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    hashed = {k: v for k, v in row.items() if k != "sha256"}
                    valid = (
                        row["sha256"] == digest(hashed)
                        and row["previous"] == previous
                        and row["sequence"] == sequence
                        and row["execution"] in executions
                        and path.name == f"events-{row['execution']}-{row['pid']}.jsonl"
                    )
                except (ValueError, KeyError, TypeError) as error:
                    raise DiagnosticError("incomplete native event evidence") from error
                if not valid:
                    raise DiagnosticError("native event identity or ordering changed")
                previous, sequence = row["sha256"], sequence + 1
                all_rows.append(row)
    return sorted(all_rows, key=lambda r: (r["monotonic_ns"], r["pid"], r["sequence"]))


def wrap(owner, name, events, event, *, details=None, after=None):
    original = getattr(owner, name, None)
    if not callable(original):
        raise DiagnosticError("pinned native observer entry point is missing: " + event)

    @functools.wraps(original)
    def observed(*args, **kwargs):
        extra = details(args, kwargs) if details else {}
        events.emit(event + ".enter", **extra)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            events.emit(event + ".error", error_type=type(error).__name__)
            raise
        more = after(args, kwargs, result) if after else {}
        events.emit(event + ".return", **extra, **more)
        return result

    setattr(owner, name, observed)


def instrument_qwen_parser(cls, events):
    # The deployed ParserManager retains compatibility adapters for structural
    # tags. Those adapters bypass parse_delta and call the extraction methods.
    # Observe the actual inherited methods as well as the direct engine API.
    for method, event in {
        "parse_delta": "parser.delta",
        "extract_tool_calls_streaming": "parser.tool_stream",
        "extract_tool_calls_from_content": "parser.tool_nonstream",
        "extract_reasoning_streaming": "parser.reasoning_stream",
        "extract_reasoning": "parser.reasoning_nonstream",
        "finish_streaming": "parser.finish",
    }.items():
        wrap(cls, method, events, event)


def instrument(module, settings, events):
    name = module.__name__
    expected = settings["observer_sources"].get(name)
    path = Path(inspect.getfile(module)).resolve()
    if expected != hashlib.sha256(path.read_bytes()).hexdigest():
        raise DiagnosticError("native observation source differs: " + name)
    events.emit("source.bound", module=name, source_sha256=expected)
    if name == "vllm.v1.worker.gpu.model_runner":
        cls = module.GPUModelRunner

        def prepared(args, kwargs, result):
            args[0]._qwen_campaign_batch = result
            return {
                "num_reqs": int(result.num_reqs),
                "num_tokens": int(result.num_tokens),
                "draft_tokens": int(result.num_draft_tokens),
            }

        wrap(cls, "prepare_inputs", events, "runner.prepare", after=prepared)

        def committed(args, kwargs, result):
            capture_at_boundary(args[0], settings, events)
            return {}

        wrap(cls, "postprocess_sampled", events, "runner.commit", after=committed)
        # This records the real replay call, not merely a graph-enabled flag.
        import torch

        wrap(torch.cuda.CUDAGraph, "replay", events, "graph.replay")
    elif name == "qwen_radiance_fair_scheduler":
        wrap(
            module.FairScheduler,
            "answer_priority",
            events,
            "priority.apply",
            details=lambda a, k: {
                key: a[1].get(key) for key in ("chat_id", "sequence", "priority", "active")
            },
        )
        wrap(
            module.RequestPhases,
            "finish",
            events,
            "response.finish",
            details=lambda a, k: {
                "chat_id": (a[1].kv_transfer_params or {}).get("qwen_chat", {}).get("id"),
            },
        )
        wrap(
            module.FairScheduler,
            "schedule",
            events,
            "scheduler.step",
            after=lambda a, k, result: {
                "scheduled": {
                    hashlib.sha256(str(r).encode()).hexdigest(): int(n)
                    for r, n in result.num_scheduled_tokens.items()
                },
                "active": a[0].banks.active,
            },
        )
        wrap(module.FairScheduler, "_retire_bank", events, "bank.retire")
        wrap(
            module.WorkerBanks,
            "before",
            events,
            "bank.activate",
            after=lambda a, k, result: {
                "active": a[0].active,
                "ram_banks": len(a[0].images),
            },
        )
    elif name == "qwen_radiance_chat_tier":
        for method, event in {
            "_load": "snapshot.load",
            "_store": "snapshot.store",
            "_flush_record": "snapshot.flush",
            "shutdown": "snapshot.shutdown",
        }.items():
            wrap(module.ChatFileSystemTierManager, method, events, event)
    elif name == "qwen_radiance_cache":
        for method, event in {"publish": "snapshot.publish", "collect": "snapshot.collect"}.items():
            wrap(module.ChatStore, method, events, event)
        original = module.atomic_write

        def atomic(path, content):
            path = Path(path)
            barrier = Path(settings["root"]) / "interrupt-write.arm"
            if path.suffix == ".qkv" and barrier.exists():
                # Leave an actual partial temporary payload beside the old
                # complete head. Only the owned child can reach this branch.
                temporary = path.parent / ".pending-conformance-interrupted"
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content[: max(1, len(content) // 2)])
                    stream.flush()
                    os.fsync(stream.fileno())
                events.emit("snapshot.partial_write", payload_bytes=len(content))
                # The controller kills THIS process group after observing the
                # receipt. A deadline prevents an abandoned controller hanging.
                deadline = time.monotonic() + 30
                while barrier.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                raise DiagnosticError("intentional interrupted snapshot write")
            return original(path, content)

        module.atomic_write = atomic
    elif name == "radiance_verifyhead":

        def head_details(args, kwargs):
            return {"fast": bool(getattr(args[0], "_radiance_fast_ok", False))}

        def head_after(args, kwargs, result):
            if not settings.get("head_audit") or not getattr(args[0], "_radiance_fast_ok", False):
                return {}
            import torch

            if torch.cuda.is_current_stream_capturing():
                return {"head_audit": "capture_not_compared"}
            state, lm_head, hidden = args[:3]
            bias = args[3] if len(args) > 3 else kwargs.get("embedding_bias")
            exact = state._radiance_exact_head(lm_head, hidden, bias)
            fast = result.float().reshape(-1, result.shape[-1])
            exact = exact.float().reshape(-1, exact.shape[-1])
            if fast.shape != exact.shape or not torch.isfinite(exact).all():
                raise DiagnosticError("invalid exact head observations")
            if settings.get("head_fault"):
                # Fault only the diagnostic comparison input, on the actual
                # device. The production result object is returned unchanged.
                fast = fast.clone()
                fast.masked_fill_(exact == exact.amax(-1, keepdim=True), -float("inf"))
                events.emit(
                    "head.fault_applied", fault="omitted_exact_maxima", device=str(fast.device)
                )
            finite = torch.isfinite(fast)
            if torch.isnan(fast).any() or torch.isposinf(fast).any() or not finite.any(-1).all():
                raise DiagnosticError("invalid fast head observations")
            omitted = exact.amax(-1) > exact.masked_fill(~finite, -float("inf")).amax(-1)
            report = {
                "head_audit": "compared",
                "rows": int(fast.shape[0]),
                "omitted_winners": int(omitted.sum()),
                "different_retained_logits": int(((fast != exact) & finite).sum()),
                "different_argmax": int((fast.argmax(-1) != exact.argmax(-1)).sum()),
            }
            if (
                report["omitted_winners"]
                or report["different_retained_logits"]
                or report["different_argmax"]
            ):
                # Preserve actual counterexample tensors, with no text decoding.
                filename = (
                    Path(settings["root"])
                    / f"head-discrepancy-{os.getpid()}-{time.monotonic_ns()}.pt"
                )
                fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    torch.save(
                        {
                            "hidden": hidden.detach().cpu(),
                            "fast": fast.detach().cpu(),
                            "exact": exact.detach().cpu(),
                        },
                        stream,
                    )
                report["counterexample"] = filename.name
            return report

        wrap(
            module,
            "_apply_head_gated",
            events,
            "head.apply",
            details=head_details,
            after=head_after,
        )
    elif name == "vllm.parser.qwen3":
        instrument_qwen_parser(module.Qwen3Parser, events)
    elif name in {
        "vllm.parser.engine.parser_engine",
        "vllm.parser.engine.adapters",
        "vllm.parser.parser_manager",
    }:
        # Bind the inherited implementation and adapter selection as well as
        # Qwen's grammar. Invocation observations are on the concrete class.
        pass
    else:
        raise DiagnosticError("unknown observer module")


def capture_at_boundary(runner, settings, events):
    """Actual synchronous post-commit capture, before the next scheduler step.

    Prefix identity comes from the controller's synthetic input, not from the
    producer being checked. This diagnostic is separate from async/race runs.
    """
    request_path = Path(settings["root"]) / "capture-request.json"
    if not request_path.exists():
        return
    from types import SimpleNamespace

    from qwen_r9700_lab.conformance_radiance import capture_committed_state

    request = private_json(request_path)
    output = Path(settings["root"]) / request["name"]
    if output.exists():
        return
    if output.parent != Path(settings["root"]) or not output.name.startswith("state-"):
        raise DiagnosticError("unsafe qualification capture destination")
    if runner.vllm_config.scheduler_config.async_scheduling:
        raise DiagnosticError("synchronous state capture cannot qualify async execution")
    batch = runner._qwen_campaign_batch
    if batch.num_reqs != 1:
        raise DiagnosticError("state capture requires one dispatched request")
    index = int(batch.idx_mapping_np[0])
    consumed = int(runner.req_states.num_computed_tokens.gpu[index].item())
    if consumed < request["consumed"]:
        return
    if consumed != request["consumed"]:
        raise DiagnosticError("native state capture missed its exact token boundary")
    pending = int(runner.req_states.last_sampled_tokens[index, 0].item())
    requests = [rid for rid, idx in runner.req_states.req_id_to_index.items() if idx == index]
    if len(requests) != 1:
        raise DiagnosticError("native capture has ambiguous request ownership")
    capture_committed_state(
        SimpleNamespace(model_runner=runner),
        output_path=str(output),
        request_id=requests[0],
        expected={
            "consumed": consumed,
            "pending": pending,
            "input_digest": request["input_digest"],
        },
        plan_path=request["plan"],
        binding_path=request["binding"],
        quiescent=True,
    )
    events.emit("state.captured", name=output.name, consumed=consumed)


class ObserverFinder(importlib.abc.MetaPathFinder):
    def __init__(self, settings, events):
        self.settings, self.events = settings, events

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.settings["observer_sources"]:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise DiagnosticError("pinned observation module was not found")
        original = spec.loader
        settings, events = self.settings, self.events

        class Loader(importlib.abc.Loader):
            def create_module(self, inner):
                return original.create_module(inner)

            def exec_module(self, module):
                if (
                    spec.origin is None
                    or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
                    != settings["observer_sources"][fullname]
                ):
                    raise DiagnosticError(
                        "native observer source differs before import: " + fullname
                    )
                original.exec_module(module)
                instrument(module, settings, events)

        spec.loader = Loader()
        return spec


def install_from_environment():
    path = os.environ.get("QWEN_CONFORMANCE_SERVER_SETTINGS")
    if not path or os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("owned native observer is not armed")
    settings = private_json(Path(path))
    root = Path(settings["root"])
    if not root.is_dir() or root.stat().st_mode & 0o077:
        raise DiagnosticError("qualification root is not private")
    events = Events(root, settings["execution"])
    for module in settings["observer_sources"]:
        if module in sys.modules:
            raise DiagnosticError("native module loaded before observer binding")
    sys.meta_path.insert(0, ObserverFinder(settings, events))
    events.emit(
        "observer.ready", scope="host_entry_return", head_audit=settings.get("head_audit", False)
    )
