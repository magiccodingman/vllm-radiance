"""Backend-neutral replay plans, independent prefill and operator capsules.

Forced tokens are diagnostic inputs, never a replacement sampler or a repair
of the user's conversation. Only the separate checked session may publish.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from qwen_r9700_lab.conformance_artifacts import reference_runtime_identity
from qwen_r9700_lab.conformance_model import Checkpoint, QuantizedQwenReference, state_names
from qwen_r9700_lab.conformance_reference import OPERATORS, reference_contract, reference_semantics
from qwen_r9700_lab.conformance_state import FrameWriter, load_arrays, read_frame
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    integer,
    require_sha,
    seal,
    semantic_identity,
    write_private,
)

PLAN_SCHEMA = "urn:qwen:conformance-replay-plan:v1"
SCHEDULE_SCHEMA = "urn:qwen:conformance-schedule:v1"
OUTPUT_COMPONENTS = ["output.logits", "output.greedy"]


def reference_code_identity():
    identity = {
        name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
        for name in (
            "conformance_model",
            "conformance_reference",
            "conformance_replay",
            "conformance_state",
            "conformance_boundaries",
            "conformance_artifacts",
            "diagnostic_contract",
            "ordered_reference_linear",
            "exact_fp8_metrics",
        )
    }
    identity["ordered_reference_linear.c"] = hashlib.sha256(
        Path(__file__).with_name("ordered_reference_linear.c").read_bytes()
    ).hexdigest()
    return identity


def validate_plan(plan: dict) -> dict:
    authenticate(plan)
    if set(plan) - {
        "accepted_widths",
        "reference_profile",
        "reference_semantics",
        "reference_runtime",
        "observation_positions",
        "reference_linear",
    } != {
        "schema",
        "contract",
        "execution",
        "adapter",
        "checkpoint",
        "checkpoint_files",
        "kv_scales",
        "prefix",
        "forced_tokens",
        "reference_arithmetic",
        "sha256",
    }:
        raise DiagnosticError("replay plan has missing or unsupported fields")
    if plan["schema"] != PLAN_SCHEMA:
        raise DiagnosticError("unsupported replay plan")
    for name in ("contract", "execution", "adapter"):
        require_sha(plan[name])
    profile = plan.get("reference_profile", "radiance-fp8")
    if plan["reference_arithmetic"] != reference_contract(profile):
        raise DiagnosticError("reference arithmetic/runtime does not match this replay plan")
    if ("reference_profile" in plan) != ("reference_semantics" in plan):
        raise DiagnosticError("reference profile needs its complete semantic contract")
    if "reference_semantics" in plan:
        require_sha(plan.get("reference_runtime"))
        semantics = plan["reference_semantics"]
        expected = reference_semantics(
            plan["checkpoint_files"], semantics["weights"]["config"], plan["kv_scales"], profile
        )
        if semantics != expected or plan["contract"] != semantic_identity(expected):
            raise DiagnosticError("plan settings differ from its reference contract")
    if plan["adapter"] != digest(reference_code_identity()):
        raise DiagnosticError("reference implementation changed; regenerate the replay plan")
    if "reference_linear" in plan:
        from qwen_r9700_lab.ordered_reference_linear import validate_binding

        validate_binding(plan["reference_linear"])
    for name in ("prefix", "forced_tokens"):
        if not isinstance(plan[name], list) or not plan[name]:
            raise DiagnosticError("initial prefill and forced-token schedule must be nonempty")
        for token in plan[name]:
            integer(token)
            if token >= 2**31:
                raise DiagnosticError("token outside portable representation")
    widths = plan.get("accepted_widths", [0] * (len(plan["forced_tokens"]) - 1))
    if not isinstance(widths, list) or any(integer(k) > 7 for k in widths):
        raise DiagnosticError("only explicit D7 accepted widths 0 through 7 are admitted")
    if 1 + sum(k + 1 for k in widths) != len(plan["forced_tokens"]):
        raise DiagnosticError("forced tokens do not cover the accepted-prefix schedule")
    positions = plan.get("observation_positions")
    if "observation_positions" in plan and (
        not isinstance(positions, list)
        or not positions
        or any(
            type(p) is not int or p < 0 or p >= len(plan["prefix"]) + len(plan["forced_tokens"]) - 1
            for p in positions
        )
        or positions != sorted(set(positions))
    ):
        raise DiagnosticError(
            "observation positions must be a nonempty ordered materialized subset"
        )
    return plan


def scheduled_inputs(plan):
    prefix = list(plan["prefix"])
    widths = plan.get("accepted_widths", [0] * (len(plan["forced_tokens"]) - 1))
    cursor = 0
    for index in range(len(widths) + 1):
        if index:
            advance = widths[index - 1] + 1
            prefix.extend(plan["forced_tokens"][cursor : cursor + advance])
            cursor += advance
        pending = plan["forced_tokens"][cursor]
        yield {
            "name": f"frame-{index:06d}",
            "phase": "prefill" if index == 0 else "step",
            "consumed": len(prefix),
            "input_digest": digest(prefix),
            "pending": pending,
        }


def observation_domain(plan):
    """Every materialized token, including earlier prefill chunks and D7 rows.

    The emitted final token is pending, so must not appear in this domain.
    Prefix identities use the same public token-ID digest as committed frames.
    """
    tokens = plan["prefix"] + plan["forced_tokens"][:-1]
    selected = plan.get("observation_positions", list(range(len(tokens))))
    admitted = frozenset(selected)
    # Hash canonical JSON prefixes incrementally; avoid quadratic re-encoding
    # for a 250K prompt. Hash copies add only the closing bracket.
    stream, identities = hashlib.sha256(b"["), {}
    for i, token in enumerate(tokens):
        stream.update(("," if i else "").encode() + str(token).encode("ascii"))
        if i in admitted:
            prefix = stream.copy()
            prefix.update(b"]")
            identities[i] = prefix.hexdigest()
    return selected, identities


def write_model_frame(model, path: Path, *, phase: str, pending: int, logits) -> dict:
    writer = FrameWriter(
        path,
        contract=model.contract,
        execution=model.execution,
        adapter=model.adapter,
        input_digest=digest(model.tokens),
        phase=phase,
        consumed=len(model.tokens),
        pending=pending,
        logical={"execution_mode": "forced_token_replay"},
        expected=state_names(model.config) + OUTPUT_COMPONENTS,
    )
    writer.array("sequence.tokens", np.asarray(model.tokens, dtype="<i4"))
    writer.array("sequence.position", np.asarray([len(model.tokens)], dtype="<i8"))
    for layer, state in model.state.items():
        for name, value in state.items():
            if name in {"keys", "values"}:
                writer.add(
                    f"layer.{layer:03d}.{name}",
                    value.tobytes(),
                    dtype=model.precision["kv_encoding"],
                    shape=value.shape,
                )
            else:
                writer.array(f"layer.{layer:03d}.{name}", value)
    logits = np.asarray(logits, dtype="<f4")
    if logits.shape != (model.config["vocab_size"],) or not np.isfinite(logits).all():
        raise DiagnosticError("invalid full-vocabulary target observation")
    writer.array("output.logits", logits)
    writer.array("output.greedy", np.asarray([np.argmax(logits)], dtype="<i4"))
    return writer.finish()


class CampaignWriter:
    def __init__(self, root: Path, plan: dict, *, coverage: list[str], backend: str):
        validate_plan(plan)
        root.mkdir(mode=0o700)
        self.root, self.plan, self.backend = root, plan, backend
        self.expected, self.frames = list(scheduled_inputs(plan)), []
        self.coverage = coverage
        write_private(root / "plan.json", plan)

    def record(self, path: Path):
        if len(self.frames) >= len(self.expected):
            raise DiagnosticError("backend produced an extra frame")
        expected = self.expected[len(self.frames)]
        frame = read_frame(path)
        if path != self.root / expected["name"] or any(
            frame[k] != expected[k] for k in expected if k != "name"
        ):
            raise DiagnosticError("backend did not follow the forced-token schedule")
        if frame["coverage"] != self.coverage or frame["contract"] != self.plan["contract"]:
            raise DiagnosticError("backend frame does not cover the reference contract")
        if frame["logical"].get("execution_mode") != "forced_token_replay":
            raise DiagnosticError("forced replay frame lacks its publication prohibition")
        self.frames.append({**expected, "sha256": frame["sha256"]})

    def finish(self):
        if len(self.frames) != len(self.expected):
            raise DiagnosticError("backend stopped before the replay finished")
        result = seal(
            {
                "schema": SCHEDULE_SCHEMA,
                "contract": self.plan["contract"],
                "plan": self.plan["sha256"],
                "frames": self.frames,
                "coverage": self.coverage,
                "backend": self.backend,
                "initial_state": "independent_zero_state",
                "published_to_session": False,
                "native_equivalence": "UNPROVED",
            }
        )
        write_private(self.root / "schedule.json", result)
        return result


def run_reference(plan: dict, root: Path):
    validate_plan(plan)
    if (
        "reference_runtime" in plan
        and plan["reference_runtime"] != reference_runtime_identity()["sha256"]
    ):
        raise DiagnosticError(
            "CPU reference executable/runtime changed; regenerate the replay plan"
        )
    checkpoint = Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"])
    model = QuantizedQwenReference(
        checkpoint,
        kv_scales=plan["kv_scales"],
        contract=plan["contract"],
        execution=plan["execution"],
        adapter=plan["adapter"],
        reference_profile=plan.get("reference_profile", "radiance-fp8"),
    )
    try:
        accelerator = None
        if "reference_linear" in plan:
            from qwen_r9700_lab import conformance_reference
            from qwen_r9700_lab.ordered_reference_linear import OrderedLinear

            accelerator = OrderedLinear(conformance_reference, plan["reference_linear"])
            model._linear = accelerator
        if (
            "reference_semantics" in plan
            and model.config != plan["reference_semantics"]["weights"]["config"]
        ):
            raise DiagnosticError("checkpoint configuration differs from the reference contract")
        campaign = CampaignWriter(
            root,
            plan,
            coverage=state_names(model.config) + OUTPUT_COMPONENTS,
            backend=(
                "independent-ordered-c-numpy-reference"
                if accelerator is not None
                else "independent-numpy-reference"
            ),
        )
        from qwen_r9700_lab.conformance_boundaries import BoundaryRecorder, detailed_stages

        positions, inputs = observation_domain(plan)
        boundaries = BoundaryRecorder(
            root / "boundaries",
            contract=model.contract,
            execution=model.execution,
            adapter=model.adapter,
            positions=positions,
            layers=model.config["num_hidden_layers"],
            input_digests=inputs,
        )
        detailed = BoundaryRecorder(
            root / "semantic",
            contract=model.contract,
            execution=model.execution,
            adapter=model.adapter,
            positions=boundaries.positions,
            layers=model.config["num_hidden_layers"],
            input_digests=boundaries.inputs,
            layer_stages=detailed_stages(model.config),
        )

        def observe(position, layer, stage, value):
            boundaries.record(position, layer, stage, value)
            detailed.record(position, layer, stage, value)

        model.capture = observe
        logits = None
        for token in plan["prefix"]:
            logits = model.step(token)
        for index, expected in enumerate(campaign.expected):
            if index:
                begin = len(model.tokens) - len(plan["prefix"])
                end = expected["consumed"] - len(plan["prefix"])
                for token in plan["forced_tokens"][begin:end]:
                    logits = model.step(token)
            path = root / expected["name"]
            write_model_frame(
                model, path, phase=expected["phase"], pending=expected["pending"], logits=logits
            )
            campaign.record(path)
        boundaries.finish()
        detailed.finish()
        if accelerator is not None:
            write_private(
                root / "ordered-linear.json",
                seal(
                    {
                        "binding": accelerator.binding,
                        "calls": accelerator.calls,
                        "fallbacks": accelerator.fallbacks,
                        "formal_equivalence": "UNPROVED",
                    }
                ),
            )
        return campaign.finish()
    finally:
        model.close()


def write_operator_capsule(
    root: Path,
    *,
    operator: str,
    inputs: dict,
    outputs: list,
    options: dict,
    contract: str,
    execution: str,
    adapter: str,
):
    if operator not in OPERATORS or not inputs or not outputs:
        raise DiagnosticError("operator capsule has no supported comparison domain")
    inputs = dict(sorted(inputs.items()))
    names = ["input." + k for k in inputs] + [f"output.{i}" for i in range(len(outputs))]
    identity = digest(
        {
            "operator": operator,
            "options": options,
            "inputs": {
                k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                for k, v in inputs.items()
            },
        }
    )
    writer = FrameWriter(
        root,
        contract=contract,
        execution=execution,
        adapter=adapter,
        input_digest=identity,
        phase="operator",
        consumed=0,
        pending=None,
        expected=names,
        logical={"operator": operator, "options": options},
    )
    for name, value in inputs.items():
        writer.array("input." + name, np.asarray(value))
    for index, value in enumerate(outputs):
        writer.array(f"output.{index}", np.asarray(value))
    return writer.finish()


def replay_operator(capsule: Path, output: Path):
    from qwen_r9700_lab.conformance_state import compare_frames

    frame, arrays = load_arrays(capsule)
    operator = frame["logical"].get("operator")
    if frame["phase"] != "operator" or operator not in OPERATORS:
        raise DiagnosticError("unsupported operator capsule")
    kwargs = {k.removeprefix("input."): v for k, v in arrays.items() if k.startswith("input.")}
    options = frame["logical"]["options"]
    if set(kwargs) & set(options):
        raise DiagnosticError("operator options overwrite captured inputs")
    result = OPERATORS[operator](**kwargs, **options)
    outputs = list(result) if isinstance(result, tuple) else [result]
    write_operator_capsule(
        output,
        operator=operator,
        inputs=kwargs,
        outputs=outputs,
        options=options,
        contract=frame["contract"],
        execution=frame["execution"],
        adapter=frame["adapter"],
    )
    return compare_frames(output, capsule)
