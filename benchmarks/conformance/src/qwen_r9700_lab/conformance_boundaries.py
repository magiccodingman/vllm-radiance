"""Canonical semantic-boundary observations, reusable across runtime adapters."""

import re
from pathlib import Path

import numpy as np

from qwen_r9700_lab.conformance_state import FrameWriter, compare_frames, read_frame
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)

STAGES = ("input_norm", "post_attention_norm", "output")


def validate_domain(positions, layers, stages):
    if (
        not isinstance(positions, list)
        or not positions
        or any(type(p) is not int or p < 0 for p in positions)
        or sorted(set(positions)) != positions
        or type(layers) is not int
        or layers < 1
        or not isinstance(stages, list)
        or len(stages) != layers
    ):
        raise DiagnosticError("invalid semantic observation domain")
    for row in stages:
        if (
            not isinstance(row, list)
            or not row
            or any(
                not isinstance(s, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", s)
                for s in row
            )
            or len(set(row)) != len(row)
        ):
            raise DiagnosticError("invalid or duplicate semantic stage")


def detailed_stages(config):
    """Native attention semantics; no inherited Quest selector requirement."""
    start = ["input", "input_norm"]
    end = [
        "attention_residual",
        "post_attention_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_intermediate",
        "mlp_down",
        "output",
    ]
    linear = [
        "gdn_qkv",
        "gdn_z",
        "gdn_b",
        "gdn_a",
        "conv_input_state",
        "conv_output",
        "conv_state",
        "gdn_q",
        "gdn_k",
        "gdn_v",
        "gdn_decay",
        "gdn_beta",
        "gdn_input_state",
        "gdn_output",
        "gdn_state",
        "gdn_gated",
        "attention_projected",
    ]
    attention = [
        "attention_q_projected",
        "attention_k_projected",
        "attention_v_projected",
        "attention_q_norm",
        "attention_k_norm",
        "attention_q_rope",
        "attention_k_rope",
        "attention_key_stored",
        "attention_value_stored",
        "attention_output",
        "attention_projected",
    ]
    return [
        start + (linear if k == "linear_attention" else attention) + end
        for k in config["layer_types"]
    ]


class BoundaryRecorder:
    def __init__(
        self,
        root: Path,
        *,
        contract: str,
        execution: str,
        adapter: str,
        positions: list[int],
        layers: int,
        input_digests: dict[int, str],
        layer_stages=None,
    ):
        root.mkdir(mode=0o700)
        self.root, self.positions, self.layers = root, positions, layers
        self.identities = {"contract": contract, "execution": execution, "adapter": adapter}
        self.inputs, self.recorded = input_digests, {}
        self.stages = (
            [list(STAGES) for _ in range(layers)] if layer_stages is None else layer_stages
        )
        validate_domain(positions, layers, self.stages)
        self.position_set = frozenset(positions)
        if (
            not positions
            or len(set(positions)) != len(positions)
            or layers < 1
            or len(self.stages) != layers
            or any(not s or len(set(s)) != len(s) for s in self.stages)
            or set(input_digests) != set(positions)
        ):
            raise DiagnosticError("invalid semantic observation domain")

    def record(self, position, layer, stage, value):
        if position not in self.position_set or layer < 0:
            return
        if not 0 <= layer < self.layers:
            raise DiagnosticError("unregistered semantic layer")
        if stage not in self.stages[layer]:
            return
        key = (position, layer, stage)
        if key in self.recorded:
            raise DiagnosticError("semantic boundary was captured twice")
        name = f"p{position:09d}-l{layer:03d}-{stage}"
        writer = FrameWriter(
            self.root / name,
            **self.identities,
            input_digest=self.inputs[position],
            phase="operator",
            consumed=position + 1,
            pending=None,
            expected=["value"],
            logical={"layer": layer, "stage": stage},
        )
        writer.array("value", np.asarray(value, dtype="<f4"))
        self.recorded[key] = {"name": name, "sha256": writer.finish()["sha256"]}

    def finish(self):
        expected = [
            (p, layer, s)
            for p in self.positions
            for layer in range(self.layers)
            for s in self.stages[layer]
        ]
        if set(expected) != set(self.recorded):
            raise DiagnosticError("semantic boundary coverage is incomplete")
        doc = seal(
            {
                "schema": "urn:qwen:boundary-schedule:v2",
                "ordering": "token/layer/stage",
                "positions": self.positions,
                "layers": self.layers,
                "layer_stages": self.stages,
                "frames": [self.recorded[k] for k in expected],
            }
        )
        write_private(self.root / "boundaries.json", doc)
        return doc


def compare_boundaries(reference: Path, candidate: Path, output: Path):
    docs = [private_json(p / "boundaries.json") for p in (reference, candidate)]
    for d in docs:
        authenticate(d)
        if (
            d.get("schema") != "urn:qwen:boundary-schedule:v2"
            or not d.get("positions")
            or not d.get("layers")
        ):
            raise DiagnosticError("incomplete semantic boundary schedule")
        stages = d.get("layer_stages")
        validate_domain(d["positions"], d["layers"], stages)
        if (
            not isinstance(stages, list)
            or len(stages) != d["layers"]
            or any(not s or len(set(s)) != len(s) for s in stages)
        ):
            raise DiagnosticError("missing semantic boundary inventory")
        names = [
            f"p{p:09d}-l{layer:03d}-{s}"
            for p in d["positions"]
            for layer in range(d["layers"])
            for s in stages[layer]
        ]
        if [f["name"] for f in d["frames"]] != names:
            raise DiagnosticError("semantic boundary schedule has omissions")
    if [f["name"] for f in docs[0]["frames"]] != [f["name"] for f in docs[1]["frames"]]:
        raise DiagnosticError("different semantic boundary domains")
    output.mkdir(mode=0o700)
    first, count = None, 0
    for a, b in zip(docs[0]["frames"], docs[1]["frames"], strict=True):
        for root, entry in ((reference, a), (candidate, b)):
            if read_frame(root / entry["name"])["sha256"] != entry["sha256"]:
                raise DiagnosticError("semantic boundary changed")
        r = compare_frames(reference / a["name"], candidate / b["name"])
        count += 1
        if not r["equal"] and first is None:
            first = {"name": a["name"], "comparison": r}
    report = seal(
        {
            "schema": "urn:qwen:boundary-comparison:v1",
            "observations": count,
            "first_difference": first,
            "equal": first is None,
            "scope": "captured positions only; common decoder semantic boundaries",
            "native_equivalence": "UNPROVED",
        }
    )
    write_private(output / "report.json", report)
    return report
