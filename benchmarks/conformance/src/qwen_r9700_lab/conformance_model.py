"""Standalone CPU reference for text-only dense Qwen3.5 MXFP4 checkpoints.

Uses checkpoint order directly, without native weight permutation, speculative
execution, fused kernels or a vLLM-generated initial state. Every token passes
through the serial finite-precision operators in conformance_reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import stat
import struct
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from qwen_r9700_lab import conformance_reference as ref
from qwen_r9700_lab.conformance_state import FrameWriter, load_arrays
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, integer

STORAGE_DTYPES = {
    "U8": "|u1",
    "I8": "|i1",
    "I32": "<i4",
    "I64": "<i8",
    "F32": "<f4",
    "F64": "<f8",
    "F16": "<f2",
    "BF16": "bf16",
}


class Checkpoint:
    """Read-only safetensors views; weights stay in their original checkpoint order."""

    def __init__(self, root: Path, expected_files: Mapping[str, str]):
        self.root, self.handles, self.maps, self.headers = root, {}, {}, {}
        self.file_identities = {}
        if not expected_files:
            raise DiagnosticError("checkpoint must be explicitly content-bound")
        for name, expected in expected_files.items():
            if Path(name).name != name:
                raise DiagnosticError("checkpoint manifest contains a nonlocal file")
            h = hashlib.sha256()
            with (root / name).open("rb") as f:
                before = os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise DiagnosticError("checkpoint must contain regular files")
                while chunk := f.read(8 * 1024 * 1024):
                    h.update(chunk)
                after = os.fstat(f.fileno())

            def identity(s):
                return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

            if identity(before) != identity(after):
                raise DiagnosticError("checkpoint changed while hashing")
            self.file_identities[name] = identity(after)
            if h.hexdigest() != expected:
                raise DiagnosticError("checkpoint artifact identity changed")
        index_name = "model.safetensors.index.json"
        if index_name not in expected_files or "config.json" not in expected_files:
            raise DiagnosticError("checkpoint index and configuration need content identities")
        self.index = json.loads((root / index_name).read_text())["weight_map"]
        self.config = json.loads((root / "config.json").read_text())
        if set(self.index.values()) - set(expected_files):
            raise DiagnosticError("checkpoint index references an unverified shard")
        self.identity = digest(dict(expected_files))

    def __contains__(self, name):
        return name in self.index

    def tensor(self, name, rows: slice | None = None):
        shard = self.index[name]
        s = (self.root / shard).stat()
        if self.file_identities[shard] != (
            s.st_dev,
            s.st_ino,
            s.st_size,
            s.st_mtime_ns,
            s.st_ctime_ns,
        ):
            raise DiagnosticError("checkpoint changed after artifact verification")
        if shard not in self.maps:
            handle = (self.root / shard).open("rb")
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            if len(mapped) < 8:
                raise DiagnosticError("truncated safetensors header")
            length = struct.unpack("<Q", mapped[:8])[0]
            if length > 64 * 1024 * 1024 or length + 8 > len(mapped):
                mapped.close()
                handle.close()
                raise DiagnosticError("invalid safetensors header")
            self.handles[shard], self.maps[shard] = handle, mapped
            self.headers[shard] = (8 + length, json.loads(mapped[8 : 8 + length]))
        offset, headers = self.headers[shard]
        descriptor = headers[name]
        dtype = STORAGE_DTYPES.get(descriptor["dtype"])
        if dtype is None:
            raise DiagnosticError("checkpoint dtype is outside this reference contract")
        storage_dtype = "<u2" if dtype == "bf16" else dtype
        begin, end = descriptor["data_offsets"]
        shape = descriptor["shape"]
        integer(begin)
        integer(end)
        if (
            not isinstance(shape, list)
            or any(integer(v) < 0 for v in shape)
            or end < begin
            or offset + end > len(self.maps[shard])
        ):
            raise DiagnosticError("checkpoint tensor is outside its shard")
        if end - begin != math.prod(shape) * np.dtype(storage_dtype).itemsize:
            raise DiagnosticError("checkpoint tensor size mismatch")
        array = np.ndarray(
            shape, dtype=storage_dtype, buffer=self.maps[shard], offset=offset + begin
        )
        if rows is not None:
            array = array[rows]
        if dtype == "bf16":
            return (array.astype(np.uint32) << 16).view(np.float32)
        return np.array(array, copy=True)

    def close(self):
        for value in self.maps.values():
            value.close()
        for value in self.handles.values():
            value.close()
        self.maps.clear()
        self.handles.clear()


def state_names(config: Mapping) -> list[str]:
    names = ["sequence.tokens", "sequence.position"]
    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"layer.{layer:03d}."
        names.extend(
            prefix + suffix
            for suffix in (
                ("gdn", "conv") if kind == "linear_attention" else ("keys", "values", "kv_scales")
            )
        )
    return names


class QuantizedQwenReference:
    def __init__(
        self,
        checkpoint,
        *,
        kv_scales: Mapping[str, list[float]],
        contract: str,
        execution: str,
        adapter: str,
        capture: Callable | None = None,
        reference_profile: str = "radiance-fp8",
    ):
        self.weights = checkpoint
        self.precision = ref.reference_precision(reference_profile)
        self.config = checkpoint.config.get("text_config", checkpoint.config)
        c = self.config
        if c["model_type"] != "qwen3_5_text" or c.get("layer_scale", False):
            raise DiagnosticError("reference supports dense text-only Qwen3.5 without layer scales")
        if c.get("hidden_act", "silu") != "silu" or c.get("attention_bias", False):
            raise DiagnosticError("unsupported activation or attention bias")
        if c.get("rope_parameters", {}).get("rope_type", "default") != "default":
            raise DiagnosticError("reference requires the default text rotary contract")
        if len(c["layer_types"]) != c["num_hidden_layers"] or any(
            k not in {"linear_attention", "full_attention"} for k in c["layer_types"]
        ):
            raise DiagnosticError("unsupported layer inventory")
        kv_scales = ref.kv_scaling(c, dict(kv_scales), reference_profile)
        self.kv_scales = {k: np.asarray(v, dtype=np.float32) for k, v in kv_scales.items()}
        self.contract, self.execution, self.adapter = contract, execution, adapter
        self.capture = capture
        self._linear = ref.linear
        self.tokens, self.state = [], {}
        self.prefix = "model.language_model."
        self.reset()

    def observe(self, position, layer, name, value):
        if self.capture is not None:
            self.capture(position, layer, name, np.array(value, copy=True))

    def reset(self):
        c = self.config
        self.tokens, self.state = [], {}
        for layer, kind in enumerate(c["layer_types"]):
            if kind == "linear_attention":
                heads, kd, vd = (
                    c["linear_num_value_heads"],
                    c["linear_key_head_dim"],
                    c["linear_value_head_dim"],
                )
                channels = 2 * c["linear_num_key_heads"] * kd + heads * vd
                self.state[layer] = {
                    "gdn": np.zeros((heads, vd, kd), dtype=np.float32),
                    "conv": np.zeros((channels, c["linear_conv_kernel_dim"] - 1), dtype=np.float32),
                }
            else:
                shape = (0, c["num_key_value_heads"], c["head_dim"])
                self.state[layer] = {
                    "keys": np.empty(shape, dtype="u1" if self.precision["kv_fp8"] else "<u2"),
                    "values": np.empty(shape, dtype="u1" if self.precision["kv_fp8"] else "<u2"),
                    "kv_scales": self.kv_scales[str(layer)].copy(),
                }

    def project(self, name, x):
        weight_name = name + ".weight"
        scale_name = name + ".weight_scale"
        if scale_name not in self.weights:
            return self._linear(x, self.weights.tensor(weight_name))
        # A row tile bounds the temporary dequantized weight allocation.
        packed = self.weights.tensor(weight_name)
        scales = self.weights.tensor(scale_name)
        outputs = []
        if self.precision["activation_fp8"]:
            code, xs = ref.activation_quantize(x)
            linear_input = ref.fp8_decode(code) * xs
        else:
            linear_input = ref.bf16(x)
        for start in range(0, packed.shape[0], 256):
            w = ref.unpack_mxfp4(packed[start : start + 256], scales[start : start + 256])
            outputs.append(self._linear(linear_input, w))
        return np.concatenate(outputs, axis=-1)

    def step(self, token: int):
        integer(token)
        c, position = self.config, len(self.tokens)
        if token >= c["vocab_size"] or position >= c["max_position_embeddings"]:
            raise DiagnosticError("token or position is outside the model contract")
        x = self.weights.tensor(self.prefix + "embed_tokens.weight", slice(token, token + 1))[0]
        x = ref.bf16(x)
        for layer, kind in enumerate(c["layer_types"]):
            base = self.prefix + f"layers.{layer}."
            self.observe(position, layer, "input", x)
            normed = ref.rms_norm(
                x,
                self.weights.tensor(base + "input_layernorm.weight"),
                c["rms_norm_eps"],
                weight_offset=1,
            )
            self.observe(position, layer, "input_norm", normed)
            if kind == "linear_attention":
                out = self.gdn(layer, base + "linear_attn.", normed, position)
            else:
                out = self.attention(layer, base + "self_attn.", normed, position)
            x = ref.bf16(np.add(x, out, dtype=np.float32))
            self.observe(position, layer, "attention_residual", x)
            normed = ref.rms_norm(
                x,
                self.weights.tensor(base + "post_attention_layernorm.weight"),
                c["rms_norm_eps"],
                weight_offset=1,
            )
            self.observe(position, layer, "post_attention_norm", normed)
            gate = self.project(base + "mlp.gate_proj", normed)
            up = self.project(base + "mlp.up_proj", normed)
            self.observe(position, layer, "mlp_gate", gate)
            self.observe(position, layer, "mlp_up", up)
            intermediate = ref.bf16(ref.silu(gate) * up)
            self.observe(position, layer, "mlp_intermediate", intermediate)
            down = self.project(base + "mlp.down_proj", intermediate)
            self.observe(position, layer, "mlp_down", down)
            x = ref.bf16(x + down)
            self.observe(position, layer, "output", x)
        self.tokens.append(token)
        hidden = ref.rms_norm(
            x, self.weights.tensor(self.prefix + "norm.weight"), c["rms_norm_eps"], weight_offset=1
        )
        logits = self.project("lm_head", hidden).astype(np.float32)
        self.observe(position, -1, "logits", logits)
        if not np.isfinite(logits).all():
            raise DiagnosticError("nonfinite reference logits")
        return logits

    def gdn(self, layer, base, x, position):
        c, state = self.config, self.state[layer]
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = self.project(base + "in_proj_qkv", x)
        z = self.project(base + "in_proj_z", x).reshape(vh, vd)
        b, a = self.project(base + "in_proj_b", x), self.project(base + "in_proj_a", x)
        for name, value in (
            ("gdn_qkv", qkv),
            ("gdn_z", z),
            ("gdn_b", b),
            ("gdn_a", a),
            ("conv_input_state", state["conv"]),
        ):
            self.observe(position, layer, name, value)
        conv_weight = self.weights.tensor(base + "conv1d.weight").reshape(qkv.size, -1)
        conv, state["conv"] = ref.convolution_step(qkv, state["conv"], conv_weight)
        self.observe(position, layer, "conv_output", conv)
        self.observe(position, layer, "conv_state", state["conv"])
        q, k, v = np.split(conv, (kh * kd, 2 * kh * kd))
        q, k, v = q.reshape(kh, kd), k.reshape(kh, kd), v.reshape(vh, vd)
        q, k = (
            ref.bf16(t / np.sqrt(ref.ordered_sum(t * t)[..., None] + np.float32(1e-6)))
            for t in (q, k)
        )
        decay = -np.exp(self.weights.tensor(base + "A_log").astype(np.float32)) * ref.softplus(
            a + self.weights.tensor(base + "dt_bias")
        )
        beta = ref.bf16(ref.sigmoid(b))
        for name, value in (
            ("gdn_q", q),
            ("gdn_k", k),
            ("gdn_v", v),
            ("gdn_decay", decay),
            ("gdn_beta", beta),
            ("gdn_input_state", state["gdn"]),
        ):
            self.observe(position, layer, name, value)
        out, state["gdn"] = ref.gdn_step(q, k, v, decay, beta, state["gdn"])
        self.observe(position, layer, "gdn_output", out)
        self.observe(position, layer, "gdn_state", state["gdn"])
        gated = ref.rms_norm(
            out, self.weights.tensor(base + "norm.weight"), c["rms_norm_eps"], output_bf16=False
        ) * ref.silu(z)
        self.observe(position, layer, "gdn_gated", gated)
        projected = self.project(base + "out_proj", ref.bf16(gated).reshape(-1))
        self.observe(position, layer, "attention_projected", projected)
        return projected

    def attention(self, layer, base, x, position):
        c, state = self.config, self.state[layer]
        heads, kv_heads, width = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q_raw = self.project(base + "q_proj", x)
        gate = None
        if c.get("attn_output_gate", True):
            q, gate = np.split(q_raw.reshape(heads, width * 2), 2, axis=-1)
        else:
            q = q_raw.reshape(heads, width)
        k, v = (
            self.project(base + name + "_proj", x).reshape(kv_heads, width) for name in ("k", "v")
        )
        for name, value in (
            ("attention_q_projected", q),
            ("attention_k_projected", k),
            ("attention_v_projected", v),
        ):
            self.observe(position, layer, name, value)
        q, k = (
            ref.rms_norm(
                t,
                self.weights.tensor(base + name + "_norm.weight"),
                c["rms_norm_eps"],
                weight_offset=1,
            )
            for name, t in (("q", q), ("k", k))
        )
        rope = c["rope_parameters"]
        self.observe(position, layer, "attention_q_norm", q)
        self.observe(position, layer, "attention_k_norm", k)
        rotary_dim = int(
            width * rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 1))
        )
        q, k = (ref.rope(t, position, rotary_dim, rope["rope_theta"]) for t in (q, k))
        self.observe(position, layer, "attention_q_rope", q)
        self.observe(position, layer, "attention_k_rope", k)

        def encode(value, scale):
            if self.precision["kv_fp8"]:
                return ref.fp8_encode(value / scale)
            return (ref.bf16(value).view("<u4") >> 16).astype("<u2")

        def decode(value, scale):
            if self.precision["kv_fp8"]:
                return ref.fp8_decode(value) * scale
            return (value.astype("<u4") << 16).view("<f4")

        state["keys"] = np.concatenate((state["keys"], encode(k, state["kv_scales"][0])[None]))
        state["values"] = np.concatenate((state["values"], encode(v, state["kv_scales"][1])[None]))
        self.observe(position, layer, "attention_key_stored", state["keys"][-1])
        self.observe(position, layer, "attention_value_stored", state["values"][-1])
        out = ref.dense_attention(
            q,
            decode(state["keys"], state["kv_scales"][0]),
            decode(state["values"], state["kv_scales"][1]),
        )
        self.observe(position, layer, "attention_output", out)
        if gate is not None:
            out = ref.bf16(out * ref.sigmoid(gate))
        projected = self.project(base + "o_proj", out.reshape(-1))
        self.observe(position, layer, "attention_projected", projected)
        return projected

    def frame(self, root: Path, *, phase: str, pending: int | None = None):
        writer = FrameWriter(
            root,
            contract=self.contract,
            execution=self.execution,
            adapter=self.adapter,
            input_digest=digest(self.tokens),
            phase=phase,
            consumed=len(self.tokens),
            pending=pending,
            expected=state_names(self.config),
        )
        writer.array("sequence.tokens", np.asarray(self.tokens, dtype="<i4"))
        writer.array("sequence.position", np.asarray([len(self.tokens)], dtype="<i8"))
        for layer, state in self.state.items():
            for name, value in state.items():
                if name in {"keys", "values"}:
                    writer.add(
                        f"layer.{layer:03d}.{name}",
                        value.tobytes(),
                        dtype=self.precision["kv_encoding"],
                        shape=value.shape,
                    )
                else:
                    writer.array(f"layer.{layer:03d}.{name}", value)
        return writer.finish()

    def restore(self, root: Path):
        document, arrays = load_arrays(root)
        if document["contract"] != self.contract or document["coverage"] != state_names(
            self.config
        ):
            raise DiagnosticError("snapshot is from a different reference contract")
        if (
            arrays["sequence.tokens"].dtype != np.dtype("<i4")
            or arrays["sequence.tokens"].ndim != 1
        ):
            raise DiagnosticError("invalid snapshot token representation")
        tokens = arrays.pop("sequence.tokens").tolist()
        if (
            arrays.pop("sequence.position").tolist() != [len(tokens)]
            or len(tokens) != document["consumed"]
        ):
            raise DiagnosticError("snapshot position does not match materialized tokens")
        if digest(tokens) != document["input_digest"]:
            raise DiagnosticError("snapshot prefix identity mismatch")
        if len(tokens) > self.config["max_position_embeddings"] or any(
            type(t) is not int or t < 0 or t >= self.config["vocab_size"] for t in tokens
        ):
            raise DiagnosticError("snapshot tokens exceed reference domain")
        state = {}
        for layer, kind in enumerate(self.config["layer_types"]):
            names = (
                ("gdn", "conv") if kind == "linear_attention" else ("keys", "values", "kv_scales")
            )
            state[layer] = {name: arrays[f"layer.{layer:03d}.{name}"] for name in names}
            for name, value in state[layer].items():
                expected = self.state[layer][name]
                if name in {"keys", "values"}:
                    encoding = document["components"][f"layer.{layer:03d}.{name}"]["dtype"]
                    if encoding != self.precision["kv_encoding"]:
                        raise DiagnosticError("snapshot KV encoding differs from the reference")
                    if not np.isfinite(ref.values(value.tobytes(), encoding)).all():
                        raise DiagnosticError("snapshot contains nonfinite KV state")
                shape = (
                    (len(tokens), *expected.shape[1:])
                    if name in {"keys", "values"}
                    else expected.shape
                )
                if (
                    value.dtype != expected.dtype
                    or value.shape != shape
                    or not np.isfinite(value).all()
                ):
                    raise DiagnosticError("snapshot state geometry or precision changed")
            if kind == "full_attention" and not np.array_equal(
                state[layer]["kv_scales"], self.kv_scales[str(layer)]
            ):
                raise DiagnosticError("snapshot KV quantizer scales changed")
        self.tokens, self.state = tokens, state

    def close(self):
        self.weights.close()
