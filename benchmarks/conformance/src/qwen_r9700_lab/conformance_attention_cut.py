"""Locate Qwen attention sub-boundaries by owners, tensor roles and checked wiring."""

import numpy as np

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate


def attention_cut(metadata, tensors, layer):
    authenticate(metadata)
    stem = f"language_model.model.layers.{layer}.self_attn."
    events = metadata["events"]
    rows = len(metadata["positions"])

    def owned(part):
        found = [
            e
            for e in events
            if any(o.startswith(stem + part + ".") for o in e.get("logical_identities", ()))
        ]
        require(len(found) == 1, "attention owner missing or ambiguous")
        return found[0]

    qkv, out, qnorm, knorm = [owned(s) for s in ("qkv_proj", "o_proj", "q_norm", "k_norm")]
    require(
        qkv["operation"] == out["operation"] == "radiance.mxfp4_linear.default",
        "unknown attention projection implementation",
    )
    middle = [e for e in events if qkv["index"] < e["index"] < out["index"]]
    attention = [
        e for e in middle if e["operation"] == "vllm.unified_attention_with_output.default"
    ]
    require(len(attention) == 1, "attention call missing or ambiguous between projections")
    attention = attention[0]
    require(
        qkv["index"] < qnorm["index"] < knorm["index"] < attention["index"] < out["index"],
        "unexpected attention cut order",
    )
    tail = [e for e in middle if e["index"] > attention["index"]]
    compiled = (
        len(tail) == 1
        and tail[0]["operation"] == "inductor/triton_poi_fused_mul_mxfp4_linear_sigmoid_view_0"
    )
    eager = [e["operation"] for e in tail] == ["aten.sigmoid.default", "aten.mul.Tensor"]
    require(compiled or eager, "unsupported attention gating sequence")

    def get(event, suffix, shape):
        key = f"{event['index']}.{suffix}"
        phase = suffix.split(".")[0]
        found = [r for r in event[phase] if r["key"] == key]
        require(len(found) == 1 and key in tensors, "missing attention tensor role")
        value = tensors[key]
        require(
            found[0]["dtype"] == "torch.bfloat16"
            and found[0]["shape"] == shape
            and list(value.shape) == shape
            and value.dtype == np.int16,
            "attention cut needs the pinned BF16 logical layout",
        )
        return value

    projection = get(qkv, "after.result", [rows, 14336])
    query_input = get(qnorm, "before.args.0", [rows, 24, 256])
    key_input = get(knorm, "before.args.0", [rows, 4, 256])
    query = get(qnorm, "after.result", [rows, 24, 256])
    key = get(knorm, "after.result", [rows, 4, 256])
    rotated_query = get(attention, "before.args.0", [rows, 24, 256])
    rotated_key = get(attention, "before.args.1", [rows, 4, 256])
    value = get(attention, "before.args.2", [rows, 4, 256])
    # args.3 is an output allocation before it has been written. Never compare
    # its prior contents as if they were a semantically meaningful model input.
    attended = get(attention, "after.mutable.output", [rows, 24, 256])
    gated = get(out, "before.args.0", [rows, 6144])
    if compiled:
        gate = get(tail[0], "before.args.1", [rows, 6144])
        gate_attention = get(tail[0], "before.args.0", [rows, 24, 256]).reshape(rows, 6144)
        gate_result = get(tail[0], "after.out_ptr0", [rows, 6144])
    else:
        gate = get(tail[0], "before.args.0", [rows, 6144])
        gate_attention = get(tail[1], "before.args.0", [rows, 6144])
        gate_result = get(tail[1], "after.result", [rows, 6144])
        require(
            np.array_equal(
                get(tail[0], "after.result", [rows, 6144]),
                get(tail[1], "before.args.1", [rows, 6144]),
            ),
            "sigmoid does not connect to attention multiplication",
        )
    packed_qg = projection[:, :12288].reshape(rows, 24, 512)
    for a, b in (
        (query_input, packed_qg[:, :, :256]),
        (gate, packed_qg[:, :, 256:].reshape(rows, 6144)),
        (key_input, projection[:, 12288:13312].reshape(rows, 4, 256)),
        (value, projection[:, 13312:].reshape(rows, 4, 256)),
        (gate_attention, attended.reshape(rows, 6144)),
        (gated, gate_result),
    ):
        require(np.array_equal(a, b), "attention tensors do not follow the declared wiring")
    return {
        "qkv_projection": projection,
        "query_after_normalization": query,
        "key_after_normalization": key,
        "query_after_rotation": rotated_query,
        "key_after_rotation": rotated_key,
        "value": value,
        "attention_output": attended,
        "gate_input": gate,
        "gated_attention_output": gated,
    }


def bf16_float(bits):
    """Decode BF16 storage exactly; the arithmetic oracle never initializes a GPU."""
    require(bits.dtype == np.int16, "expected BF16 storage")
    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_bf16(value, mode):
    """Explicit finite FP32 -> BF16 RNE/RTZ, including sign and tie-to-even."""
    require(mode in {"rne", "rtz"}, "unknown BF16 rounding mode")
    value = np.ascontiguousarray(value, dtype=np.float32)
    require(np.isfinite(value).all(), "non-finite arithmetic is outside this oracle")
    bits = value.view(np.uint32)
    if mode == "rne":
        bits = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    result = (bits >> 16).astype(np.uint16).view(np.int16)
    require(np.isfinite(bf16_float(result)).all(), "BF16 overflow is outside this oracle")
    return result


def rotary_formula(normalized, cosine, sine, product_rounding):
    """NeoX pairs with declared product rounding, RNE addition and unchanged tail."""
    require(
        normalized.ndim == 3
        and normalized.shape[2] == 256
        and normalized.shape[1] in {4, 24}
        and cosine.shape == sine.shape == (normalized.shape[0], 32),
        "unsupported pinned rotary layout",
    )
    q, c, s = bf16_float(normalized), bf16_float(cosine)[:, None, :], bf16_float(sine)[:, None, :]
    a, b = q[:, :, :32], q[:, :, 32:64]

    def product(x, y):
        return bf16_float(round_bf16(x * y, product_rounding))

    first = round_bf16(product(a, c) - product(b, s), "rne")
    second = round_bf16(product(b, c) + product(a, s), "rne")
    return np.concatenate((first, second, normalized[:, :, 64:]), axis=2)


def selected_rotary_coefficients(metadata, tensors, layer):
    """Read compiled selected coefficients, checking their input wiring to both rotations."""
    authenticate(metadata)
    owner = f"language_model.model.layers.{layer}.self_attn.rotary_emb.cos_sin_cache"
    norm_owner = f"language_model.model.layers.{layer}.self_attn.k_norm."
    norms = [
        e
        for e in metadata["events"]
        if any(o.startswith(norm_owner) for o in e.get("logical_identities", ()))
    ]
    require(len(norms) == 1, "missing unique K norm before rotary selection")
    norm_index = norms[0]["index"]
    consumers = [
        e
        for e in metadata["events"]
        if e["index"] > norm_index
        and e["operation"] == "vllm.unified_attention_with_output.default"
    ]
    require(consumers, "no attention consumer follows rotary selection")
    consumer_index = min(e["index"] for e in consumers)
    # Cache storage may be shared by multiple layers. Restrict ownership to the
    # validated K-norm -> attention interval, not every use of that allocation.
    found = [
        e
        for e in metadata["events"]
        if norm_index < e["index"] < consumer_index and owner in e.get("logical_identities", ())
    ]
    require(len(found) == 1, "missing unique compiled rotary coefficient selector")
    selector = found[0]
    require(selector["operation"].startswith("inductor/"), "coefficients need compiled capture")
    positions = metadata["positions"]
    coefficients = [tensors[f"{selector['index']}.after.out_ptr{i}"] for i in (0, 1)]
    require(
        all(x.shape == (len(positions), 32) and x.dtype == np.int16 for x in coefficients),
        "unsupported coefficient layout",
    )
    following = [
        e for e in metadata["events"] if selector["index"] < e["index"] <= selector["index"] + 2
    ]
    require(len(following) == 2, "missing rotations after coefficient selection")
    for event in following:
        require(event["operation"].startswith("inductor/"), "unknown rotation implementation")
        for i, expected in enumerate(coefficients, 1):
            require(
                np.array_equal(tensors[f"{event['index']}.before.args.{i}"], expected),
                "rotation uses different coefficient inputs",
            )
    return coefficients
