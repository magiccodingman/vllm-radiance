"""Qualify the isolated causal GDN candidate on synthetic, authenticated captures.

The fixture manifest is a synthetic full-model capture, never a Pi transcript.
Compare raw-input partitions, causal prefixes, native M1 state, memory guards,
and empty sequences. Numerical evidence is scoped to this build and these inputs.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
from pathlib import Path

from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
from qwen_r9700_lab.conformance_state import read_frame
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private

CAPTURE_SHA = "a1ee09adc7baeee1f04a577ffba20767c4a532a1239d6a7c9ba169b967fd6f30"


def load_tensor(root, row, key):
    import torch

    directory = root / f"call-{row['index']:09}" / "before"
    frame = read_frame(directory)
    if frame["sha256"] != row["before"]["frame"]:
        raise ValueError("frame identity mismatch")
    component = frame["components"][key]
    raw = (directory / component["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != component["sha256"]:
        raise ValueError("tensor identity mismatch")
    desc = row["before"]["descriptors"][key]
    return (
        torch.frombuffer(bytearray(raw), dtype=getattr(torch, desc["dtype"].removeprefix("torch.")))
        .reshape(desc["shape"])
        .clone()
    )


def captured(root, layer):
    import torch

    manifest = private_json(root / "calls.json")
    authenticate(manifest)
    if manifest["sha256"] != CAPTURE_SHA:
        raise ValueError("expected synthetic fixture manifest")
    rows = {row["index"]: row for row in manifest["calls"]}
    selected = []
    for row in rows.values():
        if row["site"] != "radiance_gdn.conv_prep" or row["logical"]["phase"] != "prefill":
            continue
        parent = row
        while (
            parent and parent["site"] != f"target.language_model.model.layers.{layer}.linear_attn"
        ):
            parent = rows.get(parent["parent"])
        if parent:
            selected.append(row)
    selected.sort(key=lambda row: row["logical"]["positions"][0])
    if [row["logical"]["positions"][0] for row in selected] != [0, 1600]:
        raise ValueError("unexpected captured partition domain")
    fixture = {
        name: torch.cat([load_tensor(root, row, key) for row in selected])
        for name, key in [("x", "args.0"), ("a", "args.6"), ("b", "args.7")]
    }
    fixture.update(
        {
            name: load_tensor(root, selected[0], key)
            for name, key in [("weight", "args.1"), ("alog", "args.8"), ("dt", "args.9")]
        }
    )
    return fixture, {
        "layer": layer,
        "manifest_record_sha256": CAPTURE_SHA,
        "calls": [row["index"] for row in selected],
    }


def exact(a, b):
    import torch

    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def comparison(a, b):
    import torch

    return {
        "exact_bytes": exact(a, b),
        "different_elements": int(torch.count_nonzero(a != b)),
        "maximum_absolute": float((a.float() - b.float()).abs().max()) if a.numel() else 0.0,
    }


def classify(rows):
    required = {"partition", "prefix", "native_recurrent", "future_gates", "empty_sequence"}
    if not rows or not required.issubset({row.get("kind") for row in rows}):
        return "INCOMPLETE"
    if any(row.get("controls_ok") is not True for row in rows):
        return "INVALID_CONTROL"
    return "TESTED" if all(row.get("exact") is True for row in rows) else "DISCREPANCY"


class Candidate:
    def __init__(self, library, native, prepared_qk=False):
        self.library = ctypes.CDLL(str(library))
        self.conv = self.library.qwen_gdn_conv_raw
        pointer, long, integer = ctypes.c_void_p, ctypes.c_long, ctypes.c_int
        self.conv.argtypes = [
            pointer,
            long,
            pointer,
            pointer,
            pointer,
            long,
            long,
            long,
            pointer,
            long,
            pointer,
            pointer,
            pointer,
            long,
            integer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            *([integer] * 7),
            ctypes.c_float,
            pointer,
        ]
        self.conv.restype = ctypes.c_int
        self.prepared_qk = prepared_qk
        self.scan = getattr(
            self.library,
            "qwen_gdn_causal_prefill_prepared" if prepared_qk else "qwen_gdn_causal_prefill",
        )
        self.scan.argtypes = (
            [ctypes.c_void_p] * 9 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        )
        self.scan.restype = ctypes.c_int
        if prepared_qk:
            self.normalize = self.library.qwen_gdn_normalize_qk
            self.normalize.argtypes = [pointer] * 4 + [integer, ctypes.c_float, pointer]
            self.normalize.restype = ctypes.c_int

    def normalized(self, values, outputs=None, scale=128**-0.5):
        import torch

        if not self.prepared_qk:
            return values
        q, k, *rest = values
        if outputs is None:
            outputs = tuple(torch.empty_like(t, dtype=torch.float32) for t in (q, k))
        status = self.normalize(
            *[t.data_ptr() for t in (q, k, *outputs)],
            q.numel() // 128,
            float(scale),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"normalization failed: {status}")
        return (*outputs, *rest)

    def prepare(self, fixture, begin, end, conv_state, initialized):
        import torch

        # Keep row strides: gate projections are often two views of a packed BA tensor.
        x, a, b = [fixture[key][begin:end] for key in ("x", "a", "b")]
        length, heads, qheads = end - begin, 48, 16
        q, k = [
            torch.empty((length, qheads, 128), device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.empty((length, heads, 128), device="cuda", dtype=torch.bfloat16)
        g, beta = [
            torch.empty((length, heads), device="cuda", dtype=torch.float32) for _ in range(2)
        ]
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        idx = torch.tensor([1], device="cuda", dtype=torch.int32)
        init = torch.tensor([initialized], device="cuda", dtype=torch.bool)
        status = self.conv(
            x.data_ptr(),
            x.stride(0),
            fixture["weight"].data_ptr(),
            0,
            conv_state.data_ptr(),
            *conv_state.stride(),
            idx.data_ptr(),
            1,
            init.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
            a.stride(0),
            a.dtype == torch.bfloat16,
            fixture["alog"].data_ptr(),
            fixture["dt"].data_ptr(),
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            g.data_ptr(),
            beta.data_ptr(),
            cu.data_ptr(),
            1,
            length,
            heads,
            qheads,
            128,
            128,
            4,
            20.0,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"raw convolution failed: {status}")
        return q, k, v, g, beta

    def recur(self, values, initial, cu=None):
        import torch

        q, k, v, g, beta = self.normalized(values)
        if cu is None:
            cu = torch.tensor([0, len(v)], device="cuda", dtype=torch.int32)
        slab = torch.full((v.numel() + 512,), 37, device="cuda", dtype=torch.bfloat16)
        output = slab[256:-256].view_as(v)
        state_slab = torch.full((initial.numel() + 512,), 37, device="cuda", dtype=torch.float32)
        final = state_slab[256:-256].view_as(initial)
        before = initial.clone()
        status = self.scan(
            *[t.data_ptr() for t in (q, k, v, g, beta, initial, output, final, cu)],
            len(cu) - 1,
            48,
            16,
            128**-0.5,
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        if status:
            raise RuntimeError(f"causal recurrence failed: {status}")
        controls = all(
            bool((t == 37).all())
            for t in (slab[:256], slab[-256:], state_slab[:256], state_slab[-256:])
        )
        controls &= exact(before, initial) and bool(torch.isfinite(output).all())
        controls &= bool(torch.isfinite(final).all())
        if not controls:
            raise RuntimeError("guard, initial-state, or finite-value control failed")
        return output, final


def run(args):
    # Torch must load its bundled ROCm runtime before either native extension.
    import torch  # isort: skip

    import r4d
    import radiance_gdn as native

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    build = private_json(args.build / "build.json")
    authenticate(build)
    binary = args.build / "candidate.so"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != build["binary_sha256"]:
        raise ValueError("candidate library hash mismatch")
    fixtures = [(f"captured_layer_{layer}", *captured(args.capture, layer)) for layer in (0, 1)]
    rows, timing = [], []
    with gpu_lease(args.output / "gpu-lease"):
        candidate = Candidate(binary, native, args.prepared_qk)
        for name, cpu, binding in fixtures:
            fixture = {key: value.cuda() for key, value in cpu.items()}
            length = len(fixture["x"])
            initial = torch.zeros((1, 48, 128, 128), device="cuda", dtype=torch.float32)
            conv_initial = torch.zeros((2, 10240, 3), device="cuda", dtype=torch.bfloat16)
            conv = conv_initial.clone()
            prepared = candidate.prepare(fixture, 0, length, conv, False)
            output, final = candidate.recur(prepared, initial)
            for widths in (
                [1600, 449],
                [1664, 385],
                [1568, 481],
                [1648, 401],
                [1, 64, 128, 1856],
                [64] * 32 + [1],
            ):
                state, history, outputs, begin = initial.clone(), conv_initial.clone(), [], 0
                producer_exact = True
                for width in widths:
                    values = candidate.prepare(fixture, begin, begin + width, history, begin != 0)
                    producer_exact &= all(
                        exact(part, full[begin : begin + width])
                        for part, full in zip(values, prepared, strict=True)
                    )
                    part, state = candidate.recur(values, state)
                    outputs.append(part)
                    begin += width
                out_cmp, state_cmp = (
                    comparison(torch.cat(outputs), output),
                    comparison(state, final),
                )
                rows.append(
                    {
                        "fixture": name,
                        "kind": "partition",
                        "widths": widths,
                        "producer_exact": producer_exact,
                        "output": out_cmp,
                        "state": state_cmp,
                        "controls_ok": begin == length,
                        "exact": producer_exact
                        and exact(history, conv)
                        and out_cmp["exact_bytes"]
                        and state_cmp["exact_bytes"],
                    }
                )
            for end in (1, 32, 48, 63, 64, 65, 129, 257):
                history = conv_initial.clone()
                values = candidate.prepare(fixture, 0, end, history, False)
                short, state = candidate.recur(values, initial)
                prefix_ok = all(
                    exact(part, full[:end]) for part, full in zip(values, prepared, strict=True)
                )
                # Native recurrence stores each processed token to slot 1. This exercises
                # its published state layout without allocating one long state per token.
                bank = torch.zeros((2, 48, 128, 128), device="cuda", dtype=torch.float32)
                native_out = torch.empty_like(short)
                cu = torch.tensor([0, end], device="cuda", dtype=torch.int32)
                sidx = torch.ones((1, end), device="cuda", dtype=torch.int32)
                native.recurrent_update(
                    *values[:3],
                    fixture["a"][:end],
                    fixture["b"][:end],
                    fixture["alog"],
                    fixture["dt"],
                    bank,
                    native_out,
                    cu,
                    sidx,
                    None,
                    1,
                    48,
                    16,
                    128**-0.5,
                )
                torch.cuda.synchronize()
                rows.append(
                    {
                        "fixture": name,
                        "kind": "prefix",
                        "tokens": end,
                        "controls_ok": prefix_ok,
                        "exact": exact(short, output[:end]),
                    }
                )
                rows.append(
                    {
                        "fixture": name,
                        "kind": "native_recurrent",
                        "tokens": end,
                        "output": comparison(short, native_out),
                        "state": comparison(state[0], bank[1]),
                        "controls_ok": True,
                        "exact": exact(short, native_out) and exact(state[0], bank[1]),
                    }
                )
            changed = {**fixture, "a": fixture["a"].clone(), "b": fixture["b"].clone()}
            changed["a"][32:] = 20
            changed["b"][32:] = -20
            changed_values = candidate.prepare(changed, 0, length, conv_initial.clone(), False)
            changed_output, _ = candidate.recur(changed_values, initial)
            rows.append(
                {
                    "fixture": name,
                    "kind": "future_gates",
                    "tokens": 32,
                    "controls_ok": not exact(changed_values[3][32:], prepared[3][32:]),
                    "exact": exact(changed_output[:32], output[:32]),
                }
            )
            # Resume from nonzero convolution/recurrent state with strided FP32 gates.
            # This exercises the other gate ABI and a real M1 convolution oracle.
            rng = torch.Generator().manual_seed(938)
            history = (torch.randn((2, 10240, 3), generator=rng) * 0.01).to(
                device="cuda", dtype=torch.bfloat16
            )
            nonzero = (torch.randn((1, 48, 128, 128), generator=rng) * 0.001).cuda()
            gate_storage = torch.zeros((65, 103), device="cuda", dtype=torch.float32)
            gate_storage[:, :48] = fixture["a"][:65]
            gate_storage[:, 48:96] = fixture["b"][:65]
            strided = {**fixture, "a": gate_storage[:, :48], "b": gate_storage[:, 48:96]}
            prefix_history = history.clone()
            resumed_values = candidate.prepare(strided, 0, 65, prefix_history, True)
            resumed_out, resumed_state = candidate.recur(resumed_values, nonzero)
            native_history = history.clone()
            decoded = []
            idx = torch.ones(1, device="cuda", dtype=torch.int32)
            single_cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
            for token in range(65):
                decoded.append(  # noqa: PERF401 - ordered calls mutate the same convolution state
                    native.conv_update(
                        fixture["x"][token : token + 1],
                        fixture["weight"],
                        None,
                        native_history,
                        3,
                        idx,
                        None,
                        single_cu,
                        1,
                        1,
                        48,
                        16,
                        1,
                    )
                )
            rows.append(
                {
                    "fixture": name,
                    "kind": "native_convolution",
                    "controls_ok": True,
                    "exact": exact(native_history, prefix_history)
                    and all(
                        exact(torch.cat([v[i] for v in decoded]), resumed_values[i])
                        for i in range(3)
                    ),
                }
            )
            bank = torch.zeros((2, 48, 128, 128), device="cuda", dtype=torch.float32)
            bank[1] = nonzero[0]
            native_out = torch.empty_like(resumed_out)
            resume_cu = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
            resume_idx = torch.ones((1, 65), device="cuda", dtype=torch.int32)
            native.recurrent_update(
                *resumed_values[:3],
                strided["a"],
                strided["b"],
                fixture["alog"],
                fixture["dt"],
                bank,
                native_out,
                resume_cu,
                resume_idx,
                None,
                1,
                48,
                16,
                128**-0.5,
            )
            rows.append(
                {
                    "fixture": name,
                    "kind": "native_recurrent",
                    "tokens": 65,
                    "nonzero_initial_state": True,
                    "strided_fp32_gates": True,
                    "controls_ok": strided["a"].stride(0) == 103,
                    "output": comparison(resumed_out, native_out),
                    "state": comparison(resumed_state[0], bank[1]),
                    "exact": exact(resumed_out, native_out) and exact(resumed_state[0], bank[1]),
                }
            )
            # Three sequences, including an empty middle sequence and nonzero initial state.
            many = initial.expand(3, -1, -1, -1).clone()
            many[1].fill_(0.125)
            cu = torch.tensor([0, 63, 63, length], device="cuda", dtype=torch.int32)
            multiple, multiple_state = candidate.recur(prepared, many, cu)
            independently = [
                candidate.recur(tuple(t[a:b].contiguous() for t in prepared), many[i : i + 1])
                for i, (a, b) in enumerate(((0, 63), (63, 63), (63, length)))
            ]
            rows.append(
                {
                    "fixture": name,
                    "kind": "empty_sequence",
                    "controls_ok": True,
                    "exact": exact(multiple_state[1], many[1])
                    and exact(multiple, torch.cat([p[0] for p in independently]))
                    and exact(multiple_state, torch.cat([p[1] for p in independently])),
                }
            )
            samples = []
            timing_cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            timing_values = candidate.normalized(prepared)
            for _ in range(12):
                start, finish = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                # Measure the recurrence itself; allocation/guards are outside the timed region.
                if args.prepared_qk:
                    candidate.normalized(prepared, timing_values[:2])
                status = candidate.scan(
                    *[
                        t.data_ptr()
                        for t in (
                            *timing_values,
                            initial,
                            output,
                            final,
                            timing_cu,
                        )
                    ],
                    1,
                    48,
                    16,
                    128**-0.5,
                    torch.cuda.current_stream().cuda_stream,
                )
                finish.record()
                finish.synchronize()
                if status:
                    raise RuntimeError(f"timed recurrence failed: {status}")
                samples.append(start.elapsed_time(finish))
            timing.append(
                {
                    "fixture": name,
                    "tokens": length,
                    "normalization_and_scan_median_ms": statistics.median(samples[2:]),
                    "binding": binding,
                }
            )
            write_private(args.output / f"{name}.json", seal({"rows": rows, "timing": timing}))
        report = seal(
            {
                "status": classify(rows),
                "rows": rows,
                "timing": timing,
                "build": build["sha256"],
                "binary": build["binary_sha256"],
                "r4d_sha256": hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest(),
                "native_wrapper_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()
                ).hexdigest(),
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prepared_fp32_qk": args.prepared_qk,
                "scope": (
                    "Two synthetic captured layers; causal raw preparation and recurrence. "
                    "Not a whole-model proof."
                ),
            }
        )
        write_private(args.output / "result.json", report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "comparisons": len(rows),
                    "exact": sum(row["exact"] for row in rows),
                    "timing": timing,
                    "result_sha256": report["sha256"],
                }
            ),
            flush=True,
        )
    return 0 if report["status"] == "TESTED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true", required=True)
    parser.add_argument("--prepared-qk", action="store_true")
    raise SystemExit(run(parser.parse_args()))
