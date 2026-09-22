"""Small exact-output and graph-timing sample for shared-KV M1 attention."""

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import private_json, seal, write_private


def qualify(args):
    spec = private_json(args.spec)
    os.environ.update(spec["environment"])
    import radiance_r4d_attn as native
    import torch
    from stock_m1_attention_shared import SharedM1Attention

    torch.set_num_threads(2)
    torch.manual_seed(19023)
    require(
        hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
        == spec["binding"]["files"]["radiance_r4d_attn.py"],
        "native attention binding changed",
    )
    candidate = SharedM1Attention(args.build)
    prefixes = (0, 9, 15, 505, 60000, 60009, 60015, 60409)
    blocks = (max(prefixes) + 8 + 15) // 16
    table = torch.zeros((1, (253792 + 15) // 16), device="cuda", dtype=torch.int32)
    table[0, :blocks] = torch.randperm(blocks, device="cuda", dtype=torch.int32)
    repeated = table.expand(8, -1).contiguous()
    guarded = torch.full((8 * 24 * 32 * 520 + 1024,), 0xA5, device="cuda", dtype=torch.uint8)
    scratch = guarded[512:-512]
    lengths = torch.empty((1,), device="cuda", dtype=torch.int32)
    row_lengths = torch.empty((8,), device="cuda", dtype=torch.int32)
    scales = [torch.full((4,), value, device="cuda", dtype=torch.float32) for value in (0.5, 1.5)]
    repeated_scales = [s.repeat(8) for s in scales]
    checks, timings = [], {}

    def mismatches(a, b):
        return int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))

    for dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.uint8):
        variant = int(dtype == torch.bfloat16)
        if dtype == torch.uint8:
            # vLLM stores FP8 as bytes; hybrid allocations can pad between pages.
            kv = torch.empty((blocks, 5, 16, 512), device="cuda", dtype=dtype)[:, :4]
            kv.copy_(torch.randn(kv.shape, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8))
        else:
            kv = torch.randn((blocks, 4, 16, 512), device="cuda").to(dtype)
        saved_kv = kv.clone()

        def launch(
            q,
            table_,
            lengths_,
            out,
            sequences,
            width,
            descales,
            *,
            function=native._DECODE[variant],
            kv=kv,
        ):
            function(
                q.data_ptr(),
                kv.data_ptr(),
                table_.data_ptr(),
                lengths_.data_ptr(),
                out.data_ptr(),
                descales[0].data_ptr(),
                descales[1].data_ptr(),
                scratch.data_ptr(),
                sequences,
                width,
                24,
                4,
                256,
                16,
                table_.shape[1],
                kv.stride(0),
                kv.stride(1),
                256**-0.5,
                32,
                253792,
                torch.cuda.current_stream().cuda_stream,
            )
            return out

        for prefix in prefixes:
            query = torch.randn((8, 24, 256), device="cuda").bfloat16()
            query[1:] *= 16
            lengths.fill_(prefix + 8)
            row_lengths.copy_(
                torch.arange(prefix + 1, prefix + 9, device="cuda", dtype=torch.int32)
            )
            oracle = torch.empty_like(query)
            for row in range(8):
                launch(
                    query[row : row + 1],
                    table,
                    row_lengths[row : row + 1],
                    oracle[row : row + 1],
                    1,
                    1,
                    scales,
                )
            old = launch(
                query, repeated, row_lengths, torch.empty_like(query), 8, 1, repeated_scales
            )
            out = candidate(query, kv, table, lengths, scratch, ks=scales[0], vs=scales[1])
            record = {
                "dtype": str(dtype),
                "prefix": prefix,
                "rows": 8,
                "elements": out.numel(),
                "baseline_mismatches": mismatches(old, oracle),
                "candidate_mismatches": mismatches(out, oracle),
            }
            checks.append(record)
            write_private(args.output / f"check-{dtype}-{prefix}.json", seal(record))
            require(
                record["baseline_mismatches"] == 0, "independent-query baseline differs from M1"
            )
            require(record["candidate_mismatches"] == 0, "shared-query attention differs from M1")
            changed = query.clone()
            changed[1:].zero_()
            shifted = candidate(changed, kv, table, lengths, scratch, ks=scales[0], vs=scales[1])
            require(mismatches(shifted[:1], out[:1]) == 0, "future query changed the first row")
            require(
                bool((guarded[:512] == 0xA5).all() and (guarded[-512:] == 0xA5).all()),
                "scratch canary changed",
            )

            if prefix != 60000:
                continue
            functions = {
                "independent_m1_queries": lambda query=query, launch=launch: launch(
                    query, repeated, row_lengths, torch.empty_like(query), 8, 1, repeated_scales
                ),
                "shared_kv_queries": lambda query=query, kv=kv: candidate(
                    query, kv, table, lengths, scratch, ks=scales[0], vs=scales[1]
                ),
            }
            graphs, samples = {}, {name: [] for name in functions}
            for name, function in functions.items():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        function()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = function()
                graph.replay()
                require(mismatches(result, oracle) == 0, "captured attention differs from M1")
                graphs[name] = (graph, result)
            rng = random.Random(3901)
            for _ in range(30):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[name][0].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(begin.elapsed_time(end))
            timings[str(dtype)] = {
                name: {"median_ms": statistics.median(values), "samples_ms": values}
                for name, values in samples.items()
            }
            del graphs
        require(torch.equal(kv.view(torch.uint8), saved_kv.view(torch.uint8)), "shared KV mutated")
        del kv, saved_kv
    broken = out.clone()
    broken.view(torch.int16)[0, 0, 0] ^= 1
    require(mismatches(broken, oracle) == 1, "negative control was not detected")
    return {
        "checks": checks,
        "timings": timings,
        "negative_control_detected": True,
        "build": candidate.manifest["sha256"],
        "graph_checks": True,
        "reference": "eight native M1 queries with the pinned 32-split graph contract",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "build", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(
        args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU admission required"
    )
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        result = qualify(args)
    report = seal(
        {
            "status": "SAMPLE_CHECKED",
            **result,
            "scope": "192 synthetic positions, 8 contexts, 3 KV layouts; not full-model proof",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "result.json", report)
    print(
        json.dumps(
            {"status": report["status"], "timings": report["timings"], "sha256": report["sha256"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
