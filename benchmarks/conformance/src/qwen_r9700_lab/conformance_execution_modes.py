"""Admit a controlled execution-mode comparison, never a universal correctness claim."""

from qwen_r9700_lab.conformance_topk import aggregate, compare_saved_rows, require
from qwen_r9700_lab.diagnostic_contract import authenticate, digest, seal

MODES = {"compiled", "compiled-no-graphs", "eager"}
CAPACITY = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "block_size",
    "cache_dtype",
    "num_gpu_blocks",
}
SOURCES = {
    "optimized_d7_worker.py",
    "optimized_stock_norm.py",
    "execution_mode_d7_worker.py",
    "isolated_d7_capture.py",
}


def numerical_config(config):
    # Everything except these explicitly varied execution controls must match.
    # In particular keep request capacity, prefill chunk size, KV, seed, model,
    # speculation and sampling defaults. Do not blacklist unknown future fields.
    result = {
        k: v
        for k, v in config.items()
        if k
        not in {
            "sha256",
            "enforce_eager",
            "compilation_config",
            "worker_cls",
        }
    }
    result["compilation_config"] = {
        k: v
        for k, v in config.get("compilation_config", {}).items()
        if k not in {"mode", "cudagraph_mode", "cudagraph_capture_sizes"}
    }
    return result


def admit_pair(left, right, *, arm="m8"):
    """Each side supplies authenticated measurement, requested config and runtime."""
    require(arm in {"m1", "m8"}, "unknown execution-mode comparison arm")
    for side in (left, right):
        for key in ("measurement", "config", "runtime", "pass"):
            authenticate(side[key])
        m, c, r = (side[k] for k in ("measurement", "config", "runtime"))
        mode = m.get("execution_mode")
        require(mode in MODES, "missing or invalid execution-mode identity")
        require(
            m["correctness"] and m["m1"] == (arm == "m1"),
            f"comparison requires forced {arm.upper()} replay on both sides",
        )
        require(m["lanes"] == ["fixed-bf16"], "comparison requires the same final fixed lane")
        eager = mode == "eager"
        require(c["enforce_eager"] == r["enforce_eager"] == eager, "eager mode was not honored")
        require((r["compilation_mode"] == 0) == eager, "unexpected compiler mode")
        graph = "NONE" if mode != "compiled" or m["isolated_capture"] else "PIECEWISE"
        require(str(r["graph_mode"]).split(".")[-1] == graph, "unexpected graph execution")
        require(r["async_scheduling"] is False, "comparison requires ordered scheduling")
        p = side["pass"]
        require(
            p["mode"] == "correctness"
            and p["execution_mode"] == mode
            and p["fixture"] == m["fixture"],
            "pass receipt does not match the requested execution",
        )
        counts = p["observation"]["observation"]["counts"]
        require(counts.get("target_forward_calls", 0) > 0, "no target execution observed")
        require(
            (counts.get("target_graph_replays", 0) > 0) == (graph == "PIECEWISE"),
            "observed graph execution contradicts configuration",
        )
        require(set(r.get("effective_capacity", {})) == CAPACITY, "actual capacity not recorded")
        require(all(v is not None for v in r["effective_capacity"].values()), "unknown capacity")
        require(set(r.get("diagnostic_sources", {})) == SOURCES, "unbound integration sources")
        require((r.get("repair") or {}).get("bundle"), "missing repair identity")
        require((r.get("performance") or {}).get("manifest"), "missing performance identity")
        authenticate(r["runtime"])
        for key in ("python", "kernel", "machine", "packages", "flags"):
            require(key in r["runtime"], f"runtime identity lacks {key}")

    lm, lc, lr = (left[k] for k in ("measurement", "config", "runtime"))
    rm, rc, rr = (right[k] for k in ("measurement", "config", "runtime"))
    for key in ("fixture", "binding", "driver_sha256", "prefix_tokens"):
        require(lm[key] == rm[key], f"comparison changes {key}")
    require(numerical_config(lc) == numerical_config(rc), "non-execution configuration differs")
    for key in ("effective_capacity", "diagnostic_sources"):
        require(lr[key] == rr[key], f"comparison changes {key}")
    for key in ("python", "kernel", "machine", "packages", "flags"):
        require(lr["runtime"][key] == rr["runtime"][key], f"runtime changes {key}")
    require(
        lr["runtime"].get("compiler_settings") == rr["runtime"].get("compiler_settings"),
        "runtime changes observed compiler settings",
    )
    require(lr["repair"]["bundle"] == rr["repair"]["bundle"], "repair bundles differ")
    require(lr["performance"]["manifest"] == rr["performance"]["manifest"], "performance differs")
    return seal(
        {
            "schema": "qwen.execution-mode-admission.v1",
            "status": "ADMITTED_MODE_COMPARISON",
            "modes": [lm["execution_mode"], rm["execution_mode"]],
            "captures": [lm["isolated_capture"], rm["isolated_capture"]],
            "arm": arm,
            "fixture": lm["fixture"],
            "binding": lm["binding"],
            "numerical_config_sha256": digest(numerical_config(lc)),
            "capacity": lr["effective_capacity"],
            "sources": lr["diagnostic_sources"],
            "repair": lr["repair"]["bundle"],
            "performance": lr["performance"]["manifest"],
            "observed_compiler_settings": lr["runtime"].get("compiler_settings"),
            "receipts": [
                [s[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
                for s in (left, right)
            ],
            "scope": "controlled replay admission; mode-selected operators may differ",
        }
    )


def compare_pair(left, right, left_rows, right_rows, *, arm="m8"):
    admission = admit_pair(left, right, arm=arm)
    for side, rows in zip((left, right), (left_rows, right_rows), strict=True):
        authenticate(rows)
        require(
            side["pass"]["observation"]["forced"]["sha256"] == rows["sha256"],
            "saved rows do not belong to this observed pass",
        )
        require(rows["continuation"] == admission["fixture"], "rows belong to another fixture")
        require(len(rows["rows"]) == 320, "mode comparison requires all 320 decode positions")
    decoded, initial = compare_saved_rows(
        left_rows, right_rows, target_rows=1 if arm == "m1" else 8
    )
    return seal(
        {
            "schema": "qwen.execution-mode-comparison.v1",
            "status": "COMPARED_SAVED_EVIDENCE",
            "admission": admission,
            "row_sources": [left_rows["sha256"], right_rows["sha256"]],
            "decode": aggregate(decoded),
            "prefill": aggregate([initial]),
            "stage_localization": "UNMEASURED; inspect aligned boundary captures separately",
            "scope": "320 forced decode predictions and one prefill prediction; no universal proof",
        }
    )
