"""Reject confounded experiments and unbound outputs before reporting agreement."""

from copy import deepcopy

import numpy as np
import pytest

from qwen_r9700_lab.conformance_execution_modes import (
    SOURCES,
    admit_pair,
    compare_pair,
    numerical_config,
)
from qwen_r9700_lab.conformance_topk import summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def reseal(document):
    return seal({k: v for k, v in document.items() if k != "sha256"})


def saved_rows(change=False):
    logits = summarize_logits(np.arange(32, dtype=np.float32))
    altered = summarize_logits(-np.arange(32, dtype=np.float32))
    return seal(
        {
            "schema": "urn:qwen:d7-equivalence-private-rows:v1",
            "continuation": "f" * 64,
            "prefill": altered if change else logits,
            "rows": [
                {
                    "position": p,
                    "absolute_position": 60000 + p,
                    "target_rows": 8,
                    "logits": altered if change and p == 16 else logits,
                }
                for p in range(320)
            ],
        }
    )


def side(mode, rows, capture=False):
    graph = "NONE" if mode != "compiled" or capture else "PIECEWISE"
    return {
        "measurement": seal(
            {
                "execution_mode": mode,
                "correctness": True,
                "m1": False,
                "lanes": ["fixed-bf16"],
                "isolated_capture": capture,
                "fixture": "f" * 64,
                "binding": "b" * 64,
                "driver_sha256": "d" * 64,
                "prefix_tokens": 60000,
            }
        ),
        "config": seal(
            {
                "enforce_eager": mode == "eager",
                "max_num_seqs": 2,
                "max_num_batched_tokens": 2048,
                "compilation_config": {
                    "mode": 0 if mode == "eager" else 3,
                    "cudagraph_mode": graph,
                },
                "worker_cls": "same-integration",
            }
        ),
        "runtime": seal(
            {
                "enforce_eager": mode == "eager",
                "compilation_mode": 0 if mode == "eager" else 3,
                "graph_mode": graph,
                "async_scheduling": False,
                "effective_capacity": {
                    "max_num_seqs": 2,
                    "max_num_batched_tokens": 2048,
                    "max_model_len": 253792,
                    "block_size": 16,
                    "cache_dtype": "fp8",
                    "num_gpu_blocks": 16000,
                },
                "diagnostic_sources": dict.fromkeys(SOURCES, "a" * 64),
                "repair": {"bundle": "r" * 64},
                "performance": {"manifest": "p" * 64},
                "runtime": seal(
                    {
                        "python": "pinned",
                        "kernel": "pinned",
                        "machine": "x86_64",
                        "packages": {"torch": "pinned"},
                        "flags": {},
                    }
                ),
            }
        ),
        "pass": seal(
            {
                "mode": "correctness",
                "execution_mode": mode,
                "fixture": "f" * 64,
                "observation": {
                    "observation": {
                        "counts": {
                            "target_forward_calls": 41,
                            "target_graph_replays": 2600 if graph != "NONE" else 0,
                        }
                    },
                    "forced": {"sha256": rows["sha256"]},
                },
            }
        ),
    }


@pytest.mark.parametrize(
    "left_mode,right_mode,capture",
    [
        ("compiled", "compiled-no-graphs", False),
        ("compiled-no-graphs", "eager", False),
        ("compiled-no-graphs", "compiled-no-graphs", True),
        ("eager", "eager", True),
    ],
)
def test_admits_controlled_modes_and_observer_bridges(left_mode, right_mode, capture):
    rows = saved_rows()
    report = compare_pair(side(left_mode, rows), side(right_mode, rows, capture), rows, rows)
    assert report["decode"]["positions"] == 320
    assert report["decode"]["full_logits_exact"] == 320
    assert report["prefill"]["full_logits_exact"] == 1
    assert report["stage_localization"].startswith("UNMEASURED")


def test_keeps_prefill_disagreement_separate_from_decode():
    a, b = saved_rows(), saved_rows(change=True)
    result = compare_pair(side("compiled-no-graphs", a), side("eager", b), a, b)
    assert result["prefill"]["full_logits_exact"] == 0
    assert result["decode"]["full_logits_exact"] == 319
    assert result["decode"]["1"]["set_exact"] == 319


def test_m1_observer_requires_explicit_arm_and_both_sides_m1():
    rows = saved_rows()
    for row in rows["rows"]:
        row["target_rows"] = 1
    rows = reseal(rows)
    a, b = side("compiled", rows), side("compiled-no-graphs", rows, capture=True)
    for record in (a, b):
        record["measurement"]["m1"] = True
        record["measurement"] = reseal(record["measurement"])
    with pytest.raises(DiagnosticError):
        compare_pair(a, b, rows, rows)
    result = compare_pair(a, b, rows, rows, arm="m1")
    assert result["admission"]["arm"] == "m1"
    assert result["decode"]["full_logits_exact"] == 320
    b["measurement"]["m1"] = False
    b["measurement"] = reseal(b["measurement"])
    with pytest.raises(DiagnosticError):
        compare_pair(a, b, rows, rows, arm="m1")


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("measurement", "binding", "different"),
        ("measurement", "driver_sha256", "different"),
        ("measurement", "fixture", "different"),
        ("measurement", "prefix_tokens", 57008),
        ("measurement", "m1", True),
        ("config", "max_num_seqs", 1),
        ("config", "max_num_batched_tokens", 4096),
        ("config", "unrecognized_future_numeric_flag", True),
        ("runtime", "repair", {"bundle": "different"}),
        ("runtime", "performance", {"manifest": "different"}),
        ("runtime", "diagnostic_sources", dict.fromkeys(SOURCES, "different")),
        ("runtime", "effective_capacity", {}),
        ("runtime", "async_scheduling", True),
    ],
)
def test_rejects_confounds_even_when_outputs_happen_to_match(section, field, value):
    rows = saved_rows()
    a, b = side("compiled-no-graphs", rows), side("eager", rows)
    b[section][field] = value
    b[section] = reseal(b[section])
    with pytest.raises(DiagnosticError):
        compare_pair(a, b, rows, rows)


def test_keeps_unknown_compiler_options_in_the_comparison():
    base = {"compilation_config": {"mode": 0, "custom_ops": ["+all"]}}
    changed = deepcopy(base)
    changed["compilation_config"]["custom_ops"] = ["-all"]
    assert numerical_config(base) != numerical_config(changed)


def test_rejects_changed_effective_capacity_and_runtime_flags():
    rows = saved_rows()
    for variant in ("capacity", "flag"):
        a, b = side("compiled-no-graphs", rows), side("eager", rows)
        if variant == "capacity":
            b["runtime"]["effective_capacity"]["max_num_seqs"] = 1
        else:
            b["runtime"]["runtime"]["flags"]["NUMERIC_FLAG"] = "1"
            b["runtime"]["runtime"] = reseal(b["runtime"]["runtime"])
        b["runtime"] = reseal(b["runtime"])
        with pytest.raises(DiagnosticError):
            admit_pair(a, b)


def test_rejects_rows_from_another_pass_and_incomplete_or_wrong_width_replays():
    good, other = saved_rows(), saved_rows(change=True)
    a, b = side("compiled-no-graphs", good), side("eager", good)
    with pytest.raises(DiagnosticError, match="observed pass"):
        compare_pair(a, b, good, other)
    for fault in ("short", "width", "position"):
        bad = deepcopy(good)
        if fault == "short":
            bad["rows"].pop()
        elif fault == "width":
            bad["rows"][0]["target_rows"] = 1
        else:
            bad["rows"][0]["absolute_position"] += 1
        bad = reseal(bad)
        with pytest.raises(DiagnosticError):
            compare_pair(a, side("eager", bad), good, bad)


def test_rejects_claimed_graph_mode_without_actual_replay():
    rows = saved_rows()
    a, b = side("compiled", rows), side("compiled-no-graphs", rows)
    a["pass"]["observation"]["observation"]["counts"]["target_graph_replays"] = 0
    a["pass"] = reseal(a["pass"])
    with pytest.raises(DiagnosticError, match="observed graph execution"):
        admit_pair(a, b)
