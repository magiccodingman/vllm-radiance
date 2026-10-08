"""Per-block capacity is necessary even when coarse ranking is exact.

CPU tests execute the actual gate's AST without importing a GPU runtime.
Set RADIANCE_TEST_NATIVE=1 to include the synthetic GPU regression.
RADIANCE_TEST_SOURCE_ROOT can select the unmodified tree for a negative control.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(os.environ.get("RADIANCE_TEST_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])))


def gate(rerank=80, candidates=8, global_topk=0):
    module = ast.parse((SOURCE / "radiance_verifyhead.py").read_text())
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_batch_is_safe", "_sampled_top_k_limit", "_global_processors_supported",
        )
    ]
    assert any(node.name == "_batch_is_safe" for node in functions)
    namespace = {
        "_dh": SimpleNamespace(RERANK=rerank, KCAND=candidates),
        "_NO_LOGPROBS": -1,
        "MAX_ROWS": 32,
        "GLOBAL_TOPK": global_topk,
        "_GLOBAL_MAX_ROWS": 32,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual-gate", "exec"), namespace)
    return namespace["_batch_is_safe"]


def batch(top_k, *, mixed=False):
    # In mixed mode the selected request indices are deliberately reordered;
    # the unselected request has unsupported sampling and must not affect the gate.
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        top_k=SimpleNamespace(np=np.array([top_k, 1024, 1024], dtype=np.int32)),
        min_p=SimpleNamespace(np=np.zeros(3, dtype=np.float32)),
        num_logprobs=np.full(3, -1, dtype=np.int32),
    )
    indices = np.array([2, 0] if mixed else [0], dtype=np.int32)
    return (
        SimpleNamespace(sampler=SimpleNamespace(sampling_states=states)),
        SimpleNamespace(idx_mapping_np=indices, num_reqs=len(indices), logits_indices=indices),
    )


@pytest.mark.parametrize("rerank,candidates", [(32, 8), (80, 8), (80, 4), (8, 8)])
@pytest.mark.parametrize("top_k", [1, 4, 8, 9, 20, 21])
@pytest.mark.parametrize("mixed", [False, True])
def test_sampling_cannot_exceed_either_candidate_stage(rerank, candidates, top_k, mixed):
    runner, request_batch = batch(top_k, mixed=mixed)
    assert gate(rerank, candidates)(runner, request_batch, None) == (
        top_k <= min(rerank // 4, candidates)
    )


@pytest.mark.parametrize("restriction", ["grammar", "logprobs", "min_p", "row_count"])
def test_existing_fallback_conditions_remain(restriction):
    runner, request_batch = batch(4)
    grammar = None
    if restriction == "grammar":
        grammar = object()
    elif restriction == "logprobs":
        runner.sampler.sampling_states.num_logprobs[0] = 0
    elif restriction == "min_p":
        runner.sampler.sampling_states.min_p.np[0] = 0.1
    else:
        request_batch.logits_indices = np.zeros(33, dtype=np.int32)
    assert gate()(runner, request_batch, grammar) is False


def test_greedy_dispatch_is_outside_this_capacity_change():
    runner, request_batch = batch(1024)
    runner.sampler.sampling_states.temperature.np[0] = 0
    assert gate()(runner, request_batch, None) is True


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.environ.get("RADIANCE_TEST_NATIVE") != "1", reason="GPU opt-in")
def test_native_clustered_top20_falls_back_to_full_logits(monkeypatch):
    import torch

    monkeypatch.setenv("RADIANCE_DRAFT_RERANK", "80")
    monkeypatch.setenv("RADIANCE_FAST_DRAFT", "1")
    monkeypatch.setenv("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", "0")  # Explicit legacy path.
    draft = load_module("radiance_drafthead")
    verify = load_module("radiance_verifyhead")
    assert draft.KCAND == 8
    weight = torch.zeros(1024, 512, dtype=torch.bfloat16, device="cuda")
    # Each relevant weight is exactly represented by its INT2 group scale.
    for index in range(20):
        weight[index, 0] = 3 * (index + 1)
        weight[index, 2] = index + 1
    hidden = torch.zeros(1, 512, dtype=torch.bfloat16, device="cuda")
    hidden[0, 2] = 1
    head = SimpleNamespace(weight=weight)
    state = SimpleNamespace(head_dtype=None, _radiance_topk_only=True)
    draft._quantize_head_now(state, head)
    full = torch.nn.functional.linear(hidden, weight)
    fast = draft._apply_head_int2(state, head, hidden, None)
    cutoff = full.topk(20, dim=-1).values[:, -1:]
    finite = torch.isfinite(fast)
    # Ties at the cutoff are excluded: these eleven are strictly required.
    assert int(((full > cutoff) & ~finite).sum()) == 11
    assert torch.equal(fast.argmax(-1), full.argmax(-1))
    assert torch.equal(fast[finite], full[finite])
    runner, request_batch = batch(20)
    state._radiance_fast_ok = verify._batch_is_safe(runner, request_batch, None)
    assert state._radiance_fast_ok is False
    state._radiance_fast_head = lambda lm, x, bias: draft._apply_head_int2(state, lm, x, bias)
    state._radiance_exact_head = lambda lm, x, bias: torch.nn.functional.linear(x, lm.weight, bias)
    corrected = verify._apply_head_gated(state, head, hidden, None)
    assert torch.equal(corrected.view(torch.uint8), full.view(torch.uint8))
