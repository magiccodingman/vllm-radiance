"""Target-only global selection and conservative fallback regressions.

CPU cases execute the real dispatch functions. Native cases exercise the actual
INT2 projection and reranker on a public synthetic fixture, with no checkpoint.
"""

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from test_verify_head_sampling_capacity import SOURCE, batch, gate, load_module


def neutral_processors(sampler):
    sampler.penalties_state = SimpleNamespace(use_penalty=np.zeros(3, dtype=bool))
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(3, dtype=bool))
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    )
    sampler.logprob_token_ids_state = SimpleNamespace(
        num_token_ids=SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    )
    sampler.thinking_budget_state = SimpleNamespace(
        enabled=True, use_thinking_budget=np.zeros(3, dtype=bool)
    )


@pytest.mark.parametrize("depth", [128, 256])
@pytest.mark.parametrize("top_k", [1, 8, 20, 32, 33, 64, 65, 1024])
@pytest.mark.parametrize("mixed", [False, True])
def test_global_gate_uses_target_budget_not_draft_tile_capacity(depth, top_k, mixed):
    runner, request = batch(top_k, mixed=mixed)
    neutral_processors(runner.sampler)
    assert gate(rerank=32, candidates=8, global_topk=depth)(runner, request, None) == (
        top_k <= depth // 4
    )


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize(
    "restriction",
    [
        "grammar",
        "logprobs",
        "penalty",
        "bias",
        "bad_words",
        "token_logprobs",
        "thinking",
        "unknown_layout",
        "rows",
        "trace_replay",
    ],
)
def test_global_fallbacks_cover_each_selected_request(restriction, greedy):
    runner, request = batch(20, mixed=True)
    sampler = runner.sampler
    neutral_processors(sampler)
    if greedy:
        sampler.sampling_states.temperature.np[0] = 0
    grammar = None
    if restriction == "grammar":
        grammar = object()
    elif restriction == "logprobs":
        sampler.sampling_states.num_logprobs[0] = 0
    elif restriction == "penalty":
        sampler.penalties_state.use_penalty[0] = True
    elif restriction == "bias":
        sampler.logit_bias_state.use_logit_bias[0] = True
    elif restriction == "bad_words":
        sampler.bad_words_state.num_bad_words.np[0] = 1
    elif restriction == "token_logprobs":
        sampler.logprob_token_ids_state.num_token_ids.np[0] = 1
    elif restriction == "thinking":
        sampler.thinking_budget_state.use_thinking_budget[0] = True
    elif restriction == "trace_replay":
        sampler.trace_replay_state = object()
    elif restriction == "unknown_layout":
        del sampler.penalties_state
    else:
        request.logits_indices = np.zeros(33, dtype=np.int32)
    assert gate(global_topk=256)(runner, request, grammar) is False


def test_unselected_request_does_not_disable_global_path():
    runner, request = batch(20, mixed=True)
    neutral_processors(runner.sampler)
    runner.sampler.penalties_state.use_penalty[1] = True
    runner.sampler.logit_bias_state.use_logit_bias[1] = True
    runner.sampler.bad_words_state.num_bad_words.np[1] = 2
    assert gate(global_topk=256)(runner, request, None) is True


@pytest.mark.parametrize("top_k,min_p", [(0, 0), (-1, 0), (20, 0.1), (20, float("nan"))])
def test_global_rejects_invalid_or_unbounded_support(top_k, min_p):
    runner, request = batch(top_k)
    neutral_processors(runner.sampler)
    runner.sampler.sampling_states.min_p.np[0] = min_p
    assert gate(global_topk=256)(runner, request, None) is False


@pytest.mark.parametrize("value", [None, "0", "128", "256", "64", "-1", "invalid"])
def test_global_option_values(value):
    tree = ast.parse((SOURCE / "radiance_verifyhead.py").read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "GLOBAL_TOPK" for t in node.targets)
        )
        or (isinstance(node, ast.If) and "GLOBAL_TOPK" in ast.unparse(node.test))
    ]
    assert len(nodes) == 2
    environ = {} if value is None else {"RADIANCE_VERIFY_HEAD_GLOBAL_TOPK": value}
    namespace = {"os": SimpleNamespace(environ=environ)}
    code = compile(ast.Module(body=nodes, type_ignores=[]), "actual-option", "exec")
    if value in {None, "0", "128", "256"}:
        exec(code, namespace)
        assert namespace["GLOBAL_TOPK"] == (256 if value is None else int(value))
        runner, request = batch(20)
        neutral_processors(runner.sampler)
        assert gate(global_topk=namespace["GLOBAL_TOPK"])(runner, request, None) == (value != "0")
    else:
        with pytest.raises(ValueError):
            exec(code, namespace)


def native_modules(monkeypatch, depth):
    monkeypatch.setenv("RADIANCE_DRAFT_RERANK", "80")
    monkeypatch.setenv("RADIANCE_FAST_DRAFT", "1")
    monkeypatch.setenv("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", str(depth))
    return load_module("radiance_drafthead"), load_module("radiance_verifyhead")


def native_fixture(torch, draft, rows, *, vocab=1024, width=512):
    weight = torch.zeros(vocab, width, dtype=torch.bfloat16, device="cuda")
    # Twenty strict winners in a single 64-token tile. The coarse arithmetic
    # is exact on these values, isolating selection capacity from INT2 error.
    for index in range(20):
        weight[index, 0] = 3 * (index + 1)
        weight[index, 2] = index + 1
    hidden = torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
    hidden[:, 2] = 1
    head = SimpleNamespace(weight=weight, tp_size=1)
    state = SimpleNamespace(head_dtype=None, _radiance_topk_only=True)
    draft._quantize_head_now(state, head)
    state._radiance_exact_head = lambda lm, x, b: torch.nn.functional.linear(x, lm.weight, b)
    return state, head, hidden


def native_enabled():
    import os

    return os.environ.get("RADIANCE_TEST_NATIVE") == "1"


@pytest.mark.skipif(not native_enabled(), reason="GPU opt-in")
@pytest.mark.parametrize("depth", [128, 256])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16, 32])
def test_native_global_recovers_clustered_top20_without_changing_drafter(monkeypatch, depth, rows):
    import torch

    draft, verify = native_modules(monkeypatch, depth)
    state, head, hidden = native_fixture(torch, draft, rows)
    full = state._radiance_exact_head(head, hidden, None)
    old = draft._apply_head_int2(state, head, hidden, None)
    cutoff = full.topk(20, dim=-1).values[:, -1:]
    assert int(((full > cutoff) & ~torch.isfinite(old)).sum()) == rows * 11
    result = verify._apply_head_global(state, head, hidden)
    assert result.shape == full.shape
    assert torch.equal(result.argmax(-1), full.argmax(-1))
    assert ((full < cutoff) | torch.isfinite(result)).all()
    finite = torch.isfinite(result)
    assert torch.equal(result[finite], full[finite])
    assert (finite.sum(-1) == depth).all()
    assert draft.KCAND == 8 and draft.RERANK == 80
    assert torch.equal(draft._apply_head_int2(state, head, hidden, None), old)


@pytest.mark.skipif(not native_enabled(), reason="GPU opt-in")
@pytest.mark.parametrize("restriction", ["bias", "tp2", "wide", "small_vocab", "small_width"])
def test_native_global_unsupported_inputs_use_complete_reference(monkeypatch, restriction):
    import torch

    draft, verify = native_modules(monkeypatch, 256)
    state, head, hidden = native_fixture(
        torch,
        draft,
        33 if restriction == "wide" else 1,
        vocab=128 if restriction == "small_vocab" else 1024,
        width=256 if restriction == "small_width" else 512,
    )
    bias = (
        torch.ones(head.weight.shape[0], dtype=head.weight.dtype, device="cuda")
        if restriction == "bias"
        else None
    )
    if restriction == "tp2":
        head.tp_size = 2  # Tests the guard, not distributed numerical execution.
    result = verify._apply_head_global(state, head, hidden, bias)
    full = state._radiance_exact_head(head, hidden, bias)
    assert torch.equal(result.view(torch.uint8), full.view(torch.uint8))
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not native_enabled(), reason="GPU opt-in")
def test_native_public_hook_switches_global_and_full_head(monkeypatch):
    import torch

    draft, verify = native_modules(monkeypatch, 256)
    _, head, hidden = native_fixture(torch, draft, 8)

    class Processor:
        head_dtype = None

        def _apply_head(self, lm_head, x, bias=None):
            return torch.nn.functional.linear(x, lm_head.weight, bias)

    lp = Processor()
    model = SimpleNamespace(logits_processor=lp, lm_head=head, named_children=lambda: [])
    runner, request = batch(20, mixed=True)
    runner.model = model
    neutral_processors(runner.sampler)
    verify.ENABLED = True
    verify.before_compute_logits(runner, request, None)
    assert verify._state["armed"] and lp._radiance_fast_ok
    assert lp._radiance_fast_head.__func__ is verify._apply_head_global
    fast = lp._apply_head(head, hidden)
    assert (torch.isfinite(fast).sum(-1) == 256).all()
    verify.before_compute_logits(runner, request, object())
    assert not lp._radiance_fast_ok
    full = torch.nn.functional.linear(hidden, head.weight)
    assert torch.equal(lp._apply_head(head, hidden).view(torch.uint8), full.view(torch.uint8))
    verify.before_compute_logits(runner, request, None)
    resumed = lp._apply_head(head, hidden)
    assert lp._radiance_fast_ok
    assert (torch.isfinite(resumed).sum(-1) == 256).all()
    # torch.topk does not promise which tied zero-score candidates occupy the
    # unused shortlist slots. Compare the actual top-20 sampling support and
    # scores, whose strictly positive values are fully specified by this fixture.
    cutoff = full.topk(20, dim=-1).values[:, -1:]
    expected = full.masked_fill(full < cutoff, -float("inf"))
    for result in (fast, resumed):
        assert torch.equal(result.masked_fill(result < cutoff, -float("inf")), expected)
    assert draft.KCAND == 8 and draft.RERANK == 80
