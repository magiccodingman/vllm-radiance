#!/usr/bin/env python3
"""Selective DFlash2 backport for stable vLLM 0.27.1.

v0.27.1 contains the older DFlash runtime but not the Qwen3.8 DFlash2 local
convolution and candidate-selector implementation.  Backport the reviewed PR
#52816 at its exact final head without floating the rest of vLLM main.  Three
new modules are fetched by immutable commit and SHA256; small shared-runtime
changes use guarded, fail-closed anchors against the pinned 0.27.1 wheel.
"""
import hashlib
import sysconfig
import urllib.request
from pathlib import Path

from _patchlib import apply


SP = Path(sysconfig.get_paths()["purelib"])
COMMIT = "3406ec1dae9916f920b90f0dbf90dcf54923d042"
RAW = f"https://raw.githubusercontent.com/vllm-project/vllm/{COMMIT}/"
NEW_FILES = {
    "vllm/model_executor/models/qwen3_dflash2.py":
        "c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f",
    "vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py":
        "e3c55cbb0d7a8bd47df6f3378835644645f5d3bc89b45793b5d5a02d013e5c58",
    "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py":
        "1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836",
}


def install_new_files() -> None:
    for relative, expected in NEW_FILES.items():
        data = urllib.request.urlopen(RAW + relative, timeout=60).read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"DFlash2 source hash mismatch for {relative}: {actual} != {expected}"
            )
        destination = SP / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() == data:
            print(f"  NOOP  dflash2 source already installed: {relative}")
            continue
        destination.write_bytes(data)
        print(f"  OK    dflash2 source installed: {relative}")


def patch_config() -> None:
    path = SP / "vllm/config/vllm.py"
    apply(
        path,
        '''        if self._dflash_needs_multi_kv_group():
            return True

        if self.model_config is not None and self.model_config.is_diffusion:
''',
        '''        if self._dflash_needs_multi_kv_group():
            return True

        # DFlash2's candidate selector exists only in the V2 speculator.
        if self._is_dflash2_draft():
            return True

        if self.model_config is not None and self.model_config.is_diffusion:
''',
        "if self._is_dflash2_draft():",
        "dflash2: require the V2 runner",
    )
    apply(
        path,
        '''        return True

    def _dflash_needs_multi_kv_group(self) -> bool:
''',
        '''        return True

    def _is_dflash2_draft(self) -> bool:
        """Whether the configured DFlash draft uses the DFlash2 architecture."""
        spec = self.speculative_config
        if spec is None or spec.method != "dflash":
            return False
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is None:
            return False
        return "DFlash2DraftModel" in (draft_config.architectures or [])

    def _dflash_needs_multi_kv_group(self) -> bool:
''',
        "def _is_dflash2_draft",
        "dflash2: architecture detector",
    )


def patch_logits_processor() -> None:
    path = SP / "vllm/model_executor/layers/logits_processor.py"
    apply(
        path,
        '''"""A layer that compute logits from hidden_stats."""

import torch
''',
        '''"""A layer that compute logits from hidden_stats."""

from collections.abc import Callable
from functools import cache

import torch
''',
        "from functools import cache",
        "dflash2: logits helper imports",
    )
    apply(
        path,
        '''from vllm.platforms import current_platform


# --8<-- [start:logits_processor]
''',
        '''from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from vllm.logger import init_logger

logger = init_logger(__name__)


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    if not current_platform.is_cuda() or not has_flashinfer():
        return None
    from flashinfer import top_k

    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    return impl(scores, k, sorted=True, deterministic=True)


# --8<-- [start:logits_processor]
''',
        "def _flashinfer_topk",
        "dflash2: vocab top-k helper",
    )
    apply(
        path,
        '''        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def extra_repr(self) -> str:
''',
        '''        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def get_top_k_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel top-k without all-gathering full logits."""
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        values, ids = _topk(logits, k)
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index
        if lm_head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, k)
            ids = ids.gather(-1, selected)

        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    def extra_repr(self) -> str:
''',
        "def get_top_k_tokens",
        "dflash2: vocab-parallel top-k",
    )


def patch_dflash_base() -> None:
    path = SP / "vllm/model_executor/models/qwen3_dflash.py"
    apply(
        path,
        '''def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:
    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return override
''',
        '''def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:
    """Resolve explicit causality before falling back to legacy layer defaults."""
    is_causal = getattr(config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return bool(override)
''',
        "is_causal = getattr(config",
        "dflash2: explicit causality",
    )
    apply(
        path,
        '''class DFlashQwen3Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
''',
        '''class DFlashQwen3Model(nn.Module):
    decoder_layer_cls = DFlashQwen3DecoderLayer

    hf_to_vllm_mapper = WeightsMapper(
''',
        "decoder_layer_cls = DFlashQwen3DecoderLayer",
        "dflash2: overridable decoder class",
    )
    apply(
        path,
        "                DFlashQwen3DecoderLayer(\n",
        "                self.decoder_layer_cls(\n",
        "self.decoder_layer_cls(\n",
        "dflash2: select decoder class",
    )
    apply(
        path,
        '''class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
''',
        '''class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    model_cls = DFlashQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
''',
        "model_cls = DFlashQwen3Model",
        "dflash2: overridable model class",
    )
    apply(
        path,
        "        self.model = DFlashQwen3Model(\n",
        "        self.model = self.model_cls(\n",
        "self.model = self.model_cls(\n",
        "dflash2: select model class",
    )
    # Quantized fused context-K/V materialization is deliberately a separate
    # guarded patch.  Keeping it outside this feature backport lets the loader
    # use the quantization method itself instead of duplicating layout/scale
    # rules for FP8, MXFP4, and future packed formats here.


def patch_registry_and_dispatch() -> None:
    registry = SP / "vllm/model_executor/models/registry.py"
    apply(
        registry,
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
        '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
        '"DFlash2DraftModel": ("qwen3_dflash2"',
        "dflash2: model registry",
    )
    dispatch = SP / "vllm/v1/worker/gpu/spec_decode/__init__.py"
    apply(
        dispatch,
        '''    if speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
''',
        '''    if speculative_config.method == "dflash":
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
''',
        "return DFlash2Speculator(vllm_config, device)",
        "dflash2: speculator dispatch",
    )


def patch_sampling() -> None:
    gumbel = SP / "vllm/v1/worker/gpu/sample/gumbel.py"
    apply(
        gumbel,
        '''@triton.jit
def gumbel_block_argmax(
''',
        '''@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0."""
    if temp != 0.0 and APPLY_TEMPERATURE:
        logits = logits / temp
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))
    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_block_argmax(
''',
        "def gumbel_noised_argmax",
        "dflash2: shared gumbel argmax",
    )
    speculator = SP / "vllm/v1/worker/gpu/spec_decode/speculator.py"
    apply(
        speculator,
        '''            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=torch.float32,
                device=device,
            )
''',
        '''            dtype, fill = self.draft_logits_spec(vllm_config)
            self.draft_logits = torch.full(
                (self.max_num_reqs, self.num_speculative_steps, self.vocab_size),
                fill,
                dtype=dtype,
                device=device,
            )
''',
        "dtype, fill = self.draft_logits_spec",
        "dflash2: configurable proposal cache",
    )
    apply(
        speculator,
        '''    def _validate_local_argmax_reduction(self) -> None:
''',
        '''    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        """Dtype and fill for the cached proposal distribution."""
        return vllm_config.model_config.head_dtype, 0.0

    def _validate_local_argmax_reduction(self) -> None:
''',
        "def draft_logits_spec",
        "dflash2: proposal cache contract",
    )


def main() -> None:
    install_new_files()
    patch_config()
    patch_logits_processor()
    patch_dflash_base()
    patch_registry_and_dispatch()
    patch_sampling()


if __name__ == "__main__":
    main()
