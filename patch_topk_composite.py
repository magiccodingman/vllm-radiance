#!/usr/bin/env python3
"""Route small-k top-k/top-p to the composite path (radiance_topk.py) without a GPU sync.

The Qrita Triton kernel runs one program per row over the whole vocabulary: 907 us/step at the
rejection sampler's 8-9 rows (2026-08-30 census, single largest non-GEMM kernel). The composite
needs to know the batch's largest top_k BEFORE touching GPU tensors, so this patch threads it
from the input batch's existing CPU numpy mirror through SamplingMetadata:

  1. SamplingMetadata gains `max_top_k: int = 0` (0 = unknown/absent -> old path).
  2. The input batch fills it from `top_k_cpu` (rows without top-k carry vocab_size there, which
     correctly disqualifies the whole batch).
  3. The rejection sampler forwards it to apply_top_k_top_p.
  4. apply_top_k_top_p routes to the composite when 0 < max_top_k <= KCAP.

RADIANCE_TOPK_COMPOSITE=0 turns the route off. See radiance_topk.py for exactness notes.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

apply(
    SP / "vllm" / "v1" / "sample" / "metadata.py",
    "    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies\n"
    "    # thinking-token-budget logits (holder may exist with an empty tracking set).\n"
    "    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None\n",
    "    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies\n"
    "    # thinking-token-budget logits (holder may exist with an empty tracking set).\n"
    "    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None\n"
    "\n"
    "    # radiance (patch_topk_composite.py): largest per-request top_k in the batch, from the\n"
    "    # CPU mirror -- lets the sampler route small-k batches without a GPU sync. 0 = unknown.\n"
    "    max_top_k: int = 0\n",
    "max_top_k: int = 0",
    "topk composite: metadata field",
)

apply(
    SP / "vllm" / "v1" / "worker" / "gpu_input_batch.py",
    "            top_k=None if self.no_top_k else self.top_k[:num_reqs],\n",
    "            top_k=None if self.no_top_k else self.top_k[:num_reqs],\n"
    "            # radiance (patch_topk_composite.py): CPU-known bound, no sync. Rows without a\n"
    "            # real top-k hold vocab_size in top_k_cpu, disqualifying the composite route.\n"
    "            max_top_k=0 if self.no_top_k else int(self.top_k_cpu[:num_reqs].max()),\n",
    "max_top_k=0 if self.no_top_k",
    "topk composite: input batch",
)

apply(
    SP / "vllm" / "v1" / "sample" / "rejection_sampler.py",
    "    return apply_top_k_top_p(logits, top_k, top_p)\n",
    "    return apply_top_k_top_p(\n"
    "        logits, top_k, top_p,\n"
    "        # radiance (patch_topk_composite.py)\n"
    "        max_top_k=getattr(sampling_metadata, \"max_top_k\", 0),\n"
    "    )\n",
    'max_top_k=getattr(sampling_metadata, "max_top_k", 0)',
    "topk composite: rejection sampler",
)

apply(
    SP / "vllm" / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
    '''def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """Apply top-k and/or top-p masking to logits.

    has_topp_only_rows must be True if any row has p < 1 and
    k >= logits.shape[-1] (no active top-k).
    """
''',
    '''_RADIANCE_TOPK_COMPOSITE = __import__("os").environ.get("RADIANCE_TOPK_COMPOSITE", "1") == "1"


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
    *, max_top_k: int = 0,
) -> torch.Tensor:
    """Apply top-k and/or top-p masking to logits.

    has_topp_only_rows must be True if any row has p < 1 and
    k >= logits.shape[-1] (no active top-k).
    """
    if _RADIANCE_TOPK_COMPOSITE and k is not None and max_top_k > 0:
        import radiance_topk
        if max_top_k <= radiance_topk.KCAP:
            return radiance_topk.apply_top_k_top_p_composite(logits, k, p)
''',
    '_RADIANCE_TOPK_COMPOSITE',
    'topk composite: sampler route',
)

# --- The build actually serves through the NEW worker (vllm/v1/worker/gpu/), whose sampler calls
# apply_top_k_top_p from two sites of its own, neither passing max_top_k -- discovered when the
# instrumented composite counted zero calls in a live serve. Both have the CPU numpy mirror
# (UvaBackedTensor.np) at hand; rows without a real top-k hold vocab_size there, which correctly
# turns the route off for the whole batch.

apply(
    SP / "vllm" / "v1" / "worker" / "gpu" / "sample" / "states.py",
    '        return apply_top_k_top_p(\n            logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows\n        )\n',
    '''        # radiance: use the existing CPU mirror; no device synchronization.
        kmax = 0 if top_k is None else int(self.top_k.np[idx_mapping_np].max())
        return apply_top_k_top_p(
            logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows,
            max_top_k=0 if kmax >= logits.shape[-1] else kmax,
        )
''',
    'kmax = 0 if top_k is None',
    'topk composite: new-worker states',
)

apply(
    SP / "vllm" / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
    '            processed_logits = apply_top_k_top_p(\n                processed_logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows\n            )\n',
    '''            _rad_kmax = (0 if top_k is None else int(
                self.sampling_states.top_k.np[idx_mapping_np].max()
            ))
            processed_logits = apply_top_k_top_p(
                processed_logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows,
                max_top_k=0 if _rad_kmax >= processed_logits.shape[-1] else _rad_kmax,
            )
''',
    '_rad_kmax',
    'topk composite: new-worker sampler',
)
