#!/usr/bin/env python3
"""Pick the hybrid KV cache group size by usable capacity instead of by smallest bucket.

vLLM slices a hybrid model's layers into equal-sized KV cache groups. Upstream picks the group
size as `min(len(bucket))` over the distinct KVCacheSpec buckets -- correct for the n:1 patterns
it was written for (Gemma3 5:1, LLaMA4 3:1), and its own FIXME says a better strategy is needed
for anything else. Qwen3.8-27B + a DFlash2 drafter is anything else. The buckets are:

    48  linear_attention (GDN, MambaSpec)
    16  full_attention   (FullAttentionSpec)
     5  drafter          (SlidingWindowSpec, window 2048)

so upstream picks 5, which divides neither 48 nor 16, and pads to 10 + 4 + 1 = 15 groups holding
75 layer slots for 69 real layers. Confirmed on the live serve:

    WARNING kv_cache_utils.py:1261 Add 2 padding layers, may waste at most 4.17% KV cache memory
    WARNING kv_cache_utils.py:1261 Add 4 padding layers, may waste at most 25.00% KV cache memory
    INFO    kv_cache_utils.py:2235 GPU KV cache size: 739,544 tokens

The wasted slots are the small half of the cost. What actually dominates is that *every group
independently rounds a request up to whole blocks*, and the groups are not equally expensive: at
block_size 1648 and max_model_len 262144 a full-attention group costs 160 blocks per request, a
GDN group costs 2 + num_speculative_blocks = 9, and the drafter's sliding-window group costs 8.
Group size 5 splits the 16 full-attention layers into *four* groups -- 4 x 160 = 640 of the 738
blocks a request reserves. Group size 8 divides both 48 and 16, giving two full-attention groups
and 382 blocks per request.

Capacity is `(available / page_size / group_size) / blocks_per_request`, so the objective is to
minimise `group_size x blocks_per_request`. That is what this patch searches for, using vLLM's own
`max_memory_usage_bytes()` for the per-group cost. Measured against the live configuration
(16.37 GiB KV, 10412 pages, TP2):

    group size   groups   blocks   blk/req    concurrency        tokens
      5 (up)       15      2082      738         2.82x          739,544
      8 (this)      9      1301      382         3.41x          892,799   +20.7%
      1             69     10412     3032        3.43x          900,212   +21.7%

Group size 1 is the true optimum -- zero padding -- but it needs 69 block tables and 69 per-request
round-ups of Python bookkeeping to buy 0.8% over group size 8, so RADIANCE_KV_GROUP_MAX_GROUPS
caps the search. Note the cheap proxy "minimise padded slots" does *not* find this: group sizes 3,
6 and 8 all waste exactly 3 slots, but score +9.8%, +9.8% and +20.7%. Padding count is not the
objective; where the padding lands is.

Candidates are restricted to canonical sizes -- those where `max(cdiv(L, cdiv(L, g)))` over the
buckets equals `g` -- so the value returned here is exactly the `group_size` the allocator will
compute downstream from `max(len(group.layer_names))`.

Knobs (all optional, upstream behaviour is one env var away):
    RADIANCE_KV_GROUP_OPT=0          restore upstream min-bucket selection
    RADIANCE_KV_GROUP_SIZE=<n>       force a group size (A/B testing)
    RADIANCE_KV_GROUP_MAX_GROUPS=<n> cap the group count, default 24

Verify in the serve log: "[radiance] kv cache groups:" reports the chosen size and what upstream
would have picked, and "GPU KV cache size" should rise from 739,544 to ~892,799 tokens.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KV = SP / "vllm/v1/core/kv_cache_utils.py"

# --- 1. thread vllm_config in so we can ask each spec what a request costs -------------------
SIG_ANCHOR = (
    "def _get_kv_cache_groups_uniform_page_size(\n"
    "    kv_cache_spec: dict[str, KVCacheSpec],\n"
    ") -> list[KVCacheGroupSpec]:\n"
)
SIG_NEW = (
    "def _radiance_pick_group_size(layer_buckets, spec_buckets, vllm_config, upstream):\n"
    '    """Group size maximising usable KV cache. See patch_kv_group_size.py."""\n'
    "    import os\n"
    "\n"
    '    if os.environ.get("RADIANCE_KV_GROUP_OPT", "1") == "0":\n'
    "        return upstream\n"
    '    forced = os.environ.get("RADIANCE_KV_GROUP_SIZE")\n'
    "    if forced:\n"
    "        return max(1, int(forced))\n"
    "    if vllm_config is None:\n"
    "        return upstream\n"
    "    try:\n"
    "        sizes = [len(layers) for layers in layer_buckets]\n"
    "        page = max(s.page_size_bytes for specs in spec_buckets for s in specs)\n"
    "        # Blocks one group of each bucket reserves for a worst-case request. This is the\n"
    "        # term upstream ignores: a full-attention group costs cdiv(max_model_len,\n"
    "        # block_size) blocks, a mamba group costs a handful, so splitting the expensive\n"
    "        # bucket into more groups costs far more than a few padded layer slots.\n"
    "        cost = [\n"
    "            max(cdiv(s.max_memory_usage_bytes(vllm_config), page) for s in specs)\n"
    "            for specs in spec_buckets\n"
    "        ]\n"
    "    except Exception:  # never fail a serve over an optimisation\n"
    "        return upstream\n"
    "\n"
    '    max_groups = int(os.environ.get("RADIANCE_KV_GROUP_MAX_GROUPS") or 24)\n'
    "    best = None\n"
    "    for g in range(1, max(sizes) + 1):\n"
    "        splits = [cdiv(L, g) for L in sizes]\n"
    "        eff = max(cdiv(L, k) for L, k in zip(sizes, splits))\n"
    "        if eff != g:\n"
    "            continue  # non-canonical: an equivalent smaller g covers this split\n"
    "        ngroups = sum(splits)\n"
    "        if ngroups > max_groups:\n"
    "            continue\n"
    "        blocks = sum(k * c for k, c in zip(splits, cost))\n"
    "        key = (g * blocks, ngroups)\n"
    "        if best is None or key < best[0]:\n"
    "            best = (key, g, ngroups, blocks)\n"
    "    if best is None:\n"
    "        return upstream\n"
    "    _, g, ngroups, blocks = best\n"
    "    logger.info(\n"
    '        "[radiance] kv cache groups: size %d, %d groups, %d blocks/request "\n'
    '        "(upstream would pick size %d)",\n'
    "        g,\n"
    "        ngroups,\n"
    "        blocks,\n"
    "        upstream,\n"
    "    )\n"
    "    return g\n"
    "\n"
    "\n"
    "def _get_kv_cache_groups_uniform_page_size(\n"
    "    kv_cache_spec: dict[str, KVCacheSpec],\n"
    "    vllm_config=None,\n"
    ") -> list[KVCacheGroupSpec]:\n"
)

# --- 2. use it ------------------------------------------------------------------------------
TAIL_ANCHOR = (
    "        # extra layers to one attention type.\n"
    "        group_size = max_num_layers\n"
    "    grouped_layers = []\n"
)
TAIL_NEW = (
    "        # extra layers to one attention type.\n"
    "        group_size = max_num_layers\n"
    "    # --- radiance (patch_kv_group_size.py) ---\n"
    "    # The rule above assumes an n:1 layer pattern; 48 GDN + 16 full + 5 drafter is not one,\n"
    "    # and picking the smallest bucket (5) splits the 16 expensive full-attention layers\n"
    "    # across 4 groups. Search for the size that maximises usable cache instead.\n"
    "    group_size = _radiance_pick_group_size(\n"
    "        layer_buckets, spec_buckets, vllm_config, group_size\n"
    "    )\n"
    "    grouped_layers = []\n"
)

# --- 3. caller ------------------------------------------------------------------------------
CALL_ANCHOR = "    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)\n"
CALL_NEW = "    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec, vllm_config)\n"

apply(KV, SIG_ANCHOR, SIG_NEW, "_radiance_pick_group_size", "kv group size: helper")
apply(KV, TAIL_ANCHOR, TAIL_NEW, "group_size = _radiance_pick_group_size(",
      "kv group size: selection")
apply(KV, CALL_ANCHOR, CALL_NEW, "uniform_page_size(filtered_spec, vllm_config)",
      "kv group size: caller")
