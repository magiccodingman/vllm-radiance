#!/usr/bin/env python3
"""Share the GDN metadata build across KV-cache groups.

The KV-group-size choice (patch_kv_group_size: capacity over padded-slot count, +20.7% KV) has a
tax nobody priced: SIX GDN kv-cache groups means GDNAttentionMetadataBuilder.build runs six times
per step on inputs that differ ONLY in the group's block table -- measured 573 us/step of python
self-time plus each group's eager kernel chain, all inside the inter-step CPU bubble.

Every field of the resulting metadata except the block-table-derived state indices is identical
across groups. But each group's FULL cudagraph captured ITS OWN builder's persistent buffers, so
a shared metadata object is not enough: on a cache hit the values must still be COPIED into this
group's buffers. That is what the fast path does -- buffer-to-buffer copies of the first group's
already-padded buffers (which also replaces the pad fill_ launches), plus this group's own
block-table slice. The numpy bookkeeping, mask building, and index construction run once.

Scope: the steady spec-decode cudagraph branch only (num_prefills == 0, num_decodes == 0, fits
the capture bound). Any other step shape misses the cache and runs the stock build per group.
RADIANCE_GDN_SHARED_BUILD=0 disables. Idempotent."""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

# --- runner loop: one shared dict per build_attn_metadata call (= per step shape) -------------
apply(
    SP / "vllm" / "v1" / "worker" / "gpu" / "attn_utils.py",
    "    attn_metadata: dict[str, Any] = {}\n"
    "    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)\n",
    "    attn_metadata: dict[str, Any] = {}\n"
    "    # radiance (patch_gdn_shared_build.py): scoped to this call, so no cross-step identity\n"
    "    # keying is needed -- every group in the loop below sees the same step.\n"
    "    _rad_gdn_shared: dict[str, Any] | None = (\n"
    "        {} if __import__(\"os\").environ.get(\"RADIANCE_GDN_SHARED_BUILD\", \"1\") == \"1\"\n"
    "        else None\n"
    "    )\n"
    "    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)\n",
    "_rad_gdn_shared",
    "gdn shared build: runner dict",
)

apply(
    SP / "vllm" / "v1" / "worker" / "gpu" / "attn_utils.py",
    "                metadata = attn_metadata_builder.build(\n"
    "                    common_prefix_len=0,\n"
    "                    common_attn_metadata=common_attn_metadata,\n"
    "                    **attn_metadata_extra_kwargs,\n"
    "                )\n",
    "                # radiance (patch_gdn_shared_build.py): GDN builders share one build.\n"
    "                if hasattr(attn_metadata_builder, \"_radiance_shared_build\"):\n"
    "                    attn_metadata_extra_kwargs[\"_radiance_shared\"] = _rad_gdn_shared\n"
    "                metadata = attn_metadata_builder.build(\n"
    "                    common_prefix_len=0,\n"
    "                    common_attn_metadata=common_attn_metadata,\n"
    "                    **attn_metadata_extra_kwargs,\n"
    "                )\n",
    '_radiance_shared"] = _rad_gdn_shared',
    "gdn shared build: runner kwarg",
)

# --- builder: marker + signature ---------------------------------------------------------------
apply(
    SP / "vllm" / "v1" / "attention" / "backends" / "gdn_attn.py",
    "    def build(  # type: ignore[override]\n"
    "        self,\n"
    "        common_prefix_len: int,\n"
    "        common_attn_metadata: CommonAttentionMetadata,\n"
    "        num_accepted_tokens: torch.Tensor | None = None,\n"
    "        num_decode_draft_tokens_cpu: torch.Tensor | None = None,\n"
    "        fast_build: bool = False,\n"
    "    ) -> GDNAttentionMetadata:\n",
    "    # radiance (patch_gdn_shared_build.py): the runner passes a per-step dict here.\n"
    "    _radiance_shared_build = True\n"
    "\n"
    "    def build(  # type: ignore[override]\n"
    "        self,\n"
    "        common_prefix_len: int,\n"
    "        common_attn_metadata: CommonAttentionMetadata,\n"
    "        num_accepted_tokens: torch.Tensor | None = None,\n"
    "        num_decode_draft_tokens_cpu: torch.Tensor | None = None,\n"
    "        fast_build: bool = False,\n"
    "        _radiance_shared: dict | None = None,\n"
    "    ) -> GDNAttentionMetadata:\n",
    "_radiance_shared: dict | None = None",
    "gdn shared build: signature",
)

# --- builder: hit fast path, straight after the metadata handle ---------------------------------
apply(
    SP / "vllm" / "v1" / "attention" / "backends" / "gdn_attn.py",
    "        m = common_attn_metadata\n"
    "\n"
    "        query_start_loc = m.query_start_loc\n",
    "        m = common_attn_metadata\n"
    "\n"
    "        # radiance (patch_gdn_shared_build.py): a previous GDN group already built this\n"
    "        # step's metadata. Only the block-table-derived indices are per-group; everything\n"
    "        # else is copied buffer-to-buffer from the first group's already-padded buffers\n"
    "        # into THIS builder's buffers (each group's cudagraph captured its own addresses).\n"
    "        if _radiance_shared and _radiance_shared.get(\"kind\") == \"spec_fast\":\n"
    "            sh = _radiance_shared\n"
    "            import dataclasses as _dc\n"
    "            bt = mamba_get_block_table_tensor(\n"
    "                m.block_table_tensor, m.seq_lens, self.kv_cache_spec,\n"
    "                self.vllm_config.cache_config.mamba_cache_mode,\n"
    "            )\n"
    "            nsd = sh[\"nsd\"]\n"
    "            bs = m.num_reqs\n"
    "            if sh[\"mask_all\"]:\n"
    "                sidx = bt[: sh[\"mask_rows\"], : self.num_spec + 1]\n"
    "            else:\n"
    "                sidx = bt[sh[\"mask_cpu\"], : self.num_spec + 1]\n"
    "            self.spec_state_indices_tensor[:nsd].copy_(sidx, non_blocking=True)\n"
    "            sst = self.spec_state_indices_tensor[:bs]\n"
    "            sst[nsd:].fill_(NULL_BLOCK_ID)\n"
    "            self.spec_sequence_masks[:bs].copy_(sh[\"masks_src\"], non_blocking=True)\n"
    "            self.spec_token_indx[: sh[\"si_n\"]].copy_(sh[\"si_src\"], non_blocking=True)\n"
    "            self.non_spec_token_indx[: sh[\"nsi_n\"]].copy_(\n"
    "                sh[\"nsi_src\"], non_blocking=True)\n"
    "            self.spec_query_start_loc[: bs + 1].copy_(sh[\"qsl_src\"], non_blocking=True)\n"
    "            self.num_accepted_tokens[:bs].copy_(sh[\"acc_src\"], non_blocking=True)\n"
    "            return _dc.replace(\n"
    "                sh[\"md\"],\n"
    "                spec_state_indices_tensor=sst,\n"
    "                spec_sequence_masks=self.spec_sequence_masks[:bs],\n"
    "                spec_token_indx=self.spec_token_indx[: sh[\"si_n\"]],\n"
    "                non_spec_token_indx=self.non_spec_token_indx[: sh[\"nsi_n\"]],\n"
    "                spec_query_start_loc=self.spec_query_start_loc[: bs + 1],\n"
    "                num_accepted_tokens=self.num_accepted_tokens[:bs],\n"
    "            )\n"
    "        _rad_fastpath = False\n"
    "        _rad_mask_all = False\n"
    "\n"
    "        query_start_loc = m.query_start_loc\n",
    'kind") == "spec_fast"',
    "gdn shared build: hit path",
)

# --- builder: remember whether the padded-slice branch was taken --------------------------------
apply(
    SP / "vllm" / "v1" / "attention" / "backends" / "gdn_attn.py",
    "                    spec_state_indices_tensor = block_table_tensor[\n"
    "                        : _mask_np.size, : self.num_spec + 1\n"
    "                    ]\n",
    "                    spec_state_indices_tensor = block_table_tensor[\n"
    "                        : _mask_np.size, : self.num_spec + 1\n"
    "                    ]\n"
    "                    _rad_mask_all = True\n",
    "_rad_mask_all = True",
    "gdn shared build: mask flag",
)

# --- builder: mark the cudagraph spec fast path -------------------------------------------------
apply(
    SP / "vllm" / "v1" / "attention" / "backends" / "gdn_attn.py",
    "            num_accepted_tokens = self.num_accepted_tokens[:batch_size]\n"
    "            num_accepted_tokens[num_spec_decodes:].fill_(1)\n",
    "            num_accepted_tokens = self.num_accepted_tokens[:batch_size]\n"
    "            num_accepted_tokens[num_spec_decodes:].fill_(1)\n"
    "            _rad_fastpath = True\n",
    "_rad_fastpath = True",
    "gdn shared build: fastpath flag",
)

# --- builder: stash on miss, right before returning ---------------------------------------------
apply(
    SP / "vllm" / "v1" / "attention" / "backends" / "gdn_attn.py",
    "            token_chunk_offset_ptr=token_chunk_offset_ptr,\n"
    "        )\n"
    "        return attn_metadata\n",
    "            token_chunk_offset_ptr=token_chunk_offset_ptr,\n"
    "        )\n"
    "        # radiance (patch_gdn_shared_build.py): make this build reusable by the step's\n"
    "        # remaining GDN groups. Only the steady spec-decode cudagraph shape qualifies.\n"
    "        if (\n"
    "            _radiance_shared is not None\n"
    "            and not _radiance_shared\n"
    "            and _rad_fastpath\n"
    "            and num_prefills == 0\n"
    "            and num_decodes == 0\n"
    "        ):\n"
    "            _radiance_shared.update(\n"
    "                kind=\"spec_fast\", md=attn_metadata, nsd=num_spec_decodes,\n"
    "                mask_all=_rad_mask_all,\n"
    "                mask_rows=(spec_sequence_masks_cpu.numel()\n"
    "                           if spec_sequence_masks_cpu is not None else 0),\n"
    "                mask_cpu=spec_sequence_masks_cpu,\n"
    "                masks_src=spec_sequence_masks,\n"
    "                si_src=spec_token_indx, si_n=spec_token_indx.size(0),\n"
    "                nsi_src=non_spec_token_indx, nsi_n=non_spec_token_indx.size(0),\n"
    "                qsl_src=spec_query_start_loc, acc_src=num_accepted_tokens,\n"
    "            )\n"
    "        return attn_metadata\n",
    'kind="spec_fast", md=attn_metadata',
    "gdn shared build: stash",
)
