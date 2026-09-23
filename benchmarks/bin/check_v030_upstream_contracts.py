#!/usr/bin/env python3
"""Installed upstream ownership checks for former DFlash/unpadding glue."""
import ast
import sysconfig
from pathlib import Path
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

q = torch.tensor([0, 1, 2], dtype=torch.int32)
metadata = CommonAttentionMetadata(query_start_loc=q, query_start_loc_cpu=q,
    seq_lens=torch.tensor([9, 12]), num_reqs=2, num_actual_tokens=2,
    max_query_len=1, max_seq_len=12, block_table_tensor=torch.zeros((2,2),dtype=torch.int32),
    slot_mapping=torch.tensor([1,2]), seq_lens_cpu_upper_bound=torch.tensor([10,13]))
trimmed = metadata.unpadded(1,1)
assert trimmed.seq_lens_cpu_upper_bound.tolist() == [10]
assert metadata.seq_lens_cpu_upper_bound.tolist() == [10,13]
assert trimmed.num_reqs == 1 and trimmed.num_actual_tokens == 1

path = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
source = path.read_text()
ast.parse(source)
# Source guards for original Triton ownership. Native gumbel/graph behavior is
# covered separately; these are not claimed to be a speculative model run.
for contract in ("sample_idx_mapping = torch.full", "sample_idx_mapping.fill_(-1)",
                 "is_valid_ctx = j < num_valid_ctx", "ctx_resident = is_valid_ctx & (ctx_block_id != 0)",
                 "q_resident = is_query & (q_block_id != 0)",
                 "tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len)"):
    assert contract in source, contract
print("v0.30 installed unpadding behavior and DFlash source ownership: PASS")
