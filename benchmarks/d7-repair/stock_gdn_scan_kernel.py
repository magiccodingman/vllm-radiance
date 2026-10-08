# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The transition body is derived from flash-linear-attention's packed GDN
# decode in vLLM (original MIT notice: Copyright (c) 2023-2025, Songlin Yang,
# Yu Zhang). Keep the transition's rounding and reduction expressions intact.
# ruff: noqa: N803, N806 - retain the stock Triton argument names for review
"""Ordered stock GDN transition; experimental until native qualification passes.

Each program owns one disjoint 32-by-128 state tile throughout the sequence.
The loop has no access to future gates. Only the chosen after-row states are
published, so rejected speculative rows cannot change earlier snapshots.
"""

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["T"])
def stock_gdn_scan_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    initial,
    output,
    states,
    indices,
    accepted,
    cu,
    T,
    scale,
    stride_qkv: tl.constexpr,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SAVE_ROWS: tl.constexpr,
    INDEXED: tl.constexpr,
    stride_state: tl.constexpr,
    stride_index_seq: tl.constexpr,
    stride_index_row: tl.constexpr,
):
    i_v, i_hv = tl.program_id(0), tl.program_id(1)
    sequence = tl.program_id(2)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    offsets = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if INDEXED:
        begin = tl.load(cu + sequence)
        end = tl.load(cu + sequence + 1)
        previous = tl.load(accepted + sequence)
        source = tl.load(
            indices + sequence * stride_index_seq + (previous - 1) * stride_index_row
        ).to(tl.int64)
        if source <= 0:
            for row in range(begin, end):
                tl.store(output + (row * HV + i_hv) * V + o_v, 0, mask=mask_v)
            return
        b_h = tl.load(initial + source * stride_state + offsets, mask=mask_h, other=0).to(
            tl.float32
        )
    else:
        begin, end = 0, T
        b_h = tl.load(initial + offsets, mask=mask_h, other=0).to(tl.float32)
    for row in range(begin, end):
        p_mixed = mixed_qkv + row * stride_qkv
        q_off = i_h * K + o_k
        k_off = H * K + i_h * K + o_k
        v_off = 2 * H * K + i_hv * V + o_v
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        a_val = tl.load(a + row * stride_a + i_hv).to(tl.float32)
        b_val = tl.load(b + row * stride_b + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
        b_h *= exp(g_val)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        p_o = output + (row * HV + i_hv) * V + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if SAVE_ROWS:
            if INDEXED:
                destination = tl.load(
                    indices + sequence * stride_index_seq + (row - begin) * stride_index_row
                ).to(tl.int64)
                tl.store(
                    states + destination * stride_state + offsets,
                    b_h,
                    mask=mask_h & (destination > 0),
                )
            else:
                tl.store(states + row * HV * V * K + offsets, b_h, mask=mask_h)
    if not SAVE_ROWS:
        tl.store(states + offsets, b_h, mask=mask_h)
