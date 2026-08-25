"""R4D attention: the RADIANCE gfx1201 (RDNA4) attention kernels, wired in as a vLLM backend.

Select it with `--attention-backend R4D`. With speculative decoding, give the drafter the same
backend: `--speculative-config '{..., "attention_backend": "R4D"}'`.

R4D is a pair of hand-written HIP kernels (r4d_attn_prefill.hip, r4d_attn_decode.hip) built around a
transposed score matrix, S^T = K.Q^T, so that a wave32 WMMA fragment gives each lane exactly one
query row and the whole softmax is lane-private, at a slightly SMALLER error against an fp32 oracle
than the kernel they replace.

Measured in the serve against the tuned AITER unified attention, same image and flags: the prefill
attention kernel is 1.65x faster, which is +14.6% end-to-end prefill throughput at 64K context
(attention is 34% of prefill GPU time there) and +4.1% at 16K (11.8%). Decode is unchanged within
noise: attention is only ~7% of a speculative decode step, which is dominated by the MoE GEMMs of
the drafter loop.

R4D is a library of gfx1201 kernels, not a model-specific one, and its entry points are named for
the geometry they are compiled for. This backend does not name them: it asks r4d.select() for the
kernel compiled for the model in front of it, and a model of another shape gets None and a refusal
rather than a kernel that does not fit. The constraint list therefore lives once, in libr4d's
registry, next to the kernels it describes.

What the kernels this resolves to cover, and therefore what this backend accepts:
  * head_dim 256, paged block size 16, 6 query heads per KV head
  * causal decoder attention, bf16 query, bf16 or fp8_e4m3 KV cache
  * no sliding window, no attention sinks, no logits soft cap, no alibi, no per-(token,head) scales
Everything else is refused at startup by validate_configuration, which prints the reason and the
backends that would work instead. This class subclasses the Triton backend, so the KV cache layout,
the cache-write path and the metadata builder are the stock ones, and anything R4D cannot take
(a non-causal batch, fused output quantisation) falls back to the Triton kernel rather than failing.

The batch is cut into runs of equal query length, one kernel launch each: a decode batch (every
request the same width) is a single launch, and a mixed batch costs one launch per run. The plan is
computed once per step by the metadata builder from the CPU copy of query_start_loc, never from
device memory, so it stays free of syncs and is stable enough to be captured into a HIP graph.
"""

import os
import sys

import torch

from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
try:
    # Post-v0.27 vLLM names cache layouts with this enum. The stable v0.27.1
    # selector still asks each backend for its legacy string layout instead.
    from vllm.v1.kv_cache_interface import KVCacheLayout
except ImportError:  # vLLM v0.27.1
    KVCacheLayout = None

def _say(msg: str) -> None:
    # This module is outside the `vllm` logger namespace, where vLLM installs its handler, so a
    # logger here would be silently dropped. The other radiance modules write to stderr too.
    sys.stderr.write(f"[radiance] {msg}\n")


# RTLD_DEEPBIND is load-bearing. tilelang (pulled in through aiter) puts its libhip_stub.so into
# the GLOBAL symbol scope, and that stub defines hipLaunchKernel to throw "install ROCm first".
# Every HIP symbol resolved through the global scope afterwards hits the stub instead of the real
# runtime, which turns the first kernel launch into a RuntimeError. DEEPBIND makes this module
# search its own DT_NEEDED libamdhip64 first; the Python C API still resolves as usual, since it
# is not one of the module's dependencies.
_dlflags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
try:
    import r4d

    _IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - only when the library was not built
    r4d = None
    _IMPORT_ERROR = e
finally:
    sys.setdlopenflags(_dlflags)

# The geometry the R4D attention kernels are compiled for. R4D is a library of gfx1201 kernels, not
# a model-specific one, and its entry-point names carry the shape each kernel serves -- so read the
# numbers out of the library instead of keeping a second copy here that can drift from it. The
# fallbacks only matter when the library is missing, in which case this backend refuses to load.
HEAD_DIM = int(r4d.ATTN_HEAD_DIM) if r4d else 256
BLOCK_SIZE = int(r4d.ATTN_BLOCK_SIZE) if r4d else 16
GQA = int(r4d.ATTN_GQA) if r4d else 6
# One decode workgroup owns this many (query position, gqa head) rows, which caps the query length
# the split-KV decode kernel can take. Longer runs go to the prefill kernel, which tiles the query.
MAX_DECODE_ROWS = int(r4d.ATTN_MAX_DECODE_ROWS) if r4d else 64
MAX_DECODE_QLEN = MAX_DECODE_ROWS // GQA

# One kernel per (phase, KV cache dtype), bound once -- by asking the library which entry point it
# has for this geometry rather than by naming them here. select() returns the name compiled for a
# geometry or None, so the constraint list stays in libr4d's registry, in one place, and this file
# cannot drift from it.
_PAGED = dict(head_dim=HEAD_DIM, gqa=GQA, block_size=BLOCK_SIZE, causal=1, q_dtype="bf16")
# Index order is what _geometry() returns for a cache tensor's dtype.
_KV_DTYPES = ("fp8_e4m3", "bf16")
# vLLM's spelling of a cache dtype -> the registry's. "auto" means the model dtype, which is bf16
# for every model this backend accepts.
_KV_DTYPE_NAMES = {"auto": "bf16", "bfloat16": "bf16", "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3"}


# The master switch for the whole libr4d integration (patch_r4d.py). This backend is the one place
# it cannot silently fall back: it was asked for by name on the command line, so it refuses to load
# and says why, rather than serving something slower than what was asked for.
USE_R4D = os.environ.get("RADIANCE_USE_R4D", "1") == "1"
_OFF_MSG = "RADIANCE_USE_R4D=0 turns the R4D kernel library off; drop --attention-backend R4D too"


def _bind(op: str, **geometry):
    """The entry point this build has for a geometry, or None if it has none."""
    if r4d is None or not USE_R4D:
        return None
    name = r4d.select(op, **geometry)
    return getattr(r4d, name) if name else None


_PREFILL = tuple(_bind("attn_prefill_paged", kv_dtype=d, **_PAGED) for d in _KV_DTYPES)
# q_len only picks the decode row apart from the prefill one; the per-call cap is MAX_DECODE_QLEN.
_DECODE = tuple(_bind("attn_decode_paged", kv_dtype=d, q_len=1, **_PAGED) for d in _KV_DTYPES)


def _plan(query_start_loc_cpu: torch.Tensor, num_reqs: int) -> tuple:
    """Cut the batch into maximal runs of consecutive requests with equal query length.

    Returns one tuple per run: (first request, request count, query length, first token). Requests
    padded in by a graph capture have no tokens at all and are dropped here.
    """
    qs = query_start_loc_cpu.tolist()
    groups = []
    i = 0
    while i < num_reqs:
        n = qs[i + 1] - qs[i]
        if n == 0:
            i += 1
            continue
        j = i + 1
        while j < num_reqs and qs[j + 1] - qs[j] == n:
            j += 1
        groups.append((i, j - i, n, qs[i]))
        i = j
    return tuple(groups)


class R4DAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    # A captured launch bakes its grid, so a replay is only safe if every request in the batch has
    # the same query length -- which is exactly what UNIFORM_BATCH promises. Mixed batches keep
    # running eagerly under piecewise graphs.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "R4D attention does not support decode context parallelism; every rank would "
                "hold only part of each sequence's KV."
            )
        self.max_ctx_bound = vllm_config.model_config.max_model_len
        # The decode kernel writes split-KV partials to scratch. Size it here, for the worst shape
        # this engine can schedule: allocating during a graph capture is not allowed, and the
        # buffer has to live at the same address at capture and at replay.
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        nbytes = max(
            r4d.attn_decode_h256_gqa6_scratch_bytes(
                n,
                MAX_DECODE_QLEN,
                self.num_heads_q,
                self.num_heads_kv,
                self.headdim,
                self.max_ctx_bound,
                0,
            )
            for n in range(1, max_seqs + 1)
        )
        self.scratch = torch.empty(nbytes, dtype=torch.uint8, device=device)
        variant = 1 if kv_cache_spec.dtype.itemsize == 2 else 0
        # Log the entry points actually bound, not a description of them: the R4D names carry the
        # geometry, so this line is the record of which kernels this engine is running.
        _say(
            f"R4D attention: {self.num_heads_q} q heads / {self.num_heads_kv} kv heads, head_dim "
            f"{self.headdim}, block {self.block_size}, "
            f"{_PREFILL[variant].__name__} + {_DECODE[variant].__name__}, "
            f"{nbytes / (1 << 20):.1f} MiB split-KV scratch"
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        m = super().build(common_prefix_len, common_attn_metadata, fast_build)
        m.r4d_plan = _plan(
            common_attn_metadata.query_start_loc_cpu, common_attn_metadata.num_reqs
        )
        m.r4d_max_ctx = common_attn_metadata.max_seq_len
        m.r4d_scratch = self.scratch
        return m

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        m = super().build_for_cudagraph_capture(common_attn_metadata)
        # The split count is chosen on the host and baked into the captured grid, so it has to be
        # the one valid for every replay: the longest context this engine can reach, not the one
        # in the dummy batch (which the capture path sets to 1).
        m.r4d_max_ctx = self.max_ctx_bound
        return m


class R4DAttentionBackend(TritonAttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]

    @staticmethod
    def get_name() -> str:
        return "R4D"

    @staticmethod
    def get_impl_cls() -> type["R4DAttentionImpl"]:
        return R4DAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[R4DAttentionMetadataBuilder]:
        return R4DAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Exactly 16, not a multiple of it. A larger framework block size is still fine: vLLM
        # splits it into 16-token kernel blocks, which is how this works alongside the 2240-token
        # pages a GDN hybrid needs for --mamba-cache-mode=align.
        return [BLOCK_SIZE]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == HEAD_DIM

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple:
        # Pinned vLLM main resolves one physical cache layout across every
        # backend before allocation. LBHNC is its head-major, block-compact
        # spelling of legacy HND: the logical tensor remains
        # (blocks, heads, slots, 2*head_size), while slots are contiguous
        # within each head exactly as the R4D kernels require.
        if KVCacheLayout is None:
            return ()
        return (KVCacheLayout.LBHNC,)

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        # v0.27.1 spelling of the same head-major, slot-contiguous layout.
        return "HND"

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        if r4d is None:
            return f"R4D kernels are not built into this image ({_IMPORT_ERROR})"
        if not USE_R4D:
            return _OFF_MSG
        return None


class R4DAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if r4d is None:
            raise RuntimeError(f"R4D kernels are not built into this image ({_IMPORT_ERROR})")
        if not USE_R4D:
            raise RuntimeError(_OFF_MSG)
        unsupported = []
        # Whether R4D has a kernel for this model's shape is the library's question to answer, not
        # a copy of its constraints kept here: select() returns one or None, and explain() names
        # the constraint that failed. The refusals below it are not geometry -- they are features
        # the kernels have no concept of, and a model that wants one needs a different backend
        # rather than a different instantiation.
        geometry = dict(
            head_dim=self.head_size,
            gqa=self.num_queries_per_kv,
            block_size=BLOCK_SIZE,
            causal=1,
            q_dtype="bf16",
            kv_dtype=_KV_DTYPE_NAMES.get(self.kv_cache_dtype, self.kv_cache_dtype),
        )
        if r4d.select("attn_prefill_paged", **geometry) is None:
            unsupported.extend(
                sorted({
                    row["reason"]
                    for row in r4d.explain("attn_prefill_paged", **geometry)
                    if not row["ok"]
                })
            )
        if self.sliding_window != (-1, -1):
            unsupported.append("sliding window")
        if self.alibi_slopes is not None:
            unsupported.append("alibi slopes")
        if self.logits_soft_cap:
            unsupported.append("a logits soft cap")
        if self.sinks is not None:
            unsupported.append("attention sinks")
        if self._is_per_token_head_quant:
            unsupported.append("per-(token, head) KV scales")
        if self.chunk_lookback != -1:
            unsupported.append("chunked local attention")
        if unsupported:
            raise NotImplementedError(
                "R4D attention does not support " + ", ".join(unsupported) + ". Serve this model "
                "with --attention-backend TRITON_ATTN or ROCM_AITER_UNIFIED_ATTN instead."
            )
        cfg = get_current_vllm_config()
        self._descale_len = cfg.scheduler_config.max_num_seqs * self.num_kv_heads
        self._kv_geometry: tuple | None = None
        self._descales: tuple | None = None

    def fused_output_quant_supported(self, quant_key) -> bool:
        # The kernels write bf16. Let the fusion pass quantise the output separately.
        return False

    def _geometry(self, kv_cache: torch.Tensor, query: torch.Tensor, out: torch.Tensor) -> tuple:
        """Variant index and cache strides, checked once and then reused for the layer's life."""
        if self._kv_geometry is None:
            if not query.is_contiguous() or not out.is_contiguous():
                raise RuntimeError("R4D needs contiguous query and output tensors")
            _, _, _, content = kv_cache.shape
            if (
                content != 2 * self.head_size
                or kv_cache.stride(3) != 1
                or kv_cache.stride(2) != content
            ):
                raise RuntimeError(
                    f"R4D needs a K/V-packed HND cache with contiguous slots, got shape "
                    f"{tuple(kv_cache.shape)} strides {kv_cache.stride()}"
                )
            variant = 1 if kv_cache.element_size() == 2 else 0
            self._kv_geometry = (variant, kv_cache.stride(0), kv_cache.stride(1))
        return self._kv_geometry

    def _descale_ptrs(self, layer, device) -> tuple[int, int]:
        """Device addresses of the K/V descales, or 0 when they are 1.0 and fold away.

        The kernels read one descale per (sequence, KV head); vLLM keeps a single per-tensor scale,
        so it is broadcast once into a buffer here rather than per step.
        """
        ks = float(getattr(layer, "_k_scale_float", 1.0))
        vs = float(getattr(layer, "_v_scale_float", 1.0))
        if ks == 1.0 and vs == 1.0:
            return 0, 0
        if self._descales is None:
            n = self._descale_len
            self._descales = (
                torch.empty(n, dtype=torch.float32, device=device),
                torch.empty(n, dtype=torch.float32, device=device),
                None,
            )
        kbuf, vbuf, cached = self._descales
        if cached != (ks, vs):
            kbuf.fill_(ks)
            vbuf.fill_(vs)
            self._descales = (kbuf, vbuf, (ks, vs))
        return kbuf.data_ptr(), vbuf.data_ptr()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)  # profiling run: no KV cache yet
        plan = getattr(attn_metadata, "r4d_plan", None)
        if (
            plan is None
            or attn_metadata.causal is not True
            or output_scale is not None
            or output_block_scale is not None
            or kv_cache.numel() == 0
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        variant, block_stride, head_stride = self._geometry(kv_cache, query, output)
        k_descale, v_descale = self._descale_ptrs(layer, query.device)
        block_table = attn_metadata.block_table
        max_blocks = block_table.shape[1]
        q_row = self.num_heads * self.head_size * query.element_size()
        o_row = self.num_heads * self.head_size * output.element_size()
        q_base, o_base = query.data_ptr(), output.data_ptr()
        bt_base, sl_base = block_table.data_ptr(), attn_metadata.seq_lens.data_ptr()
        kv_ptr = kv_cache.data_ptr()
        scratch = attn_metadata.r4d_scratch.data_ptr()
        max_ctx = attn_metadata.r4d_max_ctx
        stream = torch.cuda.current_stream().cuda_stream

        for first_req, num_seqs, q_len, first_tok in plan:
            launch = _DECODE[variant] if q_len <= MAX_DECODE_QLEN else _PREFILL[variant]
            launch(
                q_base + first_tok * q_row,
                kv_ptr,
                bt_base + first_req * max_blocks * 4,
                sl_base + first_req * 4,
                o_base + first_tok * o_row,
                k_descale,
                v_descale,
                scratch,
                num_seqs,
                q_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                BLOCK_SIZE,
                max_blocks,
                block_stride,
                head_stride,
                self.scale,
                0,  # let the kernel's split law choose
                max_ctx,
                stream,
            )
        return output
