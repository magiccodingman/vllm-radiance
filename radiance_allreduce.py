#!/usr/bin/env python3
"""radiance custom all-reduce: one-shot P2P-BAR AR for dual R9700 (gfx1201, TP=2, PCIe).

Mirrors vLLM's CustomAllreduce (ca_comm) interface, so it slots into CudaCommunicator.all_reduce's
existing size-gated dispatch. should_custom_ar returns True only for small messages in the win band;
larger or other messages fall through to RCCL.

Kernels: r4d.ar_oneshot_2rank_exact (exact bf16) and r4d.ar_oneshot_2rank_wht6 (Walsh-Hadamard rotated
6-bit payload, used for large messages when RADIANCE_USE_R4D_AR_QUANT=1). Both come from the R4D gfx1201 kernel
library; both are one-shot push all-reduces for exactly two P2P-connected ranks, which is what their
names say, and they share the same scratch, flags and seq counters. PUSH model. Each rank writes its
input into the peer's IPC scratch, then reduces (local input + peer data now in the local scratch) into out. Only scratch and flags
are IPC-shared; input/output stay local, so no cudagraph buffer registration is needed. cudagraph-safe:
the sequence number is a device-resident counter the kernel increments (a host-passed seq would freeze
at capture), and scratch is double-buffered by seq parity. Multi-block: each block owns a contiguous
chunk and drives its own flag handshake, spreading the push across CUs (PCIe) and the reduce across CUs.

Env:
  RADIANCE_USE_R4D_AR (1)      1 = install the kernel on the TP group, 0 = RCCL
  RADIANCE_USE_R4D_AR_QUANT (1) 1 = compress the payload (rotated 6-bit, group-scaled) for large messages
"""
import os
import sys
from contextlib import contextmanager

import torch
import torch.distributed as dist

_DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}

# Largest message the kernel takes; anything above it falls back to RCCL. Sized to hold one prefill
# chunk's all-reduce (4096 tokens x 5120 channels x bf16 = 40 MiB), so a serve running the shipped
# --max-num-batched-tokens keeps the kernel for prefill as well as decode. Costs 2x this in IPC
# scratch per rank, which is trivial on 32 GB. The kernel is bit-identical to RCCL at every size.
_MAX_BYTES = 49152 * 1024
# Below this the exact bf16 kernel wins: compression only pays once the transfer is bandwidth-bound.
_QUANT_MIN_BYTES = 128 * 1024


def _log(msg):
    sys.stderr.write(f"[radiance] {msg}\n")
    sys.stderr.flush()


class RadianceAllreduce:
    """Drop-in for vLLM's ca_comm (CustomAllreduce). Same public surface:
    `disabled`, `should_custom_ar(inp)`, `custom_all_reduce(inp)`, `capture()`."""

    def __init__(self, group, device):
        self.disabled = True
        self._ar_exact = None
        self._ar_wht6 = None
        self.group = group
        try:
            self.world_size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
        except Exception as e:
            _log(f"custom AR: no process group ({e!r})")
            return
        try:
            import r4d as ext
        except Exception as e:
            _log(f"custom AR disabled: r4d import failed ({e!r})")
            return
        self._ext = ext

        # "exactly 2 ranks" is a property of the kernel, not of this file, so ask the library
        # whether it has an all-reduce for this group instead of keeping a copy of the number.
        # The dtype in the question is the production one; the kernel takes fp16 and fp32 through
        # the same entry point, and should_custom_ar() is what gates a message's dtype per call.
        name = ext.select("allreduce", world_size=self.world_size, exact=1, dtype="bf16")
        if name is None:
            _log(f"custom AR disabled: no exact all-reduce kernel for "
                 f"world_size={self.world_size}")
            return
        self._ar_exact = getattr(ext, name)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        self.max_bytes = _MAX_BYTES
        self.max_bytes = (self.max_bytes // 16) * 16     # 16B (uint4) alignment
        self.slot16 = self.max_bytes // 16               # per-slot capacity in 16B words
        self.drain = 3        # s_wait_storecnt drain (correct for fine-grained/uncached scratch)
        self.acq = 0          # no explicit acquire fence (uncached reads are already fresh)
        self.nt = 1024        # threads/block (tuned)
        self.min_nb = 4       # block count scales with message size, clamped to [min_nb, max_nb]
        self.max_nb = min(24, int(ext.AR_MAX_BLOCKS))
        self.words_per_block = 1400   # 16B words/block target for the block-count heuristic
        fine = True           # fine-grained (uncached) IPC scratch: posts straight to the fabric
        maxb = int(ext.AR_MAX_BLOCKS)

        try:
            torch.cuda.set_device(self.device)
            # double-buffered scratch (2 slots) + per-block flags, both IPC-shared
            self._scratch, sc_h, fine_used = self._alloc(2 * self.max_bytes, fine)
            self._flags, fl_h, _ = self._alloc(maxb * 4, fine)
            sc_handles = [None] * self.world_size
            fl_handles = [None] * self.world_size
            dist.all_gather_object(sc_handles, sc_h, group=group)
            dist.all_gather_object(fl_handles, fl_h, group=group)
            peer = 1 - self.rank
            self._peer_scratch = ext.ar_ipc_open(sc_handles[peer])
            self._peer_flags = ext.ar_ipc_open(fl_handles[peer])
            # per-BLOCK device-resident seq counters (kernel increments -> replay-safe;
            # both ranks run identical graph sequences so seq_ctr[b] stays in lockstep)
            self._seq = torch.zeros(maxb, dtype=torch.int32, device=self.device)
        except Exception as e:
            _log(f"custom AR disabled: IPC setup failed ({e!r})")
            return

        self.disabled = False
        _log(f"custom all-reduce INSTALLED (rank={self.rank} max={self.max_bytes // 1024}KB "
             f"finegrained={fine_used} drain={self.drain} acq={self.acq} nt={self.nt} "
             f"nb={self.min_nb}..{self.max_nb})")

        # Optional compressed payload path (RADIANCE_USE_R4D_AR_QUANT). Additive; shares this class's
        # scratch, flags and seq counters. Only large (bandwidth-bound) messages take it; smaller
        # ones keep the exact bf16 kernel above. On by default; set 0 for the exact
        # (RCCL-identical) bf16 path.
        self.ar_quant = os.environ.get("RADIANCE_USE_R4D_AR_QUANT", "1") == "1"
        self.quant_min_bytes = _QUANT_MIN_BYTES
        self.qnt = 1024        # threads/block for the compressed push (wire-bound; not sensitive)
        self.qmax_nb = 48      # block cap for the compressed path
        self._qext = None
        self._qgroup = 0
        self._locpk = None
        if self.ar_quant:
            try:
                import r4d as qext
                self._qgroup = int(qext.AR_WHT6_GROUP)
                # numel is the one per-message term in the question: a group is the smallest
                # message this path will send, and _quant_ok() enforces the multiple per call.
                qname = qext.select("allreduce", world_size=self.world_size, exact=0,
                                    dtype="bf16", numel=self._qgroup)
                if qname is None:
                    raise RuntimeError("no lossy all-reduce kernel in this build")
                self._ar_wht6 = getattr(qext, qname)
                self._qext = qext
                # This rank's half, kept so the reduce folds exactly the bytes it sent. Same
                # scale_off split as the peer scratch. Held on the instance for a stable address
                # under cudagraph replay.
                loc_bytes = self.max_bytes // 2 + self.max_bytes // 32 + 4096
                self._locpk = torch.empty(loc_bytes, dtype=torch.uint8, device=self.device)
                _log(f"AR_QUANT ON (rotated {int(qext.AR_WHT6_BITS)}-bit packed payload; "
                     f"min={self.quant_min_bytes // 1024}KB group={self._qgroup} "
                     f"nt={self.qnt} max_nb={self.qmax_nb} local={loc_bytes >> 20}MB)")
            except Exception as e:
                _log(f"AR_QUANT disabled: {e!r}")
                self.ar_quant = False

    def _alloc(self, nbytes, fine):
        """Allocate a shared buffer, falling back fine->coarse if fine-grained IPC fails.
        Deterministic across ranks (same env/HW), so the fallback stays symmetric."""
        try:
            ptr, h = self._ext.ar_ipc_alloc(nbytes, fine)
            return ptr, h, fine
        except Exception as e:
            if fine:
                _log(f"fine-grained alloc failed ({e!r}); falling back to coarse")
                ptr, h = self._ext.ar_ipc_alloc(nbytes, False)
                return ptr, h, False
            raise

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in _DTYPE_CODE:
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes == 0 or nbytes % 16 != 0:  # uint4 push alignment
            return False
        if nbytes > self.max_bytes:
            return False
        return inp.is_contiguous()

    def _nblocks(self, n16: int) -> int:
        # PCIe saturates with few blocks; extra blocks only help the reduce. Scale with
        # message size, clamped. Baked per-batch-size at cudagraph capture (n16 fixed).
        nb = n16 // self.words_per_block
        if nb < self.min_nb:
            nb = self.min_nb
        if nb > self.max_nb:
            nb = self.max_nb
        if nb > n16:
            nb = max(1, n16)
        return nb

    def _quant_ok(self, inp: torch.Tensor) -> bool:
        # compressed path only for large (bandwidth-bound) bf16/fp16 messages that tile the group.
        if not self.ar_quant or self._qext is None:
            return False
        if inp.dtype not in (torch.bfloat16, torch.float16):
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes < self.quant_min_bytes:
            return False
        return inp.numel() % self._qgroup == 0

    def _nblocks_q(self, nbytes: int) -> int:
        nb = nbytes // (self.words_per_block * 16)
        if nb < self.min_nb:
            nb = self.min_nb
        if nb > self.qmax_nb:
            nb = self.qmax_nb
        return nb

    def custom_all_reduce(self, inp: torch.Tensor):
        if not self.should_custom_ar(inp):
            return None
        out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        if self._quant_ok(inp):
            self._ar_wht6(
                self._peer_scratch, self._scratch, self._peer_flags, self._flags,
                self._seq.data_ptr(), self._locpk.data_ptr(),
                self.max_bytes, self.max_bytes // 2,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], stream, self._nblocks_q(nbytes), self.qnt,
                self.drain, self.acq,
            )
        else:
            self._ar_exact(
                self._peer_scratch, self._scratch, self._peer_flags, self._flags,
                self._seq.data_ptr(), self.slot16,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,
                self.drain, self.acq,
            )
        return out

    @contextmanager
    def capture(self):
        # No graph-buffer registration needed (push model: only fixed IPC scratch/flags
        # are peer-shared; input/output stay local & are captured with stable addresses).
        yield


def install_custom_ar():
    """Wrap CudaCommunicator.all_reduce so small TP messages take the one-shot kernel and everything
    else falls through to RCCL. Wrapping rather than replacing ca_comm is required: vLLM's RocmAiter fusion
    pass asserts isinstance(ca_comm, CustomAllreduce), and ca_comm is already inert on ROCm. Env-gated
    by RADIANCE_USE_R4D_AR, idempotent."""
    # RADIANCE_USE_R4D is the master switch for the whole libr4d integration (patch_r4d.py):
    # with it off the TP group keeps RCCL, exactly as in an image built without the library.
    if os.environ.get("RADIANCE_USE_R4D", "1") != "1":
        _log("custom all-reduce not installed: RADIANCE_USE_R4D=0")
        return
    if os.environ.get("RADIANCE_USE_R4D_AR", "1") != "1":
        return
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    if getattr(CudaCommunicator, "_radiance_ar_patched", False):
        return
    _orig_init = CudaCommunicator.__init__
    _orig_all_reduce = CudaCommunicator.all_reduce

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.radiance_comm = None
        try:
            # only the TP group does custom AR (matches vLLM's own gating)
            if "tp" in getattr(self, "unique_name", "") and getattr(self, "world_size", 1) == 2:
                comm = RadianceAllreduce(self.cpu_group, self.device)
                if not comm.disabled:
                    self.radiance_comm = comm
        except Exception as e:
            _log(f"custom AR attach failed: {e!r}")

    def _patched_all_reduce(self, input_):
        rc = getattr(self, "radiance_comm", None)
        if rc is not None and not rc.disabled and rc.should_custom_ar(input_):
            out = rc.custom_all_reduce(input_)
            if out is not None:
                return out
        return _orig_all_reduce(self, input_)

    CudaCommunicator.__init__ = _patched_init
    CudaCommunicator.all_reduce = _patched_all_reduce
    CudaCommunicator._radiance_ar_patched = True
    _log("fast-reduce hook armed (RADIANCE_USE_R4D_AR=1, all_reduce wrap)")
