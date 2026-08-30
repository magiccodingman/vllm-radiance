#!/usr/bin/env python3
"""Dynamic per-request verify width for speculative decoding (scheduler-side).

WHY. Under a DFlash2 drafter the DRAFT cost is fixed -- the block-diffusion pass emits all 8
positions in one graphed pass whether or not they are used -- so the speculative depth knob only
controls the VERIFY width: the target forward runs M = batch x (width + 1) rows. The right width
is content-dependent and the spread is 2x: on BetterBench's weighted mix, code/json/file_edit run
tok/update 4.7-6.0 and want depth 7+, while prose/chat accept ~2.8 and pay 7 wide verify rows for
nothing (the whole SPEC 5-vs-7 measurement of 2026-08-29). A static depth picks one loser.

HOW. Scheduler-side, pure CPU, no proposer or GPU changes: cap each request's spec_token_ids to a
per-request width before scheduling. The cap goes in update_draft_token_ids -- under ASYNC
scheduling that is where the PLACEHOLDER spec list is set, and the placeholder length is what the
next schedule() turns into verify rows (update_draft_token_ids_in_output then trims the real
drafts to the scheduled count, an existing supported path). The width comes from a per-request
EMA of accepted counts, observed in update_from_output where the scheduler already computes
num_accepted. A request that accepts its FULL capped width observes accepted+1 so the EMA can
climb back out of its own cap (otherwise a cap at 3 could never see evidence for 5).

Numerics: speculative verification is distribution-lossless regardless of proposal length. Output
streams are not guaranteed BYTE-identical only because the decode GEMM's split-K choice depends
on M -- the same batch-size-dependent ulp drift the engine already has across concurrency levels.

Inert unless RADIANCE_DYNAMIC_WIDTH=1. Knobs: RADIANCE_DYNW_ALPHA (EMA weight, 0.35),
RADIANCE_DYNW_MARGIN (rows above the EMA, 2), RADIANCE_DYNW_MIN (floor, 2),
RADIANCE_DYNW_MIN_BATCH (cap engages only at >= this many running requests, 5 -- the local c4
gate was noise-limited with no reproducible benefit while c8 gained 9.0-9.6%, so c1-c4 retain
full K7 verification). Idempotent; run once pre-serve.
"""
import ast
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm/v1/core/sched/scheduler.py"
SENTINEL = "_radiance_dynw_observe"

HUNK_A_ANCHOR = """                num_rejected = num_draft_tokens - num_accepted
"""
HUNK_A_ADD = """                self._radiance_dynw_observe(request, num_accepted, num_draft_tokens)
"""

HUNK_B_ANCHOR = """            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output("""
HUNK_B_ADD = """            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
            self._radiance_cap_spec_width(request)

    def update_draft_token_ids_in_output("""

TAIL = '''

# ---- RADIANCE dynamic verify width (patch_dynwidth.py) --------------------------------------
import math as _rad_math
import os as _rad_os

_RAD_DYNW = _rad_os.environ.get("RADIANCE_DYNAMIC_WIDTH", "0") == "1"
_RAD_DYNW_ALPHA = float(_rad_os.environ.get("RADIANCE_DYNW_ALPHA", "0.35"))
_RAD_DYNW_MARGIN = int(_rad_os.environ.get("RADIANCE_DYNW_MARGIN", "2"))
_RAD_DYNW_MIN = int(_rad_os.environ.get("RADIANCE_DYNW_MIN", "2"))
_RAD_DYNW_MIN_BATCH = int(_rad_os.environ.get("RADIANCE_DYNW_MIN_BATCH", "5"))


def _radiance_dynw_observe(self, request, num_accepted, num_draft_tokens):
    if not _RAD_DYNW or num_draft_tokens <= 0:
        return
    # Full acceptance of a capped width is evidence the cap binds; observe one above it so the
    # EMA can climb back out of its own shadow.
    obs = float(num_accepted) + (1.0 if num_accepted >= num_draft_tokens else 0.0)
    ema = getattr(request, "_rad_dynw_ema", None)
    request._rad_dynw_ema = (
        obs if ema is None else _RAD_DYNW_ALPHA * obs + (1.0 - _RAD_DYNW_ALPHA) * ema
    )


def _radiance_cap_spec_width(self, request):
    if not _RAD_DYNW or not request.spec_token_ids:
        return
    # The cap only pays where the batch's summed verify rows cross real cost boundaries; below
    # M ~ 16 the decode GEMMs are weight-stream-bound and M-invariant, so capping a lone stream
    # (or a pair) saves nothing and can only truncate a token that would have accepted -- the
    # measured conc-2 dip. Gate the CAP on batch size, never the EMA observation above: history
    # stays warm at every concurrency, so caps apply instantly the moment a batch forms.
    if len(self.running) < _RAD_DYNW_MIN_BATCH:
        return
    ema = getattr(request, "_rad_dynw_ema", None)
    if ema is None:
        return                       # cold start: full width until there is evidence
    w = int(_rad_math.ceil(ema)) + _RAD_DYNW_MARGIN
    if w < _RAD_DYNW_MIN:
        w = _RAD_DYNW_MIN
    if w < len(request.spec_token_ids):
        del request.spec_token_ids[w:]


Scheduler._radiance_dynw_observe = _radiance_dynw_observe
Scheduler._radiance_cap_spec_width = _radiance_cap_spec_width
'''

src = TARGET.read_text()
if SENTINEL in src:
    print(f"  NOOP  {TARGET.relative_to(SP)} already applied")
    raise SystemExit(0)
ok = True
for anchor, add, tag in ((HUNK_A_ANCHOR, HUNK_A_ADD, "observe"), (HUNK_B_ANCHOR, HUNK_B_ADD, "cap")):
    if src.count(anchor) != 1:
        print(f"  FAIL  hunk {tag}: anchor matched {src.count(anchor)}x, expected 1")
        ok = False
if not ok:
    import os
    if os.environ.get("RADIANCE_DYNAMIC_WIDTH", "0") == "1":
        raise SystemExit("  FAIL  dynwidth cannot be honoured")
    print("  WARN  dynwidth not applied (inert: RADIANCE_DYNAMIC_WIDTH is off)")
    raise SystemExit(0)
src = src.replace(HUNK_A_ANCHOR, HUNK_A_ANCHOR + HUNK_A_ADD, 1)
src = src.replace(HUNK_B_ANCHOR, HUNK_B_ADD, 1)
src += TAIL
ast.parse(src)              # never write a file that would not parse
TARGET.write_text(src)
print(f"  OK    {TARGET.relative_to(SP)}")
