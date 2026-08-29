#!/usr/bin/env python3
"""Install the call site for radiance_gdnmerge (GDN in_proj_qkvz+in_proj_ba single-GEMM merge).

The merge itself lives in radiance_gdnmerge.py; this only injects the one call, immediately after
the model runner has built the main model. That point is chosen deliberately: the loader has
already run process_weights_after_loading (so weight_scale is [K/32, N] and radiance_wref is
folded), and it is before the drafter loads and before any CUDA graph is captured, so the
torch.cat the merge does is a plain allocation rather than one inside capture.

vLLM 0.27 ships TWO model runners -- the legacy `v1/worker/gpu_model_runner.py` and the newer
`v1/worker/gpu/model_runner.py` (GPUModelRunnerV2, selected in gpu_worker.py) -- with different
`load_model` bodies. Patching only one is a silent no-op: the injected call simply never runs, and
the merge reports nothing at all. Both are patched here, and at least one must take.

Inert unless RADIANCE_GDN_MERGE_INPROJ=1. Idempotent. Run once pre-serve / at image build."""
import ast
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
SENTINEL = "import radiance_gdnmerge as _radiance_gdnmerge"
CALL = (
    "{i}# RADIANCE: merge each GDN layer's two input projections into one GEMM.\n"
    "{i}import radiance_gdnmerge as _radiance_gdnmerge\n"
    "{i}_radiance_gdnmerge.merge_model(self.model)\n"
)

# (file, anchor). The two runners differ in indent and in how model_config is passed.
TARGETS = [
    (SP / "vllm/v1/worker/gpu_model_runner.py",
     "                self.model = model_loader.load_model(\n"
     "                    vllm_config=self.vllm_config, model_config=self.model_config\n"
     "                )\n"),
    (SP / "vllm/v1/worker/gpu/model_runner.py",
     "            self.model = model_loader.load_model(\n"
     "                vllm_config=self.vllm_config, model_config=self.vllm_config.model_config\n"
     "            )\n"),
]

applied = 0
for path, anchor in TARGETS:
    label = path.relative_to(SP)
    if not path.exists():
        print(f"  SKIP  {label}: not in this vLLM")
        continue
    src = path.read_text()
    if SENTINEL in src:
        print(f"  NOOP  {label} already applied")
        applied += 1
        continue
    if src.count(anchor) != 1:
        print(f"  SKIP  {label}: anchor matched {src.count(anchor)}x, expected 1")
        continue
    indent = " " * (len(anchor) - len(anchor.lstrip(" ")))
    new = src.replace(anchor, anchor + CALL.format(i=indent), 1)
    ast.parse(new)          # never write a file that would not parse
    path.write_text(new)
    print(f"  OK    {label}")
    applied += 1

if not applied:
    # Only fatal when the feature is actually switched on. This patch runs on every container
    # start; a vLLM upgrade that moves these anchors must not abort a production boot that was
    # never going to use the merge anyway.
    import os
    msg = "gdn in_proj merge: no model runner matched -- NOT applied"
    if os.environ.get("RADIANCE_GDN_MERGE_INPROJ", "0") == "1":
        raise SystemExit(f"  FAIL  {msg} (RADIANCE_GDN_MERGE_INPROJ=1 cannot be honoured)")
    print(f"  WARN  {msg} (inert: RADIANCE_GDN_MERGE_INPROJ is off)")
