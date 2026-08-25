#!/usr/bin/env python3
"""Load explicitly BF16 Qwen3.5 MTP tensors outside the global Quark recipe.

AMD's Qwen3.8-27B Quark MXFP4 checkpoint declares a global FP4 recipe, but its
embedded ``mtp.*`` tensors are ordinary BF16 tensors and are not listed in the
recipe's exclusions.  vLLM therefore constructs packed Quark parameters for the
MTP linears and later fails the weight-shape assertion while loading BF16 data.

This is intentionally opt-in.  A future checkpoint may genuinely quantize its
MTP tensors, so Radiance must not assume that every Quark MTP head is BF16.
Set ``RADIANCE_QUARK_BF16_MTP=1`` only after checking the checkpoint tensors.
The target model remains Quark MXFP4/W4A8; only the small MTP submodule and its
BF16 lm_head are constructed unquantized.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply


SP = Path(sysconfig.get_paths()["purelib"])
QWEN_MTP = SP / "vllm/model_executor/models/qwen3_5_mtp.py"


IMPORT_ANCHOR = "from collections.abc import Iterable\n\nimport torch\n"
IMPORT_NEW = "from collections.abc import Iterable\n\nimport os\n\nimport torch\n"

HELPER_ANCHOR = "logger = init_logger(__name__)\n\n\n@support_torch_compile(\n"
HELPER_NEW = '''logger = init_logger(__name__)


def _radiance_quark_bf16_mtp(quant_config) -> bool:
    """Whether this checkpoint's MTP submodule must bypass Quark packing."""
    return bool(
        quant_config
        and quant_config.get_name() == "quark"
        and os.environ.get("RADIANCE_QUARK_BF16_MTP", "0") == "1"
    )


@support_torch_compile(
'''

FC_ANCHOR = '''        fc_quant = (
            None
            if (quant_config and quant_config.get_name() == "modelopt_fp4")
            else quant_config
        )
'''
FC_NEW = '''        radiance_quark_bf16_mtp = _radiance_quark_bf16_mtp(quant_config)
        if radiance_quark_bf16_mtp:
            logger.warning_once(
                "[radiance] loading Qwen MTP tensors as BF16 outside the global "
                "Quark recipe (RADIANCE_QUARK_BF16_MTP=1)"
            )
        fc_quant = (
            None
            if (
                quant_config
                and quant_config.get_name() == "modelopt_fp4"
                or radiance_quark_bf16_mtp
            )
            else quant_config
        )
'''

LAYERS_ANCHOR = '''        original_quant = vllm_config.quant_config
        if quant_config and quant_config.get_name() not in ("modelopt_fp4",):
            hf_qc = getattr(model_config.hf_config, "quantization_config", None)
            if isinstance(hf_qc, dict):
                dynamic = hf_qc.get("dynamic", {})
                if any(k.startswith("-:") and "mtp" in k for k in dynamic):
                    vllm_config.quant_config = None
'''
LAYERS_NEW = '''        original_quant = vllm_config.quant_config
        # --- radiance (patch_quark_bf16_mtp.py): checkpoint BF16 MTP layers ---
        if radiance_quark_bf16_mtp:
            vllm_config.quant_config = None
        elif quant_config and quant_config.get_name() not in ("modelopt_fp4",):
            hf_qc = getattr(model_config.hf_config, "quantization_config", None)
            if isinstance(hf_qc, dict):
                dynamic = hf_qc.get("dynamic", {})
                if any(k.startswith("-:") and "mtp" in k for k in dynamic):
                    vllm_config.quant_config = None
'''

HEAD_ANCHOR = "        self.quant_config = vllm_config.quant_config\n"
HEAD_NEW = '''        self.quant_config = (
            None
            if _radiance_quark_bf16_mtp(vllm_config.quant_config)
            else vllm_config.quant_config
        )
'''


def main():
    apply(QWEN_MTP, IMPORT_ANCHOR, IMPORT_NEW, "import os", "quark MTP: import os")
    apply(
        QWEN_MTP,
        HELPER_ANCHOR,
        HELPER_NEW,
        "def _radiance_quark_bf16_mtp",
        "quark MTP: add guarded BF16 detector",
    )
    apply(
        QWEN_MTP,
        FC_ANCHOR,
        FC_NEW,
        "radiance_quark_bf16_mtp =",
        "quark MTP: keep BF16 fc unpacked",
    )
    apply(
        QWEN_MTP,
        LAYERS_ANCHOR,
        LAYERS_NEW,
        "checkpoint BF16 MTP layers",
        "quark MTP: keep BF16 decoder linears unpacked",
    )
    apply(
        QWEN_MTP,
        HEAD_ANCHOR,
        HEAD_NEW,
        "if _radiance_quark_bf16_mtp(vllm_config.quant_config)",
        "quark MTP: keep BF16 lm_head unpacked",
    )


if __name__ == "__main__":
    main()
