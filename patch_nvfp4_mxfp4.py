"""Select bounded NVFP4 compatibility after original metadata validation.

Adapted from GGZ14 20652ec, frozen donor 31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663.
No hooks for FP8/BF16 are installed. Upstream ignored-layer matching stays first.
"""
from pathlib import Path
import sysconfig
from _patchlib import apply


def main():
    path = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
    old = '''        if self._is_nvfp4_format(weight_quant):
            if input_quant is None:
                return CompressedTensorsW4A4Fp4(use_a16=True)

            if not self._is_nvfp4_format(input_quant):
                raise ValueError(
                    "For NVFP4 weights, input quantization must also be NVFP4 "
                    "format, None for NVFP4A16"
                )
            return CompressedTensorsW4A4Fp4()
'''
    new = '''        if self._is_nvfp4_format(weight_quant):
            if input_quant is not None and not self._is_nvfp4_format(input_quant):
                raise ValueError("NVFP4 weights require NVFP4 inputs or None")
            # Radiance: NVFP4-only load-time compatibility, original exclusions first.
            from radiance_nvfp4 import select_scheme
            scheme = select_scheme(weight_quant, input_quant, layer_name)
            if scheme is not None:
                return scheme
            return CompressedTensorsW4A4Fp4(use_a16=input_quant is None)
'''
    apply(path, old, new, "from radiance_nvfp4 import select_scheme",
          "NVFP4-only bounded load-time conversion")


if __name__ == "__main__":
    main()
