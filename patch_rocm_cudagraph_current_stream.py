#!/usr/bin/env python3
"""Backport vLLM #53818: capture ROCm graphs on the current stream.

PyTorch ROCm graph capture must use the stream selected by vLLM. Upstream
commit ``080a66a69c6fd1fe464756f88ab958baad66ce69`` adds that stream explicitly
for the general runner, multimodal encoder graphs, and Gemma4 proposer graphs.

Idempotent; exact-anchor guarded; ``ast.parse`` checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])


def main() -> None:
    general = LIB / "vllm/v1/worker/gpu/cudagraph_utils.py"
    apply(
        general,
        "from vllm.utils.math_utils import round_up\n",
        "from vllm.utils.math_utils import round_up\n"
        "from vllm.utils.torch_utils import current_stream\n",
        "from vllm.utils.torch_utils import current_stream",
        "rocm graph: import current stream",
    )
    apply(
        general,
        "                        with torch.cuda.graph(graph, self.pool):\n",
        """                        with torch.cuda.graph(
                            graph, self.pool, stream=current_stream()
                        ):
""",
        "graph, self.pool, stream=current_stream()",
        "rocm graph: general capture uses current stream",
    )

    encoder = LIB / "vllm/v1/worker/encoder_cudagraph.py"
    apply(
        encoder,
        "from vllm.utils.gpu_sync_debug import gpu_sync_allowed\n",
        "from vllm.utils.gpu_sync_debug import gpu_sync_allowed\n"
        "from vllm.utils.torch_utils import current_stream\n",
        "from vllm.utils.torch_utils import current_stream",
        "rocm graph: encoder imports current stream",
    )
    apply(
        encoder,
        """        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.graph_pool):
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
""",
        """        with (
            torch.inference_mode(),
            torch.cuda.graph(graph, pool=self.graph_pool, stream=current_stream()),
        ):
            output = self.model.encoder_cudagraph_forward({**values}, path=path)
""",
        "pool=self.graph_pool, stream=current_stream()",
        "rocm graph: encoder capture uses current stream",
    )

    gemma = LIB / "vllm/v1/spec_decode/gemma4.py"
    apply(
        gemma,
        "from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase\n",
        "from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase\n"
        "from vllm.utils.torch_utils import current_stream\n",
        "from vllm.utils.torch_utils import current_stream",
        "rocm graph: Gemma4 imports current stream",
    )
    apply(
        gemma,
        "            with torch.cuda.graph(g):\n",
        "            with torch.cuda.graph(g, stream=current_stream()):\n",
        "torch.cuda.graph(g, stream=current_stream())",
        "rocm graph: Gemma4 capture uses current stream",
    )


if __name__ == "__main__":
    main()
