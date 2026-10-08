"""Build data-movement-only GDN adapters from the authenticated repair sources."""

import argparse
import hashlib
import json
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def replace_once(text, old, new):
    require(text.count(old) == 1, "GDN transport source anchor changed")
    return text.replace(old, new)


def build(args):
    manifest = private_json(args.repair)
    authenticate(manifest)
    files = {}
    for name in ("stock_gdn_convolution_adapter.py", "stock_gdn_indexed_adapter.py"):
        path = args.sources / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(files[name] == manifest["sources"][name], "GDN repair source changed")
    convolution = (args.sources / "stock_gdn_convolution_adapter.py").read_text()
    for key in ("q", "k", "v"):
        convolution = replace_once(convolution, f"{key}.contiguous().view(", f"{key}.view(")
    recurrent = (args.sources / "stock_gdn_indexed_adapter.py").read_text()
    recurrent = replace_once(
        recurrent,
        "or not x.is_contiguous():",
        "or not (x.is_contiguous() if x is output else x.stride() == (10240, 128, 1)):",
    )
    recurrent = replace_once(
        recurrent,
        "        packed = t.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)",
        """        # These are three views of the same packed convolution result.
        # Validate storage identity and byte offsets before widening the Q view.
        if (q.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
            or q.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
            or k.data_ptr() != q.data_ptr() + 2048 * q.element_size()
            or v.data_ptr() != q.data_ptr() + 4096 * q.element_size()):
            raise DiagnosticError("GDN views do not identify one packed convolution result")
        packed = q.as_strided((count, 10240), (10240, 1))""",
    )
    args.output.mkdir(mode=0o700)
    outputs = {}
    for name, text in (("packed_convolution.py", convolution), ("packed_recurrent.py", recurrent)):
        (args.output / name).write_text(text)
        outputs[name] = hashlib.sha256(text.encode()).hexdigest()
    report = seal(
        {
            "status": "BUILT_UNTESTED",
            "gpu_used": False,
            "reference_repair": manifest["sha256"],
            "input_sources": files,
            "generated_sources": outputs,
            "scope": "Unchanged numerical kernels; preserve packed storage between adapters",
        }
    )
    write_private(args.output / "build.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repair", "sources", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())
