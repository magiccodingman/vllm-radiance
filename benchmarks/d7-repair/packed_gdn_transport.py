"""Authenticated loader for the no-copy convolution-to-GDN transport."""

import hashlib
import importlib.util
import sys
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json


def load(build, torch, convolution_update):
    build = Path(build)
    manifest = private_json(build / "build.json")
    authenticate(manifest)
    modules = {}
    for name, digest in manifest["generated_sources"].items():
        path = build / name
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            "GDN transport artifact changed",
        )
        spec = importlib.util.spec_from_file_location("qwen_packed_" + digest, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return (
        modules["packed_convolution.py"].StockConvolutionAdapter(torch, convolution_update),
        modules["packed_recurrent.py"].StockIndexedAdapter(torch),
    )
