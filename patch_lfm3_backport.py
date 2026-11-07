#!/usr/bin/env python3
"""Apply the immutable upstream LFM3 text patch with exact context guards.

The original mail patch preserves author/license/test history. Only its vllm/
files are installed; upstream tests are retained in the payload for qualification.
Each hunk requires unique old context or the exact already-applied new context
within its kernel function. The two kernel bodies must match independent
fingerprints obtained with git apply on the pinned stable source.
No fuzzy matching, external patch tool or checkpoint remote-code execution.
After verification, width-7 products use explicit FP32 intermediates: BF16
product rounding otherwise fails the unchanged upstream prefill oracle on RDNA4.
"""
import ast
import hashlib
from pathlib import Path
import re
import sysconfig

COMMIT = "c96e5f60bbdd7b523d56f6e7cb9b39e0a7013287"
PATCH_SHA256 = "5258e48cec5c9f5cfa39ad1e00513b3ddaa7a988065770493e7d05c247c004fe"
PAYLOAD = Path(__file__).parent / "upstream/vllm-lfm3/c96e5f60bbdd.patch"
KERNEL_HASHES = {
    "_causal_conv1d_fwd_kernel": "d4070e4d5c8a2836cb7182ca57ae76b1d2bd56bf022fa5d19a47b68d3f654e72",
    "_causal_conv1d_update_kernel": "83290c32ef13364eaf970f5f73c3091d0456fb7349569f29b9d4295a9041707d",
}
PRODUCT_BEFORE = "            acc += matrix_x * matrix_w  # [BLOCK_N]\n"
PRODUCT_AFTER = (
    "            # radiance: LFM3 width-7 FP32 products (RADIANCE_LFM3_FP32_PRODUCTS).\n"
    "            if KERNEL_WIDTH == 7:\n"
    "                acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)\n"
    "            else:\n"
    "                acc += matrix_x * matrix_w  # [BLOCK_N]\n"
)


def _function_span(source: str, name: str) -> tuple[int, int]:
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"LFM3 source drift: expected one function {name}")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return sum(map(len, lines[:node.lineno - 1])), sum(map(len, lines[:node.end_lineno]))


def _apply_hunk(text: str, before: str, after: str, scope: str | None = None) -> str:
    start, end = _function_span(text, scope) if scope else (0, len(text))
    region = text[start:end]
    # Forward/decode have intentionally identical arithmetic/state fragments.
    # A matching forward fragment must never suppress a pending decode hunk.
    if after and region.count(after) == 1:
        return text
    if not before:
        if text:
            raise RuntimeError("LFM3 source drift: existing new file differs")
        return after
    if region.count(before) != 1:
        raise RuntimeError(f"LFM3 source drift in {scope}: anchor {region.count(before)}x")
    return text[:start] + region.replace(before, after, 1) + text[end:]


def canonical_kernel_source(text: str) -> str:
    """Remove only our exact, complete precision adaptation before verification."""
    if "RADIANCE_LFM3_FP32_PRODUCTS" not in text:
        return text
    if text.count(PRODUCT_AFTER) != len(KERNEL_HASHES):
        raise RuntimeError("LFM3 source drift: incomplete FP32 product adaptation")
    return text.replace(PRODUCT_AFTER, PRODUCT_BEFORE)


def adapt_kernel_precision(text: str) -> str:
    """Promote width-7 products; retain established width-4 arithmetic verbatim."""
    for name in KERNEL_HASHES:
        start, end = _function_span(text, name)
        region = text[start:end]
        if region.count(PRODUCT_BEFORE) != 1:
            raise RuntimeError(f"LFM3 product precision anchor drift: {name}")
        text = text[:start] + region.replace(PRODUCT_BEFORE, PRODUCT_AFTER, 1) + text[end:]
    ast.parse(text)
    return text


def adapt_lfm3_config(source: str) -> str:
    """Preserve upstream's config representation on Transformers 5.16.1.

    The base initializer validates legacy conv labels as linear_attention and
    injects global RoPE defaults into the nested dictionary. Retain the original
    validated layer labels and the two declared RoPE dictionaries. Model/kernel
    arithmetic is unchanged; the original mail patch remains byte-identical.
    """
    before = '        super().__init__(\n'
    if source.count(before) != 1:
        raise RuntimeError("LFM3 config initialization anchor drift")
    source = source.replace(before,
        '        _lfm3_layer_types = self.layer_types\n' + before, 1)
    ending = '            **kwargs,\n        )\n'
    if source.count(ending) != 1:
        raise RuntimeError("LFM3 config post-initialization anchor drift")
    return source.replace(ending, ending +
        '        self.layer_types = _lfm3_layer_types\n'
        '        self.rope_parameters = {\n'
        '            label: self.rope_parameters[label] for label in self._rope_type_labels\n'
        '        }\n'
        '        self.validate_rope()\n', 1)


def apply_backport(site: Path, payload: Path = PAYLOAD):
    raw = payload.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PATCH_SHA256:
        raise RuntimeError("LFM3 upstream payload checksum mismatch")
    staged = {}
    for section in raw.decode().split("diff --git ")[1:]:
        header, *lines = section.splitlines(keepends=True)
        match = re.fullmatch(r"a/(\S+) b/(\S+)\n", header)
        if match is None or match[1] != match[2]:
            raise RuntimeError("Unexpected upstream diff path")
        relative = Path(match[2])
        if relative.parts[0] != "vllm":
            continue
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise RuntimeError("Unexpected LFM3 runtime path")
        target = site / relative
        original = target.read_text() if target.exists() else ""
        is_convolution = relative.as_posix().endswith("/ops/causal_conv1d.py")
        text = canonical_kernel_source(original) if is_convolution else original
        hunks = []
        old, new, scope = None, None, None
        for line in lines:
            if line.startswith("@@ "):
                if old is not None:
                    hunks.append(("".join(old), "".join(new), scope))
                old, new = [], []
                scope = None
                if is_convolution:
                    context = re.search(r"@@ def (\w+)\(", line)
                    if context is None:
                        raise RuntimeError("LFM3 causal-convolution function context drift")
                    scope = context[1]
            elif old is not None:
                if line.startswith(" "):
                    old.append(line[1:]); new.append(line[1:])
                elif line.startswith("-"):
                    old.append(line[1:])
                elif line.startswith("+"):
                    new.append(line[1:])
                else:
                    raise RuntimeError("Unexpected upstream hunk content")
        if old is not None:
            hunks.append(("".join(old), "".join(new), scope))
        if not hunks:
            raise RuntimeError(f"No hunks for {relative}")
        for before, after, scope in hunks:
            if relative.as_posix() == "vllm/transformers_utils/configs/lfm3.py":
                after = adapt_lfm3_config(after)
            text = _apply_hunk(text, before, after, scope)
        ast.parse(text)
        if is_convolution:
            for name, expected in KERNEL_HASHES.items():
                start, end = _function_span(text, name)
                actual = hashlib.sha256(text[start:end].encode()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"LFM3 upstream kernel fingerprint mismatch: {name}")
            text = adapt_kernel_precision(text)
        staged[target] = text
    if len(staged) != 6:
        raise RuntimeError(f"Expected six LFM3 runtime files, got {len(staged)}")
    # Validate every target before mutation, so source drift cannot half-install
    # the model/config/width-7 kernel registration.
    for target, text in staged.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(f"OK LFM3 official backport {COMMIT}: {len(staged)} runtime files")


if __name__ == "__main__":
    apply_backport(Path(sysconfig.get_paths()["purelib"]))
