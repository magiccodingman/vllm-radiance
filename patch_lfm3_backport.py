#!/usr/bin/env python3
"""Apply the immutable upstream LFM3 text patch with exact context guards.

The original mail patch preserves author/license/test history. Only its vllm/
files are installed; upstream tests are retained in the payload for qualification.
Each hunk requires unique old context or the exact already-applied new context.
No fuzzy matching, external patch tool or checkpoint remote-code execution.
"""
import ast
import hashlib
from pathlib import Path
import re
import sysconfig

COMMIT = "c96e5f60bbdd7b523d56f6e7cb9b39e0a7013287"
PATCH_SHA256 = "5258e48cec5c9f5cfa39ad1e00513b3ddaa7a988065770493e7d05c247c004fe"
PAYLOAD = Path(__file__).parent / "upstream/vllm-lfm3/c96e5f60bbdd.patch"


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
        text = original
        hunks = []
        old, new = None, None
        for line in lines:
            if line.startswith("@@ "):
                if old is not None:
                    hunks.append(("".join(old), "".join(new)))
                old, new = [], []
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
            hunks.append(("".join(old), "".join(new)))
        if not hunks:
            raise RuntimeError(f"No hunks for {relative}")
        for before, after in hunks:
            if relative.as_posix() == "vllm/transformers_utils/configs/lfm3.py":
                after = adapt_lfm3_config(after)
            if after in text:
                continue
            if not before:
                if text:
                    raise RuntimeError(f"Existing LFM3 file differs: {relative}")
                text = after
            elif text.count(before) == 1:
                text = text.replace(before, after, 1)
            else:
                raise RuntimeError(f"LFM3 source drift in {relative}: anchor {text.count(before)}x")
        ast.parse(text)
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
