"""Private, position-aligned decoder boundary capture for a short forced replay.

Raw activations stay beside the private replay in tmpfs. Public analysis should
emit only boundary identities, counts and numerical error statistics.
"""

import hashlib
import re
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def selected_rows(positions, selected):
    if len(positions) != len(set(positions)):
        raise DiagnosticError("duplicate positions in private boundary capture")
    return [(row, p) for row, p in enumerate(positions) if p in selected]


class PrivateRowTrace:
    def __init__(self, probe):
        self.probe = probe
        self.root = probe.root / "trace"
        self.root.mkdir(mode=0o700)
        prefix = len(probe.schedule.prefix)
        self.selected = set(range(prefix - 1, prefix + len(probe.schedule.output) - 1))
        if len(self.selected) > 65:
            raise DiagnosticError("private boundary capture requires a minimized replay")
        self.records, self.handles, self.seen = [], [], set()
        self.inventory = []
        self.active_layer = None
        registered = set()
        try:
            for name, module in probe.runner.model.named_modules(remove_duplicate=False):
                if re.search(r"(?:^|\.)layers\.\d+(?:\.|$)", name) is None:
                    continue
                self.inventory.append(name)
                if id(module) in registered:
                    continue
                registered.add(id(module))

                def before(mod, args, kwargs, *, name=name):
                    if re.search(r"(?:^|\.)layers\.\d+$", name):
                        self.active_layer = name
                    self.record(name, "before", {"args": args, "kwargs": kwargs})

                def after(mod, args, kwargs, result, *, name=name):
                    self.record(name, "after", {"result": result})
                    if re.search(r"(?:^|\.)layers\.\d+$", name):
                        self.active_layer = None

                self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(module.register_forward_hook(after, with_kwargs=True))
            if sum(bool(re.search(r"(?:^|\.)layers\.\d+$", n)) for n in self.inventory) != 64:
                raise DiagnosticError("private decoder inventory differs from the pinned model")
        except BaseException:
            self.close()
            raise

    def record(self, module, phase, tree):
        import torch

        if self.active_layer is None:
            raise DiagnosticError("private decoder call has no active logical layer")
        module = re.sub(r"^.*?layers\.\d+", self.active_layer, module, count=1)
        positions = self.probe.pending_positions
        if positions is None:
            return
        selected = selected_rows(positions, self.selected)
        if not selected:
            return

        def visit(value, path):
            if isinstance(value, torch.Tensor):
                # GDN flattens token and value-head dimensions before its gated
                # norm. Recover logical token rows so this boundary is not
                # silently omitted by the generic first-dimension check.
                if module.endswith(".linear_attn.norm") and value.ndim == 2:
                    if value.shape != (len(positions) * 48, 128):
                        raise DiagnosticError("private GDN norm shape changed")
                    value = value.reshape(len(positions), 48, 128)
                if value.ndim == 0 or value.shape[0] != len(positions):
                    return
                for row, position in selected:
                    key = (position, module, phase, path)
                    if key in self.seen:
                        write_private(
                            self.root / "capture-failure.json",
                            seal(
                                {
                                    "reason": "duplicate boundary",
                                    "boundary": list(key),
                                    "records": self.records,
                                }
                            ),
                        )
                        raise DiagnosticError("private decoder boundary executed twice")
                    self.seen.add(key)
                    tensor = value[row].detach().cpu().contiguous()
                    # Canonicalize singleton strides before byte reinterpretation.
                    packed = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                    packed.copy_(tensor)
                    raw = packed.reshape(-1).view(torch.uint8).numpy().tobytes()
                    filename = f"tensor-{len(self.records):06d}.bin"
                    with (self.root / filename).open("xb") as f:
                        f.write(raw)
                    self.records.append(
                        {
                            "position": position,
                            "module": module,
                            "phase": phase,
                            "path": path,
                            "dtype": str(tensor.dtype),
                            "shape": list(tensor.shape),
                            "bytes": len(raw),
                            "file": filename,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    visit(child, f"{path}.{i}")
            elif isinstance(value, dict):
                for name, child in sorted(value.items()):
                    visit(child, f"{path}.{name}")

        visit(tree, "call")

    def finish(self):
        for position in self.selected:
            for name in self.inventory:
                if name.endswith("post_attention_layernorm") and not any(
                    (position, name, "after", p) in self.seen
                    for p in ("call.result", "call.result.0")
                ):
                    raise DiagnosticError("private decoder boundary coverage is incomplete")
        doc = seal(
            {
                "schema": "urn:qwen:private-decoder-row-trace:v1",
                "positions": sorted(self.selected),
                "inventory": self.inventory,
                "records": self.records,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        write_private(self.root / "manifest.json", doc)
        return {"sha256": doc["sha256"], "records": len(self.records)}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
