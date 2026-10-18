"""Reproducible FP8 attention/KV-scale calibration for Radiance.

The calibration run observes the full-precision tensors entering vLLM's
``Attention`` modules.  It never edits the source checkpoint.  The resulting
scalar q/k/v/prob scales live in a small safetensors sidecar which Radiance can
load last with ``RADIANCE_FP8_KV_SCALES=/path/to/scales.safetensors``.

This module intentionally uses only public PyTorch module hooks and vLLM's
documented worker-extension/collective-RPC mechanism.  GPU-heavy imports are
kept inside the calibration command so artifact verification remains cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SCALE_SUFFIXES = ("q_scale", "k_scale", "v_scale", "prob_scale")
_SCALE_NAME = re.compile(r"^.+\.(?:q|k|v|prob)_scale$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_metadata(model: Path, filename: str) -> tuple[str | None, str | None]:
    path = model / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    if not path.is_file():
        return None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    return (lines[0] if lines else None, lines[1] if len(lines) > 1 else None)


def model_identity(model: str | Path) -> dict[str, Any]:
    """Return a path-independent, cheap identity for a local HF checkpoint."""
    root = Path(model).resolve()
    if not root.is_dir():
        raise ValueError(f"calibration requires a local model directory: {root}")
    weights: list[dict[str, Any]] = []
    revisions: set[str] = set()
    for path in sorted(root.glob("*.safetensors")):
        revision, oid = _hf_metadata(root, path.name)
        if revision:
            revisions.add(revision)
        record: dict[str, Any] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
        }
        if oid:
            record["hf_content_oid"] = oid
        else:
            # A local checkpoint without Hub metadata still gets an exact
            # identity. This is intentionally expensive only once, when the
            # artifact is produced or explicitly verified at load time.
            record["sha256"] = _sha256(path)
        weights.append(record)
    if not weights:
        raise ValueError(f"no safetensors weights found beneath {root}")
    config = root / "config.json"
    if not config.is_file():
        raise ValueError(f"missing model config: {config}")
    config_revision, config_oid = _hf_metadata(root, config.name)
    if config_revision:
        revisions.add(config_revision)
    result: dict[str, Any] = {
        "config_sha256": _sha256(config),
        "weights": weights,
    }
    if config_oid:
        result["config_hf_oid"] = config_oid
    if len(revisions) == 1:
        result["hf_revision"] = next(iter(revisions))
    return result


def manifest_path(sidecar: str | Path) -> Path:
    path = Path(sidecar)
    return path.with_suffix(".manifest.json")


def _validate_manifest(sidecar: Path, source_model: str | Path | None) -> dict[str, Any]:
    path = manifest_path(sidecar)
    if not path.is_file():
        raise ValueError(f"missing FP8-KV calibration manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported FP8-KV manifest schema {data.get('schema_version')!r}"
        )
    expected = data.get("artifact", {}).get("sha256")
    actual = _sha256(sidecar)
    if expected != actual:
        raise ValueError(
            f"FP8-KV sidecar checksum mismatch: expected {expected}, got {actual}"
        )
    if source_model is not None:
        actual_model = model_identity(source_model)
        if data.get("source_model") != actual_model:
            raise ValueError(
                "FP8-KV sidecar was calibrated for a different checkpoint; "
                "set RADIANCE_FP8_KV_SCALES_VERIFY=0 only for deliberate diagnostics"
            )
    return data


def validate_scale_sidecar(
    sidecar: str | Path,
    *,
    source_model: str | Path | None = None,
    verify_manifest: bool = True,
) -> list[str]:
    """Validate names, scalar values, checksum, and optional model identity."""
    from safetensors import safe_open

    path = Path(sidecar).resolve()
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"FP8-KV sidecar must be a .safetensors file: {path}")
    if verify_manifest:
        _validate_manifest(path, source_model)
    keys: list[str] = []
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            if not _SCALE_NAME.fullmatch(key):
                raise ValueError(f"non-scale tensor is forbidden in FP8-KV sidecar: {key}")
            value = handle.get_tensor(key)
            if value.size != 1:
                raise ValueError(f"FP8-KV scale {key} is not scalar: shape={value.shape}")
            scalar = float(value.reshape(-1)[0])
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(f"FP8-KV scale {key} must be finite and positive: {scalar}")
            keys.append(key)
    if not keys:
        raise ValueError(f"FP8-KV sidecar contains no scales: {path}")
    return sorted(keys)


def iter_scale_sidecar(
    sidecar: str | Path,
    *,
    source_model: str | Path | None = None,
    verify_manifest: bool = True,
) -> Iterable[tuple[str, Any]]:
    """Yield a validated sidecar as torch tensors for vLLM's weight loader."""
    from safetensors import safe_open

    path = Path(sidecar).resolve()
    keys = validate_scale_sidecar(
        path, source_model=source_model, verify_manifest=verify_manifest
    )
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in keys:
            yield key, handle.get_tensor(key)


class RadianceKVCalibrationExtension:
    """vLLM worker extension which observes attention inputs without CPU syncs."""

    def radiance_kv_calibration_start(self) -> dict[str, Any]:
        import torch
        from vllm.model_executor.layers.attention import Attention

        self.radiance_kv_calibration_stop()
        model = self.get_model()
        if self.device is None:
            raise RuntimeError("worker device is not initialized")
        stats: dict[str, dict[str, Any]] = {}
        handles = []

        for name, module in model.named_modules():
            if not isinstance(module, Attention):
                continue
            entry = {
                "q": torch.zeros((), dtype=torch.float32, device=self.device),
                "k": torch.zeros((), dtype=torch.float32, device=self.device),
                "v": torch.zeros((), dtype=torch.float32, device=self.device),
                "calls": 0,
            }
            stats[name] = entry

            def observe(_module, inputs, *, _entry=entry):
                _entry["calls"] += 1
                for label, tensor in zip(("q", "k", "v"), inputs[:3]):
                    if tensor is None or tensor.numel() == 0:
                        continue
                    # Keep accumulation entirely on-device. One scalar per
                    # layer is synchronized only when the report is requested.
                    amax = tensor.detach().abs().amax().float()
                    amax = torch.nan_to_num(
                        amax, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
                    )
                    _entry[label] = torch.maximum(_entry[label], amax)

            handles.append(module.register_forward_pre_hook(observe))

        if not stats:
            raise RuntimeError("no vLLM Attention modules were found for calibration")
        self._radiance_kv_calibration_stats = stats
        self._radiance_kv_calibration_handles = handles
        return {"rank": int(self.rank), "layers": len(stats)}

    def radiance_kv_calibration_report(self) -> dict[str, Any]:
        stats = getattr(self, "_radiance_kv_calibration_stats", None)
        if not stats:
            raise RuntimeError("FP8-KV calibration was not started on this worker")
        layers = {
            name: {
                "q": float(entry["q"].item()),
                "k": float(entry["k"].item()),
                "v": float(entry["v"].item()),
                "calls": int(entry["calls"]),
            }
            for name, entry in stats.items()
        }
        return {"rank": int(self.rank), "layers": layers}

    def radiance_kv_calibration_stop(self) -> bool:
        handles = getattr(self, "_radiance_kv_calibration_handles", ())
        for handle in handles:
            handle.remove()
        self._radiance_kv_calibration_handles = []
        return True


def aggregate_worker_stats(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not reports:
        raise ValueError("no worker calibration reports")
    merged: dict[str, dict[str, Any]] = {}
    for report in reports:
        for name, values in report["layers"].items():
            calls = values["calls"]
            if type(calls) is not int or calls < 1:
                raise ValueError(f"attention layer was never observed on rank {report.get('rank')}: {name}")
            entry = merged.setdefault(name, {"q": 0.0, "k": 0.0, "v": 0.0, "calls": 0})
            for kind in ("q", "k", "v"):
                value = float(values[kind])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"invalid observed {kind} amax on rank {report.get('rank')} for {name}: {value}")
                entry[kind] = max(float(entry[kind]), value)
            entry["calls"] += calls
    for name, values in merged.items():
        for kind in ("q", "k", "v"):
            value = float(values[kind])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid observed {kind} amax for {name}: {value}")
        if not values["calls"]:
            raise ValueError(f"attention layer was never observed: {name}")
    return dict(sorted(merged.items()))


def _checked_scale(name: str, value: float) -> float:
    """The artifact stores FP32; validate that cast without importing Torch."""
    try:
        stored = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise ValueError(f"FP8-KV scale {name} overflows FP32: {value}") from error
    if not math.isfinite(stored) or stored <= 0.0:
        raise ValueError(f"FP8-KV scale {name} must remain finite and positive in FP32: {value}")
    return value


def _validate_scale_parameters(fp8_max: float, margin: float) -> None:
    if not math.isfinite(fp8_max) or fp8_max <= 0 or not math.isfinite(margin) or margin < 1.0:
        raise ValueError("fp8_max must be finite and positive and margin must be finite and >= 1")
    _checked_scale("prob_scale", margin / fp8_max)


def make_scale_values(
    observations: dict[str, dict[str, Any]],
    *,
    fp8_max: float = 448.0,
    margin: float = 1.05,
) -> dict[str, float]:
    """Convert activation maxima to OCP-E4M3 checkpoint scale values."""
    _validate_scale_parameters(fp8_max, margin)
    scales: dict[str, float] = {}
    for name, values in observations.items():
        for kind in ("q", "k", "v"):
            value = float(values[kind])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid observed {kind} amax for {name}: {value}")
            key = f"{name}.{kind}_scale"
            scales[key] = _checked_scale(
                key, max(value * margin / fp8_max, 1.0e-12)
            )
        # Softmax probabilities are bounded by one. This scale leaves the
        # same explicit headroom as q/k/v without needing backend internals.
        scales[f"{name}.prob_scale"] = margin / fp8_max
    return scales


def _read_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not ({"text", "messages"} & item.keys()):
            raise ValueError(f"{path}:{line_no}: expected text or messages")
        repeat = int(item.get("repeat", 1))
        if repeat < 1 or repeat > 4096:
            raise ValueError(f"{path}:{line_no}: repeat must be in [1,4096]")
        item["repeat"] = repeat
        items.append(item)
    if not items:
        raise ValueError(f"calibration corpus is empty: {path}")
    return items, _sha256(path)


def _render_prompts(llm: Any, items: list[dict[str, Any]]) -> list[str]:
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    for item in items:
        if "text" in item:
            rendered = str(item["text"])
        else:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if item.get("tools") is not None:
                kwargs["tools"] = item["tools"]
            rendered = tokenizer.apply_chat_template(item["messages"], **kwargs)
        prompts.append(rendered * int(item["repeat"]))
    return prompts


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _write_artifact(
    output: Path,
    scales: dict[str, float],
    observations: dict[str, dict[str, Any]],
    *,
    source_model: Path,
    corpus: Path,
    corpus_sha256: str,
    prompt_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    import torch
    from safetensors.torch import save_file

    output = output.resolve()
    manifest = manifest_path(output)
    if output.exists() or manifest.exists():
        raise FileExistsError(
            f"immutable calibration output already exists: {output} or {manifest}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    tensors = {name: torch.tensor(value, dtype=torch.float32) for name, value in scales.items()}
    save_file(tensors, temporary, metadata={"format": "radiance-fp8-kv-scales-v1"})
    temporary.replace(output)
    data = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "filename": output.name,
            "sha256": _sha256(output),
            "tensor_count": len(scales),
        },
        "source_model": model_identity(source_model),
        "calibration": {
            "corpus": corpus.name,
            "corpus_sha256": corpus_sha256,
            "fp8_format": "OCP E4M3",
            "fp8_max": args.fp8_max,
            "margin": args.margin,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "batch_size": args.batch_size,
            "attention_backend": args.attention_backend,
            "attention_layers": len(observations),
            "prompts": prompt_records,
            "observed_amax": observations,
        },
        "software": {
            "radiance": os.environ.get("RADIANCE_VERSION"),
            "vllm": _dist_version("vllm"),
            "torch": _dist_version("torch"),
            "triton": _dist_version("triton"),
            "aiter": _dist_version("amd-aiter"),
            "python": platform.python_version(),
        },
    }
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibrate(args: argparse.Namespace) -> None:
    # Import only after arguments/model identity are validated. This command is
    # intentionally GPU-consuming and should be run in a maintenance window.
    source_model = Path(args.model).resolve()
    corpus = Path(args.corpus).resolve()
    output = Path(args.output).resolve()
    if output.exists() or manifest_path(output).exists():
        raise FileExistsError(f"immutable calibration output already exists: {output}")
    if args.batch_size < 1 or args.tensor_parallel_size < 1:
        raise ValueError("batch size and tensor parallel size must be positive")
    if args.max_model_len < 1 or args.max_num_batched_tokens < 1:
        raise ValueError("model/token limits must be positive")
    _validate_scale_parameters(args.fp8_max, args.margin)
    model_identity(source_model)
    items, corpus_sha256 = _read_corpus(corpus)
    from vllm import LLM, SamplingParams

    quantization = None if args.quantization == "auto" else args.quantization
    llm = LLM(
        model=str(source_model),
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        quantization=quantization,
        kv_cache_dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=args.language_model_only,
        attention_backend=args.attention_backend,
        worker_extension_cls=(
            "radiance_kv_calibration.RadianceKVCalibrationExtension"
        ),
    )
    prompts = _render_prompts(llm, items)
    tokenizer = llm.get_tokenizer()
    prompt_records = [
        {
            "id": item.get("id", f"prompt-{index}"),
            "characters": len(prompt),
            "tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        }
        for index, (item, prompt) in enumerate(zip(items, prompts))
    ]
    too_long = [record for record in prompt_records if record["tokens"] >= args.max_model_len]
    if too_long:
        raise ValueError(
            f"calibration prompts must be shorter than max_model_len={args.max_model_len}: "
            f"{too_long}"
        )
    starts = llm.collective_rpc("radiance_kv_calibration_start")
    print(f"[radiance] FP8-KV calibration active: {starts}", file=sys.stderr)
    params = SamplingParams(temperature=0.0, max_tokens=1, seed=args.seed)
    try:
        for offset in range(0, len(prompts), args.batch_size):
            llm.generate(
                prompts[offset : offset + args.batch_size],
                params,
                use_tqdm=True,
            )
        reports = llm.collective_rpc("radiance_kv_calibration_report")
    finally:
        llm.collective_rpc("radiance_kv_calibration_stop")
    observations = aggregate_worker_stats(reports)
    scales = make_scale_values(
        observations, fp8_max=args.fp8_max, margin=args.margin
    )
    _write_artifact(
        output,
        scales,
        observations,
        source_model=source_model,
        corpus=corpus,
        corpus_sha256=corpus_sha256,
        prompt_records=prompt_records,
        args=args,
    )
    print(f"[radiance] wrote immutable FP8-KV sidecar: {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="verify a calibration sidecar and manifest")
    verify.add_argument("sidecar")
    verify.add_argument("--model")

    run = sub.add_parser("calibrate", help="run an eager full-precision-KV calibration")
    run.add_argument("--model", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--tensor-parallel-size", type=int, default=2)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--max-model-len", type=int, default=16384)
    run.add_argument("--max-num-batched-tokens", type=int, default=4096)
    run.add_argument("--attention-backend", default="R4D")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--quantization", default="auto")
    run.add_argument("--fp8-max", type=float, default=448.0)
    run.add_argument("--margin", type=float, default=1.05)
    run.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify":
        keys = validate_scale_sidecar(args.sidecar, source_model=args.model)
        print(json.dumps({"status": "ok", "tensor_count": len(keys), "keys": keys}, indent=2))
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
